import json
import os
import time
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from unittest.mock import patch

import openai.resources.chat.completions

from ._interceptor import (
    _assemble_from_chunks,
    MockAsyncStream,
    MockStream,
    async_recording_context,
    async_replaying_context,
    recording_context,
    replaying_context,
)
from ._models import RecordedCall
from ._pricing import estimate_cost
from ._serializer import deserialize_chunk, deserialize_response, serialize_request


def _should_update() -> bool:
    """Return True when AGENTPROBE_UPDATE=1 is set — forces re-record."""
    return os.environ.get("AGENTPROBE_UPDATE", "").strip() == "1"


class AssertionProxy:
    """Fluent assertion interface over a recorded or replayed session trace."""

    def __init__(self, calls: List[RecordedCall]):
        self._calls = calls

    # ── Iteration assertions ──────────────────────────────────────────────

    def assert_max_iterations(self, n: int) -> "AssertionProxy":
        actual = len(self._calls)
        assert actual <= n, (
            f"agentprobe: expected at most {n} LLM call(s), got {actual}"
        )
        return self

    def assert_min_iterations(self, n: int) -> "AssertionProxy":
        actual = len(self._calls)
        assert actual >= n, (
            f"agentprobe: expected at least {n} LLM call(s), got {actual}"
        )
        return self

    def assert_iteration_count(self, n: int) -> "AssertionProxy":
        actual = len(self._calls)
        assert actual == n, (
            f"agentprobe: expected exactly {n} LLM call(s), got {actual}"
        )
        return self

    # ── Tool call assertions ──────────────────────────────────────────────

    def assert_tool_called(self, tool_name: str) -> "AssertionProxy":
        names = self._all_tool_names()
        assert tool_name in names, (
            f"agentprobe: expected tool '{tool_name}' to be called, "
            f"but only these were called: {sorted(names) or '(none)'}"
        )
        return self

    def assert_not_tool_called(self, tool_name: str) -> "AssertionProxy":
        names = self._all_tool_names()
        assert tool_name not in names, (
            f"agentprobe: expected tool '{tool_name}' NOT to be called, but it was"
        )
        return self

    def assert_tool_called_with(self, tool_name: str, **expected_input: Any) -> "AssertionProxy":
        """Assert *tool_name* was called at least once with all *expected_input* key/values."""
        inputs = self._tool_inputs(tool_name)
        assert inputs, f"agentprobe: tool '{tool_name}' was never called"
        for actual in inputs:
            if all(actual.get(k) == v for k, v in expected_input.items()):
                return self
        raise AssertionError(
            f"agentprobe: tool '{tool_name}' was called {len(inputs)} time(s) "
            f"but never with {expected_input!r}.\n"
            f"Actual inputs seen: {inputs}"
        )

    def assert_tool_call_count(self, tool_name: str, n: int) -> "AssertionProxy":
        """Assert *tool_name* was called exactly *n* times."""
        actual = len(self._tool_inputs(tool_name))
        assert actual == n, (
            f"agentprobe: expected tool '{tool_name}' to be called {n} time(s), "
            f"got {actual}"
        )
        return self

    def assert_tool_called_before(self, first: str, second: str) -> "AssertionProxy":
        first_idx = self._first_tool_index(first)
        second_idx = self._first_tool_index(second)
        assert first_idx is not None, f"agentprobe: tool '{first}' was never called"
        assert second_idx is not None, f"agentprobe: tool '{second}' was never called"
        assert first_idx < second_idx, (
            f"agentprobe: expected '{first}' to be called before '{second}', "
            f"but order was reversed (LLM call indices {first_idx} vs {second_idx})"
        )
        return self

    # ── Output assertions ─────────────────────────────────────────────────

    def assert_output_contains(self, text: str) -> "AssertionProxy":
        out = self.final_output
        assert out is not None, "agentprobe: no text output found in session"
        assert text in out, (
            f"agentprobe: expected output to contain {text!r}, "
            f"got: {out[:300]!r}"
        )
        return self

    def assert_output_not_contains(self, text: str) -> "AssertionProxy":
        out = self.final_output
        if out:
            assert text not in out, (
                f"agentprobe: expected output NOT to contain {text!r}"
            )
        return self

    def assert_all_outputs_contain(self, text: str) -> "AssertionProxy":
        """Assert *text* appears in every text block across all calls."""
        texts = self._all_text_blocks()
        assert texts, "agentprobe: no text output found in session"
        for t in texts:
            assert text in t, (
                f"agentprobe: expected all text blocks to contain {text!r}, "
                f"but found block that does not: {t[:200]!r}"
            )
        return self

    def assert_stop_reason(self, reason: str) -> "AssertionProxy":
        assert self._calls, "agentprobe: no calls in session"
        choices = self._calls[-1].response.get("choices", [])
        actual = choices[-1]["finish_reason"] if choices else None
        assert actual == reason, (
            f"agentprobe: expected finish_reason '{reason}', got '{actual}'"
        )
        return self

    # ── Token / cost assertions ───────────────────────────────────────────

    def assert_max_tokens(self, n: int) -> "AssertionProxy":
        actual = self.total_tokens
        assert actual <= n, (
            f"agentprobe: expected at most {n} total tokens, got {actual} "
            f"({self.total_input_tokens} in + {self.total_output_tokens} out)"
        )
        return self

    def assert_max_cost(self, usd: float) -> "AssertionProxy":
        """Assert the estimated cost is at most *usd* dollars."""
        actual = self.estimated_cost_usd
        assert actual <= usd, (
            f"agentprobe: expected cost <= ${usd:.4f}, got ${actual:.4f}"
        )
        return self

    # ── Timing assertions ─────────────────────────────────────────────────

    def assert_max_duration_ms(self, ms: float) -> "AssertionProxy":
        """Assert the total wall-clock time for all recorded calls is at most *ms* ms."""
        actual = self.total_duration_ms
        assert actual <= ms, (
            f"agentprobe: expected total duration <= {ms:.0f}ms, got {actual:.0f}ms"
        )
        return self

    # ── Model assertions ──────────────────────────────────────────────────

    def assert_model_used(self, model: str) -> "AssertionProxy":
        """Assert every call in the session used *model*."""
        for i, call in enumerate(self._calls):
            actual = call.request.get("model", "")
            assert actual == model, (
                f"agentprobe: call {i+1} used model '{actual}', expected '{model}'"
            )
        return self

    def assert_no_tool_calls(self) -> "AssertionProxy":
        """Assert no tool calls were made in the entire session."""
        names = self._all_tool_names()
        assert not names, (
            f"agentprobe: expected no tool calls but got: {sorted(names)}"
        )
        return self

    def assert_finish_reason_all(self, reason: str) -> "AssertionProxy":
        """Assert every call in the session ended with *reason*."""
        for i, call in enumerate(self._calls):
            choices = call.response.get("choices", [])
            actual = choices[-1]["finish_reason"] if choices else None
            assert actual == reason, (
                f"agentprobe: call {i+1} has finish_reason '{actual}', expected '{reason}'"
            )
        return self

    def assert_all_calls_consumed(self) -> "AssertionProxy":
        """Assert that *all* fixture calls were consumed during the test.

        Real enforcement uses ``Session.replay(strict=True)`` which raises on
        under-consumption. This method is a post-context alias for chaining.
        """
        return self

    def assert_response_json_at(self, n: int, **expected: Any) -> "AssertionProxy":
        """Assert the *n*-th call's text output is JSON containing *expected* key/values.

        Shortcut for ``probe.call(n).assert_output_json_contains(**expected)``::

            probe.assert_response_json_at(1, status="ok", count=3)
        """
        return self.call(n).assert_output_json_contains(**expected)

    def received_at(self, n: int) -> Dict[str, Any]:
        """Return the assistant message received at the *n*-th call (0-indexed).

        Returns a dict with ``content``, ``tool_calls``, and ``finish_reason``::

            msg = probe.received_at(0)
            assert msg["tool_calls"][0]["function"]["name"] == "bash"
        """
        msgs = self.messages_received
        if n < 0 or n >= len(msgs):
            raise IndexError(
                f"agentprobe: call index {n} out of range "
                f"(session has {len(msgs)} call(s))"
            )
        return msgs[n]

    # ── Cost / token per-call assertions ─────────────────────────────────

    def assert_cost_per_call(self, usd: float) -> "AssertionProxy":
        """Assert no individual call exceeded *usd* in estimated cost."""
        for i, call in enumerate(self._calls):
            model = call.request.get("model", "")
            usage = call.response.get("usage") or {}
            cost = estimate_cost(
                model,
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
            )
            assert cost <= usd, (
                f"agentprobe: call {i + 1} estimated cost ${cost:.6f} exceeds "
                f"limit ${usd:.6f} (model={model!r})"
            )
        return self

    def assert_messages_count(self, n: int) -> "AssertionProxy":
        """Assert the total number of messages sent across all calls equals *n*.

        Counts the ``messages`` list in each request::

            # A two-turn agent typically sends 1, then 3 messages = 4 total
            probe.assert_messages_count(4)
        """
        total = sum(len(call.request.get("messages", [])) for call in self._calls)
        assert total == n, (
            f"agentprobe: expected {n} total messages sent, got {total}"
        )
        return self

    def assert_all_models_in(self, *allowed: str) -> "AssertionProxy":
        """Assert every call used one of *allowed* model names.

        Useful for enforcing model governance in CI::

            probe.assert_all_models_in("gpt-4o", "gpt-4o-mini")
        """
        allowed_set = set(allowed)
        for i, call in enumerate(self._calls):
            model = call.request.get("model", "")
            assert model in allowed_set, (
                f"agentprobe: call {i + 1} used disallowed model '{model}'. "
                f"Allowed: {sorted(allowed_set)}"
            )
        return self

    def assert_no_empty_responses(self) -> "AssertionProxy":
        """Assert every call returned either text content or tool calls — no blank replies."""
        for i, call in enumerate(self._calls):
            choices = call.response.get("choices", [])
            for choice in choices:
                msg = choice.get("message", {})
                has_content = bool(msg.get("content"))
                has_tools = bool(msg.get("tool_calls"))
                assert has_content or has_tools, (
                    f"agentprobe: call {i + 1} returned an empty response "
                    f"(no content and no tool_calls)"
                )
        return self

    # ── Session export / comparison ───────────────────────────────────────

    def export_json(self, indent: int = 2) -> str:
        """Export the full session as a JSON string — useful for CI artefacts.

        Output structure: ``{"summary": {...}, "calls": [...]}``
        """
        return json.dumps({
            "summary": self.summary_dict(),
            "calls": self.call_log,
        }, indent=indent, default=str)

    def matches_fixture(self, path: Union[str, Path], *,
                        ignore_content: bool = False) -> "AssertionProxy":
        """Assert this session's tool/model/finish_reason trace matches *path*.

        Useful for regression testing — after refactoring an agent, confirm it
        makes exactly the same API calls as the golden fixture::

            with session.record("new_run.jsonl") as probe:
                my_agent(client)
            probe.matches_fixture("golden.jsonl")

        Set *ignore_content* to skip text output comparison (only checks
        structure: models, tools called, finish reasons).
        """
        golden_calls = _load_calls(Path(path))

        assert len(self._calls) == len(golden_calls), (
            f"agentprobe: fixture mismatch — session has {len(self._calls)} "
            f"call(s), golden fixture has {len(golden_calls)}"
        )

        for i, (actual, golden) in enumerate(zip(self._calls, golden_calls)):
            a_model = actual.request.get("model")
            g_model = golden.request.get("model")
            assert a_model == g_model, (
                f"agentprobe: call {i + 1} model mismatch: "
                f"{a_model!r} (actual) != {g_model!r} (golden)"
            )

            a_choices = actual.response.get("choices", [])
            g_choices = golden.response.get("choices", [])
            a_finish = a_choices[-1]["finish_reason"] if a_choices else None
            g_finish = g_choices[-1]["finish_reason"] if g_choices else None
            assert a_finish == g_finish, (
                f"agentprobe: call {i + 1} finish_reason mismatch: "
                f"{a_finish!r} (actual) != {g_finish!r} (golden)"
            )

            a_tools = sorted(
                tc["function"]["name"]
                for ch in a_choices
                for tc in (ch["message"].get("tool_calls") or [])
            )
            g_tools = sorted(
                tc["function"]["name"]
                for ch in g_choices
                for tc in (ch["message"].get("tool_calls") or [])
            )
            assert a_tools == g_tools, (
                f"agentprobe: call {i + 1} tools mismatch: "
                f"{a_tools} (actual) != {g_tools} (golden)"
            )

            if not ignore_content:
                a_text = next(
                    (ch["message"].get("content") for ch in reversed(a_choices)
                     if ch["message"].get("content")), None
                )
                g_text = next(
                    (ch["message"].get("content") for ch in reversed(g_choices)
                     if ch["message"].get("content")), None
                )
                assert a_text == g_text, (
                    f"agentprobe: call {i + 1} content mismatch:\n"
                    f"  actual:  {a_text!r}\n"
                    f"  golden:  {g_text!r}"
                )

        return self

    def assert_response_latency_percentile(self, p: float, ms: float) -> "AssertionProxy":
        """Assert the *p*-th percentile of recorded call durations is under *ms*.

        ``p`` is 0–100. Requires at least one call with a recorded ``duration_ms``::

            probe.assert_response_latency_percentile(95, 3000)  # p95 under 3 s
        """
        durations = sorted(c.duration_ms for c in self._calls if c.duration_ms is not None)
        if not durations:
            return self
        idx = max(0, int(len(durations) * p / 100) - 1) if p < 100 else len(durations) - 1
        value = durations[idx]
        assert value <= ms, (
            f"agentprobe: p{p:.0f} latency {value:.1f}ms exceeds limit {ms:.1f}ms"
        )
        return self

    def assert_all_responses_under_tokens(self, n: int) -> "AssertionProxy":
        """Assert every individual API call returned fewer than *n* total tokens.

        Complements ``assert_max_tokens`` which checks the session total::

            probe.assert_all_responses_under_tokens(1000)  # each call under 1K tokens
        """
        for i, call in enumerate(self._calls):
            usage = call.response.get("usage") or {}
            total = (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0)
            if total > 0:
                assert total <= n, (
                    f"agentprobe: call {i + 1} used {total} token(s), exceeds per-call limit {n}"
                )
        return self

    def assert_tool_arg_type(self, tool_name: str, key: str,
                             expected_type: str) -> "AssertionProxy":
        """Assert all calls to *tool_name* have ``input[key]`` of type *expected_type*.

        *expected_type* is a string: ``"str"``, ``"int"``, ``"float"``, ``"bool"``,
        ``"list"``, ``"dict"``, or ``"null"``::

            probe.assert_tool_arg_type("search", "page", "int")
        """
        _type_map = {
            "str": str, "int": int, "float": float, "bool": bool,
            "list": list, "dict": dict, "null": type(None),
        }
        if expected_type not in _type_map:
            raise ValueError(
                f"agentprobe: unknown type {expected_type!r}. "
                f"Supported: {sorted(_type_map)}"
            )
        expected = _type_map[expected_type]
        inputs = self._tool_inputs(tool_name)
        assert inputs, f"agentprobe: tool '{tool_name}' was never called"
        for i, inp in enumerate(inputs):
            assert key in inp, (
                f"agentprobe: tool '{tool_name}' call {i + 1} missing key '{key}'"
            )
            actual = type(inp[key])
            if expected_type == "bool":
                assert isinstance(inp[key], bool), (
                    f"agentprobe: tool '{tool_name}' call {i + 1} '{key}' is {actual.__name__}, "
                    f"expected bool"
                )
            else:
                assert isinstance(inp[key], expected) and not (expected_type == "int" and isinstance(inp[key], bool)), (
                    f"agentprobe: tool '{tool_name}' call {i + 1} '{key}' is {actual.__name__}, "
                    f"expected {expected_type}"
                )
        return self

    def assert_no_empty_system_prompt(self) -> "AssertionProxy":
        """Assert every call that included a system message had non-empty content.

        Catches agents where the system prompt is set to ``""`` or ``None``::

            probe.assert_no_empty_system_prompt()
        """
        for i, call in enumerate(self._calls):
            for msg in (call.request.get("messages") or []):
                if msg.get("role") == "system":
                    content = msg.get("content") or ""
                    if isinstance(content, list):
                        has_text = any(
                            p.get("text", "").strip() if isinstance(p, dict) else str(p).strip()
                            for p in content
                        )
                    else:
                        has_text = bool(str(content).strip())
                    assert has_text, (
                        f"agentprobe: call {i + 1} has an empty system message"
                    )
        return self

    def assert_tool_inputs_unique(self, tool_name: str) -> "AssertionProxy":
        """Assert no two calls to *tool_name* used identical serialized inputs.

        Detects redundant/repeated tool invocations across the session::

            probe.assert_tool_inputs_unique("search")
        """
        inputs = self._tool_inputs(tool_name)
        serialized = [json.dumps(inp, sort_keys=True) for inp in inputs]
        seen: set = set()
        for i, s in enumerate(serialized):
            assert s not in seen, (
                f"agentprobe: tool '{tool_name}' call {i + 1} duplicates a previous input: "
                f"{inputs[i]}"
            )
            seen.add(s)
        return self

    def assert_output_not_empty(self) -> "AssertionProxy":
        """Assert the final text output is non-empty and non-whitespace::

            probe.assert_output_not_empty()
        """
        out = self.final_output
        assert out and out.strip(), (
            "agentprobe: final output is empty or whitespace-only"
        )
        return self

    def assert_tool_never_called_with(self, tool_name: str,
                                      **forbidden_input: Any) -> "AssertionProxy":
        """Assert *tool_name* was never called with all of *forbidden_input* key/values.

        The inverse of ``assert_tool_called_with``::

            # Assert agent never searched with safe_search disabled
            probe.assert_tool_never_called_with("search", safe_search=False)
        """
        inputs = self._tool_inputs(tool_name)
        for inp in inputs:
            if all(inp.get(k) == v for k, v in forbidden_input.items()):
                raise AssertionError(
                    f"agentprobe: tool '{tool_name}' WAS called with forbidden input "
                    f"{forbidden_input}. Actual: {inp}"
                )
        return self

    def assert_response_format(self, fmt: str) -> "AssertionProxy":
        """Assert the final output matches a named format: ``'json'`` or ``'markdown'``.

        ``'json'`` — output must parse as valid JSON.
        ``'markdown'`` — output must contain at least one Markdown heading or list.

        ::

            probe.assert_response_format("json")
        """
        text = self.final_output or ""
        if fmt == "json":
            try:
                json.loads(text)
            except json.JSONDecodeError as e:
                raise AssertionError(
                    f"agentprobe: expected output to be valid JSON: {e}. "
                    f"Output: {text[:200]!r}"
                ) from None
        elif fmt == "markdown":
            import re
            has_heading = bool(re.search(r'^#{1,6} ', text, re.MULTILINE))
            has_list = bool(re.search(r'^\s*[-*+] ', text, re.MULTILINE))
            has_numbered = bool(re.search(r'^\s*\d+\. ', text, re.MULTILINE))
            assert has_heading or has_list or has_numbered, (
                f"agentprobe: expected output in markdown format (headings/lists), "
                f"but none found. Output: {text[:200]!r}"
            )
        else:
            raise ValueError(
                f"agentprobe: unknown format {fmt!r}. Supported: 'json', 'markdown'"
            )
        return self

    def assert_prompt_growth_bounded(self, max_ratio: float) -> "AssertionProxy":
        """Assert each call's prompt tokens are < *max_ratio* × the previous call's.

        More granular than ``assert_context_growth`` — checks every consecutive pair::

            probe.assert_prompt_growth_bounded(2.0)  # no call may double the previous
        """
        prev: Optional[int] = None
        for i, call in enumerate(self._calls):
            curr = (call.response.get("usage") or {}).get("prompt_tokens")
            if curr and prev:
                ratio = curr / prev
                assert ratio <= max_ratio, (
                    f"agentprobe: prompt tokens grew {ratio:.2f}x from call {i} to {i + 1} "
                    f"(limit: {max_ratio}x). {prev} → {curr}"
                )
            if curr:
                prev = curr
        return self

    def assert_first_response_latency_under(self, ms: float) -> "AssertionProxy":
        """Assert the first recorded call completed within *ms* milliseconds.

        Useful for cold-start / time-to-first-token SLAs::

            probe.assert_first_response_latency_under(1500)
        """
        if not self._calls or self._calls[0].duration_ms is None:
            return self
        first = self._calls[0].duration_ms
        assert first <= ms, (
            f"agentprobe: first call took {first:.1f}ms, exceeds cold-start limit {ms:.1f}ms"
        )
        return self

    def assert_output_contains_all(self, *substrings: str) -> "AssertionProxy":
        """Assert all *substrings* appear in the final output.

        Shorthand for chaining multiple ``assert_output_contains`` calls::

            probe.assert_output_contains_all("success", "3 results", "cached")
        """
        out = self.final_output or ""
        missing = [s for s in substrings if s not in out]
        assert not missing, (
            f"agentprobe: final output missing: {missing}\n"
            f"Output: {out[:200]!r}"
        )
        return self

    def assert_tool_call_args_match(self, tool_name: str, pattern: str) -> "AssertionProxy":
        """Assert at least one call to *tool_name* has serialized input matching *pattern* (regex).

        Useful when you want a flexible structural check without full schema
        validation::

            probe.assert_tool_call_args_match("search", r'"language":\\s*"en"')
        """
        import re
        inputs = self._tool_inputs(tool_name)
        assert inputs, f"agentprobe: tool '{tool_name}' was never called"
        for inp in inputs:
            if re.search(pattern, json.dumps(inp)):
                return self
        raise AssertionError(
            f"agentprobe: no call to '{tool_name}' matched pattern {pattern!r}. "
            f"Inputs: {inputs}"
        )

    def assert_tool_called_n_times(self, tool_name: str, n: int) -> "AssertionProxy":
        """Assert *tool_name* was called exactly *n* times across the entire session.

        Complements ``assert_tool_call_count_per_call`` which checks per-call counts::

            probe.assert_tool_called_n_times("search", 3)
        """
        total = sum(
            tc["function"]["name"] == tool_name
            for call in self._calls
            for choice in call.response.get("choices", [])
            for tc in (choice.get("message", {}).get("tool_calls") or [])
        )
        assert total == n, (
            f"agentprobe: tool '{tool_name}' was called {total} time(s) total, expected {n}"
        )
        return self

    def assert_no_sensitive_in_messages(self, *patterns: str) -> "AssertionProxy":
        """Assert no outgoing request message content matches any of *patterns* (regex).

        Guards against agents forwarding PII or secrets in the conversation
        messages sent to the API (not just tool inputs)::

            probe.assert_no_sensitive_in_messages(r"sk-[A-Za-z0-9]{32,}")  # API key leak
        """
        import re
        for i, call in enumerate(self._calls):
            for msg in (call.request.get("messages") or []):
                content = msg.get("content") or ""
                if isinstance(content, list):
                    content = " ".join(
                        p.get("text", "") if isinstance(p, dict) else str(p)
                        for p in content
                    )
                for pat in patterns:
                    m = re.search(pat, str(content))
                    if m:
                        raise AssertionError(
                            f"agentprobe: call {i + 1} message matched sensitive pattern "
                            f"{pat!r}: {m.group()!r}"
                        )
        return self

    def assert_tool_input_contains(self, tool_name: str, key: str,
                                   value: Any) -> "AssertionProxy":
        """Assert at least one call to *tool_name* had *key* = *value* in its input.

        Shorthand for ``assert_tool_called_with`` when you only care about one key::

            probe.assert_tool_input_contains("search", "language", "en")
        """
        inputs = self._tool_inputs(tool_name)
        assert inputs, f"agentprobe: tool '{tool_name}' was never called"
        for inp in inputs:
            if inp.get(key) == value:
                return self
        raise AssertionError(
            f"agentprobe: tool '{tool_name}' was never called with {key}={value!r}. "
            f"Actual inputs: {inputs}"
        )

    def assert_tool_result_contains(self, tool_name: str, text: str) -> "AssertionProxy":
        """Assert at least one result returned for *tool_name* contains *text*.

        Checks the ``role: "tool"`` messages fed back to the model, not the
        inputs sent to the tool.  Useful for verifying that downstream tool
        output actually contains expected data::

            probe.assert_tool_result_contains("search", "agentprobe")
        """
        # Build id → name map from all responses
        id_to_name: dict = {}
        for call in self._calls:
            for choice in call.response.get("choices", []):
                for tc in (choice.get("message", {}).get("tool_calls") or []):
                    id_to_name[tc["id"]] = tc["function"]["name"]

        for call in self._calls:
            for msg in (call.request.get("messages") or []):
                if msg.get("role") != "tool":
                    continue
                if id_to_name.get(msg.get("tool_call_id", "")) != tool_name:
                    continue
                content = msg.get("content") or ""
                if text in content:
                    return self

        raise AssertionError(
            f"agentprobe: no result for tool '{tool_name}' contained {text!r}"
        )

    def assert_final_tool_not_called(self, tool_name: str) -> "AssertionProxy":
        """Assert the last API call did not invoke *tool_name*.

        Guards against agents that finish on a tool call instead of a final
        text response::

            probe.assert_final_tool_not_called("search")
        """
        if not self._calls:
            return self
        last = self._calls[-1]
        names = [
            tc["function"]["name"]
            for choice in last.response.get("choices", [])
            for tc in (choice.get("message", {}).get("tool_calls") or [])
        ]
        assert tool_name not in names, (
            f"agentprobe: final call invoked '{tool_name}' — agent did not finish cleanly"
        )
        return self

    def assert_output_word_count(self, min_words: int = 0,
                                 max_words: Optional[int] = None) -> "AssertionProxy":
        """Assert the final output word count is within [*min_words*, *max_words*].

        Useful for catching over-verbose or too-terse responses::

            probe.assert_output_word_count(5, 500)
        """
        text = self.final_output or ""
        words = len(text.split()) if text.strip() else 0
        assert words >= min_words, (
            f"agentprobe: output has {words} word(s), expected at least {min_words}"
        )
        if max_words is not None:
            assert words <= max_words, (
                f"agentprobe: output has {words} word(s), expected at most {max_words}"
            )
        return self

    def assert_output_char_count(self, min_chars: int = 0,
                                 max_chars: Optional[int] = None) -> "AssertionProxy":
        """Assert the final output character count is within [*min_chars*, *max_chars*].

        Complements ``assert_output_word_count`` for byte-precise length checks::

            probe.assert_output_char_count(10, 2000)
        """
        text = self.final_output or ""
        chars = len(text)
        assert chars >= min_chars, (
            f"agentprobe: output has {chars} char(s), expected at least {min_chars}"
        )
        if max_chars is not None:
            assert chars <= max_chars, (
                f"agentprobe: output has {chars} char(s), expected at most {max_chars}"
            )
        return self

    def assert_no_pii_in_tool_inputs(self, *patterns: str) -> "AssertionProxy":
        """Assert none of the tool input arguments match any of *patterns* (regex).

        Guards against agents accidentally forwarding PII (emails, phone numbers,
        API keys) into tool calls::

            probe.assert_no_pii_in_tool_inputs(
                r"[\\w.+-]+@[\\w-]+\\.[\\w.]+",  # email
                r"\\b\\d{3}-\\d{2}-\\d{4}\\b",   # SSN
            )
        """
        import re
        for i, call in enumerate(self._calls):
            for choice in call.response.get("choices", []):
                for tc in (choice.get("message", {}).get("tool_calls") or []):
                    name = tc["function"]["name"]
                    args_str = tc["function"].get("arguments", "")
                    for pat in patterns:
                        m = re.search(pat, args_str)
                        if m:
                            raise AssertionError(
                                f"agentprobe: call {i + 1} tool '{name}' input matched "
                                f"PII pattern {pat!r}: {m.group()!r}"
                            )
        return self

    def assert_tool_call_count_per_call(self, tool_name: str, n: int) -> "AssertionProxy":
        """Assert every iteration that called *tool_name* called it exactly *n* times.

        Useful for multi-step agents where each turn must invoke a tool a fixed
        number of times::

            # Each API call that used "search" must have called it exactly once
            probe.assert_tool_call_count_per_call("search", 1)
        """
        for i, call in enumerate(self._calls):
            names = [
                tc["function"]["name"]
                for choice in call.response.get("choices", [])
                for tc in (choice.get("message", {}).get("tool_calls") or [])
            ]
            count = names.count(tool_name)
            if count > 0:
                assert count == n, (
                    f"agentprobe: call {i + 1} invoked tool '{tool_name}' {count} time(s), "
                    f"expected {n}"
                )
        return self

    def assert_no_tool_call_cycles(self) -> "AssertionProxy":
        """Assert the agent never called the same tool twice in a row.

        Detects the common stuck-loop pattern where an agent calls tool A,
        gets no result, then calls tool A again::

            probe.assert_no_tool_call_cycles()
        """
        ordered = self._tools_in_call_order()
        for i in range(1, len(ordered)):
            assert ordered[i] != ordered[i - 1], (
                f"agentprobe: tool '{ordered[i]}' was called twice in a row "
                f"(positions {i} and {i + 1})"
            )
        return self

    def assert_tool_call_order(self, *names: str) -> "AssertionProxy":
        """Assert tools were called in *names* order (allowing other calls between them).

        Unlike ``assert_tool_sequence``, this is a partial-order check — other
        tools may appear between the specified ones::

            # search must come before summarize, but other tools can appear in between
            probe.assert_tool_call_order("search", "summarize")
        """
        actual = self._tools_in_call_order()
        pos = 0
        for name in names:
            found = next((i for i in range(pos, len(actual)) if actual[i] == name), None)
            assert found is not None, (
                f"agentprobe: tool '{name}' not found after position {pos} in call order {actual}"
            )
            pos = found + 1
        return self

    def assert_no_empty_tool_inputs(self) -> "AssertionProxy":
        """Assert every tool call had at least one input argument.

        Catches agents calling tools with empty ``{}`` inputs::

            probe.assert_no_empty_tool_inputs()
        """
        for i, call in enumerate(self._calls):
            for choice in call.response.get("choices", []):
                for tc in (choice.get("message", {}).get("tool_calls") or []):
                    name = tc["function"]["name"]
                    try:
                        args = json.loads(tc["function"].get("arguments", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    assert args, (
                        f"agentprobe: call {i + 1} tool '{name}' was called with empty input"
                    )
        return self

    def assert_average_latency_under(self, ms: float) -> "AssertionProxy":
        """Assert the average ``duration_ms`` across all calls is under *ms*.

        Only counts calls that have a recorded duration::

            probe.assert_average_latency_under(2000)  # avg under 2 s
        """
        durations = [c.duration_ms for c in self._calls if c.duration_ms is not None]
        if not durations:
            return self
        avg = sum(durations) / len(durations)
        assert avg <= ms, (
            f"agentprobe: average latency {avg:.1f}ms exceeds limit {ms:.1f}ms "
            f"({len(durations)} call(s) measured)"
        )
        return self

    def assert_no_repeated_messages(self) -> "AssertionProxy":
        """Assert no consecutive pair of calls sent the same last user message.

        Catches agents stuck in a loop re-sending the identical prompt::

            probe.assert_no_repeated_messages()
        """
        prev_last = None
        for i, call in enumerate(self._calls):
            messages = call.request.get("messages") or []
            user_msgs = [m.get("content") for m in messages if m.get("role") == "user"]
            last = user_msgs[-1] if user_msgs else None
            if last is not None and last == prev_last:
                raise AssertionError(
                    f"agentprobe: call {i + 1} repeated the same user message as call {i}: "
                    f"{last!r:.80}"
                )
            if last is not None:
                prev_last = last
        return self

    def assert_output_language(self, lang: str) -> "AssertionProxy":
        """Assert the final text output is in *lang* (ISO 639-1 code, e.g. ``'en'``).

        Requires ``langdetect`` (``pip install langdetect``)::

            probe.assert_output_language("en")  # assert agent replied in English
        """
        try:
            from langdetect import detect
        except ImportError:
            raise ImportError(
                "agentprobe: langdetect is required for assert_output_language. "
                "Install it with: pip install langdetect"
            )
        text = self.final_output
        assert text, "agentprobe: no text output found to check language"
        detected = detect(text)
        assert detected == lang, (
            f"agentprobe: expected output language '{lang}', detected '{detected}'. "
            f"Output: {text[:100]!r}"
        )
        return self

    def assert_token_ratio(self, call_n: int, max_ratio: float) -> "AssertionProxy":
        """Assert call *call_n* used at most *max_ratio* times the prompt tokens of call 0.

        Useful as a per-call context-growth guard relative to the first call::

            probe.assert_token_ratio(2, 3.0)  # call 2 must use <3x tokens of call 0
        """
        if len(self._calls) < 2 or call_n >= len(self._calls):
            return self
        base = (self._calls[0].response.get("usage") or {}).get("prompt_tokens")
        curr = (self._calls[call_n].response.get("usage") or {}).get("prompt_tokens")
        if base and curr:
            ratio = curr / base
            assert ratio <= max_ratio, (
                f"agentprobe: call {call_n + 1} used {ratio:.2f}x the tokens of call 1 "
                f"(limit: {max_ratio}x). {base} -> {curr} prompt tokens."
            )
        return self

    def assert_no_hallucinated_tool_calls(self, *allowed: str) -> "AssertionProxy":
        """Assert every tool call used a name from *allowed*.

        Guards against agents inventing tool names not in their tool list::

            probe.assert_no_hallucinated_tool_calls("search", "read_file", "bash")
        """
        allowed_set = set(allowed)
        for i, call in enumerate(self._calls):
            for choice in call.response.get("choices", []):
                for tc in (choice.get("message", {}).get("tool_calls") or []):
                    name = tc["function"]["name"]
                    assert name in allowed_set, (
                        f"agentprobe: call {i + 1} used undeclared tool '{name}'. "
                        f"Allowed: {sorted(allowed_set)}"
                    )
        return self

    def assert_max_tool_calls(self, n: int) -> "AssertionProxy":
        """Assert the total number of tool calls across all iterations is at most *n*."""
        total = sum(
            len(choice.get("message", {}).get("tool_calls") or [])
            for call in self._calls
            for choice in call.response.get("choices", [])
        )
        assert total <= n, (
            f"agentprobe: total tool calls {total} exceeds limit {n}"
        )
        return self

    def assert_min_tool_calls(self, n: int) -> "AssertionProxy":
        """Assert the total number of tool calls across all iterations is at least *n*.

        Symmetric counterpart to ``assert_max_tool_calls``; catches agents that
        silently skip tool use when they should be calling tools::

            probe.assert_min_tool_calls(1)  # agent must use at least one tool
        """
        total = sum(
            len(choice.get("message", {}).get("tool_calls") or [])
            for call in self._calls
            for choice in call.response.get("choices", [])
        )
        assert total >= n, (
            f"agentprobe: total tool calls {total} is below minimum {n}"
        )
        return self

    def assert_system_prompt_present(self) -> "AssertionProxy":
        """Assert at least one call included a system message in its request.

        Useful for catching misconfigured agents that forget the system prompt::

            probe.assert_system_prompt_present()
        """
        for call in self._calls:
            messages = call.request.get("messages") or []
            if any(m.get("role") == "system" for m in messages):
                return self
        raise AssertionError(
            "agentprobe: no call included a system message in its request"
        )

    def assert_response_time_under(self, ms: float) -> "AssertionProxy":
        """Assert every call completed within *ms* milliseconds.

        Useful as a latency SLA gate in CI — only meaningful on real recorded
        fixtures where ``duration_ms`` reflects actual network time::

            probe.assert_response_time_under(5000)  # each call under 5 s
        """
        for i, call in enumerate(self._calls):
            if call.duration_ms is not None:
                assert call.duration_ms <= ms, (
                    f"agentprobe: call {i + 1} took {call.duration_ms:.1f}ms, "
                    f"exceeds limit {ms:.1f}ms"
                )
        return self

    def assert_tool_input_schema(self, tool_name: str, schema: Dict[str, Any]) -> "AssertionProxy":
        """Assert every call to *tool_name* has inputs that validate against *schema*.

        Requires the ``jsonschema`` library (``pip install jsonschema``)::

            probe.assert_tool_input_schema("search", {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            })
        """
        try:
            import jsonschema
        except ImportError:
            raise ImportError(
                "agentprobe: jsonschema is required for assert_tool_input_schema. "
                "Install it with: pip install jsonschema"
            )
        inputs = self._tool_inputs(tool_name)
        assert inputs, f"agentprobe: tool '{tool_name}' was never called"
        for i, inp in enumerate(inputs):
            try:
                jsonschema.validate(inp, schema)
            except jsonschema.ValidationError as e:
                raise AssertionError(
                    f"agentprobe: tool '{tool_name}' call {i + 1} input failed "
                    f"schema validation: {e.message}"
                ) from None
        return self

    def assert_tool_called_before_output(self) -> "AssertionProxy":
        """Assert at least one tool was called AND the final response contains text output.

        Verifies the canonical agentic pattern: tool calls first, text answer last::

            probe.assert_tool_called_before_output()
        """
        assert self.tools_called, "agentprobe: no tool calls found in session"
        last_choices = self._calls[-1].response.get("choices", [])
        has_final_text = any(ch["message"].get("content") for ch in last_choices)
        assert has_final_text, (
            "agentprobe: expected final response to contain text output after tool calls, "
            "but the last call has no text content"
        )
        return self

    # ── Duplicate / growth assertions ─────────────────────────────────────

    def assert_no_duplicate_tool_calls(self) -> "AssertionProxy":
        """Assert no tool was called twice with identical arguments.

        Catches agents stuck in a loop repeating the same tool invocation::

            probe.assert_no_duplicate_tool_calls()
        """
        seen: set = set()
        for i, call in enumerate(self._calls):
            for choice in call.response.get("choices", []):
                for tc in (choice.get("message", {}).get("tool_calls") or []):
                    name = tc["function"]["name"]
                    try:
                        args_norm = json.dumps(
                            json.loads(tc["function"]["arguments"]), sort_keys=True
                        )
                    except (json.JSONDecodeError, TypeError):
                        args_norm = tc["function"]["arguments"]
                    key = (name, args_norm)
                    assert key not in seen, (
                        f"agentprobe: call {i + 1} is a duplicate tool call — "
                        f"'{name}' with args {args_norm} was already invoked"
                    )
                    seen.add(key)
        return self

    def assert_context_growth(self, max_ratio: float) -> "AssertionProxy":
        """Assert context size (prompt tokens) never grows by more than *max_ratio*
        between consecutive calls.

        Useful to catch unbounded context accumulation::

            probe.assert_context_growth(2.0)  # context must not double each call
        """
        prev: Optional[int] = None
        for i, call in enumerate(self._calls):
            curr = (call.response.get("usage") or {}).get("prompt_tokens")
            if curr and prev:
                ratio = curr / prev
                assert ratio <= max_ratio, (
                    f"agentprobe: context grew {ratio:.2f}x from call {i} to call {i + 1} "
                    f"(limit: {max_ratio}x). {prev} -> {curr} prompt tokens."
                )
            if curr:
                prev = curr
        return self

    # ── Output regex assertion ────────────────────────────────────────────

    def assert_output_matches(self, pattern: str) -> "AssertionProxy":
        """Assert the final output matches *pattern* (Python regex, re.search)."""
        import re
        out = self.final_output
        assert out is not None, "agentprobe: no text output found in session"
        assert re.search(pattern, out) is not None, (
            f"agentprobe: expected output to match pattern {pattern!r}, "
            f"got: {out[:300]!r}"
        )
        return self

    # ── Tool sequence assertion ───────────────────────────────────────────

    def assert_tool_sequence(self, *names: str) -> "AssertionProxy":
        """Assert tools were called in the exact order given (one per API call).

        Checks the first tool call seen in each API call, in order::

            probe.assert_tool_sequence("search", "bash", "bash")
        """
        actual = []
        for call in self._calls:
            for choice in call.response.get("choices", []):
                tcs = choice.get("message", {}).get("tool_calls") or []
                if tcs:
                    actual.append(tcs[0]["function"]["name"])
                    break
        assert list(names) == actual, (
            f"agentprobe: expected tool sequence {list(names)}, got {actual}"
        )
        return self

    # ── Fixture export ────────────────────────────────────────────────────

    def dump_fixture(self, path: Union[str, Path]) -> "AssertionProxy":
        """Save this session's calls to *path* as a JSONL fixture.

        Useful for persisting inject() sessions or exporting a subset of calls::

            with session.inject(resp1, resp2) as probe:
                my_agent(client)
            probe.dump_fixture("tests/fixtures/my_session.jsonl")
        """
        _save_calls(self._calls, Path(path))
        return self

    # ── Introspection properties ──────────────────────────────────────────

    @property
    def iteration_count(self) -> int:
        return len(self._calls)

    @property
    def tools_called(self) -> List[str]:
        return sorted(self._all_tool_names())

    @property
    def total_duration_ms(self) -> float:
        """Sum of recorded duration_ms across all calls (0.0 for replay-only sessions)."""
        return sum(call.duration_ms or 0.0 for call in self._calls)

    @property
    def call_log(self) -> List[Dict[str, Any]]:
        """Full request/response pairs as plain dicts, suitable for custom assertions."""
        return [
            {"request": call.request, "response": call.response}
            for call in self._calls
        ]

    @property
    def messages_sent(self) -> List[List[Dict[str, Any]]]:
        """Messages sent to the API for each call, in order.

        Returns a list of message lists — one per API call::

            # Check the first call's messages
            assert probe.messages_sent[0][0]["content"] == "list files in /tmp"
        """
        return [call.request.get("messages", []) for call in self._calls]

    @property
    def models_used(self) -> List[str]:
        """Ordered list of model names used across all calls."""
        return [call.request.get("model", "") for call in self._calls]

    @property
    def messages_received(self) -> List[Dict[str, Any]]:
        """Assistant messages received across all calls.

        Each entry has ``call_index``, ``content``, ``tool_calls``,
        and ``finish_reason``::

            for msg in probe.messages_received:
                print(msg["call_index"], msg["content"])
        """
        received = []
        for i, call in enumerate(self._calls):
            for choice in call.response.get("choices", []):
                msg = choice.get("message", {})
                received.append({
                    "call_index": i,
                    "content": msg.get("content"),
                    "tool_calls": msg.get("tool_calls"),
                    "finish_reason": choice.get("finish_reason"),
                })
        return received

    @property
    def tool_call_inputs(self) -> Dict[str, List[Dict[str, Any]]]:
        """Dict mapping each tool name to the list of argument dicts it was called with.

        Useful for bulk inspection without looping::

            inputs = probe.tool_call_inputs
            assert inputs["bash"][0]["command"] == "ls /tmp"
        """
        result: Dict[str, List[Dict[str, Any]]] = {}
        for call in self._calls:
            for choice in call.response.get("choices", []):
                for tc in (choice.get("message", {}).get("tool_calls") or []):
                    name = tc["function"]["name"]
                    args = json.loads(tc["function"]["arguments"])
                    result.setdefault(name, []).append(args)
        return result

    def summary_dict(self) -> Dict[str, Any]:
        """Return a plain dict summarising the session — useful for logging/reporting."""
        return {
            "iteration_count": self.iteration_count,
            "tools_called": self.tools_called,
            "first_tool_called": self.first_tool_called,
            "final_output": self.final_output,
            "total_tokens": self.total_tokens,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "total_duration_ms": self.total_duration_ms,
            "models_used": self.models_used,
        }

    def duration_percentile(self, p: float) -> float:
        """Return the *p*-th percentile of per-call duration_ms (0–100).

        Only counts calls that have a recorded duration.  Returns 0.0 if no
        durations are recorded (replay-only sessions)::

            p50 = probe.duration_percentile(50)
            p95 = probe.duration_percentile(95)
        """
        durations = [call.duration_ms for call in self._calls if call.duration_ms is not None]
        if not durations:
            return 0.0
        durations.sort()
        if p <= 0:
            return durations[0]
        if p >= 100:
            return durations[-1]
        idx = (p / 100) * (len(durations) - 1)
        lo, hi = int(idx), min(int(idx) + 1, len(durations) - 1)
        frac = idx - lo
        return round(durations[lo] + frac * (durations[hi] - durations[lo]), 4)

    @property
    def first_tool_called(self) -> Optional[str]:
        """Name of the very first tool called in the session, or None."""
        for call in self._calls:
            for choice in call.response.get("choices", []):
                for tc in (choice.get("message", {}).get("tool_calls") or []):
                    return tc["function"]["name"]
        return None

    @property
    def last_tool_called(self) -> Optional[str]:
        """Name of the last tool called in the session, or None."""
        result = None
        for call in self._calls:
            for choice in call.response.get("choices", []):
                for tc in (choice.get("message", {}).get("tool_calls") or []):
                    result = tc["function"]["name"]
        return result

    @property
    def final_output(self) -> Optional[str]:
        for call in reversed(self._calls):
            for choice in call.response.get("choices", []):
                content = choice.get("message", {}).get("content")
                if content:
                    return content
        return None

    @property
    def total_input_tokens(self) -> int:
        return sum(
            (call.response.get("usage") or {}).get("prompt_tokens", 0)
            for call in self._calls
        )

    @property
    def total_output_tokens(self) -> int:
        return sum(
            (call.response.get("usage") or {}).get("completion_tokens", 0)
            for call in self._calls
        )

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def estimated_cost_usd(self) -> float:
        """Estimated total cost in USD based on model pricing table."""
        total = 0.0
        for call in self._calls:
            model = call.request.get("model", "")
            usage = call.response.get("usage", {})
            total += estimate_cost(
                model,
                (usage or {}).get("prompt_tokens", 0),
                (usage or {}).get("completion_tokens", 0),
            )
        return round(total, 8)

    def assert_token_efficiency(self, min_ratio: float) -> "AssertionProxy":
        """Assert output_tokens / input_tokens >= *min_ratio*.

        Useful for catching runaway verbosity or near-empty responses::

            probe.assert_token_efficiency(0.1)  # at least 10% output vs input
        """
        inp = self.total_input_tokens
        out = self.total_output_tokens
        if inp == 0:
            return self
        ratio = out / inp
        assert ratio >= min_ratio, (
            f"agentprobe: token efficiency {ratio:.4f} < {min_ratio} "
            f"({out} output / {inp} input tokens)"
        )
        return self

    # ── Per-call access ───────────────────────────────────────────────────

    def call(self, n: int) -> "AssertionProxy":
        """Return an AssertionProxy scoped to just the *n*-th call (0-indexed).

        Useful for asserting on a specific iteration::

            probe.call(0).assert_tool_called("bash")
            probe.call(1).assert_stop_reason("stop")
        """
        if n < 0 or n >= len(self._calls):
            raise IndexError(
                f"agentprobe: call index {n} out of range "
                f"(session has {len(self._calls)} call(s))"
            )
        return AssertionProxy([self._calls[n]])

    # ── JSON output assertions ────────────────────────────────────────────

    def assert_output_is_json(self) -> "AssertionProxy":
        """Assert the final text output is valid JSON."""
        out = self.final_output
        assert out is not None, "agentprobe: no text output found in session"
        try:
            json.loads(out)
        except json.JSONDecodeError as e:
            raise AssertionError(
                f"agentprobe: expected output to be valid JSON, got parse error: {e}\n"
                f"Output: {out[:200]!r}"
            ) from None
        return self

    def assert_output_json_contains(self, **expected: Any) -> "AssertionProxy":
        """Assert the final output is JSON and contains all *expected* key/value pairs."""
        self.assert_output_is_json()
        data = json.loads(self.final_output)  # type: ignore[arg-type]
        for key, value in expected.items():
            assert key in data, (
                f"agentprobe: expected JSON output to contain key '{key}', "
                f"but keys found: {list(data.keys())}"
            )
            assert data[key] == value, (
                f"agentprobe: expected JSON['{key}'] == {value!r}, got {data[key]!r}"
            )
        return self

    # ── Helpers ───────────────────────────────────────────────────────────

    def _tools_in_call_order(self) -> List[str]:
        tools = []
        for call in self._calls:
            for choice in call.response.get("choices", []):
                for tc in (choice.get("message", {}).get("tool_calls") or []):
                    tools.append(tc["function"]["name"])
        return tools

    def _all_tool_names(self):
        names = set()
        for call in self._calls:
            for choice in call.response.get("choices", []):
                for tc in (choice.get("message", {}).get("tool_calls") or []):
                    names.add(tc["function"]["name"])
        return names

    def _tool_inputs(self, tool_name: str) -> List[Dict[str, Any]]:
        inputs = []
        for call in self._calls:
            for choice in call.response.get("choices", []):
                for tc in (choice.get("message", {}).get("tool_calls") or []):
                    if tc["function"]["name"] == tool_name:
                        inputs.append(json.loads(tc["function"]["arguments"]))
        return inputs

    def _first_tool_index(self, tool_name: str) -> Optional[int]:
        for i, call in enumerate(self._calls):
            for choice in call.response.get("choices", []):
                for tc in (choice.get("message", {}).get("tool_calls") or []):
                    if tc["function"]["name"] == tool_name:
                        return i
        return None

    def _all_text_blocks(self) -> List[str]:
        texts = []
        for call in self._calls:
            for choice in call.response.get("choices", []):
                content = choice.get("message", {}).get("content")
                if content:
                    texts.append(content)
        return texts


def _load_calls(path: Path) -> List[RecordedCall]:
    if path.suffix == ".gz":
        import gzip
        opener = gzip.open(path, "rt", encoding="utf-8")
    else:
        opener = open(path)
    with opener as f:
        # Skip _meta header lines transparently — they are not RecordedCall rows
        return [
            RecordedCall(**json.loads(line))
            for line in f
            if line.strip() and "_meta" not in json.loads(line)
        ]


def _build_meta_line(extra: Optional[Dict[str, Any]] = None,
                     label: Optional[str] = None) -> str:
    """Return a JSON _meta header line with version + timestamp."""
    from agentprobe import __version__
    meta: Dict[str, Any] = {
        "agentprobe_version": __version__,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_version": __import__("sys").version.split()[0],
    }
    if label:
        meta["label"] = label
    if extra:
        meta.update(extra)
    return json.dumps({"_meta": meta})


def _save_calls(calls: List[RecordedCall], path: Path,
                meta_extra: Optional[Dict[str, Any]] = None,
                label: Optional[str] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [_build_meta_line(meta_extra, label)]
    for call in calls:
        data: dict = {
            "request": call.request,
            "response": call.response,
            "timestamp": call.timestamp,
            "duration_ms": call.duration_ms,
        }
        if call.chunks is not None:
            data["chunks"] = call.chunks
        lines.append(json.dumps(data))

    content = "\n".join(lines) + "\n"

    if path.suffix == ".gz":
        import gzip
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(content)
    else:
        # Write atomically via a temp file + rename so concurrent readers never
        # see a partially-written fixture. This also avoids deadlock when auto()
        # holds a FileLock while calling record() → _save_calls().
        import tempfile
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(content)
            Path(tmp).replace(path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


def _check_exists(path: Path) -> None:
    # Also accept the compressed twin automatically: replay("foo.jsonl") works
    # even if the file on disk is "foo.jsonl.gz".
    if not path.exists():
        gz = path.with_suffix(path.suffix + ".gz")
        if gz.exists():
            return  # caller will open gz path via _load_calls
        raise FileNotFoundError(
            f"agentprobe: no fixture at '{path}'. "
            "Run with session.record(...) first to capture a session."
        )


class Session:
    """Context-manager entry point for agentprobe record/replay sessions."""

    # ── Sync ─────────────────────────────────────────────────────────────

    @contextmanager
    def record(self, path: Union[str, Path]):
        """Intercept real OpenAI API calls and save them to *path* (JSONL)."""
        path = Path(path)
        calls: List[RecordedCall] = []
        with recording_context(calls):
            yield AssertionProxy(calls)
        _save_calls(calls, path)

    @contextmanager
    def replay(self, path: Union[str, Path], *, strict: bool = False):
        """Replay calls from a previously recorded fixture at *path* (JSONL).

        When *strict=True*, raises if the agent exits without consuming every
        call in the fixture — useful to detect agents that stop early::

            with session.replay("fixture.jsonl", strict=True) as probe:
                result = my_agent(client)
        """
        path = Path(path)
        _check_exists(path)
        # Also resolve gz twin transparently
        if not path.exists():
            path = path.with_suffix(path.suffix + ".gz")
        calls = _load_calls(path)
        index: List[int] = [0]

        if strict:
            from ._interceptor import _strict_replaying_context
            with _strict_replaying_context(calls, index):
                yield AssertionProxy(calls)
            if index[0] < len(calls):
                raise AssertionError(
                    f"agentprobe: strict replay — fixture has {len(calls)} call(s) "
                    f"but the agent only consumed {index[0]}. "
                    "Ensure the agent makes all expected API calls."
                )
        else:
            with replaying_context(calls):
                yield AssertionProxy(calls)

    @contextmanager
    def replay_chain(self, *paths: Union[str, Path]):
        """Replay multiple fixtures end-to-end as a single session.

        Calls from each fixture are served in order — when the first is
        exhausted the next is used transparently::

            with session.replay_chain("warm_up.jsonl", "task.jsonl") as probe:
                agent(client)  # consumes calls from both fixtures in sequence
        """
        all_calls: List[RecordedCall] = []
        for p in paths:
            p = Path(p)
            _check_exists(p)
            if not p.exists():
                p = p.with_suffix(p.suffix + ".gz")
            all_calls.extend(_load_calls(p))
        with replaying_context(all_calls):
            yield AssertionProxy(all_calls)

    @contextmanager
    def auto(self, path: Union[str, Path]):
        """Record if fixture missing or AGENTPROBE_UPDATE=1; replay otherwise.

        xdist-safe: if filelock is installed and another worker is recording the
        same fixture, waits for it to finish then replays rather than double-recording.
        """
        path = Path(path)
        if path.exists() and not _should_update():
            with self.replay(path) as proxy:
                yield proxy
            return

        lock_path = path.with_suffix(path.suffix + ".lock")
        try:
            import filelock as _fl
            lock = _fl.FileLock(str(lock_path), timeout=60)
            try:
                lock.acquire(timeout=0)          # non-blocking
                acquired = True
            except _fl.Timeout:
                acquired = False
        except ImportError:
            lock = None
            acquired = True

        if not acquired:
            # Another worker is recording; wait then replay
            with _fl.FileLock(str(lock_path), timeout=60):
                pass  # waits until the recording worker releases
            with self.replay(path) as proxy:
                yield proxy
        else:
            try:
                # Re-check: another worker may have finished while we waited
                if path.exists() and not _should_update():
                    with self.replay(path) as proxy:
                        yield proxy
                else:
                    with self.record(path) as proxy:
                        yield proxy
            finally:
                if lock is not None:
                    lock.release()

    @contextmanager
    def inject(self, *responses):
        """Replay an explicit sequence of response dicts (or ChatCompletion objects).

        Useful for quick unit tests where you don't want a fixture file on disk::

            resp_dict = {"id": "chatcmpl-x", "object": "chat.completion", ...}
            with session.inject(resp_dict) as probe:
                result = my_agent(client)
            probe.assert_output_contains("hello")

        Each positional argument is one API response, consumed in order.
        Pass a plain dict (validated via ChatCompletion.model_validate) or an
        already-constructed ChatCompletion object.
        """
        calls: List[RecordedCall] = []
        for resp in responses:
            if isinstance(resp, dict):
                validated = deserialize_response(resp)
                calls.append(RecordedCall(request={}, response=validated.model_dump()))
            else:
                calls.append(RecordedCall(request={}, response=resp.model_dump()))
        with replaying_context(calls):
            yield AssertionProxy(calls)

    @contextmanager
    def inject_error(self, exception: Exception):
        """Make the next API call raise *exception* instead of returning a response.

        Useful for testing agent error-handling paths without a real fixture::

            with session.inject_error(openai.RateLimitError(...)) as probe:
                with pytest.raises(openai.RateLimitError):
                    my_agent(client)
        """
        def patched(self_inner, **kwargs):
            raise exception

        with patch.object(openai.resources.chat.completions.Completions, "create", patched):
            yield

    # ── Async ─────────────────────────────────────────────────────────────

    @asynccontextmanager
    async def async_record(self, path: Union[str, Path]):
        """Async version of record() — for agents built with AsyncOpenAI."""
        path = Path(path)
        calls: List[RecordedCall] = []
        async with async_recording_context(calls):
            yield AssertionProxy(calls)
        _save_calls(calls, path)

    @asynccontextmanager
    async def async_replay(self, path: Union[str, Path]):
        """Async version of replay() — for agents built with AsyncOpenAI."""
        path = Path(path)
        _check_exists(path)
        calls = _load_calls(path)
        async with async_replaying_context(calls):
            yield AssertionProxy(calls)

    @asynccontextmanager
    async def async_auto(self, path: Union[str, Path]):
        """Async version of auto() — record-on-first-run, replay thereafter."""
        path = Path(path)
        if path.exists() and not _should_update():
            async with self.async_replay(path) as proxy:
                yield proxy
        else:
            async with self.async_record(path) as proxy:
                yield proxy

    # ── Append mode ───────────────────────────────────────────────────────

    @contextmanager
    def record_append(self, path: Union[str, Path]):
        """Like record() but appends new calls to an existing fixture.

        Useful for building up a fixture incrementally across multiple runs::

            with session.record_append("tests/fixtures/my_agent.jsonl") as probe:
                my_agent(client, "first query")

            with session.record_append("tests/fixtures/my_agent.jsonl") as probe:
                my_agent(client, "second query")
        """
        path = Path(path)
        new_calls: List[RecordedCall] = []
        with recording_context(new_calls):
            yield AssertionProxy(new_calls)
        existing = _load_calls(path) if path.exists() else []
        _save_calls(existing + new_calls, path)

    @asynccontextmanager
    async def async_record_append(self, path: Union[str, Path]):
        """Async version of record_append()."""
        path = Path(path)
        new_calls: List[RecordedCall] = []
        async with async_recording_context(new_calls):
            yield AssertionProxy(new_calls)
        existing = _load_calls(path) if path.exists() else []
        _save_calls(existing + new_calls, path)


# ── Per-client helpers (shared by MultiSession) ───────────────────────────────

def _make_sync_recorder(calls: List[RecordedCall], original):
    def patched(**kwargs):
        start = time.time()
        if kwargs.get("stream"):
            raw = list(original(**kwargs))
            ser = [c.model_dump() for c in raw]
            calls.append(RecordedCall(
                request=serialize_request(kwargs),
                response=_assemble_from_chunks(ser),
                chunks=ser,
                duration_ms=(time.time() - start) * 1000,
            ))
            return MockStream(raw)
        resp = original(**kwargs)
        calls.append(RecordedCall(
            request=serialize_request(kwargs),
            response=resp.model_dump(),
            duration_ms=(time.time() - start) * 1000,
        ))
        return resp
    return patched


def _make_sync_replayer(calls: List[RecordedCall], index: List[int]):
    def patched(**kwargs):
        if index[0] >= len(calls):
            raise RuntimeError(
                f"agentprobe: replay exhausted — fixture has {len(calls)} call(s) "
                f"but the agent made more. Re-record or update the fixture."
            )
        call = calls[index[0]]
        index[0] += 1
        call.request = serialize_request(kwargs)
        if kwargs.get("stream"):
            if call.chunks is None:
                raise RuntimeError(
                    "agentprobe: agent used stream=True but this fixture was recorded "
                    "without streaming. Re-record with stream=True."
                )
            return MockStream([deserialize_chunk(c) for c in call.chunks])
        return deserialize_response(call.response)
    return patched


def _make_async_recorder(calls: List[RecordedCall], original):
    async def patched(**kwargs):
        start = time.time()
        if kwargs.get("stream"):
            raw = [c async for c in await original(**kwargs)]
            ser = [c.model_dump() for c in raw]
            calls.append(RecordedCall(
                request=serialize_request(kwargs),
                response=_assemble_from_chunks(ser),
                chunks=ser,
                duration_ms=(time.time() - start) * 1000,
            ))
            return MockAsyncStream(raw)
        resp = await original(**kwargs)
        calls.append(RecordedCall(
            request=serialize_request(kwargs),
            response=resp.model_dump(),
            duration_ms=(time.time() - start) * 1000,
        ))
        return resp
    return patched


def _make_async_replayer(calls: List[RecordedCall], index: List[int]):
    async def patched(**kwargs):
        if index[0] >= len(calls):
            raise RuntimeError(
                f"agentprobe: replay exhausted — fixture has {len(calls)} call(s) "
                f"but the agent made more. Re-record or update the fixture."
            )
        call = calls[index[0]]
        index[0] += 1
        call.request = serialize_request(kwargs)
        if kwargs.get("stream"):
            if call.chunks is None:
                raise RuntimeError(
                    "agentprobe: agent used stream=True but this fixture was recorded "
                    "without streaming. Re-record with stream=True."
                )
            return MockAsyncStream([deserialize_chunk(c) for c in call.chunks])
        return deserialize_response(call.response)
    return patched


class MultiSession:
    """Per-client record/replay for multi-agent (subagent) scenarios.

    Unlike Session which patches openai at the class level (all clients share
    the same intercept), MultiSession patches each client instance independently.
    This means two agents using different clients can be replayed simultaneously
    without their call sequences interfering with each other.

    Usage::

        multi = MultiSession()
        client_main = openai.OpenAI(api_key="...")
        client_sub  = openai.OpenAI(api_key="...")

        with multi.replay(client_main, "fixtures/main.jsonl") as probe_main:
            with multi.replay(client_sub, "fixtures/sub.jsonl") as probe_sub:
                run_orchestrator(client_main, client_sub)

        probe_main.assert_tool_called("bash")
        probe_sub.assert_max_iterations(3)
    """

    # ── Sync ─────────────────────────────────────────────────────────────

    @contextmanager
    def record(self, client, path: Union[str, Path]):
        """Intercept *client*'s calls and save them to *path* (JSONL)."""
        path = Path(path)
        calls: List[RecordedCall] = []
        original = client.chat.completions.create
        with patch.object(client.chat.completions, "create",
                          _make_sync_recorder(calls, original)):
            yield AssertionProxy(calls)
        _save_calls(calls, path)

    @contextmanager
    def replay(self, client, path: Union[str, Path]):
        """Replay *client*'s calls from a previously recorded fixture."""
        path = Path(path)
        _check_exists(path)
        calls = _load_calls(path)
        with patch.object(client.chat.completions, "create",
                          _make_sync_replayer(calls, [0])):
            yield AssertionProxy(calls)

    @contextmanager
    def auto(self, client, path: Union[str, Path]):
        """Record if fixture missing or AGENTPROBE_UPDATE=1; replay otherwise."""
        path = Path(path)
        if path.exists() and not _should_update():
            with self.replay(client, path) as proxy:
                yield proxy
        else:
            with self.record(client, path) as proxy:
                yield proxy

    # ── Async ─────────────────────────────────────────────────────────────

    @asynccontextmanager
    async def async_record(self, client, path: Union[str, Path]):
        """Async version of record() — for agents using AsyncOpenAI."""
        path = Path(path)
        calls: List[RecordedCall] = []
        original = client.chat.completions.create
        with patch.object(client.chat.completions, "create",
                          _make_async_recorder(calls, original)):
            yield AssertionProxy(calls)
        _save_calls(calls, path)

    @asynccontextmanager
    async def async_replay(self, client, path: Union[str, Path]):
        """Async version of replay() — for agents using AsyncOpenAI."""
        path = Path(path)
        _check_exists(path)
        calls = _load_calls(path)
        with patch.object(client.chat.completions, "create",
                          _make_async_replayer(calls, [0])):
            yield AssertionProxy(calls)

    @asynccontextmanager
    async def async_auto(self, client, path: Union[str, Path]):
        """Async version of auto() — record-on-first-run, replay thereafter."""
        path = Path(path)
        if path.exists() and not _should_update():
            async with self.async_replay(client, path) as proxy:
                yield proxy
        else:
            async with self.async_record(client, path) as proxy:
                yield proxy

    # ── Multi-client chained replay ───────────────────────────────────────

    @contextmanager
    def replay_chain(self, *client_path_pairs):
        """Replay multiple chained fixtures for multiple clients simultaneously.

        Each argument is a ``(client, paths)`` pair where *paths* is a single
        path or list of paths that are concatenated in order for that client::

            with multi.replay_chain(
                (client_orchestrator, ["warmup.jsonl", "task.jsonl"]),
                (client_subagent, "sub.jsonl"),
            ) as probes:
                run_pipeline(client_orchestrator, client_subagent)

            probes[client_orchestrator].assert_tool_called("search")
            probes[client_subagent].assert_max_iterations(3)
        """
        all_patches = []
        all_probes: Dict[Any, "AssertionProxy"] = {}

        for client, paths in client_path_pairs:
            path_list = paths if isinstance(paths, list) else [paths]
            calls: List[RecordedCall] = []
            for p in path_list:
                _check_exists(Path(p))
                calls.extend(_load_calls(Path(p)))
            pat = patch.object(client.chat.completions, "create",
                               _make_sync_replayer(calls, [0]))
            pat.start()
            all_patches.append(pat)
            all_probes[client] = AssertionProxy(calls)

        yield all_probes

        for pat in all_patches:
            pat.stop()


# ── Anthropic support ─────────────────────────────────────────────────────────

class AnthropicAssertionProxy:
    """Fluent assertion interface for sessions recorded against the Anthropic API.

    Mirrors :class:`AssertionProxy` but reads Anthropic ``Message`` response
    format (``content`` list of blocks, ``stop_reason``, ``usage.input_tokens``,
    etc.) rather than OpenAI's ``choices`` structure.
    """

    def __init__(self, calls: List[RecordedCall]):
        self._calls = calls

    # ── Iteration assertions ──────────────────────────────────────────────

    def assert_max_iterations(self, n: int) -> "AnthropicAssertionProxy":
        actual = len(self._calls)
        assert actual <= n, (
            f"agentprobe: expected at most {n} LLM call(s), got {actual}"
        )
        return self

    def assert_min_iterations(self, n: int) -> "AnthropicAssertionProxy":
        actual = len(self._calls)
        assert actual >= n, (
            f"agentprobe: expected at least {n} LLM call(s), got {actual}"
        )
        return self

    def assert_iteration_count(self, n: int) -> "AnthropicAssertionProxy":
        actual = len(self._calls)
        assert actual == n, (
            f"agentprobe: expected exactly {n} LLM call(s), got {actual}"
        )
        return self

    # ── Tool call assertions ──────────────────────────────────────────────

    def assert_tool_called(self, tool_name: str) -> "AnthropicAssertionProxy":
        names = self._all_tool_names()
        assert tool_name in names, (
            f"agentprobe: expected tool '{tool_name}' to be called, "
            f"but only these were called: {sorted(names) or '(none)'}"
        )
        return self

    def assert_not_tool_called(self, tool_name: str) -> "AnthropicAssertionProxy":
        names = self._all_tool_names()
        assert tool_name not in names, (
            f"agentprobe: expected tool '{tool_name}' NOT to be called, but it was"
        )
        return self

    def assert_tool_called_with(self, tool_name: str, **expected_input: Any) -> "AnthropicAssertionProxy":
        """Assert *tool_name* was called at least once with all *expected_input* key/values."""
        inputs = self._tool_inputs(tool_name)
        assert inputs, f"agentprobe: tool '{tool_name}' was never called"
        for actual in inputs:
            if all(actual.get(k) == v for k, v in expected_input.items()):
                return self
        raise AssertionError(
            f"agentprobe: tool '{tool_name}' was called but never with "
            f"the expected inputs: {expected_input}. Actual inputs: {inputs}"
        )

    def assert_tool_call_count(self, tool_name: str, n: int) -> "AnthropicAssertionProxy":
        count = len(self._tool_inputs(tool_name))
        assert count == n, (
            f"agentprobe: expected tool '{tool_name}' to be called {n} time(s), "
            f"got {count}"
        )
        return self

    def assert_no_tool_calls(self) -> "AnthropicAssertionProxy":
        names = self._all_tool_names()
        assert not names, (
            f"agentprobe: expected no tool calls, but these tools were called: {sorted(names)}"
        )
        return self

    def assert_tool_sequence(self, *names: str) -> "AnthropicAssertionProxy":
        """Assert tools were called in the exact order *names* specifies."""
        actual = self.tools_called
        assert list(names) == actual, (
            f"agentprobe: expected tool sequence {list(names)}, got {actual}"
        )
        return self

    # ── Stop reason assertions ────────────────────────────────────────────

    def assert_stop_reason(self, reason: str) -> "AnthropicAssertionProxy":
        if not self._calls:
            raise AssertionError("agentprobe: no calls recorded in session")
        actual = self._calls[-1].response.get("stop_reason")
        assert actual == reason, (
            f"agentprobe: expected stop_reason '{reason}', got '{actual}'"
        )
        return self

    # ── Output assertions ─────────────────────────────────────────────────

    def assert_output_contains(self, substring: str) -> "AnthropicAssertionProxy":
        out = self.final_output or ""
        assert substring in out, (
            f"agentprobe: expected output to contain '{substring}'\n"
            f"Actual output: {out[:200]!r}"
        )
        return self

    def assert_output_not_contains(self, substring: str) -> "AnthropicAssertionProxy":
        out = self.final_output or ""
        assert substring not in out, (
            f"agentprobe: expected output NOT to contain '{substring}'"
        )
        return self

    def assert_output_matches(self, pattern: str) -> "AnthropicAssertionProxy":
        import re
        out = self.final_output or ""
        assert re.search(pattern, out), (
            f"agentprobe: expected output to match pattern '{pattern}'\n"
            f"Actual output: {out[:200]!r}"
        )
        return self

    # ── Token / cost assertions ───────────────────────────────────────────

    def assert_max_tokens(self, n: int) -> "AnthropicAssertionProxy":
        total = self.total_tokens
        assert total <= n, (
            f"agentprobe: total tokens {total} exceeds limit {n}"
        )
        return self

    def assert_max_cost(self, usd: float) -> "AnthropicAssertionProxy":
        cost = self.estimated_cost_usd
        assert cost <= usd, (
            f"agentprobe: estimated cost ${cost:.6f} exceeds limit ${usd:.6f}"
        )
        return self

    def assert_cost_per_call(self, usd: float) -> "AnthropicAssertionProxy":
        from ._pricing import estimate_cost_anthropic
        for i, call in enumerate(self._calls):
            model = call.response.get("model") or call.request.get("model", "")
            usage = call.response.get("usage") or {}
            cost = estimate_cost_anthropic(
                model,
                usage.get("input_tokens", 0) or 0,
                usage.get("output_tokens", 0) or 0,
            )
            assert cost <= usd, (
                f"agentprobe: call {i + 1} estimated cost ${cost:.6f} exceeds "
                f"limit ${usd:.6f} (model={model!r})"
            )
        return self

    def assert_messages_count(self, n: int) -> "AnthropicAssertionProxy":
        total = sum(len(call.request.get("messages", [])) for call in self._calls)
        assert total == n, (
            f"agentprobe: expected {n} total messages sent, got {total}"
        )
        return self

    def assert_model_used(self, model: str) -> "AnthropicAssertionProxy":
        for i, call in enumerate(self._calls):
            actual = call.response.get("model") or call.request.get("model", "")
            assert actual == model, (
                f"agentprobe: call {i + 1} used model '{actual}', expected '{model}'"
            )
        return self

    def assert_all_models_in(self, *allowed: str) -> "AnthropicAssertionProxy":
        allowed_set = set(allowed)
        for i, call in enumerate(self._calls):
            model = call.response.get("model") or call.request.get("model", "")
            assert model in allowed_set, (
                f"agentprobe: call {i + 1} used disallowed model '{model}'. "
                f"Allowed: {sorted(allowed_set)}"
            )
        return self

    def assert_no_empty_responses(self) -> "AnthropicAssertionProxy":
        for i, call in enumerate(self._calls):
            blocks = call.response.get("content") or []
            has_text = any(b.get("type") == "text" and b.get("text") for b in blocks)
            has_tools = any(b.get("type") == "tool_use" for b in blocks)
            assert has_text or has_tools, (
                f"agentprobe: call {i + 1} returned an empty response "
                f"(no text blocks and no tool_use blocks)"
            )
        return self

    def assert_response_latency_percentile(self, p: float, ms: float) -> "AnthropicAssertionProxy":
        """Assert the *p*-th percentile of recorded call durations is under *ms*."""
        durations = sorted(c.duration_ms for c in self._calls if c.duration_ms is not None)
        if not durations:
            return self
        idx = max(0, int(len(durations) * p / 100) - 1) if p < 100 else len(durations) - 1
        value = durations[idx]
        assert value <= ms, (
            f"agentprobe: p{p:.0f} latency {value:.1f}ms exceeds limit {ms:.1f}ms"
        )
        return self

    def assert_all_responses_under_tokens(self, n: int) -> "AnthropicAssertionProxy":
        """Assert every individual API call returned fewer than *n* total tokens."""
        for i, call in enumerate(self._calls):
            usage = call.response.get("usage") or {}
            total = (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0)
            if total > 0:
                assert total <= n, (
                    f"agentprobe: call {i + 1} used {total} token(s), exceeds per-call limit {n}"
                )
        return self

    def assert_tool_arg_type(self, tool_name: str, key: str,
                             expected_type: str) -> "AnthropicAssertionProxy":
        """Assert all calls to *tool_name* have ``input[key]`` of type *expected_type*."""
        _type_map = {
            "str": str, "int": int, "float": float, "bool": bool,
            "list": list, "dict": dict, "null": type(None),
        }
        if expected_type not in _type_map:
            raise ValueError(f"agentprobe: unknown type {expected_type!r}.")
        expected = _type_map[expected_type]
        inputs = self._tool_inputs(tool_name)
        assert inputs, f"agentprobe: tool '{tool_name}' was never called"
        for i, inp in enumerate(inputs):
            assert key in inp, (
                f"agentprobe: tool '{tool_name}' call {i + 1} missing key '{key}'"
            )
            actual = type(inp[key])
            if expected_type == "bool":
                assert isinstance(inp[key], bool), (
                    f"agentprobe: tool '{tool_name}' call {i + 1} '{key}' is {actual.__name__}, "
                    f"expected bool"
                )
            else:
                assert isinstance(inp[key], expected) and not (expected_type == "int" and isinstance(inp[key], bool)), (
                    f"agentprobe: tool '{tool_name}' call {i + 1} '{key}' is {actual.__name__}, "
                    f"expected {expected_type}"
                )
        return self

    def assert_no_empty_system_prompt(self) -> "AnthropicAssertionProxy":
        """Assert every call that included a system prompt had non-empty content."""
        for i, call in enumerate(self._calls):
            sys_prompt = call.request.get("system")
            if sys_prompt is not None:
                if isinstance(sys_prompt, list):
                    has_text = any(
                        p.get("text", "").strip() if isinstance(p, dict) else str(p).strip()
                        for p in sys_prompt
                    )
                else:
                    has_text = bool(str(sys_prompt).strip())
                assert has_text, (
                    f"agentprobe: call {i + 1} has an empty system prompt"
                )
            for msg in (call.request.get("messages") or []):
                if msg.get("role") == "system":
                    content = msg.get("content") or ""
                    assert bool(str(content).strip()), (
                        f"agentprobe: call {i + 1} has an empty system message"
                    )
        return self

    def assert_tool_inputs_unique(self, tool_name: str) -> "AnthropicAssertionProxy":
        """Assert no two calls to *tool_name* used identical inputs."""
        inputs = self._tool_inputs(tool_name)
        serialized = [json.dumps(inp, sort_keys=True) for inp in inputs]
        seen: set = set()
        for i, s in enumerate(serialized):
            assert s not in seen, (
                f"agentprobe: tool '{tool_name}' call {i + 1} duplicates a previous input: "
                f"{inputs[i]}"
            )
            seen.add(s)
        return self

    def assert_output_not_empty(self) -> "AnthropicAssertionProxy":
        """Assert the final text output is non-empty and non-whitespace."""
        out = self.final_output
        assert out and out.strip(), (
            "agentprobe: final output is empty or whitespace-only"
        )
        return self

    def assert_tool_never_called_with(self, tool_name: str,
                                      **forbidden_input: Any) -> "AnthropicAssertionProxy":
        """Assert *tool_name* was never called with all of *forbidden_input* key/values."""
        inputs = self._tool_inputs(tool_name)
        for inp in inputs:
            if all(inp.get(k) == v for k, v in forbidden_input.items()):
                raise AssertionError(
                    f"agentprobe: tool '{tool_name}' WAS called with forbidden input "
                    f"{forbidden_input}. Actual: {inp}"
                )
        return self

    def assert_response_format(self, fmt: str) -> "AnthropicAssertionProxy":
        """Assert the final output matches a named format: ``'json'`` or ``'markdown'``."""
        text = self.final_output or ""
        if fmt == "json":
            try:
                json.loads(text)
            except json.JSONDecodeError as e:
                raise AssertionError(
                    f"agentprobe: expected output to be valid JSON: {e}. "
                    f"Output: {text[:200]!r}"
                ) from None
        elif fmt == "markdown":
            import re
            has_heading = bool(re.search(r'^#{1,6} ', text, re.MULTILINE))
            has_list = bool(re.search(r'^\s*[-*+] ', text, re.MULTILINE))
            has_numbered = bool(re.search(r'^\s*\d+\. ', text, re.MULTILINE))
            assert has_heading or has_list or has_numbered, (
                f"agentprobe: expected output in markdown format (headings/lists), "
                f"but none found. Output: {text[:200]!r}"
            )
        else:
            raise ValueError(
                f"agentprobe: unknown format {fmt!r}. Supported: 'json', 'markdown'"
            )
        return self

    def assert_prompt_growth_bounded(self, max_ratio: float) -> "AnthropicAssertionProxy":
        """Assert each call's input tokens are < *max_ratio* × the previous call's."""
        prev: Optional[int] = None
        for i, call in enumerate(self._calls):
            curr = (call.response.get("usage") or {}).get("input_tokens")
            if curr and prev:
                ratio = curr / prev
                assert ratio <= max_ratio, (
                    f"agentprobe: input tokens grew {ratio:.2f}x from call {i} to {i + 1} "
                    f"(limit: {max_ratio}x). {prev} → {curr}"
                )
            if curr:
                prev = curr
        return self

    def assert_first_response_latency_under(self, ms: float) -> "AnthropicAssertionProxy":
        """Assert the first recorded call completed within *ms* milliseconds."""
        if not self._calls or self._calls[0].duration_ms is None:
            return self
        first = self._calls[0].duration_ms
        assert first <= ms, (
            f"agentprobe: first call took {first:.1f}ms, exceeds cold-start limit {ms:.1f}ms"
        )
        return self

    def assert_output_contains_all(self, *substrings: str) -> "AnthropicAssertionProxy":
        """Assert all *substrings* appear in the final output."""
        out = self.final_output or ""
        missing = [s for s in substrings if s not in out]
        assert not missing, (
            f"agentprobe: final output missing: {missing}\n"
            f"Output: {out[:200]!r}"
        )
        return self

    def assert_tool_call_args_match(self, tool_name: str, pattern: str) -> "AnthropicAssertionProxy":
        """Assert at least one call to *tool_name* has serialized input matching *pattern* (regex)."""
        import re
        inputs = self._tool_inputs(tool_name)
        assert inputs, f"agentprobe: tool '{tool_name}' was never called"
        for inp in inputs:
            if re.search(pattern, json.dumps(inp)):
                return self
        raise AssertionError(
            f"agentprobe: no call to '{tool_name}' matched pattern {pattern!r}. "
            f"Inputs: {inputs}"
        )

    def assert_tool_called_n_times(self, tool_name: str, n: int) -> "AnthropicAssertionProxy":
        """Assert *tool_name* was called exactly *n* times across the entire session."""
        total = sum(
            1 for call in self._calls
            for block in (call.response.get("content") or [])
            if block.get("type") == "tool_use" and block["name"] == tool_name
        )
        assert total == n, (
            f"agentprobe: tool '{tool_name}' was called {total} time(s) total, expected {n}"
        )
        return self

    def assert_no_sensitive_in_messages(self, *patterns: str) -> "AnthropicAssertionProxy":
        """Assert no outgoing request message content matches any of *patterns* (regex)."""
        import re
        for i, call in enumerate(self._calls):
            for msg in (call.request.get("messages") or []):
                content = msg.get("content") or ""
                if isinstance(content, list):
                    content = " ".join(
                        p.get("text", "") if isinstance(p, dict) else str(p)
                        for p in content
                    )
                for pat in patterns:
                    m = re.search(pat, str(content))
                    if m:
                        raise AssertionError(
                            f"agentprobe: call {i + 1} message matched sensitive pattern "
                            f"{pat!r}: {m.group()!r}"
                        )
        return self

    def assert_tool_input_contains(self, tool_name: str, key: str,
                                   value: Any) -> "AnthropicAssertionProxy":
        """Assert at least one call to *tool_name* had *key* = *value* in its input."""
        inputs = self._tool_inputs(tool_name)
        assert inputs, f"agentprobe: tool '{tool_name}' was never called"
        for inp in inputs:
            if inp.get(key) == value:
                return self
        raise AssertionError(
            f"agentprobe: tool '{tool_name}' was never called with {key}={value!r}. "
            f"Actual inputs: {inputs}"
        )

    def assert_tool_result_contains(self, tool_name: str, text: str) -> "AnthropicAssertionProxy":
        """Assert at least one result returned for *tool_name* contains *text*.

        Checks ``tool_result`` blocks fed back in user messages, not inputs sent to the tool.
        """
        # Build id → name map from all responses
        id_to_name: dict = {}
        for call in self._calls:
            for block in (call.response.get("content") or []):
                if block.get("type") == "tool_use":
                    id_to_name[block["id"]] = block["name"]

        for call in self._calls:
            for msg in (call.request.get("messages") or []):
                if msg.get("role") != "user":
                    continue
                content = msg.get("content") or []
                if not isinstance(content, list):
                    continue
                for block in content:
                    if not isinstance(block, dict) or block.get("type") != "tool_result":
                        continue
                    if id_to_name.get(block.get("tool_use_id", "")) != tool_name:
                        continue
                    result_content = block.get("content") or ""
                    if isinstance(result_content, list):
                        result_content = " ".join(
                            b.get("text", "") for b in result_content
                            if isinstance(b, dict)
                        )
                    if text in result_content:
                        return self

        raise AssertionError(
            f"agentprobe: no result for tool '{tool_name}' contained {text!r}"
        )

    def assert_final_tool_not_called(self, tool_name: str) -> "AnthropicAssertionProxy":
        """Assert the last API call did not invoke *tool_name*."""
        if not self._calls:
            return self
        last_blocks = self._calls[-1].response.get("content") or []
        names = [b["name"] for b in last_blocks if b.get("type") == "tool_use"]
        assert tool_name not in names, (
            f"agentprobe: final call invoked '{tool_name}' — agent did not finish cleanly"
        )
        return self

    def assert_output_word_count(self, min_words: int = 0,
                                 max_words: Optional[int] = None) -> "AnthropicAssertionProxy":
        """Assert the final output word count is within [*min_words*, *max_words*]."""
        text = self.final_output or ""
        words = len(text.split()) if text.strip() else 0
        assert words >= min_words, (
            f"agentprobe: output has {words} word(s), expected at least {min_words}"
        )
        if max_words is not None:
            assert words <= max_words, (
                f"agentprobe: output has {words} word(s), expected at most {max_words}"
            )
        return self

    def assert_output_char_count(self, min_chars: int = 0,
                                 max_chars: Optional[int] = None) -> "AnthropicAssertionProxy":
        """Assert the final output character count is within [*min_chars*, *max_chars*]."""
        text = self.final_output or ""
        chars = len(text)
        assert chars >= min_chars, (
            f"agentprobe: output has {chars} char(s), expected at least {min_chars}"
        )
        if max_chars is not None:
            assert chars <= max_chars, (
                f"agentprobe: output has {chars} char(s), expected at most {max_chars}"
            )
        return self

    def assert_no_pii_in_tool_inputs(self, *patterns: str) -> "AnthropicAssertionProxy":
        """Assert no tool input arguments match any of *patterns* (regex)."""
        import re
        for i, call in enumerate(self._calls):
            for block in (call.response.get("content") or []):
                if block.get("type") != "tool_use":
                    continue
                name = block["name"]
                inp_str = json.dumps(block.get("input", {}))
                for pat in patterns:
                    m = re.search(pat, inp_str)
                    if m:
                        raise AssertionError(
                            f"agentprobe: call {i + 1} tool '{name}' input matched "
                            f"PII pattern {pat!r}: {m.group()!r}"
                        )
        return self

    def assert_tool_call_count_per_call(self, tool_name: str, n: int) -> "AnthropicAssertionProxy":
        """Assert every iteration that called *tool_name* called it exactly *n* times."""
        for i, call in enumerate(self._calls):
            count = sum(
                1 for block in (call.response.get("content") or [])
                if block.get("type") == "tool_use" and block["name"] == tool_name
            )
            if count > 0:
                assert count == n, (
                    f"agentprobe: call {i + 1} invoked tool '{tool_name}' {count} time(s), "
                    f"expected {n}"
                )
        return self

    def assert_no_tool_call_cycles(self) -> "AnthropicAssertionProxy":
        """Assert the agent never called the same tool twice in a row."""
        ordered = self._tools_in_call_order()
        for i in range(1, len(ordered)):
            assert ordered[i] != ordered[i - 1], (
                f"agentprobe: tool '{ordered[i]}' was called twice in a row "
                f"(positions {i} and {i + 1})"
            )
        return self

    def assert_tool_call_order(self, *names: str) -> "AnthropicAssertionProxy":
        """Assert tools were called in *names* order (partial-order, gaps allowed)."""
        actual = self._tools_in_call_order()
        pos = 0
        for name in names:
            found = next((i for i in range(pos, len(actual)) if actual[i] == name), None)
            assert found is not None, (
                f"agentprobe: tool '{name}' not found after position {pos} in call order {actual}"
            )
            pos = found + 1
        return self

    def assert_no_empty_tool_inputs(self) -> "AnthropicAssertionProxy":
        """Assert every tool call had at least one input argument."""
        for i, call in enumerate(self._calls):
            for block in (call.response.get("content") or []):
                if block.get("type") == "tool_use":
                    name = block["name"]
                    inp = block.get("input") or {}
                    assert inp, (
                        f"agentprobe: call {i + 1} tool '{name}' was called with empty input"
                    )
        return self

    def assert_average_latency_under(self, ms: float) -> "AnthropicAssertionProxy":
        """Assert the average ``duration_ms`` across all calls is under *ms*."""
        durations = [c.duration_ms for c in self._calls if c.duration_ms is not None]
        if not durations:
            return self
        avg = sum(durations) / len(durations)
        assert avg <= ms, (
            f"agentprobe: average latency {avg:.1f}ms exceeds limit {ms:.1f}ms "
            f"({len(durations)} call(s) measured)"
        )
        return self

    def assert_no_repeated_messages(self) -> "AnthropicAssertionProxy":
        """Assert no consecutive pair of calls sent the same last user message."""
        prev_last = None
        for i, call in enumerate(self._calls):
            messages = call.request.get("messages") or []
            user_msgs = [m.get("content") for m in messages if m.get("role") == "user"]
            last = user_msgs[-1] if user_msgs else None
            if last is not None and last == prev_last:
                raise AssertionError(
                    f"agentprobe: call {i + 1} repeated the same user message as call {i}: "
                    f"{last!r:.80}"
                )
            if last is not None:
                prev_last = last
        return self

    def assert_output_language(self, lang: str) -> "AnthropicAssertionProxy":
        """Assert the final text output is in *lang* (ISO 639-1 code)."""
        try:
            from langdetect import detect
        except ImportError:
            raise ImportError(
                "agentprobe: langdetect is required for assert_output_language. "
                "Install it with: pip install langdetect"
            )
        text = self.final_output
        assert text, "agentprobe: no text output found to check language"
        detected = detect(text)
        assert detected == lang, (
            f"agentprobe: expected output language '{lang}', detected '{detected}'. "
            f"Output: {text[:100]!r}"
        )
        return self

    def assert_token_ratio(self, call_n: int, max_ratio: float) -> "AnthropicAssertionProxy":
        """Assert call *call_n* used at most *max_ratio* times the input tokens of call 0."""
        if len(self._calls) < 2 or call_n >= len(self._calls):
            return self
        base = (self._calls[0].response.get("usage") or {}).get("input_tokens")
        curr = (self._calls[call_n].response.get("usage") or {}).get("input_tokens")
        if base and curr:
            ratio = curr / base
            assert ratio <= max_ratio, (
                f"agentprobe: call {call_n + 1} used {ratio:.2f}x the tokens of call 1 "
                f"(limit: {max_ratio}x). {base} -> {curr} input tokens."
            )
        return self

    def assert_no_hallucinated_tool_calls(self, *allowed: str) -> "AnthropicAssertionProxy":
        """Assert every tool call used a name from *allowed*."""
        allowed_set = set(allowed)
        for i, call in enumerate(self._calls):
            for block in (call.response.get("content") or []):
                if block.get("type") == "tool_use":
                    name = block["name"]
                    assert name in allowed_set, (
                        f"agentprobe: call {i + 1} used undeclared tool '{name}'. "
                        f"Allowed: {sorted(allowed_set)}"
                    )
        return self

    def assert_max_tool_calls(self, n: int) -> "AnthropicAssertionProxy":
        """Assert the total number of tool calls across all iterations is at most *n*."""
        total = sum(
            1 for call in self._calls
            for block in (call.response.get("content") or [])
            if block.get("type") == "tool_use"
        )
        assert total <= n, (
            f"agentprobe: total tool calls {total} exceeds limit {n}"
        )
        return self

    def assert_min_tool_calls(self, n: int) -> "AnthropicAssertionProxy":
        """Assert the total number of tool calls across all iterations is at least *n*."""
        total = sum(
            1 for call in self._calls
            for block in (call.response.get("content") or [])
            if block.get("type") == "tool_use"
        )
        assert total >= n, (
            f"agentprobe: total tool calls {total} is below minimum {n}"
        )
        return self

    def assert_system_prompt_present(self) -> "AnthropicAssertionProxy":
        """Assert at least one call included a system prompt in its request."""
        for call in self._calls:
            if call.request.get("system"):
                return self
            messages = call.request.get("messages") or []
            if any(m.get("role") == "system" for m in messages):
                return self
        raise AssertionError(
            "agentprobe: no call included a system prompt in its request"
        )

    def assert_response_time_under(self, ms: float) -> "AnthropicAssertionProxy":
        """Assert every call completed within *ms* milliseconds."""
        for i, call in enumerate(self._calls):
            if call.duration_ms is not None:
                assert call.duration_ms <= ms, (
                    f"agentprobe: call {i + 1} took {call.duration_ms:.1f}ms, "
                    f"exceeds limit {ms:.1f}ms"
                )
        return self

    def assert_tool_input_schema(self, tool_name: str, schema: Dict[str, Any]) -> "AnthropicAssertionProxy":
        """Assert every call to *tool_name* has inputs that validate against *schema*."""
        try:
            import jsonschema
        except ImportError:
            raise ImportError(
                "agentprobe: jsonschema is required for assert_tool_input_schema. "
                "Install it with: pip install jsonschema"
            )
        inputs = self._tool_inputs(tool_name)
        assert inputs, f"agentprobe: tool '{tool_name}' was never called"
        for i, inp in enumerate(inputs):
            try:
                jsonschema.validate(inp, schema)
            except jsonschema.ValidationError as e:
                raise AssertionError(
                    f"agentprobe: tool '{tool_name}' call {i + 1} input failed "
                    f"schema validation: {e.message}"
                ) from None
        return self

    def assert_tool_called_before_output(self) -> "AnthropicAssertionProxy":
        """Assert at least one tool was called AND the final response contains text output."""
        assert self.tools_called, "agentprobe: no tool calls found in session"
        last_blocks = self._calls[-1].response.get("content") or []
        has_final_text = any(b.get("type") == "text" and b.get("text") for b in last_blocks)
        assert has_final_text, (
            "agentprobe: expected final response to contain text output after tool calls, "
            "but the last call has no text blocks"
        )
        return self

    # ── Duplicate / growth assertions ─────────────────────────────────────

    def assert_no_duplicate_tool_calls(self) -> "AnthropicAssertionProxy":
        seen: set = set()
        for i, call in enumerate(self._calls):
            for block in (call.response.get("content") or []):
                if block.get("type") != "tool_use":
                    continue
                name = block["name"]
                args_norm = json.dumps(block.get("input", {}), sort_keys=True)
                key = (name, args_norm)
                assert key not in seen, (
                    f"agentprobe: call {i + 1} is a duplicate tool call — "
                    f"'{name}' with input {args_norm} was already invoked"
                )
                seen.add(key)
        return self

    def assert_context_growth(self, max_ratio: float) -> "AnthropicAssertionProxy":
        prev: Optional[int] = None
        for i, call in enumerate(self._calls):
            curr = (call.response.get("usage") or {}).get("input_tokens")
            if curr and prev:
                ratio = curr / prev
                assert ratio <= max_ratio, (
                    f"agentprobe: context grew {ratio:.2f}x from call {i} to call {i + 1} "
                    f"(limit: {max_ratio}x). {prev} -> {curr} input tokens."
                )
            if curr:
                prev = curr
        return self

    # ── Per-call access ───────────────────────────────────────────────────

    def call(self, n: int) -> "AnthropicAssertionProxy":
        if n < 0 or n >= len(self._calls):
            raise IndexError(
                f"agentprobe: call index {n} out of range "
                f"(session has {len(self._calls)} call(s))"
            )
        return AnthropicAssertionProxy([self._calls[n]])

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def iteration_count(self) -> int:
        return len(self._calls)

    @property
    def tools_called(self) -> List[str]:
        tools = []
        for call in self._calls:
            for block in (call.response.get("content") or []):
                if block.get("type") == "tool_use":
                    tools.append(block["name"])
        return tools

    @property
    def first_tool_called(self) -> Optional[str]:
        for call in self._calls:
            for block in (call.response.get("content") or []):
                if block.get("type") == "tool_use":
                    return block["name"]
        return None

    @property
    def last_tool_called(self) -> Optional[str]:
        last = None
        for call in self._calls:
            for block in (call.response.get("content") or []):
                if block.get("type") == "tool_use":
                    last = block["name"]
        return last

    @property
    def final_output(self) -> Optional[str]:
        if not self._calls:
            return None
        for block in reversed(self._calls[-1].response.get("content") or []):
            if block.get("type") == "text" and block.get("text"):
                return block["text"]
        return None

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def total_input_tokens(self) -> int:
        return sum(
            (call.response.get("usage") or {}).get("input_tokens", 0) or 0
            for call in self._calls
        )

    @property
    def total_output_tokens(self) -> int:
        return sum(
            (call.response.get("usage") or {}).get("output_tokens", 0) or 0
            for call in self._calls
        )

    @property
    def estimated_cost_usd(self) -> float:
        from ._pricing import estimate_cost_anthropic
        total = 0.0
        for call in self._calls:
            model = call.response.get("model") or call.request.get("model", "")
            usage = call.response.get("usage") or {}
            total += estimate_cost_anthropic(
                model,
                usage.get("input_tokens", 0) or 0,
                usage.get("output_tokens", 0) or 0,
            )
        return round(total, 8)

    @property
    def models_used(self) -> List[str]:
        return [
            call.response.get("model") or call.request.get("model", "")
            for call in self._calls
        ]

    @property
    def call_log(self) -> List[Dict[str, Any]]:
        return [{"request": c.request, "response": c.response} for c in self._calls]

    @property
    def tool_call_inputs(self) -> Dict[str, List[Dict[str, Any]]]:
        result: Dict[str, List[Dict[str, Any]]] = {}
        for call in self._calls:
            for block in (call.response.get("content") or []):
                if block.get("type") == "tool_use":
                    result.setdefault(block["name"], []).append(block.get("input", {}))
        return result

    @property
    def messages_sent(self) -> List[List[Any]]:
        return [call.request.get("messages", []) for call in self._calls]

    # ── Export ────────────────────────────────────────────────────────────

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "iteration_count": self.iteration_count,
            "total_tokens": self.total_tokens,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "tools_called": self.tools_called,
            "models_used": self.models_used,
            "final_output": self.final_output,
        }

    def export_json(self, indent: int = 2) -> str:
        return json.dumps({
            "summary": self.summary_dict(),
            "calls": self.call_log,
        }, indent=indent, default=str)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _tools_in_call_order(self) -> List[str]:
        tools = []
        for call in self._calls:
            for block in (call.response.get("content") or []):
                if block.get("type") == "tool_use":
                    tools.append(block["name"])
        return tools

    def _all_tool_names(self) -> set:
        names: set = set()
        for call in self._calls:
            for block in (call.response.get("content") or []):
                if block.get("type") == "tool_use":
                    names.add(block["name"])
        return names

    def _tool_inputs(self, tool_name: str) -> List[Dict[str, Any]]:
        inputs = []
        for call in self._calls:
            for block in (call.response.get("content") or []):
                if block.get("type") == "tool_use" and block["name"] == tool_name:
                    inputs.append(block.get("input", {}))
        return inputs


class AnthropicSession:
    """Context-manager entry point for agentprobe record/replay with the Anthropic API.

    Works identically to :class:`Session` but patches
    ``anthropic.resources.messages.Messages.create`` (sync) and
    ``AsyncMessages.create`` (async) instead of the OpenAI endpoint.

    Usage::

        import anthropic
        session = AnthropicSession()
        client = anthropic.Anthropic(api_key="...")

        # Record a real session once
        with session.record("tests/fixtures/my_agent.jsonl") as probe:
            my_agent(client)

        # Replay deterministically in CI
        with session.replay("tests/fixtures/my_agent.jsonl") as probe:
            my_agent(client)
        probe.assert_tool_called("search")
        probe.assert_stop_reason("end_turn")
    """

    # ── Sync ─────────────────────────────────────────────────────────────

    @contextmanager
    def record(self, path: Union[str, Path]):
        """Intercept real Anthropic API calls and save them to *path* (JSONL)."""
        from ._anthropic_interceptor import anthropic_recording_context
        path = Path(path)
        calls: List[RecordedCall] = []
        with anthropic_recording_context(calls):
            yield AnthropicAssertionProxy(calls)
        _save_calls(calls, path)

    @contextmanager
    def replay(self, path: Union[str, Path], *, strict: bool = False):
        """Replay calls from *path* without hitting the Anthropic API."""
        from ._anthropic_interceptor import (
            anthropic_replaying_context,
            _anthropic_strict_replaying_context,
        )
        path = Path(path)
        _check_exists(path)
        calls = _load_calls(path)
        index: List[int] = [0]

        if strict:
            with _anthropic_strict_replaying_context(calls, index):
                yield AnthropicAssertionProxy(calls)
            if index[0] < len(calls):
                raise AssertionError(
                    f"agentprobe: strict replay — fixture has {len(calls)} call(s) "
                    f"but the agent only consumed {index[0]}."
                )
        else:
            with anthropic_replaying_context(calls):
                yield AnthropicAssertionProxy(calls)

    @contextmanager
    def replay_chain(self, *paths: Union[str, Path]):
        """Replay multiple fixtures end-to-end as a single session."""
        from ._anthropic_interceptor import anthropic_replaying_context
        all_calls: List[RecordedCall] = []
        for p in paths:
            p = Path(p)
            _check_exists(p)
            all_calls.extend(_load_calls(p))
        with anthropic_replaying_context(all_calls):
            yield AnthropicAssertionProxy(all_calls)

    @contextmanager
    def auto(self, path: Union[str, Path]):
        """Record if fixture missing or AGENTPROBE_UPDATE=1; replay otherwise."""
        path = Path(path)
        if path.exists() and not _should_update():
            with self.replay(path) as proxy:
                yield proxy
        else:
            with self.record(path) as proxy:
                yield proxy

    @contextmanager
    def inject(self, *responses):
        """Replay explicit Anthropic ``Message`` objects (or dicts) without a fixture.

        Pass ``anthropic.types.Message`` objects or raw dicts that can be
        validated by ``Message.model_validate``::

            resp = anthropic.types.Message.model_validate({...})
            with session.inject(resp) as probe:
                my_agent(client)
        """
        import anthropic.types
        from ._anthropic_interceptor import anthropic_replaying_context
        calls: List[RecordedCall] = []
        for resp in responses:
            if isinstance(resp, dict):
                validated = anthropic.types.Message.model_validate(resp)
                calls.append(RecordedCall(request={}, response=validated.model_dump()))
            else:
                calls.append(RecordedCall(request={}, response=resp.model_dump()))
        with anthropic_replaying_context(calls):
            yield AnthropicAssertionProxy(calls)

    @contextmanager
    def record_append(self, path: Union[str, Path]):
        """Like record() but appends new calls to an existing fixture."""
        from ._anthropic_interceptor import anthropic_recording_context
        path = Path(path)
        new_calls: List[RecordedCall] = []
        with anthropic_recording_context(new_calls):
            yield AnthropicAssertionProxy(new_calls)
        existing = _load_calls(path) if path.exists() else []
        _save_calls(existing + new_calls, path)

    # ── Async ─────────────────────────────────────────────────────────────

    @asynccontextmanager
    async def async_record(self, path: Union[str, Path]):
        """Async version of record() — for agents using ``AsyncAnthropic``."""
        from ._anthropic_interceptor import async_anthropic_recording_context
        path = Path(path)
        calls: List[RecordedCall] = []
        async with async_anthropic_recording_context(calls):
            yield AnthropicAssertionProxy(calls)
        _save_calls(calls, path)

    @asynccontextmanager
    async def async_replay(self, path: Union[str, Path]):
        """Async version of replay() — for agents using ``AsyncAnthropic``."""
        from ._anthropic_interceptor import async_anthropic_replaying_context
        path = Path(path)
        _check_exists(path)
        calls = _load_calls(path)
        async with async_anthropic_replaying_context(calls):
            yield AnthropicAssertionProxy(calls)

    @asynccontextmanager
    async def async_auto(self, path: Union[str, Path]):
        """Async version of auto()."""
        path = Path(path)
        if path.exists() and not _should_update():
            async with self.async_replay(path) as proxy:
                yield proxy
        else:
            async with self.async_record(path) as proxy:
                yield proxy


# ── AnthropicMultiSession ─────────────────────────────────────────────────────

def _make_anthropic_sync_recorder(calls: List[RecordedCall], original):
    from ._anthropic_interceptor import _serialize_anthropic_request
    def patched(**kwargs):
        import time as _time
        start = _time.time()
        resp = original(**kwargs)
        calls.append(RecordedCall(
            request=_serialize_anthropic_request(kwargs),
            response=resp.model_dump(),
            duration_ms=(_time.time() - start) * 1000,
        ))
        return resp
    return patched


def _make_anthropic_sync_replayer(calls: List[RecordedCall], index: List[int]):
    from ._anthropic_interceptor import _deserialize_anthropic_response, _serialize_anthropic_request
    def patched(**kwargs):
        if index[0] >= len(calls):
            raise RuntimeError(
                f"agentprobe: replay exhausted — fixture has {len(calls)} call(s) "
                f"but the agent made more. Re-record or update the fixture."
            )
        call = calls[index[0]]
        index[0] += 1
        call.request = _serialize_anthropic_request(kwargs)
        return _deserialize_anthropic_response(call.response)
    return patched


def _make_anthropic_async_recorder(calls: List[RecordedCall], original):
    from ._anthropic_interceptor import _serialize_anthropic_request
    async def patched(**kwargs):
        import time as _time
        start = _time.time()
        resp = await original(**kwargs)
        calls.append(RecordedCall(
            request=_serialize_anthropic_request(kwargs),
            response=resp.model_dump(),
            duration_ms=(_time.time() - start) * 1000,
        ))
        return resp
    return patched


def _make_anthropic_async_replayer(calls: List[RecordedCall], index: List[int]):
    from ._anthropic_interceptor import _deserialize_anthropic_response, _serialize_anthropic_request
    async def patched(**kwargs):
        if index[0] >= len(calls):
            raise RuntimeError(
                f"agentprobe: replay exhausted — fixture has {len(calls)} call(s) "
                f"but the agent made more. Re-record or update the fixture."
            )
        call = calls[index[0]]
        index[0] += 1
        call.request = _serialize_anthropic_request(kwargs)
        return _deserialize_anthropic_response(call.response)
    return patched


class AnthropicMultiSession:
    """Per-client record/replay for multi-agent Anthropic scenarios.

    Patches each Anthropic client instance independently so that two agents
    using different clients can be replayed simultaneously without interference.

    Usage::

        multi = AnthropicMultiSession()
        orchestrator = anthropic.Anthropic(api_key="...")
        subagent    = anthropic.Anthropic(api_key="...")

        with multi.replay(orchestrator, "fixtures/orch.jsonl") as probe_orch:
            with multi.replay(subagent, "fixtures/sub.jsonl") as probe_sub:
                run_pipeline(orchestrator, subagent)

        probe_orch.assert_tool_called("search")
        probe_sub.assert_max_iterations(3)
    """

    # ── Sync ─────────────────────────────────────────────────────────────

    @contextmanager
    def record(self, client, path: Union[str, Path]):
        """Intercept *client*'s Anthropic messages and save to *path* (JSONL)."""
        path = Path(path)
        calls: List[RecordedCall] = []
        original = client.messages.create
        with patch.object(client.messages, "create",
                          _make_anthropic_sync_recorder(calls, original)):
            yield AnthropicAssertionProxy(calls)
        _save_calls(calls, path)

    @contextmanager
    def replay(self, client, path: Union[str, Path]):
        """Replay *client*'s messages from a previously recorded fixture."""
        path = Path(path)
        _check_exists(path)
        calls = _load_calls(path)
        with patch.object(client.messages, "create",
                          _make_anthropic_sync_replayer(calls, [0])):
            yield AnthropicAssertionProxy(calls)

    @contextmanager
    def auto(self, client, path: Union[str, Path]):
        """Record if fixture missing or AGENTPROBE_UPDATE=1; replay otherwise."""
        path = Path(path)
        if path.exists() and not _should_update():
            with self.replay(client, path) as proxy:
                yield proxy
        else:
            with self.record(client, path) as proxy:
                yield proxy

    @contextmanager
    def replay_chain(self, *client_path_pairs):
        """Replay chained fixtures for multiple Anthropic clients simultaneously.

        Each argument is a ``(client, paths)`` pair where *paths* is a single
        path or list of paths concatenated in order for that client::

            with multi.replay_chain(
                (orchestrator, ["warmup.jsonl", "task.jsonl"]),
                (subagent, "sub.jsonl"),
            ) as probes:
                run_pipeline(orchestrator, subagent)
        """
        all_patches = []
        all_probes: Dict[Any, AnthropicAssertionProxy] = {}

        for client, paths in client_path_pairs:
            path_list = paths if isinstance(paths, list) else [paths]
            calls: List[RecordedCall] = []
            for p in path_list:
                _check_exists(Path(p))
                calls.extend(_load_calls(Path(p)))
            pat = patch.object(client.messages, "create",
                               _make_anthropic_sync_replayer(calls, [0]))
            pat.start()
            all_patches.append(pat)
            all_probes[client] = AnthropicAssertionProxy(calls)

        yield all_probes

        for pat in all_patches:
            pat.stop()

    # ── Async ─────────────────────────────────────────────────────────────

    @asynccontextmanager
    async def async_record(self, client, path: Union[str, Path]):
        """Async version of record() — for agents using ``AsyncAnthropic``."""
        path = Path(path)
        calls: List[RecordedCall] = []
        original = client.messages.create
        with patch.object(client.messages, "create",
                          _make_anthropic_async_recorder(calls, original)):
            yield AnthropicAssertionProxy(calls)
        _save_calls(calls, path)

    @asynccontextmanager
    async def async_replay(self, client, path: Union[str, Path]):
        """Async version of replay() — for agents using ``AsyncAnthropic``."""
        path = Path(path)
        _check_exists(path)
        calls = _load_calls(path)
        with patch.object(client.messages, "create",
                          _make_anthropic_async_replayer(calls, [0])):
            yield AnthropicAssertionProxy(calls)
