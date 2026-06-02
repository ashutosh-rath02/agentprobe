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


def _build_meta_line() -> str:
    """Return a JSON _meta header line with version + timestamp."""
    from agentprobe import __version__
    return json.dumps({
        "_meta": {
            "agentprobe_version": __version__,
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python_version": __import__("sys").version.split()[0],
        }
    })


def _save_calls(calls: List[RecordedCall], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [_build_meta_line()]
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
