import json
import os
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ._interceptor import (
    async_recording_context,
    async_replaying_context,
    recording_context,
    replaying_context,
)
from ._models import RecordedCall
from ._pricing import estimate_cost


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
            f"agentprobe: expected cost ≤ ${usd:.4f}, got ${actual:.4f}"
        )
        return self

    # ── Introspection properties ──────────────────────────────────────────

    @property
    def iteration_count(self) -> int:
        return len(self._calls)

    @property
    def tools_called(self) -> List[str]:
        return sorted(self._all_tool_names())

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
    with open(path) as f:
        return [RecordedCall(**json.loads(line)) for line in f if line.strip()]


def _save_calls(calls: List[RecordedCall], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for call in calls:
            data: dict = {
                "request": call.request,
                "response": call.response,
                "timestamp": call.timestamp,
                "duration_ms": call.duration_ms,
            }
            if call.chunks is not None:
                data["chunks"] = call.chunks
            f.write(json.dumps(data) + "\n")


def _check_exists(path: Path) -> None:
    if not path.exists():
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
    def replay(self, path: Union[str, Path]):
        """Replay calls from a previously recorded fixture at *path* (JSONL)."""
        path = Path(path)
        _check_exists(path)
        calls = _load_calls(path)
        with replaying_context(calls):
            yield AssertionProxy(calls)

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
