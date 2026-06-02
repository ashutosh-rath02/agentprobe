import json
import time
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Dict, List
from unittest.mock import patch

import anthropic.resources.messages

from ._models import RecordedCall
from ._serializer import _deep_serialize


def _serialize_anthropic_request(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _deep_serialize(v) for k, v in kwargs.items()}


def _deserialize_anthropic_response(data: Dict[str, Any]) -> "anthropic.types.Message":
    return anthropic.types.Message.model_validate(data)


def _deserialize_anthropic_event(data: Dict[str, Any]) -> Any:
    import anthropic.types as t
    mapping = {
        "message_start":       t.RawMessageStartEvent,
        "content_block_start": t.RawContentBlockStartEvent,
        "content_block_delta": t.RawContentBlockDeltaEvent,
        "content_block_stop":  t.RawContentBlockStopEvent,
        "message_delta":       t.RawMessageDeltaEvent,
        "message_stop":        t.RawMessageStopEvent,
    }
    cls = mapping.get(data.get("type", ""))
    return cls.model_validate(data) if cls else data


def _assemble_anthropic_from_events(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Reassemble a Message dict from serialized RawMessageStreamEvent dicts."""
    message_data: Dict[str, Any] = {}
    content_blocks: Dict[int, Dict[str, Any]] = {}
    stop_reason = None
    output_tokens = 0

    for ev in events:
        etype = ev.get("type")
        if etype == "message_start":
            message_data = ev.get("message", {})
        elif etype == "content_block_start":
            idx = ev.get("index", 0)
            block = ev.get("content_block", {})
            content_blocks[idx] = {
                "type": block.get("type", "text"),
                "text": block.get("text", ""),
                "id": block.get("id"),
                "name": block.get("name"),
                "input": "",
                "caller": None,
            }
        elif etype == "content_block_delta":
            idx = ev.get("index", 0)
            delta = ev.get("delta", {})
            dtype = delta.get("type")
            if dtype == "text_delta":
                content_blocks.setdefault(idx, {"type": "text", "text": "", "id": None, "name": None, "input": "", "caller": None})
                content_blocks[idx]["text"] = content_blocks[idx].get("text", "") + delta.get("text", "")
            elif dtype == "input_json_delta":
                content_blocks.setdefault(idx, {"type": "tool_use", "text": "", "id": None, "name": None, "input": "", "caller": None})
                content_blocks[idx]["input"] = content_blocks[idx].get("input", "") + delta.get("partial_json", "")
        elif etype == "message_delta":
            delta = ev.get("delta", {})
            stop_reason = delta.get("stop_reason")
            usage = ev.get("usage") or {}
            output_tokens = usage.get("output_tokens", 0) or 0

    content = []
    for i in sorted(content_blocks.keys()):
        block = content_blocks[i]
        btype = block.get("type", "text")
        if btype == "text":
            content.append({"type": "text", "text": block.get("text", ""), "citations": None})
        elif btype == "tool_use":
            inp_raw = block.get("input", "")
            if isinstance(inp_raw, str):
                try:
                    inp = json.loads(inp_raw) if inp_raw else {}
                except (json.JSONDecodeError, ValueError):
                    inp = {}
            else:
                inp = inp_raw
            content.append({
                "type": "tool_use",
                "id": block.get("id"),
                "name": block.get("name"),
                "input": inp,
                "caller": None,
            })

    input_tokens = (message_data.get("usage") or {}).get("input_tokens", 0) or 0
    return {
        "id": message_data.get("id", ""),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": message_data.get("model", ""),
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "container": None,
        "stop_details": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation": None,
            "cache_creation_input_tokens": None,
            "cache_read_input_tokens": None,
            "inference_geo": None,
            "output_tokens_details": None,
            "server_tool_use": None,
            "service_tier": None,
        },
    }


def _make_anthropic_stream_events(response: Dict[str, Any]) -> List[Any]:
    """Synthesize RawMessageStreamEvent objects from an assembled Message dict.

    Used during replay to reconstruct a plausible event stream from a
    non-streaming fixture or from a fixture recorded without ``stream=True``.
    """
    import anthropic.types as t
    events = []
    usage = response.get("usage") or {}

    events.append(t.RawMessageStartEvent.model_validate({
        "type": "message_start",
        "message": {
            "id": response.get("id", "msg_mock"),
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": response.get("model", ""),
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": usage.get("input_tokens", 0) or 0, "output_tokens": 0},
        },
    }))

    for idx, block in enumerate(response.get("content") or []):
        btype = block.get("type")
        if btype == "text":
            events.append(t.RawContentBlockStartEvent.model_validate({
                "type": "content_block_start", "index": idx,
                "content_block": {"type": "text", "text": ""},
            }))
            text = block.get("text", "")
            if text:
                events.append(t.RawContentBlockDeltaEvent.model_validate({
                    "type": "content_block_delta", "index": idx,
                    "delta": {"type": "text_delta", "text": text},
                }))
            events.append(t.RawContentBlockStopEvent.model_validate(
                {"type": "content_block_stop", "index": idx}
            ))
        elif btype == "tool_use":
            events.append(t.RawContentBlockStartEvent.model_validate({
                "type": "content_block_start", "index": idx,
                "content_block": {
                    "type": "tool_use",
                    "id": block.get("id", "toolu_mock"),
                    "name": block.get("name", ""),
                    "input": {},
                },
            }))
            inp = block.get("input", {})
            inp_str = json.dumps(inp) if isinstance(inp, dict) else str(inp)
            events.append(t.RawContentBlockDeltaEvent.model_validate({
                "type": "content_block_delta", "index": idx,
                "delta": {"type": "input_json_delta", "partial_json": inp_str},
            }))
            events.append(t.RawContentBlockStopEvent.model_validate(
                {"type": "content_block_stop", "index": idx}
            ))

    events.append(t.RawMessageDeltaEvent.model_validate({
        "type": "message_delta",
        "delta": {"stop_reason": response.get("stop_reason", "end_turn"), "stop_sequence": None},
        "usage": {"output_tokens": usage.get("output_tokens", 0) or 0},
    }))
    events.append(t.RawMessageStopEvent.model_validate({"type": "message_stop"}))
    return events


# ── Stream mocks ──────────────────────────────────────────────────────────────

class MockAnthropicStream:
    """Sync mock for ``messages.create(stream=True)`` — replays recorded events."""

    def __init__(self, events: list, assembled_response: Dict[str, Any]):
        self._events = events
        self._assembled = assembled_response

    def __iter__(self):
        return iter(self._events)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def close(self):
        pass

    def get_final_message(self) -> "anthropic.types.Message":
        return _deserialize_anthropic_response(self._assembled)

    def get_final_text(self) -> str:
        content = self._assembled.get("content") or []
        return "".join(b.get("text", "") for b in content if b.get("type") == "text")


class MockAnthropicAsyncStream:
    """Async mock for ``messages.create(stream=True)`` with ``AsyncAnthropic``."""

    def __init__(self, events: list, assembled_response: Dict[str, Any]):
        self._events = events
        self._assembled = assembled_response

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for event in self._events:
            yield event

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def close(self):
        pass

    async def get_final_message(self) -> "anthropic.types.Message":
        return _deserialize_anthropic_response(self._assembled)


# ── Sync ──────────────────────────────────────────────────────────────────────

@contextmanager
def anthropic_recording_context(calls: List[RecordedCall]):
    original = anthropic.resources.messages.Messages.create

    def patched(self, **kwargs):
        start = time.time()
        if kwargs.get("stream"):
            real_stream = original(self, **kwargs)
            raw_events = list(real_stream)
            serialized = [e.model_dump() for e in raw_events]
            assembled = _assemble_anthropic_from_events(serialized)
            calls.append(RecordedCall(
                request=_serialize_anthropic_request(kwargs),
                response=assembled,
                chunks=serialized,
                duration_ms=(time.time() - start) * 1000,
            ))
            return MockAnthropicStream(raw_events, assembled)
        response = original(self, **kwargs)
        calls.append(RecordedCall(
            request=_serialize_anthropic_request(kwargs),
            response=response.model_dump(),
            duration_ms=(time.time() - start) * 1000,
        ))
        return response

    with patch.object(anthropic.resources.messages.Messages, "create", patched):
        yield


@contextmanager
def anthropic_replaying_context(calls: List[RecordedCall]):
    index = [0]

    def patched(self, **kwargs):
        if index[0] >= len(calls):
            raise RuntimeError(
                f"agentprobe: replay exhausted — fixture has {len(calls)} call(s) "
                f"but the agent made more. Re-record or update the fixture."
            )
        call = calls[index[0]]
        index[0] += 1
        call.request = _serialize_anthropic_request(kwargs)
        if kwargs.get("stream"):
            events = (
                [_deserialize_anthropic_event(e) for e in call.chunks]
                if call.chunks is not None
                else _make_anthropic_stream_events(call.response)
            )
            return MockAnthropicStream(events, call.response)
        return _deserialize_anthropic_response(call.response)

    with patch.object(anthropic.resources.messages.Messages, "create", patched):
        yield


@contextmanager
def _anthropic_strict_replaying_context(calls: List[RecordedCall], index: List[int]):
    def patched(self, **kwargs):
        if index[0] >= len(calls):
            raise RuntimeError(
                f"agentprobe: replay exhausted — fixture has {len(calls)} call(s) "
                f"but the agent made more. Re-record or update the fixture."
            )
        call = calls[index[0]]
        index[0] += 1
        call.request = _serialize_anthropic_request(kwargs)
        if kwargs.get("stream"):
            events = (
                [_deserialize_anthropic_event(e) for e in call.chunks]
                if call.chunks is not None
                else _make_anthropic_stream_events(call.response)
            )
            return MockAnthropicStream(events, call.response)
        return _deserialize_anthropic_response(call.response)

    with patch.object(anthropic.resources.messages.Messages, "create", patched):
        yield


# ── Async ─────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def async_anthropic_recording_context(calls: List[RecordedCall]):
    original = anthropic.resources.messages.AsyncMessages.create

    async def patched(self, **kwargs):
        start = time.time()
        if kwargs.get("stream"):
            real_stream = await original(self, **kwargs)
            raw_events = [e async for e in real_stream]
            serialized = [e.model_dump() for e in raw_events]
            assembled = _assemble_anthropic_from_events(serialized)
            calls.append(RecordedCall(
                request=_serialize_anthropic_request(kwargs),
                response=assembled,
                chunks=serialized,
                duration_ms=(time.time() - start) * 1000,
            ))
            return MockAnthropicAsyncStream(raw_events, assembled)
        response = await original(self, **kwargs)
        calls.append(RecordedCall(
            request=_serialize_anthropic_request(kwargs),
            response=response.model_dump(),
            duration_ms=(time.time() - start) * 1000,
        ))
        return response

    with patch.object(anthropic.resources.messages.AsyncMessages, "create", patched):
        yield


@asynccontextmanager
async def async_anthropic_replaying_context(calls: List[RecordedCall]):
    index = [0]

    async def patched(self, **kwargs):
        if index[0] >= len(calls):
            raise RuntimeError(
                f"agentprobe: replay exhausted — fixture has {len(calls)} call(s) "
                f"but the agent made more. Re-record or update the fixture."
            )
        call = calls[index[0]]
        index[0] += 1
        call.request = _serialize_anthropic_request(kwargs)
        if kwargs.get("stream"):
            events = (
                [_deserialize_anthropic_event(e) for e in call.chunks]
                if call.chunks is not None
                else _make_anthropic_stream_events(call.response)
            )
            return MockAnthropicAsyncStream(events, call.response)
        return _deserialize_anthropic_response(call.response)

    with patch.object(anthropic.resources.messages.AsyncMessages, "create", patched):
        yield
