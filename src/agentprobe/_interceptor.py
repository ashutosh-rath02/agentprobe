import time
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Dict, List
from unittest.mock import patch

import openai.resources.chat.completions

from ._models import RecordedCall
from ._serializer import deserialize_chunk, deserialize_response, serialize_request


# ── Stream mocks ──────────────────────────────────────────────────────────────

class MockStream:
    """Iterable context manager that replays recorded stream chunks synchronously."""

    def __init__(self, chunks: list):
        self._chunks = chunks

    def __iter__(self):
        return iter(self._chunks)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class MockAsyncStream:
    """Async iterable context manager that replays recorded stream chunks."""

    def __init__(self, chunks: list):
        self._chunks = chunks

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for chunk in self._chunks:
            yield chunk

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


# ── Chunk assembly ────────────────────────────────────────────────────────────

def _assemble_from_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Reassemble serialized ChatCompletionChunk dicts into a ChatCompletion-like dict."""
    if not chunks:
        return {}

    first = chunks[0]
    content_parts: List[str] = []
    tool_calls: Dict[int, Dict] = {}
    finish_reason = None
    usage = None

    for chunk in chunks:
        for choice in chunk.get("choices", []):
            delta = choice.get("delta") or {}

            if delta.get("content"):
                content_parts.append(delta["content"])

            for tc in (delta.get("tool_calls") or []):
                idx = tc.get("index", 0)
                if idx not in tool_calls:
                    tool_calls[idx] = {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                if tc.get("id"):
                    tool_calls[idx]["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    tool_calls[idx]["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    tool_calls[idx]["function"]["arguments"] += fn["arguments"]

            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

        if chunk.get("usage"):
            usage = chunk["usage"]

    content = "".join(content_parts) or None
    tool_calls_list = [tool_calls[i] for i in sorted(tool_calls)] or None

    return {
        "id": first.get("id", ""),
        "object": "chat.completion",
        "created": first.get("created", 0),
        "model": first.get("model", ""),
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls_list,
            },
            "finish_reason": finish_reason,
            "logprobs": None,
        }],
        "usage": usage,
        "system_fingerprint": first.get("system_fingerprint"),
    }


# ── Sync ──────────────────────────────────────────────────────────────────────

@contextmanager
def recording_context(calls: List[RecordedCall]):
    original = openai.resources.chat.completions.Completions.create

    def patched(self, **kwargs):
        start = time.time()
        if kwargs.get("stream"):
            real_stream = original(self, **kwargs)
            raw_chunks = list(real_stream)
            serialized = [c.model_dump() for c in raw_chunks]
            calls.append(RecordedCall(
                request=serialize_request(kwargs),
                response=_assemble_from_chunks(serialized),
                chunks=serialized,
                duration_ms=(time.time() - start) * 1000,
            ))
            return MockStream(raw_chunks)
        response = original(self, **kwargs)
        calls.append(RecordedCall(
            request=serialize_request(kwargs),
            response=response.model_dump(),
            duration_ms=(time.time() - start) * 1000,
        ))
        return response

    with patch.object(openai.resources.chat.completions.Completions, "create", patched):
        yield


@contextmanager
def replaying_context(calls: List[RecordedCall]):
    index = [0]

    def patched(self, **kwargs):
        if index[0] >= len(calls):
            raise RuntimeError(
                f"agentprobe: replay exhausted — fixture has {len(calls)} call(s) "
                f"but the agent made more. Re-record or update the fixture."
            )
        call = calls[index[0]]
        index[0] += 1
        if kwargs.get("stream"):
            if call.chunks is None:
                raise RuntimeError(
                    "agentprobe: agent used stream=True but this fixture was recorded "
                    "without streaming. Re-record with stream=True."
                )
            return MockStream([deserialize_chunk(c) for c in call.chunks])
        return deserialize_response(call.response)

    with patch.object(openai.resources.chat.completions.Completions, "create", patched):
        yield


# ── Async ─────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def async_recording_context(calls: List[RecordedCall]):
    original = openai.resources.chat.completions.AsyncCompletions.create

    async def patched(self, **kwargs):
        start = time.time()
        if kwargs.get("stream"):
            real_stream = await original(self, **kwargs)
            raw_chunks = [c async for c in real_stream]
            serialized = [c.model_dump() for c in raw_chunks]
            calls.append(RecordedCall(
                request=serialize_request(kwargs),
                response=_assemble_from_chunks(serialized),
                chunks=serialized,
                duration_ms=(time.time() - start) * 1000,
            ))
            return MockAsyncStream(raw_chunks)
        response = await original(self, **kwargs)
        calls.append(RecordedCall(
            request=serialize_request(kwargs),
            response=response.model_dump(),
            duration_ms=(time.time() - start) * 1000,
        ))
        return response

    with patch.object(openai.resources.chat.completions.AsyncCompletions, "create", patched):
        yield


@asynccontextmanager
async def async_replaying_context(calls: List[RecordedCall]):
    index = [0]

    async def patched(self, **kwargs):
        if index[0] >= len(calls):
            raise RuntimeError(
                f"agentprobe: replay exhausted — fixture has {len(calls)} call(s) "
                f"but the agent made more. Re-record or update the fixture."
            )
        call = calls[index[0]]
        index[0] += 1
        if kwargs.get("stream"):
            if call.chunks is None:
                raise RuntimeError(
                    "agentprobe: agent used stream=True but this fixture was recorded "
                    "without streaming. Re-record with stream=True."
                )
            return MockAsyncStream([deserialize_chunk(c) for c in call.chunks])
        return deserialize_response(call.response)

    with patch.object(openai.resources.chat.completions.AsyncCompletions, "create", patched):
        yield
