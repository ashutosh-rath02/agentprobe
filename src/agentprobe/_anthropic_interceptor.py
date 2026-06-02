import time
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Dict, List
from unittest.mock import patch

import anthropic.resources.messages

from ._models import RecordedCall
from ._serializer import _deep_serialize


def _serialize_anthropic_request(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _deep_serialize(v) for k, v in kwargs.items()}


def _deserialize_anthropic_response(data: Dict[str, Any]) -> anthropic.types.Message:
    return anthropic.types.Message.model_validate(data)


# ── Sync ──────────────────────────────────────────────────────────────────────

@contextmanager
def anthropic_recording_context(calls: List[RecordedCall]):
    original = anthropic.resources.messages.Messages.create

    def patched(self, **kwargs):
        start = time.time()
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
        return _deserialize_anthropic_response(call.response)

    with patch.object(anthropic.resources.messages.Messages, "create", patched):
        yield


# ── Async ─────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def async_anthropic_recording_context(calls: List[RecordedCall]):
    original = anthropic.resources.messages.AsyncMessages.create

    async def patched(self, **kwargs):
        start = time.time()
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
        return _deserialize_anthropic_response(call.response)

    with patch.object(anthropic.resources.messages.AsyncMessages, "create", patched):
        yield
