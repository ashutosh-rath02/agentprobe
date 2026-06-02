"""Tests for v0.11.0: Anthropic streaming, show/diff Anthropic support,
assert_response_time_under, assert_tool_input_schema."""
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import anthropic
import anthropic.types as at
import openai
import pytest

from agentprobe import AnthropicSession, Session
from agentprobe._anthropic_interceptor import (
    MockAnthropicStream,
    _assemble_anthropic_from_events,
    _make_anthropic_stream_events,
)

ANTH_SIMPLE = "tests/fixtures/anthropic_simple.jsonl"
ANTH_TOOLS  = "tests/fixtures/anthropic_tools.jsonl"
OAI_FIXTURE = "tests/fixtures/bash_session.jsonl"


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "agentprobe._cli", *args],
        capture_output=True, text=True,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_text_events(text="Hello streaming", model="claude-sonnet-4-6",
                      input_tokens=20, output_tokens=8):
    """Build a list of real RawMessageStreamEvent objects for a text response."""
    return [
        at.RawMessageStartEvent.model_validate({
            "type": "message_start",
            "message": {
                "id": "msg_stream1", "type": "message", "role": "assistant",
                "content": [], "model": model, "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            },
        }),
        at.RawContentBlockStartEvent.model_validate({
            "type": "content_block_start", "index": 0,
            "content_block": {"type": "text", "text": ""},
        }),
        at.RawContentBlockDeltaEvent.model_validate({
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": text},
        }),
        at.RawContentBlockStopEvent.model_validate(
            {"type": "content_block_stop", "index": 0}
        ),
        at.RawMessageDeltaEvent.model_validate({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        }),
        at.RawMessageStopEvent.model_validate({"type": "message_stop"}),
    ]


def _make_tool_events(tool_name="search", tool_input=None,
                      model="claude-sonnet-4-6", input_tokens=30, output_tokens=15):
    inp = tool_input or {"q": "test"}
    return [
        at.RawMessageStartEvent.model_validate({
            "type": "message_start",
            "message": {
                "id": "msg_stream2", "type": "message", "role": "assistant",
                "content": [], "model": model, "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            },
        }),
        at.RawContentBlockStartEvent.model_validate({
            "type": "content_block_start", "index": 0,
            "content_block": {
                "type": "tool_use", "id": "toolu_stream1",
                "name": tool_name, "input": {},
            },
        }),
        at.RawContentBlockDeltaEvent.model_validate({
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": json.dumps(inp)},
        }),
        at.RawContentBlockStopEvent.model_validate(
            {"type": "content_block_stop", "index": 0}
        ),
        at.RawMessageDeltaEvent.model_validate({
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        }),
        at.RawMessageStopEvent.model_validate({"type": "message_stop"}),
    ]


# ── _assemble_anthropic_from_events ──────────────────────────────────────────

def test_assemble_text_response():
    events = _make_text_events("Hello world")
    serialized = [e.model_dump() for e in events]
    assembled = _assemble_anthropic_from_events(serialized)
    assert assembled["stop_reason"] == "end_turn"
    assert len(assembled["content"]) == 1
    assert assembled["content"][0]["type"] == "text"
    assert assembled["content"][0]["text"] == "Hello world"
    assert assembled["usage"]["input_tokens"] == 20
    assert assembled["usage"]["output_tokens"] == 8


def test_assemble_tool_use_response():
    events = _make_tool_events("search", {"q": "agentprobe"})
    serialized = [e.model_dump() for e in events]
    assembled = _assemble_anthropic_from_events(serialized)
    assert assembled["stop_reason"] == "tool_use"
    assert len(assembled["content"]) == 1
    assert assembled["content"][0]["type"] == "tool_use"
    assert assembled["content"][0]["name"] == "search"
    assert assembled["content"][0]["input"] == {"q": "agentprobe"}


# ── _make_anthropic_stream_events (synthesis) ─────────────────────────────────

def test_make_stream_events_from_text_response():
    resp = {
        "id": "msg_synth", "type": "message", "role": "assistant",
        "content": [{"type": "text", "text": "Hi there", "citations": None}],
        "model": "claude-sonnet-4-6", "stop_reason": "end_turn",
        "stop_sequence": None, "container": None, "stop_details": None,
        "usage": {"input_tokens": 10, "output_tokens": 5,
                  "cache_creation": None, "cache_creation_input_tokens": None,
                  "cache_read_input_tokens": None, "inference_geo": None,
                  "output_tokens_details": None, "server_tool_use": None,
                  "service_tier": None},
    }
    events = _make_anthropic_stream_events(resp)
    # Should have: start, block_start, block_delta, block_stop, msg_delta, msg_stop
    types = [e.type for e in events]
    assert "message_start" in types
    assert "content_block_delta" in types
    assert "message_delta" in types
    assert "message_stop" in types


# ── MockAnthropicStream ───────────────────────────────────────────────────────

def test_mock_stream_iteration():
    events = _make_text_events("Hello")
    assembled = _assemble_anthropic_from_events([e.model_dump() for e in events])
    stream = MockAnthropicStream(events, assembled)
    collected = list(stream)
    assert len(collected) == len(events)
    assert collected[0].type == "message_start"


def test_mock_stream_context_manager():
    events = _make_text_events("World")
    assembled = _assemble_anthropic_from_events([e.model_dump() for e in events])
    stream = MockAnthropicStream(events, assembled)
    with stream as s:
        texts = [e.delta.text for e in s if e.type == "content_block_delta"]
    assert texts == ["World"]


def test_mock_stream_get_final_message():
    events = _make_text_events("Final")
    assembled = _assemble_anthropic_from_events([e.model_dump() for e in events])
    stream = MockAnthropicStream(events, assembled)
    msg = stream.get_final_message()
    assert isinstance(msg, anthropic.types.Message)
    assert msg.content[0].text == "Final"


def test_mock_stream_get_final_text():
    events = _make_text_events("Stream text")
    assembled = _assemble_anthropic_from_events([e.model_dump() for e in events])
    stream = MockAnthropicStream(events, assembled)
    assert stream.get_final_text() == "Stream text"


# ── AnthropicSession replay with stream=True ─────────────────────────────────

def test_anthropic_stream_replay_from_non_streaming_fixture():
    """Replay with stream=True synthesizes events from a non-streaming fixture."""
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    with session.replay(ANTH_SIMPLE) as probe:
        stream = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=1024,
            messages=[{"role": "user", "content": "x"}],
            stream=True,
        )
        texts = [e.delta.text for e in stream if e.type == "content_block_delta"]
    assert "42" in "".join(texts)
    assert probe.iteration_count == 1


def test_anthropic_stream_replay_from_streaming_fixture(tmp_path):
    """Record a streaming call and replay it — events round-trip correctly."""
    fixture = tmp_path / "stream.jsonl"
    session = AnthropicSession()
    real_events = _make_text_events("Streaming reply")
    assembled = _assemble_anthropic_from_events([e.model_dump() for e in real_events])

    # Simulate recording by building a mock stream that create() returns
    mock_stream = MockAnthropicStream(real_events, assembled)

    client = anthropic.Anthropic(api_key="dummy")
    with patch.object(anthropic.resources.messages.Messages, "create", return_value=mock_stream):
        with session.record(str(fixture)) as probe:
            stream = client.messages.create(
                model="claude-sonnet-4-6", max_tokens=1024,
                messages=[{"role": "user", "content": "x"}],
                stream=True,
            )
            collected_texts = [e.delta.text for e in stream if e.type == "content_block_delta"]

    assert "".join(collected_texts) == "Streaming reply"
    assert fixture.exists()

    # Now replay
    client2 = anthropic.Anthropic(api_key="dummy")
    with session.replay(str(fixture)) as probe2:
        stream2 = client2.messages.create(
            model="claude-sonnet-4-6", max_tokens=1024,
            messages=[{"role": "user", "content": "x"}],
            stream=True,
        )
        replay_texts = [e.delta.text for e in stream2 if e.type == "content_block_delta"]

    assert "".join(replay_texts) == "Streaming reply"
    assert stream2.get_final_text() == "Streaming reply"


def test_anthropic_stream_inject():
    """inject() with stream=True synthesizes events from the injected message."""
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    msg = anthropic.types.Message.model_validate({
        "id": "msg_inj", "type": "message", "role": "assistant",
        "content": [{"type": "text", "text": "Injected stream"}],
        "model": "claude-sonnet-4-6", "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    })
    with session.inject(msg) as probe:
        stream = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=1024,
            messages=[{"role": "user", "content": "x"}],
            stream=True,
        )
        texts = [e.delta.text for e in stream if e.type == "content_block_delta"]
    assert "Injected stream" in "".join(texts)


# ── assert_response_time_under ────────────────────────────────────────────────

def test_assert_response_time_under_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(OAI_FIXTURE) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    probe.assert_response_time_under(1_000_000)  # fixture has recorded durations ~200ms


def test_assert_response_time_under_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(OAI_FIXTURE) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    with pytest.raises(AssertionError, match="ms"):
        probe.assert_response_time_under(0.001)  # impossible threshold


def test_anthropic_assert_response_time_under_passes():
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    with session.replay(ANTH_SIMPLE) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
    probe.assert_response_time_under(1_000_000)


# ── assert_tool_input_schema ──────────────────────────────────────────────────

def test_assert_tool_input_schema_passes():
    pytest.importorskip("jsonschema")
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    resp = {
        "id": "chatcmpl-s", "object": "chat.completion", "created": 1748700000,
        "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": None,
                     "tool_calls": [{"id": "c1", "type": "function",
                     "function": {"name": "search", "arguments": '{"query": "hello"}'}}]},
                     "finish_reason": "tool_calls", "logprobs": None}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "system_fingerprint": None,
    }
    with session.inject(resp) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    probe.assert_tool_input_schema("search", {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    })


def test_assert_tool_input_schema_fails():
    pytest.importorskip("jsonschema")
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    resp = {
        "id": "chatcmpl-s2", "object": "chat.completion", "created": 1748700000,
        "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": None,
                     "tool_calls": [{"id": "c2", "type": "function",
                     "function": {"name": "search", "arguments": '{"wrong_key": 123}'}}]},
                     "finish_reason": "tool_calls", "logprobs": None}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "system_fingerprint": None,
    }
    with session.inject(resp) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError, match="schema validation"):
        probe.assert_tool_input_schema("search", {
            "type": "object",
            "required": ["query"],
            "properties": {"query": {"type": "string"}},
            "additionalProperties": False,
        })


def test_anthropic_assert_tool_input_schema_passes():
    pytest.importorskip("jsonschema")
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    msg = anthropic.types.Message.model_validate({
        "id": "msg_s", "type": "message", "role": "assistant",
        "content": [{"type": "tool_use", "id": "t1", "name": "weather",
                     "input": {"city": "NYC", "units": "celsius"}}],
        "model": "claude-sonnet-4-6", "stop_reason": "tool_use",
        "stop_sequence": None, "usage": {"input_tokens": 20, "output_tokens": 10},
    })
    with session.inject(msg) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
    probe.assert_tool_input_schema("weather", {
        "type": "object",
        "properties": {"city": {"type": "string"}, "units": {"type": "string"}},
        "required": ["city"],
    })


# ── CLI show Anthropic support ────────────────────────────────────────────────

def test_cli_show_anthropic_fixture():
    r = run_cli("show", ANTH_SIMPLE)
    assert r.returncode == 0, r.stderr
    assert "claude-sonnet-4-6" in r.stdout
    assert "end_turn" in r.stdout


def test_cli_show_anthropic_tools_fixture():
    r = run_cli("show", ANTH_TOOLS)
    assert r.returncode == 0, r.stderr
    assert "tool_use" in r.stdout or "search" in r.stdout


def test_cli_show_json_anthropic_fixture():
    r = run_cli("show", "--json", ANTH_SIMPLE)
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    assert data["calls"][0]["provider"] == "anthropic"
    assert data["calls"][0]["stop_reason"] == "end_turn"


# ── CLI diff Anthropic support ────────────────────────────────────────────────

def test_cli_diff_anthropic_identical():
    r = run_cli("diff", ANTH_SIMPLE, ANTH_SIMPLE)
    assert r.returncode == 0, r.stderr
    assert "no differences" in r.stdout


def test_cli_diff_anthropic_different():
    r = run_cli("diff", ANTH_SIMPLE, ANTH_TOOLS)
    # Different call counts (1 vs 2) and content
    assert r.returncode == 1


def test_cli_diff_json_anthropic():
    r = run_cli("diff", "--json", ANTH_SIMPLE, ANTH_SIMPLE)
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    assert data["differences"] == []
