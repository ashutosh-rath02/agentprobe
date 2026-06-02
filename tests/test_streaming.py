"""Tests for stream=True record/replay (sync and async)."""
import json
import pytest
import openai

from agentprobe import Session

FIXTURE = "tests/fixtures/streaming_session.jsonl"
NON_STREAMING_FIXTURE = "tests/fixtures/bash_session.jsonl"


# ── Stream consumer helpers ───────────────────────────────────────────────────

def _consume_stream(stream):
    """Drain a sync stream; return (content, tool_calls_list, finish_reason)."""
    content_parts = []
    tool_call_buffers = {}
    finish_reason = None
    for chunk in stream:
        choice = chunk.choices[0]
        delta = choice.delta
        if delta.content:
            content_parts.append(delta.content)
        for tc in (delta.tool_calls or []):
            idx = tc.index
            if idx not in tool_call_buffers:
                tool_call_buffers[idx] = {"id": "", "name": "", "arguments": ""}
            if tc.id:
                tool_call_buffers[idx]["id"] = tc.id
            if tc.function:
                if tc.function.name:
                    tool_call_buffers[idx]["name"] += tc.function.name
                if tc.function.arguments:
                    tool_call_buffers[idx]["arguments"] += tc.function.arguments
        if choice.finish_reason:
            finish_reason = choice.finish_reason
    tool_calls = [tool_call_buffers[i] for i in sorted(tool_call_buffers)]
    return "".join(content_parts), tool_calls, finish_reason


async def _consume_async_stream(stream):
    """Drain an async stream; return (content, tool_calls_list, finish_reason)."""
    content_parts = []
    tool_call_buffers = {}
    finish_reason = None
    async for chunk in stream:
        choice = chunk.choices[0]
        delta = choice.delta
        if delta.content:
            content_parts.append(delta.content)
        for tc in (delta.tool_calls or []):
            idx = tc.index
            if idx not in tool_call_buffers:
                tool_call_buffers[idx] = {"id": "", "name": "", "arguments": ""}
            if tc.id:
                tool_call_buffers[idx]["id"] = tc.id
            if tc.function:
                if tc.function.name:
                    tool_call_buffers[idx]["name"] += tc.function.name
                if tc.function.arguments:
                    tool_call_buffers[idx]["arguments"] += tc.function.arguments
        if choice.finish_reason:
            finish_reason = choice.finish_reason
    tool_calls = [tool_call_buffers[i] for i in sorted(tool_call_buffers)]
    return "".join(content_parts), tool_calls, finish_reason


def _fake_streaming_agent(client: openai.OpenAI) -> str:
    tools = [{"type": "function", "function": {"name": "bash", "description": "Run bash", "parameters": {
        "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]
    }}}]
    messages = [{"role": "user", "content": "list files in /tmp"}]

    while True:
        stream = client.chat.completions.create(
            model="gpt-4o", messages=messages, tools=tools, stream=True
        )
        content, tool_calls, finish_reason = _consume_stream(stream)

        if not tool_calls:
            return content

        messages.append({
            "role": "assistant",
            "content": content or None,
            "tool_calls": [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                for tc in tool_calls
            ],
        })
        for tc in tool_calls:
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": "file1.txt\nfile2.txt\ntemp.log"})


async def _fake_async_streaming_agent(client: openai.AsyncOpenAI) -> str:
    tools = [{"type": "function", "function": {"name": "bash", "description": "Run bash", "parameters": {
        "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]
    }}}]
    messages = [{"role": "user", "content": "list files in /tmp"}]

    while True:
        stream = await client.chat.completions.create(
            model="gpt-4o", messages=messages, tools=tools, stream=True
        )
        content, tool_calls, finish_reason = await _consume_async_stream(stream)

        if not tool_calls:
            return content

        messages.append({
            "role": "assistant",
            "content": content or None,
            "tool_calls": [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                for tc in tool_calls
            ],
        })
        for tc in tool_calls:
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": "file1.txt\nfile2.txt\ntemp.log"})


# ── Sync replay tests ─────────────────────────────────────────────────────────

def test_streaming_replay_runs_without_real_api():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(FIXTURE) as probe:
        result = _fake_streaming_agent(client)
    assert "file1.txt" in result


def test_streaming_replay_yields_chunks_in_order():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    collected = []
    with session.replay(FIXTURE):
        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "list files in /tmp"}],
            stream=True,
        )
        for chunk in stream:
            collected.append(chunk)
    assert len(collected) == 4  # fixture call 1 has 4 chunks
    assert collected[0].choices[0].delta.role == "assistant"
    assert collected[-1].choices[0].finish_reason == "tool_calls"


def test_streaming_replay_iteration_count():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(FIXTURE) as probe:
        _fake_streaming_agent(client)
    assert probe.iteration_count == 2


def test_streaming_replay_tool_called():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(FIXTURE) as probe:
        _fake_streaming_agent(client)
    probe.assert_tool_called("bash")
    probe.assert_not_tool_called("web_search")


def test_streaming_replay_tool_called_with():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(FIXTURE) as probe:
        _fake_streaming_agent(client)
    probe.assert_tool_called_with("bash", command="ls /tmp")


def test_streaming_replay_output_contains():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(FIXTURE) as probe:
        _fake_streaming_agent(client)
    probe.assert_output_contains("file1.txt")


def test_streaming_replay_stop_reason():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(FIXTURE) as probe:
        _fake_streaming_agent(client)
    probe.assert_stop_reason("stop")


def test_streaming_mismatch_raises():
    """Replaying a non-streaming fixture with stream=True should raise."""
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with pytest.raises(RuntimeError, match="stream=True"):
        with session.replay(NON_STREAMING_FIXTURE):
            client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            )


def test_streaming_record_round_trip(tmp_path):
    fixture = tmp_path / "stream_rt.jsonl"
    session = Session()
    client = openai.OpenAI(api_key="dummy")

    with session.replay(FIXTURE):
        with session.record(fixture):
            _fake_streaming_agent(client)

    assert fixture.exists()
    from conftest import fixture_lines
    lines = fixture_lines(fixture)
    assert len(lines) == 2
    data = json.loads(lines[0])
    assert "chunks" in data
    assert len(data["chunks"]) == 4  # tool-use call has 4 chunks


# ── Async replay tests ────────────────────────────────────────────────────────

async def test_async_streaming_replay_runs_without_real_api():
    session = Session()
    client = openai.AsyncOpenAI(api_key="dummy")
    async with session.async_replay(FIXTURE) as probe:
        result = await _fake_async_streaming_agent(client)
    assert "file1.txt" in result


async def test_async_streaming_replay_iteration_count():
    session = Session()
    client = openai.AsyncOpenAI(api_key="dummy")
    async with session.async_replay(FIXTURE) as probe:
        await _fake_async_streaming_agent(client)
    assert probe.iteration_count == 2


async def test_async_streaming_replay_tool_called():
    session = Session()
    client = openai.AsyncOpenAI(api_key="dummy")
    async with session.async_replay(FIXTURE) as probe:
        await _fake_async_streaming_agent(client)
    probe.assert_tool_called("bash")


async def test_async_streaming_replay_output_contains():
    session = Session()
    client = openai.AsyncOpenAI(api_key="dummy")
    async with session.async_replay(FIXTURE) as probe:
        await _fake_async_streaming_agent(client)
    probe.assert_output_contains("file1.txt")


async def test_async_streaming_record_round_trip(tmp_path):
    fixture = tmp_path / "async_stream_rt.jsonl"
    session = Session()
    client = openai.AsyncOpenAI(api_key="dummy")

    async with session.async_replay(FIXTURE):
        async with session.async_record(fixture):
            await _fake_async_streaming_agent(client)

    assert fixture.exists()
    from conftest import fixture_lines
    lines = fixture_lines(fixture)
    assert len(lines) == 2
    data = json.loads(lines[0])
    assert "chunks" in data
