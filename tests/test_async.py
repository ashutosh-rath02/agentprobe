"""Tests for async session — AsyncOpenAI record/replay."""
import pytest
import openai

from agentprobe import Session

FIXTURE = "tests/fixtures/bash_session.jsonl"


async def _fake_async_agent(client: openai.AsyncOpenAI) -> str:
    tools = [{"type": "function", "function": {"name": "bash", "description": "Run bash", "parameters": {
        "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]
    }}}]
    messages = [{"role": "user", "content": "list files in /tmp"}]

    while True:
        resp = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
        )
        choice = resp.choices[0]
        tool_calls = choice.message.tool_calls or []
        if not tool_calls:
            return choice.message.content

        messages.append({
            "role": "assistant",
            "content": choice.message.content,
            "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in tool_calls
            ]
        })
        for tc in tool_calls:
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": "file1.txt\nfile2.txt\ntemp.log"})


async def test_async_replay_runs_without_real_api():
    session = Session()
    client = openai.AsyncOpenAI(api_key="dummy-key-async-test")
    async with session.async_replay(FIXTURE) as probe:
        result = await _fake_async_agent(client)
    assert "file1.txt" in result


async def test_async_replay_iteration_count():
    session = Session()
    client = openai.AsyncOpenAI(api_key="dummy-key-async-test")
    async with session.async_replay(FIXTURE) as probe:
        await _fake_async_agent(client)
    assert probe.iteration_count == 2


async def test_async_assert_tool_called():
    session = Session()
    client = openai.AsyncOpenAI(api_key="dummy-key-async-test")
    async with session.async_replay(FIXTURE) as probe:
        await _fake_async_agent(client)
        probe.assert_tool_called("bash")
        probe.assert_not_tool_called("web_search")


async def test_async_assert_output_contains():
    session = Session()
    client = openai.AsyncOpenAI(api_key="dummy-key-async-test")
    async with session.async_replay(FIXTURE) as probe:
        await _fake_async_agent(client)
        probe.assert_output_contains("file1.txt")


async def test_async_auto_mode_replays_when_exists():
    session = Session()
    client = openai.AsyncOpenAI(api_key="dummy-key-async-test")
    async with session.async_auto(FIXTURE) as probe:
        result = await _fake_async_agent(client)
    assert "file1.txt" in result


async def test_async_record_round_trip(tmp_path):
    fixture = tmp_path / "async_session.jsonl"
    session = Session()
    client = openai.AsyncOpenAI(api_key="dummy-key-async-test")

    async with session.async_replay(FIXTURE):
        async with session.async_record(fixture):
            await _fake_async_agent(client)

    assert fixture.exists()
    from conftest import fixture_lines
    lines = fixture_lines(fixture)
    assert len(lines) == 2


async def test_async_missing_fixture_raises():
    session = Session()
    with pytest.raises(FileNotFoundError, match="agentprobe"):
        async with session.async_replay("tests/fixtures/nonexistent.jsonl"):
            pass
