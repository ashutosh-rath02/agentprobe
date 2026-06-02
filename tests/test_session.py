"""Tests for agentprobe record/replay and assertion machinery."""
import json
import pytest
import openai

from agentprobe import Session


FIXTURE = "tests/fixtures/bash_session.jsonl"


def _fake_agent(client: openai.OpenAI) -> str:
    """Minimal two-turn tool-use agent used across tests."""
    tools = [{"type": "function", "function": {"name": "bash", "description": "Run bash", "parameters": {
        "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]
    }}}]
    messages = [{"role": "user", "content": "list files in /tmp"}]

    while True:
        resp = client.chat.completions.create(
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


# ── Replay tests ──────────────────────────────────────────────────────────────

def test_replay_runs_without_real_api():
    session = Session()
    client = openai.OpenAI(api_key="dummy-key-replay-test")
    with session.replay(FIXTURE) as probe:
        result = _fake_agent(client)
    assert "file1.txt" in result


def test_replay_iteration_count():
    session = Session()
    client = openai.OpenAI(api_key="dummy-key-replay-test")
    with session.replay(FIXTURE) as probe:
        _fake_agent(client)
    assert probe.iteration_count == 2


def test_assert_tool_called_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy-key-replay-test")
    with session.replay(FIXTURE) as probe:
        _fake_agent(client)
        probe.assert_tool_called("bash")


def test_assert_tool_called_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy-key-replay-test")
    with session.replay(FIXTURE) as probe:
        _fake_agent(client)
        with pytest.raises(AssertionError, match="web_search"):
            probe.assert_tool_called("web_search")


def test_assert_not_tool_called_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy-key-replay-test")
    with session.replay(FIXTURE) as probe:
        _fake_agent(client)
        probe.assert_not_tool_called("web_search")


def test_assert_not_tool_called_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy-key-replay-test")
    with session.replay(FIXTURE) as probe:
        _fake_agent(client)
        with pytest.raises(AssertionError, match="bash"):
            probe.assert_not_tool_called("bash")


def test_assert_max_iterations_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy-key-replay-test")
    with session.replay(FIXTURE) as probe:
        _fake_agent(client)
        probe.assert_max_iterations(5)


def test_assert_max_iterations_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy-key-replay-test")
    with session.replay(FIXTURE) as probe:
        _fake_agent(client)
        with pytest.raises(AssertionError, match="at most 1"):
            probe.assert_max_iterations(1)


def test_assert_output_contains_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy-key-replay-test")
    with session.replay(FIXTURE) as probe:
        _fake_agent(client)
        probe.assert_output_contains("file1.txt")


def test_assert_output_contains_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy-key-replay-test")
    with session.replay(FIXTURE) as probe:
        _fake_agent(client)
        with pytest.raises(AssertionError, match="missing_file"):
            probe.assert_output_contains("missing_file")


def test_tools_called_property():
    session = Session()
    client = openai.OpenAI(api_key="dummy-key-replay-test")
    with session.replay(FIXTURE) as probe:
        _fake_agent(client)
    assert probe.tools_called == ["bash"]


def test_final_output_property():
    session = Session()
    client = openai.OpenAI(api_key="dummy-key-replay-test")
    with session.replay(FIXTURE) as probe:
        _fake_agent(client)
    assert probe.final_output is not None
    assert "file1.txt" in probe.final_output


def test_replay_exhaustion_raises():
    session = Session()
    client = openai.OpenAI(api_key="dummy-key-replay-test")
    with session.replay(FIXTURE) as probe:
        _fake_agent(client)
        with pytest.raises(RuntimeError, match="replay exhausted"):
            client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "extra call"}],
            )


# ── Chained assertions ────────────────────────────────────────────────────────

def test_chained_assertions():
    session = Session()
    client = openai.OpenAI(api_key="dummy-key-replay-test")
    with session.replay(FIXTURE) as probe:
        _fake_agent(client)
    (probe
        .assert_tool_called("bash")
        .assert_not_tool_called("web_search")
        .assert_max_iterations(5)
        .assert_output_contains("file1.txt"))


# ── pytest fixture ────────────────────────────────────────────────────────────

def test_pytest_fixture(agentprobe):
    client = openai.OpenAI(api_key="dummy-key-replay-test")
    with agentprobe.replay(FIXTURE) as probe:
        _fake_agent(client)
        probe.assert_tool_called("bash")
        probe.assert_max_iterations(3)


# ── Record + replay round-trip ────────────────────────────────────────────────

def test_record_round_trip(tmp_path):
    fixture = tmp_path / "session.jsonl"
    session = Session()

    client = openai.OpenAI(api_key="dummy-key-replay-test")
    with session.replay(FIXTURE):
        with session.record(fixture):
            _fake_agent(client)

    assert fixture.exists()
    lines = fixture.read_text().strip().split("\n")
    assert len(lines) == 2
    data = json.loads(lines[0])
    assert "request" in data
    assert "response" in data


def test_auto_mode_records_when_missing(tmp_path):
    fixture = tmp_path / "new_session.jsonl"
    session = Session()
    client = openai.OpenAI(api_key="dummy-key-replay-test")

    with session.replay(FIXTURE):
        with session.auto(fixture) as probe:
            _fake_agent(client)

    assert fixture.exists()


def test_auto_mode_replays_when_exists():
    session = Session()
    client = openai.OpenAI(api_key="dummy-key-replay-test")
    with session.auto(FIXTURE) as probe:
        result = _fake_agent(client)
    assert "file1.txt" in result


def test_missing_fixture_raises():
    session = Session()
    with pytest.raises(FileNotFoundError, match="agentprobe"):
        with session.replay("tests/fixtures/nonexistent.jsonl"):
            pass
