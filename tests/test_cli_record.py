"""Tests for agentprobe record and show --json CLI commands."""
import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import openai
import openai.types.chat
import pytest

FIXTURE = "tests/fixtures/bash_session.jsonl"
STREAMING_FIXTURE = "tests/fixtures/streaming_session.jsonl"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_mock_response(content="Hello!", tool_calls=None):
    return openai.types.chat.ChatCompletion.model_validate({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1748700000,
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content, "tool_calls": tool_calls},
            "finish_reason": "stop",
            "logprobs": None,
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "system_fingerprint": None,
    })


def _script(tmp_path: Path, code: str) -> Path:
    p = tmp_path / "agent.py"
    p.write_text(code)
    return p


# ── cmd_record tests ──────────────────────────────────────────────────────────

def test_record_captures_single_call(tmp_path):
    script = _script(tmp_path, """
import openai
client = openai.OpenAI(api_key="dummy")
client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "hello"}],
)
""")
    output = tmp_path / "recorded.jsonl"
    mock_resp = _make_mock_response("Hello!")

    from agentprobe._cli import cmd_record
    args = argparse.Namespace(script=str(script), output=str(output))

    with patch("openai.resources.chat.completions.Completions.create", return_value=mock_resp):
        cmd_record(args)

    assert output.exists()
    from conftest import fixture_lines
    lines = fixture_lines(output)
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["request"]["model"] == "gpt-4o"
    assert data["response"]["choices"][0]["message"]["content"] == "Hello!"


def test_record_captures_multiple_calls(tmp_path):
    script = _script(tmp_path, """
import openai
client = openai.OpenAI(api_key="dummy")
for _ in range(3):
    client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "ping"}],
    )
""")
    output = tmp_path / "multi.jsonl"
    mock_resp = _make_mock_response("pong")

    from agentprobe._cli import cmd_record
    args = argparse.Namespace(script=str(script), output=str(output))

    with patch("openai.resources.chat.completions.Completions.create", return_value=mock_resp):
        cmd_record(args)

    from conftest import fixture_lines
    lines = fixture_lines(output)
    assert len(lines) == 3


def test_record_missing_script_exits(tmp_path, capsys):
    from agentprobe._cli import cmd_record
    args = argparse.Namespace(script=str(tmp_path / "nonexistent.py"), output=str(tmp_path / "out.jsonl"))
    with pytest.raises(SystemExit) as exc:
        cmd_record(args)
    assert exc.value.code == 1


def test_record_output_is_valid_fixture(tmp_path):
    """Recorded file must be replayable by Session."""
    script = _script(tmp_path, """
import openai
client = openai.OpenAI(api_key="dummy")
client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
""")
    output = tmp_path / "replay_me.jsonl"
    mock_resp = _make_mock_response("Hi back!")

    from agentprobe._cli import cmd_record
    args = argparse.Namespace(script=str(script), output=str(output))
    with patch("openai.resources.chat.completions.Completions.create", return_value=mock_resp):
        cmd_record(args)

    # Replay the recorded fixture
    from agentprobe import Session
    client = openai.OpenAI(api_key="dummy")
    with Session().replay(str(output)) as probe:
        resp = client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}]
        )
    assert resp.choices[0].message.content == "Hi back!"
    assert probe.iteration_count == 1


# ── show --json tests ─────────────────────────────────────────────────────────

def run_cli(*args):
    import subprocess
    return subprocess.run(
        [sys.executable, "-m", "agentprobe._cli", *args],
        capture_output=True, text=True,
    )


def test_show_json_is_valid_json():
    r = run_cli("show", "--json", FIXTURE)
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert "calls" in data and "summary" in data
    assert len(data["calls"]) == 2


def test_show_json_contains_expected_fields():
    r = run_cli("show", "--json", FIXTURE)
    data = json.loads(r.stdout)
    entry = data["calls"][0]
    assert "model" in entry
    assert "finish_reason" in entry
    assert "prompt_tokens" in entry
    assert "completion_tokens" in entry
    assert "tools_called" in entry
    assert "streaming" in entry


def test_show_json_tool_name_present():
    r = run_cli("show", "--json", FIXTURE)
    data = json.loads(r.stdout)
    all_tools = [t for entry in data["calls"] for t in entry["tools_called"]]
    assert "bash" in all_tools


def test_show_json_streaming_flag():
    r = run_cli("show", "--json", STREAMING_FIXTURE)
    data = json.loads(r.stdout)
    assert all(entry["streaming"] is True for entry in data["calls"])


def test_show_json_non_streaming_flag():
    r = run_cli("show", "--json", FIXTURE)
    data = json.loads(r.stdout)
    assert all(entry["streaming"] is False for entry in data["calls"])
