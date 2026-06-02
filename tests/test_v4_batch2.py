"""Tests for v0.4 batch 2: streaming cost estimation, agentprobe record async."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import openai
import openai.types.chat
import pytest

from agentprobe import Session

STREAMING_USAGE_FIXTURE = "tests/fixtures/streaming_with_usage.jsonl"


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "agentprobe._cli", *args],
        capture_output=True, text=True,
    )


# ── Streaming cost estimation ─────────────────────────────────────────────────

def test_streaming_with_usage_has_tokens():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(STREAMING_USAGE_FIXTURE) as probe:
        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )
        for _ in stream:
            pass
    assert probe.total_input_tokens == 10
    assert probe.total_output_tokens == 5
    assert probe.total_tokens == 15


def test_streaming_with_usage_cost_nonzero():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(STREAMING_USAGE_FIXTURE) as probe:
        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )
        for _ in stream:
            pass
    assert probe.estimated_cost_usd > 0


def test_streaming_with_usage_max_tokens_assertion():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(STREAMING_USAGE_FIXTURE) as probe:
        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )
        for _ in stream:
            pass
    probe.assert_max_tokens(100)


def test_streaming_validate_with_usage_fixture():
    r = run_cli("validate", STREAMING_USAGE_FIXTURE)
    assert r.returncode == 0
    assert "OK" in r.stdout


def test_streaming_show_json_reports_usage():
    r = run_cli("show", "--json", STREAMING_USAGE_FIXTURE)
    data = json.loads(r.stdout)
    assert data["calls"][0]["prompt_tokens"] == 10
    assert data["calls"][0]["completion_tokens"] == 5


# ── agentprobe record async script detection ──────────────────────────────────

def _make_mock_response(content="Hi!"):
    return openai.types.chat.ChatCompletion.model_validate({
        "id": "chatcmpl-async",
        "object": "chat.completion",
        "created": 1748700000,
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content, "tool_calls": None},
            "finish_reason": "stop",
            "logprobs": None,
        }],
        "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
        "system_fingerprint": None,
    })


def test_record_async_script_captures_calls(tmp_path):
    """An async script using asyncio.run() should be captured correctly."""
    script = tmp_path / "async_agent.py"
    script.write_text("""
import asyncio
import openai

async def main():
    client = openai.AsyncOpenAI(api_key="dummy")
    await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hello async"}],
    )

asyncio.run(main())
""")
    output = tmp_path / "async_recorded.jsonl"

    from agentprobe._cli import cmd_record

    mock_resp = _make_mock_response("Async response!")
    args = argparse.Namespace(script=str(script), output=str(output))

    with patch("openai.resources.chat.completions.AsyncCompletions.create",
               new=AsyncMock(return_value=mock_resp)):
        cmd_record(args)

    assert output.exists()
    lines = output.read_text().strip().split("\n")
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["response"]["choices"][0]["message"]["content"] == "Async response!"


def test_record_async_note_in_output(tmp_path, capsys):
    """Output should say '(async script)' for async scripts."""
    script = tmp_path / "async_agent.py"
    script.write_text("""
import asyncio
import openai

async def main():
    client = openai.AsyncOpenAI(api_key="dummy")
    await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
    )

asyncio.run(main())
""")
    output = tmp_path / "out.jsonl"
    from agentprobe._cli import cmd_record
    args = argparse.Namespace(script=str(script), output=str(output))

    mock_resp = _make_mock_response()
    with patch("openai.resources.chat.completions.AsyncCompletions.create",
               new=AsyncMock(return_value=mock_resp)):
        cmd_record(args)

    captured = capsys.readouterr()
    assert "async script" in captured.out


def test_record_sync_no_async_note(tmp_path, capsys):
    """Sync scripts should NOT have '(async script)' in output."""
    script = tmp_path / "sync_agent.py"
    script.write_text("""
import openai
client = openai.OpenAI(api_key="dummy")
client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "sync hi"}],
)
""")
    output = tmp_path / "sync_out.jsonl"
    from agentprobe._cli import cmd_record
    args = argparse.Namespace(script=str(script), output=str(output))

    mock_resp = _make_mock_response("Sync!")
    with patch("openai.resources.chat.completions.Completions.create",
               return_value=mock_resp):
        cmd_record(args)

    captured = capsys.readouterr()
    assert "async script" not in captured.out
