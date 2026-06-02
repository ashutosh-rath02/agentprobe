"""Tests for v0.12.0: assert_no_hallucinated_tool_calls, assert_max_tool_calls,
assert_system_prompt_present, stats --by-model, record --provider anthropic."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import anthropic
import openai
import openai.types.chat
import pytest

from agentprobe import AnthropicSession, Session

OAI_FIXTURE  = "tests/fixtures/bash_session.jsonl"
ANTH_SIMPLE  = "tests/fixtures/anthropic_simple.jsonl"
ANTH_TOOLS   = "tests/fixtures/anthropic_tools.jsonl"


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "agentprobe._cli", *args],
        capture_output=True, text=True,
    )


# ── helpers ───────────────────────────────────────────────────────────────────

def _oai_resp(content="ok", tool_name=None, tool_args=None):
    tool_calls = None
    if tool_name:
        tool_calls = [{"id": "c1", "type": "function",
                       "function": {"name": tool_name, "arguments": json.dumps(tool_args or {})}}]
    return {
        "id": "chatcmpl-v12", "object": "chat.completion", "created": 1748700000,
        "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant",
                     "content": content, "tool_calls": tool_calls},
                     "finish_reason": "stop", "logprobs": None}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "system_fingerprint": None,
    }


def _anth_msg(text="Hi", stop_reason="end_turn"):
    return anthropic.types.Message.model_validate({
        "id": "msg_v12", "type": "message", "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": "claude-sonnet-4-6", "stop_reason": stop_reason,
        "stop_sequence": None, "usage": {"input_tokens": 20, "output_tokens": 8},
    })


def _anth_tool(name="search", inp=None):
    return anthropic.types.Message.model_validate({
        "id": "msg_t12", "type": "message", "role": "assistant",
        "content": [{"type": "tool_use", "id": "t1", "name": name,
                     "input": inp or {"q": "x"}}],
        "model": "claude-sonnet-4-6", "stop_reason": "tool_use",
        "stop_sequence": None, "usage": {"input_tokens": 30, "output_tokens": 10},
    })


# ── assert_no_hallucinated_tool_calls (OpenAI) ────────────────────────────────

def test_no_hallucinated_tools_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(
        _oai_resp(tool_name="search"),
        _oai_resp(tool_name="read_file"),
    ) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "a"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "b"}])
    probe.assert_no_hallucinated_tool_calls("search", "read_file", "bash")


def test_no_hallucinated_tools_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai_resp(tool_name="unknown_tool")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError, match="undeclared tool"):
        probe.assert_no_hallucinated_tool_calls("search", "read_file")


def test_no_hallucinated_tools_no_tools_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai_resp("Hello")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    probe.assert_no_hallucinated_tool_calls("search")  # no tools called = pass


# ── assert_no_hallucinated_tool_calls (Anthropic) ────────────────────────────

def test_anthropic_no_hallucinated_tools_passes():
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    with session.inject(_anth_tool("search")) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
    probe.assert_no_hallucinated_tool_calls("search", "bash")


def test_anthropic_no_hallucinated_tools_fails():
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    with session.inject(_anth_tool("phantom_tool")) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError, match="undeclared tool"):
        probe.assert_no_hallucinated_tool_calls("search")


# ── assert_max_tool_calls ─────────────────────────────────────────────────────

def test_assert_max_tool_calls_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai_resp(tool_name="search"), _oai_resp("done")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "a"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "b"}])
    probe.assert_max_tool_calls(3)


def test_assert_max_tool_calls_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(
        _oai_resp(tool_name="s1"),
        _oai_resp(tool_name="s2"),
        _oai_resp(tool_name="s3"),
    ) as probe:
        for _ in range(3):
            client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError, match="tool calls"):
        probe.assert_max_tool_calls(2)


def test_anthropic_assert_max_tool_calls_passes():
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    with session.inject(_anth_tool(), _anth_msg("done")) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "a"}])
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "b"}])
    probe.assert_max_tool_calls(5)


# ── assert_system_prompt_present ─────────────────────────────────────────────

def test_assert_system_prompt_present_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai_resp("hello")) as probe:
        client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hi"},
            ]
        )
    probe.assert_system_prompt_present()


def test_assert_system_prompt_present_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai_resp("hello")) as probe:
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
        )
    with pytest.raises(AssertionError, match="system message"):
        probe.assert_system_prompt_present()


def test_anthropic_assert_system_prompt_present_passes():
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    with session.inject(_anth_msg("hello")) as probe:
        client.messages.create(
            model="claude-sonnet-4-6", max_tokens=1024,
            system="You are a helpful assistant.",
            messages=[{"role": "user", "content": "hi"}],
        )
    probe.assert_system_prompt_present()


def test_anthropic_assert_system_prompt_present_fails():
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    with session.inject(_anth_msg("hello")) as probe:
        client.messages.create(
            model="claude-sonnet-4-6", max_tokens=1024,
            messages=[{"role": "user", "content": "hi"}],
        )
    with pytest.raises(AssertionError, match="system prompt"):
        probe.assert_system_prompt_present()


# ── CLI stats --by-model ──────────────────────────────────────────────────────

def test_stats_by_model_exits_zero():
    r = run_cli("stats", "--by-model", "tests/fixtures")
    assert r.returncode == 0, r.stderr
    assert "gpt-4o" in r.stdout or "claude" in r.stdout


def test_stats_by_model_json():
    r = run_cli("stats", "--by-model", "--json", "tests/fixtures")
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    assert isinstance(data, dict)
    # Should have at least one model entry
    assert len(data) > 0


def test_stats_by_model_includes_anthropic():
    r = run_cli("stats", "--by-model", "--json", "tests/fixtures")
    data = json.loads(r.stdout)
    # anthropic_simple.jsonl and anthropic_tools.jsonl should show claude model
    assert any("claude" in m for m in data)


# ── CLI record --provider anthropic ─────────────────────────────────────────

def test_record_provider_anthropic_cli(tmp_path):
    """record --provider anthropic intercepts Anthropic messages.create."""
    script = tmp_path / "agent.py"
    script.write_text(
        "import anthropic\n"
        "client = anthropic.Anthropic(api_key='dummy')\n"
        "client.messages.create(model='claude-sonnet-4-6', max_tokens=1024,\n"
        "    messages=[{'role': 'user', 'content': 'hi'}])\n"
    )
    output = str(tmp_path / "anthropic_recorded.jsonl")
    mock_msg = anthropic.types.Message.model_validate({
        "id": "msg_rec", "type": "message", "role": "assistant",
        "content": [{"type": "text", "text": "recorded"}],
        "model": "claude-sonnet-4-6", "stop_reason": "end_turn",
        "stop_sequence": None, "usage": {"input_tokens": 5, "output_tokens": 3},
    })
    from agentprobe._cli import cmd_record
    args = argparse.Namespace(
        script=str(script), output=output, env=None, output_format=None,
        watch=False, interval=1.0, timeout=None, dry_run=False, append=False,
        capture_stdout=False, provider="anthropic",
    )
    with patch("anthropic.resources.messages.Messages.create", return_value=mock_msg):
        cmd_record(args)

    fixture = Path(output)
    assert fixture.exists()
    lines = [l for l in fixture.read_text().splitlines() if l.strip() and "_meta" not in l]
    assert len(lines) == 1
    call = json.loads(lines[0])
    assert call["response"].get("stop_reason") == "end_turn"
    assert call["response"].get("role") == "assistant"
