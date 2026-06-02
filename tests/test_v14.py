"""Tests for v0.14.0: assert_tool_call_order, assert_no_empty_tool_inputs,
assert_average_latency_under, record --max-calls."""
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

OAI_FIXTURE = "tests/fixtures/bash_session.jsonl"


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "agentprobe._cli", *args],
        capture_output=True, text=True,
    )


# ── helpers ───────────────────────────────────────────────────────────────────

def _oai_tool(name, args_dict=None, duration_ms=200.0):
    from agentprobe._models import RecordedCall
    resp = {
        "id": "c1", "object": "chat.completion", "created": 1748700000, "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": None,
                     "tool_calls": [{"id": "t1", "type": "function",
                     "function": {"name": name, "arguments": json.dumps(args_dict or {"q": "x"})}}]},
                     "finish_reason": "tool_calls", "logprobs": None}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "system_fingerprint": None,
    }
    return resp


def _oai(content="ok", duration_ms=200.0):
    return {
        "id": "c2", "object": "chat.completion", "created": 1748700000, "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content,
                     "tool_calls": None}, "finish_reason": "stop", "logprobs": None}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "system_fingerprint": None,
    }


def _anth_tool(name, inp=None):
    return anthropic.types.Message.model_validate({
        "id": "m1", "type": "message", "role": "assistant",
        "content": [{"type": "tool_use", "id": "t1", "name": name, "input": inp or {"q": "x"}}],
        "model": "claude-sonnet-4-6", "stop_reason": "tool_use", "stop_sequence": None,
        "usage": {"input_tokens": 30, "output_tokens": 10},
    })


def _anth_tool_empty(name):
    return anthropic.types.Message.model_validate({
        "id": "m2", "type": "message", "role": "assistant",
        "content": [{"type": "tool_use", "id": "t2", "name": name, "input": {}}],
        "model": "claude-sonnet-4-6", "stop_reason": "tool_use", "stop_sequence": None,
        "usage": {"input_tokens": 25, "output_tokens": 8},
    })


# ── assert_tool_call_order (OpenAI) ──────────────────────────────────────────

def test_tool_call_order_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(
        _oai_tool("search"),
        _oai_tool("read_file"),
        _oai("done"),
    ) as probe:
        for _ in range(3):
            client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    probe.assert_tool_call_order("search", "read_file")


def test_tool_call_order_with_gaps_passes():
    """Other tools can appear between the required ones."""
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(
        _oai_tool("search"),
        _oai_tool("bash"),      # gap tool
        _oai_tool("summarize"),
    ) as probe:
        for _ in range(3):
            client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    probe.assert_tool_call_order("search", "summarize")  # bash is allowed in between


def test_tool_call_order_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai_tool("read_file"), _oai_tool("search")) as probe:
        for _ in range(2):
            client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError, match="not found after"):
        probe.assert_tool_call_order("search", "read_file")  # search comes after read_file


# ── assert_tool_call_order (Anthropic) ───────────────────────────────────────

def test_anthropic_tool_call_order_passes():
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    with session.inject(_anth_tool("fetch"), _anth_tool("parse")) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "a"}])
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "b"}])
    probe.assert_tool_call_order("fetch", "parse")


def test_anthropic_tool_call_order_fails():
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    with session.inject(_anth_tool("parse"), _anth_tool("fetch")) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "a"}])
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "b"}])
    with pytest.raises(AssertionError, match="not found after"):
        probe.assert_tool_call_order("fetch", "parse")


# ── assert_no_empty_tool_inputs (OpenAI) ─────────────────────────────────────

def test_no_empty_tool_inputs_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai_tool("search", {"q": "hello"})) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    probe.assert_no_empty_tool_inputs()


def test_no_empty_tool_inputs_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    # Create a response with empty tool args
    resp = {
        "id": "c3", "object": "chat.completion", "created": 1748700000, "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": None,
                     "tool_calls": [{"id": "t1", "type": "function",
                     "function": {"name": "search", "arguments": "{}"}}]},
                     "finish_reason": "tool_calls", "logprobs": None}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "system_fingerprint": None,
    }
    with session.inject(resp) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError, match="empty input"):
        probe.assert_no_empty_tool_inputs()


# ── assert_no_empty_tool_inputs (Anthropic) ──────────────────────────────────

def test_anthropic_no_empty_tool_inputs_passes():
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    with session.inject(_anth_tool("search", {"q": "hello"})) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
    probe.assert_no_empty_tool_inputs()


def test_anthropic_no_empty_tool_inputs_fails():
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    with session.inject(_anth_tool_empty("search")) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError, match="empty input"):
        probe.assert_no_empty_tool_inputs()


# ── assert_average_latency_under ─────────────────────────────────────────────

def test_average_latency_under_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(OAI_FIXTURE) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    probe.assert_average_latency_under(1_000_000)  # fixture has ~200ms recorded


def test_average_latency_under_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(OAI_FIXTURE) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    with pytest.raises(AssertionError, match="average latency"):
        probe.assert_average_latency_under(0.001)


def test_average_latency_no_durations_passes():
    """When duration_ms is None on all calls, should pass (no data to check)."""
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai()) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    # inject() creates calls with duration_ms=None
    probe.assert_average_latency_under(1.0)  # no durations → always pass


# ── CLI record --max-calls ────────────────────────────────────────────────────

def test_record_max_calls_truncates(tmp_path):
    """--max-calls N truncates the fixture to N calls."""
    script = tmp_path / "agent.py"
    script.write_text(
        "import openai\n"
        "client = openai.OpenAI(api_key='dummy')\n"
        "for _ in range(5):\n"
        "    client.chat.completions.create(model='gpt-4o', "
        "messages=[{'role': 'user', 'content': 'x'}])\n"
    )
    output = str(tmp_path / "truncated.jsonl")
    mock_resp = openai.types.chat.ChatCompletion.model_validate({
        "id": "chatcmpl-mc", "object": "chat.completion", "created": 1748700000,
        "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok",
                     "tool_calls": None}, "finish_reason": "stop", "logprobs": None}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        "system_fingerprint": None,
    })
    from agentprobe._cli import cmd_record
    args = argparse.Namespace(
        script=str(script), output=output, env=None, output_format=None,
        watch=False, interval=1.0, timeout=None, dry_run=False, append=False,
        capture_stdout=False, provider="openai", max_calls=3,
    )
    with patch("openai.resources.chat.completions.Completions.create", return_value=mock_resp):
        cmd_record(args)

    fixture = Path(output)
    assert fixture.exists()
    from conftest import fixture_lines
    lines = fixture_lines(fixture)
    assert len(lines) == 3  # truncated to 3
