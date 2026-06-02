"""Tests for v0.16.0: assert_final_tool_not_called, assert_output_word_count,
assert_no_pii_in_tool_inputs, record --label, show --stdout."""
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


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "agentprobe._cli", *args],
        capture_output=True, text=True,
    )


# ── helpers ───────────────────────────────────────────────────────────────────

def _oai_tool(name, args_dict=None):
    return {
        "id": "c1", "object": "chat.completion", "created": 1748700000, "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": None,
                     "tool_calls": [{"id": "t1", "type": "function",
                     "function": {"name": name, "arguments": json.dumps(args_dict or {"q": "x"})}}]},
                     "finish_reason": "tool_calls", "logprobs": None}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "system_fingerprint": None,
    }


def _oai(content="ok"):
    return {
        "id": "c2", "object": "chat.completion", "created": 1748700000, "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content,
                     "tool_calls": None}, "finish_reason": "stop", "logprobs": None}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "system_fingerprint": None,
    }


def _anth(text="Hello world"):
    return anthropic.types.Message.model_validate({
        "id": "m1", "type": "message", "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": "claude-sonnet-4-6", "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 20, "output_tokens": 8},
    })


def _anth_tool(name, inp=None):
    return anthropic.types.Message.model_validate({
        "id": "m2", "type": "message", "role": "assistant",
        "content": [{"type": "tool_use", "id": "t1", "name": name, "input": inp or {"q": "x"}}],
        "model": "claude-sonnet-4-6", "stop_reason": "tool_use", "stop_sequence": None,
        "usage": {"input_tokens": 30, "output_tokens": 10},
    })


# ── assert_final_tool_not_called (OpenAI) ────────────────────────────────────

def test_final_tool_not_called_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai_tool("search"), _oai("done")) as probe:
        for _ in range(2):
            client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    probe.assert_final_tool_not_called("search")


def test_final_tool_not_called_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai_tool("search")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError, match="final call"):
        probe.assert_final_tool_not_called("search")


# ── assert_final_tool_not_called (Anthropic) ─────────────────────────────────

def test_anthropic_final_tool_not_called_passes():
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    with session.inject(_anth_tool("search"), _anth("done")) as probe:
        for _ in range(2):
            client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                                   messages=[{"role": "user", "content": "x"}])
    probe.assert_final_tool_not_called("search")


def test_anthropic_final_tool_not_called_fails():
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    with session.inject(_anth_tool("bash")) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError, match="final call"):
        probe.assert_final_tool_not_called("bash")


# ── assert_output_word_count ──────────────────────────────────────────────────

def test_output_word_count_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai("Hello world test")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    probe.assert_output_word_count(1, 10)


def test_output_word_count_too_short():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai("Hi")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError, match="at least"):
        probe.assert_output_word_count(5)


def test_output_word_count_too_long():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai("one two three four five")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError, match="at most"):
        probe.assert_output_word_count(0, 3)


def test_anthropic_output_word_count_passes():
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    with session.inject(_anth("The quick brown fox")) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
    probe.assert_output_word_count(3, 10)


# ── assert_no_pii_in_tool_inputs ─────────────────────────────────────────────

def test_no_pii_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai_tool("search", {"q": "weather in NYC"})) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    probe.assert_no_pii_in_tool_inputs(r"\b\d{3}-\d{2}-\d{4}\b")  # SSN pattern


def test_no_pii_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai_tool("send_email", {"to": "user@example.com", "body": "test"})) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError, match="PII pattern"):
        probe.assert_no_pii_in_tool_inputs(r"[\w.+-]+@[\w-]+\.[\w.]+")  # email pattern


def test_anthropic_no_pii_passes():
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    with session.inject(_anth_tool("search", {"q": "safe query"})) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
    probe.assert_no_pii_in_tool_inputs(r"\b\d{16}\b")  # credit card pattern


def test_anthropic_no_pii_fails():
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    with session.inject(_anth_tool("lookup", {"email": "secret@corp.com"})) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError, match="PII pattern"):
        probe.assert_no_pii_in_tool_inputs(r"[\w.+-]+@[\w-]+\.[\w.]+")


# ── record --label ────────────────────────────────────────────────────────────

def test_record_label_stored_in_meta(tmp_path):
    script = tmp_path / "agent.py"
    script.write_text(
        "import openai\n"
        "client = openai.OpenAI(api_key='dummy')\n"
        "client.chat.completions.create(model='gpt-4o', "
        "messages=[{'role': 'user', 'content': 'hi'}])\n"
    )
    output = str(tmp_path / "labeled.jsonl")
    mock_resp = openai.types.chat.ChatCompletion.model_validate({
        "id": "chatcmpl-lbl", "object": "chat.completion", "created": 1748700000,
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
        capture_stdout=False, provider="openai", max_calls=None, label="ci-run-42",
    )
    with patch("openai.resources.chat.completions.Completions.create", return_value=mock_resp):
        cmd_record(args)

    from agentprobe._cli import _load_meta
    meta = _load_meta(output)
    assert meta.get("label") == "ci-run-42"


# ── show --stdout ─────────────────────────────────────────────────────────────

def test_show_stdout_no_capture():
    """--stdout on a fixture without captured output shows a hint."""
    r = run_cli("show", "--stdout", OAI_FIXTURE)
    assert r.returncode == 0, r.stderr
    assert "no captured stdout" in r.stdout


def test_show_stdout_with_capture(tmp_path):
    """--stdout on a fixture recorded with --capture-stdout shows the output."""
    from agentprobe._session import _save_calls, _build_meta_line
    from agentprobe._models import RecordedCall
    fixture = tmp_path / "with_stdout.jsonl"
    call = RecordedCall(
        request={"model": "gpt-4o", "messages": []},
        response={
            "id": "c1", "object": "chat.completion", "created": 1748700000,
            "model": "gpt-4o",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi",
                         "tool_calls": None}, "finish_reason": "stop", "logprobs": None}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            "system_fingerprint": None,
        },
        duration_ms=100.0,
    )
    _save_calls([call], fixture, meta_extra={"stdout": "hello from script\n"})

    r = run_cli("show", "--stdout", str(fixture))
    assert r.returncode == 0, r.stderr
    assert "hello from script" in r.stdout
