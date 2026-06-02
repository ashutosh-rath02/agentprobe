"""Tests for v0.17.0: assert_tool_called_n_times, assert_no_sensitive_in_messages,
assert_tool_input_contains, fixtures --label."""
import json
import subprocess
import sys
from pathlib import Path

import anthropic
import openai
import pytest

from agentprobe import AnthropicSession, Session

OAI_FIXTURE = "tests/fixtures/bash_session.jsonl"
ANTH_SIMPLE = "tests/fixtures/anthropic_simple.jsonl"


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


def _anth_tool(name, inp=None):
    return anthropic.types.Message.model_validate({
        "id": "m1", "type": "message", "role": "assistant",
        "content": [{"type": "tool_use", "id": "t1", "name": name, "input": inp or {"q": "x"}}],
        "model": "claude-sonnet-4-6", "stop_reason": "tool_use", "stop_sequence": None,
        "usage": {"input_tokens": 30, "output_tokens": 10},
    })


# ── assert_tool_called_n_times (OpenAI) ──────────────────────────────────────

def test_tool_called_n_times_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(
        _oai_tool("search"),
        _oai_tool("search"),
        _oai_tool("search"),
    ) as probe:
        for _ in range(3):
            client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    probe.assert_tool_called_n_times("search", 3)


def test_tool_called_n_times_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai_tool("search"), _oai_tool("search")) as probe:
        for _ in range(2):
            client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError, match="2 time"):
        probe.assert_tool_called_n_times("search", 3)


# ── assert_tool_called_n_times (Anthropic) ───────────────────────────────────

def test_anthropic_tool_called_n_times_passes():
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    with session.inject(_anth_tool("bash"), _anth_tool("bash")) as probe:
        for _ in range(2):
            client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                                   messages=[{"role": "user", "content": "x"}])
    probe.assert_tool_called_n_times("bash", 2)


# ── assert_no_sensitive_in_messages (OpenAI) ─────────────────────────────────

def test_no_sensitive_messages_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai_tool("search")) as probe:
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "what is the weather?"}],
        )
    probe.assert_no_sensitive_in_messages(r"sk-[A-Za-z0-9]{20,}")


def test_no_sensitive_messages_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai_tool("search")) as probe:
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "my api key is sk-abcdefghij1234567890"}],
        )
    with pytest.raises(AssertionError, match="sensitive pattern"):
        probe.assert_no_sensitive_in_messages(r"sk-[A-Za-z0-9]{20,}")


# ── assert_no_sensitive_in_messages (Anthropic) ──────────────────────────────

def test_anthropic_no_sensitive_messages_passes():
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    with session.inject(_anth_tool("lookup")) as probe:
        client.messages.create(
            model="claude-sonnet-4-6", max_tokens=1024,
            messages=[{"role": "user", "content": "safe content"}],
        )
    probe.assert_no_sensitive_in_messages(r"\b\d{16}\b")


def test_anthropic_no_sensitive_messages_fails():
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    with session.inject(_anth_tool("lookup")) as probe:
        client.messages.create(
            model="claude-sonnet-4-6", max_tokens=1024,
            messages=[{"role": "user", "content": "my ssn is 123-45-6789"}],
        )
    with pytest.raises(AssertionError, match="sensitive pattern"):
        probe.assert_no_sensitive_in_messages(r"\d{3}-\d{2}-\d{4}")


# ── assert_tool_input_contains (OpenAI) ──────────────────────────────────────

def test_tool_input_contains_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai_tool("search", {"q": "agentprobe", "lang": "en"})) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    probe.assert_tool_input_contains("search", "lang", "en")


def test_tool_input_contains_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai_tool("search", {"q": "agentprobe"})) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError, match="never called with"):
        probe.assert_tool_input_contains("search", "lang", "en")


# ── assert_tool_input_contains (Anthropic) ───────────────────────────────────

def test_anthropic_tool_input_contains_passes():
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    with session.inject(_anth_tool("weather", {"city": "NYC", "units": "metric"})) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
    probe.assert_tool_input_contains("weather", "units", "metric")


def test_anthropic_tool_input_contains_fails():
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    with session.inject(_anth_tool("weather", {"city": "NYC"})) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError, match="never called with"):
        probe.assert_tool_input_contains("weather", "units", "metric")


# ── CLI fixtures --label ──────────────────────────────────────────────────────

def test_fixtures_label_finds_match(tmp_path):
    from agentprobe._session import _save_calls
    from agentprobe._models import RecordedCall
    call = RecordedCall(request={"model": "gpt-4o"}, response={
        "id": "c1", "object": "chat.completion", "created": 1748700000, "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi",
                     "tool_calls": None}, "finish_reason": "stop", "logprobs": None}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        "system_fingerprint": None,
    })
    _save_calls([call], tmp_path / "labeled.jsonl", label="my-tag")
    _save_calls([call], tmp_path / "nolabel.jsonl")

    r = run_cli("fixtures", "--label", "my-tag", str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert "labeled.jsonl" in r.stdout
    assert "nolabel.jsonl" not in r.stdout


def test_fixtures_label_no_match(tmp_path):
    from agentprobe._session import _save_calls
    from agentprobe._models import RecordedCall
    call = RecordedCall(request={}, response={})
    _save_calls([call], tmp_path / "nolabel.jsonl")
    r = run_cli("fixtures", "--label", "nonexistent-tag", str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert "no fixtures" in r.stdout.lower()


def test_fixtures_label_json(tmp_path):
    from agentprobe._session import _save_calls
    from agentprobe._models import RecordedCall
    call = RecordedCall(request={"model": "gpt-4o"}, response={
        "id": "c1", "object": "chat.completion", "created": 1748700000, "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi",
                     "tool_calls": None}, "finish_reason": "stop", "logprobs": None}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        "system_fingerprint": None,
    })
    _save_calls([call], tmp_path / "run1.jsonl", label="prod")
    r = run_cli("fixtures", "--label", "prod", "--json", str(tmp_path))
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    assert data["label"] == "prod"
    assert any("run1.jsonl" in f for f in data["fixtures"])
