"""Tests for v0.20.0: assert_no_empty_system_prompt, assert_tool_inputs_unique,
assert_output_not_empty, fixtures --count, fixtures --delete-old, fixtures --confirm."""
import json
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import anthropic
import openai
import pytest

from agentprobe import AnthropicSession, Session

OAI_FIXTURE = "tests/fixtures/bash_session.jsonl"


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "agentprobe._cli", *args],
        capture_output=True, text=True,
    )


# ── helpers ───────────────────────────────────────────────────────────────────

def _oai(content="ok"):
    return {
        "id": "c1", "object": "chat.completion", "created": 1748700000, "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content,
                     "tool_calls": None}, "finish_reason": "stop", "logprobs": None}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "system_fingerprint": None,
    }


def _oai_tool(name, args_dict=None):
    return {
        "id": "c2", "object": "chat.completion", "created": 1748700000, "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": None,
                     "tool_calls": [{"id": "t1", "type": "function",
                     "function": {"name": name, "arguments": json.dumps(args_dict or {})}}]},
                     "finish_reason": "tool_calls", "logprobs": None}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "system_fingerprint": None,
    }


def _anth(text="Hello"):
    return anthropic.types.Message.model_validate({
        "id": "m1", "type": "message", "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": "claude-sonnet-4-6", "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 20, "output_tokens": 8},
    })


def _anth_tool(name, inp=None):
    return anthropic.types.Message.model_validate({
        "id": "m2", "type": "message", "role": "assistant",
        "content": [{"type": "tool_use", "id": "t1", "name": name, "input": inp or {}}],
        "model": "claude-sonnet-4-6", "stop_reason": "tool_use", "stop_sequence": None,
        "usage": {"input_tokens": 30, "output_tokens": 10},
    })


# ── assert_no_empty_system_prompt (OpenAI) ────────────────────────────────────

def test_no_empty_system_prompt_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai("hello")) as probe:
        client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "hi"},
            ]
        )
    probe.assert_no_empty_system_prompt()


def test_no_empty_system_prompt_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai("hello")) as probe:
        client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": ""},  # empty!
                {"role": "user", "content": "hi"},
            ]
        )
    with pytest.raises(AssertionError, match="empty system"):
        probe.assert_no_empty_system_prompt()


def test_no_system_at_all_passes():
    """Calls without any system message should pass."""
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai("hello")) as probe:
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}]
        )
    probe.assert_no_empty_system_prompt()


# ── assert_no_empty_system_prompt (Anthropic) ─────────────────────────────────

def test_anthropic_no_empty_system_prompt_passes():
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    with session.inject(_anth("hello")) as probe:
        client.messages.create(
            model="claude-sonnet-4-6", max_tokens=1024,
            system="You are a helpful assistant.",
            messages=[{"role": "user", "content": "hi"}]
        )
    probe.assert_no_empty_system_prompt()


def test_anthropic_no_empty_system_prompt_fails():
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    with session.inject(_anth("hello")) as probe:
        client.messages.create(
            model="claude-sonnet-4-6", max_tokens=1024,
            system="",  # empty!
            messages=[{"role": "user", "content": "hi"}]
        )
    with pytest.raises(AssertionError, match="empty system"):
        probe.assert_no_empty_system_prompt()


# ── assert_tool_inputs_unique ─────────────────────────────────────────────────

def test_tool_inputs_unique_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(
        _oai_tool("search", {"q": "cats"}),
        _oai_tool("search", {"q": "dogs"}),
    ) as probe:
        for _ in range(2):
            client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    probe.assert_tool_inputs_unique("search")


def test_tool_inputs_unique_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(
        _oai_tool("search", {"q": "same"}),
        _oai_tool("search", {"q": "same"}),
    ) as probe:
        for _ in range(2):
            client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError, match="duplicates"):
        probe.assert_tool_inputs_unique("search")


def test_anthropic_tool_inputs_unique_passes():
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    with session.inject(
        _anth_tool("fetch", {"url": "https://a.com"}),
        _anth_tool("fetch", {"url": "https://b.com"}),
    ) as probe:
        for _ in range(2):
            client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                                   messages=[{"role": "user", "content": "x"}])
    probe.assert_tool_inputs_unique("fetch")


# ── assert_output_not_empty ───────────────────────────────────────────────────

def test_output_not_empty_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai("Hello world!")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    probe.assert_output_not_empty()


def test_output_not_empty_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai("   ")) as probe:  # whitespace only
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError, match="empty"):
        probe.assert_output_not_empty()


def test_anthropic_output_not_empty_passes():
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    with session.inject(_anth("Non-empty response")) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
    probe.assert_output_not_empty()


# ── CLI fixtures --count ──────────────────────────────────────────────────────

def test_fixtures_count_exits_zero():
    r = run_cli("fixtures", "--count", "tests/fixtures")
    assert r.returncode == 0, r.stderr
    assert "fixture" in r.stdout


def test_fixtures_count_json():
    r = run_cli("fixtures", "--count", "--json", "tests/fixtures")
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    assert isinstance(data["count"], int)
    assert data["count"] > 0


# ── CLI fixtures --delete-old ─────────────────────────────────────────────────

def test_delete_old_requires_confirm(tmp_path):
    from agentprobe._session import _save_calls
    from agentprobe._models import RecordedCall
    _save_calls([RecordedCall(request={}, response={})], tmp_path / "f.jsonl")
    r = run_cli("fixtures", "--delete-old", "0", str(tmp_path))  # without --confirm
    assert r.returncode != 0
    assert "--confirm" in r.stderr


def test_delete_old_with_confirm(tmp_path):
    from agentprobe._session import _save_calls
    from agentprobe._models import RecordedCall
    fixture = tmp_path / "old.jsonl"
    _save_calls([RecordedCall(request={}, response={})], fixture)
    # Override the _meta recorded_at to be 100 days old
    old_date = (datetime.now(timezone.utc) - timedelta(days=100)).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = fixture.read_text().splitlines()
    lines[0] = json.dumps({"_meta": {"agentprobe_version": "0.20.0",
                                      "recorded_at": old_date, "python_version": "3.13"}})
    fixture.write_text("\n".join(lines) + "\n")

    r = run_cli("fixtures", "--delete-old", "30", "--confirm", str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert not fixture.exists()  # was deleted
    assert "deleted 1" in r.stdout
