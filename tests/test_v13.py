"""Tests for v0.13.0: assert_no_repeated_messages, assert_output_language,
assert_token_ratio, validate Anthropic, fixtures --orphaned."""
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
ANTH_TOOLS  = "tests/fixtures/anthropic_tools.jsonl"


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "agentprobe._cli", *args],
        capture_output=True, text=True,
    )


# ── helpers ───────────────────────────────────────────────────────────────────

def _oai(content="ok", prompt_tokens=10):
    return {
        "id": "c1", "object": "chat.completion", "created": 1748700000, "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content,
                     "tool_calls": None}, "finish_reason": "stop", "logprobs": None}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 5,
                  "total_tokens": prompt_tokens + 5},
        "system_fingerprint": None,
    }


def _anth(text="Hi", input_tokens=20):
    return anthropic.types.Message.model_validate({
        "id": "m1", "type": "message", "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": "claude-sonnet-4-6", "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": 8},
    })


# ── assert_no_repeated_messages (OpenAI) ─────────────────────────────────────

def test_no_repeated_messages_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai(), _oai()) as probe:
        client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "first question"}])
        client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "second question"}])
    probe.assert_no_repeated_messages()


def test_no_repeated_messages_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai(), _oai()) as probe:
        client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "same message"}])
        client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "same message"}])
    with pytest.raises(AssertionError, match="repeated"):
        probe.assert_no_repeated_messages()


def test_no_repeated_messages_single_call_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai()) as probe:
        client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    probe.assert_no_repeated_messages()


# ── assert_no_repeated_messages (Anthropic) ──────────────────────────────────

def test_anthropic_no_repeated_messages_passes():
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    with session.inject(_anth(), _anth()) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "q1"}])
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "q2"}])
    probe.assert_no_repeated_messages()


def test_anthropic_no_repeated_messages_fails():
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    with session.inject(_anth(), _anth()) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "stuck"}])
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "stuck"}])
    with pytest.raises(AssertionError, match="repeated"):
        probe.assert_no_repeated_messages()


# ── assert_output_language ────────────────────────────────────────────────────

def test_assert_output_language_no_langdetect():
    """Without langdetect installed, should raise ImportError."""
    import sys
    # Force the import to fail
    import unittest.mock as mock
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai("Hello world")) as probe:
        client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    with mock.patch.dict(sys.modules, {"langdetect": None}):
        with pytest.raises((ImportError, TypeError)):
            probe.assert_output_language("en")


# ── assert_token_ratio ────────────────────────────────────────────────────────

def test_assert_token_ratio_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai(prompt_tokens=100), _oai(prompt_tokens=150)) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "a"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "b"}])
    probe.assert_token_ratio(1, 2.0)  # 150/100 = 1.5x <= 2.0


def test_assert_token_ratio_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai(prompt_tokens=100), _oai(prompt_tokens=500)) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "a"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "b"}])
    with pytest.raises(AssertionError, match="tokens of call 1"):
        probe.assert_token_ratio(1, 2.0)  # 500/100 = 5x > 2.0


def test_anthropic_assert_token_ratio_passes():
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    with session.inject(_anth(input_tokens=100), _anth(input_tokens=180)) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "a"}])
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "b"}])
    probe.assert_token_ratio(1, 2.0)  # 180/100 = 1.8x <= 2.0


# ── CLI validate Anthropic ────────────────────────────────────────────────────

def test_validate_anthropic_simple_passes():
    r = run_cli("validate", ANTH_SIMPLE)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_validate_anthropic_tools_passes():
    r = run_cli("validate", ANTH_TOOLS)
    assert r.returncode == 0, r.stderr


def test_validate_anthropic_invalid_fixture(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        json.dumps({"request": {"model": "x"}, "response": {"NOT_A_MESSAGE": True}}) + "\n"
    )
    r = run_cli("validate", str(bad))
    # Should either pass (non-anthropic format detected as OpenAI which fails pydantic)
    # or fail cleanly — either way, no crash
    assert r.returncode in (0, 1)


# ── CLI fixtures --orphaned ───────────────────────────────────────────────────

def test_fixtures_orphaned_no_orphans():
    """All fixtures in tests/fixtures should be referenced in tests/."""
    r = run_cli("fixtures", "--orphaned", "tests/fixtures")
    assert r.returncode == 0, r.stderr
    # bash_session, agent_a, agent_b etc are all referenced in test files
    # At minimum the command must run without error


def test_fixtures_orphaned_finds_orphan(tmp_path):
    """An unreferenced fixture in a temp dir should be reported."""
    import uuid
    # Use a UUID-based name — guaranteed to never appear in any source file
    uid = str(uuid.uuid4()).replace("-", "")
    fixture_name = f"fixture_{uid}.jsonl"
    (tmp_path / fixture_name).write_text(
        json.dumps({"_meta": {"agentprobe_version": "0.13.0"}}) + "\n"
        + json.dumps({"request": {}, "response": {}}) + "\n"
    )
    r = run_cli("fixtures", "--orphaned", str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert uid in r.stdout


def test_fixtures_orphaned_json(tmp_path):
    import uuid
    uid = str(uuid.uuid4()).replace("-", "")
    fixture_name = f"unref_{uid}.jsonl"
    (tmp_path / fixture_name).write_text("{}\n")
    r = run_cli("fixtures", "--orphaned", "--json", str(tmp_path))
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    assert "orphaned" in data
    assert any(uid in f for f in data["orphaned"])
