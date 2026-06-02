"""Tests for v0.18.0: assert_first_response_latency_under, assert_output_contains_all,
assert_tool_call_args_match, fixtures --age-days, migrate --strip-pii."""
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
ANTH_TOOLS  = "tests/fixtures/anthropic_tools.jsonl"


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


# ── assert_first_response_latency_under ──────────────────────────────────────

def test_first_response_latency_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(OAI_FIXTURE) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    probe.assert_first_response_latency_under(1_000_000)


def test_first_response_latency_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(OAI_FIXTURE) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError, match="cold-start"):
        probe.assert_first_response_latency_under(0.001)


def test_first_response_latency_no_duration_passes():
    """No duration recorded → skip assertion."""
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai()) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    probe.assert_first_response_latency_under(0.001)  # no duration → pass


# ── assert_output_contains_all ───────────────────────────────────────────────

def test_output_contains_all_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai("success: 3 results cached")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    probe.assert_output_contains_all("success", "3 results", "cached")


def test_output_contains_all_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai("success: 3 results")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError, match="missing"):
        probe.assert_output_contains_all("success", "3 results", "cached")


def test_anthropic_output_contains_all_passes():
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    msg = anthropic.types.Message.model_validate({
        "id": "m1", "type": "message", "role": "assistant",
        "content": [{"type": "text", "text": "alpha beta gamma"}],
        "model": "claude-sonnet-4-6", "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    })
    with session.inject(msg) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
    probe.assert_output_contains_all("alpha", "beta", "gamma")


# ── assert_tool_call_args_match ───────────────────────────────────────────────

def test_tool_call_args_match_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai_tool("search", {"q": "agentprobe", "language": "en"})) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    probe.assert_tool_call_args_match("search", r'"language":\s*"en"')


def test_tool_call_args_match_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai_tool("search", {"q": "agentprobe"})) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError, match="pattern"):
        probe.assert_tool_call_args_match("search", r'"language":\s*"en"')


def test_anthropic_tool_call_args_match_passes():
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    with session.inject(_anth_tool("weather", {"city": "NYC", "units": "metric"})) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
    probe.assert_tool_call_args_match("weather", r'"units":\s*"metric"')


# ── CLI fixtures --age-days ───────────────────────────────────────────────────

def test_fixtures_age_days_finds_old(tmp_path):
    from agentprobe._session import _save_calls, _build_meta_line
    from agentprobe._models import RecordedCall
    # Create fixture with old recorded_at
    old_date = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    fixture = tmp_path / "old_fixture.jsonl"
    call = RecordedCall(request={}, response={})
    from agentprobe._session import _save_calls
    _save_calls([call], fixture)
    # Overwrite _meta with old date
    content = fixture.read_text()
    lines = content.splitlines()
    lines[0] = json.dumps({"_meta": {"agentprobe_version": "0.18.0", "recorded_at": old_date,
                                      "python_version": "3.13.0"}})
    fixture.write_text("\n".join(lines) + "\n")

    r = run_cli("fixtures", "--age-days", "7", str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert "old_fixture" in r.stdout


def test_fixtures_age_days_no_old(tmp_path):
    from agentprobe._session import _save_calls
    from agentprobe._models import RecordedCall
    call = RecordedCall(request={}, response={})
    _save_calls([call], tmp_path / "new_fixture.jsonl")
    # This fixture was just recorded, so it's NOT older than 365 days
    r = run_cli("fixtures", "--age-days", "365", str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert "no fixtures" in r.stdout.lower()


# ── CLI migrate --strip-pii ───────────────────────────────────────────────────

def test_migrate_strip_pii(tmp_path):
    """--strip-pii redacts matching patterns in tool call arguments."""
    fixture = tmp_path / "with_pii.jsonl"
    fixture.write_text(
        json.dumps({"_meta": {"agentprobe_version": "0.18.0",
                               "recorded_at": "2026-06-02T00:00:00Z",
                               "python_version": "3.13.0"}}) + "\n"
        + json.dumps({
            "request": {"model": "gpt-4o"},
            "response": {
                "id": "c1", "object": "chat.completion", "created": 1748700000,
                "model": "gpt-4o",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": None,
                             "tool_calls": [{"id": "t1", "type": "function",
                             "function": {"name": "send", "arguments":
                                          json.dumps({"email": "user@example.com", "msg": "hello"})}}]},
                             "finish_reason": "tool_calls", "logprobs": None}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                "system_fingerprint": None,
            },
        }) + "\n"
    )
    output = str(tmp_path / "stripped.jsonl")
    r = run_cli("migrate", str(fixture), output,
                "--strip-pii", r"[\w.+-]+@[\w-]+\.[\w.]+")
    assert r.returncode == 0, r.stderr
    assert "redaction" in r.stdout

    out_lines = [l for l in Path(output).read_text().splitlines()
                 if l.strip() and "_meta" not in l]
    call = json.loads(out_lines[0])
    args = json.loads(call["response"]["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    assert args["email"] == "[REDACTED]"
    assert args["msg"] == "hello"  # untouched
