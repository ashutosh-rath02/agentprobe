"""Tests for v0.5 batch 5: messages_sent, duration_percentile, assert_output_matches,
assert_tool_sequence, dump_fixture, show --json summary, validate linting."""
import json
import re
import subprocess
import sys
from pathlib import Path

import openai
import pytest

from agentprobe import Session

FIXTURE = "tests/fixtures/bash_session.jsonl"


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "agentprobe._cli", *args],
        capture_output=True, text=True,
    )


def _resp(content=None, model="gpt-4o", finish="stop"):
    return {
        "id": "chatcmpl-t",
        "object": "chat.completion",
        "created": 1748700000,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content, "tool_calls": None},
            "finish_reason": finish,
            "logprobs": None,
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "system_fingerprint": None,
    }


# ── probe.messages_sent ───────────────────────────────────────────────────────

def test_messages_sent_returns_list_per_call():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(FIXTURE) as probe:
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "list files in /tmp"}],
        )
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "done"}],
        )
    assert len(probe.messages_sent) == 2
    assert probe.messages_sent[0][0]["content"] == "list files in /tmp"


def test_messages_sent_first_call_content():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_resp("hi")) as probe:
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
        )
    assert probe.messages_sent[0][0]["role"] == "user"
    assert probe.messages_sent[0][0]["content"] == "hello"


# ── probe.duration_percentile ─────────────────────────────────────────────────

def test_duration_percentile_replay_fixture():
    """bash_session.jsonl has duration_ms=312.5 and 280.0."""
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(FIXTURE) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    p50 = probe.duration_percentile(50)
    assert 280.0 <= p50 <= 312.5  # median of two values
    assert probe.duration_percentile(0) == 280.0
    assert probe.duration_percentile(100) == 312.5


def test_duration_percentile_no_durations():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_resp("hi")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    assert probe.duration_percentile(50) == 0.0


# ── probe.assert_output_matches ───────────────────────────────────────────────

def test_assert_output_matches_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_resp("The answer is 42")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    probe.assert_output_matches(r"answer is \d+")


def test_assert_output_matches_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_resp("Hello world")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError, match="pattern"):
        probe.assert_output_matches(r"^\d+$")


def test_assert_output_matches_case_insensitive_via_flag():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_resp("HELLO WORLD")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    probe.assert_output_matches(r"(?i)hello world")


# ── probe.assert_tool_sequence ────────────────────────────────────────────────

def test_assert_tool_sequence_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(FIXTURE) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    # bash_session: call 0 uses bash, call 1 has no tools
    probe.assert_tool_sequence("bash")


def test_assert_tool_sequence_wrong_order_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(FIXTURE) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    with pytest.raises(AssertionError, match="sequence"):
        probe.assert_tool_sequence("web_search", "bash")


def test_assert_tool_sequence_empty_passes_no_tools():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_resp("hi"), _resp("bye")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    probe.assert_tool_sequence()  # empty sequence, no tools


# ── probe.dump_fixture ────────────────────────────────────────────────────────

def test_dump_fixture_saves_inject_session(tmp_path):
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    output = tmp_path / "dumped.jsonl"
    with session.inject(_resp("Hello from inject!")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    probe.dump_fixture(output)

    assert output.exists()
    data = json.loads(output.read_text().strip())
    assert data["response"]["choices"][0]["message"]["content"] == "Hello from inject!"


def test_dump_fixture_is_replayable(tmp_path):
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    output = tmp_path / "replay_me.jsonl"

    with session.inject(_resp("Saved!")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    probe.dump_fixture(output)

    client2 = openai.OpenAI(api_key="dummy")
    with Session().replay(output) as probe2:
        resp = client2.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])

    assert resp.choices[0].message.content == "Saved!"


def test_dump_fixture_is_chainable(tmp_path):
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    output = tmp_path / "chain.jsonl"
    with session.inject(_resp("chained")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    result = probe.dump_fixture(output).assert_output_contains("chained")
    assert result is probe  # chained


# ── show --json summary ───────────────────────────────────────────────────────

def test_show_json_has_summary():
    r = run_cli("show", "--json", FIXTURE)
    data = json.loads(r.stdout)
    assert "summary" in data
    s = data["summary"]
    assert s["total_calls"] == 2
    assert s["total_prompt_tokens"] == 130
    assert s["total_completion_tokens"] == 55
    assert s["total_tokens"] == 185
    assert s["streaming_calls"] == 0


def test_show_json_summary_streaming():
    r = run_cli("show", "--json", "tests/fixtures/streaming_session.jsonl")
    data = json.loads(r.stdout)
    assert data["summary"]["streaming_calls"] == 2


def test_show_json_summary_cost_nonzero():
    r = run_cli("show", "--json", FIXTURE)
    data = json.loads(r.stdout)
    assert data["summary"]["estimated_total_cost_usd"] > 0


# ── validate linting ──────────────────────────────────────────────────────────

def test_validate_warns_on_no_duration(tmp_path):
    """A fixture with no duration_ms should show a WARN but not fail."""
    no_dur = tmp_path / "no_dur.jsonl"
    no_dur.write_text(json.dumps({
        "request": {"model": "gpt-4o"},
        "response": {
            "id": "x",
            "object": "chat.completion",
            "created": 123,
            "model": "gpt-4o",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi", "tool_calls": None},
                         "finish_reason": "stop", "logprobs": None}],
            "usage": None,
            "system_fingerprint": None,
        },
        # no duration_ms
    }) + "\n")
    r = run_cli("validate", str(no_dur))
    assert r.returncode == 0  # linting doesn't fail
    assert "WARN" in r.stdout
    assert "duration_ms" in r.stdout


def test_validate_ok_message_on_good_fixture():
    r = run_cli("validate", FIXTURE)
    assert r.returncode == 0
    assert "OK" in r.stdout
