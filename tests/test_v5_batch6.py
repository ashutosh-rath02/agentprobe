"""Tests for v0.5 batch 6/7: strict replay, assert_finish_reason_all,
messages_sent fix, record --env, xdist auto() coordination."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import openai
import openai.types.chat
import pytest

from agentprobe import Session

FIXTURE = "tests/fixtures/bash_session.jsonl"
AGENT_A = "tests/fixtures/agent_a.jsonl"


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "agentprobe._cli", *args],
        capture_output=True, text=True,
    )


def _resp(content="Hi!", finish="stop"):
    return {
        "id": "chatcmpl-t", "object": "chat.completion", "created": 1748700000,
        "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content,
                     "tool_calls": None}, "finish_reason": finish, "logprobs": None}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        "system_fingerprint": None,
    }


# ── messages_sent captures actual kwargs ─────────────────────────────────────

def test_messages_sent_inject_captures_actual_messages():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_resp("hi")) as probe:
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello from inject"}],
        )
    assert len(probe.messages_sent) == 1
    assert probe.messages_sent[0][0]["content"] == "hello from inject"


def test_messages_sent_replay_captures_actual_messages():
    """messages_sent should reflect what was passed in the test, not stored fixture values."""
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(AGENT_A) as probe:
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "What is 2+2?"}],
        )
    assert probe.messages_sent[0][0]["content"] == "What is 2+2?"


# ── Session.replay(strict=True) ───────────────────────────────────────────────

def test_strict_replay_passes_when_all_calls_consumed():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    # AGENT_A has 1 call — consume it
    with session.replay(AGENT_A, strict=True) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    assert probe.iteration_count == 1


def test_strict_replay_raises_when_calls_not_consumed():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    # FIXTURE has 2 calls — consume only 1
    with pytest.raises(AssertionError, match="strict replay"):
        with session.replay(FIXTURE, strict=True) as probe:
            client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
            # second call NOT made → strict mode raises on exit


def test_strict_replay_passes_when_all_two_consumed():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(FIXTURE, strict=True) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])


# ── assert_finish_reason_all ──────────────────────────────────────────────────

def test_assert_finish_reason_all_stop():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_resp("a", "stop"), _resp("b", "stop")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    probe.assert_finish_reason_all("stop")


def test_assert_finish_reason_all_fails_on_mismatch():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_resp("a", "stop"), _resp("b", "length")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    with pytest.raises(AssertionError, match="finish_reason"):
        probe.assert_finish_reason_all("stop")


# ── agentprobe record --env ───────────────────────────────────────────────────

def test_record_env_loads_vars(tmp_path):
    """--env should load variables from the file before running the script."""
    env_file = tmp_path / ".env"
    env_file.write_text("MY_TEST_VAR=hello_from_env\n")

    script = tmp_path / "check_env.py"
    script.write_text("""
import os
import openai
# Just verify the env var is set — no actual API call
assert os.environ.get("MY_TEST_VAR") == "hello_from_env", f"Got: {os.environ.get('MY_TEST_VAR')!r}"
""")
    output = tmp_path / "out.jsonl"
    from agentprobe._cli import cmd_record
    args = argparse.Namespace(script=str(script), output=str(output), env=str(env_file))
    cmd_record(args)  # would raise if env not loaded
    # No API calls made, so output may be empty but no exception
    assert True


def test_record_env_missing_file_exits(tmp_path):
    script = tmp_path / "agent.py"
    script.write_text("pass\n")
    output = tmp_path / "out.jsonl"
    from agentprobe._cli import cmd_record
    args = argparse.Namespace(script=str(script), output=str(output), env=str(tmp_path / "missing.env"))
    with pytest.raises(SystemExit) as exc:
        cmd_record(args)
    assert exc.value.code == 1


def test_record_env_does_not_override_existing(tmp_path):
    """--env uses setdefault so it won't overwrite already-set vars."""
    env_file = tmp_path / ".env"
    env_file.write_text("ALREADY_SET=from_env\n")
    os.environ["ALREADY_SET"] = "from_os"

    script = tmp_path / "check.py"
    script.write_text("""
import os
assert os.environ.get("ALREADY_SET") == "from_os"
""")
    output = tmp_path / "out.jsonl"
    from agentprobe._cli import cmd_record
    args = argparse.Namespace(script=str(script), output=str(output), env=str(env_file))
    try:
        cmd_record(args)
    finally:
        os.environ.pop("ALREADY_SET", None)


# ── xdist auto() coordination (file-based test) ───────────────────────────────

def test_auto_with_filelock_records_once(tmp_path):
    """Simulate two sequential auto() calls on the same missing fixture — only records once."""
    fixture = tmp_path / "auto_test.jsonl"
    session = Session()
    mock_resp = openai.types.chat.ChatCompletion.model_validate({
        "id": "chatcmpl-x", "object": "chat.completion", "created": 1748700000,
        "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello!", "tool_calls": None},
                     "finish_reason": "stop", "logprobs": None}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        "system_fingerprint": None,
    })

    call_count = [0]
    original_create = openai.resources.chat.completions.Completions.create

    def counting_create(self, **kwargs):
        call_count[0] += 1
        return mock_resp

    client1 = openai.OpenAI(api_key="dummy")
    with patch.object(openai.resources.chat.completions.Completions, "create", counting_create):
        with session.auto(fixture) as probe:
            client1.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])

    assert fixture.exists()
    recorded_calls = call_count[0]

    # Second auto() call — fixture exists, should replay without making real API calls
    client2 = openai.OpenAI(api_key="dummy")
    with session.auto(fixture) as probe2:
        client2.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])

    assert call_count[0] == recorded_calls  # no extra real calls made


# ── validate linting: missing model field ─────────────────────────────────────

def test_validate_warns_on_missing_model(tmp_path):
    no_model = tmp_path / "no_model.jsonl"
    no_model.write_text(json.dumps({
        "request": {},  # no model
        "response": {
            "id": "x", "object": "chat.completion", "created": 123, "model": "gpt-4o",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi", "tool_calls": None},
                         "finish_reason": "stop", "logprobs": None}],
            "usage": None, "system_fingerprint": None,
        },
        "duration_ms": 100.0,
    }) + "\n")
    r = run_cli("validate", str(no_model))
    assert r.returncode == 0
    assert "WARN" in r.stdout
    assert "model" in r.stdout
