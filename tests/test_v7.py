"""Tests for v0.7.0: assert_response_json_at, received_at, fixture metadata,
record --timeout, show --json meta field."""
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

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


def _resp(content=None, model="gpt-4o"):
    return {
        "id": "chatcmpl-t7", "object": "chat.completion", "created": 1748700000,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content,
                     "tool_calls": None}, "finish_reason": "stop", "logprobs": None}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        "system_fingerprint": None,
    }


# ── probe.assert_response_json_at ─────────────────────────────────────────────

def test_assert_response_json_at_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(
        _resp('{"status": "ok", "count": 3}'),
        _resp("plain text"),
    ) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    probe.assert_response_json_at(0, status="ok", count=3)


def test_assert_response_json_at_wrong_value_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_resp('{"status": "error"}')) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError):
        probe.assert_response_json_at(0, status="ok")


def test_assert_response_json_at_not_json_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_resp("plain text")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError, match="JSON"):
        probe.assert_response_json_at(0, key="value")


def test_assert_response_json_at_out_of_range():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_resp("hi")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(IndexError):
        probe.assert_response_json_at(5, key="value")


# ── probe.received_at ─────────────────────────────────────────────────────────

def test_received_at_returns_message_dict():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_resp("Hello"), _resp("World")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    msg0 = probe.received_at(0)
    assert msg0["content"] == "Hello"
    assert msg0["call_index"] == 0
    assert msg0["finish_reason"] == "stop"


def test_received_at_second_call():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_resp("First"), _resp("Second")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    assert probe.received_at(1)["content"] == "Second"


def test_received_at_out_of_range():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_resp("hi")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(IndexError):
        probe.received_at(99)


def test_received_at_has_tool_calls():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(FIXTURE) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    msg = probe.received_at(0)
    assert msg["tool_calls"] is not None
    assert msg["tool_calls"][0]["function"]["name"] == "bash"


# ── Fixture metadata (_meta header) ───────────────────────────────────────────

def test_save_calls_writes_meta_header(tmp_path):
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    fixture = tmp_path / "meta_test.jsonl"
    with session.inject(_resp("hi")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    probe.dump_fixture(fixture)

    lines = fixture.read_text().strip().split("\n")
    assert len(lines) == 2  # _meta line + 1 call
    meta_line = json.loads(lines[0])
    assert "_meta" in meta_line
    assert "agentprobe_version" in meta_line["_meta"]
    assert "recorded_at" in meta_line["_meta"]
    assert "python_version" in meta_line["_meta"]


def test_load_calls_skips_meta_header(tmp_path):
    """Fixtures with _meta header load correctly with the right call count."""
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    fixture = tmp_path / "with_meta.jsonl"

    with session.inject(_resp("a"), _resp("b")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    probe.dump_fixture(fixture)

    # Replay the fixture — should see 2 calls, not 3 (meta skipped)
    client2 = openai.OpenAI(api_key="dummy")
    with Session().replay(str(fixture)) as probe2:
        r1 = client2.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        r2 = client2.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    assert r1.choices[0].message.content == "a"
    assert r2.choices[0].message.content == "b"
    assert probe2.iteration_count == 2


def test_show_json_includes_meta_field(tmp_path):
    """agentprobe show --json should expose the _meta field in output."""
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    fixture = tmp_path / "meta_show.jsonl"
    with session.inject(_resp("hi")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    probe.dump_fixture(fixture)

    r = run_cli("show", "--json", str(fixture))
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert "meta" in data
    assert data["meta"].get("agentprobe_version") is not None


def test_show_human_includes_meta(tmp_path):
    """agentprobe show should print version/timestamp in the header."""
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    fixture = tmp_path / "meta_human.jsonl"
    with session.inject(_resp("hi")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    probe.dump_fixture(fixture)

    r = run_cli("show", str(fixture))
    assert r.returncode == 0
    assert "recorded" in r.stdout or "agentprobe_version" in r.stdout or "v0." in r.stdout


def test_record_round_trip_preserves_meta(tmp_path):
    """A fixture produced by Session.record should have a _meta header."""
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    fixture = tmp_path / "rt.jsonl"
    source = "tests/fixtures/bash_session.jsonl"

    with Session().replay(source):
        with session.record(str(fixture)):
            client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
            client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])

    lines = fixture.read_text().strip().split("\n")
    first = json.loads(lines[0])
    assert "_meta" in first
    # Remaining lines should be the 2 calls
    assert len([l for l in lines if "_meta" not in json.loads(l)]) == 2


# ── record --timeout ──────────────────────────────────────────────────────────

def test_record_timeout_saves_partial_results(tmp_path):
    """A script that sleeps should be killed after timeout; calls made before are saved."""
    import argparse
    from unittest.mock import patch as mpatch

    script = tmp_path / "slow_agent.py"
    script.write_text("""
import openai, time
client = openai.OpenAI(api_key="dummy")
client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "quick"}])
time.sleep(30)  # will be killed by timeout
client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "never"}])
""")
    output = str(tmp_path / "timeout_test.jsonl")
    mock_resp = openai.types.chat.ChatCompletion.model_validate({
        "id": "chatcmpl-to", "object": "chat.completion", "created": 1748700000, "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "quick response",
                     "tool_calls": None}, "finish_reason": "stop", "logprobs": None}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        "system_fingerprint": None,
    })
    from agentprobe._cli import cmd_record
    args = argparse.Namespace(script=str(script), output=output, env=None,
                              output_format=None, watch=False, interval=1.0, timeout=2.0)
    with mpatch("openai.resources.chat.completions.Completions.create", return_value=mock_resp):
        cmd_record(args)

    assert Path(output).exists()
    lines = [l for l in Path(output).read_text().strip().split("\n") if "_meta" not in l]
    # At least the first call was captured
    assert len(lines) >= 1
    data = json.loads(lines[0])
    assert data["response"]["choices"][0]["message"]["content"] == "quick response"
