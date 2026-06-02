"""Tests for v0.4 batch 1: call(n), first/last_tool_called, JSON output assertions,
Pydantic validate, diff --json content diffs."""
import json
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


def _make_resp(content=None, tool_calls=None, finish="stop", model="gpt-4o"):
    return {
        "id": "chatcmpl-t",
        "object": "chat.completion",
        "created": 1748700000,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content, "tool_calls": tool_calls},
            "finish_reason": finish,
            "logprobs": None,
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "system_fingerprint": None,
    }


# ── probe.call(n) ─────────────────────────────────────────────────────────────

def test_call_n_scopes_to_single_call():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(FIXTURE) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])

    # Call 0 should see the tool_use call; call 1 sees end_turn
    probe.call(0).assert_tool_called("bash")
    probe.call(1).assert_stop_reason("stop")
    probe.call(1).assert_not_tool_called("bash")


def test_call_n_index_error():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_make_resp("hi")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(IndexError, match="out of range"):
        probe.call(5)


def test_call_negative_index_error():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_make_resp("hi")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(IndexError):
        probe.call(-1)


# ── first_tool_called / last_tool_called ──────────────────────────────────────

def test_first_tool_called():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(FIXTURE) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    assert probe.first_tool_called == "bash"


def test_last_tool_called():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(FIXTURE) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    assert probe.last_tool_called == "bash"


def test_first_tool_called_none_when_no_tools():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_make_resp("hi")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    assert probe.first_tool_called is None
    assert probe.last_tool_called is None


# ── assert_output_is_json / assert_output_json_contains ──────────────────────

def test_assert_output_is_json_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_make_resp('{"status": "ok", "count": 3}')) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    probe.assert_output_is_json()


def test_assert_output_is_json_fails_on_plain_text():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_make_resp("This is plain text, not JSON")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError, match="valid JSON"):
        probe.assert_output_is_json()


def test_assert_output_json_contains_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_make_resp('{"status": "ok", "count": 3}')) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    probe.assert_output_json_contains(status="ok", count=3)


def test_assert_output_json_contains_wrong_value():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_make_resp('{"status": "ok"}')) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError, match="status"):
        probe.assert_output_json_contains(status="error")


def test_assert_output_json_contains_missing_key():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_make_resp('{"status": "ok"}')) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError, match="missing_key"):
        probe.assert_output_json_contains(missing_key="x")


# ── agentprobe validate with Pydantic ────────────────────────────────────────

def test_validate_pydantic_passes_on_valid_fixture():
    r = run_cli("validate", FIXTURE)
    assert r.returncode == 0


def test_validate_pydantic_passes_on_streaming_fixture():
    r = run_cli("validate", "tests/fixtures/streaming_session.jsonl")
    assert r.returncode == 0


def test_validate_pydantic_fails_on_bad_response(tmp_path):
    bad = tmp_path / "bad.jsonl"
    # Write a response where 'choices' has wrong structure for Pydantic
    bad.write_text(json.dumps({
        "request": {"model": "gpt-4o"},
        "response": {
            "id": "x",
            "object": "chat.completion",
            "created": 123,
            "model": "gpt-4o",
            "choices": "not-a-list",  # wrong type
        },
    }) + "\n")
    r = run_cli("validate", str(bad))
    assert r.returncode == 1
    assert "ERROR" in r.stderr


# ── diff --json content comparison ────────────────────────────────────────────

def test_diff_json_detects_content_change(tmp_path):
    # Create a second fixture with different text content
    orig_lines = Path(FIXTURE).read_text().strip().split("\n")
    modified_line = json.loads(orig_lines[1])
    modified_line["response"]["choices"][0]["message"]["content"] = "Different content!"
    modified = tmp_path / "modified.jsonl"
    modified.write_text(orig_lines[0] + "\n" + json.dumps(modified_line) + "\n")

    r = run_cli("diff", "--json", FIXTURE, str(modified))
    assert r.returncode == 1
    data = json.loads(r.stdout)
    assert any(d["type"] == "content" for d in data["differences"])


def test_diff_json_detects_tool_arg_change(tmp_path):
    orig_lines = Path(FIXTURE).read_text().strip().split("\n")
    modified_line = json.loads(orig_lines[0])
    tc = modified_line["response"]["choices"][0]["message"]["tool_calls"][0]
    tc["function"]["arguments"] = '{"command": "ls /different"}'
    modified = tmp_path / "mod_args.jsonl"
    modified.write_text(json.dumps(modified_line) + "\n" + orig_lines[1] + "\n")

    r = run_cli("diff", "--json", FIXTURE, str(modified))
    assert r.returncode == 1
    data = json.loads(r.stdout)
    assert any(d["type"] == "tool_arguments" for d in data["differences"])
