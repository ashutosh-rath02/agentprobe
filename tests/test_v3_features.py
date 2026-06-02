"""Tests for v0.3.0 features: AssertionProxy extensions, Session.inject, CLI additions."""
import json
import subprocess
import sys
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


# ── Helper ────────────────────────────────────────────────────────────────────

def _make_response(content="Hello!", model="gpt-4o"):
    return {
        "id": "chatcmpl-inject",
        "object": "chat.completion",
        "created": 1748700000,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content, "tool_calls": None},
            "finish_reason": "stop",
            "logprobs": None,
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "system_fingerprint": None,
    }


# ── Session.inject ────────────────────────────────────────────────────────────

def test_inject_single_response():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_make_response("Injected!")) as probe:
        resp = client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}]
        )
    assert resp.choices[0].message.content == "Injected!"
    assert probe.iteration_count == 1


def test_inject_multiple_responses():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_make_response("First"), _make_response("Second")) as probe:
        r1 = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "1"}])
        r2 = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "2"}])
    assert r1.choices[0].message.content == "First"
    assert r2.choices[0].message.content == "Second"
    assert probe.iteration_count == 2


def test_inject_assertions_work():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_make_response("Hi there!")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    probe.assert_output_contains("Hi there!")
    probe.assert_no_tool_calls()
    probe.assert_stop_reason("stop")


def test_inject_accepts_pydantic_object():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    resp_obj = openai.types.chat.ChatCompletion.model_validate(_make_response("Pydantic!"))
    with session.inject(resp_obj) as probe:
        resp = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    assert resp.choices[0].message.content == "Pydantic!"


# ── Session.inject_error ──────────────────────────────────────────────────────

def test_inject_error_raises():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    error = RuntimeError("Simulated API failure")
    with session.inject_error(error):
        with pytest.raises(RuntimeError, match="Simulated API failure"):
            client.chat.completions.create(
                model="gpt-4o", messages=[{"role": "user", "content": "hi"}]
            )


# ── AssertionProxy: assert_model_used ─────────────────────────────────────────

def test_assert_model_used_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(FIXTURE) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    probe.assert_model_used("gpt-4o")


def test_assert_model_used_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(FIXTURE) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    with pytest.raises(AssertionError, match="gpt-4-turbo"):
        probe.assert_model_used("gpt-4-turbo")


# ── AssertionProxy: assert_no_tool_calls ─────────────────────────────────────

def test_assert_no_tool_calls_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_make_response("No tools here")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    probe.assert_no_tool_calls()


def test_assert_no_tool_calls_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(FIXTURE) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    with pytest.raises(AssertionError, match="bash"):
        probe.assert_no_tool_calls()


# ── AssertionProxy: call_log ──────────────────────────────────────────────────

def test_call_log_has_correct_length():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(FIXTURE) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    assert len(probe.call_log) == 2


def test_call_log_contains_request_and_response():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(FIXTURE) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    for entry in probe.call_log:
        assert "request" in entry
        assert "response" in entry


# ── AssertionProxy: models_used ───────────────────────────────────────────────

def test_models_used():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(FIXTURE) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    assert probe.models_used == ["gpt-4o", "gpt-4o"]


# ── AssertionProxy: total_duration_ms ────────────────────────────────────────

def test_total_duration_ms_is_float():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(FIXTURE) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    # Replay doesn't record real durations, but the fixture has recorded values
    assert isinstance(probe.total_duration_ms, float)
    assert probe.total_duration_ms >= 0


def test_assert_max_duration_ms_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(FIXTURE) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    probe.assert_max_duration_ms(10_000)  # 10 seconds — always passes for fixture calls


def test_assert_max_duration_ms_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_make_response()) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    # inject creates calls with duration_ms=None → 0.0 total; but we test the failure path
    # with a very tight limit on a replay fixture
    session2 = Session()
    client2 = openai.OpenAI(api_key="dummy")
    with session2.replay(FIXTURE) as probe2:
        client2.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client2.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    with pytest.raises(AssertionError, match="duration"):
        probe2.assert_max_duration_ms(1)  # 1ms is too tight


# ── CLI: diff --json ──────────────────────────────────────────────────────────

def test_diff_json_identical():
    r = run_cli("diff", "--json", FIXTURE, FIXTURE)
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert data["identical"] is True
    assert data["differences"] == []


def test_diff_json_different(tmp_path):
    one_call = tmp_path / "one.jsonl"
    lines = Path(FIXTURE).read_text().strip().split("\n")
    one_call.write_text(lines[0] + "\n")
    r = run_cli("diff", "--json", FIXTURE, str(one_call))
    assert r.returncode == 1
    data = json.loads(r.stdout)
    assert data["identical"] is False
    assert any(d["type"] == "call_count" for d in data["differences"])


# ── CLI: show --json chunk_count ─────────────────────────────────────────────

def test_show_json_chunk_count_present_for_streaming():
    r = run_cli("show", "--json", "tests/fixtures/streaming_session.jsonl")
    data = json.loads(r.stdout)
    assert all(entry["chunk_count"] is not None for entry in data)
    assert all(entry["chunk_count"] > 0 for entry in data)


def test_show_json_chunk_count_null_for_non_streaming():
    r = run_cli("show", "--json", FIXTURE)
    data = json.loads(r.stdout)
    assert all(entry["chunk_count"] is None for entry in data)


# ── CLI: fixtures list ────────────────────────────────────────────────────────

def test_fixtures_list_default_dir():
    r = run_cli("fixtures", "tests/fixtures")
    assert r.returncode == 0
    assert "bash_session.jsonl" in r.stdout


def test_fixtures_list_json():
    r = run_cli("fixtures", "--json", "tests/fixtures")
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert isinstance(data, list)
    paths = [e["path"] for e in data]
    assert any("bash_session.jsonl" in p for p in paths)


def test_fixtures_list_json_includes_streaming():
    r = run_cli("fixtures", "--json", "tests/fixtures")
    data = json.loads(r.stdout)
    streaming = next((e for e in data if "streaming_session" in e["path"]), None)
    assert streaming is not None
    assert streaming["streaming_calls"] == 2


def test_fixtures_list_missing_dir():
    r = run_cli("fixtures", "does_not_exist/")
    assert r.returncode == 1
