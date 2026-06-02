"""Tests for v0.8.0 features: new AssertionProxy methods, Session.record_append,
MultiSession.replay_chain, CLI validate --strict, diff --by-call, fixtures --clean,
stats --by-date, record --dry-run, record --append."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import openai
import openai.types.chat
import pytest

from agentprobe import MultiSession, Session

FIXTURE = "tests/fixtures/bash_session.jsonl"
AGENT_A = "tests/fixtures/agent_a.jsonl"
AGENT_B = "tests/fixtures/agent_b.jsonl"


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "agentprobe._cli", *args],
        capture_output=True, text=True,
    )


def _resp(content="Hi", model="gpt-4o", finish="stop"):
    return {
        "id": "chatcmpl-v8", "object": "chat.completion", "created": 1748700000,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content,
                     "tool_calls": None}, "finish_reason": finish, "logprobs": None}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "system_fingerprint": None,
    }


# ── probe.assert_cost_per_call ────────────────────────────────────────────────

def test_assert_cost_per_call_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_resp()) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    probe.assert_cost_per_call(1.0)  # 10 prompt + 5 completion tokens @ $2.50/$10 per mtok = tiny


def test_assert_cost_per_call_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_resp()) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError, match="cost"):
        probe.assert_cost_per_call(0.0)  # any cost > $0 should fail


# ── probe.assert_messages_count ───────────────────────────────────────────────

def test_assert_messages_count_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_resp(), _resp()) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[
            {"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}
        ])
        client.chat.completions.create(model="gpt-4o", messages=[
            {"role": "user", "content": "c"}
        ])
    probe.assert_messages_count(3)  # 2 + 1


def test_assert_messages_count_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_resp()) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError, match="messages"):
        probe.assert_messages_count(99)


# ── probe.assert_all_models_in ────────────────────────────────────────────────

def test_assert_all_models_in_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(FIXTURE) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    probe.assert_all_models_in("gpt-4o", "gpt-4o-mini")


def test_assert_all_models_in_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(FIXTURE) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    with pytest.raises(AssertionError, match="disallowed"):
        probe.assert_all_models_in("gpt-4-turbo")


# ── probe.assert_no_empty_responses ──────────────────────────────────────────

def test_assert_no_empty_responses_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_resp("Hello!")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    probe.assert_no_empty_responses()


def test_assert_no_empty_responses_with_tool_calls():
    """Tool-call responses (no content) should also pass."""
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(FIXTURE) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    probe.assert_no_empty_responses()


# ── probe.export_json ─────────────────────────────────────────────────────────

def test_export_json_is_valid_json():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_resp("Hello!")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    j = probe.export_json()
    data = json.loads(j)
    assert "summary" in data
    assert "calls" in data
    assert data["summary"]["iteration_count"] == 1


def test_export_json_roundtrip():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(FIXTURE) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    data = json.loads(probe.export_json())
    assert data["summary"]["total_tokens"] == 185
    assert len(data["calls"]) == 2


# ── probe.matches_fixture ─────────────────────────────────────────────────────

def test_matches_fixture_identical():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(FIXTURE) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    probe.matches_fixture(FIXTURE)


def test_matches_fixture_count_mismatch():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(AGENT_A) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError, match="mismatch"):
        probe.matches_fixture(FIXTURE)  # FIXTURE has 2 calls; probe has 1


def test_matches_fixture_ignore_content():
    """ignore_content=True should pass even if text differs."""
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(FIXTURE) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    probe.matches_fixture(FIXTURE, ignore_content=True)


# ── Session.record_append ─────────────────────────────────────────────────────

def test_record_append_adds_to_existing(tmp_path):
    fixture = tmp_path / "appended.jsonl"
    session = Session()
    client = openai.OpenAI(api_key="dummy")

    # First pass: 1 call from AGENT_A
    with session.replay(AGENT_A):
        with session.record_append(fixture):
            client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])

    # Second pass: 1 call from AGENT_B
    with session.replay(AGENT_B):
        with session.record_append(fixture):
            client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])

    from conftest import fixture_lines
    lines = fixture_lines(fixture)
    assert len(lines) == 2  # both calls preserved


def test_record_append_is_replayable(tmp_path):
    fixture = tmp_path / "two_calls.jsonl"
    session = Session()
    client = openai.OpenAI(api_key="dummy")

    with session.replay(AGENT_A):
        with session.record_append(fixture):
            client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    with session.replay(AGENT_B):
        with session.record_append(fixture):
            client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])

    client2 = openai.OpenAI(api_key="dummy")
    with Session().replay(str(fixture)) as probe:
        r1 = client2.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        r2 = client2.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    assert "4" in r1.choices[0].message.content
    assert "6" in r2.choices[0].message.content


def test_record_append_creates_file_if_missing(tmp_path):
    fixture = tmp_path / "new_file.jsonl"
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(AGENT_A):
        with session.record_append(fixture):
            client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    assert fixture.exists()
    from conftest import fixture_lines
    assert len(fixture_lines(fixture)) == 1


# ── MultiSession.replay_chain ─────────────────────────────────────────────────

def test_multi_replay_chain_two_clients():
    multi = MultiSession()
    client_a = openai.OpenAI(api_key="dummy")
    client_b = openai.OpenAI(api_key="dummy")

    with multi.replay_chain(
        (client_a, AGENT_A),
        (client_b, AGENT_B),
    ) as probes:
        resp_a = client_a.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        resp_b = client_b.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])

    assert "4" in resp_a.choices[0].message.content
    assert "6" in resp_b.choices[0].message.content
    assert probes[client_a].iteration_count == 1
    assert probes[client_b].iteration_count == 1


def test_multi_replay_chain_list_paths():
    multi = MultiSession()
    client_a = openai.OpenAI(api_key="dummy")

    with multi.replay_chain((client_a, [AGENT_A, AGENT_B])) as probes:
        r1 = client_a.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        r2 = client_a.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])

    assert "4" in r1.choices[0].message.content
    assert "6" in r2.choices[0].message.content
    assert probes[client_a].iteration_count == 2


# ── CLI validate --strict ─────────────────────────────────────────────────────

def test_validate_strict_fails_on_warnings(tmp_path):
    """A fixture with no duration_ms should pass normally but fail with --strict."""
    no_dur = tmp_path / "no_dur.jsonl"
    no_dur.write_text(json.dumps({
        "request": {"model": "gpt-4o"},
        "response": {
            "id": "x", "object": "chat.completion", "created": 123, "model": "gpt-4o",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi",
                         "tool_calls": None}, "finish_reason": "stop", "logprobs": None}],
            "usage": None, "system_fingerprint": None,
        },
    }) + "\n")
    r_normal = run_cli("validate", str(no_dur))
    assert r_normal.returncode == 0  # no --strict: warnings don't fail

    r_strict = run_cli("validate", "--strict", str(no_dur))
    assert r_strict.returncode == 1  # --strict: warning is an error


# ── CLI diff --by-call ────────────────────────────────────────────────────────

def test_diff_by_call_identical():
    r = run_cli("diff", "--by-call", FIXTURE, FIXTURE)
    assert r.returncode == 0
    assert "No differences" in r.stdout


def test_diff_by_call_different(tmp_path):
    one = tmp_path / "one.jsonl"
    lines = Path(FIXTURE).read_text().strip().split("\n")
    one.write_text(lines[0] + "\n")
    r = run_cli("diff", "--by-call", FIXTURE, str(one))
    assert r.returncode == 1


# ── CLI fixtures --clean ──────────────────────────────────────────────────────

def test_fixtures_clean_removes_lock_files(tmp_path):
    (tmp_path / "a.jsonl.lock").write_text("")
    (tmp_path / "b.jsonl.lock").write_text("")
    r = run_cli("fixtures", "--clean", str(tmp_path))
    assert r.returncode == 0
    assert not list(tmp_path.glob("*.lock"))
    assert "2" in r.stdout


def test_fixtures_clean_no_locks(tmp_path):
    r = run_cli("fixtures", "--clean", str(tmp_path))
    assert r.returncode == 0
    assert "no .lock" in r.stdout


# ── CLI stats --by-date ───────────────────────────────────────────────────────

def test_stats_by_date_exits_zero():
    r = run_cli("stats", "--by-date", "tests/fixtures")
    assert r.returncode == 0


def test_stats_by_date_json():
    r = run_cli("stats", "--by-date", "--json", "tests/fixtures")
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert isinstance(data, dict)


# ── CLI record --dry-run ──────────────────────────────────────────────────────

def test_record_dry_run_does_not_save(tmp_path):
    script = tmp_path / "agent.py"
    script.write_text("""
import openai
client = openai.OpenAI(api_key="dummy")
client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
""")
    output = str(tmp_path / "should_not_exist.jsonl")
    mock_resp = openai.types.chat.ChatCompletion.model_validate({
        "id": "chatcmpl-dr", "object": "chat.completion", "created": 1748700000, "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok", "tool_calls": None},
                     "finish_reason": "stop", "logprobs": None}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        "system_fingerprint": None,
    })
    from agentprobe._cli import cmd_record
    args = argparse.Namespace(script=str(script), output=output, env=None,
                              output_format=None, watch=False, interval=1.0,
                              timeout=None, dry_run=True, append=False)
    with patch("openai.resources.chat.completions.Completions.create", return_value=mock_resp):
        cmd_record(args)

    assert not Path(output).exists(), "dry-run should not write the fixture"


# ── CLI record --append ───────────────────────────────────────────────────────

def test_record_append_via_cli(tmp_path):
    script = tmp_path / "agent.py"
    script.write_text("""
import openai
client = openai.OpenAI(api_key="dummy")
client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
""")
    output = str(tmp_path / "appended.jsonl")
    mock_resp = openai.types.chat.ChatCompletion.model_validate({
        "id": "chatcmpl-app", "object": "chat.completion", "created": 1748700000, "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok", "tool_calls": None},
                     "finish_reason": "stop", "logprobs": None}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        "system_fingerprint": None,
    })
    from agentprobe._cli import cmd_record
    base_args = argparse.Namespace(script=str(script), output=output, env=None,
                                   output_format=None, watch=False, interval=1.0,
                                   timeout=None, dry_run=False, append=True)

    with patch("openai.resources.chat.completions.Completions.create", return_value=mock_resp):
        cmd_record(base_args)
        cmd_record(base_args)  # second pass — should append

    from conftest import fixture_lines
    lines = fixture_lines(Path(output))
    assert len(lines) == 2  # both calls
