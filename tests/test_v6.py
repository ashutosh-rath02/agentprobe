"""Tests for v0.6.0: messages_received, tool_call_inputs, summary_dict,
assert_token_efficiency, replay_chain, migrate CLI, fixtures stats,
show --model, record --output-format gz, record --watch."""
import json
import subprocess
import sys
import time
from pathlib import Path

import openai
import pytest

from agentprobe import Session

FIXTURE = "tests/fixtures/bash_session.jsonl"
AGENT_A = "tests/fixtures/agent_a.jsonl"
AGENT_B = "tests/fixtures/agent_b.jsonl"


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "agentprobe._cli", *args],
        capture_output=True, text=True,
    )


def _resp(content="Hi!", model="gpt-4o", finish="stop"):
    return {
        "id": "chatcmpl-t", "object": "chat.completion", "created": 1748700000,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content,
                     "tool_calls": None}, "finish_reason": finish, "logprobs": None}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "system_fingerprint": None,
    }


# ── probe.messages_received ───────────────────────────────────────────────────

def test_messages_received_has_one_entry_per_call():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_resp("First"), _resp("Second")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    assert len(probe.messages_received) == 2
    assert probe.messages_received[0]["content"] == "First"
    assert probe.messages_received[1]["content"] == "Second"


def test_messages_received_has_call_index():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(FIXTURE) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    assert probe.messages_received[0]["call_index"] == 0
    assert probe.messages_received[1]["call_index"] == 1


def test_messages_received_tool_calls_present():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(FIXTURE) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    # First call has tool_calls; second has content
    assert probe.messages_received[0]["tool_calls"] is not None
    assert probe.messages_received[1]["content"] is not None


# ── probe.tool_call_inputs ────────────────────────────────────────────────────

def test_tool_call_inputs_returns_dict():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(FIXTURE) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    inputs = probe.tool_call_inputs
    assert "bash" in inputs
    assert inputs["bash"][0]["command"] == "ls /tmp"


def test_tool_call_inputs_empty_when_no_tools():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_resp("hi")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    assert probe.tool_call_inputs == {}


# ── probe.summary_dict ────────────────────────────────────────────────────────

def test_summary_dict_has_expected_keys():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_resp("hi")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    s = probe.summary_dict()
    for key in ["iteration_count", "tools_called", "final_output", "total_tokens",
                "estimated_cost_usd", "total_duration_ms", "models_used"]:
        assert key in s, f"Missing key: {key}"


def test_summary_dict_values_match_properties():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(FIXTURE) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    s = probe.summary_dict()
    assert s["iteration_count"] == probe.iteration_count
    assert s["tools_called"] == probe.tools_called
    assert s["total_tokens"] == probe.total_tokens


# ── probe.assert_token_efficiency ────────────────────────────────────────────

def test_assert_token_efficiency_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(FIXTURE) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    # bash_session: 130 in, 55 out → ratio ≈ 0.423
    probe.assert_token_efficiency(0.1)


def test_assert_token_efficiency_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(FIXTURE) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    with pytest.raises(AssertionError, match="efficiency"):
        probe.assert_token_efficiency(10.0)  # 10× output vs input — impossible


# ── Session.replay_chain ──────────────────────────────────────────────────────

def test_replay_chain_concatenates_fixtures():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay_chain(AGENT_A, AGENT_B) as probe:
        resp_a = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        resp_b = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    assert "4" in resp_a.choices[0].message.content  # from agent_a
    assert "6" in resp_b.choices[0].message.content  # from agent_b
    assert probe.iteration_count == 2


def test_replay_chain_assertions_work():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay_chain(AGENT_A, AGENT_B) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    probe.assert_max_iterations(5)
    probe.assert_output_contains("6")  # last output from agent_b


def test_replay_chain_exhaustion_raises():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with pytest.raises(RuntimeError, match="replay exhausted"):
        with session.replay_chain(AGENT_A):
            client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
            client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])


# ── agentprobe migrate ────────────────────────────────────────────────────────

def test_migrate_rename_model(tmp_path):
    output = tmp_path / "migrated.jsonl"
    r = run_cli("migrate", FIXTURE, str(output), "--rename-model", "gpt-4o=gpt-4o-mini")
    assert r.returncode == 0
    data = json.loads(output.read_text().strip().split("\n")[0])
    assert data["request"]["model"] == "gpt-4o-mini"
    assert data["response"]["model"] == "gpt-4o-mini"


def test_migrate_set_model(tmp_path):
    output = tmp_path / "set_model.jsonl"
    r = run_cli("migrate", FIXTURE, str(output), "--set-model", "gpt-4-turbo")
    assert r.returncode == 0
    for line in output.read_text().strip().split("\n"):
        data = json.loads(line)
        assert data["request"]["model"] == "gpt-4-turbo"


def test_migrate_rename_tool(tmp_path):
    output = tmp_path / "renamed_tool.jsonl"
    r = run_cli("migrate", FIXTURE, str(output), "--rename-tool", "bash=shell")
    assert r.returncode == 0
    data = json.loads(output.read_text().strip().split("\n")[0])
    tc = data["response"]["choices"][0]["message"]["tool_calls"][0]
    assert tc["function"]["name"] == "shell"


def test_migrate_missing_input_fails():
    r = run_cli("migrate", "nonexistent.jsonl", "out.jsonl")
    assert r.returncode == 1


def test_migrate_output_is_replayable(tmp_path):
    output = tmp_path / "migrated.jsonl"
    run_cli("migrate", FIXTURE, str(output), "--rename-model", "gpt-4o=gpt-4o")
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(str(output)) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    assert probe.iteration_count == 2


# ── agentprobe stats ──────────────────────────────────────────────────────────

def test_stats_exits_zero():
    r = run_cli("stats", "tests/fixtures")
    assert r.returncode == 0


def test_stats_json_has_expected_fields():
    r = run_cli("stats", "--json", "tests/fixtures")
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert "total_calls" in data
    assert "total_tokens" in data
    assert "estimated_total_cost_usd" in data
    assert "by_model" in data


def test_stats_json_total_calls_positive():
    r = run_cli("stats", "--json", "tests/fixtures")
    data = json.loads(r.stdout)
    assert data["total_calls"] > 0


def test_stats_missing_dir():
    r = run_cli("stats", "does_not_exist_dir/")
    assert r.returncode == 1


# ── agentprobe show --model ───────────────────────────────────────────────────

def test_show_model_filter_matching():
    r = run_cli("show", "--model", "gpt-4o", FIXTURE)
    assert r.returncode == 0
    assert "2 call(s)" in r.stdout


def test_show_model_filter_no_match():
    r = run_cli("show", "--model", "gpt-4-turbo", FIXTURE)
    assert r.returncode == 0
    assert "0 call(s)" in r.stdout


def test_show_json_model_filter():
    r = run_cli("show", "--json", "--model", "gpt-4o", FIXTURE)
    data = json.loads(r.stdout)
    assert data["summary"]["total_calls"] == 2


def test_show_json_model_filter_no_match():
    r = run_cli("show", "--json", "--model", "unknown-model", FIXTURE)
    data = json.loads(r.stdout)
    assert data["summary"]["total_calls"] == 0


# ── agentprobe record --output-format gz ─────────────────────────────────────

def test_record_output_format_gz(tmp_path):
    import argparse
    import openai.types.chat
    from unittest.mock import patch as mpatch

    script = tmp_path / "agent.py"
    script.write_text("""
import openai
client = openai.OpenAI(api_key="dummy")
client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
""")
    output = str(tmp_path / "session.jsonl")
    mock_resp = openai.types.chat.ChatCompletion.model_validate({
        "id": "chatcmpl-gz", "object": "chat.completion", "created": 1748700000, "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi", "tool_calls": None},
                     "finish_reason": "stop", "logprobs": None}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        "system_fingerprint": None,
    })
    from agentprobe._cli import cmd_record
    args = argparse.Namespace(script=str(script), output=output, env=None,
                              output_format="gz", watch=False, interval=1.0)
    with mpatch("openai.resources.chat.completions.Completions.create", return_value=mock_resp):
        cmd_record(args)

    gz_path = Path(output + ".gz")
    assert gz_path.exists()

    import gzip
    with gzip.open(gz_path, "rt") as f:
        lines = [l for l in f if l.strip() and '"_meta"' not in l]
    data = json.loads(lines[0])
    assert data["request"]["model"] == "gpt-4o"
