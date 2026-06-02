"""Tests for production-readiness features: validate, init, pytest option, agentprobe_multi fixture."""
import json
import os
import subprocess
import sys
from pathlib import Path

import openai
import pytest

from agentprobe import MultiSession

FIXTURE = "tests/fixtures/bash_session.jsonl"


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "agentprobe._cli", *args],
        capture_output=True, text=True,
    )


# ── agentprobe validate ───────────────────────────────────────────────────────

def test_validate_valid_fixture():
    r = run_cli("validate", FIXTURE)
    assert r.returncode == 0
    assert "OK" in r.stdout


def test_validate_streaming_fixture():
    r = run_cli("validate", "tests/fixtures/streaming_session.jsonl")
    assert r.returncode == 0
    assert "OK" in r.stdout
    assert "streaming" in r.stdout


def test_validate_missing_file():
    r = run_cli("validate", "tests/fixtures/does_not_exist.jsonl")
    assert r.returncode == 1


def test_validate_malformed_json(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("not valid json\n")
    r = run_cli("validate", str(bad))
    assert r.returncode == 1
    assert "ERROR" in r.stderr


def test_validate_missing_response_field(tmp_path):
    bad = tmp_path / "missing_field.jsonl"
    bad.write_text(json.dumps({"request": {"model": "gpt-4o"}}) + "\n")
    r = run_cli("validate", str(bad))
    assert r.returncode == 1
    assert "response" in r.stderr


# ── agentprobe init ───────────────────────────────────────────────────────────

def test_init_creates_fixtures_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = run_cli("init")
    assert r.returncode == 0
    assert (tmp_path / "tests" / "fixtures").is_dir()


def test_init_creates_conftest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = run_cli("init")
    assert (tmp_path / "tests" / "conftest.py").exists()


def test_init_skips_existing_conftest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    conftest = tmp_path / "tests" / "conftest.py"
    conftest.parent.mkdir(parents=True)
    conftest.write_text("# existing\n")
    r = run_cli("init")
    assert r.returncode == 0
    assert "skipping" in r.stdout
    assert conftest.read_text() == "# existing\n"


# ── pytest --agentprobe-update option ────────────────────────────────────────

def test_agentprobe_update_option_sets_env(tmp_path):
    """Running pytest --agentprobe-update should set AGENTPROBE_UPDATE=1."""
    # We test this by running a small pytest session that checks the env var.
    test_file = tmp_path / "test_env_check.py"
    test_file.write_text("""
import os
def test_update_env():
    assert os.environ.get("AGENTPROBE_UPDATE") == "1"
""")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "--agentprobe-update", "-v"],
        capture_output=True, text=True, cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_agentprobe_update_not_set_by_default(tmp_path):
    test_file = tmp_path / "test_no_env.py"
    test_file.write_text("""
import os
def test_no_update_env():
    assert os.environ.get("AGENTPROBE_UPDATE") != "1"
""")
    # Ensure the env var isn't set in our environment
    env = {k: v for k, v in os.environ.items() if k != "AGENTPROBE_UPDATE"}
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-v"],
        capture_output=True, text=True, cwd=str(tmp_path), env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# ── agentprobe_multi pytest fixture ──────────────────────────────────────────

def test_agentprobe_multi_fixture(agentprobe_multi):
    assert isinstance(agentprobe_multi, MultiSession)


def test_agentprobe_multi_fixture_works(agentprobe_multi):
    client = openai.OpenAI(api_key="dummy")
    with agentprobe_multi.replay(client, "tests/fixtures/agent_a.jsonl") as probe:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "What is 2+2?"}],
        )
    assert probe.iteration_count == 1
    assert "4" in resp.choices[0].message.content


# ── agentprobe show [stream] annotation ──────────────────────────────────────

def test_show_annotates_streaming():
    r = run_cli("show", "tests/fixtures/streaming_session.jsonl")
    assert r.returncode == 0
    assert "[stream]" in r.stdout


def test_show_no_stream_annotation_for_non_streaming():
    r = run_cli("show", FIXTURE)
    assert r.returncode == 0
    assert "[stream]" not in r.stdout
