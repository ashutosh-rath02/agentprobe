"""Tests for the agentprobe CLI (show / diff)."""
import subprocess
import sys
import json
from pathlib import Path

PYTHON = sys.executable
FIXTURE = "tests/fixtures/bash_session.jsonl"


def run_cli(*args):
    result = subprocess.run(
        [PYTHON, "-m", "agentprobe._cli", *args],
        capture_output=True, text=True,
    )
    return result


def test_show_exits_zero():
    r = run_cli("show", FIXTURE)
    assert r.returncode == 0


def test_show_output_contains_call_count():
    r = run_cli("show", FIXTURE)
    assert "2 call(s)" in r.stdout


def test_show_output_contains_tool_name():
    r = run_cli("show", FIXTURE)
    assert "bash" in r.stdout


def test_show_output_contains_token_summary():
    r = run_cli("show", FIXTURE)
    assert "total tokens" in r.stdout


def test_diff_identical_exits_zero():
    r = run_cli("diff", FIXTURE, FIXTURE)
    assert r.returncode == 0
    assert "no differences" in r.stdout


def test_diff_different_fixtures_exits_nonzero(tmp_path):
    # Build a one-call fixture
    one_call = tmp_path / "one_call.jsonl"
    lines = Path(FIXTURE).read_text().strip().split("\n")
    one_call.write_text(lines[0] + "\n")

    r = run_cli("diff", FIXTURE, str(one_call))
    assert r.returncode == 1
    assert "differ" in r.stdout or "count" in r.stdout


def test_show_missing_file_exits_nonzero():
    r = run_cli("show", "tests/fixtures/does_not_exist.jsonl")
    assert r.returncode == 1
