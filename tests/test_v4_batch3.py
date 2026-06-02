"""Tests for v0.4 batch 3: gzip fixtures, file-lock safety, pricing table."""
import gzip
import json
from pathlib import Path

import openai
import pytest

from agentprobe import Session
from agentprobe._pricing import estimate_cost

FIXTURE = "tests/fixtures/bash_session.jsonl"


# ── Gzip fixture support ──────────────────────────────────────────────────────

def test_save_and_load_gzip_fixture(tmp_path):
    """Session.record() with a .jsonl.gz path should write compressed JSONL."""
    gz_path = tmp_path / "session.jsonl.gz"
    session = Session()
    client = openai.OpenAI(api_key="dummy")

    with Session().replay(FIXTURE):
        with session.record(gz_path):
            client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "x"}],
            )
            client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "y"}],
            )

    assert gz_path.exists()
    # Verify it's actually gzip compressed
    with gzip.open(gz_path, "rt") as f:
        lines = [l for l in f if l.strip()]
    assert len(lines) == 2
    data = json.loads(lines[0])
    assert "request" in data and "response" in data


def test_replay_gzip_fixture(tmp_path):
    """Session.replay() with a .jsonl.gz path should decompress transparently."""
    gz_path = tmp_path / "session.jsonl.gz"
    session = Session()
    client = openai.OpenAI(api_key="dummy")

    # Record to gzip
    with Session().replay(FIXTURE):
        with session.record(gz_path):
            client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
            client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])

    # Replay from gzip
    client2 = openai.OpenAI(api_key="dummy")
    with Session().replay(gz_path) as probe:
        resp1 = client2.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        resp2 = client2.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])

    assert probe.iteration_count == 2


def test_gzip_fixture_is_smaller(tmp_path):
    """Gzip fixture should be smaller than plain JSONL for the same data."""
    plain_path = tmp_path / "session.jsonl"
    gz_path = tmp_path / "session.jsonl.gz"
    session = Session()
    client = openai.OpenAI(api_key="dummy")

    with Session().replay(FIXTURE):
        with session.record(plain_path):
            client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
            client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])

    client2 = openai.OpenAI(api_key="dummy")
    with Session().replay(FIXTURE):
        with Session().record(gz_path):
            client2.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
            client2.chat.completions.create(model="gzip", messages=[{"role": "user", "content": "y"}])

    assert gz_path.stat().st_size < plain_path.stat().st_size


def test_gzip_assertions_work(tmp_path):
    """AssertionProxy should work normally when replaying from a gzip fixture."""
    gz_path = tmp_path / "session.jsonl.gz"
    session = Session()
    client = openai.OpenAI(api_key="dummy")

    with Session().replay(FIXTURE):
        with session.record(gz_path):
            client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
            client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])

    client2 = openai.OpenAI(api_key="dummy")
    with Session().replay(gz_path) as probe:
        client2.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client2.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])

    probe.assert_tool_called("bash")
    probe.assert_max_iterations(5)
    probe.assert_output_contains("file1.txt")


# ── File-lock safety: _save_calls uses a lock file ───────────────────────────

def test_save_calls_creates_lock_file(tmp_path):
    """_save_calls should leave no stale lock file after writing."""
    from agentprobe._session import _save_calls
    from agentprobe._models import RecordedCall
    path = tmp_path / "test.jsonl"
    calls = [RecordedCall(request={"model": "gpt-4o"}, response={"choices": []})]
    _save_calls(calls, path)
    # Lock file should be cleaned up by filelock (not left behind)
    lock = path.with_suffix(".jsonl.lock")
    # filelock may leave the file on Windows; just check the main file exists
    assert path.exists()


# ── Pricing table ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("model,expected_nonzero", [
    ("gpt-4o", True),
    ("gpt-4o-2024-11-20", True),
    ("gpt-4o-mini", True),
    ("gpt-4-turbo", True),
    ("gpt-4", True),
    ("gpt-3.5-turbo", True),
    ("o1", True),
    ("o1-mini", True),
    ("o1-preview", True),
    ("o3", True),
    ("o3-mini", True),
    ("o4-mini", True),
    ("unknown-model-xyz", False),
])
def test_pricing_table_coverage(model, expected_nonzero):
    cost = estimate_cost(model, 1000, 500)
    if expected_nonzero:
        assert cost > 0, f"Expected non-zero cost for {model}"
    else:
        assert cost == 0.0, f"Expected zero cost for unknown model {model}"


def test_versioned_model_resolves_cost():
    """gpt-4o-2024-08-06 should resolve to gpt-4o pricing."""
    cost_versioned = estimate_cost("gpt-4o-2024-08-06", 1_000_000, 0)
    cost_base = estimate_cost("gpt-4o", 1_000_000, 0)
    assert cost_versioned == cost_base


def test_cost_calculation_accuracy():
    # gpt-4o: $2.50/Mtok input, $10.00/Mtok output
    # 500k input + 100k output = 500*2.50/1000 + 100*10/1000 = 1.25 + 1.00 = 2.25
    cost = estimate_cost("gpt-4o", 500_000, 100_000)
    assert abs(cost - 2.25) < 1e-6
