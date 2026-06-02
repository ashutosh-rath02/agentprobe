"""Tests for MultiSession — per-client record/replay for multi-agent scenarios."""
import json
import pytest
import openai

from agentprobe import MultiSession, Session

FIXTURE_A = "tests/fixtures/agent_a.jsonl"
FIXTURE_B = "tests/fixtures/agent_b.jsonl"
BASH_FIXTURE = "tests/fixtures/bash_session.jsonl"


# ── Isolation: two clients don't share a call sequence ───────────────────────

def test_multi_replay_isolates_clients():
    """Each client replays its own independent call sequence."""
    multi = MultiSession()
    client_a = openai.OpenAI(api_key="dummy")
    client_b = openai.OpenAI(api_key="dummy")

    with multi.replay(client_a, FIXTURE_A) as probe_a:
        with multi.replay(client_b, FIXTURE_B) as probe_b:
            resp_a = client_a.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "What is 2+2?"}],
            )
            resp_b = client_b.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "What is 3+3?"}],
            )

    assert "4" in resp_a.choices[0].message.content
    assert "6" in resp_b.choices[0].message.content
    assert probe_a.iteration_count == 1
    assert probe_b.iteration_count == 1


def test_multi_vs_class_level_interference():
    """Demonstrate that nested Session (class-level) would share a sequence,
    while MultiSession isolates per client — both fixtures used here are 1-call,
    so nested Session would exhaust on the second call if not isolated."""
    multi = MultiSession()
    client_a = openai.OpenAI(api_key="dummy")
    client_b = openai.OpenAI(api_key="dummy")

    # Both A and B use their own 1-call fixtures simultaneously — no exhaustion.
    with multi.replay(client_a, FIXTURE_A):
        with multi.replay(client_b, FIXTURE_B):
            client_a.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "What is 2+2?"}],
            )
            client_b.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "What is 3+3?"}],
            )


def test_multi_replay_assertions_are_per_client():
    multi = MultiSession()
    client_a = openai.OpenAI(api_key="dummy")
    client_b = openai.OpenAI(api_key="dummy")

    with multi.replay(client_a, FIXTURE_A) as probe_a:
        with multi.replay(client_b, FIXTURE_B) as probe_b:
            client_a.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
            client_b.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])

    probe_a.assert_output_contains("4")
    probe_b.assert_output_contains("6")
    probe_a.assert_max_iterations(1)
    probe_b.assert_max_iterations(1)


# ── Record round-trip ─────────────────────────────────────────────────────────

def test_multi_record_round_trip(tmp_path):
    fixture_a = tmp_path / "a.jsonl"
    fixture_b = tmp_path / "b.jsonl"
    multi = MultiSession()
    client_a = openai.OpenAI(api_key="dummy")
    client_b = openai.OpenAI(api_key="dummy")

    # Record: use nested replay as the "real API" source
    with multi.replay(client_a, FIXTURE_A):
        with multi.replay(client_b, FIXTURE_B):
            with multi.record(client_a, fixture_a):
                with multi.record(client_b, fixture_b):
                    client_a.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
                    client_b.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])

    assert fixture_a.exists() and fixture_b.exists()
    data_a = json.loads(fixture_a.read_text().strip())
    data_b = json.loads(fixture_b.read_text().strip())
    assert data_a["response"]["choices"][0]["message"]["content"] == "2+2 equals 4."
    assert data_b["response"]["choices"][0]["message"]["content"] == "3+3 equals 6."


# ── Auto mode ────────────────────────────────────────────────────────────────

def test_multi_auto_replays_when_exists():
    multi = MultiSession()
    client_a = openai.OpenAI(api_key="dummy")

    with multi.auto(client_a, FIXTURE_A) as probe:
        resp = client_a.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])

    assert "4" in resp.choices[0].message.content


def test_multi_exhaustion_raises():
    multi = MultiSession()
    client_a = openai.OpenAI(api_key="dummy")

    with pytest.raises(RuntimeError, match="replay exhausted"):
        with multi.replay(client_a, FIXTURE_A):
            # Fixture A has 1 call; making 2 should raise
            client_a.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "1"}])
            client_a.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "2"}])


# ── Async multi-agent ─────────────────────────────────────────────────────────

async def test_async_multi_replay_isolates_clients():
    multi = MultiSession()
    client_a = openai.AsyncOpenAI(api_key="dummy")
    client_b = openai.AsyncOpenAI(api_key="dummy")

    async with multi.async_replay(client_a, FIXTURE_A) as probe_a:
        async with multi.async_replay(client_b, FIXTURE_B) as probe_b:
            resp_a = await client_a.chat.completions.create(
                model="gpt-4o", messages=[{"role": "user", "content": "x"}]
            )
            resp_b = await client_b.chat.completions.create(
                model="gpt-4o", messages=[{"role": "user", "content": "y"}]
            )

    assert "4" in resp_a.choices[0].message.content
    assert "6" in resp_b.choices[0].message.content


async def test_async_multi_record_round_trip(tmp_path):
    fixture_a = tmp_path / "async_a.jsonl"
    fixture_b = tmp_path / "async_b.jsonl"
    multi = MultiSession()
    client_a = openai.AsyncOpenAI(api_key="dummy")
    client_b = openai.AsyncOpenAI(api_key="dummy")

    async with multi.async_replay(client_a, FIXTURE_A):
        async with multi.async_replay(client_b, FIXTURE_B):
            async with multi.async_record(client_a, fixture_a):
                async with multi.async_record(client_b, fixture_b):
                    await client_a.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
                    await client_b.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])

    assert fixture_a.exists() and fixture_b.exists()
