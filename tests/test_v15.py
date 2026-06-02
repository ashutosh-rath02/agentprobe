"""Tests for v0.15.0: assert_tool_call_count_per_call, assert_no_tool_call_cycles,
fixtures --summarize, compare --score."""
import json
import subprocess
import sys

import anthropic
import openai
import pytest

from agentprobe import AnthropicSession, Session

OAI_FIXTURE  = "tests/fixtures/bash_session.jsonl"
ANTH_SIMPLE  = "tests/fixtures/anthropic_simple.jsonl"
ANTH_TOOLS   = "tests/fixtures/anthropic_tools.jsonl"


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "agentprobe._cli", *args],
        capture_output=True, text=True,
    )


# ── helpers ───────────────────────────────────────────────────────────────────

def _oai_tool(name, args_dict=None):
    return {
        "id": "c1", "object": "chat.completion", "created": 1748700000, "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": None,
                     "tool_calls": [{"id": "t1", "type": "function",
                     "function": {"name": name, "arguments": json.dumps(args_dict or {"q": "x"})}}]},
                     "finish_reason": "tool_calls", "logprobs": None}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "system_fingerprint": None,
    }


def _oai_two_tools(name1, name2):
    return {
        "id": "c2", "object": "chat.completion", "created": 1748700000, "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": None,
                     "tool_calls": [
                         {"id": "t1", "type": "function", "function": {"name": name1, "arguments": "{}"}},
                         {"id": "t2", "type": "function", "function": {"name": name2, "arguments": "{}"}},
                     ]}, "finish_reason": "tool_calls", "logprobs": None}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "system_fingerprint": None,
    }


def _anth_tool(name, inp=None):
    return anthropic.types.Message.model_validate({
        "id": "m1", "type": "message", "role": "assistant",
        "content": [{"type": "tool_use", "id": "t1", "name": name, "input": inp or {"q": "x"}}],
        "model": "claude-sonnet-4-6", "stop_reason": "tool_use", "stop_sequence": None,
        "usage": {"input_tokens": 30, "output_tokens": 10},
    })


# ── assert_tool_call_count_per_call (OpenAI) ─────────────────────────────────

def test_tool_call_count_per_call_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai_tool("search"), _oai_tool("search")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "a"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "b"}])
    probe.assert_tool_call_count_per_call("search", 1)


def test_tool_call_count_per_call_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    # One call has 2 "search" tool calls in the same response
    with session.inject(_oai_two_tools("search", "search")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "a"}])
    with pytest.raises(AssertionError, match="2 time"):
        probe.assert_tool_call_count_per_call("search", 1)


def test_tool_call_count_per_call_skips_zero():
    """Calls that don't invoke the tool at all are not checked."""
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(
        _oai_tool("search"),
        {"id": "c3", "object": "chat.completion", "created": 1748700000, "model": "gpt-4o",
         "choices": [{"index": 0, "message": {"role": "assistant", "content": "done",
                      "tool_calls": None}, "finish_reason": "stop", "logprobs": None}],
         "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
         "system_fingerprint": None},
    ) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "a"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "b"}])
    probe.assert_tool_call_count_per_call("search", 1)


# ── assert_tool_call_count_per_call (Anthropic) ───────────────────────────────

def test_anthropic_tool_call_count_per_call_passes():
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    with session.inject(_anth_tool("weather"), _anth_tool("weather")) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "a"}])
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "b"}])
    probe.assert_tool_call_count_per_call("weather", 1)


# ── assert_no_tool_call_cycles (OpenAI) ──────────────────────────────────────

def test_no_tool_call_cycles_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(
        _oai_tool("search"),
        _oai_tool("read_file"),  # different from previous
        _oai_tool("search"),    # same as call 1 but not consecutive
    ) as probe:
        for _ in range(3):
            client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    probe.assert_no_tool_call_cycles()


def test_no_tool_call_cycles_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(
        _oai_tool("search"),
        _oai_tool("search"),  # same as previous = cycle
    ) as probe:
        for _ in range(2):
            client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError, match="twice in a row"):
        probe.assert_no_tool_call_cycles()


# ── assert_no_tool_call_cycles (Anthropic) ───────────────────────────────────

def test_anthropic_no_tool_call_cycles_passes():
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    with session.inject(_anth_tool("search"), _anth_tool("bash")) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "a"}])
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "b"}])
    probe.assert_no_tool_call_cycles()


def test_anthropic_no_tool_call_cycles_fails():
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    with session.inject(_anth_tool("bash"), _anth_tool("bash")) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "a"}])
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "b"}])
    with pytest.raises(AssertionError, match="twice in a row"):
        probe.assert_no_tool_call_cycles()


# ── CLI fixtures --summarize ──────────────────────────────────────────────────

def test_fixtures_summarize_exits_zero():
    r = run_cli("fixtures", "--summarize", "tests/fixtures")
    assert r.returncode == 0, r.stderr
    assert "call" in r.stdout


def test_fixtures_summarize_json():
    r = run_cli("fixtures", "--summarize", "--json", "tests/fixtures")
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    assert isinstance(data, list)
    assert len(data) > 0
    assert "calls" in data[0]


def test_fixtures_summarize_includes_anthropic():
    r = run_cli("fixtures", "--summarize", "--json", "tests/fixtures")
    data = json.loads(r.stdout)
    anthropic_fixtures = [d for d in data if "anthropic" in d.get("file", "")]
    assert len(anthropic_fixtures) > 0


# ── CLI compare --score ───────────────────────────────────────────────────────

def test_compare_score_identical():
    r = run_cli("compare", OAI_FIXTURE, OAI_FIXTURE)
    assert r.returncode == 0, r.stderr
    assert "100" in r.stdout


def test_compare_score_identical_json():
    r = run_cli("compare", "--json", OAI_FIXTURE, OAI_FIXTURE)
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    assert data["score"] == 100
    assert data["differences"] == 0


def test_compare_score_different():
    r = run_cli("compare", "--json", OAI_FIXTURE, ANTH_SIMPLE)
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    assert data["score"] < 100


def test_compare_score_identical_anthropic():
    r = run_cli("compare", "--json", ANTH_TOOLS, ANTH_TOOLS)
    data = json.loads(r.stdout)
    assert data["score"] == 100


def test_compare_score_anthropic_vs_oai():
    r = run_cli("compare", "--json", ANTH_SIMPLE, OAI_FIXTURE)
    data = json.loads(r.stdout)
    # Different call counts and formats → score < 100
    assert isinstance(data["score"], int)
