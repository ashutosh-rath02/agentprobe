"""Tests for v0.21.0: assert_response_latency_percentile, assert_all_responses_under_tokens,
assert_tool_arg_type."""
import json
import subprocess
import sys

import anthropic
import openai
import pytest

from agentprobe import AnthropicSession, Session
from agentprobe._models import RecordedCall

OAI_FIXTURE = "tests/fixtures/bash_session.jsonl"


# ── helpers ───────────────────────────────────────────────────────────────────

def _oai(content="ok", prompt_tokens=10, completion_tokens=5):
    return {
        "id": "c1", "object": "chat.completion", "created": 1748700000, "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content,
                     "tool_calls": None}, "finish_reason": "stop", "logprobs": None}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
        "system_fingerprint": None,
    }


def _oai_tool(name, args_dict=None):
    return {
        "id": "c2", "object": "chat.completion", "created": 1748700000, "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": None,
                     "tool_calls": [{"id": "t1", "type": "function",
                     "function": {"name": name, "arguments": json.dumps(args_dict or {})}}]},
                     "finish_reason": "tool_calls", "logprobs": None}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "system_fingerprint": None,
    }


def _anth_tool(name, inp=None):
    return anthropic.types.Message.model_validate({
        "id": "m1", "type": "message", "role": "assistant",
        "content": [{"type": "tool_use", "id": "t1", "name": name, "input": inp or {}}],
        "model": "claude-sonnet-4-6", "stop_reason": "tool_use", "stop_sequence": None,
        "usage": {"input_tokens": 30, "output_tokens": 10},
    })


# ── assert_response_latency_percentile ───────────────────────────────────────

def test_latency_percentile_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(OAI_FIXTURE) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "a"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "b"}])
    probe.assert_response_latency_percentile(100, 1_000_000)


def test_latency_percentile_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(OAI_FIXTURE) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "a"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "b"}])
    with pytest.raises(AssertionError, match="latency"):
        probe.assert_response_latency_percentile(100, 0.001)


def test_latency_percentile_no_durations_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai()) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    probe.assert_response_latency_percentile(95, 0.001)  # no durations → pass


def test_anthropic_latency_percentile_passes():
    from agentprobe._session import _load_calls
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    with session.replay("tests/fixtures/anthropic_simple.jsonl") as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
    probe.assert_response_latency_percentile(50, 1_000_000)


# ── assert_all_responses_under_tokens (OpenAI) ───────────────────────────────

def test_all_responses_under_tokens_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai(prompt_tokens=100, completion_tokens=50)) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    probe.assert_all_responses_under_tokens(200)  # 150 < 200


def test_all_responses_under_tokens_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai(prompt_tokens=100, completion_tokens=50)) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError, match="per-call limit"):
        probe.assert_all_responses_under_tokens(100)  # 150 > 100


# ── assert_all_responses_under_tokens (Anthropic) ────────────────────────────

def test_anthropic_all_responses_under_tokens_passes():
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    with session.replay("tests/fixtures/anthropic_simple.jsonl") as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
    probe.assert_all_responses_under_tokens(10_000)


# ── assert_tool_arg_type (OpenAI) ─────────────────────────────────────────────

def test_tool_arg_type_str_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai_tool("search", {"q": "hello", "page": 1})) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    probe.assert_tool_arg_type("search", "q", "str")


def test_tool_arg_type_int_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai_tool("search", {"q": "hello", "page": 1})) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    probe.assert_tool_arg_type("search", "page", "int")


def test_tool_arg_type_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai_tool("search", {"q": "hello", "page": "not_an_int"})) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError, match="expected int"):
        probe.assert_tool_arg_type("search", "page", "int")


def test_tool_arg_type_bool_not_confused_with_int():
    """bool is a subclass of int — assert_tool_arg_type must distinguish them."""
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai_tool("search", {"enabled": True})) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    probe.assert_tool_arg_type("search", "enabled", "bool")
    with pytest.raises(AssertionError, match="expected int"):
        probe.assert_tool_arg_type("search", "enabled", "int")


# ── assert_tool_arg_type (Anthropic) ─────────────────────────────────────────

def test_anthropic_tool_arg_type_passes():
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    with session.inject(_anth_tool("weather", {"city": "NYC", "days": 5})) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
    probe.assert_tool_arg_type("weather", "city", "str")
    probe.assert_tool_arg_type("weather", "days", "int")
