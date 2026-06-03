"""Tests for v0.22.0: assert_min_tool_calls, assert_output_char_count,
assert_tool_result_contains."""
import json

import anthropic
import openai
import pytest

from agentprobe import AnthropicSession, Session

OAI_FIXTURE = "tests/fixtures/bash_session.jsonl"


# ── helpers ───────────────────────────────────────────────────────────────────

def _oai(content="hello world"):
    return {
        "id": "c1", "object": "chat.completion", "created": 1748700000, "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content,
                     "tool_calls": None}, "finish_reason": "stop", "logprobs": None}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
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


def _anth_text(content="hello world"):
    return anthropic.types.Message.model_validate({
        "id": "m2", "type": "message", "role": "assistant",
        "content": [{"type": "text", "text": content}],
        "model": "claude-sonnet-4-6", "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 20, "output_tokens": 5},
    })


# ── assert_min_tool_calls (OpenAI) ────────────────────────────────────────────

def test_min_tool_calls_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.replay(OAI_FIXTURE) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    probe.assert_min_tool_calls(1)


def test_min_tool_calls_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai()) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError, match="below minimum"):
        probe.assert_min_tool_calls(1)


def test_min_tool_calls_zero_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai()) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    probe.assert_min_tool_calls(0)


# ── assert_min_tool_calls (Anthropic) ─────────────────────────────────────────

def test_anthropic_min_tool_calls_passes():
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    with session.inject(_anth_tool("search", {"q": "test"})) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
    probe.assert_min_tool_calls(1)


def test_anthropic_min_tool_calls_fails():
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    with session.inject(_anth_text()) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError, match="below minimum"):
        probe.assert_min_tool_calls(1)


# ── assert_output_char_count (OpenAI) ────────────────────────────────────────

def test_output_char_count_min_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai("hello world")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    probe.assert_output_char_count(min_chars=5)


def test_output_char_count_max_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai("hi")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    probe.assert_output_char_count(max_chars=10)


def test_output_char_count_min_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai("hi")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError, match="at least 100"):
        probe.assert_output_char_count(min_chars=100)


def test_output_char_count_max_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai("hello world")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError, match="at most 3"):
        probe.assert_output_char_count(max_chars=3)


# ── assert_output_char_count (Anthropic) ─────────────────────────────────────

def test_anthropic_output_char_count_passes():
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    with session.inject(_anth_text("hello world")) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
    probe.assert_output_char_count(min_chars=5, max_chars=100)


# ── assert_tool_result_contains (OpenAI) ─────────────────────────────────────

def _make_oai_with_tool_result(tool_call_id, tool_name, tool_result):
    """Two-call inject: first returns a tool call, second receives the result."""
    call1 = {
        "id": "c1", "object": "chat.completion", "created": 1748700000, "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": None,
                     "tool_calls": [{"id": tool_call_id, "type": "function",
                     "function": {"name": tool_name, "arguments": "{}"}}]},
                     "finish_reason": "tool_calls", "logprobs": None}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "system_fingerprint": None,
    }
    call2 = {
        "id": "c2", "object": "chat.completion", "created": 1748700000, "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "done",
                     "tool_calls": None}, "finish_reason": "stop", "logprobs": None}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23},
        "system_fingerprint": None,
    }
    return call1, call2, tool_result


def test_tool_result_contains_passes():
    """Simulate a real agent: second call sends accumulated messages with tool result."""
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    call1, call2, result_text = _make_oai_with_tool_result("tc1", "search", "agentprobe rocks")
    with session.inject(call1, call2) as probe:
        resp1 = client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "find it"}])
        # Simulate agent feeding tool result back
        tc = resp1.choices[0].message.tool_calls[0]
        client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": "find it"},
                {"role": "assistant", "content": None,
                 "tool_calls": [{"id": tc.id, "type": "function",
                                 "function": {"name": tc.function.name,
                                              "arguments": tc.function.arguments}}]},
                {"role": "tool", "tool_call_id": tc.id, "content": result_text},
            ],
        )
    probe.assert_tool_result_contains("search", "agentprobe")


def test_tool_result_contains_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    call1, call2, result_text = _make_oai_with_tool_result("tc1", "search", "agentprobe rocks")
    with session.inject(call1, call2) as probe:
        resp1 = client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        tc = resp1.choices[0].message.tool_calls[0]
        client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "tool", "tool_call_id": tc.id, "content": result_text},
            ],
        )
    with pytest.raises(AssertionError, match="no result for tool 'search'"):
        probe.assert_tool_result_contains("search", "DOES_NOT_EXIST_XYZ")


def test_tool_result_contains_wrong_tool_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    call1, call2, result_text = _make_oai_with_tool_result("tc1", "search", "agentprobe rocks")
    with session.inject(call1, call2) as probe:
        resp1 = client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        tc = resp1.choices[0].message.tool_calls[0]
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "tool", "tool_call_id": tc.id, "content": result_text}],
        )
    with pytest.raises(AssertionError, match="no result for tool 'nonexistent'"):
        probe.assert_tool_result_contains("nonexistent", "agentprobe")
