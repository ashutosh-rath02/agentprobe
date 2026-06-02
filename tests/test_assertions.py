"""Tests for new assertions: assert_tool_called_with, assert_stop_reason, token metrics."""
import pytest
import openai

from agentprobe import Session

FIXTURE = "tests/fixtures/bash_session.jsonl"


def _run(session, fixture=FIXTURE):
    client = openai.OpenAI(api_key="dummy")
    tools = [{"type": "function", "function": {"name": "bash", "description": "Run bash", "parameters": {
        "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]
    }}}]
    messages = [{"role": "user", "content": "list files in /tmp"}]
    with session.replay(fixture) as probe:
        while True:
            resp = client.chat.completions.create(model="gpt-4o", messages=messages, tools=tools)
            choice = resp.choices[0]
            tool_calls = choice.message.tool_calls or []
            if not tool_calls:
                break
            messages.append({
                "role": "assistant",
                "content": choice.message.content,
                "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in tool_calls
                ]
            })
            for tc in tool_calls:
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": "file1.txt"})
    return probe


# ── assert_tool_called_with ───────────────────────────────────────────────────

def test_assert_tool_called_with_passes():
    probe = _run(Session())
    probe.assert_tool_called_with("bash", command="ls /tmp")


def test_assert_tool_called_with_partial_match():
    probe = _run(Session())
    probe.assert_tool_called_with("bash", command="ls /tmp")


def test_assert_tool_called_with_wrong_args_fails():
    probe = _run(Session())
    with pytest.raises(AssertionError, match="never with"):
        probe.assert_tool_called_with("bash", command="rm -rf /")


def test_assert_tool_called_with_wrong_tool_fails():
    probe = _run(Session())
    with pytest.raises(AssertionError, match="never called"):
        probe.assert_tool_called_with("web_search", query="test")


# ── assert_stop_reason ────────────────────────────────────────────────────────

def test_assert_stop_reason_stop():
    probe = _run(Session())
    probe.assert_stop_reason("stop")


def test_assert_stop_reason_wrong_fails():
    probe = _run(Session())
    with pytest.raises(AssertionError, match="finish_reason"):
        probe.assert_stop_reason("max_tokens")


# ── token metrics ─────────────────────────────────────────────────────────────

def test_total_input_tokens():
    probe = _run(Session())
    assert probe.total_input_tokens == 130  # 50 + 80 from fixture


def test_total_output_tokens():
    probe = _run(Session())
    assert probe.total_output_tokens == 55  # 30 + 25 from fixture


def test_total_tokens():
    probe = _run(Session())
    assert probe.total_tokens == 185


def test_assert_max_tokens_passes():
    probe = _run(Session())
    probe.assert_max_tokens(1000)


def test_assert_max_tokens_fails():
    probe = _run(Session())
    with pytest.raises(AssertionError, match="total tokens"):
        probe.assert_max_tokens(10)
