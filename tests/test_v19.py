"""Tests for v0.19.0: assert_tool_never_called_with, assert_response_format,
assert_prompt_growth_bounded, show --calls N."""
import json
import subprocess
import sys

import anthropic
import openai
import pytest

from agentprobe import AnthropicSession, Session

OAI_FIXTURE = "tests/fixtures/bash_session.jsonl"


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "agentprobe._cli", *args],
        capture_output=True, text=True,
    )


# ── helpers ───────────────────────────────────────────────────────────────────

def _oai(content="ok", prompt_tokens=10):
    return {
        "id": "c1", "object": "chat.completion", "created": 1748700000, "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content,
                     "tool_calls": None}, "finish_reason": "stop", "logprobs": None}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 5,
                  "total_tokens": prompt_tokens + 5},
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


def _anth(text="Hello"):
    return anthropic.types.Message.model_validate({
        "id": "m1", "type": "message", "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": "claude-sonnet-4-6", "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 20, "output_tokens": 8},
    })


def _anth_tool(name, inp=None):
    return anthropic.types.Message.model_validate({
        "id": "m2", "type": "message", "role": "assistant",
        "content": [{"type": "tool_use", "id": "t1", "name": name, "input": inp or {}}],
        "model": "claude-sonnet-4-6", "stop_reason": "tool_use", "stop_sequence": None,
        "usage": {"input_tokens": 30, "output_tokens": 10},
    })


# ── assert_tool_never_called_with (OpenAI) ────────────────────────────────────

def test_tool_never_called_with_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai_tool("search", {"q": "safe", "safe_search": True})) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    probe.assert_tool_never_called_with("search", safe_search=False)  # False was never used


def test_tool_never_called_with_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai_tool("search", {"q": "bad", "safe_search": False})) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError, match="forbidden input"):
        probe.assert_tool_never_called_with("search", safe_search=False)


# ── assert_tool_never_called_with (Anthropic) ────────────────────────────────

def test_anthropic_tool_never_called_with_passes():
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    with session.inject(_anth_tool("fetch", {"url": "https://safe.example.com", "js": False})) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
    probe.assert_tool_never_called_with("fetch", js=True)


def test_anthropic_tool_never_called_with_fails():
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    with session.inject(_anth_tool("fetch", {"url": "https://x.com", "js": True})) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError, match="forbidden input"):
        probe.assert_tool_never_called_with("fetch", js=True)


# ── assert_response_format json ───────────────────────────────────────────────

def test_response_format_json_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai('{"result": "ok", "count": 3}')) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    probe.assert_response_format("json")


def test_response_format_json_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai("This is plain text")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError, match="valid JSON"):
        probe.assert_response_format("json")


# ── assert_response_format markdown ──────────────────────────────────────────

def test_response_format_markdown_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai("# Title\n\n- item 1\n- item 2")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    probe.assert_response_format("markdown")


def test_response_format_markdown_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai("plain text with no headings or lists")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError, match="markdown format"):
        probe.assert_response_format("markdown")


def test_response_format_anthropic_json():
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    with session.inject(_anth('{"status": "done"}')) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
    probe.assert_response_format("json")


# ── assert_prompt_growth_bounded ──────────────────────────────────────────────

def test_prompt_growth_bounded_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(
        _oai(prompt_tokens=100),
        _oai(prompt_tokens=150),
        _oai(prompt_tokens=200),
    ) as probe:
        for _ in range(3):
            client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    probe.assert_prompt_growth_bounded(2.0)  # max 1.5x growth per call


def test_prompt_growth_bounded_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(
        _oai(prompt_tokens=100),
        _oai(prompt_tokens=400),  # 4x growth!
    ) as probe:
        for _ in range(2):
            client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError, match="grew"):
        probe.assert_prompt_growth_bounded(2.0)


# ── CLI show --calls N ────────────────────────────────────────────────────────

def test_show_calls_first_one():
    r = run_cli("show", "--calls", "1", OAI_FIXTURE)
    assert r.returncode == 0, r.stderr
    assert "Call 1/1" in r.stdout


def test_show_calls_last_one():
    """Negative N shows last N calls."""
    r = run_cli("show", "--calls", "-1", OAI_FIXTURE)
    assert r.returncode == 0, r.stderr
    assert "1 call" in r.stdout


def test_show_calls_all():
    r_all = run_cli("show", OAI_FIXTURE)
    r_filtered = run_cli("show", "--calls", "100", OAI_FIXTURE)
    # Filtering more than exists shows same as all
    assert r_all.stdout == r_filtered.stdout
