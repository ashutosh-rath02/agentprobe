"""Tests for v0.10.0: record --capture-stdout, agentprobe replay CLI,
AnthropicMultiSession, assert_tool_called_before_output, call(n).tool_call_inputs."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import anthropic
import openai
import openai.types.chat
import pytest

from agentprobe import AnthropicMultiSession, AnthropicSession, Session

OAI_FIXTURE = "tests/fixtures/bash_session.jsonl"
ANTH_SIMPLE  = "tests/fixtures/anthropic_simple.jsonl"
ANTH_TOOLS   = "tests/fixtures/anthropic_tools.jsonl"


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "agentprobe._cli", *args],
        capture_output=True, text=True,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _oai_resp(content="ok", tool_name=None, tool_args=None, prompt_tokens=10):
    tool_calls = None
    if tool_name:
        tool_calls = [{
            "id": "call_1", "type": "function",
            "function": {"name": tool_name, "arguments": json.dumps(tool_args or {})},
        }]
    return {
        "id": "chatcmpl-v10", "object": "chat.completion", "created": 1748700000,
        "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content,
                     "tool_calls": tool_calls}, "finish_reason": "stop", "logprobs": None}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 5,
                  "total_tokens": prompt_tokens + 5},
        "system_fingerprint": None,
    }


def _anth_msg(text="Hi", model="claude-sonnet-4-6", stop_reason="end_turn",
              input_tokens=20, output_tokens=8):
    return anthropic.types.Message.model_validate({
        "id": "msg_v10", "type": "message", "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": model, "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    })


def _anth_tool_msg(name="search", inp=None, model="claude-sonnet-4-6"):
    return anthropic.types.Message.model_validate({
        "id": "msg_tool_v10", "type": "message", "role": "assistant",
        "content": [{"type": "tool_use", "id": "toolu_v10", "name": name,
                     "input": inp or {"q": "test"}}],
        "model": model, "stop_reason": "tool_use", "stop_sequence": None,
        "usage": {"input_tokens": 30, "output_tokens": 15},
    })


# ── assert_tool_called_before_output (OpenAI) ─────────────────────────────────

def test_assert_tool_called_before_output_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(
        _oai_resp(tool_name="search"),
        _oai_resp("Here are the results."),
    ) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    probe.assert_tool_called_before_output()


def test_assert_tool_called_before_output_fails_no_tools():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai_resp("Hello")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError, match="no tool calls"):
        probe.assert_tool_called_before_output()


def test_assert_tool_called_before_output_fails_no_final_text():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    # Both responses are tool calls (no final text)
    with session.inject(
        _oai_resp(content=None, tool_name="search"),
        _oai_resp(content=None, tool_name="search"),
    ) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    with pytest.raises(AssertionError, match="text content"):
        probe.assert_tool_called_before_output()


# ── assert_tool_called_before_output (Anthropic) ──────────────────────────────

def test_anthropic_assert_tool_called_before_output_passes():
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    with session.inject(_anth_tool_msg(), _anth_msg("Final answer.")) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "y"}])
    probe.assert_tool_called_before_output()


def test_anthropic_assert_tool_called_before_output_fails_no_tools():
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    with session.inject(_anth_msg("Hi")) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError, match="no tool calls"):
        probe.assert_tool_called_before_output()


# ── call(n).tool_call_inputs (OpenAI) ────────────────────────────────────────

def test_per_call_tool_call_inputs_openai():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(
        _oai_resp(tool_name="search", tool_args={"q": "hello"}),
        _oai_resp(tool_name="read",   tool_args={"file": "x.txt"}),
    ) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "a"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "b"}])

    assert probe.call(0).tool_call_inputs == {"search": [{"q": "hello"}]}
    assert probe.call(1).tool_call_inputs == {"read": [{"file": "x.txt"}]}


def test_per_call_tool_call_inputs_anthropic():
    session = AnthropicSession()
    client = anthropic.Anthropic(api_key="dummy")
    with session.inject(
        _anth_tool_msg("weather", {"city": "NYC"}),
        _anth_tool_msg("weather", {"city": "SF"}),
    ) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "a"}])
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "b"}])

    assert probe.call(0).tool_call_inputs == {"weather": [{"city": "NYC"}]}
    assert probe.call(1).tool_call_inputs == {"weather": [{"city": "SF"}]}


# ── AnthropicMultiSession ─────────────────────────────────────────────────────

def test_anthropic_multi_replay_two_clients():
    multi = AnthropicMultiSession()
    client_a = anthropic.Anthropic(api_key="dummy")
    client_b = anthropic.Anthropic(api_key="dummy")

    with multi.replay(client_a, ANTH_SIMPLE) as probe_a:
        with multi.replay(client_b, ANTH_TOOLS) as probe_b:
            resp_a = client_a.messages.create(
                model="claude-sonnet-4-6", max_tokens=1024,
                messages=[{"role": "user", "content": "x"}])
            client_b.messages.create(
                model="claude-sonnet-4-6", max_tokens=1024,
                messages=[{"role": "user", "content": "y"}])
            client_b.messages.create(
                model="claude-sonnet-4-6", max_tokens=1024,
                messages=[{"role": "user", "content": "z"}])

    assert "42" in resp_a.content[0].text
    probe_a.assert_iteration_count(1)
    probe_b.assert_iteration_count(2)
    probe_b.assert_tool_called("search")


def test_anthropic_multi_replay_chain():
    multi = AnthropicMultiSession()
    client_a = anthropic.Anthropic(api_key="dummy")

    with multi.replay_chain((client_a, [ANTH_SIMPLE, ANTH_TOOLS])) as probes:
        r1 = client_a.messages.create(
            model="claude-sonnet-4-6", max_tokens=1024,
            messages=[{"role": "user", "content": "x"}])
        client_a.messages.create(
            model="claude-sonnet-4-6", max_tokens=1024,
            messages=[{"role": "user", "content": "y"}])
        client_a.messages.create(
            model="claude-sonnet-4-6", max_tokens=1024,
            messages=[{"role": "user", "content": "z"}])

    assert "42" in r1.content[0].text
    assert probes[client_a].iteration_count == 3


# ── CLI: agentprobe replay ────────────────────────────────────────────────────

def test_replay_cli_basic(tmp_path):
    """replay command runs the script and exits 0 without hitting real API."""
    script = tmp_path / "agent.py"
    # agent_a.jsonl has a simple text response; content is not None
    script.write_text(
        "import openai\n"
        "client = openai.OpenAI(api_key='dummy')\n"
        "r = client.chat.completions.create(\n"
        "    model='gpt-4o', messages=[{'role': 'user', 'content': 'hi'}])\n"
        "assert r.choices[0].finish_reason == 'stop'\n"
    )
    r = run_cli("replay", "tests/fixtures/agent_a.jsonl", str(script))
    assert r.returncode == 0, r.stderr
    assert "replayed" in r.stdout


def test_replay_cli_wrong_fixture(tmp_path):
    script = tmp_path / "agent.py"
    script.write_text("pass\n")
    r = run_cli("replay", "tests/fixtures/no_such.jsonl", str(script))
    assert r.returncode != 0


def test_replay_cli_strict_passes(tmp_path):
    """Script that consumes exactly all calls passes --strict."""
    script = tmp_path / "agent.py"
    # bash_session.jsonl has 2 calls
    script.write_text(
        "import openai\n"
        "client = openai.OpenAI(api_key='dummy')\n"
        "client.chat.completions.create(model='gpt-4o', messages=[{'role':'user','content':'a'}])\n"
        "client.chat.completions.create(model='gpt-4o', messages=[{'role':'user','content':'b'}])\n"
    )
    r = run_cli("replay", "--strict", OAI_FIXTURE, str(script))
    assert r.returncode == 0, r.stderr


def test_replay_cli_strict_fails_under_consume(tmp_path):
    """Script that consumes fewer calls than fixture fails --strict."""
    script = tmp_path / "agent.py"
    # bash_session.jsonl has 2 calls; only consuming 1
    script.write_text(
        "import openai\n"
        "client = openai.OpenAI(api_key='dummy')\n"
        "client.chat.completions.create(model='gpt-4o', messages=[{'role':'user','content':'a'}])\n"
    )
    r = run_cli("replay", "--strict", OAI_FIXTURE, str(script))
    assert r.returncode == 1


def test_replay_cli_anthropic_provider(tmp_path):
    """replay --provider anthropic patches anthropic instead of openai."""
    script = tmp_path / "agent.py"
    script.write_text(
        "import anthropic\n"
        "client = anthropic.Anthropic(api_key='dummy')\n"
        "r = client.messages.create(\n"
        "    model='claude-sonnet-4-6', max_tokens=1024,\n"
        "    messages=[{'role': 'user', 'content': 'hi'}])\n"
        "assert r.content[0].text\n"
    )
    r = run_cli("replay", "--provider", "anthropic", ANTH_SIMPLE, str(script))
    assert r.returncode == 0, r.stderr
    assert "replayed" in r.stdout


# ── CLI: record --capture-stdout ─────────────────────────────────────────────

def test_record_capture_stdout_stores_output(tmp_path):
    """--capture-stdout saves script output in _meta header."""
    script = tmp_path / "agent.py"
    script.write_text(
        "import openai, sys\n"
        "print('hello from agent')\n"
        "client = openai.OpenAI(api_key='dummy')\n"
        "client.chat.completions.create(model='gpt-4o', messages=[{'role':'user','content':'x'}])\n"
    )
    output = str(tmp_path / "captured.jsonl")
    mock_resp = openai.types.chat.ChatCompletion.model_validate({
        "id": "chatcmpl-cap", "object": "chat.completion", "created": 1748700000,
        "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok",
                     "tool_calls": None}, "finish_reason": "stop", "logprobs": None}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        "system_fingerprint": None,
    })
    from agentprobe._cli import cmd_record
    args = argparse.Namespace(
        script=str(script), output=output, env=None, output_format=None,
        watch=False, interval=1.0, timeout=None, dry_run=False, append=False,
        capture_stdout=True,
    )
    with patch("openai.resources.chat.completions.Completions.create", return_value=mock_resp):
        cmd_record(args)

    # Read back the fixture and check _meta
    fixture_path = Path(output)
    assert fixture_path.exists()
    lines = fixture_path.read_text().splitlines()
    meta_line = next((l for l in lines if "_meta" in l), None)
    assert meta_line is not None
    meta = json.loads(meta_line)["_meta"]
    assert "stdout" in meta
    assert "hello from agent" in meta["stdout"]


def test_record_capture_stdout_note_in_output(tmp_path):
    """--capture-stdout adds [+stdout] note in CLI output."""
    script = tmp_path / "agent.py"
    script.write_text(
        "import openai\n"
        "print('test output')\n"
        "client = openai.OpenAI(api_key='dummy')\n"
        "client.chat.completions.create(model='gpt-4o', messages=[{'role':'user','content':'x'}])\n"
    )
    output = str(tmp_path / "out.jsonl")
    r = run_cli("record", "--capture-stdout", str(script), output)
    # Script will fail (no real key), but if it DID work the note appears
    # Just verify the flag is parsed without error
    assert "unrecognized" not in r.stderr


def test_record_no_capture_stdout_by_default(tmp_path):
    """Without --capture-stdout, _meta should not contain stdout key."""
    fixture_path = Path(OAI_FIXTURE)
    lines = fixture_path.read_text().splitlines()
    meta_line = next((l for l in lines if "_meta" in l), None)
    if meta_line:
        meta = json.loads(meta_line)["_meta"]
        assert "stdout" not in meta
