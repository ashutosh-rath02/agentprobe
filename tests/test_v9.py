"""Tests for v0.9.0: AnthropicSession/AnthropicAssertionProxy, assert_no_duplicate_tool_calls,
assert_context_growth."""
import json
import pytest
import anthropic
import anthropic.types
import openai

from agentprobe import AnthropicSession, Session

SIMPLE = "tests/fixtures/anthropic_simple.jsonl"
TOOLS  = "tests/fixtures/anthropic_tools.jsonl"
OAI_FIXTURE = "tests/fixtures/bash_session.jsonl"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _anthropic_client():
    return anthropic.Anthropic(api_key="dummy")


def _msg(text="Hello", model="claude-sonnet-4-6", stop_reason="end_turn",
         input_tokens=20, output_tokens=8):
    return anthropic.types.Message.model_validate({
        "id": "msg_test", "type": "message", "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": model, "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    })


def _tool_msg(name="search", inp=None, model="claude-sonnet-4-6",
              input_tokens=30, output_tokens=15):
    return anthropic.types.Message.model_validate({
        "id": "msg_tool", "type": "message", "role": "assistant",
        "content": [{"type": "tool_use", "id": "toolu_01", "name": name,
                     "input": inp or {"query": "test"}}],
        "model": model, "stop_reason": "tool_use", "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    })


# ── AnthropicSession.replay + AnthropicAssertionProxy properties ──────────────

def test_replay_simple_iteration_count():
    session = AnthropicSession()
    client = _anthropic_client()
    with session.replay(SIMPLE) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
    assert probe.iteration_count == 1


def test_replay_simple_final_output():
    session = AnthropicSession()
    client = _anthropic_client()
    with session.replay(SIMPLE) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
    assert "42" in probe.final_output


def test_replay_tools_tool_called():
    session = AnthropicSession()
    client = _anthropic_client()
    with session.replay(TOOLS) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "y"}])
    probe.assert_tool_called("search")
    probe.assert_not_tool_called("bash")
    assert probe.tools_called == ["search"]
    assert probe.first_tool_called == "search"
    assert probe.last_tool_called == "search"


def test_replay_tools_token_counts():
    session = AnthropicSession()
    client = _anthropic_client()
    with session.replay(TOOLS) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "y"}])
    assert probe.total_input_tokens == 90   # 30 + 60
    assert probe.total_output_tokens == 27  # 15 + 12
    assert probe.total_tokens == 117


def test_replay_tools_models_used():
    session = AnthropicSession()
    client = _anthropic_client()
    with session.replay(TOOLS) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "y"}])
    assert probe.models_used == ["claude-sonnet-4-6", "claude-sonnet-4-6"]


# ── AnthropicSession.inject ───────────────────────────────────────────────────

def test_inject_simple():
    session = AnthropicSession()
    client = _anthropic_client()
    with session.inject(_msg("Hi there!")) as probe:
        resp = client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                                      messages=[{"role": "user", "content": "hello"}])
    assert resp.content[0].text == "Hi there!"
    probe.assert_output_contains("Hi there!")
    probe.assert_iteration_count(1)


def test_inject_tool_called_with():
    session = AnthropicSession()
    client = _anthropic_client()
    with session.inject(_tool_msg("weather", {"location": "SF"})) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "weather?"}])
    probe.assert_tool_called("weather")
    probe.assert_tool_called_with("weather", location="SF")


def test_inject_multiple():
    session = AnthropicSession()
    client = _anthropic_client()
    with session.inject(_tool_msg("search"), _msg("Done")) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "a"}])
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "b"}])
    probe.assert_iteration_count(2)
    probe.assert_tool_called("search")
    probe.assert_output_contains("Done")


# ── AssertionProxy assertions on Anthropic proxy ─────────────────────────────

def test_assert_stop_reason_passes():
    session = AnthropicSession()
    client = _anthropic_client()
    with session.inject(_msg(stop_reason="end_turn")) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
    probe.assert_stop_reason("end_turn")


def test_assert_stop_reason_fails():
    session = AnthropicSession()
    client = _anthropic_client()
    with session.inject(_msg(stop_reason="end_turn")) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError, match="stop_reason"):
        probe.assert_stop_reason("tool_use")


def test_assert_max_tokens_passes():
    session = AnthropicSession()
    client = _anthropic_client()
    with session.inject(_msg(input_tokens=20, output_tokens=8)) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
    probe.assert_max_tokens(1000)


def test_assert_max_tokens_fails():
    session = AnthropicSession()
    client = _anthropic_client()
    with session.inject(_msg(input_tokens=20, output_tokens=8)) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError, match="tokens"):
        probe.assert_max_tokens(1)


def test_assert_all_models_in_passes():
    session = AnthropicSession()
    client = _anthropic_client()
    with session.inject(_msg()) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
    probe.assert_all_models_in("claude-sonnet-4-6", "claude-opus-4-8")


def test_assert_all_models_in_fails():
    session = AnthropicSession()
    client = _anthropic_client()
    with session.inject(_msg()) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError, match="disallowed"):
        probe.assert_all_models_in("claude-opus-4-8")


def test_assert_no_empty_responses_passes():
    session = AnthropicSession()
    client = _anthropic_client()
    with session.inject(_msg("Hello")) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
    probe.assert_no_empty_responses()


def test_assert_no_empty_responses_with_tool_use():
    session = AnthropicSession()
    client = _anthropic_client()
    with session.inject(_tool_msg()) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
    probe.assert_no_empty_responses()


def test_export_json_structure():
    session = AnthropicSession()
    client = _anthropic_client()
    with session.inject(_msg("Hello")) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
    data = json.loads(probe.export_json())
    assert "summary" in data
    assert "calls" in data
    assert data["summary"]["iteration_count"] == 1
    assert data["summary"]["total_input_tokens"] == 20


def test_tool_call_inputs_property():
    session = AnthropicSession()
    client = _anthropic_client()
    with session.inject(_tool_msg("search", {"query": "hello"})) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
    inputs = probe.tool_call_inputs
    assert "search" in inputs
    assert inputs["search"][0]["query"] == "hello"


def test_per_call_proxy():
    session = AnthropicSession()
    client = _anthropic_client()
    with session.inject(_tool_msg(), _msg("Done")) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "a"}])
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "b"}])
    probe.call(0).assert_tool_called("search")
    probe.call(1).assert_output_contains("Done")


# ── assert_no_duplicate_tool_calls (OpenAI) ───────────────────────────────────

def _oai_resp(content="ok", tool_name=None, tool_args=None):
    tool_calls = None
    if tool_name:
        tool_calls = [{
            "id": "call_1", "type": "function",
            "function": {"name": tool_name, "arguments": json.dumps(tool_args or {})},
        }]
    return {
        "id": "chatcmpl-v9", "object": "chat.completion", "created": 1748700000,
        "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content,
                     "tool_calls": tool_calls}, "finish_reason": "stop", "logprobs": None}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "system_fingerprint": None,
    }


def test_no_duplicate_tool_calls_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(
        _oai_resp(tool_name="search", tool_args={"q": "a"}),
        _oai_resp(tool_name="search", tool_args={"q": "b"}),  # different args: OK
    ) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    probe.assert_no_duplicate_tool_calls()


def test_no_duplicate_tool_calls_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(
        _oai_resp(tool_name="search", tool_args={"q": "same"}),
        _oai_resp(tool_name="search", tool_args={"q": "same"}),  # identical: bad
    ) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    with pytest.raises(AssertionError, match="duplicate"):
        probe.assert_no_duplicate_tool_calls()


def test_no_duplicate_tool_calls_no_tools_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(_oai_resp("hello")) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    probe.assert_no_duplicate_tool_calls()  # no tools = no duplicates


# ── assert_no_duplicate_tool_calls (Anthropic) ────────────────────────────────

def test_anthropic_no_duplicate_tool_calls_passes():
    session = AnthropicSession()
    client = _anthropic_client()
    with session.inject(
        _tool_msg("search", {"q": "a"}),
        _tool_msg("search", {"q": "b"}),
    ) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "y"}])
    probe.assert_no_duplicate_tool_calls()


def test_anthropic_no_duplicate_tool_calls_fails():
    session = AnthropicSession()
    client = _anthropic_client()
    with session.inject(
        _tool_msg("search", {"q": "same"}),
        _tool_msg("search", {"q": "same"}),
    ) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "y"}])
    with pytest.raises(AssertionError, match="duplicate"):
        probe.assert_no_duplicate_tool_calls()


# ── assert_context_growth (OpenAI) ────────────────────────────────────────────

def _oai_resp_with_usage(prompt_tokens):
    return {
        "id": "chatcmpl-ctx", "object": "chat.completion", "created": 1748700000,
        "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok",
                     "tool_calls": None}, "finish_reason": "stop", "logprobs": None}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 5,
                  "total_tokens": prompt_tokens + 5},
        "system_fingerprint": None,
    }


def test_context_growth_passes():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(
        _oai_resp_with_usage(100),
        _oai_resp_with_usage(150),  # 1.5x growth
    ) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    probe.assert_context_growth(2.0)  # 1.5x <= 2.0: pass


def test_context_growth_fails():
    session = Session()
    client = openai.OpenAI(api_key="dummy")
    with session.inject(
        _oai_resp_with_usage(100),
        _oai_resp_with_usage(300),  # 3x growth
    ) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "y"}])
    with pytest.raises(AssertionError, match="context grew"):
        probe.assert_context_growth(2.0)  # 3x > 2.0: fail


# ── assert_context_growth (Anthropic) ────────────────────────────────────────

def test_anthropic_context_growth_passes():
    session = AnthropicSession()
    client = _anthropic_client()
    with session.inject(
        _msg(input_tokens=100, output_tokens=10),
        _msg(input_tokens=150, output_tokens=10),
    ) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "y"}])
    probe.assert_context_growth(2.0)


def test_anthropic_context_growth_fails():
    session = AnthropicSession()
    client = _anthropic_client()
    with session.inject(
        _msg(input_tokens=100, output_tokens=10),
        _msg(input_tokens=500, output_tokens=10),
    ) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "y"}])
    with pytest.raises(AssertionError, match="context grew"):
        probe.assert_context_growth(2.0)


# ── AnthropicSession.record / replay round-trip ───────────────────────────────

def test_record_replay_roundtrip(tmp_path):
    fixture = tmp_path / "roundtrip.jsonl"
    session = AnthropicSession()
    client = _anthropic_client()

    # Record with inject (pretend we're recording)
    with session.inject(_msg("Recorded response")) as probe:
        resp = client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                                      messages=[{"role": "user", "content": "x"}])

    # Save to fixture manually via dump_fixture equivalent
    from agentprobe._session import _save_calls
    _save_calls(probe._calls, fixture)

    # Replay
    client2 = _anthropic_client()
    with session.replay(str(fixture)) as probe2:
        resp2 = client2.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                                        messages=[{"role": "user", "content": "x"}])
    assert resp2.content[0].text == "Recorded response"
    probe2.assert_output_contains("Recorded response")


# ── AnthropicSession estimated cost ──────────────────────────────────────────

def test_estimated_cost_usd_nonzero():
    session = AnthropicSession()
    client = _anthropic_client()
    with session.inject(_msg(input_tokens=1_000_000, output_tokens=0)) as probe:
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
    assert probe.estimated_cost_usd == pytest.approx(3.0, rel=1e-3)  # $3/Mtok input


def test_estimated_cost_usd_zero_for_unknown_model():
    session = AnthropicSession()
    client = _anthropic_client()
    msg = anthropic.types.Message.model_validate({
        "id": "msg_x", "type": "message", "role": "assistant",
        "content": [{"type": "text", "text": "hi"}],
        "model": "unknown-model-xyz", "stop_reason": "end_turn",
        "stop_sequence": None, "usage": {"input_tokens": 1000, "output_tokens": 500},
    })
    with session.inject(msg) as probe:
        client.messages.create(model="unknown-model-xyz", max_tokens=1024,
                               messages=[{"role": "user", "content": "x"}])
    assert probe.estimated_cost_usd == 0.0
