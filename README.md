<p align="center">
  <h1 align="center">agentprobe</h1>
  <p align="center">Deterministic record-and-replay test harness for AI agents</p>
</p>

<p align="center">
  <a href="https://pypi.org/project/pytest-agentprobe/"><img alt="PyPI" src="https://img.shields.io/pypi/v/pytest-agentprobe?color=blue&label=PyPI"></a>
  <a href="https://pypi.org/project/pytest-agentprobe/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/pytest-agentprobe"></a>
  <a href="https://github.com/ashutosh-rath02/agentprobe/blob/master/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-green"></a>
</p>

<br>

Record your agent's API calls once. Replay them in CI forever — **no network, no cost, no flakiness.**

Works with **OpenAI** and **Anthropic (Claude)**. No changes to your agent code.

```python
def test_file_agent(agentprobe):
    with agentprobe.replay("tests/fixtures/list_files.jsonl") as probe:
        my_agent.run(client, "list files in /tmp")

    probe.assert_tool_called("bash")
    probe.assert_tool_called_with("bash", command="ls /tmp")
    probe.assert_max_iterations(3)
    probe.assert_output_contains("/tmp")
    probe.assert_average_latency_under(500)
```

---

## Contents

- [Install](#install)
- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [Anthropic / Claude](#anthropic--claude)
- [Async agents](#async-agents)
- [Assertions](#assertions)
- [CLI reference](#cli-reference)
- [Fixture format](#fixture-format)

---

## Install

```bash
pip install pytest-agentprobe
```

Requires Python 3.9+ and `openai >= 1.0.0`. For Anthropic Claude agents, also install `anthropic`.

---

## How it works

`agentprobe` patches `openai.ChatCompletions.create` (or `anthropic.messages.create`) at the class level — no changes to your agent code required.

| Mode | Behaviour |
|---|---|
| `record` | Runs the agent against the real API; saves every request/response pair to a `.jsonl` fixture file |
| `replay` | Returns saved responses instead of hitting the API — instant, free, deterministic |
| `auto` | Records on first run, replays on every subsequent run |
| `inject` | Supplies hand-crafted responses without a file — ideal for unit tests |

---

## Quick start

### 1. Record a session

Run this once against the real API to capture a fixture:

```python
from agentprobe import Session
import openai

session = Session()
client = openai.OpenAI()  # uses OPENAI_API_KEY

with session.record("tests/fixtures/my_agent.jsonl") as probe:
    my_agent.run(client, "what files are in /tmp?")
```

Commit the generated `.jsonl` file to your repository.

### 2. Replay in CI

```python
def test_my_agent(agentprobe):
    client = openai.OpenAI(api_key="dummy")  # key is not used during replay

    with agentprobe.replay("tests/fixtures/my_agent.jsonl") as probe:
        my_agent.run(client, "what files are in /tmp?")

    probe.assert_tool_called("bash")
    probe.assert_not_tool_called("web_search")
    probe.assert_tool_called_with("bash", command="ls /tmp")
    probe.assert_max_iterations(4)
    probe.assert_output_contains("/tmp")
    probe.assert_max_tokens(500)
```

> **No `conftest.py` needed.** The `agentprobe` pytest fixture is registered automatically as a plugin.

### 3. Auto mode

Records on first run; replays on every run after that:

```python
def test_my_agent(agentprobe):
    with agentprobe.auto("tests/fixtures/my_agent.jsonl") as probe:
        my_agent.run(client, "what files are in /tmp?")

    probe.assert_tool_called("bash")
```

### 4. Inject (unit tests without fixture files)

Supply a raw response dict inline — no file I/O, no network:

```python
def test_tool_type_validation():
    session = Session()
    client = openai.OpenAI(api_key="dummy")

    with session.inject(mock_tool_response) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[...])

    probe.assert_tool_arg_type("search", "page", "int")
    probe.assert_output_not_empty()
```

---

## Anthropic / Claude

Use `AnthropicSession` — the assertion API is identical:

```python
from agentprobe import AnthropicSession
import anthropic

session = AnthropicSession()
client = anthropic.Anthropic(api_key="dummy")

with session.replay("tests/fixtures/claude_agent.jsonl") as probe:
    my_claude_agent.run(client, "summarise this document")

probe.assert_tool_called("read_file")
probe.assert_stop_reason("end_turn")
probe.assert_all_responses_under_tokens(2000)
probe.assert_response_latency_percentile(95, 3000)
```

---

## Async agents

Full `asyncio` support via `async_record`, `async_replay`, `async_auto`:

```python
import pytest
from agentprobe import Session
import openai

@pytest.mark.asyncio
async def test_async_agent():
    session = Session()
    client = openai.AsyncOpenAI(api_key="dummy")

    async with session.async_replay("tests/fixtures/my_agent.jsonl") as probe:
        await my_async_agent.run(client, "list files in /tmp")

    probe.assert_tool_called("bash")
    probe.assert_max_iterations(3)
```

---

## Assertions

All assertion methods return `self` for chaining. Methods that have no data to check (e.g. no recorded durations) skip silently rather than fail.

### Iterations

```python
probe.assert_max_iterations(n)        # at most n LLM calls
probe.assert_min_iterations(n)        # at least n LLM calls
probe.assert_iteration_count(n)       # exactly n LLM calls
```

### Tool calls

```python
probe.assert_tool_called("name")                   # called at least once
probe.assert_not_tool_called("name")               # never called
probe.assert_no_tool_calls()                       # zero tool calls in session
probe.assert_tool_called_with("name", key=val)     # at least one call had these inputs
probe.assert_tool_never_called_with("name", key=val)
probe.assert_tool_call_count("name", n)            # called exactly n times
probe.assert_min_tool_calls(n)                     # at least n calls total
probe.assert_max_tool_calls(n)                     # at most n calls total
probe.assert_tool_called_before("a", "b")          # a was called before b
probe.assert_tool_call_order("a", "b", "c")        # tools in this exact order
probe.assert_tool_sequence("a", "b")               # contiguous subsequence
probe.assert_tool_called_before_output()           # a tool ran before final text
probe.assert_final_tool_not_called("name")         # last call didn't invoke this
probe.assert_tool_call_count_per_call("name", n)   # at most n uses per LLM call
probe.assert_no_tool_call_cycles()                 # no tool called twice in a row
probe.assert_no_duplicate_tool_calls()             # no identical tool+args repeated
probe.assert_no_empty_tool_inputs()                # no tool called with empty {}
probe.assert_tool_inputs_unique("name")            # no two calls with same inputs
probe.assert_tool_input_contains("name", key, val) # at least one call had input[key]==val
probe.assert_tool_input_schema("name", schema)     # inputs match JSON Schema dict
probe.assert_tool_call_args_match("name", regex)   # serialized inputs match pattern
probe.assert_tool_arg_type("name", key, "int")     # type check: "str"/"int"/"float"/"bool"/"list"/"dict"/"null"
probe.assert_tool_result_contains("name", text)    # a result fed back for this tool contains text
probe.assert_no_hallucinated_tool_calls("a", "b")  # only listed tools were called
probe.assert_no_pii_in_tool_inputs(r"\b\d{3}-\d{2}-\d{4}\b")  # no regex match in inputs
```

### Output

```python
probe.assert_output_contains(text)             # final response contains text
probe.assert_output_not_contains(text)         # final response does not contain text
probe.assert_output_contains_all("a", "b")     # final response contains all of these
probe.assert_all_outputs_contain(text)         # every LLM response contains text
probe.assert_output_not_empty()                # final output is not None/blank
probe.assert_output_matches(r"pattern")        # final output matches regex
probe.assert_output_word_count(min=5, max=500) # word count in range
probe.assert_output_char_count(min=10, max=2000) # character count in range
probe.assert_output_language("en")             # ISO 639-1 language code (requires langdetect)
probe.assert_response_format("json")           # "json" or "markdown"
probe.assert_no_empty_responses()             # no call returned empty text
probe.assert_stop_reason("stop")               # final stop reason
probe.assert_finish_reason_all("stop")         # all calls ended with this reason
```

### Latency

```python
probe.assert_response_time_under(ms)                 # every call under ms
probe.assert_max_duration_ms(ms)                     # total session under ms
probe.assert_average_latency_under(ms)               # mean call latency under ms
probe.assert_first_response_latency_under(ms)        # cold-start (first call) under ms
probe.assert_response_latency_percentile(95, 3000)   # p95 under 3 s
```

### Tokens and cost

```python
probe.assert_max_tokens(n)                  # total tokens ≤ n
probe.assert_max_cost(0.05)                 # total estimated cost ≤ $0.05
probe.assert_cost_per_call(0.01)            # average cost per call ≤ $0.01
probe.assert_all_responses_under_tokens(n)  # every individual call < n tokens
probe.assert_token_ratio(n, max_ratio)      # call n used ≤ max_ratio × call n-1 tokens
```

### Models

```python
probe.assert_model_used("gpt-4o")
probe.assert_all_models_in("gpt-4o", "gpt-4o-mini")
```

### Context and messages

```python
probe.assert_context_growth(max_ratio)          # total tokens grew ≤ max_ratio across session
probe.assert_prompt_growth_bounded(max_ratio)   # no single step grew prompt > max_ratio
probe.assert_messages_count(n)                  # session sent exactly n messages total
probe.assert_no_repeated_messages()             # no consecutive call repeated last user message
probe.assert_no_sensitive_in_messages(r"sk-.*") # no outgoing message matches regex
probe.assert_system_prompt_present()            # at least one call had a system prompt
probe.assert_no_empty_system_prompt()           # no call used an empty system prompt
probe.assert_all_calls_consumed()               # all fixture calls were used
```

### Introspection properties

```python
probe.iteration_count       # int   — number of LLM calls made
probe.tools_called          # list  — sorted unique tool names used
probe.final_output          # str | None — last text block in session
probe.total_tokens          # int   — input + output tokens across all calls
probe.total_input_tokens    # int
probe.total_output_tokens   # int
```

---

## CLI reference

Inspect, compare, and manage fixture files without writing Python.

### `show` — inspect a fixture

```bash
agentprobe show tests/fixtures/my_agent.jsonl
agentprobe show --tool-results tests/fixtures/my_agent.jsonl   # include tool results
agentprobe show --calls 2 tests/fixtures/my_agent.jsonl        # first 2 calls only
agentprobe show --calls -1 tests/fixtures/my_agent.jsonl       # last call only
agentprobe show --model gpt-4o tests/fixtures/my_agent.jsonl   # filter by model
agentprobe show --json tests/fixtures/my_agent.jsonl           # machine-readable JSON
```

```
fixture: tests/fixtures/my_agent.jsonl  (2 call(s))

-- Call 1/2  model=gpt-4o  stop=tool_calls  in=50 out=30  312ms
  [tool_call] bash({"command": "ls /tmp"})

-- Call 2/2  model=gpt-4o  stop=stop  in=80 out=25  280ms
  [tool_result] file1.txt
file2.txt
temp.log
  [text] The /tmp directory contains: file1.txt, file2.txt, temp.log

total tokens: 185  (130 in + 55 out)
```

### `diff` — compare two fixtures

```bash
agentprobe diff tests/fixtures/v1.jsonl tests/fixtures/v2.jsonl
agentprobe diff --by-call tests/fixtures/v1.jsonl tests/fixtures/v2.jsonl
agentprobe compare tests/fixtures/v1.jsonl tests/fixtures/v2.jsonl   # similarity score 0–100
```

### `stats` — aggregate stats across a directory

```bash
agentprobe stats tests/fixtures/
agentprobe stats --by-model tests/fixtures/
agentprobe stats --by-date tests/fixtures/
agentprobe stats --latency-percentile 95 tests/fixtures/
```

### `fixtures` — manage fixture files

```bash
agentprobe fixtures tests/fixtures/                          # list all fixtures
agentprobe fixtures --summarize tests/fixtures/              # one-line summary per fixture
agentprobe fixtures --tool-names tests/fixtures/             # all unique tool names used
agentprobe fixtures --by-token-count tests/fixtures/         # sorted by token usage
agentprobe fixtures --label smoke tests/fixtures/            # filter by label
agentprobe fixtures --age-days 30 tests/fixtures/            # fixtures older than 30 days
agentprobe fixtures --count tests/fixtures/                  # count only
agentprobe fixtures --orphaned tests/fixtures/               # not referenced in any test
agentprobe fixtures --delete-old 90 --confirm tests/fixtures/  # delete fixtures > 90 days old
```

### `migrate` — transform a fixture

```bash
agentprobe migrate in.jsonl out.jsonl --rename-model gpt-4=gpt-4o
agentprobe migrate in.jsonl out.jsonl --rename-tool old_name=new_name
agentprobe migrate in.jsonl out.jsonl --strip-pii '\b[\w.+-]+@[\w-]+\.[\w.]+\b'
```

### `record` — capture a script's API calls

```bash
agentprobe record my_agent.py tests/fixtures/output.jsonl
agentprobe record-watch my_agent.py tests/fixtures/output.jsonl   # re-record on file change
```

---

## Fixture format

Fixtures are newline-delimited JSON (`.jsonl`) — one line per API call, preceded by a `_meta` header:

```
{"_meta": {"agentprobe_version": "0.22.0", "recorded_at": "2026-06-03T10:00:00Z"}}
{"request": {...}, "response": {...}, "timestamp": 1748700000.0, "duration_ms": 312.5}
{"request": {...}, "response": {...}, "timestamp": 1748700001.0, "duration_ms": 280.1}
```

- **Plain text** — safe to commit, review in PRs, and edit by hand
- **Gzip supported** — rename to `.jsonl.gz`; agentprobe reads both transparently
- **Provider-agnostic** — OpenAI and Anthropic fixtures use the same format

---

## License

MIT © [Ashutosh Rath](https://github.com/ashutosh-rath02)
