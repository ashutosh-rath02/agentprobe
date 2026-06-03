# agentprobe

**Deterministic record-and-replay test harness for AI agents — OpenAI and Anthropic.**

Record your agent's API calls once. Replay them forever — no network, no cost, no flakiness.

```python
def test_agent_uses_bash(agentprobe):
    with agentprobe.replay("tests/fixtures/list_files.jsonl") as probe:
        my_agent.run(client, "list files in /tmp")

    probe.assert_tool_called("bash")
    probe.assert_tool_called_with("bash", command="ls /tmp")
    probe.assert_max_iterations(3)
    probe.assert_output_contains("/tmp")
    probe.assert_average_latency_under(500)
```

---

## Install

```bash
pip install pytest-agentprobe
```

Requires Python 3.9+ and `openai>=1.0.0`. For Anthropic support, also install `anthropic`.

---

## How it works

`agentprobe` patches `openai.ChatCompletions.create` (or `anthropic.messages.create`) at the class level — no changes to your agent code required.

| Mode | What it does |
|---|---|
| **record** | Runs your agent against the real API and saves every request/response pair to a `.jsonl` fixture |
| **replay** | Feeds saved responses back to the agent — instant, free, deterministic |
| **auto** | Records on first run; replays on every subsequent run |
| **inject** | Supplies hand-crafted response objects without touching the filesystem — ideal for unit tests |

---

## Quick start

### Record a session

```python
from agentprobe import Session
import openai

session = Session()
client = openai.OpenAI()  # uses OPENAI_API_KEY

with session.record("tests/fixtures/my_agent.jsonl") as probe:
    my_agent.run(client, "what files are in /tmp?")
    probe.assert_tool_called("bash")  # optional during recording

# fixture is written to disk — commit it to your repo
```

### Replay in CI (zero API calls)

```python
def test_my_agent(agentprobe):
    client = openai.OpenAI(api_key="dummy")  # not used during replay

    with agentprobe.replay("tests/fixtures/my_agent.jsonl") as probe:
        my_agent.run(client, "what files are in /tmp?")

    probe.assert_tool_called("bash")
    probe.assert_not_tool_called("web_search")
    probe.assert_tool_called_with("bash", command="ls /tmp")
    probe.assert_max_iterations(4)
    probe.assert_output_contains("/tmp")
    probe.assert_stop_reason("stop")
    probe.assert_max_tokens(500)
    probe.assert_average_latency_under(600)
```

### Auto mode (record once, replay always)

```python
def test_my_agent(agentprobe):
    with agentprobe.auto("tests/fixtures/my_agent.jsonl") as probe:
        my_agent.run(client, "what files are in /tmp?")
        probe.assert_tool_called("bash")
```

### Inject (unit tests without fixture files)

```python
def test_tool_type_check():
    session = Session()
    client = openai.OpenAI(api_key="dummy")

    mock_response = {...}  # raw OpenAI response dict

    with session.inject(mock_response) as probe:
        client.chat.completions.create(model="gpt-4o", messages=[...])

    probe.assert_tool_arg_type("search", "page", "int")
    probe.assert_output_not_empty()
```

---

## Anthropic support

Use `AnthropicSession` and `AnthropicAssertionProxy` for Claude agents. The API is identical:

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

Async Claude agents use `async_replay`, `async_record`, `async_auto` on `AnthropicSession`.

---

## Assertion API

All assertions return `self` for chaining. Assertions that have no recorded data to check skip silently rather than fail.

### Iteration control

| Assertion | Description |
|---|---|
| `assert_max_iterations(n)` | At most *n* LLM calls |
| `assert_min_iterations(n)` | At least *n* LLM calls |
| `assert_iteration_count(n)` | Exactly *n* LLM calls |

### Tool calls

| Assertion | Description |
|---|---|
| `assert_tool_called(name)` | Tool was called at least once |
| `assert_not_tool_called(name)` | Tool was never called |
| `assert_no_tool_calls()` | No tool calls in the entire session |
| `assert_tool_called_with(name, **kw)` | At least one call had these input fields |
| `assert_tool_never_called_with(name, **kw)` | No call had these input fields |
| `assert_tool_call_count(name, n)` | Tool called exactly *n* times across session |
| `assert_tool_called_n_times(name, n)` | Alias for above |
| `assert_min_tool_calls(n)` | At least *n* total tool invocations across all calls |
| `assert_max_tool_calls(n)` | At most *n* total tool invocations across all calls |
| `assert_tool_called_before(first, second)` | *first* was called before *second* |
| `assert_tool_call_order(*names)` | Tools appear in exactly this order |
| `assert_tool_sequence(*names)` | Contiguous subsequence of tool calls matches |
| `assert_tool_called_before_output()` | At least one tool was called before the final text response |
| `assert_final_tool_not_called(name)` | Last call did not invoke this tool |
| `assert_tool_call_count_per_call(name, n)` | Each LLM call invokes the tool at most *n* times |
| `assert_no_tool_call_cycles()` | No tool is called twice in a row in the same call |
| `assert_no_duplicate_tool_calls()` | No identical tool+arguments combination repeated |
| `assert_no_empty_tool_inputs()` | No tool was called with an entirely empty input dict |
| `assert_tool_inputs_unique(name)` | No two calls to *name* used the same serialized inputs |
| `assert_tool_input_contains(name, key, value)` | At least one call had `input[key] == value` |
| `assert_tool_input_schema(name, schema)` | All inputs match a JSON Schema dict |
| `assert_tool_call_args_match(name, pattern)` | Serialized inputs match regex *pattern* |
| `assert_tool_arg_type(name, key, type)` | All calls have `input[key]` of type `"str"/"int"/"float"/"bool"/"list"/"dict"/"null"` |
| `assert_tool_result_contains(name, text)` | At least one result fed back for *name* contains *text* |
| `assert_no_hallucinated_tool_calls(*allowed)` | Only tools in *allowed* were called |
| `assert_no_pii_in_tool_inputs(*patterns)` | No tool input matches any regex pattern |

### Output

| Assertion | Description |
|---|---|
| `assert_output_contains(text)` | Final text response contains *text* |
| `assert_output_not_contains(text)` | Final text response does not contain *text* |
| `assert_output_contains_all(*texts)` | Final text contains all of *texts* |
| `assert_all_outputs_contain(text)` | Every LLM response contains *text* |
| `assert_output_not_empty()` | Final output is not None, empty, or whitespace |
| `assert_output_matches(pattern)` | Final output matches regex *pattern* |
| `assert_output_word_count(min, max)` | Final output word count in `[min, max]` |
| `assert_output_char_count(min, max)` | Final output character count in `[min, max]` |
| `assert_output_language(lang)` | Final output is in ISO 639-1 language code *lang* (requires `langdetect`) |
| `assert_response_format(fmt)` | Final output format: `"json"` or `"markdown"` |
| `assert_no_empty_responses()` | No call returned an empty text response |
| `assert_stop_reason(reason)` | Final call stop reason equals *reason* |
| `assert_finish_reason_all(reason)` | All calls ended with *reason* |
| `assert_response_json_at(n, **kw)` | Call *n*'s output parses as JSON with these fields |

### Latency

| Assertion | Description |
|---|---|
| `assert_response_time_under(ms)` | Every call completed within *ms* milliseconds |
| `assert_max_duration_ms(ms)` | Total session duration under *ms* milliseconds |
| `assert_average_latency_under(ms)` | Mean call latency under *ms* milliseconds |
| `assert_first_response_latency_under(ms)` | First call (cold-start) under *ms* milliseconds |
| `assert_response_latency_percentile(p, ms)` | p*N* latency (0–100) under *ms* milliseconds |

### Tokens and cost

| Assertion | Description |
|---|---|
| `assert_max_tokens(n)` | Total tokens across all calls ≤ *n* |
| `assert_max_cost(usd)` | Total estimated cost ≤ *usd* |
| `assert_cost_per_call(usd)` | Average cost per call ≤ *usd* |
| `assert_all_responses_under_tokens(n)` | Every individual call used fewer than *n* tokens |
| `assert_token_ratio(n, max_ratio)` | Call *n* used ≤ `max_ratio × call_(n-1)` tokens |

### Models

| Assertion | Description |
|---|---|
| `assert_model_used(model)` | Every call used this model |
| `assert_all_models_in(*allowed)` | Only models in *allowed* were used |

### Context and messages

| Assertion | Description |
|---|---|
| `assert_context_growth(max_ratio)` | Total token count grew by at most `max_ratio` across session |
| `assert_prompt_growth_bounded(max_ratio)` | No single step grew the prompt by more than `max_ratio` |
| `assert_messages_count(n)` | Session sent exactly *n* total messages across all calls |
| `assert_no_repeated_messages()` | No consecutive call repeated the same final user message |
| `assert_no_sensitive_in_messages(*patterns)` | No outgoing message matches any regex pattern |
| `assert_system_prompt_present()` | At least one call included a system prompt |
| `assert_no_empty_system_prompt()` | No call used an empty system prompt |
| `assert_all_calls_consumed()` | All fixture calls were replayed (no calls skipped) |

### Introspection

```python
probe.iteration_count       # int  — number of LLM calls made
probe.tools_called          # list[str] — sorted unique tool names used
probe.final_output          # str | None — last text block in session
probe.total_tokens          # int — input + output tokens across all calls
probe.total_input_tokens    # int
probe.total_output_tokens   # int
```

---

## Async agents

Full async/await support via `AsyncOpenAI` or `AsyncAnthropic`:

```python
import pytest
from agentprobe import Session

@pytest.mark.asyncio
async def test_async_agent():
    session = Session()
    client = openai.AsyncOpenAI(api_key="dummy")

    async with session.async_replay("tests/fixtures/my_agent.jsonl") as probe:
        await my_async_agent.run(client, "list files in /tmp")

    probe.assert_tool_called("bash")
    probe.assert_max_iterations(3)
```

Async equivalents: `async_record`, `async_replay`, `async_auto`.

---

## Multi-session

Test agents that make API calls across multiple fixtures (e.g. multi-step workflows):

```python
from agentprobe import MultiSession

session = MultiSession()
with session.replay_chain(
    "tests/fixtures/step1.jsonl",
    "tests/fixtures/step2.jsonl",
) as probe:
    orchestrator.run(client)

probe.assert_tool_called("fetch_data")
probe.assert_max_iterations(6)
```

---

## CLI

Inspect, compare, and manage fixtures from the command line.

### Inspect fixtures

```bash
# Pretty-print a fixture
agentprobe show tests/fixtures/my_agent.jsonl

# Also print tool results fed back to the model
agentprobe show --tool-results tests/fixtures/my_agent.jsonl

# Show only the first 2 calls (negative = last N)
agentprobe show --calls 2 tests/fixtures/my_agent.jsonl

# Filter by model
agentprobe show --model gpt-4o tests/fixtures/my_agent.jsonl

# Machine-readable JSON
agentprobe show --json tests/fixtures/my_agent.jsonl
```

Example output:

```
fixture: tests/fixtures/my_agent.jsonl  (2 call(s))

-- Call 1/2  model=gpt-4o  stop=tool_calls  in=50 out=30  312ms
  [tool_call] bash({"command": "ls /tmp"})

-- Call 2/2  model=gpt-4o  stop=stop  in=80 out=25  280ms
  [tool_result] file1.txt\nfile2.txt\ntemp.log
  [text] The /tmp directory contains: file1.txt, file2.txt, temp.log

total tokens: 185  (130 in + 55 out)
```

### Compare fixtures

```bash
# Human-readable diff
agentprobe diff tests/fixtures/v1.jsonl tests/fixtures/v2.jsonl

# Side-by-side per-call comparison
agentprobe diff --by-call tests/fixtures/v1.jsonl tests/fixtures/v2.jsonl

# Similarity score (0-100)
agentprobe compare tests/fixtures/v1.jsonl tests/fixtures/v2.jsonl
```

### Aggregate stats

```bash
# Summary across all fixtures in a directory
agentprobe stats tests/fixtures/

# Group by model
agentprobe stats --by-model tests/fixtures/

# Group by recording date
agentprobe stats --by-date tests/fixtures/

# p95 latency across all calls
agentprobe stats --latency-percentile 95 tests/fixtures/
```

### Manage fixtures

```bash
# List all fixtures
agentprobe fixtures tests/fixtures/

# One-line summary per fixture (calls, tokens, tools)
agentprobe fixtures --summarize tests/fixtures/

# List all unique tool names called across fixtures
agentprobe fixtures --tool-names tests/fixtures/

# List fixtures sorted by total token usage
agentprobe fixtures --by-token-count tests/fixtures/

# List fixtures with a specific label
agentprobe fixtures --label smoke tests/fixtures/

# List fixtures older than 30 days
agentprobe fixtures --age-days 30 tests/fixtures/

# Count fixtures
agentprobe fixtures --count tests/fixtures/

# Delete fixtures older than 90 days
agentprobe fixtures --delete-old 90 --confirm tests/fixtures/

# List fixture files not referenced in any test
agentprobe fixtures --orphaned tests/fixtures/
```

### Transform fixtures

```bash
# Rename a model in a fixture
agentprobe migrate input.jsonl output.jsonl --rename-model gpt-4=gpt-4o

# Rename a tool
agentprobe migrate input.jsonl output.jsonl --rename-tool old_name=new_name

# Redact PII from tool inputs
agentprobe migrate input.jsonl output.jsonl --strip-pii '\b[\w.+-]+@[\w-]+\.[\w.]+\b'
```

### Record from a script

```bash
# Record all API calls made by a Python script
agentprobe record my_agent_script.py tests/fixtures/output.jsonl

# Watch a script and re-record on file change
agentprobe record-watch my_agent_script.py tests/fixtures/output.jsonl
```

---

## Fixture format

Fixtures are newline-delimited JSON (`.jsonl`) — one line per API call plus a `_meta` header:

```json
{"_meta": {"agentprobe_version": "0.22.0", "recorded_at": "2026-06-03T10:00:00Z"}}
{"request": {"model": "gpt-4o", "messages": [...], "tools": [...]}, "response": {"id": "...", "choices": [...], "usage": {...}}, "timestamp": 1748700000.0, "duration_ms": 312.5}
```

Fixtures are plain text — safe to commit, diff in PRs, and edit by hand. Compress large fixtures to `.jsonl.gz` — agentprobe reads both transparently.

---

## pytest plugin

`agentprobe` auto-registers as a pytest plugin. No `conftest.py` needed:

```python
def test_something(agentprobe):
    with agentprobe.replay("tests/fixtures/session.jsonl") as probe:
        my_agent.run(client, "do something")
    probe.assert_tool_called("search")
```

To use `Session` directly outside pytest (scripts, notebooks):

```python
from agentprobe import Session

session = Session()
with session.replay("tests/fixtures/session.jsonl") as probe:
    my_agent.run(client, "do something")
probe.assert_tool_called("search")
```

---

## License

MIT
