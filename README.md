# agentprobe

**pytest-compatible test harness for AI agents — deterministic record & replay for Anthropic Claude.**

Test your Claude agents in CI without hitting the real API on every run. Record once, replay forever — zero cost, zero flakiness.

```python
def test_agent_uses_bash(agentprobe):
    with agentprobe.replay("tests/fixtures/list_files.jsonl") as probe:
        result = my_agent.run("list files in /tmp")
        probe.assert_tool_called("bash")
        probe.assert_max_iterations(3)
        probe.assert_output_contains("/tmp")
```

---

## Install

```bash
pip install pytest-agentprobe
```

Requires Python 3.9+ and `anthropic>=0.40.0`.

---

## How it works

`agentprobe` intercepts calls to `anthropic.Anthropic.messages.create` (and the async equivalent) at the class level — no changes to your agent code needed.

- **Record mode** — runs your agent against the real API and saves every request/response pair to a JSONL fixture file.
- **Replay mode** — feeds the saved responses back to your agent instead of making real API calls. Deterministic, instant, free.
- **Auto mode** — records on first run, replays on every subsequent run.

---

## Quick start

### 1. Record a session

```python
from agentprobe import Session
import anthropic

session = Session()
client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY

with session.record("tests/fixtures/my_agent.jsonl") as probe:
    result = my_agent.run(client, "what files are in /tmp?")
    # assertions are optional during recording
    probe.assert_tool_called("bash")

# fixture is written to disk — commit it to your repo
```

### 2. Replay in CI

```python
def test_my_agent(agentprobe):
    client = anthropic.Anthropic(api_key="dummy")  # not used during replay

    with agentprobe.replay("tests/fixtures/my_agent.jsonl") as probe:
        result = my_agent.run(client, "what files are in /tmp?")

        probe.assert_tool_called("bash")
        probe.assert_not_tool_called("web_search")
        probe.assert_tool_called_with("bash", command="ls /tmp")
        probe.assert_max_iterations(4)
        probe.assert_output_contains("/tmp")
        probe.assert_stop_reason("end_turn")
        probe.assert_max_tokens(500)
```

### 3. Auto mode (record-on-first-run)

```python
def test_my_agent(agentprobe):
    with agentprobe.auto("tests/fixtures/my_agent.jsonl") as probe:
        result = my_agent.run(client, "what files are in /tmp?")
        probe.assert_tool_called("bash")
```

---

## Async agents

Full async support via `AsyncAnthropic`:

```python
import pytest
import anthropic
from agentprobe import Session

@pytest.mark.asyncio
async def test_async_agent():
    session = Session()
    client = anthropic.AsyncAnthropic(api_key="dummy")

    async with session.async_replay("tests/fixtures/my_agent.jsonl") as probe:
        result = await my_async_agent.run(client, "list files in /tmp")
        probe.assert_tool_called("bash")
        probe.assert_max_iterations(3)
```

Async equivalents: `async_record`, `async_replay`, `async_auto`.

---

## Assertion API

All assertions return `self` for chaining.

### Iteration count

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
| `assert_tool_called_with(name, **kwargs)` | Tool was called with these input fields |
| `assert_tool_called_before(first, second)` | First tool was called before second |

### Output

| Assertion | Description |
|---|---|
| `assert_output_contains(text)` | Final text response contains *text* |
| `assert_output_not_contains(text)` | Final text response does not contain *text* |
| `assert_stop_reason(reason)` | Final call stop reason equals *reason* (e.g. `"end_turn"`) |

### Token budget

| Assertion | Description |
|---|---|
| `assert_max_tokens(n)` | Total tokens across all calls ≤ *n* |

### Introspection

```python
probe.iteration_count       # int — number of LLM calls
probe.tools_called          # list[str] — sorted tool names used
probe.final_output          # str | None — last text block in session
probe.total_tokens          # int — input + output tokens across all calls
probe.total_input_tokens    # int
probe.total_output_tokens   # int
```

---

## CLI

Inspect and compare fixtures without writing Python:

```bash
# Pretty-print a fixture
agentprobe show tests/fixtures/my_agent.jsonl

# Compare two fixtures (exits 1 if differences found)
agentprobe diff tests/fixtures/v1.jsonl tests/fixtures/v2.jsonl
```

Example `show` output:

```
fixture: tests/fixtures/my_agent.jsonl  (2 call(s))

── Call 1/2  model=claude-opus-4-8  stop=tool_use  in=50 out=30  312ms
  [tool_use] bash({"command": "ls /tmp"})

── Call 2/2  model=claude-opus-4-8  stop=end_turn  in=80 out=25  280ms
  [text] The /tmp directory contains: file1.txt, file2.txt, temp.log

total tokens: 185  (130 in + 55 out)
```

---

## pytest fixture

`agentprobe` is auto-registered as a pytest plugin. The `agentprobe` fixture is available in all tests without any `conftest.py` setup:

```python
def test_something(agentprobe):
    with agentprobe.replay("tests/fixtures/session.jsonl") as probe:
        ...
```

To use `Session` directly (e.g. in scripts or non-pytest contexts):

```python
from agentprobe import Session

session = Session()
with session.replay("tests/fixtures/session.jsonl") as probe:
    ...
```

---

## Fixture format

Fixtures are newline-delimited JSON (`.jsonl`). Each line is one `messages.create` call:

```json
{"request": {"model": "...", "messages": [...], "tools": [...]}, "response": {"id": "...", "content": [...], "stop_reason": "tool_use", "usage": {"input_tokens": 50, "output_tokens": 30}}, "timestamp": 1748700000.0, "duration_ms": 312.5}
```

Fixtures are plain text — safe to commit to git, diff in PRs, and edit by hand.

---

## License

MIT
