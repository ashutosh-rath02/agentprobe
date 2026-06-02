# agentprobe — Project Roadmap

## Status: v0.2.0 (unreleased)

---

## Done

### Core library
- [x] `Session.record(path)` — intercepts `openai.OpenAI.chat.completions.create`, saves to JSONL
- [x] `Session.replay(path)` — replays fixture, zero API cost, deterministic
- [x] `Session.auto(path)` — record on first run, replay thereafter
- [x] Async parity: `Session.async_record / async_replay / async_auto` for `AsyncOpenAI`
- [x] Deep serialization of nested Pydantic objects in request kwargs
- [x] **OpenAI adapter** — patches `openai.resources.chat.completions.Completions.create`
- [x] **Streaming support** — `MockStream` / `MockAsyncStream` for `stream=True` (sync + async)
- [x] **`MultiSession`** — per-client instance-level patching for multi-agent / subagent patterns

### AssertionProxy — full assertion DSL (all chainable)
- [x] `assert_tool_called(name)` / `assert_not_tool_called(name)`
- [x] `assert_tool_called_with(name, **input_kwargs)` / `assert_tool_called_before(first, second)`
- [x] `assert_tool_call_count(name, n)`
- [x] `assert_max_iterations(n)` / `assert_min_iterations(n)` / `assert_iteration_count(n)`
- [x] `assert_output_contains(text)` / `assert_output_not_contains(text)` / `assert_all_outputs_contain(text)`
- [x] `assert_stop_reason(reason)` / `assert_max_tokens(n)` / `assert_max_cost(usd)`
- [x] Properties: `iteration_count`, `tools_called`, `final_output`, `total_tokens`,
  `total_input_tokens`, `total_output_tokens`, `estimated_cost_usd`

### pytest integration
- [x] `agentprobe` fixture — auto-registered via `entry_points["pytest11"]`
- [x] `agentprobe_multi` fixture — convenience fixture for `MultiSession`
- [x] `asyncio_mode = "auto"` in `pyproject.toml`
- [x] `pytest --agentprobe-update` option — force re-record from the CLI
- [x] `pytest-asyncio` promoted to a required dependency

### CLI (`agentprobe`)
- [x] `agentprobe show <fixture.jsonl>` — pretty-print calls, tools, tokens
- [x] `agentprobe show --json <fixture.jsonl>` — machine-readable JSON output
- [x] `agentprobe diff <a.jsonl> <b.jsonl>` — compare two fixtures, exit 1 on differences
- [x] `agentprobe record <script.py> [output.jsonl]` — run a script, capture all OpenAI calls
- [x] `agentprobe validate <fixture.jsonl>` — validate fixture format and structure
- [x] `agentprobe init` — scaffold `tests/fixtures/` and a sample `conftest.py`

### Fixture update mode
- [x] `AGENTPROBE_UPDATE=1` / `pytest --agentprobe-update` — force re-record

### Cost estimation
- [x] `probe.estimated_cost_usd` — estimated USD cost based on OpenAI model pricing
- [x] `assert_max_cost(usd)` — assert total cost stays under a threshold

### Tests
- [x] 75 tests, 0 failures
- [x] Covers: sync/async replay, streaming, multi-agent, CLI record, show --json,
  record round-trip, auto mode, all assertions, error cases

### Infrastructure
- [x] `pyproject.toml` with classifiers, URLs, hard deps (openai, pytest, pytest-asyncio)
- [x] GitHub Actions CI: Python 3.9–3.13 matrix
- [x] CHANGELOG.md
- [x] `.gitignore`

---

## Next — v0.3.0

### High priority
- [ ] **pytest-xdist compatibility** — class-level Session patching conflicts with
  parallel test workers; add a warning or switch to per-worker fixture isolation
- [ ] **`agentprobe record` async scripts** — detect `asyncio.run()` entry point
  and handle async scripts transparently
- [ ] **Structured fixture schema** — JSON Schema for fixture validation; stricter
  `agentprobe validate` with Pydantic deserialization check

### Medium priority
- [ ] **`agentprobe show --json` streaming detail** — include chunk count per call
- [ ] **`probe.call_log`** — expose the full request/response sequence for custom assertions
- [ ] **Cost estimation for streaming** — track `stream_options={"include_usage": True}`
- [ ] **`agentprobe diff --json`** — machine-readable diff output

### Lower priority
- [ ] **VS Code extension** — inline fixture viewer in the editor
- [ ] **PyPI publish automation** — GitHub Action to publish on tag push

---

## Publish checklist (v0.2.0)
- [ ] Rotate PyPI token
- [ ] Tag release: `git tag v0.2.0 && git push --tags`
- [ ] Add GitHub release with CHANGELOG excerpt
- [ ] Submit to [awesome-pytest](https://github.com/augustogoulart/awesome-pytest)
- [ ] Post to r/Python and Hacker News "Show HN"
