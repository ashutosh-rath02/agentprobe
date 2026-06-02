# agentprobe — Project Roadmap

## Status: v0.4.0 (unreleased)

---

## Done

### Core library
- [x] `Session.record(path)` — intercepts `openai.OpenAI.chat.completions.create`, saves to JSONL
- [x] `Session.replay(path)` — replays fixture, zero API cost, deterministic
- [x] `Session.auto(path)` — record on first run, replay thereafter
- [x] `Session.inject(*responses)` — ad-hoc replay from dicts/objects, no file needed
- [x] `Session.inject_error(exception)` — inject API errors for testing error paths
- [x] Async parity: `Session.async_record / async_replay / async_auto` for `AsyncOpenAI`
- [x] Deep serialization of nested Pydantic objects in request kwargs
- [x] **OpenAI adapter** — patches `openai.resources.chat.completions.Completions.create`
- [x] **Streaming support** — `MockStream` / `MockAsyncStream` for `stream=True` (sync + async)
- [x] **Streaming cost** — `stream_options={"include_usage": True}` usage captured correctly
- [x] **`MultiSession`** — per-client instance-level patching for multi-agent / subagent patterns
- [x] **Gzip fixtures** — `.jsonl.gz` read/write transparent; saves ~60-70% disk space
- [x] **xdist file-lock** — `FileLock` prevents concurrent workers from corrupting fixtures

### AssertionProxy — full assertion DSL (all chainable)
- [x] `assert_tool_called(name)` / `assert_not_tool_called(name)` / `assert_no_tool_calls()`
- [x] `assert_tool_called_with(name, **input_kwargs)` / `assert_tool_called_before(first, second)`
- [x] `assert_tool_call_count(name, n)`
- [x] `assert_max_iterations(n)` / `assert_min_iterations(n)` / `assert_iteration_count(n)`
- [x] `assert_output_contains(text)` / `assert_output_not_contains(text)` / `assert_all_outputs_contain(text)`
- [x] `assert_output_is_json()` / `assert_output_json_contains(**kv)`
- [x] `assert_stop_reason(reason)` / `assert_max_tokens(n)` / `assert_max_cost(usd)`
- [x] `assert_model_used(model)` / `assert_max_duration_ms(ms)`
- [x] `call(n)` — scoped proxy for the nth call
- [x] Properties: `iteration_count`, `tools_called`, `first_tool_called`, `last_tool_called`,
  `final_output`, `total_tokens`, `total_input_tokens`, `total_output_tokens`,
  `estimated_cost_usd`, `total_duration_ms`, `call_log`, `models_used`

### pytest integration
- [x] `agentprobe` fixture / `agentprobe_multi` fixture
- [x] `pytest --agentprobe-update` option
- [x] `asyncio_mode = "auto"` / `pytest-asyncio` as hard dependency
- [x] pytest-xdist detection warning

### CLI (`agentprobe`)
- [x] `agentprobe show <fixture>` / `agentprobe show --json`
- [x] `agentprobe diff <a> <b>` / `agentprobe diff --json` (incl. content + tool arg diffs)
- [x] `agentprobe record <script.py>` — sync + async script support
- [x] `agentprobe validate <fixture>` — Pydantic-level deserialization check
- [x] `agentprobe init` / `agentprobe fixtures [dir]`

### Infrastructure
- [x] `py.typed` marker (PEP 561)
- [x] GitHub Actions CI (Python 3.9–3.13) + PyPI publish on tag
- [x] Pricing table: all current OpenAI models + versioned IDs
- [x] CHANGELOG.md / `.gitignore`
- [x] `pip install pytest-agentprobe[xdist]` for file-lock dep

### Tests
- [x] 156 tests, 0 failures
- [x] Full coverage: inject/inject_error, streaming, multi-agent, gzip, xdist lock,
  async record, pricing, JSON output, call(n), Pydantic validate, diff content

---

## Next — v0.5.0

### High priority
- [ ] **pytest-xdist full worker isolation** — beyond file-lock: fixture namespacing per
  worker so parallel recordings don't even attempt the same file
- [ ] **`probe.assert_response_contains_json` on arbitrary call** — currently only works
  on `final_output`; add per-call JSON assertions via `probe.call(n).assert_output_is_json()`
- [ ] **Fixture migration CLI** — `agentprobe migrate <old.jsonl> <new.jsonl>` to re-schema
  fixtures when the agent's tool schema changes

### Medium priority
- [ ] **`probe.messages_sent`** — full list of messages sent across all calls
- [ ] **`agentprobe show --json` stats** — total token cost, total duration in the summary
- [ ] **Response latency histogram** — `probe.duration_percentile(p)` for p50/p95
- [ ] **`Session.record` context without `as`** — `with session.record(path):` (no proxy)

### Lower priority
- [ ] **VS Code extension** — inline fixture viewer
- [ ] **Fixture linting** — warn on fixtures with no recorded `duration_ms` (replay-only)
- [ ] **`agentprobe record --env <file>`** — load a .env file before running the script

---

## Publish checklist (v0.4.0)
- [ ] `git tag v0.4.0 && git push --tags`
- [ ] GitHub release with CHANGELOG excerpt
- [ ] Submit to [awesome-pytest](https://github.com/augustogoulart/awesome-pytest)
