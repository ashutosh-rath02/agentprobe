# agentprobe — Project Roadmap

## Status: v0.5.0 (unreleased)

---

## Done

### Core library
- [x] `Session.record(path)` — saves to JSONL; atomic write via temp+rename (never partially written)
- [x] `Session.replay(path)` — deterministic replay, zero API cost
- [x] `Session.replay(path, strict=True)` — raises if agent doesn't consume all fixture calls
- [x] `Session.auto(path)` — record-on-first-run, replay thereafter; xdist-safe via FileLock
- [x] `Session.inject(*responses)` — ad-hoc replay without a file
- [x] `Session.inject_error(exception)` — inject API errors
- [x] Async parity: all Session methods have `async_*` variants for `AsyncOpenAI`
- [x] `MultiSession` — per-client instance-level patching for multi-agent patterns
- [x] Gzip fixtures (`.jsonl.gz`) — transparent read/write
- [x] Streaming support (`stream=True`) — sync + async, with `stream_options` usage tracking
- [x] xdist coordination in `auto()` — only one worker records; others replay

### AssertionProxy — full assertion DSL (all chainable)
- [x] Tool: `assert_tool_called` / `assert_not_tool_called` / `assert_no_tool_calls` / `assert_tool_called_with` / `assert_tool_called_before` / `assert_tool_call_count`
- [x] Sequence: `assert_tool_sequence(*names)` — exact ordered tool sequence
- [x] Iteration: `assert_max_iterations` / `assert_min_iterations` / `assert_iteration_count`
- [x] Output: `assert_output_contains` / `assert_output_not_contains` / `assert_all_outputs_contain` / `assert_output_is_json` / `assert_output_json_contains` / `assert_output_matches(regex)`
- [x] Stop: `assert_stop_reason` / `assert_finish_reason_all`
- [x] Tokens: `assert_max_tokens` / `assert_max_cost` / `assert_max_duration_ms`
- [x] Model: `assert_model_used`
- [x] Properties: `iteration_count`, `tools_called`, `first_tool_called`, `last_tool_called`, `final_output`, `total_tokens`, `total_input_tokens`, `total_output_tokens`, `estimated_cost_usd`, `total_duration_ms`, `call_log`, `models_used`, `messages_sent`, `duration_percentile(p)`
- [x] Per-call: `call(n)` — scoped proxy for nth call
- [x] Export: `dump_fixture(path)` — save session to JSONL

### CLI (`agentprobe`)
- [x] `show` / `show --json` (with `calls` + `summary` block, `estimated_cost_usd`)
- [x] `diff` / `diff --json` (content + tool argument diffs)
- [x] `record <script.py> [output] [--env FILE]` — sync + async script support
- [x] `validate <fixture>` — Pydantic deserialization + structural linting (WARN on no duration, no model)
- [x] `init` — scaffolds `tests/fixtures/` + `conftest.py`
- [x] `fixtures [dir]` — list fixtures with stats

### pytest integration
- [x] `agentprobe` / `agentprobe_multi` fixtures
- [x] `pytest --agentprobe-update`
- [x] pytest-xdist detection warning
- [x] `pytest-asyncio` as hard dep

### Infrastructure
- [x] `py.typed` (PEP 561)
- [x] GitHub Actions CI (Python 3.9–3.13) + PyPI OIDC publish
- [x] Pricing: all current OpenAI models + versioned IDs
- [x] CHANGELOG through v0.5.0
- [x] 186 tests, 0 failures

---

## Next — v0.6.0

### High priority
- [ ] **Fixture migration CLI** — `agentprobe migrate <old.jsonl> <new.jsonl>` to update tool schemas or model names across a fixture
- [ ] **`probe.assert_response_json_at(n, **kv)`** — per-call JSON output assertion (currently only on `final_output`)
- [ ] **`agentprobe record --watch`** — re-record automatically when the script file changes (file watcher)

### Medium priority
- [ ] **`probe.messages_received`** — list of assistant messages received across calls
- [ ] **Response cost breakdown per call** — `probe.call(0).estimated_cost_usd`
- [ ] **`agentprobe fixtures stats`** — aggregate token/cost/duration stats across all fixtures in a directory
- [ ] **`agentprobe show --model <name>`** — filter show output to only calls using a specific model

### Lower priority
- [ ] **VS Code extension** — inline fixture viewer
- [ ] **`agentprobe record --output-format gz`** — force gzip output from the CLI
- [ ] **Fixture versioning** — store `agentprobe_version` in fixture metadata for migration hints

---

## Publish checklist (v0.5.0)
- [ ] `git tag v0.5.0 && git push --tags`
- [ ] GitHub release with CHANGELOG excerpt
- [ ] Submit to [awesome-pytest](https://github.com/augustogoulart/awesome-pytest)
