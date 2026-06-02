# agentprobe — Project Roadmap

## Status: v0.18.0

---

## Done

### Core library
- [x] `Session.record / replay / auto` — JSONL fixtures, atomic write, xdist-safe
- [x] `Session.replay(strict=True)` — raises if agent doesn't consume all fixture calls
- [x] `Session.replay_chain(*paths)` — concatenate multiple fixtures in one session
- [x] `Session.record_append(path)` / `async_record_append(path)` — append calls to existing fixture
- [x] `Session.inject(*responses)` / `Session.inject_error(exception)`
- [x] Async parity: all Session methods have `async_*` variants
- [x] `MultiSession` — per-client instance-level patching for multi-agent patterns
- [x] `MultiSession.replay_chain(*client_path_pairs)` — chained multi-client replay
- [x] Gzip fixtures (`.jsonl.gz`) — transparent read/write
- [x] Streaming support (`stream=True`) — sync + async, usage tracking
- [x] xdist coordination in `auto()` — FileLock + atomic write
- [x] Fixture metadata header (`_meta`) — `agentprobe_version`, `recorded_at`

### AssertionProxy — complete DSL
- [x] Tool assertions: called / not_called / no_tool_calls / called_with / called_before / call_count / sequence
- [x] Iteration: max / min / exact count
- [x] Output: contains / not_contains / all_contain / is_json / json_contains / matches(regex)
- [x] Stop/finish: assert_stop_reason / assert_finish_reason_all
- [x] Tokens: max_tokens / max_cost / max_duration_ms / token_efficiency
- [x] Model: assert_model_used / assert_all_models_in
- [x] Per-call: `call(n)` scoped proxy / `assert_response_json_at(n, **kv)`
- [x] Cost: `assert_cost_per_call(usd)`
- [x] Messages: `assert_messages_count(n)`
- [x] Quality: `assert_no_empty_responses()`
- [x] Regression: `matches_fixture(path, ignore_content=False)`
- [x] Export: `dump_fixture(path)` / `summary_dict()` / `export_json()`
- [x] Properties: `iteration_count`, `tools_called`, `first/last_tool_called`, `final_output`, `total_tokens`, `total_input/output_tokens`, `estimated_cost_usd`, `total_duration_ms`, `duration_percentile(p)`, `call_log`, `messages_sent`, `messages_received`, `models_used`, `tool_call_inputs`

### CLI — full toolchain
- [x] `show` / `show --json` / `show --model`
- [x] `diff` / `diff --json` / `diff --by-call`
- [x] `record <script> [--env] [--output-format gz] [--watch] [--timeout] [--dry-run] [--append]`
- [x] `validate` / `validate --strict`
- [x] `migrate` — rename models/tools across a fixture
- [x] `init` — scaffold project
- [x] `fixtures [dir]` / `fixtures --clean` / `fixtures --by-date`
- [x] `stats [dir]` / `stats --by-date`

### pytest integration
- [x] `agentprobe` / `agentprobe_multi` fixtures
- [x] `pytest --agentprobe-update`
- [x] pytest-xdist detection warning
- [x] `pytest-asyncio` as hard dep

### Infrastructure
- [x] `py.typed` (PEP 561)
- [x] GitHub Actions CI (Python 3.9–3.13) + PyPI OIDC publish
- [x] Pricing: all current OpenAI models + versioned IDs

---

## Done (v0.9.0)
- [x] **Anthropic Claude support** — `AnthropicSession` + `AnthropicAssertionProxy`; full record/replay/inject parity
- [x] **`probe.assert_no_duplicate_tool_calls()`** — OpenAI and Anthropic
- [x] **`probe.assert_context_growth(max_ratio)`** — OpenAI and Anthropic

---

## Done (v0.11.0)
- [x] Anthropic streaming (`create(stream=True)`) — record/replay, `MockAnthropicStream`, event assembly/synthesis
- [x] `agentprobe show` + `show --json` Anthropic support
- [x] `agentprobe diff` + `diff --json` Anthropic support
- [x] `probe.assert_response_time_under(ms)` (OpenAI + Anthropic)
- [x] `probe.assert_tool_input_schema(name, schema)` (OpenAI + Anthropic)

---

## Done (v0.18.0)
- [x] `assert_first_response_latency_under`, `assert_output_contains_all`, `assert_tool_call_args_match`
- [x] `fixtures --age-days N`, `migrate --strip-pii PATTERN`

---

## Next — v0.19.0

### High priority
- [ ] **`probe.assert_response_format(format)`** — assert final output matches a named format (e.g. `"json"`, `"markdown"`)
- [ ] **`agentprobe fixtures --delete-old N`** — delete fixtures older than N days (destructive, requires `--confirm`)
- [ ] **`probe.assert_tool_never_called_with(name, **kv)`** — negative input assertion

### Medium priority
- [ ] **`messages.stream()` recording support** — `MessageStreamManager` higher-level API
- [ ] **`probe.assert_prompt_growth_bounded(max_ratio)`** — assert each call's prompt tokens are < max_ratio × previous
- [ ] **`agentprobe show --calls N`** — show only the first/last N calls

### Lower priority
- [ ] **`agentprobe record --watch` real file watcher** — swap polling for `watchdog`
- [ ] **VS Code extension** — inline fixture viewer

---

## Publish checklist (v0.18.0)
- [ ] `git tag v0.18.0 && git push --tags`
- [ ] GitHub release with CHANGELOG excerpt
