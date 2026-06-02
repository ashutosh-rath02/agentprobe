# agentprobe — Project Roadmap

## Status: v0.8.0

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

## Next — v0.9.0

### High priority
- [ ] **Anthropic Claude support** — patch `anthropic.Anthropic().messages.create` (sync + async); dual-provider parity so the same fixture format works for both OpenAI and Claude agents
- [ ] **`probe.assert_no_duplicate_tool_calls()`** — detect when an agent calls the same tool with identical args twice (hallucination / loop signal)
- [ ] **`probe.assert_context_growth(max_ratio)`** — assert `prompt_tokens[n] / prompt_tokens[n-1] <= max_ratio` across calls; catches unbounded context accumulation

### Medium priority
- [ ] **`agentprobe record --capture-stdout`** — capture script stdout/stderr into `_meta.stdout` in the fixture; useful for debugging agent log output
- [ ] **`agentprobe replay <fixture> <script>`** — CLI command to run a script in pure replay mode without pytest; developer convenience for debugging
- [ ] **`probe.call(n).tool_call_inputs`** — expose `tool_call_inputs` on the scoped per-call proxy (currently only on the full session proxy)

### Lower priority
- [ ] **`agentprobe record --watch` with real file watcher** — swap polling for `watchdog` event-based re-recording
- [ ] **VS Code extension** — inline fixture viewer

---

## Publish checklist (v0.8.0)
- [ ] `git tag v0.8.0 && git push --tags`
- [ ] GitHub release with CHANGELOG excerpt
