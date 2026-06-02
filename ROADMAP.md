# agentprobe — Project Roadmap

## Status: v0.6.0 (unreleased)

---

## Done

### Core library
- [x] `Session.record / replay / auto` — JSONL fixtures, atomic write, xdist-safe
- [x] `Session.replay(strict=True)` — raises if agent doesn't consume all fixture calls
- [x] `Session.replay_chain(*paths)` — concatenate multiple fixtures in one session
- [x] `Session.inject(*responses)` / `Session.inject_error(exception)`
- [x] Async parity: all Session methods have `async_*` variants
- [x] `MultiSession` — per-client instance-level patching for multi-agent patterns
- [x] Gzip fixtures (`.jsonl.gz`) — transparent read/write
- [x] Streaming support (`stream=True`) — sync + async, usage tracking
- [x] xdist coordination in `auto()` — FileLock + atomic write

### AssertionProxy — complete DSL
- [x] Tool assertions: called / not_called / no_tool_calls / called_with / called_before / call_count / sequence
- [x] Iteration: max / min / exact count
- [x] Output: contains / not_contains / all_contain / is_json / json_contains / matches(regex)
- [x] Stop/finish: assert_stop_reason / assert_finish_reason_all
- [x] Tokens: max_tokens / max_cost / max_duration_ms / token_efficiency
- [x] Model: assert_model_used
- [x] Per-call: `call(n)` scoped proxy
- [x] Export: `dump_fixture(path)` / `summary_dict()`
- [x] Properties: `iteration_count`, `tools_called`, `first/last_tool_called`, `final_output`, `total_tokens`, `total_input/output_tokens`, `estimated_cost_usd`, `total_duration_ms`, `duration_percentile(p)`, `call_log`, `messages_sent`, `messages_received`, `models_used`, `tool_call_inputs`

### CLI — full toolchain
- [x] `show` / `show --json` / `show --model` — pretty-print or JSON with summary block
- [x] `diff` / `diff --json` — content + tool argument diffs
- [x] `record <script> [output] [--env] [--output-format gz] [--watch]`
- [x] `validate` — Pydantic deserialization + structural linting
- [x] `migrate` — rename models/tools across a fixture
- [x] `init` — scaffold project
- [x] `fixtures [dir]` — list fixtures
- [x] `stats [dir]` — aggregate token/cost/duration stats

### pytest integration
- [x] `agentprobe` / `agentprobe_multi` fixtures
- [x] `pytest --agentprobe-update`
- [x] pytest-xdist detection warning
- [x] `pytest-asyncio` as hard dep

### Infrastructure
- [x] `py.typed` (PEP 561)
- [x] GitHub Actions CI (Python 3.9–3.13) + PyPI OIDC publish
- [x] Pricing: all current OpenAI models + versioned IDs
- [x] CHANGELOG through v0.6.0
- [x] 212 tests, 0 failures

---

## Next — v0.7.0

### High priority
- [ ] **`agentprobe record --watch` with real file watcher** — use `watchdog` library for event-based re-recording instead of polling
- [ ] **Per-call cost in `probe.call(n)`** — `probe.call(0).estimated_cost_usd` (already works via scoped proxy — document it)
- [ ] **`probe.assert_response_json_at(n, **kv)`** — per-call JSON assertion without `probe.call(n).assert_output_json_contains(...)`

### Medium priority  
- [ ] **`agentprobe record --timeout N`** — kill script after N seconds (for hung async scripts)
- [ ] **`probe.messages_received[n]` direct access** — `probe.received_at(n)` for cleaner API
- [ ] **`agentprobe fixtures stats --by-date`** — group stats by recording date
- [ ] **Fixture versioning metadata** — optional `_meta` header line with `agentprobe_version` + `recorded_at`

### Lower priority
- [ ] **VS Code extension** — inline fixture viewer
- [ ] **`agentprobe diff --by-call`** — side-by-side per-call comparison view
- [ ] **`agentprobe record --capture-stdout`** — capture script stdout in fixture metadata

---

## Publish checklist (v0.6.0)
- [ ] `git tag v0.6.0 && git push --tags`
- [ ] GitHub release with CHANGELOG excerpt
- [ ] Submit to [awesome-pytest](https://github.com/augustogoulart/awesome-pytest)
