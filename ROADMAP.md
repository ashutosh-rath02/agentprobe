# agentprobe — Project Roadmap

## Status: v0.1.0 published to PyPI as `pytest-agentprobe`

---

## Done

### Core library
- [x] `Session.record(path)` — intercepts `openai.OpenAI.chat.completions.create`, saves to JSONL
- [x] `Session.replay(path)` — replays fixture, zero API cost, deterministic
- [x] `Session.auto(path)` — record on first run, replay thereafter
- [x] Async parity: `Session.async_record / async_replay / async_auto` for `AsyncOpenAI`
- [x] Deep serialization of nested Pydantic objects in request kwargs
- [x] **OpenAI adapter** — patches `openai.resources.chat.completions.Completions.create` (sync + async)

### AssertionProxy — full assertion DSL (all chainable)
- [x] `assert_tool_called(name)`
- [x] `assert_not_tool_called(name)`
- [x] `assert_tool_called_with(name, **input_kwargs)`
- [x] `assert_tool_called_before(first, second)`
- [x] `assert_tool_call_count(name, n)`
- [x] `assert_max_iterations(n)` / `assert_min_iterations(n)` / `assert_iteration_count(n)`
- [x] `assert_output_contains(text)` / `assert_output_not_contains(text)` / `assert_all_outputs_contain(text)`
- [x] `assert_stop_reason(reason)`
- [x] `assert_max_tokens(n)` / `assert_max_cost(usd)`
- [x] Properties: `iteration_count`, `tools_called`, `final_output`, `total_tokens`, `total_input_tokens`, `total_output_tokens`, `estimated_cost_usd`

### pytest integration
- [x] `agentprobe` fixture auto-registered via `entry_points["pytest11"]`
- [x] `asyncio_mode = "auto"` in `pyproject.toml`

### CLI (`agentprobe`)
- [x] `agentprobe show <fixture.jsonl>` — pretty-print calls, tools, tokens
- [x] `agentprobe diff <a.jsonl> <b.jsonl>` — compare two fixtures, exit 1 on differences

### Fixture update mode
- [x] `AGENTPROBE_UPDATE=1` — forces re-record over an existing fixture (like Jest `--updateSnapshot`)

### Cost estimation
- [x] `probe.estimated_cost_usd` — estimated USD cost based on OpenAI model pricing table
- [x] `assert_max_cost(usd)` — assert total cost stays under a threshold

### Tests
- [x] 44 tests, 0 failures
- [x] Covers: sync replay, async replay, record round-trip, auto mode, all assertions, CLI, error cases

### Infrastructure
- [x] `pyproject.toml` with classifiers, URLs, `[async]` optional extra
- [x] GitHub repo: https://github.com/ashutosh-rath02/agentprobe
- [x] GitHub Actions CI: Python 3.9–3.12 matrix
- [x] Published to PyPI: `pip install pytest-agentprobe`

---

## Next — v0.2.0

### High priority
- [ ] **Streaming support** — record/replay `stream=True` calls (`ChatCompletionStream`)
- [ ] **`agentprobe record <script.py>`** — CLI record mode: run a script and capture all calls automatically without modifying agent code

### Medium priority
- [ ] **Multi-agent support** — record/replay sessions with multiple OpenAI clients active simultaneously (subagent patterns)
- [ ] **`agentprobe show --json`** — machine-readable output for tooling

### Lower priority
- [ ] **VS Code extension** — inline fixture viewer
- [ ] **`agentprobe init`** — scaffold `tests/fixtures/` directory with example
- [ ] **pytest-asyncio as hard dep** — currently optional; consider making it required for simplicity

---

## Publish checklist (next release)
- [ ] Rotate PyPI token
- [ ] Tag release in git: `git tag v0.2.0 && git push --tags`
- [ ] Add GitHub release with changelog
- [ ] Submit to [awesome-pytest](https://github.com/augustogoulart/awesome-pytest)
- [ ] Post to r/Python and Hacker News "Show HN"
