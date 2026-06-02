# agentprobe — Project Roadmap

## Status: v0.3.0 (unreleased)

---

## Done

### Core library
- [x] `Session.record(path)` — intercepts `openai.OpenAI.chat.completions.create`, saves to JSONL
- [x] `Session.replay(path)` — replays fixture, zero API cost, deterministic
- [x] `Session.auto(path)` — record on first run, replay thereafter
- [x] `Session.inject(*responses)` — ad-hoc replay without a fixture file
- [x] `Session.inject_error(exception)` — inject API errors for testing error paths
- [x] Async parity: `Session.async_record / async_replay / async_auto` for `AsyncOpenAI`
- [x] Deep serialization of nested Pydantic objects in request kwargs
- [x] **OpenAI adapter** — patches `openai.resources.chat.completions.Completions.create`
- [x] **Streaming support** — `MockStream` / `MockAsyncStream` for `stream=True` (sync + async)
- [x] **`MultiSession`** — per-client instance-level patching for multi-agent / subagent patterns

### AssertionProxy — full assertion DSL (all chainable)
- [x] `assert_tool_called(name)` / `assert_not_tool_called(name)` / `assert_no_tool_calls()`
- [x] `assert_tool_called_with(name, **input_kwargs)` / `assert_tool_called_before(first, second)`
- [x] `assert_tool_call_count(name, n)`
- [x] `assert_max_iterations(n)` / `assert_min_iterations(n)` / `assert_iteration_count(n)`
- [x] `assert_output_contains(text)` / `assert_output_not_contains(text)` / `assert_all_outputs_contain(text)`
- [x] `assert_stop_reason(reason)` / `assert_max_tokens(n)` / `assert_max_cost(usd)`
- [x] `assert_model_used(model)` / `assert_max_duration_ms(ms)`
- [x] Properties: `iteration_count`, `tools_called`, `final_output`, `total_tokens`,
  `total_input_tokens`, `total_output_tokens`, `estimated_cost_usd`, `total_duration_ms`,
  `call_log`, `models_used`

### pytest integration
- [x] `agentprobe` fixture — auto-registered via `entry_points["pytest11"]`
- [x] `agentprobe_multi` fixture — convenience fixture for `MultiSession`
- [x] `asyncio_mode = "auto"` in `pyproject.toml`
- [x] `pytest --agentprobe-update` option — force re-record from the CLI
- [x] pytest-xdist detection warning
- [x] `pytest-asyncio` as a required dependency

### CLI (`agentprobe`)
- [x] `agentprobe show <fixture.jsonl>` — pretty-print calls, tools, tokens
- [x] `agentprobe show --json <fixture.jsonl>` — machine-readable JSON (incl. chunk_count)
- [x] `agentprobe diff <a.jsonl> <b.jsonl>` — compare two fixtures, exit 1 on differences
- [x] `agentprobe diff --json` — machine-readable diff output
- [x] `agentprobe record <script.py> [output.jsonl]` — run a script, capture all OpenAI calls
- [x] `agentprobe validate <fixture.jsonl>` — validate fixture format and structure
- [x] `agentprobe init` — scaffold `tests/fixtures/` and a sample `conftest.py`
- [x] `agentprobe fixtures [dir]` — list all fixtures with call counts and streaming stats

### Infrastructure
- [x] `py.typed` marker (PEP 561) — mypy/pyright compatibility for library consumers
- [x] `pyproject.toml` with classifiers, URLs, hard deps (openai, pytest, pytest-asyncio)
- [x] GitHub Actions CI: Python 3.9–3.13 matrix
- [x] CHANGELOG.md
- [x] `.gitignore`

### Tests
- [x] 112 tests, 0 failures
- [x] Covers: inject, inject_error, model assertions, timing assertions, call_log, diff --json,
  fixtures list, streaming, multi-agent, CLI record, validation, pytest option, error cases

---

## Next — v0.4.0

### High priority
- [ ] **pytest-xdist full compatibility** — fixture-level isolation using tmp_path
  or worker-specific fixture directories; no just a warning but a working solution
- [ ] **`agentprobe record` async scripts** — detect `asyncio.run()` entry point
  automatically; currently requires the script to use sync OpenAI
- [ ] **Structured fixture validation with Pydantic** — `agentprobe validate` should
  attempt full `ChatCompletion.model_validate` on each fixture response, not just
  check for key presence

### Medium priority
- [ ] **`probe.first_tool_called`** — convenience property for the name of the first tool invoked
- [ ] **`probe.assert_response_contains_json(key, value)`** — assert structured JSON in output
- [ ] **Cost estimation for streaming** — extract usage when `stream_options={"include_usage": True}`
- [ ] **`agentprobe diff --json` full content diff** — include content/tool argument diffs, not just metadata
- [ ] **Session context manager without `as`** — allow `with session.replay(path):` without assigning proxy

### Lower priority
- [ ] **VS Code extension** — inline fixture viewer in the editor
- [ ] **PyPI publish automation** — GitHub Action to publish on tag push
- [ ] **Fixture compression** — optional gzip for large multi-turn fixtures

---

## Publish checklist (v0.3.0)
- [ ] Rotate PyPI token
- [ ] Tag release: `git tag v0.3.0 && git push --tags`
- [ ] Add GitHub release with CHANGELOG excerpt
- [ ] Submit to [awesome-pytest](https://github.com/augustogoulart/awesome-pytest)
