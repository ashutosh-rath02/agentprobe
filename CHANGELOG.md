# Changelog

All notable changes to `pytest-agentprobe` are documented here.

---

## [0.6.0] — 2026-06-02

### Added

**AssertionProxy**
- `probe.messages_received` — list of assistant messages received, each with `call_index`, `content`, `tool_calls`, `finish_reason`
- `probe.tool_call_inputs` — `{tool_name: [args_dict, ...]}` for all tool calls; quick bulk inspection without looping
- `probe.summary_dict()` — exportable dict summary of the session (iteration count, tokens, cost, tools, model list)
- `probe.assert_token_efficiency(min_ratio)` — assert `output_tokens / input_tokens >= min_ratio`

**Session**
- `Session.replay_chain(*paths)` — concatenate multiple fixtures end-to-end into a single session; calls exhausted in order

**CLI**
- `agentprobe migrate <input> <output>` — transform fixtures: `--rename-model OLD=NEW`, `--rename-tool OLD=NEW`, `--set-model MODEL`
- `agentprobe stats [dir]` — aggregate token/cost/duration stats across all fixtures; `--json` for machine-readable output
- `agentprobe show --model MODEL` — filter show/show --json output to calls matching a specific model
- `agentprobe record --output-format gz` — force gzip output regardless of filename extension
- `agentprobe record --watch [--interval N]` — poll script for changes and re-record automatically (Ctrl+C to stop)

---

## [0.5.0] — 2026-06-02

### Added

**AssertionProxy**
- `assert_output_matches(pattern)` — regex assertion on final output (`re.search`)
- `assert_tool_sequence(*names)` — assert exact ordered sequence of tools called
- `assert_finish_reason_all(reason)` — every call must end with the given finish reason
- `dump_fixture(path)` — save the current session to a JSONL fixture file (useful for persisting `inject()` sessions)
- `messages_sent` property — list of messages passed per API call; now captures **actual** kwargs passed during replay/inject, not just stored fixture values
- `duration_percentile(p)` — p-th percentile of per-call `duration_ms` (0–100)

**Session**
- `Session.replay(path, strict=True)` — raises `AssertionError` on exit if the agent did not consume all fixture calls; detects under-consuming agents
- `Session.auto()` — xdist-safe via `FileLock` coordination: if another worker holds the record lock, waits and then replays
- `_save_calls` now uses atomic temp-file + rename instead of `FileLock` — eliminates the deadlock between `auto()`'s coordination lock and `_save_calls`'s write-safety; fixtures are never seen in a partially-written state

**CLI**
- `agentprobe record --env FILE` — load a `.env` file before running the script (`setdefault` semantics; doesn't overwrite already-set vars)
- `agentprobe show --json` output restructured to `{"calls": [...], "summary": {...}}` with `total_calls`, `total_tokens`, `total_duration_ms`, `estimated_total_cost_usd`, `streaming_calls`
- `agentprobe validate` linting: warns (WARN, exit 0) on fixture lines missing `duration_ms` or `request.model`

### Fixed
- `replaying_context` (and `_make_sync_replayer`, `_make_async_replayer`) now update `call.request` with actual kwargs on each replay, so `probe.messages_sent` correctly reflects what was passed in the test rather than stored fixture values

---

## [0.4.0] — 2026-06-02

### Added

**AssertionProxy**
- `probe.call(n)` — scoped proxy for the nth call (0-indexed); enables per-iteration assertions
- `probe.first_tool_called` / `probe.last_tool_called` — convenience properties
- `probe.assert_output_is_json()` — assert final output parses as valid JSON
- `probe.assert_output_json_contains(**kv)` — assert key/value pairs inside JSON output

**Session**
- `Session.record / replay` now support `.jsonl.gz` compressed fixtures transparently
- `_save_calls` acquires a `FileLock` (when `filelock` is installed) to prevent concurrent
  pytest-xdist workers from corrupting the same fixture during parallel recording

**CLI**
- `agentprobe validate` now performs full `ChatCompletion.model_validate` + per-chunk
  `ChatCompletionChunk.model_validate` — catches malformed fixtures before a test run fails
- `agentprobe diff --json` now includes `tool_arguments` and `content` diffs
- `agentprobe record` detects async scripts (`asyncio.run()` / `async def main`) and
  notes them in the output; both sync and async scripts work correctly

**Infrastructure**
- Pricing table expanded: versioned model IDs (`gpt-4o-2024-11-20`, `o1-2024-12-17` etc),
  `gpt-4-32k`, `gpt-3.5-turbo-0125` — prefix matching resolves versioned suffixes
- `pip install pytest-agentprobe[xdist]` installs `filelock` for parallel-test safety
- `.github/workflows/publish.yml` — OIDC trusted publishing on `git tag v*`

---

## [0.3.0] — 2026-06-02

### Added

**AssertionProxy extensions**
- `probe.assert_model_used(model)` — assert every call used the specified model
- `probe.assert_no_tool_calls()` — explicit assertion that no tools were invoked
- `probe.assert_max_duration_ms(ms)` — wall-clock timing assertion
- `probe.total_duration_ms` — sum of recorded durations across all calls
- `probe.call_log` — list of `{"request": ..., "response": ...}` dicts for custom assertions
- `probe.models_used` — ordered list of model names across all calls

**Session enhancements**
- `Session.inject(*responses)` — ad-hoc replay of explicit response dicts or Pydantic objects without a fixture file on disk; useful for quick unit tests
- `Session.inject_error(exception)` — make the next API call raise an exception; for testing agent error-handling paths

**CLI additions**
- `agentprobe diff --json` — machine-readable diff output
- `agentprobe fixtures [dir]` — list all `.jsonl` fixtures in a directory with call counts and streaming stats; `--json` for machine-readable output
- `agentprobe show --json` now includes `chunk_count` field for streaming calls

**Infrastructure**
- `py.typed` marker (PEP 561) — enables mypy/pyright type checking for library consumers
- pytest-xdist detection warning — warns when parallel workers are active and `Session` (class-level patching) is in use; recommends `MultiSession` instead

### Fixed
- `assert_max_cost` error message used Unicode `≤` causing cp1252 crash on Windows; replaced with `<=`

---

All notable changes to `pytest-agentprobe` are documented here.

---

## [0.2.0] — 2026-06-02

### Added
- **OpenAI adapter** — migrated from Anthropic to OpenAI SDK; patches
  `openai.resources.chat.completions.Completions.create` (sync + async)
- **Streaming support** (`stream=True`) — `MockStream` / `MockAsyncStream` replay
  recorded chunks faithfully; `_assemble_from_chunks` builds a `ChatCompletion`-
  compatible assembled response so all `AssertionProxy` assertions work unchanged
- **`MultiSession`** — per-client instance-level patching for multi-agent /
  subagent scenarios where two clients must not share a call sequence
- **`agentprobe record <script.py>`** — run any Python script and capture all
  OpenAI calls to a JSONL fixture without modifying the agent code
- **`agentprobe show --json`** — machine-readable JSON output per call
- **`agentprobe validate <fixture>`** — validates fixture format and structure
- **`agentprobe init`** — scaffolds `tests/fixtures/` and a sample `conftest.py`
- **`pytest --agentprobe-update`** option — force re-record from the CLI instead
  of setting the `AGENTPROBE_UPDATE` env var
- **`agentprobe_multi` pytest fixture** — convenience fixture for `MultiSession`
- GitHub Actions CI matrix (Python 3.9–3.13)

### Changed
- `total_input_tokens` / `total_output_tokens` now safely handle `None` usage
  (streaming calls without `include_usage`)
- `_save_calls` persists the `chunks` field only when present (streaming calls)
- `agentprobe show` annotates streaming calls with `[stream]`
- `pyproject.toml`: version bumped to `0.2.0`; `pytest-asyncio` promoted to a
  required dependency

### Fixed
- CLI `show` / `diff` crash on Windows cp1252 terminal (replaced `──` with `--`)

---

## [0.1.0] — 2025-05-31

### Added
- Initial release: `Session.record / replay / auto` with JSONL fixtures
- Full `AssertionProxy` assertion DSL (tools, iterations, output, tokens, cost)
- `agentprobe show` and `agentprobe diff` CLI commands
- `agentprobe` pytest fixture
- Async parity via `async_record / async_replay / async_auto`
