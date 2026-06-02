# Changelog

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
