# Changelog

All notable changes to `pytest-agentprobe` are documented here.

---

## [0.22.0] — 2026-06-03

### Added

**AssertionProxy + AnthropicAssertionProxy**
- `probe.assert_min_tool_calls(n)` — symmetric counterpart to `assert_max_tool_calls`; fails if the total tool invocation count falls below *n*, catching agents that silently skip tool use
- `probe.assert_output_char_count(min, max)` — character-level bounds on the final output; complements `assert_output_word_count` for byte-precise length checks
- `probe.assert_tool_result_contains(name, text)` — asserts at least one result fed back for *name* contains *text*; checks `role: "tool"` messages (OpenAI) or `tool_result` blocks (Anthropic), not the inputs sent to the tool — the first assertion to validate what comes *back* from tool calls

**CLI**
- `agentprobe show --tool-results` — inline tool result messages in the human-readable `show` output so you can see the full request/response cycle without inspecting raw JSON
- `agentprobe fixtures --tool-names [dir]` — list all unique tool names called across every fixture in a directory; useful for auditing what tools an agent actually invokes

---

## [0.21.0] — 2026-06-03

### Added

**AssertionProxy + AnthropicAssertionProxy**
- `probe.assert_response_latency_percentile(p, ms)` — asserts the *p*-th percentile (0–100) of recorded call durations is under *ms*; useful for p95/p99 SLA checks across multi-call sessions
- `probe.assert_all_responses_under_tokens(n)` — asserts every individual API call used fewer than *n* total tokens; complements `assert_max_tokens` which checks the session total
- `probe.assert_tool_arg_type(name, key, type)` — asserts all calls to *name* have `input[key]` of the given Python type string (`"str"`, `"int"`, `"float"`, `"bool"`, `"list"`, `"dict"`, `"null"`); correctly distinguishes `bool` from `int` (since `bool` is a subclass of `int` in Python)

**CLI**
- `agentprobe stats --latency-percentile N` — report pN latency (0–100) across all calls in a fixture directory; `--json` for structured output
- `agentprobe fixtures --by-token-count [dir]` — list fixtures sorted by total token usage (descending); useful for identifying the most expensive sessions

---

## [0.20.0] — 2026-06-02

### Added

**AssertionProxy + AnthropicAssertionProxy**
- `probe.assert_no_empty_system_prompt()` — fails if any call with a system message/prompt has empty content; catches misconfigured agents that set `system=""` or `None`
- `probe.assert_tool_inputs_unique(name)` — fails if any two calls to *name* used identical serialized inputs; detects redundant invocations
- `probe.assert_output_not_empty()` — fails if the final output is `None`, empty, or whitespace-only

**CLI**
- `agentprobe fixtures --count [dir]` — prints a count of fixture files without listing them; `--json` for structured output
- `agentprobe fixtures --delete-old N --confirm [dir]` — deletes fixtures whose `_meta.recorded_at` is at least N days old; requires `--confirm` to prevent accidental deletion

---

## [0.19.0] — 2026-06-02

### Added

**AssertionProxy + AnthropicAssertionProxy**
- `probe.assert_tool_never_called_with(name, **forbidden)` — inverse of `assert_tool_called_with`; asserts the tool was never called with the given key/value combination
- `probe.assert_response_format(fmt)` — checks final output format: `"json"` validates JSON parse, `"markdown"` checks for headings/lists
- `probe.assert_prompt_growth_bounded(max_ratio)` — checks every consecutive call pair; fails if any single step grows by more than `max_ratio` (more granular than `assert_context_growth`)

**CLI**
- `agentprobe show --calls N` — show only first N calls (`N > 0`) or last N calls (`N < 0`); useful for long fixtures

---

## [0.18.0] — 2026-06-02

### Added

**AssertionProxy + AnthropicAssertionProxy**
- `probe.assert_first_response_latency_under(ms)` — cold-start SLA check on the first call only; skips if no `duration_ms` recorded
- `probe.assert_output_contains_all(*substrings)` — asserts all given substrings appear in the final output; shorthand for multiple `assert_output_contains` chains
- `probe.assert_tool_call_args_match(name, pattern)` — regex match on the full serialized tool input dict; flexible structural check without full schema validation

**CLI**
- `agentprobe fixtures --age-days N [dir]` — lists fixtures whose `_meta.recorded_at` is at least *N* days old; useful for stale-fixture cleanup
- `agentprobe migrate --strip-pii PATTERN` — redacts all regex matches with `[REDACTED]` in tool call inputs (OpenAI and Anthropic); repeatable flag for multiple patterns

---

## [0.17.0] — 2026-06-02

### Added

**AssertionProxy + AnthropicAssertionProxy**
- `probe.assert_tool_called_n_times(name, n)` — asserts the total invocation count of *name* across the full session; complements `assert_tool_call_count_per_call` which checks per-turn counts
- `probe.assert_no_sensitive_in_messages(*patterns)` — regex-guards all outgoing request messages against sensitive data (API keys, SSNs, emails); checks both user and assistant message content
- `probe.assert_tool_input_contains(name, key, value)` — asserts at least one call to *name* had `input[key] == value`; shorthand for single-field input checks without full schema

**CLI**
- `agentprobe fixtures --label TAG [dir]` — lists all fixtures whose `_meta.label` equals *TAG*; `--json` for machine-readable output; complements `record --label`

---

## [0.16.0] — 2026-06-02

### Added

**AssertionProxy + AnthropicAssertionProxy**
- `probe.assert_final_tool_not_called(name)` — fails if the last API call invoked *name*; guards against agents that end on a tool call instead of a text response
- `probe.assert_output_word_count(min_words=0, max_words=None)` — asserts the final output word count is in range; catches over-verbose and too-terse responses
- `probe.assert_no_pii_in_tool_inputs(*patterns)` — regex-based PII guard on all tool call inputs; raises if any pattern matches

**CLI**
- `agentprobe record --label TAG` — embeds a custom label string into `_meta.label`; useful for grouping fixtures by CI run, environment, or feature branch
- `agentprobe show --stdout` — prints captured stdout/stderr from `_meta` below the call listing (from fixtures recorded with `record --capture-stdout`)

---

## [0.15.0] — 2026-06-02

### Added

**AssertionProxy + AnthropicAssertionProxy**
- `probe.assert_tool_call_count_per_call(name, n)` — for every API call that invoked *name*, assert it was called exactly *n* times (skips calls that didn't invoke it at all)
- `probe.assert_no_tool_call_cycles()` — fails if the same tool was called twice in a row; detects the most common stuck-loop pattern

**CLI**
- `agentprobe fixtures --summarize [dir]` — prints a one-line summary per fixture (call count, token counts, tools, model, recording date); `--json` for machine-readable output; handles both OpenAI and Anthropic fixtures
- `agentprobe compare <fixture_a> <fixture_b>` — structural similarity score (0–100) based on call count, stop_reason, tools, and model agreement per call; `--json` for structured output with `score`, `matches`, `total_checks`, `differences`

---

## [0.14.0] — 2026-06-02

### Added

**AssertionProxy + AnthropicAssertionProxy**
- `probe.assert_tool_call_order(*names)` — partial-order check; asserts tools were called in the given sequence with any tools allowed in between
- `probe.assert_no_empty_tool_inputs()` — fails if any tool was called with empty `{}` inputs (catches under-specified tool calls)
- `probe.assert_average_latency_under(ms)` — asserts the average `duration_ms` across all calls is below the limit; skips calls with no recorded duration

**CLI**
- `agentprobe record --max-calls N` — truncates fixture to the first N recorded calls; useful for capping long agent runs

### Fixed
- `assert_tool_call_order` and `assert_tool_sequence` now correctly use call-order rather than the alphabetically-sorted `tools_called` property; added `_tools_in_call_order()` helper to both `AssertionProxy` and `AnthropicAssertionProxy`

---

## [0.13.0] — 2026-06-02

### Added

**AssertionProxy + AnthropicAssertionProxy**
- `probe.assert_no_repeated_messages()` — fails if any consecutive calls sent identical last user messages; detects stuck-loop agents
- `probe.assert_output_language(lang)` — detects response language via `langdetect` (optional dep); raises `ImportError` if not installed
- `probe.assert_token_ratio(call_n, max_ratio)` — asserts call N used at most `max_ratio × call 0` tokens; per-call growth guard relative to baseline

**CLI: validate Anthropic support**
- `agentprobe validate` now auto-detects Anthropic fixture format and validates with `anthropic.types.Message.model_validate`; validates streaming chunks against `Raw*Event` Pydantic types; correctly skips `_meta` header lines in both error-checking and lint passes

**CLI: fixtures --orphaned**
- `agentprobe fixtures --orphaned [dir]` — lists `.jsonl`/`.jsonl.gz` fixture files not referenced (by name or stem) in any Python test file under `tests/`; `--json` for machine-readable output

---

## [0.12.0] — 2026-06-02

### Added

**AssertionProxy + AnthropicAssertionProxy**
- `probe.assert_no_hallucinated_tool_calls(*allowed)` — fails if any tool name is not in the declared allowed set; guards against agents inventing tool names
- `probe.assert_max_tool_calls(n)` — fails if the total number of tool calls across all iterations exceeds `n`
- `probe.assert_system_prompt_present()` — fails if no call included a system message/prompt; catches misconfigured agents

**CLI**
- `agentprobe record --provider anthropic` — intercept `messages.create` instead of `chat.completions.create`; records Anthropic fixtures from the CLI
- `agentprobe stats --by-model` — group aggregate token/cost stats by model name across all fixtures; `--json` for machine-readable output

---

## [0.11.0] — 2026-06-02

### Added

**Anthropic streaming**
- `messages.create(stream=True)` — full record/replay support; recording consumes events upfront (same behaviour as OpenAI streaming); replay deserializes stored `chunks` back into `RawMessageStreamEvent` objects
- `MockAnthropicStream` — sync mock with `__iter__`, `__enter__`/`__exit__`, `get_final_message()`, `get_final_text()`; `MockAnthropicAsyncStream` for async
- `_assemble_anthropic_from_events(events)` — reassembles a `Message` dict from serialized stream events
- `_make_anthropic_stream_events(response)` — synthesizes plausible stream events from any non-streaming fixture, enabling `stream=True` replay on fixtures recorded without streaming

**AssertionProxy + AnthropicAssertionProxy**
- `probe.assert_response_time_under(ms)` — latency SLA gate; asserts every call's recorded `duration_ms` is under the limit
- `probe.assert_tool_input_schema(name, schema)` — validates tool call inputs against a JSON Schema (requires `jsonschema`)

**CLI: Anthropic fixture support**
- `agentprobe show` — auto-detects Anthropic format; renders `stop_reason`, `input_tokens`/`output_tokens`, text and `tool_use` blocks
- `agentprobe show --json` — includes `provider: "anthropic"` field; uses `input_tokens`/`output_tokens` and Anthropic pricing
- `agentprobe diff` — diffs Anthropic fixtures on `stop_reason`, tools, `input_tokens`, content
- `agentprobe diff --json` — machine-readable Anthropic-aware diff

---

## [0.10.0] — 2026-06-02

### Added

**CLI**
- `agentprobe replay <fixture> <script>` — run a Python script in pure replay mode without pytest; supports `--provider anthropic`, `--strict` (fail if not all calls consumed), `--env FILE`
- `agentprobe record --capture-stdout` — capture script stdout/stderr and store in `_meta.stdout` / `_meta.stderr` inside the fixture; outputs `[+stdout]` note on save

**AssertionProxy** (OpenAI)
- `probe.assert_tool_called_before_output()` — assert at least one tool was called AND the final response contains text output; verifies the canonical agentic pattern
- `probe.call(n).tool_call_inputs` — already worked via single-call proxy; confirmed and tested

**AnthropicAssertionProxy**
- `probe.assert_tool_called_before_output()` — same assertion for Anthropic sessions
- `probe.call(n).tool_call_inputs` — confirmed working on per-call proxy

**AnthropicMultiSession**
- Per-client instance-level patching for multi-agent Anthropic scenarios (mirrors `MultiSession` for OpenAI)
- `record(client, path)`, `replay(client, path)`, `auto(client, path)`, `replay_chain(*pairs)`, `async_record`, `async_replay`

---

## [0.9.0] — 2026-06-02

### Added

**Anthropic Claude support**
- `AnthropicSession` — full record/replay/auto/inject/record_append/replay_chain for the Anthropic `messages.create` API (sync + async via `AsyncAnthropic`)
- `AnthropicAssertionProxy` — complete assertion DSL mirroring `AssertionProxy` but reading Anthropic response format (`content` blocks, `stop_reason`, `usage.input_tokens`/`output_tokens`)
- Anthropic pricing table: Claude 4 (Opus/Sonnet/Haiku), Claude 3.5, and Claude 3 families
- `estimate_cost_anthropic(model, input_tokens, output_tokens)` in `_pricing`

**AssertionProxy** (OpenAI)
- `probe.assert_no_duplicate_tool_calls()` — fails if any tool is invoked twice with identical arguments (detects looping agents)
- `probe.assert_context_growth(max_ratio)` — asserts prompt tokens never grow by more than `max_ratio` between consecutive calls

**AnthropicAssertionProxy** (Anthropic)
- All methods above plus full parity with `AssertionProxy`: `assert_tool_called`, `assert_tool_called_with`, `assert_tool_sequence`, `assert_stop_reason`, `assert_output_contains`, `assert_max_tokens`, `assert_max_cost`, `assert_cost_per_call`, `assert_messages_count`, `assert_model_used`, `assert_all_models_in`, `assert_no_empty_responses`, `assert_no_duplicate_tool_calls`, `assert_context_growth`
- Properties: `iteration_count`, `tools_called`, `first/last_tool_called`, `final_output`, `total_tokens`, `total_input/output_tokens`, `estimated_cost_usd`, `models_used`, `call_log`, `tool_call_inputs`, `messages_sent`
- `call(n)` scoped per-call proxy, `summary_dict()`, `export_json()`

---

## [0.8.0] — 2026-06-02

### Added

**AssertionProxy**
- `probe.assert_cost_per_call(usd)` — assert no individual call exceeded a cost budget
- `probe.assert_messages_count(n)` — assert total messages sent across all calls equals `n`
- `probe.assert_all_models_in(*allowed)` — model governance: every call must use an allowed model
- `probe.assert_no_empty_responses()` — assert every call returned content or tool calls
- `probe.export_json(indent=2)` — export full session as a JSON string (`{"summary": ..., "calls": [...]}`) for CI artefacts
- `probe.matches_fixture(path, *, ignore_content=False)` — structural regression comparison against a golden fixture (model, finish_reason, tools, content)

**Session**
- `Session.record_append(path)` / `Session.async_record_append(path)` — append new calls to an existing fixture without overwriting; creates the file if it doesn't exist

**MultiSession**
- `MultiSession.replay_chain(*client_path_pairs)` — replay multiple chained fixtures for multiple clients simultaneously; each pair is `(client, path_or_list_of_paths)`

**CLI**
- `agentprobe validate --strict` — treat lint warnings as errors (exit 1 if any warnings present)
- `agentprobe diff --by-call` — human-readable side-by-side per-call comparison
- `agentprobe fixtures --clean` — remove stale `.lock` files left by interrupted recordings
- `agentprobe fixtures --by-date` / `agentprobe stats --by-date` — group fixture stats by recording date from `_meta` header; `--json` for machine-readable output
- `agentprobe record --dry-run` — run the script and show what would be captured without saving
- `agentprobe record --append` — append captured calls to an existing fixture instead of overwriting

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
