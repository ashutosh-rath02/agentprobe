"""agentprobe CLI — inspect, diff, and record session fixtures."""
import argparse
import json
import sys
from pathlib import Path


def _load(path: str):
    p = Path(path)
    if not p.exists():
        print(f"agentprobe: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _fmt_choice(choice: dict) -> list[str]:
    lines = []
    msg = choice.get("message", {})
    content = msg.get("content")
    if content:
        lines.append(f"  [text] {content[:120]}{'...' if len(content) > 120 else ''}")
    for tc in (msg.get("tool_calls") or []):
        fn = tc.get("function", {})
        name = fn.get("name", "?")
        args = fn.get("arguments", "{}")
        lines.append(f"  [tool_call] {name}({args[:100]}{'...' if len(args) > 100 else ''})")
    return lines


def cmd_show(args):
    calls = _load(args.fixture)
    print(f"fixture: {args.fixture}  ({len(calls)} call(s))\n")
    for i, call in enumerate(calls, 1):
        req = call.get("request", {})
        resp = call.get("response", {})
        usage = resp.get("usage") or {}
        choices = resp.get("choices", [])
        finish = choices[0]["finish_reason"] if choices else "?"
        ms = call.get("duration_ms")
        ms_str = f"  {ms:.0f}ms" if ms else ""
        streaming = " [stream]" if call.get("chunks") else ""
        print(f"-- Call {i}/{len(calls)}  model={req.get('model', '?')}  "
              f"stop={finish}"
              f"  in={usage.get('prompt_tokens', '?')} out={usage.get('completion_tokens', '?')}"
              f"{ms_str}{streaming}")
        for choice in choices:
            for line in _fmt_choice(choice):
                print(line)
        print()

    total_in = sum((c.get("response", {}).get("usage") or {}).get("prompt_tokens", 0) for c in calls)
    total_out = sum((c.get("response", {}).get("usage") or {}).get("completion_tokens", 0) for c in calls)
    print(f"total tokens: {total_in + total_out}  ({total_in} in + {total_out} out)")


def cmd_show_json(args):
    calls = _load(args.fixture)
    out = []
    for call in calls:
        resp = call.get("response", {})
        usage = resp.get("usage") or {}
        choices = resp.get("choices", [])
        chunks = call.get("chunks")
        entry = {
            "model": call.get("request", {}).get("model"),
            "finish_reason": choices[0]["finish_reason"] if choices else None,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "duration_ms": call.get("duration_ms"),
            "streaming": bool(chunks),
            "chunk_count": len(chunks) if chunks else None,
            "tools_called": [
                tc["function"]["name"]
                for ch in choices
                for tc in (ch["message"].get("tool_calls") or [])
            ],
            "final_text": next(
                (ch["message"].get("content") for ch in reversed(choices) if ch["message"].get("content")),
                None,
            ),
        }
        out.append(entry)
    print(json.dumps(out, indent=2))


def cmd_diff(args):
    a_calls = _load(args.fixture_a)
    b_calls = _load(args.fixture_b)

    max_len = max(len(a_calls), len(b_calls))
    diffs = 0

    print(f"diff {args.fixture_a} ({len(a_calls)} calls)  vs  {args.fixture_b} ({len(b_calls)} calls)\n")

    if len(a_calls) != len(b_calls):
        print(f"  call count differs: {len(a_calls)} vs {len(b_calls)}")
        diffs += 1

    for i in range(max_len):
        a = a_calls[i] if i < len(a_calls) else None
        b = b_calls[i] if i < len(b_calls) else None
        if a is None:
            print(f"  call {i+1}: only in {args.fixture_b}")
            diffs += 1
            continue
        if b is None:
            print(f"  call {i+1}: only in {args.fixture_a}")
            diffs += 1
            continue

        a_resp = a.get("response", {})
        b_resp = b.get("response", {})

        a_choices = a_resp.get("choices", [])
        b_choices = b_resp.get("choices", [])
        a_stop = a_choices[0]["finish_reason"] if a_choices else None
        b_stop = b_choices[0]["finish_reason"] if b_choices else None
        if a_stop != b_stop:
            print(f"  call {i+1}: finish_reason {a_stop!r} -> {b_stop!r}")
            diffs += 1

        a_tools = [tc["function"]["name"] for ch in a_choices for tc in (ch["message"].get("tool_calls") or [])]
        b_tools = [tc["function"]["name"] for ch in b_choices for tc in (ch["message"].get("tool_calls") or [])]
        if a_tools != b_tools:
            print(f"  call {i+1}: tools {a_tools} -> {b_tools}")
            diffs += 1

        a_in = (a_resp.get("usage") or {}).get("prompt_tokens", 0)
        b_in = (b_resp.get("usage") or {}).get("prompt_tokens", 0)
        if a_in != b_in:
            print(f"  call {i+1}: prompt_tokens {a_in} -> {b_in}")
            diffs += 1

    if diffs == 0:
        print("  no differences found")
    else:
        print(f"\n{diffs} difference(s) found")
        sys.exit(1)


def cmd_diff_json(args):
    """Machine-readable JSON diff between two fixtures."""
    a_calls = _load(args.fixture_a)
    b_calls = _load(args.fixture_b)
    differences = []

    if len(a_calls) != len(b_calls):
        differences.append({
            "type": "call_count",
            "a": len(a_calls),
            "b": len(b_calls),
        })

    for i in range(max(len(a_calls), len(b_calls))):
        a = a_calls[i] if i < len(a_calls) else None
        b = b_calls[i] if i < len(b_calls) else None
        if a is None or b is None:
            differences.append({"call": i + 1, "type": "missing", "present_in": "b" if a is None else "a"})
            continue

        a_choices = a.get("response", {}).get("choices", [])
        b_choices = b.get("response", {}).get("choices", [])

        a_stop = a_choices[0]["finish_reason"] if a_choices else None
        b_stop = b_choices[0]["finish_reason"] if b_choices else None
        if a_stop != b_stop:
            differences.append({"call": i + 1, "type": "finish_reason", "a": a_stop, "b": b_stop})

        a_tools = [tc["function"]["name"] for ch in a_choices for tc in (ch["message"].get("tool_calls") or [])]
        b_tools = [tc["function"]["name"] for ch in b_choices for tc in (ch["message"].get("tool_calls") or [])]
        if a_tools != b_tools:
            differences.append({"call": i + 1, "type": "tools", "a": a_tools, "b": b_tools})

        # Compare tool arguments per tool call
        a_args = {tc["function"]["name"]: tc["function"].get("arguments") for ch in a_choices for tc in (ch["message"].get("tool_calls") or [])}
        b_args = {tc["function"]["name"]: tc["function"].get("arguments") for ch in b_choices for tc in (ch["message"].get("tool_calls") or [])}
        for name in set(a_args) & set(b_args):
            if a_args[name] != b_args[name]:
                differences.append({"call": i + 1, "type": "tool_arguments", "tool": name, "a": a_args[name], "b": b_args[name]})

        # Compare text content
        a_content = next((ch["message"].get("content") for ch in reversed(a_choices) if ch["message"].get("content")), None)
        b_content = next((ch["message"].get("content") for ch in reversed(b_choices) if ch["message"].get("content")), None)
        if a_content != b_content:
            differences.append({"call": i + 1, "type": "content", "a": a_content, "b": b_content})

        a_in = (a.get("response", {}).get("usage") or {}).get("prompt_tokens", 0)
        b_in = (b.get("response", {}).get("usage") or {}).get("prompt_tokens", 0)
        if a_in != b_in:
            differences.append({"call": i + 1, "type": "prompt_tokens", "a": a_in, "b": b_in})

    result = {
        "fixture_a": args.fixture_a,
        "fixture_b": args.fixture_b,
        "calls_a": len(a_calls),
        "calls_b": len(b_calls),
        "identical": len(differences) == 0,
        "differences": differences,
    }
    print(json.dumps(result, indent=2))
    if differences:
        sys.exit(1)


def cmd_fixtures_list(args):
    """List all .jsonl fixture files under a directory with summary stats."""
    import os
    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"agentprobe: not a directory: {args.directory}", file=sys.stderr)
        sys.exit(1)

    fixtures = sorted(directory.rglob("*.jsonl"))
    if not fixtures:
        print(f"agentprobe: no .jsonl files found in {args.directory}")
        return

    if getattr(args, "json", False):
        entries = []
        for f in fixtures:
            try:
                calls = _load(str(f))
                streaming = sum(1 for c in calls if c.get("chunks"))
                entries.append({
                    "path": str(f),
                    "calls": len(calls),
                    "streaming_calls": streaming,
                    "size_bytes": f.stat().st_size,
                })
            except Exception as e:
                entries.append({"path": str(f), "error": str(e)})
        print(json.dumps(entries, indent=2))
    else:
        print(f"Fixtures in {args.directory}:\n")
        for f in fixtures:
            try:
                calls = _load(str(f))
                streaming = sum(1 for c in calls if c.get("chunks"))
                stream_note = f"  ({streaming} streaming)" if streaming else ""
                print(f"  {f}  [{len(calls)} call(s){stream_note}]")
            except Exception:
                print(f"  {f}  [INVALID]")
        print(f"\n{len(fixtures)} fixture(s) found")


def cmd_validate(args):
    """Validate fixture structure and attempt full Pydantic deserialization."""
    import openai.types.chat as oai

    p = Path(args.fixture)
    if not p.exists():
        print(f"agentprobe: file not found: {args.fixture}", file=sys.stderr)
        sys.exit(1)

    lines = [l for l in p.read_text().splitlines() if l.strip()]
    errors = []

    for i, line in enumerate(lines, 1):
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"line {i}: invalid JSON — {e}")
            continue

        for field in ("request", "response"):
            if field not in data:
                errors.append(f"line {i}: missing '{field}' field")

        # Attempt Pydantic deserialization of the assembled response
        resp = data.get("response", {})
        try:
            oai.ChatCompletion.model_validate(resp)
        except Exception as e:
            errors.append(f"line {i}: response failed Pydantic validation — {e}")

        # Validate each streaming chunk if present
        chunks = data.get("chunks")
        if chunks is not None:
            if not isinstance(chunks, list):
                errors.append(f"line {i}: 'chunks' must be a list")
            else:
                for j, chunk in enumerate(chunks):
                    try:
                        oai.ChatCompletionChunk.model_validate(chunk)
                    except Exception as e:
                        errors.append(f"line {i} chunk {j}: Pydantic validation failed — {e}")

    if errors:
        for err in errors:
            print(f"  ERROR: {err}", file=sys.stderr)
        print(f"\nagentprobe: {len(errors)} error(s) in {args.fixture}", file=sys.stderr)
        sys.exit(1)

    streaming = sum(1 for d in [json.loads(l) for l in lines] if d.get("chunks"))
    print(f"agentprobe: {args.fixture} OK  ({len(lines)} call(s), {streaming} streaming)")


def cmd_init(args):
    """Scaffold tests/fixtures/ and a sample conftest.py."""
    fixtures_dir = Path("tests/fixtures")
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    print(f"agentprobe: created {fixtures_dir}/")

    conftest = Path("tests/conftest.py")
    if conftest.exists():
        print(f"agentprobe: {conftest} already exists, skipping")
    else:
        conftest.write_text(
            "# tests/conftest.py\n"
            "# agentprobe fixtures live in tests/fixtures/\n"
            "# Re-record with: pytest --agentprobe-update\n"
        )
        print(f"agentprobe: created {conftest}")

    print("\nagentprobe: ready!  Record your first fixture:")
    print("  agentprobe record my_agent.py tests/fixtures/my_session.jsonl")
    print("\nThen replay in tests:")
    print("  with session.replay('tests/fixtures/my_session.jsonl') as probe:")
    print("      result = my_agent(client)")
    print("      probe.assert_tool_called('bash')")


def cmd_record(args):
    """Run a Python script and capture all OpenAI chat.completions calls to a JSONL fixture.

    Works for both sync scripts and async scripts that use asyncio.run().
    The script is executed in-process via runpy so the OpenAI patches are active
    before any import inside the script runs.
    """
    import runpy
    import time
    from unittest.mock import patch
    import openai.resources.chat.completions
    from agentprobe._models import RecordedCall
    from agentprobe._serializer import serialize_request
    from agentprobe._interceptor import (
        _assemble_from_chunks, MockStream, MockAsyncStream, recording_context,
    )
    from agentprobe._session import _save_calls

    script = Path(args.script)
    if not script.exists():
        print(f"agentprobe: script not found: {args.script}", file=sys.stderr)
        sys.exit(1)

    output = Path(args.output)
    calls: list = []

    # Patch sync completions via recording_context and async completions manually.
    # Both patches share the same `calls` list so ordering is preserved.
    orig_async = openai.resources.chat.completions.AsyncCompletions.create

    async def _async_patch(self, **kwargs):
        start = time.time()
        if kwargs.get("stream"):
            real = await orig_async(self, **kwargs)
            raw = [c async for c in real]
            ser = [c.model_dump() for c in raw]
            calls.append(RecordedCall(
                request=serialize_request(kwargs),
                response=_assemble_from_chunks(ser),
                chunks=ser,
                duration_ms=(time.time() - start) * 1000,
            ))
            return MockAsyncStream(raw)
        resp = await orig_async(self, **kwargs)
        calls.append(RecordedCall(
            request=serialize_request(kwargs),
            response=resp.model_dump(),
            duration_ms=(time.time() - start) * 1000,
        ))
        return resp

    # Detect whether the script is async (contains 'asyncio.run' or 'async def main').
    script_source = script.read_text()
    is_async = "asyncio.run(" in script_source or "async def main" in script_source

    with recording_context(calls):
        with patch.object(openai.resources.chat.completions.AsyncCompletions, "create", _async_patch):
            try:
                # For both sync and async scripts: run directly.
                # Async scripts call asyncio.run() themselves; the class-level
                # patches are already active so calls inside that loop are captured.
                runpy.run_path(str(script), run_name="__main__")
            except SystemExit:
                pass

    _save_calls(calls, output)
    async_note = " (async script)" if is_async else ""
    print(f"agentprobe: recorded {len(calls)} call(s) to {output}{async_note}")


def main():
    parser = argparse.ArgumentParser(
        prog="agentprobe",
        description="Inspect, diff, and record agentprobe session fixtures",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_show = sub.add_parser("show", help="Pretty-print a session fixture")
    p_show.add_argument("fixture", help="Path to .jsonl fixture file")
    p_show.add_argument("--json", action="store_true", help="Output as machine-readable JSON")
    p_show.set_defaults(func=lambda a: cmd_show_json(a) if a.json else cmd_show(a))

    p_diff = sub.add_parser("diff", help="Compare two session fixtures")
    p_diff.add_argument("fixture_a")
    p_diff.add_argument("fixture_b")
    p_diff.add_argument("--json", action="store_true", help="Output as machine-readable JSON")
    p_diff.set_defaults(func=lambda a: cmd_diff_json(a) if a.json else cmd_diff(a))

    p_list = sub.add_parser("fixtures", help="List fixture files in a directory")
    p_list.add_argument("directory", nargs="?", default="tests/fixtures",
                        help="Directory to search (default: tests/fixtures)")
    p_list.add_argument("--json", action="store_true", help="Output as machine-readable JSON")
    p_list.set_defaults(func=cmd_fixtures_list)

    p_record = sub.add_parser(
        "record",
        help="Run a Python script and capture all OpenAI calls to a fixture",
    )
    p_record.add_argument("script", help="Path to Python script to execute")
    p_record.add_argument(
        "output",
        nargs="?",
        default="agentprobe_session.jsonl",
        help="Output JSONL path (default: agentprobe_session.jsonl)",
    )
    p_record.set_defaults(func=cmd_record)

    p_validate = sub.add_parser("validate", help="Validate a fixture file for correctness")
    p_validate.add_argument("fixture", help="Path to .jsonl fixture file")
    p_validate.set_defaults(func=cmd_validate)

    p_init = sub.add_parser("init", help="Scaffold tests/fixtures/ and a sample conftest.py")
    p_init.set_defaults(func=cmd_init)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
