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
        entry = {
            "model": call.get("request", {}).get("model"),
            "finish_reason": choices[0]["finish_reason"] if choices else None,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "duration_ms": call.get("duration_ms"),
            "streaming": bool(call.get("chunks")),
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


def cmd_record(args):
    """Run a Python script and capture all OpenAI chat.completions calls to a JSONL fixture."""
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

    # recording_context handles sync; we also patch async so async agents work.
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

    with recording_context(calls):
        with patch.object(openai.resources.chat.completions.AsyncCompletions, "create", _async_patch):
            try:
                runpy.run_path(str(script), run_name="__main__")
            except SystemExit:
                pass

    _save_calls(calls, output)
    print(f"agentprobe: recorded {len(calls)} call(s) to {output}")


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
    p_diff.set_defaults(func=cmd_diff)

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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
