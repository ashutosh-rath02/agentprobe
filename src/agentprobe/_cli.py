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
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return [r for r in rows if "_meta" not in r]


def _load_meta(path: str) -> dict:
    """Return the _meta header dict from a fixture, or empty dict if absent."""
    p = Path(path)
    if not p.exists():
        return {}
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            if "_meta" in d:
                return d["_meta"]
        except json.JSONDecodeError:
            pass
    return {}


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


def _filter_calls(calls: list, model_filter: str | None) -> list:
    if not model_filter:
        return calls
    return [c for c in calls if c.get("request", {}).get("model") == model_filter]


def cmd_show(args):
    calls = _filter_calls(_load(args.fixture), getattr(args, "model", None))
    meta = _load_meta(args.fixture)
    meta_str = ""
    if meta:
        meta_str = f"  [v{meta.get('agentprobe_version', '?')} recorded {meta.get('recorded_at', '?')}]"
    print(f"fixture: {args.fixture}  ({len(calls)} call(s)){meta_str}\n")
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
    from agentprobe._pricing import estimate_cost
    calls = _filter_calls(_load(args.fixture), getattr(args, "model", None))
    out = []
    total_prompt = 0
    total_completion = 0
    total_duration = 0.0
    total_cost = 0.0
    for call in calls:
        resp = call.get("response", {})
        usage = resp.get("usage") or {}
        choices = resp.get("choices", [])
        chunks = call.get("chunks")
        prompt_tok = usage.get("prompt_tokens") or 0
        completion_tok = usage.get("completion_tokens") or 0
        dur = call.get("duration_ms") or 0.0
        model = call.get("request", {}).get("model") or ""
        cost = estimate_cost(model, prompt_tok, completion_tok)
        total_prompt += prompt_tok
        total_completion += completion_tok
        total_duration += dur
        total_cost += cost
        entry = {
            "model": model or None,
            "finish_reason": choices[0]["finish_reason"] if choices else None,
            "prompt_tokens": prompt_tok or None,
            "completion_tokens": completion_tok or None,
            "duration_ms": call.get("duration_ms"),
            "streaming": bool(chunks),
            "chunk_count": len(chunks) if chunks else None,
            "estimated_cost_usd": cost if cost else None,
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
    result = {
        "calls": out,
        "summary": {
            "total_calls": len(out),
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "total_duration_ms": total_duration,
            "estimated_total_cost_usd": round(total_cost, 8),
            "streaming_calls": sum(1 for e in out if e["streaming"]),
        },
        "meta": _load_meta(args.fixture),
    }
    print(json.dumps(result, indent=2))


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


def cmd_fixtures_stats(args):
    """Aggregate token/cost/duration stats across all fixtures in a directory."""
    from agentprobe._pricing import estimate_cost
    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"agentprobe: not a directory: {args.directory}", file=sys.stderr)
        sys.exit(1)

    fixtures = sorted(directory.rglob("*.jsonl")) + sorted(directory.rglob("*.jsonl.gz"))
    if not fixtures:
        print(f"agentprobe: no fixtures found in {args.directory}")
        return

    total_calls = 0
    total_prompt = 0
    total_completion = 0
    total_duration = 0.0
    total_cost = 0.0
    total_streaming = 0
    by_model: dict = {}
    errors = 0

    for f in fixtures:
        try:
            calls = _load(str(f))
        except Exception:
            errors += 1
            continue
        for c in calls:
            total_calls += 1
            usage = c.get("response", {}).get("usage") or {}
            p_tok = usage.get("prompt_tokens", 0) or 0
            c_tok = usage.get("completion_tokens", 0) or 0
            dur = c.get("duration_ms") or 0.0
            model = c.get("request", {}).get("model") or "unknown"
            cost = estimate_cost(model, p_tok, c_tok)
            total_prompt += p_tok
            total_completion += c_tok
            total_duration += dur
            total_cost += cost
            if c.get("chunks"):
                total_streaming += 1
            entry = by_model.setdefault(model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0})
            entry["calls"] += 1
            entry["prompt_tokens"] += p_tok
            entry["completion_tokens"] += c_tok
            entry["cost"] += cost

    if getattr(args, "json", False):
        print(json.dumps({
            "fixtures": len(fixtures),
            "error_fixtures": errors,
            "total_calls": total_calls,
            "streaming_calls": total_streaming,
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "total_duration_ms": total_duration,
            "estimated_total_cost_usd": round(total_cost, 6),
            "by_model": by_model,
        }, indent=2))
    else:
        print(f"Stats for {args.directory}  ({len(fixtures)} fixture(s), {total_calls} call(s))\n")
        print(f"  Tokens:    {total_prompt + total_completion:,}  ({total_prompt:,} in + {total_completion:,} out)")
        print(f"  Duration:  {total_duration/1000:.1f}s total")
        print(f"  Cost est.: ${total_cost:.4f}")
        print(f"  Streaming: {total_streaming} call(s)")
        if errors:
            print(f"  Errors:    {errors} fixture(s) could not be read")
        if by_model:
            print("\n  By model:")
            for model, info in sorted(by_model.items()):
                print(f"    {model}: {info['calls']} call(s), ${info['cost']:.4f}")


def cmd_migrate(args):
    """Transform a fixture file: rename models, rename tools, or set model."""
    p = Path(args.input)
    if not p.exists():
        print(f"agentprobe: file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    rename_models = {}
    for s in getattr(args, "rename_model", []) or []:
        if "=" not in s:
            print(f"agentprobe: --rename-model requires old=new format, got: {s!r}", file=sys.stderr)
            sys.exit(1)
        old, new = s.split("=", 1)
        rename_models[old.strip()] = new.strip()

    rename_tools = {}
    for s in getattr(args, "rename_tool", []) or []:
        if "=" not in s:
            print(f"agentprobe: --rename-tool requires old=new format, got: {s!r}", file=sys.stderr)
            sys.exit(1)
        old, new = s.split("=", 1)
        rename_tools[old.strip()] = new.strip()

    set_model = getattr(args, "set_model", None)
    lines = [l for l in p.read_text().splitlines() if l.strip()]
    transformed = 0

    out_lines = []
    for line in lines:
        data = json.loads(line)
        req = data.get("request", {})
        resp = data.get("response", {})

        # Rename/set model in request
        if set_model:
            req["model"] = set_model
            resp["model"] = set_model
        elif req.get("model") in rename_models:
            new = rename_models[req["model"]]
            req["model"] = new
            resp["model"] = new

        # Rename tools in response choices
        for choice in resp.get("choices", []):
            for tc in (choice.get("message", {}).get("tool_calls") or []):
                old = tc.get("function", {}).get("name", "")
                if old in rename_tools:
                    tc["function"]["name"] = rename_tools[old]
                    transformed += 1

        # Rename tools in request tools list
        for tool in req.get("tools", []):
            old = tool.get("function", {}).get("name", "")
            if old in rename_tools:
                tool["function"]["name"] = rename_tools[old]

        data["request"] = req
        data["response"] = resp
        out_lines.append(json.dumps(data))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n")
    print(f"agentprobe: migrated {len(lines)} call(s) to {out_path}  ({transformed} tool rename(s))")


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

    # Lint: soft warnings (don't fail)
    warnings = []
    for i, line in enumerate(lines, 1):
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if data.get("duration_ms") is None:
            warnings.append(f"line {i}: no duration_ms (recorded without timing — replay-only fixture)")
        if not data.get("request", {}).get("model"):
            warnings.append(f"line {i}: request has no model field")

    for w in warnings:
        print(f"  WARN: {w}")

    if warnings and getattr(args, "strict", False):
        print(f"\nagentprobe: {len(warnings)} warning(s) treated as errors (--strict)", file=sys.stderr)
        sys.exit(1)

    streaming = sum(1 for d in [json.loads(l) for l in lines] if d.get("chunks"))
    status = f"OK ({len(warnings)} warning(s))" if warnings else "OK"
    print(f"agentprobe: {args.fixture} {status}  ({len(lines)} call(s), {streaming} streaming)")


def cmd_diff_by_call(args):
    """Human-readable side-by-side per-call comparison of two fixtures."""
    a_calls = _load(args.fixture_a)
    b_calls = _load(args.fixture_b)
    max_len = max(len(a_calls), len(b_calls), 1)

    print(f"agentprobe diff: {args.fixture_a}  vs  {args.fixture_b}\n")

    diffs = 0
    for i in range(max_len):
        a = a_calls[i] if i < len(a_calls) else None
        b = b_calls[i] if i < len(b_calls) else None

        label_a = f"A call {i+1}"
        label_b = f"B call {i+1}"
        print(f"{'=' * 60}")
        print(f"  {label_a:<28}  {label_b}")
        print(f"{'=' * 60}")

        if a is None:
            print(f"  (absent)                          {_call_summary(b)}")
            diffs += 1
            print()
            continue
        if b is None:
            print(f"  {_call_summary(a):<34}(absent)")
            diffs += 1
            print()
            continue

        fields = [
            ("model",         _call_model(a),       _call_model(b)),
            ("finish_reason", _call_finish(a),       _call_finish(b)),
            ("prompt_tokens", _call_prompt_tok(a),   _call_prompt_tok(b)),
            ("tools",         _call_tools_str(a),    _call_tools_str(b)),
            ("content",       _call_content_snip(a), _call_content_snip(b)),
        ]
        for name, va, vb in fields:
            marker = "!!" if va != vb else "  "
            if va != vb:
                diffs += 1
            print(f"  {marker} {name:<16} {str(va):<30}  {vb}")
        print()

    if diffs == 0:
        print("No differences found.")
    else:
        print(f"\n{diffs} difference(s) found.")
        sys.exit(1)


def _call_summary(c): return f"model={c.get('request',{}).get('model','?')} stop={_call_finish(c)}"
def _call_model(c): return c.get("request", {}).get("model", "?")
def _call_finish(c):
    ch = c.get("response", {}).get("choices", [])
    return ch[0]["finish_reason"] if ch else "?"
def _call_prompt_tok(c): return (c.get("response", {}).get("usage") or {}).get("prompt_tokens", "?")
def _call_tools_str(c):
    ch = c.get("response", {}).get("choices", [])
    names = [tc["function"]["name"] for ch_ in ch for tc in (ch_["message"].get("tool_calls") or [])]
    return ",".join(names) if names else "(none)"
def _call_content_snip(c):
    ch = c.get("response", {}).get("choices", [])
    text = next((ch_["message"].get("content") for ch_ in reversed(ch) if ch_["message"].get("content")), None)
    if text is None:
        return "(none)"
    return text[:40] + "..." if len(text) > 40 else text


def cmd_fixtures_clean(args):
    """Remove stale .lock files left by interrupted recordings."""
    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"agentprobe: not a directory: {args.directory}", file=sys.stderr)
        sys.exit(1)
    locks = list(directory.rglob("*.lock"))
    if not locks:
        print(f"agentprobe: no .lock files found in {args.directory}")
        return
    for lock in locks:
        lock.unlink()
        print(f"  removed {lock}")
    print(f"\nagentprobe: removed {len(locks)} lock file(s)")


def cmd_stats_by_date(args):
    """Aggregate stats grouped by the recording date from _meta headers."""
    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"agentprobe: not a directory: {args.directory}", file=sys.stderr)
        sys.exit(1)

    from agentprobe._pricing import estimate_cost
    by_date: dict = {}
    for f in sorted(directory.rglob("*.jsonl")) + sorted(directory.rglob("*.jsonl.gz")):
        meta = _load_meta(str(f))
        date = meta.get("recorded_at", "")[:10] if meta else "unknown"
        try:
            calls = _load(str(f))
        except Exception:
            continue
        entry = by_date.setdefault(date, {"fixtures": 0, "calls": 0, "tokens": 0, "cost": 0.0})
        entry["fixtures"] += 1
        for c in calls:
            usage = c.get("response", {}).get("usage") or {}
            p = usage.get("prompt_tokens", 0) or 0
            co = usage.get("completion_tokens", 0) or 0
            entry["calls"] += 1
            entry["tokens"] += p + co
            entry["cost"] += estimate_cost(c.get("request", {}).get("model", ""), p, co)

    if getattr(args, "json", False):
        print(json.dumps(by_date, indent=2))
    else:
        print(f"Fixtures by recording date ({args.directory}):\n")
        for date in sorted(by_date):
            e = by_date[date]
            print(f"  {date}  {e['fixtures']} fixture(s)  {e['calls']} call(s)  "
                  f"{e['tokens']:,} tokens  ${e['cost']:.4f}")


def cmd_replay(args):
    """Run a Python script in pure replay mode against a saved fixture."""
    import runpy, time
    import openai.resources.chat.completions
    from agentprobe._session import _load_calls, _check_exists
    from agentprobe._interceptor import _strict_replaying_context, replaying_context

    fixture = Path(args.fixture)
    _check_exists(fixture)
    calls = _load_calls(fixture)

    script = Path(args.script)
    if not script.exists():
        print(f"agentprobe: script not found: {args.script}", file=sys.stderr)
        sys.exit(1)

    env_file = getattr(args, "env", None)
    if env_file:
        import os as _os
        env_path = Path(env_file)
        if not env_path.exists():
            print(f"agentprobe: env file not found: {env_file}", file=sys.stderr)
            sys.exit(1)
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            _os.environ.setdefault(key.strip(), value.strip())

    provider = getattr(args, "provider", "openai")
    strict = getattr(args, "strict", False)
    index: list = [0]

    if provider == "anthropic":
        from agentprobe._anthropic_interceptor import (
            _anthropic_strict_replaying_context,
            anthropic_replaying_context,
        )
        ctx = _anthropic_strict_replaying_context(calls, index) if strict else anthropic_replaying_context(calls)
    else:
        ctx = _strict_replaying_context(calls, index) if strict else replaying_context(calls)

    script_source = script.read_text()
    is_async = "asyncio.run(" in script_source or "async def main" in script_source

    try:
        with ctx:
            try:
                runpy.run_path(str(script), run_name="__main__")
            except SystemExit as e:
                if e.code not in (0, None):
                    print(f"agentprobe: script exited with code {e.code}", file=sys.stderr)
                    sys.exit(e.code)
    except RuntimeError as e:
        if "replay exhausted" in str(e):
            print(f"agentprobe: {e}", file=sys.stderr)
            sys.exit(1)
        raise

    consumed = index[0] if strict else len(calls)
    if strict and index[0] < len(calls):
        print(
            f"agentprobe: strict replay — fixture has {len(calls)} call(s) "
            f"but script only consumed {index[0]}.",
            file=sys.stderr,
        )
        sys.exit(1)

    async_note = " (async script)" if is_async else ""
    print(f"agentprobe: replayed {consumed} call(s) from {fixture}{async_note}")


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

    # Load .env file if --env was specified
    env_file = getattr(args, "env", None)
    if env_file:
        env_path = Path(env_file)
        if not env_path.exists():
            print(f"agentprobe: env file not found: {env_file}", file=sys.stderr)
            sys.exit(1)
        import os
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

    output_str = args.output
    fmt = getattr(args, "output_format", None)
    if fmt == "gz" and not output_str.endswith(".gz"):
        output_str = output_str + ".gz"
    output = Path(output_str)
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

    timeout_s = getattr(args, "timeout", None)

    capture_stdout = getattr(args, "capture_stdout", False)
    import io, contextlib
    stdout_buf = io.StringIO() if capture_stdout else None
    stderr_buf = io.StringIO() if capture_stdout else None

    def _run_script():
        runpy.run_path(str(script), run_name="__main__")

    def _run_with_capture():
        if capture_stdout:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                _run_script()
        else:
            _run_script()

    with recording_context(calls):
        with patch.object(openai.resources.chat.completions.AsyncCompletions, "create", _async_patch):
            if timeout_s:
                import threading
                exc_box: list = []
                def _target():
                    try:
                        _run_with_capture()
                    except SystemExit:
                        pass
                    except Exception as e:
                        exc_box.append(e)
                t = threading.Thread(target=_target, daemon=True)
                t.start()
                t.join(timeout=timeout_s)
                if t.is_alive():
                    print(f"agentprobe: script timed out after {timeout_s}s — "
                          f"saving {len(calls)} call(s) recorded so far", file=sys.stderr)
                elif exc_box:
                    raise exc_box[0]
            else:
                try:
                    _run_with_capture()
                except SystemExit:
                    pass

    dry_run = getattr(args, "dry_run", False)
    append = getattr(args, "append", False)
    async_note = " (async script)" if is_async else ""
    gz_note = " [gzip]" if output.suffix == ".gz" else ""

    meta_extra = None
    if capture_stdout and (stdout_buf.getvalue() or stderr_buf.getvalue()):
        meta_extra = {}
        if stdout_buf.getvalue():
            meta_extra["stdout"] = stdout_buf.getvalue()
        if stderr_buf.getvalue():
            meta_extra["stderr"] = stderr_buf.getvalue()

    if dry_run:
        print(f"agentprobe: dry-run — would record {len(calls)} call(s) to {output}{async_note}")
        for i, c in enumerate(calls, 1):
            model = c.request.get("model", "?")
            tokens = (c.response.get("usage") or {}).get("prompt_tokens", "?")
            print(f"  call {i}: model={model} prompt_tokens={tokens}")
        if capture_stdout and meta_extra:
            print(f"  captured stdout ({len(meta_extra.get('stdout',''))} chars), "
                  f"stderr ({len(meta_extra.get('stderr',''))} chars)")
    elif append:
        from agentprobe._session import _load_calls as _lc
        existing = _lc(output) if output.exists() else []
        _save_calls(existing + calls, output, meta_extra)
        print(f"agentprobe: appended {len(calls)} call(s) to {output} "
              f"(total: {len(existing) + len(calls)}){async_note}{gz_note}")
    else:
        _save_calls(calls, output, meta_extra)
        stdout_note = " [+stdout]" if meta_extra else ""
        print(f"agentprobe: recorded {len(calls)} call(s) to {output}{async_note}{gz_note}{stdout_note}")


def cmd_record_watch(args):
    """Watch a script for changes and re-record automatically."""
    import time
    interval = getattr(args, "interval", 1.0)
    last_mtime = None
    print(f"agentprobe: watching {args.script} (interval {interval}s, Ctrl+C to stop)...")
    try:
        while True:
            try:
                mtime = Path(args.script).stat().st_mtime
            except FileNotFoundError:
                time.sleep(interval)
                continue
            if mtime != last_mtime:
                if last_mtime is not None:
                    print(f"\nagentprobe: change detected — re-recording...")
                last_mtime = mtime
                try:
                    cmd_record(args)
                except Exception as e:
                    print(f"agentprobe: record failed: {e}", file=sys.stderr)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nagentprobe: watch stopped.")


def main():
    parser = argparse.ArgumentParser(
        prog="agentprobe",
        description="Inspect, diff, and record agentprobe session fixtures",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_show = sub.add_parser("show", help="Pretty-print a session fixture")
    p_show.add_argument("fixture", help="Path to .jsonl fixture file")
    p_show.add_argument("--json", action="store_true", help="Output as machine-readable JSON")
    p_show.add_argument("--model", metavar="MODEL", help="Filter to calls using this model")
    p_show.set_defaults(func=lambda a: cmd_show_json(a) if a.json else cmd_show(a))

    p_diff = sub.add_parser("diff", help="Compare two session fixtures")
    p_diff.add_argument("fixture_a")
    p_diff.add_argument("fixture_b")
    p_diff.add_argument("--json", action="store_true", help="Output as machine-readable JSON")
    p_diff.add_argument("--by-call", action="store_true",
                        help="Human-readable side-by-side per-call comparison")
    p_diff.set_defaults(func=lambda a: cmd_diff_by_call(a) if a.by_call else (cmd_diff_json(a) if a.json else cmd_diff(a)))

    p_list = sub.add_parser("fixtures", help="List or manage fixture files in a directory")
    p_list.add_argument("directory", nargs="?", default="tests/fixtures",
                        help="Directory to search (default: tests/fixtures)")
    p_list.add_argument("--json", action="store_true", help="Output as machine-readable JSON")
    p_list.add_argument("--clean", action="store_true", help="Remove stale .lock files")
    p_list.add_argument("--by-date", action="store_true", help="Group stats by recording date")
    p_list.set_defaults(func=lambda a: cmd_fixtures_clean(a) if a.clean else (
        cmd_stats_by_date(a) if a.by_date else cmd_fixtures_list(a)))

    p_stats = sub.add_parser("stats", help="Aggregate stats across all fixtures in a directory")
    p_stats.add_argument("directory", nargs="?", default="tests/fixtures",
                         help="Directory to scan (default: tests/fixtures)")
    p_stats.add_argument("--json", action="store_true", help="Output as machine-readable JSON")
    p_stats.add_argument("--by-date", action="store_true", help="Group stats by recording date")
    p_stats.set_defaults(func=lambda a: cmd_stats_by_date(a) if a.by_date else cmd_fixtures_stats(a))

    p_migrate = sub.add_parser("migrate", help="Transform a fixture: rename models/tools")
    p_migrate.add_argument("input", help="Input .jsonl fixture")
    p_migrate.add_argument("output", help="Output .jsonl fixture")
    p_migrate.add_argument("--rename-model", dest="rename_model", action="append",
                           metavar="OLD=NEW", help="Rename a model (repeatable)")
    p_migrate.add_argument("--rename-tool", dest="rename_tool", action="append",
                           metavar="OLD=NEW", help="Rename a tool (repeatable)")
    p_migrate.add_argument("--set-model", dest="set_model", metavar="MODEL",
                           help="Force all calls to use this model")
    p_migrate.set_defaults(func=cmd_migrate)

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
    p_record.add_argument("--env", metavar="FILE",
                          help="Load environment variables from FILE before running the script")
    p_record.add_argument("--output-format", choices=["jsonl", "gz"],
                          help="Force output format (gz adds .gz suffix if missing)")
    p_record.add_argument("--watch", action="store_true",
                          help="Watch script for changes and re-record automatically")
    p_record.add_argument("--interval", type=float, default=1.0,
                          help="Polling interval in seconds for --watch (default: 1.0)")
    p_record.add_argument("--timeout", type=float, metavar="SECONDS",
                          help="Kill the script after SECONDS (saves calls recorded so far)")
    p_record.add_argument("--dry-run", action="store_true",
                          help="Run script and show what would be captured, but don't save")
    p_record.add_argument("--append", action="store_true",
                          help="Append captured calls to an existing fixture (don't overwrite)")
    p_record.add_argument("--capture-stdout", dest="capture_stdout", action="store_true",
                          help="Capture script stdout/stderr and store in fixture _meta header")
    p_record.set_defaults(func=lambda a: cmd_record_watch(a) if a.watch else cmd_record(a))

    p_replay = sub.add_parser(
        "replay",
        help="Run a Python script in pure replay mode against a saved fixture",
    )
    p_replay.add_argument("fixture", help="Path to .jsonl fixture file")
    p_replay.add_argument("script", help="Path to Python script to run against the fixture")
    p_replay.add_argument("--provider", choices=["openai", "anthropic"], default="openai",
                          help="API provider to intercept (default: openai)")
    p_replay.add_argument("--strict", action="store_true",
                          help="Fail if the script doesn't consume every call in the fixture")
    p_replay.add_argument("--env", metavar="FILE",
                          help="Load environment variables from FILE before running")
    p_replay.set_defaults(func=cmd_replay)

    p_validate = sub.add_parser("validate", help="Validate a fixture file for correctness")
    p_validate.add_argument("fixture", help="Path to .jsonl fixture file")
    p_validate.add_argument("--strict", action="store_true",
                            help="Treat lint warnings as errors (exit 1 if any)")
    p_validate.set_defaults(func=cmd_validate)

    p_init = sub.add_parser("init", help="Scaffold tests/fixtures/ and a sample conftest.py")
    p_init.set_defaults(func=cmd_init)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
