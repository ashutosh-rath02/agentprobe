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


def _is_anthropic_response(resp: dict) -> bool:
    """Return True if *resp* is an Anthropic Message dict (not OpenAI choices format)."""
    return "choices" not in resp and isinstance(resp.get("content"), list)


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


def _fmt_anthropic_content(resp: dict) -> list[str]:
    lines = []
    for block in (resp.get("content") or []):
        btype = block.get("type")
        if btype == "text":
            text = block.get("text", "")
            lines.append(f"  [text] {text[:120]}{'...' if len(text) > 120 else ''}")
        elif btype == "tool_use":
            name = block.get("name", "?")
            inp = json.dumps(block.get("input", {}))
            lines.append(f"  [tool_use] {name}({inp[:100]}{'...' if len(inp) > 100 else ''})")
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
    total_in = 0
    total_out = 0
    for i, call in enumerate(calls, 1):
        req = call.get("request", {})
        resp = call.get("response", {})
        usage = resp.get("usage") or {}
        ms = call.get("duration_ms")
        ms_str = f"  {ms:.0f}ms" if ms else ""
        streaming = " [stream]" if call.get("chunks") else ""
        model = resp.get("model") or req.get("model", "?")
        if _is_anthropic_response(resp):
            stop = resp.get("stop_reason", "?")
            in_tok = usage.get("input_tokens", "?")
            out_tok = usage.get("output_tokens", "?")
            total_in += in_tok if isinstance(in_tok, int) else 0
            total_out += out_tok if isinstance(out_tok, int) else 0
            print(f"-- Call {i}/{len(calls)}  model={model}  stop={stop}"
                  f"  in={in_tok} out={out_tok}{ms_str}{streaming}")
            for line in _fmt_anthropic_content(resp):
                print(line)
        else:
            choices = resp.get("choices", [])
            finish = choices[0]["finish_reason"] if choices else "?"
            in_tok = usage.get("prompt_tokens", "?")
            out_tok = usage.get("completion_tokens", "?")
            total_in += in_tok if isinstance(in_tok, int) else 0
            total_out += out_tok if isinstance(out_tok, int) else 0
            print(f"-- Call {i}/{len(calls)}  model={model}  stop={finish}"
                  f"  in={in_tok} out={out_tok}{ms_str}{streaming}")
            for choice in choices:
                for line in _fmt_choice(choice):
                    print(line)
        print()

    print(f"total tokens: {total_in + total_out}  ({total_in} in + {total_out} out)")

    if getattr(args, "stdout", False):
        meta = _load_meta(args.fixture)
        captured = meta.get("stdout", "")
        err_captured = meta.get("stderr", "")
        if captured or err_captured:
            print(f"\n--- captured stdout ---")
            if captured:
                print(captured.rstrip())
            if err_captured:
                print(f"\n--- captured stderr ---")
                print(err_captured.rstrip())
        else:
            print("\n(no captured stdout in _meta — record with --capture-stdout)")


def cmd_show_json(args):
    from agentprobe._pricing import estimate_cost, estimate_cost_anthropic
    calls = _filter_calls(_load(args.fixture), getattr(args, "model", None))
    out = []
    total_in = 0
    total_out = 0
    total_duration = 0.0
    total_cost = 0.0
    for call in calls:
        resp = call.get("response", {})
        usage = resp.get("usage") or {}
        chunks = call.get("chunks")
        dur = call.get("duration_ms") or 0.0
        model = resp.get("model") or call.get("request", {}).get("model") or ""
        if _is_anthropic_response(resp):
            in_tok = usage.get("input_tokens") or 0
            out_tok = usage.get("output_tokens") or 0
            cost = estimate_cost_anthropic(model, in_tok, out_tok)
            content = resp.get("content") or []
            tools = [b["name"] for b in content if b.get("type") == "tool_use"]
            final_text = next((b["text"] for b in reversed(content) if b.get("type") == "text" and b.get("text")), None)
            entry = {
                "provider": "anthropic",
                "model": model or None,
                "stop_reason": resp.get("stop_reason"),
                "input_tokens": in_tok or None,
                "output_tokens": out_tok or None,
                "duration_ms": dur or None,
                "streaming": bool(chunks),
                "chunk_count": len(chunks) if chunks else None,
                "estimated_cost_usd": cost if cost else None,
                "tools_called": tools,
                "final_text": final_text,
            }
        else:
            choices = resp.get("choices", [])
            in_tok = usage.get("prompt_tokens") or 0
            out_tok = usage.get("completion_tokens") or 0
            cost = estimate_cost(model, in_tok, out_tok)
            entry = {
                "provider": "openai",
                "model": model or None,
                "finish_reason": choices[0]["finish_reason"] if choices else None,
                "prompt_tokens": in_tok or None,
                "completion_tokens": out_tok or None,
                "duration_ms": dur or None,
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
        total_in += in_tok
        total_out += out_tok
        total_duration += dur
        total_cost += cost
        out.append(entry)
    result = {
        "calls": out,
        "summary": {
            "total_calls": len(out),
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_tokens": total_in + total_out,
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

        def _stop(resp):
            if _is_anthropic_response(resp):
                return resp.get("stop_reason")
            ch = resp.get("choices", [])
            return ch[0]["finish_reason"] if ch else None

        def _tools(resp):
            if _is_anthropic_response(resp):
                return [b["name"] for b in (resp.get("content") or []) if b.get("type") == "tool_use"]
            ch = resp.get("choices", [])
            return [tc["function"]["name"] for c in ch for tc in (c["message"].get("tool_calls") or [])]

        def _in_tok(resp):
            u = resp.get("usage") or {}
            return u.get("input_tokens") or u.get("prompt_tokens", 0)

        a_stop, b_stop = _stop(a_resp), _stop(b_resp)
        if a_stop != b_stop:
            print(f"  call {i+1}: stop_reason {a_stop!r} -> {b_stop!r}")
            diffs += 1

        a_tools, b_tools = _tools(a_resp), _tools(b_resp)
        if a_tools != b_tools:
            print(f"  call {i+1}: tools {a_tools} -> {b_tools}")
            diffs += 1

        a_in, b_in = _in_tok(a_resp), _in_tok(b_resp)
        if a_in != b_in:
            print(f"  call {i+1}: input_tokens {a_in} -> {b_in}")
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

        ar, br = a.get("response", {}), b.get("response", {})

        def _stop_j(resp):
            if _is_anthropic_response(resp):
                return resp.get("stop_reason")
            ch = resp.get("choices", [])
            return ch[0]["finish_reason"] if ch else None

        def _tools_j(resp):
            if _is_anthropic_response(resp):
                return [b["name"] for b in (resp.get("content") or []) if b.get("type") == "tool_use"]
            ch = resp.get("choices", [])
            return [tc["function"]["name"] for c in ch for tc in (c["message"].get("tool_calls") or [])]

        def _tool_args_j(resp):
            if _is_anthropic_response(resp):
                return {b["name"]: json.dumps(b.get("input", {})) for b in (resp.get("content") or []) if b.get("type") == "tool_use"}
            ch = resp.get("choices", [])
            return {tc["function"]["name"]: tc["function"].get("arguments") for c in ch for tc in (c["message"].get("tool_calls") or [])}

        def _text_j(resp):
            if _is_anthropic_response(resp):
                return next((b["text"] for b in reversed(resp.get("content") or []) if b.get("type") == "text" and b.get("text")), None)
            ch = resp.get("choices", [])
            return next((c["message"].get("content") for c in reversed(ch) if c["message"].get("content")), None)

        def _in_tok_j(resp):
            u = resp.get("usage") or {}
            return u.get("input_tokens") or u.get("prompt_tokens", 0)

        a_stop, b_stop = _stop_j(ar), _stop_j(br)
        if a_stop != b_stop:
            differences.append({"call": i + 1, "type": "stop_reason", "a": a_stop, "b": b_stop})

        a_tools, b_tools = _tools_j(ar), _tools_j(br)
        if a_tools != b_tools:
            differences.append({"call": i + 1, "type": "tools", "a": a_tools, "b": b_tools})

        a_args, b_args = _tool_args_j(ar), _tool_args_j(br)
        for name in set(a_args) & set(b_args):
            if a_args[name] != b_args[name]:
                differences.append({"call": i + 1, "type": "tool_arguments", "tool": name, "a": a_args[name], "b": b_args[name]})

        a_content, b_content = _text_j(ar), _text_j(br)
        if a_content != b_content:
            differences.append({"call": i + 1, "type": "content", "a": a_content, "b": b_content})

        a_in, b_in = _in_tok_j(ar), _in_tok_j(br)
        if a_in != b_in:
            differences.append({"call": i + 1, "type": "input_tokens", "a": a_in, "b": b_in})

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
    strip_pii = getattr(args, "strip_pii", None) or []

    import re as _re
    lines = [l for l in p.read_text().splitlines() if l.strip()]
    transformed = 0
    redactions = 0

    def _redact(text: str) -> str:
        nonlocal redactions
        for pat in strip_pii:
            result, count = _re.subn(pat, "[REDACTED]", text)
            redactions += count
            text = result
        return text

    out_lines = []
    for line in lines:
        data = json.loads(line)
        if "_meta" in data:
            out_lines.append(line)
            continue
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

        # Rename tools in response choices / redact args
        for choice in resp.get("choices", []):
            for tc in (choice.get("message", {}).get("tool_calls") or []):
                old = tc.get("function", {}).get("name", "")
                if old in rename_tools:
                    tc["function"]["name"] = rename_tools[old]
                    transformed += 1
                if strip_pii and tc.get("function", {}).get("arguments"):
                    tc["function"]["arguments"] = _redact(tc["function"]["arguments"])

        # Rename tools / redact in Anthropic format
        for block in (resp.get("content") or []):
            if block.get("type") == "tool_use":
                if block.get("name") in rename_tools:
                    block["name"] = rename_tools[block["name"]]
                    transformed += 1
                if strip_pii and block.get("input"):
                    inp_str = _redact(json.dumps(block["input"]))
                    try:
                        block["input"] = json.loads(inp_str)
                    except json.JSONDecodeError:
                        block["input"] = inp_str

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
    pii_note = f"  ({redactions} redaction(s))" if redactions else ""
    print(f"agentprobe: migrated {len(lines)} call(s) to {out_path}  ({transformed} tool rename(s)){pii_note}")


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

        if "_meta" in data:
            continue  # skip meta header lines

        for field in ("request", "response"):
            if field not in data:
                errors.append(f"line {i}: missing '{field}' field")

        # Attempt Pydantic deserialization of the assembled response
        resp = data.get("response", {})

        if _is_anthropic_response(resp):
            # Anthropic format
            try:
                import anthropic.types as _at
                _at.Message.model_validate(resp)
            except Exception as e:
                errors.append(f"line {i}: Anthropic response failed Pydantic validation — {e}")
            chunks = data.get("chunks")
            if chunks is not None:
                if not isinstance(chunks, list):
                    errors.append(f"line {i}: 'chunks' must be a list")
                else:
                    _event_mapping = {
                        "message_start": "RawMessageStartEvent",
                        "content_block_start": "RawContentBlockStartEvent",
                        "content_block_delta": "RawContentBlockDeltaEvent",
                        "content_block_stop": "RawContentBlockStopEvent",
                        "message_delta": "RawMessageDeltaEvent",
                        "message_stop": "RawMessageStopEvent",
                    }
                    for j, chunk in enumerate(chunks):
                        etype = chunk.get("type", "")
                        cls_name = _event_mapping.get(etype)
                        if cls_name:
                            try:
                                getattr(_at, cls_name).model_validate(chunk)
                            except Exception as e:
                                errors.append(f"line {i} chunk {j}: Anthropic event validation failed — {e}")
        else:
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
        if "_meta" in data:
            continue
        if data.get("duration_ms") is None:
            warnings.append(f"line {i}: no duration_ms (recorded without timing — replay-only fixture)")
        resp = data.get("response", {})
        model = resp.get("model") or data.get("request", {}).get("model")
        if not model:
            warnings.append(f"line {i}: request has no model field")

    for w in warnings:
        print(f"  WARN: {w}")

    if warnings and getattr(args, "strict", False):
        print(f"\nagentprobe: {len(warnings)} warning(s) treated as errors (--strict)", file=sys.stderr)
        sys.exit(1)

    data_lines = [json.loads(l) for l in lines if "_meta" not in json.loads(l)]
    streaming = sum(1 for d in data_lines if d.get("chunks"))
    status = f"OK ({len(warnings)} warning(s))" if warnings else "OK"
    print(f"agentprobe: {args.fixture} {status}  ({len(data_lines)} call(s), {streaming} streaming)")


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


def cmd_fixtures_summarize(args):
    """Print a one-line summary for every fixture file in a directory."""
    from agentprobe._pricing import estimate_cost, estimate_cost_anthropic
    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"agentprobe: not a directory: {args.directory}", file=sys.stderr)
        sys.exit(1)

    fixtures = sorted(directory.rglob("*.jsonl")) + sorted(directory.rglob("*.jsonl.gz"))
    if not fixtures:
        print(f"agentprobe: no fixtures found in {args.directory}")
        return

    rows = []
    for f in fixtures:
        try:
            calls = _load(str(f))
        except Exception:
            continue
        meta = _load_meta(str(f))
        recorded = meta.get("recorded_at", "")[:10] if meta else ""
        total_in = total_out = 0
        tools: list = []
        models: set = set()
        for c in calls:
            resp = c.get("response", {})
            usage = resp.get("usage") or {}
            model = resp.get("model") or c.get("request", {}).get("model", "")
            models.add(model)
            if _is_anthropic_response(resp):
                total_in += usage.get("input_tokens", 0) or 0
                total_out += usage.get("output_tokens", 0) or 0
                for block in (resp.get("content") or []):
                    if block.get("type") == "tool_use":
                        tools.append(block["name"])
            else:
                total_in += usage.get("prompt_tokens", 0) or 0
                total_out += usage.get("completion_tokens", 0) or 0
                for ch in resp.get("choices", []):
                    for tc in (ch["message"].get("tool_calls") or []):
                        tools.append(tc["function"]["name"])
        unique_tools = sorted(set(tools))
        rows.append({
            "file": str(f.relative_to(directory) if f.is_relative_to(directory) else f),
            "calls": len(calls), "in": total_in, "out": total_out,
            "tools": ",".join(unique_tools) or "(none)",
            "models": ",".join(sorted(models - {""})) or "?",
            "date": recorded,
        })

    if getattr(args, "json", False):
        print(json.dumps(rows, indent=2))
    else:
        for r in rows:
            tools_str = f"  tools=[{r['tools']}]" if r["tools"] != "(none)" else ""
            print(f"  {r['file']:<40} {r['calls']:>3} call(s)  "
                  f"{r['in']:>6} in / {r['out']:>5} out  {r['date']}{tools_str}")


def cmd_compare_score(args):
    """Compute a structural similarity score (0–100) between two fixtures."""
    a_calls = _load(args.fixture_a)
    b_calls = _load(args.fixture_b)

    if not a_calls and not b_calls:
        print(json.dumps({"score": 100, "reason": "both empty"}) if getattr(args, "json", False)
              else "agentprobe: score 100/100 (both empty)")
        return

    total_checks = 0
    matches = 0

    # Call count match
    total_checks += 1
    if len(a_calls) == len(b_calls):
        matches += 1

    # Per-call structural comparison
    for i in range(min(len(a_calls), len(b_calls))):
        ar, br = a_calls[i].get("response", {}), b_calls[i].get("response", {})

        def _stop(resp):
            if _is_anthropic_response(resp):
                return resp.get("stop_reason")
            ch = resp.get("choices", [])
            return ch[0]["finish_reason"] if ch else None

        def _tools(resp):
            if _is_anthropic_response(resp):
                return sorted(b["name"] for b in (resp.get("content") or []) if b.get("type") == "tool_use")
            ch = resp.get("choices", [])
            return sorted(tc["function"]["name"] for c in ch for tc in (c["message"].get("tool_calls") or []))

        def _model(resp, req):
            return resp.get("model") or req.get("model", "")

        # stop_reason/finish_reason
        total_checks += 1
        if _stop(ar) == _stop(br):
            matches += 1

        # tools
        total_checks += 1
        if _tools(ar) == _tools(br):
            matches += 1

        # model
        total_checks += 1
        a_model = _model(ar, a_calls[i].get("request", {}))
        b_model = _model(br, b_calls[i].get("request", {}))
        if a_model == b_model:
            matches += 1

    score = round((matches / total_checks) * 100) if total_checks else 100
    diff_count = total_checks - matches

    if getattr(args, "json", False):
        print(json.dumps({
            "score": score,
            "matches": matches,
            "total_checks": total_checks,
            "differences": diff_count,
        }))
    else:
        bar = "#" * (score // 5) + "." * (20 - score // 5)
        print(f"agentprobe compare: {args.fixture_a}  vs  {args.fixture_b}")
        print(f"  similarity: {score:>3}/100  [{bar}]  ({diff_count} difference(s))")
        if score == 100:
            print("  Fixtures are structurally identical.")
        elif score >= 80:
            print("  Minor differences — likely a safe refactor.")
        elif score >= 50:
            print("  Moderate differences — review before merging.")
        else:
            print("  Significant differences — fixtures may represent different agent behaviors.")


def cmd_fixtures_by_age(args):
    """List fixture files older than N days (by _meta.recorded_at)."""
    from datetime import datetime, timezone
    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"agentprobe: not a directory: {args.directory}", file=sys.stderr)
        sys.exit(1)

    age_days = args.age_days
    now = datetime.now(timezone.utc)
    fixtures = sorted(directory.rglob("*.jsonl")) + sorted(directory.rglob("*.jsonl.gz"))
    old_fixtures = []
    for f in fixtures:
        meta = _load_meta(str(f))
        recorded = meta.get("recorded_at", "")
        if not recorded:
            continue
        try:
            ts = datetime.fromisoformat(recorded.replace("Z", "+00:00"))
            age = (now - ts).days
            if age >= age_days:
                old_fixtures.append({"file": str(f), "age_days": age, "recorded_at": recorded})
        except ValueError:
            continue

    if not old_fixtures:
        print(f"agentprobe: no fixtures older than {age_days} day(s) found in {args.directory}")
        return

    if getattr(args, "json", False):
        print(json.dumps(old_fixtures, indent=2))
    else:
        print(f"Fixtures older than {age_days} day(s) in {args.directory}:\n")
        for entry in old_fixtures:
            print(f"  {entry['file']}  ({entry['age_days']} days, recorded {entry['recorded_at'][:10]})")
        print(f"\n{len(old_fixtures)} fixture(s)")


def cmd_fixtures_by_label(args):
    """List fixtures that have a specific label in their _meta header."""
    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"agentprobe: not a directory: {args.directory}", file=sys.stderr)
        sys.exit(1)

    label_filter = args.label
    fixtures = sorted(directory.rglob("*.jsonl")) + sorted(directory.rglob("*.jsonl.gz"))
    matches = []
    for f in fixtures:
        meta = _load_meta(str(f))
        if meta.get("label") == label_filter:
            matches.append(str(f))

    if not matches:
        print(f"agentprobe: no fixtures with label '{label_filter}' found in {args.directory}")
        return

    if getattr(args, "json", False):
        print(json.dumps({"label": label_filter, "fixtures": matches}))
    else:
        print(f"Fixtures with label='{label_filter}' in {args.directory}:\n")
        for path in matches:
            print(f"  {path}")
        print(f"\n{len(matches)} fixture(s)")


def cmd_fixtures_orphaned(args):
    """List fixture files not referenced in any Python test file."""
    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"agentprobe: not a directory: {args.directory}", file=sys.stderr)
        sys.exit(1)

    # Collect all fixture filenames (stem + suffix combos users might write)
    fixtures = list(directory.rglob("*.jsonl")) + list(directory.rglob("*.jsonl.gz"))

    # Collect all Python test file content under tests/
    test_root = Path("tests")
    test_content = ""
    if test_root.is_dir():
        for pyfile in test_root.rglob("*.py"):
            try:
                test_content += pyfile.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                pass

    orphaned = []
    for fixture in sorted(fixtures):
        name = fixture.name
        stem = fixture.stem  # e.g. "my_agent" from "my_agent.jsonl"
        if name not in test_content and stem not in test_content:
            orphaned.append(str(fixture))

    if not orphaned:
        print(f"agentprobe: no orphaned fixtures found in {args.directory}")
        return

    if getattr(args, "json", False):
        print(json.dumps({"orphaned": orphaned}))
    else:
        print(f"Orphaned fixtures (not referenced in tests/):\n")
        for path in orphaned:
            print(f"  {path}")
        print(f"\n{len(orphaned)} orphaned fixture(s)")


def cmd_stats_by_model(args):
    """Aggregate stats grouped by model name across all fixtures in a directory."""
    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"agentprobe: not a directory: {args.directory}", file=sys.stderr)
        sys.exit(1)

    from agentprobe._pricing import estimate_cost, estimate_cost_anthropic
    by_model: dict = {}
    for f in sorted(directory.rglob("*.jsonl")) + sorted(directory.rglob("*.jsonl.gz")):
        try:
            calls = _load(str(f))
        except Exception:
            continue
        for c in calls:
            resp = c.get("response", {})
            usage = resp.get("usage") or {}
            model = resp.get("model") or c.get("request", {}).get("model", "unknown")
            if _is_anthropic_response(resp):
                inp = usage.get("input_tokens", 0) or 0
                out = usage.get("output_tokens", 0) or 0
                cost = estimate_cost_anthropic(model, inp, out)
            else:
                inp = usage.get("prompt_tokens", 0) or 0
                out = usage.get("completion_tokens", 0) or 0
                cost = estimate_cost(model, inp, out)
            entry = by_model.setdefault(model, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0})
            entry["calls"] += 1
            entry["input_tokens"] += inp
            entry["output_tokens"] += out
            entry["cost"] += cost

    if getattr(args, "json", False):
        print(json.dumps(by_model, indent=2))
    else:
        print(f"Stats by model ({args.directory}):\n")
        for model in sorted(by_model):
            e = by_model[model]
            print(f"  {model:<40} {e['calls']:>5} call(s)  "
                  f"{e['input_tokens']:>8,} in  {e['output_tokens']:>8,} out  ${e['cost']:.4f}")


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
    provider = getattr(args, "provider", "openai")

    # Detect whether the script is async (contains 'asyncio.run' or 'async def main').
    script_source = script.read_text()
    is_async = "asyncio.run(" in script_source or "async def main" in script_source

    timeout_s = getattr(args, "timeout", None)
    max_calls = getattr(args, "max_calls", None)

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

    if provider == "anthropic":
        from agentprobe._anthropic_interceptor import (
            anthropic_recording_context, async_anthropic_recording_context,
        )
        import anthropic.resources.messages

        orig_async_anth = anthropic.resources.messages.AsyncMessages.create

        async def _async_anth_patch(self, **kwargs):
            from agentprobe._anthropic_interceptor import (
                _serialize_anthropic_request, _assemble_anthropic_from_events,
                MockAnthropicAsyncStream,
            )
            start = time.time()
            if kwargs.get("stream"):
                real = await orig_async_anth(self, **kwargs)
                raw = [e async for e in real]
                ser = [e.model_dump() for e in raw]
                assembled = _assemble_anthropic_from_events(ser)
                calls.append(RecordedCall(
                    request=_serialize_anthropic_request(kwargs),
                    response=assembled, chunks=ser,
                    duration_ms=(time.time() - start) * 1000,
                ))
                return MockAnthropicAsyncStream(raw, assembled)
            resp = await orig_async_anth(self, **kwargs)
            calls.append(RecordedCall(
                request=_serialize_anthropic_request(kwargs),
                response=resp.model_dump(),
                duration_ms=(time.time() - start) * 1000,
            ))
            return resp

        outer_ctx = anthropic_recording_context(calls)
        async_patch_ctx = patch.object(
            anthropic.resources.messages.AsyncMessages, "create", _async_anth_patch
        )
    else:
        # Default: OpenAI
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

        outer_ctx = recording_context(calls)
        async_patch_ctx = patch.object(
            openai.resources.chat.completions.AsyncCompletions, "create", _async_patch
        )

    with outer_ctx:
        with async_patch_ctx:
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

    if max_calls is not None and len(calls) > max_calls:
        print(f"agentprobe: truncating to {max_calls} call(s) (recorded {len(calls)})")
        calls = calls[:max_calls]

    async_note = " (async script)" if is_async else ""
    gz_note = " [gzip]" if output.suffix == ".gz" else ""

    label = getattr(args, "label", None)
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
        _save_calls(existing + calls, output, meta_extra, label)
        print(f"agentprobe: appended {len(calls)} call(s) to {output} "
              f"(total: {len(existing) + len(calls)}){async_note}{gz_note}")
    else:
        _save_calls(calls, output, meta_extra, label)
        label_note = f" [label={label!r}]" if label else ""
        stdout_note = " [+stdout]" if meta_extra else ""
        print(f"agentprobe: recorded {len(calls)} call(s) to {output}{async_note}{gz_note}{stdout_note}{label_note}")


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
    p_show.add_argument("--stdout", action="store_true",
                        help="Also print captured stdout/stderr from _meta header")
    p_show.set_defaults(func=lambda a: cmd_show_json(a) if a.json else cmd_show(a))

    p_diff = sub.add_parser("diff", help="Compare two session fixtures")
    p_diff.add_argument("fixture_a")
    p_diff.add_argument("fixture_b")
    p_diff.add_argument("--json", action="store_true", help="Output as machine-readable JSON")
    p_diff.add_argument("--by-call", action="store_true",
                        help="Human-readable side-by-side per-call comparison")
    p_diff.set_defaults(func=lambda a: cmd_diff_by_call(a) if a.by_call else (cmd_diff_json(a) if a.json else cmd_diff(a)))

    p_compare = sub.add_parser("compare", help="Structural similarity score between two fixtures")
    p_compare.add_argument("fixture_a")
    p_compare.add_argument("fixture_b")
    p_compare.add_argument("--score", action="store_true",
                           help="Output similarity score (0-100) — always on, kept for ergonomics")
    p_compare.add_argument("--json", action="store_true", help="Machine-readable JSON output")
    p_compare.set_defaults(func=cmd_compare_score)

    p_list = sub.add_parser("fixtures", help="List or manage fixture files in a directory")
    p_list.add_argument("directory", nargs="?", default="tests/fixtures",
                        help="Directory to search (default: tests/fixtures)")
    p_list.add_argument("--json", action="store_true", help="Output as machine-readable JSON")
    p_list.add_argument("--clean", action="store_true", help="Remove stale .lock files")
    p_list.add_argument("--by-date", action="store_true", help="Group stats by recording date")
    p_list.add_argument("--orphaned", action="store_true",
                        help="List fixture files not referenced in any test file")
    p_list.add_argument("--summarize", action="store_true",
                        help="Print a one-line summary (calls, tokens, tools) per fixture")
    p_list.add_argument("--label", metavar="TAG",
                        help="List fixtures that have this label in their _meta header")
    p_list.add_argument("--age-days", dest="age_days", type=int, metavar="N",
                        help="List fixtures older than N days (by _meta.recorded_at)")
    p_list.set_defaults(func=lambda a: cmd_fixtures_clean(a) if a.clean else (
        cmd_stats_by_date(a) if a.by_date else (
        cmd_fixtures_orphaned(a) if a.orphaned else (
        cmd_fixtures_summarize(a) if a.summarize else (
        cmd_fixtures_by_label(a) if a.label else (
        cmd_fixtures_by_age(a) if a.age_days else cmd_fixtures_list(a)))))))

    p_stats = sub.add_parser("stats", help="Aggregate stats across all fixtures in a directory")
    p_stats.add_argument("directory", nargs="?", default="tests/fixtures",
                         help="Directory to scan (default: tests/fixtures)")
    p_stats.add_argument("--json", action="store_true", help="Output as machine-readable JSON")
    p_stats.add_argument("--by-date", action="store_true", help="Group stats by recording date")
    p_stats.add_argument("--by-model", action="store_true", help="Group stats by model name")
    p_stats.set_defaults(func=lambda a: (
        cmd_stats_by_date(a) if a.by_date else
        cmd_stats_by_model(a) if a.by_model else
        cmd_fixtures_stats(a)
    ))

    p_migrate = sub.add_parser("migrate", help="Transform a fixture: rename models/tools")
    p_migrate.add_argument("input", help="Input .jsonl fixture")
    p_migrate.add_argument("output", help="Output .jsonl fixture")
    p_migrate.add_argument("--rename-model", dest="rename_model", action="append",
                           metavar="OLD=NEW", help="Rename a model (repeatable)")
    p_migrate.add_argument("--rename-tool", dest="rename_tool", action="append",
                           metavar="OLD=NEW", help="Rename a tool (repeatable)")
    p_migrate.add_argument("--set-model", dest="set_model", metavar="MODEL",
                           help="Force all calls to use this model")
    p_migrate.add_argument("--strip-pii", dest="strip_pii", action="append", metavar="PATTERN",
                           help="Redact regex matches with [REDACTED] in tool inputs (repeatable)")
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
    p_record.add_argument("--provider", choices=["openai", "anthropic"], default="openai",
                          help="API provider to intercept (default: openai)")
    p_record.add_argument("--max-calls", dest="max_calls", type=int, metavar="N",
                          help="Stop after recording N calls (truncates fixture)")
    p_record.add_argument("--label", metavar="TAG",
                          help="Embed a custom label in the fixture _meta header")
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
