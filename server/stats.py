#!/usr/bin/env python3
"""Live stats for a searchbox job: token usage (the budget), tool distribution, activity feed.

Token totals come from the append-only Pi session JSONL (compaction-safe; see run_searchbox.py),
using the same fields Pi uses (input/output/cacheRead/cacheWrite). The BUDGET is `input`.
"""
import json, os, re, shlex, urllib.request
from pathlib import Path

BUDGET_METRIC = os.environ.get("BUDGET_METRIC", "input")
LLAMA_URL = os.environ.get("LLAMA_URL", "http://127.0.0.1:8080")


def llama_tps() -> dict:
    """Live throughput from llama.cpp Prometheus /metrics (needs --metrics). pi reports no tps,
    so this is the speed source (same as dataroom). Only meaningful while this job is the one
    actively running on the shared llama-server."""
    out = {}
    try:
        with urllib.request.urlopen(f"{LLAMA_URL}/metrics", timeout=3) as r:
            for line in r.read().decode(errors="ignore").splitlines():
                if line.startswith("#") or " " not in line:
                    continue
                k, v = line.rsplit(" ", 1)
                try:
                    val = float(v)
                except ValueError:
                    continue
                if k == "llamacpp:predicted_tokens_seconds":
                    out["decode"] = round(val, 1)
                elif k == "llamacpp:prompt_tokens_seconds":
                    out["prefill"] = round(val, 1)
    except Exception:
        pass
    return out


def read_timing(job_dir: Path) -> dict:
    p = job_dir / "timing.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {"wall_ms": 0, "llm_ms": 0, "tool_ms": 0, "by_tool_ms": {}, "by_tool_n": {}}


def _session_file(job_dir: Path):
    sd = job_dir / ".pi-agent" / "sessions"
    if not sd.exists():
        return None
    files = sorted(sd.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def session_usage(job_dir: Path) -> dict:
    """Cumulative assistant usage (compaction-safe) + last-call context occupancy.

    context_tokens = the latest assistant message's input+cacheRead, i.e. the actual prompt
    context the model held on its most recent call (what compaction works against)."""
    out = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
    last_ctx = 0
    sf = _session_file(job_dir)
    if not sf or not sf.exists():
        out["total"] = 0
        out["context_tokens"] = 0
        return out
    for line in open(sf, errors="ignore"):
        if '"usage"' not in line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        msg = o.get("message") if isinstance(o, dict) else None
        if not (isinstance(msg, dict) and msg.get("role") == "assistant"):
            continue
        u = msg.get("usage")
        if isinstance(u, dict):
            for k in out:
                out[k] += int(u.get(k) or 0)
            last_ctx = int(u.get("input") or 0) + int(u.get("cacheRead") or 0)
    out["total"] = out["input"] + out["output"] + out["cacheRead"] + out["cacheWrite"]
    out["context_tokens"] = last_ctx
    return out


# Job-internal plumbing the file tree should not surface (logs, configs, the raw upload,
# the pi agent + embedding cache dirs). Everything else (corpus/, ANSWER.md, NOTES.md, any
# scratch the agent writes) shows up in the tree.
_HIDE = {".pi-agent", ".corpus_cache", "input.zip", "pi.log", "corpus.log",
         "orchestrator.log", "meta.json", "run_meta.json", "query.txt", "control", "answer.zip"}


def _walk_tree(root: Path):
    total = 0

    def node(p: Path):
        nonlocal total
        if p.is_dir():
            ch = []
            for c in sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name)):
                if c.name in _HIDE or c.name.startswith("."):
                    continue
                ch.append(node(c))
            return {"name": p.name, "type": "dir", "children": ch}
        sz = p.stat().st_size
        total += sz
        return {"name": p.name, "type": "file", "size": sz}

    if not root.exists():
        return {"name": root.name, "type": "dir", "children": []}, 0
    return node(root), total


MAX_LOG_BYTES = 24 * 1024 * 1024


def _iter_events(log_path: Path):
    if not log_path.exists():
        return
    size = log_path.stat().st_size
    src = open(log_path, "rb")
    if size > MAX_LOG_BYTES:
        src.seek(size - MAX_LOG_BYTES); src.readline()
    try:
        for raw in src:
            if b'"type":"message_update"' in raw:
                continue
            if b"===== RPC SESSION" in raw:
                yield {"type": "_session_start"}; continue
            if not raw.lstrip().startswith(b"{"):
                continue
            try:
                yield json.loads(raw.decode("utf-8", "ignore"))
            except Exception:
                continue
    finally:
        src.close()


def _summarize(tool: str, args) -> str:
    a = args if isinstance(args, dict) else {}
    try:
        if tool == "semantic_search":
            return f"search {str(a.get('query',''))[:80]}".strip()
        if tool == "passage_rerank":
            return f"rerank {str(a.get('query',''))[:80]}".strip()
        if tool == "bash":
            cmd = re.sub(r"\s+", " ", str(a.get("command") or a.get("cmd") or "")).strip()
            return f"$ {cmd[:400]}".strip()
        if tool in ("read", "write", "edit"):
            return f"{tool} {str(a.get('path') or a.get('file') or '')[:80]}".strip()
    except Exception:
        pass
    return tool


# Shell operators after which a new command starts (so the next token is a program name).
_CMD_SEP = {"|", "||", "&&", ";", "&", "(", "{", "|&"}
# Keywords whose following token is also a command position.
_CMD_KW = {"then", "do", "else", "elif", "!", "time", "xargs", "sudo", "nohup", "command", "exec", "env"}
_PROG_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._+-]*$")


def _bash_programs(cmd: str):
    """Names of programs actually invoked in a bash command.

    Uses shlex so quoted text (e.g. grep -iE "v3|v4") is one token and never mistaken for a
    command. A program name is only taken at a command position: the very start, or right after
    a shell operator (| && || ; & ( {) or a command-introducing keyword. Redirections, flags,
    quoted args, VAR= prefixes, and pipe-pattern fragments are skipped."""
    try:
        toks = shlex.split(cmd, posix=True)
    except ValueError:
        toks = cmd.split()
    progs, expect = [], True  # expect a command name at the start
    for tk in toks:
        if tk in _CMD_SEP:
            expect = True
            continue
        if tk in _CMD_KW:
            expect = True  # the wrapper itself is plumbing; the real program is the next token
            continue
        if not expect:
            continue
        # at a command position now
        if "=" in tk and _PROG_RE.match(tk.split("=", 1)[0]):
            continue  # VAR=val prefix -> command is still the next token
        expect = False
        p = tk.split("/")[-1]
        if _PROG_RE.match(p):
            progs.append(p)
    return progs or ["bash"]


def parse_pi_log(log_path: Path) -> dict:
    tool_counts, tool_calls = {}, 0
    bash_counts = {}
    turns = steps = compactions = 0
    recent, errors = [], []
    for ev in _iter_events(log_path):
        t = ev.get("type")
        if t == "_session_start":
            errors = []
        elif t == "agent_start":
            turns += 1
        elif t == "turn_start":
            steps += 1
        elif t == "compaction_start":
            compactions += 1
            recent.append({"turn": turns, "tool": "compaction", "text": f"context compacted (#{compactions})"})
        elif t == "tool_execution_start":
            name = ev.get("toolName") or "unknown"
            tool_counts[name] = tool_counts.get(name, 0) + 1
            tool_calls += 1
            args = ev.get("args") or {}
            recent.append({"turn": turns, "tool": name, "text": _summarize(name, args)})
            if name == "bash":
                cmd = str(args.get("command") or args.get("cmd") or "")
                for prog in _bash_programs(cmd):
                    bash_counts[prog] = bash_counts.get(prog, 0) + 1
        elif t == "tool_execution_end":
            res = ev.get("result") or {}
            is_err = ev.get("isError") or (isinstance(res, dict) and res.get("isError"))
            if is_err:
                txt = ""
                if isinstance(res, dict):
                    cont = res.get("content")
                    if isinstance(cont, list) and cont:
                        txt = str((cont[0] or {}).get("text") or "")
                txt = txt or str(ev.get("error") or "tool error")
                tool = ev.get("toolName") or "?"
                benign = False
                if tool == "bash":
                    body = re.sub(r'(?i)command exited with code\s*\d+\.?', '', txt).replace("(no output)", "").strip()
                    benign = not body
                if not benign:
                    errors.append({"turn": turns, "tool": tool, "text": txt[:200]})
        elif t == "message_end":
            txt = (ev.get("message") or {}).get("text") or ev.get("text")
            if txt:
                recent.append({"turn": turns, "tool": "say", "text": str(txt)[:120]})
        if len(recent) > 60:
            recent = recent[-60:]
        if len(errors) > 50:
            errors = errors[-50:]
    return {
        "tool_calls": tool_calls,
        "tool_distribution": dict(sorted(tool_counts.items(), key=lambda kv: -kv[1])),
        "turns": turns, "steps": steps, "compactions": compactions,
        "recent": recent, "errors": errors,
        "bash_distribution": dict(sorted(bash_counts.items(), key=lambda kv: -kv[1])),
    }


def job_stats(job_dir: Path, budget: int = None, live: bool = False) -> dict:
    corpus = job_dir / "corpus"
    # Walk the whole job dir so the agent's outputs (ANSWER.md, NOTES.md, any scratch) appear
    # in the tree next to corpus/, not just the read-only corpus. Internal plumbing is hidden
    # by _walk_tree. Relabel the root to 'output' so it reads as the job's working dir.
    tree, size = _walk_tree(job_dir)
    tree["name"] = "output"
    log = parse_pi_log(job_dir / "pi.log")
    usage = session_usage(job_dir)
    spent = usage.get(BUDGET_METRIC, 0)
    ctx_window = int(os.environ.get("CONTEXT_WINDOW", os.environ.get("CTX_SIZE", "131072")))

    stop_reason, done = None, False
    rm = job_dir / "run_meta.json"
    bdg = budget
    if rm.exists():
        try:
            meta = json.loads(rm.read_text())
            stop_reason = meta.get("stop_reason")
            done = bool(meta.get("done"))
            bdg = bdg or meta.get("budget")
            if meta.get("tokens"):
                usage = meta["tokens"]; spent = usage.get(BUDGET_METRIC, spent)
        except Exception:
            pass
    bdg = bdg or int(os.environ.get("INPUT_TOKEN_BUDGET", "500000"))

    answer = ""
    ap = job_dir / "ANSWER.md"
    if ap.exists():
        answer = ap.read_text(errors="ignore")[:200000]

    return {
        **log,
        "budget_metric": BUDGET_METRIC,
        "tokens": usage,
        "budget": {"target": bdg, "spent": spent,
                   "percent": round(min(100, 100 * spent / bdg), 1) if bdg else 0},
        "context_window": ctx_window,
        "context_tokens": usage.get("context_tokens", 0),
        "context_percent": round(min(100, 100 * usage.get("context_tokens", 0) / ctx_window), 1) if ctx_window else 0,
        "corpus": {"tree": tree, "size_bytes": size,
                   "file_count": sum(1 for p in corpus.rglob("*") if p.is_file()) if corpus.exists() else 0},
        "answer_md": answer,
        "timing": read_timing(job_dir),
        "tps": llama_tps() if live else {},
        "stop_reason": stop_reason,
        "done": done,
        "zip_ready": (job_dir / "answer.zip").exists(),
    }
