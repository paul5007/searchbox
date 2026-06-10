#!/usr/bin/env python3
"""Orchestrator: run one searchbox job on a minimal Pi harness.

Inputs: a PROMPT, a CORPUS (.zip or folder), an INPUT-TOKEN BUDGET.
The agent answers the prompt grounded ONLY in the corpus, using two local-only retrieval tools
(jina-embeddings-v5-text-small + jina-reranker-v3 over the corpus). No web.

DESIGN: keep it minimal. Pi is already a complete agent (it loops, calls tools, and
auto-compacts its own context). We add exactly ONE thing on top of vanilla Pi: do not let the
run finish until it has SPENT the input-token budget. So this file only:
  1. unzips the corpus + boots the retrieval sidecar,
  2. sends the task once,
  3. re-nudges ("keep going") each time Pi goes idle while the budget is unspent,
  4. stops when budget is spent and ANSWER.md exists.
No STATUS files, no saturation/consolidation machinery, no extra prompt scaffolding.

TOKEN ACCOUNTING (verified against pi source): pi has no cumulative counter, and
get_session_stats sums only in-memory messages, which compaction prunes - so it UNDERCOUNTS a
long run. The session JSONL is append-only (compaction only appends a summary; it never deletes
the assistant entries that carry usage), so we sum usage from the session file. We use the SAME
fields pi uses (input/output/cacheRead/cacheWrite). The BUDGET is measured against `input`
(fresh prefill tokens the model actually processed); all four components are recorded for the UI.

ABLATION (all via env, no code edits):
  base model : LLAMA_URL / MODEL_ID / CONTEXT_WINDOW
  tools      : SEARCHBOX_TOOLS="corpus_search,corpus_rerank" | "corpus_search" | "" (none)
  retrieval  : EMBED_MODEL / RERANK_MODEL
  budget     : INPUT_TOKEN_BUDGET, and BUDGET_METRIC (default "input")
"""
import argparse, json, os, subprocess, sys, time, zipfile, signal, socket, threading, urllib.request, urllib.error, shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# Which session-usage field the budget is measured against. Default: fresh prefill ("input").
BUDGET_METRIC = os.environ.get("BUDGET_METRIC", "input")


def free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close()
    return p


def write_pi_config(agent_dir: Path, llama_url: str):
    """Per-job Pi config: default model = the configured local model. Compaction left ON
    (Pi's own, default settings) - we do not tune it."""
    agent_dir.mkdir(parents=True, exist_ok=True)
    ctx = int(os.environ.get("CONTEXT_WINDOW", os.environ.get("CTX_SIZE", "131072")))
    model_id = os.environ.get("MODEL_ID", "qwen3.6")
    max_tokens = int(os.environ.get("MAX_OUTPUT_TOKENS", "8192"))
    (agent_dir / "models.json").write_text(json.dumps({
        "providers": {
            "local": {
                "baseUrl": f"{llama_url}/v1",
                "api": "openai-completions",
                "apiKey": os.environ.get("LLAMA_API_KEY", "sk-local"),
                "compat": {"supportsDeveloperRole": False, "supportsReasoningEffort": False},
                "models": [{"id": model_id, "contextWindow": ctx, "maxTokens": max_tokens}],
            }
        }
    }, indent=2))
    (agent_dir / "settings.json").write_text(json.dumps({
        "defaultProvider": "local",
        "defaultModel": model_id,
        "defaultThinkingLevel": os.environ.get("THINKING_LEVEL", "high"),
        "enableInstallTelemetry": False,
    }, indent=2))


def boot_corpus(job_dir: Path, corpus_dir: Path, port: int) -> subprocess.Popen:
    env = dict(os.environ)
    env["CORPUS_DIR"] = str(corpus_dir)
    env["CORPUS_PORT"] = str(port)
    env["CORPUS_CACHE_DIR"] = str(job_dir / ".corpus_cache")
    logf = open(job_dir / "corpus.log", "a")
    logf.write(f"\n===== CORPUS SIDECAR @ {time.ctime()} =====\n"); logf.flush()
    proc = subprocess.Popen([sys.executable, str(HERE / "corpus_service.py")],
                            env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
    logf.close()
    return proc


def wait_http(url: str, timeout: int = 600) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=3); return True
        except urllib.error.HTTPError:
            return True
        except Exception:
            time.sleep(2)
    return False


def _strip_single_wrapper(corpus_dir: Path):
    """If everything sits under one top-level folder (the common zip wrapper, e.g. a GitHub
    `repo-main/` or a hand-made `corpus/`), hoist its contents up one level so the corpus root
    is the real content, not a redundant nesting."""
    entries = [p for p in corpus_dir.iterdir() if not p.name.startswith(".")]
    if len(entries) == 1 and entries[0].is_dir():
        inner = entries[0]
        for item in list(inner.iterdir()):
            shutil.move(str(item), str(corpus_dir / item.name))
        inner.rmdir()


def prepare_corpus(src: Path, corpus_dir: Path):
    corpus_dir.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        for item in src.iterdir():
            dst = corpus_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dst)
        _strip_single_wrapper(corpus_dir)
        return
    if src.is_file() and zipfile.is_zipfile(src):
        base = corpus_dir.resolve()
        with zipfile.ZipFile(src) as z:
            for member in z.namelist():
                tgt = (corpus_dir / member).resolve()
                if str(tgt).startswith(str(base)):
                    z.extract(member, corpus_dir)
        _strip_single_wrapper(corpus_dir)
        return
    raise SystemExit(f"ERROR: corpus must be a .zip or a folder: {src}")


def session_file(agent_dir: Path) -> Path | None:
    """The newest Pi session JSONL under the per-job agent dir."""
    sd = agent_dir / "sessions"
    if not sd.exists():
        return None
    files = sorted(sd.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def session_usage(sfile: Path | None) -> dict:
    """Sum assistant-message usage from the append-only session JSONL.

    This is the authoritative, compaction-safe token total (see module docstring). Uses the same
    fields Pi's getSessionStats uses: input / output / cacheRead / cacheWrite."""
    out = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
    if not sfile or not sfile.exists():
        return out
    for line in open(sfile, errors="ignore"):
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
    out["total"] = out["input"] + out["output"] + out["cacheRead"] + out["cacheWrite"]
    return out


def count_tool_calls(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    return sum(1 for raw in open(log_path, "rb") if b'"type":"tool_execution_start"' in raw)


def answer_present(work_dir: Path) -> bool:
    a = work_dir / "ANSWER.md"
    return a.exists() and a.stat().st_size > 200


# The task is delivered as `/skill:searchbox <question>`. Pi expands this to the FULL SKILL.md
# body + the question as one user message (verified), which both (a) guarantees the skill is
# actually loaded - pi otherwise only puts the skill's name/description in context and the model
# may never `read` the body - and (b) carries the question. So there is no separate task prompt:
# all task text lives in SKILL.md (single source of truth). Nothing here hints at method, tools,
# thinking style, or answer format - the experiment observes the model's own behavior.
TASK_COMMAND = "/skill:searchbox {query}"
# Sent only to keep the run going until the input-token budget is spent (the one mechanism we
# add over vanilla pi). No guidance on method or content.
KEEP_GOING = (
    "Continue. (input tokens used: {spent}/{budget})"
)


def drive(job_dir, work_dir, agent_dir, corpus_dir, args, budget):
    env = dict(os.environ)
    env["PI_CODING_AGENT_DIR"] = str(agent_dir)
    env["PI_SKIP_VERSION_CHECK"] = "1"
    cmd = [os.environ.get("PI_BIN", "pi"), "--mode", "rpc",
           "--skill", str(REPO / "pi" / "skills" / "searchbox"),
           "--extension", str(REPO / "pi" / "extensions" / "corpus-search.ts")]
    log = open(job_dir / "pi.log", "a")
    log.write(f"\n\n===== RPC SESSION @ {time.ctime()} =====\n"); log.flush()
    log_path = job_dir / "pi.log"

    # cwd = work_dir (sandbox: only corpus/ + ANSWER.md). Plumbing lives in the parent job_dir.
    proc = subprocess.Popen(cmd, cwd=str(work_dir), env=env,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
    lock = threading.Lock()

    def send(obj):
        with lock:
            try:
                proc.stdin.write(json.dumps(obj) + "\n"); proc.stdin.flush()
            except Exception:
                pass

    hard = {"reason": None}

    def hard_kill(reason):
        hard["reason"] = reason
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    global_wd = threading.Timer(args.max_seconds + 30, lambda: hard_kill("ceiling_seconds"))
    global_wd.start()

    # Wall-clock timing accumulators. pi emits no timing/tps (verified against source), so we
    # stopwatch its event stream here: LLM busy = message_start..message_end, tool busy =
    # tool_execution_start..tool_execution_end. Per-tool totals power the UI breakdown.
    timing = {"llm_ms": 0.0, "tool_ms": 0.0, "by_tool_ms": {}, "by_tool_n": {}}
    tmark = {"msg": None, "tool": {}}
    timing_path = job_dir / "timing.json"

    def record_timing(line: str):
        now = time.time() * 1000.0
        try:
            if '"type":"message_start"' in line:
                tmark["msg"] = now
            elif '"type":"message_end"' in line:
                if tmark["msg"] is not None:
                    timing["llm_ms"] += now - tmark["msg"]; tmark["msg"] = None
            elif '"type":"tool_execution_start"' in line:
                ev = json.loads(line)
                tmark["tool"][ev.get("toolCallId", "")] = (now, ev.get("toolName", "?"))
            elif '"type":"tool_execution_end"' in line:
                ev = json.loads(line)
                cid = ev.get("toolCallId", "")
                st = tmark["tool"].pop(cid, None)
                if st:
                    dur, name = now - st[0], st[1]
                    timing["tool_ms"] += dur
                    timing["by_tool_ms"][name] = timing["by_tool_ms"].get(name, 0.0) + dur
                    timing["by_tool_n"][name] = timing["by_tool_n"].get(name, 0) + 1
        except Exception:
            pass

    def flush_timing(wall_ms):
        try:
            out = {"wall_ms": wall_ms, "llm_ms": round(timing["llm_ms"]),
                   "tool_ms": round(timing["tool_ms"]),
                   "by_tool_ms": {k: round(v) for k, v in timing["by_tool_ms"].items()},
                   "by_tool_n": timing["by_tool_n"]}
            timing_path.write_text(json.dumps(out))
        except Exception:
            pass

    start = time.time()
    turn, stop_reason = 0, "error_pi_exited"
    sfile = None

    def spent_now():
        nonlocal sfile
        if sfile is None:
            sfile = session_file(agent_dir)
        u = session_usage(sfile)
        return u.get(BUDGET_METRIC, 0), u

    send({"type": "prompt", "message": TASK_COMMAND.format(query=args.query)})

    try:
        while True:
            cycle_wd = threading.Timer(max(1, args.turn_timeout), lambda: send({"type": "abort"}))
            cycle_wd.start()
            ended = False
            try:
                while True:
                    line = proc.stdout.readline()
                    if line == "":
                        break
                    if '"type":"message_update"' in line:
                        continue
                    log.write(line)
                    record_timing(line)
                    if '"type":"agent_end"' in line:
                        ended = True; break
            finally:
                cycle_wd.cancel()
            flush_timing(round((time.time() - start) * 1000))

            if not ended:
                stop_reason = hard["reason"] or "error_pi_exited"; break

            turn += 1
            log.flush()

            cf = job_dir / "control"
            if cf.exists():
                ctl = cf.read_text(errors="ignore").strip()
                if ctl in ("cancel", "pause"):
                    stop_reason = ctl if ctl == "paused" else "cancelled"
                    stop_reason = "paused" if ctl == "pause" else "cancelled"; break

            spent, usage = spent_now()
            elapsed = time.time() - start
            print(f"[searchbox] cycle {turn} {BUDGET_METRIC}={spent}/{budget} "
                  f"({round(100*spent/budget,1) if budget else 0}%) "
                  f"tools={count_tool_calls(log_path)} ans={answer_present(work_dir)}", flush=True)

            if elapsed > args.max_seconds:
                stop_reason = "ceiling_seconds"; break
            if turn >= args.max_turns:
                stop_reason = "ceiling_turns"; break

            if spent >= budget:
                if answer_present(work_dir):
                    stop_reason = "budget_spent"; break
                send({"type": "prompt", "message":
                      "Write your answer to ANSWER.md now."})
                continue

            send({"type": "prompt", "message": KEEP_GOING.format(spent=spent, budget=budget)})
    except KeyboardInterrupt:
        stop_reason = "interrupted"
    finally:
        global_wd.cancel()
        send({"type": "abort"})
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except Exception:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
        log.flush()
        flush_timing(round((time.time() - start) * 1000))

    _, usage = spent_now()
    return turn, stop_reason, usage


def write_run_meta(job_dir: Path, **fields):
    try:
        (job_dir / "run_meta.json").write_text(json.dumps(fields, indent=2))
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="./out")
    ap.add_argument("--budget", type=int, default=int(os.environ.get("INPUT_TOKEN_BUDGET", "500000")))
    ap.add_argument("--max-turns", type=int, default=int(os.environ.get("MAX_TURNS", "500")))
    ap.add_argument("--max-seconds", type=int, default=int(os.environ.get("MAX_SECONDS", "21600")))
    ap.add_argument("--turn-timeout", type=int, default=int(os.environ.get("TURN_TIMEOUT", "1200")))
    args = ap.parse_args()
    if args.budget < 1:
        print("ERROR: --budget must be >= 1", file=sys.stderr); sys.exit(2)

    signal.signal(signal.SIGTERM, lambda *_a: (_ for _ in ()).throw(KeyboardInterrupt()))

    llama_url = os.environ.get("LLAMA_URL", "http://localhost:8080")
    job_dir = Path(args.out).resolve(); job_dir.mkdir(parents=True, exist_ok=True)
    # work_dir is pi's cwd: a clean sandbox holding ONLY corpus/ and the model's ANSWER.md.
    # All plumbing (input.zip, logs, meta, .pi-agent) stays in job_dir, the parent, which pi's
    # cwd cannot see - so the model can't ls/cat/unzip the raw input or read its own logs.
    work_dir = job_dir / "work"; work_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir = work_dir / "corpus"
    agent_dir = job_dir / ".pi-agent"

    prepare_corpus(Path(args.corpus).resolve(), corpus_dir)
    (job_dir / "query.txt").write_text(args.query)

    port = free_port()
    os.environ["CORPUS_INDEX_URL"] = f"http://127.0.0.1:{port}"
    write_pi_config(agent_dir, llama_url)
    cs = boot_corpus(job_dir, corpus_dir, port)
    if not wait_http(f"http://127.0.0.1:{port}/stats", int(os.environ.get("CORPUS_BOOT_TIMEOUT", "600"))):
        print("ERROR: corpus sidecar did not come up", file=sys.stderr)
        try:
            os.killpg(os.getpgid(cs.pid), signal.SIGTERM)
        except Exception:
            cs.terminate()
        write_run_meta(job_dir, stop_reason="error_corpus_boot", turns=0, done=False)
        sys.exit(3)

    start = time.time()
    turn, stop_reason, usage = 0, "error_pi_exited", {}
    try:
        turn, stop_reason, usage = drive(job_dir, work_dir, agent_dir, corpus_dir, args, args.budget)
    except KeyboardInterrupt:
        stop_reason = "interrupted"
    finally:
        try:
            os.killpg(os.getpgid(cs.pid), signal.SIGTERM)
            try:
                cs.wait(timeout=10)
            except Exception:
                os.killpg(os.getpgid(cs.pid), signal.SIGKILL)
        except Exception:
            cs.terminate()

    spent = usage.get(BUDGET_METRIC, 0)
    done = stop_reason == "budget_spent" and answer_present(work_dir)
    write_run_meta(job_dir, stop_reason=stop_reason, turns=turn, done=done,
                   budget=args.budget, budget_metric=BUDGET_METRIC,
                   input_tokens_spent=spent, budget_pct=round(100*spent/args.budget, 1) if args.budget else None,
                   tokens=usage, answer_present=answer_present(work_dir),
                   tool_calls=count_tool_calls(job_dir / "pi.log"),
                   elapsed_seconds=round(time.time() - start, 1),
                   tools_enabled=os.environ.get("SEARCHBOX_TOOLS", "all"),
                   model_id=os.environ.get("MODEL_ID", "qwen3.6"))
    print(f"[searchbox] done (stop_reason={stop_reason}, {BUDGET_METRIC}={spent}/{args.budget})")
    print(json.dumps({"out": str(job_dir), "turns": turn, "done": done,
                      "stop_reason": stop_reason, "tokens": usage, "budget": args.budget,
                      "answer": str(work_dir / "ANSWER.md")}))


if __name__ == "__main__":
    main()
