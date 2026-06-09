#!/usr/bin/env python3
"""Orchestrator: run the autonomous Pi harness for one searchbox job.

Inputs: a PROMPT, a CORPUS (a .zip or an already-unzipped folder), and an INPUT-TOKEN BUDGET.
The agent answers the prompt grounded ONLY in the corpus, using local-only retrieval tools
(jina-embeddings-v5-text-small + jina-reranker-v3 over the corpus, served by the corpus
sidecar). NO web access.

This is dataroom inverted on two axes:
  1. Tools are CLOSED-CORPUS (the uploaded zip), not the open web.
  2. Stopping is BUDGET-FIRST, not floor-first: the run must SPEND the input-token budget on
     exploration before an answer is accepted. A premature `DONE` is rejected with the budget
     remaining, and the agent is nudged to keep exploring. Turns / seconds / tool-calls are
     only hard ceilings.

"Input-token budget" = cumulative prompt tokens the LLM has processed across all turns,
measured from llama.cpp's `llamacpp:prompt_tokens_total` counter (falls back to summed pi.log
`message_end.usage.input`). This is the honest measure of how much corpus context the model
actually chewed through.

ABLATION-READY (driven entirely by env, no code edits needed):
  BASE_MODEL ablation : LLAMA_URL / MODEL_ID / CONTEXT_WINDOW point Pi at any OpenAI-compatible
                        model server -> swap the base LLM.
  TOOL ablation       : SEARCHBOX_TOOLS="corpus_search" (or "corpus_rerank", or "" for none)
                        gates which corpus tools the extension registers. BASH_ENABLED=0 also
                        drops the bash/read fallback path nudge from the prompt.
  RETRIEVAL ablation  : EMBED_MODEL / RERANK_MODEL swap the retrieval models in the sidecar.
  BUDGET ablation     : INPUT_TOKEN_BUDGET sets the spend floor.
"""
import argparse, json, os, re, subprocess, sys, time, zipfile, signal, socket, threading, urllib.request, urllib.error, shutil
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def write_pi_config(agent_dir: Path, llama_url: str):
    """Per-job, isolated Pi agent dir so default LLM = the configured local model."""
    agent_dir.mkdir(parents=True, exist_ok=True)
    ctx = int(os.environ.get("CONTEXT_WINDOW", os.environ.get("CTX_SIZE", "131072")))
    model_id = os.environ.get("MODEL_ID", "qwen3.6")
    max_tokens = int(os.environ.get("MAX_OUTPUT_TOKENS", "8192"))
    reserve = max(max_tokens + 2048, ctx // 8)
    keep_recent = max(16000, ctx // 3)
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
        "compaction": {"enabled": True, "keepRecentTokens": keep_recent,
                       "reserveTokens": reserve},
    }, indent=2))


def boot_corpus(job_dir: Path, corpus_dir: Path, port: int) -> subprocess.Popen:
    env = dict(os.environ)
    env["CORPUS_DIR"] = str(corpus_dir)
    env["CORPUS_PORT"] = str(port)
    env["CORPUS_CACHE_DIR"] = str(job_dir / ".corpus_cache")
    logf = open(job_dir / "corpus.log", "a")
    logf.write(f"\n===== CORPUS SIDECAR @ {time.ctime()} =====\n")
    logf.flush()
    proc = subprocess.Popen(
        [sys.executable, str(HERE / "corpus_service.py")],
        env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True,
    )
    logf.close()
    return proc


def wait_http(url: str, timeout: int = 600) -> bool:
    """Up = any HTTP response. The corpus sidecar embeds the whole corpus at boot, which can
    take a while on a big zip + CPU embedder, so the default timeout is generous."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=3)
            return True
        except urllib.error.HTTPError:
            return True
        except Exception:
            time.sleep(2)
    return False


def prepare_corpus(src: Path, corpus_dir: Path):
    """Materialize the corpus under corpus_dir. src may be a .zip or an existing folder."""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        for item in src.iterdir():
            dst = corpus_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dst)
        return
    if src.is_file() and zipfile.is_zipfile(src):
        with zipfile.ZipFile(src) as z:
            # path-traversal guard: skip any member that escapes corpus_dir
            base = corpus_dir.resolve()
            for member in z.namelist():
                tgt = (corpus_dir / member).resolve()
                if not str(tgt).startswith(str(base)):
                    continue
                z.extract(member, corpus_dir)
        return
    raise SystemExit(f"ERROR: corpus must be a .zip or a folder: {src}")


# --- input-token budget accounting -------------------------------------------
def llama_prompt_tokens(llama_url: str) -> int | None:
    """Cumulative prompt (input) tokens processed by llama.cpp, from Prometheus /metrics
    (needs --metrics). This counter is monotonic for the life of the server, so it is read
    relative to a baseline captured at job start."""
    try:
        with urllib.request.urlopen(f"{llama_url}/metrics", timeout=3) as r:
            for line in r.read().decode(errors="ignore").splitlines():
                if line.startswith("llamacpp:prompt_tokens_total"):
                    return int(float(line.rsplit(" ", 1)[1]))
    except Exception:
        return None
    return None


def usage_input_tokens(log_path: Path) -> int:
    """Fallback: sum message_end.usage.input across pi.log (used if llama /metrics is absent)."""
    total = 0
    if not log_path.exists():
        return 0
    cap = 64 * 1024 * 1024
    size = log_path.stat().st_size
    with open(log_path, "rb") as f:
        if size > cap:
            f.seek(size - cap); f.readline()
        for raw in f:
            if b'"type":"message_end"' not in raw:
                continue
            try:
                ev = json.loads(raw.decode("utf-8", "ignore"))
            except Exception:
                continue
            u = ((ev.get("message") or {}).get("usage")) or {}
            total += int(u.get("input") or u.get("inputTokens") or u.get("promptTokens") or 0)
    return total


def count_tool_calls(log_path: Path) -> int:
    n = 0
    if not log_path.exists():
        return 0
    for raw in open(log_path, "rb"):
        if b'"type":"tool_execution_start"' in raw:
            n += 1
    return n


def answer_present(job_dir: Path) -> bool:
    a = job_dir / "ANSWER.md"
    return a.exists() and a.stat().st_size > 200


def status_done(job_dir: Path) -> bool:
    sp = job_dir / "STATUS.md"
    if not sp.exists():
        return False
    first = sp.read_text(errors="ignore").lstrip().splitlines()[:1]
    if not first:
        return False
    head = first[0].strip().upper()
    return head.startswith("DONE") or head.startswith("STATUS: DONE") or head.startswith("STATUS:DONE")


def write_run_meta(job_dir: Path, **fields):
    try:
        (job_dir / "run_meta.json").write_text(json.dumps(fields, indent=2))
    except Exception:
        pass


# Prompts ---------------------------------------------------------------------
FIRST_PROMPT = (
    "Question to answer: {query}\n\n"
    "Load and follow the `searchbox` skill. The corpus is at `{corpus}` (read-only). Answer "
    "ONLY from that corpus - no outside knowledge, no web. You have an input-token budget of "
    "{budget} tokens that you must SPEND exploring the corpus before answering: read widely, "
    "cross-check, verify. Write NOTES.md and the final ANSWER.md in your working directory "
    "(NOT inside corpus/). Begin now: map the corpus, decompose the question, then investigate "
    "the highest-value sub-question."
)
CONT_PROMPT = (
    "Continue per the skill. You have spent {spent}/{budget} input tokens ({pct}%). Keep "
    "exploring the corpus: open the next sub-question in NOTES.md with NEW corpus_search / "
    "corpus_rerank queries, read the full files behind the best chunks, and cross-check. Do "
    "not answer yet."
)
STALL_PROMPT = (
    "The last cycle made little progress. Do NOT repeat the same searches. Open NEW angles: "
    "different phrasings, adjacent files, references between files, edge cases. Read whole "
    "files (not just chunks) and record findings with their corpus citations in NOTES.md."
)
CORRECTIVE_PROMPT = (
    "You wrote DONE but only spent {spent}/{budget} input tokens ({pct}%) - the budget is NOT "
    "exhausted, so the exploration is not deep enough yet. Remove DONE from STATUS.md and keep "
    "investigating: enumerate corpus files you have not read, open new sub-questions, verify "
    "numbers with bash, and reconcile any inconsistencies. Only write DONE again once the "
    "budget is spent AND every sub-question is closed AND ANSWER.md is complete and fully cited."
)
ANSWER_MISSING_PROMPT = (
    "The budget is spent but ANSWER.md is missing or too thin. Write a complete, fully-cited "
    "ANSWER.md now (direct answer first, then per-sub-question reasoning, every claim citing a "
    "corpus path, a final ## Sources section), then set STATUS.md to `STATUS: DONE`."
)


def drive_rpc(job_dir, agent_dir, corpus_dir, args, budget, llama_url,
              sat_window, min_new_tokens, consolidate_every):
    env = dict(os.environ)
    env["PI_CODING_AGENT_DIR"] = str(agent_dir)
    env["PI_SKIP_VERSION_CHECK"] = "1"
    pi_bin = os.environ.get("PI_BIN", "pi")
    cmd = [
        pi_bin, "--mode", "rpc",
        "--skill", str(REPO / "pi" / "skills" / "searchbox"),
        "--extension", str(REPO / "pi" / "extensions" / "corpus-search.ts"),
    ]
    log = open(job_dir / "pi.log", "a")
    log.write(f"\n\n===== RPC SESSION @ {time.ctime()} =====\n")
    log.flush()
    log_path = job_dir / "pi.log"

    proc = subprocess.Popen(cmd, cwd=str(job_dir), env=env,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                            start_new_session=True)
    lock = threading.Lock()

    def send(obj):
        with lock:
            try:
                proc.stdin.write(json.dumps(obj) + "\n")
                proc.stdin.flush()
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

    # Baseline for the monotonic llama prompt-token counter (shared server may have prior usage).
    base_metric = llama_prompt_tokens(llama_url)
    use_metric = base_metric is not None

    def spent_tokens() -> int:
        if use_metric:
            cur = llama_prompt_tokens(llama_url)
            if cur is not None:
                return max(0, cur - base_metric)
        return usage_input_tokens(log_path)

    start = time.time()
    turn = 0
    stop_reason = "error_pi_exited"
    recent_deltas = deque(maxlen=sat_window)
    prev_spent = 0

    send({"type": "prompt", "message": FIRST_PROMPT.format(
        query=args.query, corpus=str(corpus_dir), budget=budget)})

    try:
        while True:
            cycle_wd = threading.Timer(max(1, args.turn_timeout),
                                       lambda: send({"type": "abort"}))
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
                    if '"type":"agent_end"' in line:
                        ended = True
                        break
            finally:
                cycle_wd.cancel()

            if not ended:
                stop_reason = hard["reason"] or "error_pi_exited"
                break

            turn += 1
            log.flush()

            cf = job_dir / "control"
            if cf.exists():
                ctl = cf.read_text(errors="ignore").strip()
                if ctl == "cancel":
                    stop_reason = "cancelled"; break
                if ctl == "pause":
                    stop_reason = "paused"; break

            elapsed = time.time() - start
            spent = spent_tokens()
            recent_deltas.append(spent - prev_spent)
            prev_spent = spent
            pct = round(100 * spent / budget, 1) if budget else 100.0
            tool_calls = count_tool_calls(log_path)
            print(f"[searchbox] cycle {turn} spent={spent}/{budget} ({pct}%) "
                  f"tools={tool_calls} ans={answer_present(job_dir)}", flush=True)

            # Hard ceilings (backstops).
            if elapsed > args.max_seconds:
                stop_reason = "ceiling_seconds"; break
            if turn >= args.max_turns:
                stop_reason = "ceiling_turns"; break
            if tool_calls > args.max_tool_calls:
                stop_reason = "ceiling_tool_calls"; break

            budget_spent = spent >= budget

            # DONE is honored only once the budget is spent AND a real answer exists.
            if status_done(job_dir):
                if budget_spent and answer_present(job_dir):
                    stop_reason = "done_budget_spent"; break
                if not budget_spent:
                    print(f"[searchbox] DONE rejected, budget unspent: {spent}/{budget}")
                    send({"type": "prompt", "message": CORRECTIVE_PROMPT.format(
                        spent=spent, budget=budget, pct=pct)})
                    continue
                # budget spent but answer missing/thin
                send({"type": "prompt", "message": ANSWER_MISSING_PROMPT})
                continue

            # Budget spent but the agent hasn't declared DONE yet -> ask for the final answer.
            if budget_spent:
                if answer_present(job_dir):
                    stop_reason = "budget_spent"; break
                send({"type": "prompt", "message": ANSWER_MISSING_PROMPT})
                continue

            # Saturation guard: if it stalls on token spend WHILE under budget, push new angles.
            stalled = (len(recent_deltas) == sat_window and
                       all(d < min_new_tokens for d in recent_deltas))
            if stalled:
                msg = STALL_PROMPT
            else:
                msg = CONT_PROMPT.format(spent=spent, budget=budget, pct=pct)
            send({"type": "prompt", "message": msg})
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
        try:
            log.flush()
        except Exception:
            pass

    return turn, stop_reason, spent_tokens()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="the question to answer from the corpus")
    ap.add_argument("--corpus", required=True, help="path to the corpus .zip or folder")
    ap.add_argument("--out", default="./out", help="job working directory")
    ap.add_argument("--budget", type=int,
                    default=int(os.environ.get("INPUT_TOKEN_BUDGET", "500000")),
                    help="input-token budget the run must spend before answering")
    ap.add_argument("--max-turns", type=int, default=int(os.environ.get("MAX_TURNS", "300")))
    ap.add_argument("--max-seconds", type=int, default=int(os.environ.get("MAX_SECONDS", "21600")))
    ap.add_argument("--turn-timeout", type=int, default=int(os.environ.get("TURN_TIMEOUT", "1200")))
    ap.add_argument("--max-tool-calls", type=int, default=int(os.environ.get("MAX_TOOL_CALLS", "5000")))
    args = ap.parse_args()

    if args.budget < 1:
        print("ERROR: --budget must be >= 1", file=sys.stderr); sys.exit(2)

    signal.signal(signal.SIGTERM, lambda *_a: (_ for _ in ()).throw(KeyboardInterrupt()))

    sat_window = int(os.environ.get("SATURATION_WINDOW", "3"))
    min_new_tokens = int(os.environ.get("MIN_NEW_TOKENS_PER_TURN", "1000"))
    consolidate_every = int(os.environ.get("CONSOLIDATE_EVERY", "0"))

    llama_url = os.environ.get("LLAMA_URL", "http://localhost:8080")

    job_dir = Path(args.out).resolve()
    job_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir = job_dir / "corpus"
    agent_dir = job_dir / ".pi-agent"

    prepare_corpus(Path(args.corpus).resolve(), corpus_dir)
    (job_dir / "query.txt").write_text(args.query)

    corpus_port = free_port()
    corpus_url = f"http://127.0.0.1:{corpus_port}"
    os.environ["CORPUS_INDEX_URL"] = corpus_url

    write_pi_config(agent_dir, llama_url)
    cs = boot_corpus(job_dir, corpus_dir, corpus_port)
    if not wait_http(f"{corpus_url}/stats", int(os.environ.get("CORPUS_BOOT_TIMEOUT", "600"))):
        print("ERROR: corpus sidecar did not come up", file=sys.stderr)
        try:
            os.killpg(os.getpgid(cs.pid), signal.SIGTERM)
        except Exception:
            cs.terminate()
        write_run_meta(job_dir, stop_reason="error_corpus_boot", turns=0, done=False)
        sys.exit(3)

    start = time.time()
    turn, stop_reason, spent = 0, "error_pi_exited", 0
    try:
        turn, stop_reason, spent = drive_rpc(
            job_dir, agent_dir, corpus_dir, args, args.budget, llama_url,
            sat_window, min_new_tokens, consolidate_every)
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

    done = stop_reason in ("done_budget_spent", "budget_spent") and answer_present(job_dir)
    write_run_meta(job_dir, stop_reason=stop_reason, turns=turn, done=done,
                   budget=args.budget, input_tokens_spent=spent,
                   budget_pct=round(100 * spent / args.budget, 1) if args.budget else None,
                   answer_present=answer_present(job_dir),
                   tool_calls=count_tool_calls(job_dir / "pi.log"),
                   elapsed_seconds=round(time.time() - start, 1),
                   tools_enabled=os.environ.get("SEARCHBOX_TOOLS", "all"),
                   model_id=os.environ.get("MODEL_ID", "qwen3.6"))
    print(f"[searchbox] done (stop_reason={stop_reason}, spent={spent}/{args.budget})")
    print(json.dumps({"out": str(job_dir), "turns": turn, "done": done,
                      "stop_reason": stop_reason, "input_tokens_spent": spent,
                      "budget": args.budget, "answer": str(job_dir / "ANSWER.md")}))


if __name__ == "__main__":
    main()
