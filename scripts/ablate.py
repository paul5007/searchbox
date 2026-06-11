#!/usr/bin/env python3
"""Ablation runner for searchbox.

Runs the same (prompt, dataroom, budget) through a matrix of configurations and collects the
result of each into one results.jsonl, so you can compare how the answer / token-spend /
tool-usage change as you toggle tools or swap the base model.

Each config is a dict of ENV OVERRIDES applied to a single `server.run_searchbox` invocation.
The reserved ablation knobs (no code edits needed to vary any of these):

  SEARCHBOX_TOOLS   which dataroom tools are registered. "" = none, "sentence_embed",
                    "passage_rerank", or "sentence_embed,passage_rerank" (default = all).
  LLAMA_URL         OpenAI-compatible base-model server (swap the base LLM).
  MODEL_ID          agent-facing model id for that server.
  CONTEXT_WINDOW    context window for the chosen model.
  EMBED_MODEL       retrieval embedder (e.g. v5-text-small vs v5-text-nano).
  RERANK_MODEL      cross-encoder reranker.
  TURN_BUDGET       the budget, in turns.

Define the matrix in a JSON file (see config/ablations.example.json) or use the built-in
default matrix (tool ablation). Example:

  python -m scripts.ablate --query "..." --dataroom path/to/dataroom.zip \\
      --budget 10 --matrix config/ablations.example.json --out runs/exp1

Results: runs/exp1/<config_name>/ holds the full job (ANSWER.md, NOTES.md, pi.log, run_meta.json);
runs/exp1/results.jsonl has one summary row per config.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Default matrix: the tool ablation. Each entry is {name, env:{...}}.
DEFAULT_MATRIX = [
    {"name": "full",          "env": {"SEARCHBOX_TOOLS": "sentence_embed,passage_rerank"}},
    {"name": "search_only",   "env": {"SEARCHBOX_TOOLS": "sentence_embed"}},
    {"name": "rerank_only",   "env": {"SEARCHBOX_TOOLS": "passage_rerank"}},
    {"name": "no_tools",      "env": {"SEARCHBOX_TOOLS": ""}},  # bash/read only
]


def load_matrix(path: str | None):
    if not path:
        return DEFAULT_MATRIX
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict) and "matrix" in data:
        data = data["matrix"]
    if not isinstance(data, list):
        raise SystemExit("matrix file must be a JSON list of {name, env} objects")
    return data


def run_one(cfg, args, exp_dir: Path) -> dict:
    name = cfg.get("name") or f"cfg{int(time.time())}"
    job_dir = exp_dir / name
    job_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({k: str(v) for k, v in (cfg.get("env") or {}).items()})
    # Per-config budget override allowed; else the sweep-wide budget.
    budget = str(cfg.get("budget", args.budget))
    cmd = [sys.executable, "-m", "server.run_searchbox",
           "--query", args.query, "--dataroom", str(Path(args.dataroom).resolve()),
           "--out", str(job_dir), "--budget", budget]
    if args.max_seconds:
        cmd += ["--max-seconds", str(args.max_seconds)]

    print(f"\n=== ablation: {name}  env={cfg.get('env')}  budget={budget} ===", flush=True)
    t0 = time.time()
    log = open(job_dir / "ablate.log", "w")
    rc = subprocess.call(cmd, cwd=str(REPO), env=env, stdout=log, stderr=subprocess.STDOUT)
    log.close()

    meta = {}
    rm = job_dir / "run_meta.json"
    if rm.exists():
        try:
            meta = json.loads(rm.read_text())
        except Exception:
            meta = {}
    answer = job_dir / "ANSWER.md"
    tokens = meta.get("tokens") or {}
    row = {
        "name": name,
        "env": cfg.get("env"),
        "rc": rc,
        "wall_seconds": round(time.time() - t0, 1),
        "stop_reason": meta.get("stop_reason"),
        "done": meta.get("done"),
        "budget": meta.get("budget"),                 # in turns
        "budget_pct": meta.get("budget_pct"),
        "token_metric": meta.get("token_metric"),
        "tokens_spent": meta.get("tokens_spent"),
        "tokens_input": tokens.get("input"),
        "tokens_output": tokens.get("output"),
        "tokens_cacheRead": tokens.get("cacheRead"),
        "tokens_cacheWrite": tokens.get("cacheWrite"),
        "tokens_total": tokens.get("total"),
        "turns": meta.get("turns"),
        "tool_calls": meta.get("tool_calls"),
        "answer_present": meta.get("answer_present"),
        "answer_bytes": answer.stat().st_size if answer.exists() else 0,
        "answer_path": str(answer) if answer.exists() else None,
        "model_id": meta.get("model_id"),
        "tools_enabled": meta.get("tools_enabled"),
    }
    print(f"=== {name}: stop={row['stop_reason']} turns={row['turns']}/{row['budget']} "
          f"tools={row['tool_calls']} answer={row['answer_bytes']}B ===", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--dataroom", required=True)
    ap.add_argument("--budget", type=int, default=int(os.environ.get("TURN_BUDGET", "10")))
    ap.add_argument("--matrix", help="JSON file with the ablation matrix; default = tool ablation")
    ap.add_argument("--out", default="./runs/ablation")
    ap.add_argument("--max-seconds", type=int, default=0)
    args = ap.parse_args()

    matrix = load_matrix(args.matrix)
    exp_dir = Path(args.out).resolve()
    exp_dir.mkdir(parents=True, exist_ok=True)
    results_path = exp_dir / "results.jsonl"

    rows = []
    with open(results_path, "w") as rf:
        for cfg in matrix:
            row = run_one(cfg, args, exp_dir)
            rows.append(row)
            rf.write(json.dumps(row, ensure_ascii=False) + "\n")
            rf.flush()

    print(f"\n===== ablation summary ({len(rows)} configs) =====")
    print(f"{'config':<16}{'stop_reason':<20}{'turns/budget':<18}{'tools':<8}{'answerB':<9}")
    for r in rows:
        sb = f"{r['turns']}/{r['budget']}"
        print(f"{r['name']:<16}{str(r['stop_reason']):<20}{sb:<18}"
              f"{str(r['tool_calls']):<8}{str(r['answer_bytes']):<9}")
    print(f"\nresults.jsonl -> {results_path}")


if __name__ == "__main__":
    main()
