# Searchbox ops runbook

Operational commands for running searchbox live. Expanded by the OPS issues (#34–#36).

## llama-server (the agent LLM)

`stats.llama_tps()` (dashboard throughput panel) reads llama.cpp's Prometheus `/metrics`
(`llamacpp:predicted_tokens_seconds` = decode, `llamacpp:prompt_tokens_seconds` = prefill).
**These are only exposed when llama-server is launched with `--metrics`.** Without it,
`/metrics` 404s and `llama_tps()` returns `{}` (the dashboard tps panel is blank — not an error).

Launch command (the `--metrics` flag is the fix for #3):

```bash
llama-server \
  -m /path/to/model.gguf \
  --host 127.0.0.1 --port 8080 \
  --metrics \           # REQUIRED: exposes llamacpp:*_tokens_seconds on /metrics for stats.llama_tps()
  -c 131072 \           # context window (match CONTEXT_WINDOW); smaller for tiny models
  -np 1                 # single slot (searchbox runs one job per slot; see the scheduler)
```

Point searchbox at it with `LLAMA_URL=http://127.0.0.1:8080` (default).

### Verify it works (FSV)

```bash
# 1. props returns real n_ctx + model_path (not nulls / "not found")
curl -s "$LLAMA_URL/props" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['default_generation_settings']['n_ctx'], d['model_path'])"

# 2. after at least one completion, /metrics exposes both throughput lines
curl -s "$LLAMA_URL/metrics" | grep -E '^llamacpp:(predicted|prompt)_tokens_seconds'
#   llamacpp:prompt_tokens_seconds 666.667
#   llamacpp:predicted_tokens_seconds 93.4579

# 3. stats.llama_tps() returns {decode, prefill}
LLAMA_URL=$LLAMA_URL python3 -c "import sys;sys.path.insert(0,'server');import stats;print(stats.llama_tps())"
#   {'prefill': 666.7, 'decode': 93.5}
```

### Rollback

Drop `--metrics`: `/metrics` 404s and the dashboard tps panel goes blank (`llama_tps()` → `{}`).
No crash — this is the documented degraded state.

### Notes / gotchas

- `/metrics` Prometheus counters are **empty until the first completion** runs (the
  `*_tokens_seconds` gauges are computed from the most recent decode/prefill). Trigger one
  completion before asserting they exist.
- A bare `/health` returning `{"status":"ok"}` does **not** prove this is llama-server — a stub
  or unrelated server can answer `/health`. Confirm `/props` returns real `n_ctx`/`model_path`
  and `/v1/models` lists the GGUF.

## Benchmark / eval environment (#2)

One job = `server.run_searchbox` boots its own dataroom sidecar (jina models, lazy-embed), drives
`pi` against the running llama-server, and writes artifacts to `--out`. The sidecar picks a free
port per run (`free_port()`), so concurrent runs do not collide.

Required env (local backend, no network):

```bash
export LLAMA_URL=http://127.0.0.1:8089       # the llama-server above
export EMBED_BACKEND=local                    # jina models from local cache, no download
export CONTEXT_WINDOW=4096                     # match llama-server -c
export PI_BIN=$(command -v pi)                 # pi CLI on PATH
# tools the agent may call (subset of pi/tools-catalog.json); fewer tools = fewer models warmed
export SEARCHBOX_TOOLS=search_dataroom,answer_question
```

### Single smoke run

```bash
.venv/bin/python -m server.run_searchbox \
  --query "What is the battery life of the Atlas-7?" \
  --dataroom data/default-dataroom.zip --budget 1 --out ./out/smoke
```

Source of truth = the artifacts, not stdout:

```bash
cat out/smoke/run_meta.json   # done=true, answer_present=true, tokens.output>0
cat out/smoke/timing.json     # llm_ms>0, by_tool_ms populated
cat out/smoke/work/ANSWER.md   # the model's grounded answer
```

`done=true` requires a natural stop AND a non-empty ANSWER.md AND `tokens.output>0` (a real
answer costs output tokens — see #38/#39). A run whose LLM is unreachable reports `done=false`
with no ANSWER.md, and ends in seconds (it does not hang to `--max-seconds`).

### Ablation matrix

```bash
.venv/bin/python -m scripts.ablate \
  --query "What error code did incident INC-2041 report?" \
  --dataroom data/default-dataroom.zip --budget 1 \
  --matrix config/ablations.example.json --out runs/exp1
```

`runs/exp1/results.jsonl` has one summary row per config; each config also gets its own
`runs/exp1/<name>/{run_meta.json,timing.json}`. The matrix is a JSON list of `{name, env:{...}}`
where `env` is per-config ENV overrides applied to a single `run_searchbox` invocation.

### Default dataroom

`data/default-dataroom.zip` is the frozen Meridian Robotics corpus (the #1 eval corpus, 10 docs);
regenerate it and `data/prd/benchmarks/eval/queryset.jsonl` with
`data/prd/benchmarks/eval/build_queryset.py`.

### Failure modes (verified)

- **llama-server down** → sidecar still boots; pi's LLM calls error (`Connection error.` in
  pi.log), run ends in ~10s with `done=false`, `tokens.output=0`, no ANSWER.md. No hang.
- **port collision** → `free_port()` binds a fresh ephemeral port per run; concurrent runs differ.
- **corrupt / non-zip dataroom** → `prepare_dataroom` exits with
  `ERROR: dataroom must be a .zip or a folder` (SystemExit, not a partial run).
