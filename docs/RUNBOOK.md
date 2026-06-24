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
