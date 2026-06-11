# Searchbox

Give it a **prompt**, a **`.zip`** (or folder), and a **turn budget**. A self-hosted model in a
minimal [Pi](https://pi.dev) harness explores the dataroom with local retrieval
(jina-embeddings-v5-text-small + jina-reranker-v3, no web) for that many turns, then answers
(`ANSWER.md`). Token cost is recorded per turn but is not the stop condition.

The point is to watch what a model does under a maximally restrained harness: given atomic
primitives (an embedder, a reranker) but no instruction on how to use them, does it compose them
into a retrieval pipeline (embed -> store -> similarity search), or fall back to grep? Nothing the
model sees prescribes which tool to use or how.

## How it works

`server/run_searchbox.py` drives a `pi --mode rpc` session:

1. The dataroom is unzipped to `dataroom/` (read-only; a single wrapper dir is stripped). The
   sidecar (`server/dataroom_service.py`) indexes **nothing** at boot - it exposes only atomic
   model primitives (embed, rerank, similarity, ...), so the model decides if/when/how to embed,
   store, and search.
2. The task is appended to Pi's **system prompt** (`--append-system-prompt`): answer from
   `dataroom/`, no network, use any tools or build your own, write the answer to `ANSWER.md`. It
   is present every turn and never compacted, so the task stays stable for the whole budget. The
   question is sent once as the first user message. No skill.
3. Pi runs its own loop and compaction, untouched. The only thing added over vanilla Pi: while
   the budget is unspent and Pi goes idle, send a bare `Continue.`. As a backstop the harness
   also captures the model's final non-thinking message to `ANSWER.md` each turn, so there is an
   answer even if the model never wrote the file itself (it never clobbers a model-written one).
4. Force-budget ON (default): run until the turn budget is used (one turn = one `Continue.` ->
   agent works -> idle). OFF: stop after the first turn. `run_meta.json` records stop reason,
   turns, per-turn token breakdown, tool calls, and config. (We stop at a turn boundary because a
   run cannot be cleanly interrupted mid-turn anyway, and turns are the user-legible unit.)

## Tools

The model's external tools come from one source of truth, `pi/tools-catalog.json` (~22 entries),
registered by `pi/extensions/dataroom-search.ts` and validated by `server/app.py`. Every tool is
a thin wrapper that reduces to the sidecar's embed/rerank/similarity backends - they differ only
in name, description, and argument shape (modeled on Jina's MCP + CLI surface) so we can observe
which framing a model reaches for. Built-in Pi tools (`read`, `bash`, `grep`, `find`, `ls`,
`edit`, `write`) keep their stock descriptions.

`SEARCHBOX_TOOLS` gates which tools a run registers (the tool-ablation axis):

- **unset** -> the catalog `default: true` set: `sentence_embed`, `passage_rerank`
- **`""`** (empty) -> none; Pi built-ins only (grep/read baseline)
- **`a,b,...`** -> exactly those (e.g. `semantic_search,passage_rerank` for a high-level
  embed->rank->top-k pipeline instead of the atomic primitives)

## Components

```
server/dataroom_service.py        sidecar: /embed /rerank /search /similarity /classify
                                  /deduplicate /cluster /stats over the unzipped dataroom
server/run_searchbox.py           orchestrator: unzip -> drive Pi -> stop on budget
server/app.py + web/              upload UI + live dashboard + job queue
pi/tools-catalog.json             external-tool catalog (single source of truth)
pi/extensions/dataroom-search.ts  registers the catalog tools, gated by SEARCHBOX_TOOLS
scripts/run.sh                    one-shot CLI run
scripts/ablate.py                 ablation sweep
tests/test_scheduler.py           scheduler tests (no GPU)
```

## Get started

```bash
cp .env.example .env          # point LLAMA_URL at your OpenAI-compatible model server
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch -r server/requirements.txt huggingface-hub
npm install -g @earendil-works/pi-coding-agent@0.79.1   # pinned; see PI_VERSION

# one-shot CLI: prompt, dataroom, budget(turns), outdir
bash scripts/run.sh "Where is auth handled?" ./dataroom.zip 30 ./out
cat ./out/ANSWER.md
```

Web UI: `python -m server.app`, then open `http://localhost:8000`.

Retrieval runs on local weights by default; set `EMBED_BACKEND=api RERANK_BACKEND=api` (+
`JINA_API_KEY`) to call the Jina cloud instead. Same tools either way - only where embed/rerank
runs changes (a clean local-vs-api ablation axis).

## Scheduling (single slot)

One model slot, so jobs run serially:

- **Foreground FIFO** - fresh submits and explicit resumes run oldest-first.
- **Preemption** - a new foreground job preempts whatever is running (incl. another foreground
  job) and takes the slot now (`PREEMPT_FOREGROUND=1`). The preempted job returns to the pool and
  resumes ahead of bulk backfill.
- **Auto-backfill** - when nothing is queued, the oldest paused job is auto-resumed (`--continue`)
  to keep the slot busy (`AUTO_BACKFILL=1`); such a run is preemptible.
- **Pause is sticky** - `POST /jobs/{id}/pause` -> `held`; not auto-resumed, only an explicit
  `POST /jobs/{id}/resume` revives it. (Preemption uses a separate, preemptible `paused` state.)

State survives an app restart: mid-flight jobs return to the resumable pool, held jobs stay held.
Logic is covered by `tests/test_scheduler.py` (no GPU; run `python tests/test_scheduler.py`).

## Ablation

Everything is an env knob - no code edits:

| Knob | Ablates |
| --- | --- |
| `SEARCHBOX_TOOLS` | which tools the model gets (see [Tools](#tools)) |
| `EMBED_BACKEND` / `RERANK_BACKEND` | retrieval on local weights vs Jina API |
| `LLAMA_URL` + `MODEL_ID` + `CONTEXT_WINDOW` | the base LLM |
| `EMBED_MODEL` / `RERANK_MODEL` | the retrieval models |
| `TURN_BUDGET` | the budget, in turns |

```bash
python -m scripts.ablate --query "..." --dataroom ./dataroom.zip --budget 10 \
  --matrix config/ablations.example.json --out ./runs/exp1
cat ./runs/exp1/results.jsonl    # one row per config
```

Default matrix (omit `--matrix`): `full`, `search_only`, `rerank_only`, `no_tools`.

## Token accounting

The budget is a turn count, but per-turn token cost is still recorded (`turns.jsonl`,
`run_meta.json`, the dashboard). The reported field is **`input`** (fresh prefill tokens); the UI
also shows `output`, `cacheRead`, `cacheWrite`, `total`. Change the reported field with
`BUDGET_METRIC`.

Source of truth is the append-only Pi session file, summed over assistant-message `usage`. Pi's
`get_session_stats` sums only in-memory messages, which compaction prunes, so it undercounts a
long run; the session file is never rewritten, so it is compaction-safe. Timing (wall / LLM /
tool) is stopwatched from Pi's event stream; tok/s comes from llama.cpp `/metrics`.

## License

MIT
