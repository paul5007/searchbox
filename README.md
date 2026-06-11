# Searchbox

Give it a **prompt**, a **`.zip`** (or folder), and an **input-token budget**. A self-hosted
model in a minimal [Pi](https://pi.dev) harness explores the dataroom with local retrieval
(jina-embeddings-v5-text-small + jina-reranker-v3, no web) until it has spent the budget, then
answers. The harness captures the model's final message as the answer (`ANSWER.md`).

The point is to watch what a model does under a maximally restrained harness: given raw atomic
primitives (an embedder and a reranker) but no instructions on how to use them, does it compose
them into a retrieval pipeline (embed -> store -> similarity search), or fall back to grep? So
nothing the model sees prescribes method, format, or tool choice.

## How it works

`server/run_searchbox.py` drives a `pi --mode rpc` session:

1. The dataroom is unzipped to `dataroom/` (read-only; a single wrapper dir is stripped). The
   sidecar (`server/dataroom_service.py`) indexes **nothing** at boot - it exposes only atomic
   model primitives (`sentence_embed`, `passage_rerank`), so the model decides if/when/how to
   embed, store, and search.
2. The task framing is appended to Pi's **system prompt** (`--append-system-prompt`): answer
   from `dataroom/`, no network, use any tools or build your own workflow. The question itself
   is sent once as the first user message. There is no skill - the system prompt is present on
   every turn and is never compacted, so the task stays stable for the whole budget. Pi runs its
   own loop and compaction, untouched; the only thing added over vanilla Pi is: while the budget
   is unspent and Pi goes idle, send a bare `Continue.`.
3. **The harness captures the answer itself.** The model is never told to write `ANSWER.md`.
   After each turn the harness takes the model's final non-thinking message text and saves it to
   `ANSWER.md` (overwriting). This keeps the framework lean - no file-writing instruction, no
   "write your answer" nudges. Input tokens spent are read each cycle from the Pi session file.
4. Force-budget ON (default): run until the input-token budget is spent. OFF: stop after the
   first turn. `run_meta.json` records the stop reason, token breakdown, tool calls, and config.

## What the model sees

Every piece of model-facing text, and nothing else:

| What | Where |
| --- | --- |
| System prompt | Pi's default coding-assistant prompt + our appended task ([`SYSTEM_TASK`](server/run_searchbox.py)): answer from `dataroom/`, no network, use any tools or build your own workflow. Present every turn, never compacted. |
| Task delivery | [`TASK_COMMAND`](server/run_searchbox.py) — the bare question, sent once as the first user message. No skill. |
| Keep-going nudge | [`KEEP_GOING`](server/run_searchbox.py) — `Continue.` |
| Answer capture | the harness saves the model's final non-thinking message to `ANSWER.md` each turn; the model is never asked to write it. |
| `sentence_embed` | [dataroom-search.ts](pi/extensions/dataroom-search.ts) — embed text(s) with jina-embeddings-v5-text-small; APPENDS vectors to a jsonl in the work dir and returns only {path,count,dim}; the model reads the file back and does its own similarity/search |
| `passage_rerank` | [dataroom-search.ts](pi/extensions/dataroom-search.ts) — score caller-supplied passages by relevance (jina-reranker-v3) |

Built-in Pi tools (`read`, `bash`, `edit`, `write`, `grep`, `find`, `ls`) keep Pi's stock
descriptions. Unzip is in the orchestrator: the sidecar must embed the dataroom before Pi starts,
and every ablation run must begin from an identical dataroom.

## Components

```
server/dataroom_service.py   FastAPI sidecar: /embed, /rerank, /search, /stats
server/run_searchbox.py    orchestrator: unzip -> drive Pi -> stop on budget (task in system prompt)
server/app.py + web/       upload UI + live dashboard
pi/extensions/dataroom-search.ts   sentence_embed + passage_rerank (default); semantic_search (opt-in)
scripts/run.sh             one-shot CLI run
scripts/ablate.py          ablation sweep
```

## Get started

```bash
cp .env.example .env          # point LLAMA_URL at your OpenAI-compatible model server
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch -r server/requirements.txt huggingface-hub
npm install -g @earendil-works/pi-coding-agent@0.78.0

bash scripts/run.sh "Where is auth handled?" ./dataroom.zip 300000 ./out
cat ./out/ANSWER.md
```

Web UI: `python -m server.app` then open `http://localhost:8000`.

## Ablation

Everything is an env knob — no code edits:

| Knob | Ablates |
| --- | --- |
| `SEARCHBOX_TOOLS` | tools the model gets. Unset => default `sentence_embed,passage_rerank`. Set explicitly to pick, e.g. `sentence_embed` / `passage_rerank` / `` (none). Opt-in `semantic_search` (high-level embed->rank->top-k pipeline) is kept in the repo but only loads when listed here, e.g. `semantic_search,passage_rerank`. |
| `LLAMA_URL` + `MODEL_ID` + `CONTEXT_WINDOW` | the base LLM |
| `EMBED_MODEL` / `RERANK_MODEL` | the retrieval models |
| `INPUT_TOKEN_BUDGET` | the budget |

```bash
python -m scripts.ablate --query "..." --dataroom ./dataroom.zip --budget 300000 \
  --matrix config/ablations.example.json --out ./runs/exp1
cat ./runs/exp1/results.jsonl    # one row per config
```

Default matrix (omit `--matrix`): `full`, `search_only`, `rerank_only`, `no_tools`.

## Token accounting

The budget is measured against **`input`** (fresh prefill tokens the model processed); the UI
also shows `output`, `cacheRead`, `cacheWrite`, `total`. Change the budgeted field with
`BUDGET_METRIC`.

Source of truth is the append-only Pi session file, summed over assistant-message `usage`. Pi's
`get_session_stats` sums only in-memory messages, which compaction prunes, so it undercounts a
long run; the session file is never rewritten, so it is compaction-safe. Timing (wall / LLM /
tool) is stopwatched from Pi's event stream and tok/s comes from llama.cpp `/metrics` — Pi
reports neither.

## License

MIT
