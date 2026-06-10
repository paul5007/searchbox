# Searchbox

Give it a **prompt**, a **`.zip`** (or folder), and an **input-token budget**. A self-hosted
model in a minimal [Pi](https://pi.dev) harness explores the corpus with local retrieval
(jina-embeddings-v5-text-small + jina-reranker-v3, no web) until it has spent the budget, then
writes `ANSWER.md`.

The point is to watch what a model does under a maximally restrained harness: given raw atomic
primitives (an embedder and a reranker) but no instructions on how to use them, does it compose
them into a retrieval pipeline (embed -> store -> similarity search), or fall back to grep? So
nothing the model sees prescribes method, format, or tool choice.

## How it works

`server/run_searchbox.py` drives a `pi --mode rpc` session:

1. The corpus is unzipped to `corpus/` (read-only; a single wrapper dir is stripped). The
   sidecar (`server/corpus_service.py`) indexes **nothing** at boot - it exposes only atomic
   model primitives (`sentence_embed`, `passage_rerank`), so the model decides if/when/how to
   embed, store, and search.
2. The task is sent once as `/skill:searchbox <question>`. Pi runs its own loop and its own
   compaction, untouched. The only thing added over vanilla Pi: while the budget is unspent and
   Pi goes idle, send a bare `Continue.`.
3. Input tokens spent are read each cycle from the Pi session file (see
   [Token accounting](#token-accounting)).
4. When the budget is spent and `ANSWER.md` exists, the run stops. `run_meta.json` records the
   stop reason, token breakdown, tool calls, timing, and config.

## What the model sees

Every piece of model-facing text, and nothing else:

| What | Where |
| --- | --- |
| System prompt | none of ours — Pi's default (no `--system-prompt`, no `SYSTEM.md`) |
| Task | [`pi/skills/searchbox/SKILL.md`](pi/skills/searchbox/SKILL.md) — answer from `corpus/`, no network, write `ANSWER.md` |
| Task delivery | [`TASK_COMMAND`](server/run_searchbox.py#L183) — `/skill:searchbox <q>`; Pi expands it to the full SKILL.md body + question, which guarantees the skill is loaded (otherwise only its name/description is in context) |
| Keep-going nudge | [`KEEP_GOING`](server/run_searchbox.py#L186-L188) — `Continue. (input tokens used: x/y)` |
| Final-answer nudge | [run_searchbox.py#L330](server/run_searchbox.py#L330) — `Write your answer to ANSWER.md now.` (only if budget spent but no `ANSWER.md`) |
| `sentence_embed` | [corpus-search.ts](pi/extensions/corpus-search.ts) — embed text(s) with jina-embeddings-v5-text-small; APPENDS vectors to a jsonl in the work dir and returns only {path,count,dim}; the model reads the file back and does its own similarity/search |
| `passage_rerank` | [corpus-search.ts#L85-L87](pi/extensions/corpus-search.ts#L85-L87) — re-order passages you supply by relevance |

Built-in Pi tools (`read`, `bash`, `edit`, `write`, `grep`, `find`, `ls`) keep Pi's stock
descriptions. Unzip is in the orchestrator, not the skill: the sidecar must embed the corpus
before Pi starts, and every ablation run must begin from an identical corpus.

## Components

```
server/corpus_service.py   FastAPI sidecar: /search, /rerank, /stats
server/run_searchbox.py    orchestrator: unzip -> embed -> drive Pi -> stop on budget
server/app.py + web/       upload UI + live dashboard
pi/extensions/corpus-search.ts   sentence_embed + passage_rerank (gated by SEARCHBOX_TOOLS)
pi/skills/searchbox/SKILL.md     the task
scripts/run.sh             one-shot CLI run
scripts/ablate.py          ablation sweep
```

## Get started

```bash
cp .env.example .env          # point LLAMA_URL at your OpenAI-compatible model server
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch -r server/requirements.txt huggingface-hub
npm install -g @earendil-works/pi-coding-agent@0.78.0

bash scripts/run.sh "Where is auth handled?" ./corpus.zip 300000 ./out
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
python -m scripts.ablate --query "..." --corpus ./corpus.zip --budget 300000 \
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
