# Searchbox

Give it a **prompt**, a **`.zip`** (or folder), and a **turn budget**. A self-hosted model in a
minimal [Pi](https://pi.dev) harness explores the dataroom with local retrieval
(jina-embeddings-v5-text-small + jina-reranker-v3, no web) for that many turns, then answers
(`ANSWER.md`). Token cost is recorded per turn but is not the stop condition.

The point is to watch what a model does under a restrained harness: it gets two tiers of retrieval
tools - **high-level** task tools that run the whole embed->rank pipeline over the dataroom, and
**low-level** stateless model primitives (embed, rerank, similarity, ...) - plus the stock Pi
built-ins (grep/read/...). Which does it reach for, and can it answer in a single turn?

![Searchbox main UI](docs/img/main-ui.png)

The per-job dashboard streams the live run: context window, token usage, turn budget, tool-call
distribution (with the actual programs each `bash` ran), wall-time split, and an ACTIVITY feed
showing every tool call **with its parameters**.

![Searchbox run dashboard](docs/img/dashboard.png)

## How it works

`server/run_searchbox.py` drives a `pi --mode rpc` session:

1. The dataroom is unzipped to `dataroom/` (read-only; a single wrapper dir is stripped). The
   sidecar (`server/dataroom_service.py`) indexes **nothing** at boot - retrieval scopes are
   embedded lazily on first use, and embeddings are cached per `(text, role)` for the life of the
   job process so identical text is never embedded twice (across turns or tools).
2. The task is appended to Pi's **system prompt** (`--append-system-prompt`): answer from
   `dataroom/`, no network, use any tools or build your own, write the answer to `ANSWER.md`. It
   is present every turn and never compacted, so the task stays stable for the whole budget. The
   question is sent once as the first user message. No skill.
3. Pi runs its own loop and compaction, untouched. The only thing added over vanilla Pi: while
   the budget is unspent and Pi goes idle, send a bare `Continue.`. As a backstop the harness
   also captures the model's final non-thinking message to `ANSWER.md` each turn, so there is an
   answer even if the model never wrote the file itself (it never clobbers a model-written one).
4. Force-budget OFF (default): stop after the **first turn** (turn=1 probe - the common case for
   measuring single-shot answer quality). ON: run until the turn budget is used (one turn = one
   `Continue.` -> agent works -> idle). `run_meta.json` records stop reason, turns, per-turn token
   breakdown, tool calls, and config. (We stop at a turn boundary because a run cannot be cleanly
   interrupted mid-turn anyway, and turns are the user-legible unit.)

## Tools

The external tools come from one source of truth, `pi/tools-catalog.json`, registered by
`pi/extensions/dataroom-search.ts` and validated by `server/app.py`. Every tool is a thin wrapper
over the sidecar's two models (jina-embeddings-v5-text-small + jina-reranker-v3). They are split
into **two tiers** (the `group` field drives the UI grouping + per-group master toggle):

**High-level (task tools)** - dataroom-aware, one call does the whole job. The tool reaches into
the uploaded dataroom for you (this is where the implicit corpus state lives, by design - it is
explicit in the tool's contract):

| Tool | What it does |
| --- | --- |
| `search_dataroom` | Embed corpus + query, return top-k relevant chunks `{path,chunk,score,text}`. One-stop semantic search over the dataroom. |
| `answer_question` | Two-stage: dense-retrieve a wide candidate set, then cross-encoder **rerank** to the few best supporting passages. |

**Low-level (model primitives)** - stateless single-model ops that act **only on data you pass
in** (no hidden corpus/state); the model composes its own pipeline (grep/read to get text, then
these):

| Tool | What it does |
| --- | --- |
| `embed_texts` | Embed caller text -> append vectors to a jsonl file (returns `{path,count,dim}`, not the vectors). |
| `rerank` | Cross-encoder score caller-supplied documents against a query. |
| `similarity` | Cosine similarity over caller text (pairwise if equal-length, else full matrix). |
| `classify` | Zero-shot label each text by nearest-label embedding similarity. |
| `cluster` | Greedy-threshold cluster caller text into near-duplicate groups. |
| `select_diverse` | Facility-location pick of the top-k most diverse texts (drop near-dups). |

This replaced an earlier flat catalog of ~22 near-duplicate entries (many were the same backend
op under different names/argument shapes). The split makes intent obvious - a model that just
wants an answer uses the high-level tools; one that wants control composes the primitives - and
keeps the only stateful behavior (reaching into the corpus) confined to the explicitly
dataroom-aware high-level tools. Built-in Pi tools (`read`, `bash`, `grep`, `find`, `ls`, `edit`,
`write`) keep their stock descriptions.

All embedding flows funnel through one cached `_encode(text, role)`, so repeated text (same chunk
across turns, same query reused, overlap between tools) is embedded at most once per job.

`SEARCHBOX_TOOLS` gates which tools a run registers (the tool-ablation axis):

- **unset** -> the catalog `default: true` set: `search_dataroom`, `answer_question`
- **`""`** (empty) -> none; Pi built-ins only (grep/read baseline)
- **`a,b,...`** -> exactly those (e.g. `embed_texts,rerank` to force the low-level path)

## Components

```
server/dataroom_service.py        sidecar: /search /answer /embed /rerank /similarity /classify
                                  /deduplicate /cluster /stats over the unzipped dataroom
                                  (per-(text,role) embed cache)
server/run_searchbox.py           orchestrator: unzip -> drive Pi -> stop on budget
server/app.py + web/              upload UI + live dashboard + job queue
server/stats.py                   per-turn token/timing/tool parse from the Pi session + log
pi/tools-catalog.json             external-tool catalog (single source of truth, high/low tiers)
pi/extensions/dataroom-search.ts  registers the catalog tools, gated by SEARCHBOX_TOOLS
docs/img/                         UI screenshots used in this README
scripts/run.sh                    one-shot CLI run
scripts/ablate.py                 ablation sweep
tests/test_scheduler.py           scheduler tests (no GPU)
```

## Get started

```bash
cp .env.example .env          # point LLAMA_URL at your OpenAI-compatible model server
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch -r server/requirements.txt huggingface-hub
npm install -g @earendil-works/pi-coding-agent@0.80.1   # pinned; see PI_VERSION

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

Default matrix (omit `--matrix`): `high_level` (search_dataroom + answer_question), `low_level`
(embed_texts + rerank + similarity), `no_tools` (grep/read baseline).

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
