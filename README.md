# Searchbox

Hand it a **prompt**, a **`.zip`** (or folder), and an **input-token budget**. A local model
in a [Pi](https://pi.dev) harness loops search-rerank-read over the unzipped corpus until it
has **spent the budget** exploring, then writes a fully-cited answer grounded only in that
corpus. No web access.

It is the closed-corpus inversion of [dataroom](https://github.com/hanxiao/dataroom):

| | dataroom | searchbox |
| --- | --- | --- |
| Input | a query | a query + a zip + a token budget |
| Tools | open web (jina CLI search/read) | local only: corpus embeddings + reranker |
| Models | self-hosted LLM | self-hosted LLM + jina-embeddings-v5-text-small + jina-reranker-v3 |
| Stop rule | a comprehensiveness **floor** (budget is a ceiling) | spend the input-token **budget** (turns/seconds are ceilings) |
| Output | a knowledge-base `.zip` | a cited `ANSWER.md` |

## Why

For a grounded answer over a fixed body of material (a codebase, a data room, a document
dump), the bottleneck is not reasoning - it is reading enough of the corpus to answer
honestly. Searchbox forces that: it must consume an input-token budget of real corpus context
through tool calls before it is allowed to answer, and a premature answer is rejected with the
budget remaining. The result is an answer whose every claim cites a file you handed it.

## How it works

A persistent `pi --mode rpc` session is driven by `server/run_searchbox.py`:

1. The corpus zip is unzipped to `corpus/` (read-only) and a sidecar
   (`server/corpus_service.py`) chunks + embeds it once with **jina-embeddings-v5-text-small**
   (cached on disk by content hash, so ablation reruns reuse the index).
2. The agent loops: `corpus_search` (v5-text-small semantic search) -> `corpus_rerank`
   (jina-reranker-v3 cross-encoder) -> `read`/`bash` the full files -> record cited findings.
3. The orchestrator measures **input tokens spent** (llama.cpp `prompt_tokens_total`, with a
   pi.log usage fallback) each cycle. It keeps nudging the agent to explore until `spent >=
   budget`. A `DONE` written before the budget is spent is rejected.
4. When the budget is spent and `ANSWER.md` exists, the run stops. `run_meta.json` records the
   stop reason, tokens spent, tool calls, and config (for ablation).

## Components

```
server/
  corpus_service.py    # FastAPI sidecar: /search (v5-text-small), /rerank (reranker-v3), /stats
  run_searchbox.py     # orchestrator: unzip -> boot sidecar -> drive Pi -> budget-gated stop
pi/
  extensions/corpus-search.ts   # Pi tools: corpus_search, corpus_rerank (gated by SEARCHBOX_TOOLS)
  skills/searchbox/SKILL.md     # the agent's one-page methodology (closed-corpus, budget-burning)
scripts/
  run.sh               # one-shot CLI run
  ablate.py            # ablation sweep runner
config/
  ablations.example.json
```

## Get started

```bash
cd searchbox
cp .env.example .env          # point LLAMA_URL at your OpenAI-compatible model server

# python env (corpus sidecar deps: jina v5-text-small + reranker-v3)
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch -r server/requirements.txt huggingface-hub

# the Pi agent
npm install -g @earendil-works/pi-coding-agent@0.78.0

# a model server (any OpenAI-compatible /v1; e.g. llama-server on :8080)
# then:
bash scripts/run.sh "What does this codebase do and where is auth handled?" ./mycorpus.zip 300000 ./out
cat ./out/ANSWER.md
```

## Ablation

Everything that matters for an ablation study is an **env knob** - no code edits:

| Knob | What it ablates |
| --- | --- |
| `SEARCHBOX_TOOLS` | which corpus tools the agent gets: `corpus_search,corpus_rerank` / `corpus_search` / `corpus_rerank` / `` (none; bash+read only) |
| `LLAMA_URL` + `MODEL_ID` + `CONTEXT_WINDOW` | the base LLM |
| `EMBED_MODEL` / `RERANK_MODEL` | the retrieval models |
| `INPUT_TOKEN_BUDGET` (or per-config `budget`) | the spend floor |

Run a sweep:

```bash
python -m scripts.ablate \
  --query "your question" --corpus ./mycorpus.zip --budget 300000 \
  --matrix config/ablations.example.json --out ./runs/exp1

cat ./runs/exp1/results.jsonl    # one summary row per config (tokens spent, tool calls, answer size, stop reason)
```

The default matrix (omit `--matrix`) is the tool ablation: `full`, `search_only`,
`rerank_only`, `no_tools`. Each config's full job (ANSWER.md, NOTES.md, pi.log, run_meta.json)
lands in `runs/exp1/<config_name>/`.

## License

MIT
