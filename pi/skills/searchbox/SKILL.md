---
name: searchbox
description: Methodology for exhaustively answering a question from a fixed local corpus (an unzipped folder) by spending an input-token budget on tool-driven exploration before answering. Use whenever the task is to answer a prompt grounded ONLY in a provided folder of files.
---

# Searchbox

You answer ONE question using ONLY a fixed, read-only corpus: the `corpus/` directory in
your working directory (the orchestrator gives you its absolute path). There is NO web
access. Every claim in your answer must be grounded in a file inside `corpus/`.

You are not optimizing for speed. You are given an **input-token budget** and your job is to
**spend it** doing real exploration of the corpus before you answer. A shallow answer that
leaves the budget unspent is a failure. Read widely, cross-check, follow leads, verify, and
only then write the final answer. The orchestrator will reject a premature answer and tell
you to keep exploring until the budget is consumed.

## Tools you have

Local-only retrieval over the corpus (no internet):

- **`corpus_search({query, k})`** — semantic search over the corpus (jina-embeddings-v5-text-small).
  Returns top-k chunks with `path`, `chunk`, `score`, and full `text`. Your primary discovery
  tool. Run it many times with **different phrasings and sub-questions** to widen coverage.
- **`corpus_rerank({query, k, top_n})`** — cross-encoder rerank (jina-reranker-v3) over search
  hits (or an explicit `documents` list). Higher precision than search for picking the best
  passages. Use it to confirm which of several candidate chunks actually answer a sub-question.
- **`read` / `bash`** — read whole corpus files for full context (search returns chunks; read
  the surrounding file when a chunk looks load-bearing). Use `bash` (grep/find/wc/jq/etc.) to
  enumerate the corpus, follow references between files, count things, and verify specifics.
  `corpus/` is READ-ONLY: never write into it.
- **`write` / `edit`** — only for your scratch notes and the final answer, written OUTSIDE
  `corpus/` (in the working directory root). You may also write+run small scripts to compute
  or verify a number, then cite the file(s) the inputs came from.

## What you produce

Write these in the working directory (NOT inside `corpus/`):

```
NOTES.md     # running scratchpad: sub-questions, findings, each with the corpus file(s) cited
ANSWER.md    # the final answer (see format below) - this is the deliverable
```

`ANSWER.md` format:
- Lead with the direct answer to the prompt.
- Then the supporting reasoning, organized by sub-question.
- **Every factual claim cites its corpus source(s)** inline as `[path]` or `[path#chunk]`.
- A final `## Sources` section listing every corpus file you relied on.
- If the corpus genuinely does not contain the answer (or only partially), say so explicitly
  and state exactly what is missing — do NOT invent facts or pull from outside knowledge.

## The loop (repeat until the budget is spent)

1. **Map the corpus first.** `bash`: list the tree (`find corpus -type f | head -200`), sizes,
   file types, any README/index/manifest. Build a mental model of what is in there.
2. **Decompose the prompt** into concrete sub-questions. Write them into `NOTES.md` as a
   checklist (`- [ ]` open, `- [x]` answered).
3. **Pick the highest-value open sub-question** (80/20).
4. **Retrieve**: `corpus_search` with several phrasings; `corpus_rerank` to sharpen. `read`
   the full files behind the best chunks — do not answer from a single chunk when the file
   has more context.
5. **Cross-check**: look for corroborating or contradicting passages elsewhere in the corpus.
   Follow references/links/imports/citations between files. Distinguish fact vs inference.
6. **Record** the finding in `NOTES.md` with the exact corpus path(s) it came from; check the
   sub-question off; add any new sub-questions you discovered.
7. **Keep going.** When the obvious sub-questions are closed, deepen: read adjacent files,
   chase edge cases, verify numbers with `bash`, reconcile inconsistencies. There is almost
   always more signal in the corpus — find it until the budget is spent.

## Discipline

- **Grounded only.** No claim without a corpus citation. No outside knowledge in the answer.
- **Read whole files, not just chunks**, whenever a chunk is load-bearing.
- **Vary your queries.** Re-running the same search wastes budget and adds nothing; open new
  angles, synonyms, and sub-topics each time.
- **Verify, don't assume.** Use `bash`/scripts to check numbers, counts, and cross-references.
- **Note dead ends** so you don't re-chase them.
- **Spend the budget on exploration, not on padding** the answer. Depth of investigation, not
  length of prose.

## Stopping

This is a budget-bounded job. Keep exploring until you have **spent the input-token budget**
the orchestrator set. Only then write the final answer:
1. Make sure every sub-question in `NOTES.md` is closed (`- [x]`) or explicitly marked
   unanswerable-from-corpus.
2. Write `ANSWER.md` in the format above, fully cited.
3. Set the first line of `STATUS.md` (in the working directory) to `STATUS: DONE`.

If you write `DONE` before the input-token budget is spent, the orchestrator will reject it,
tell you how much budget remains, and nudge you to keep exploring — open new sub-questions and
read more of the corpus rather than re-stating what you already have.
