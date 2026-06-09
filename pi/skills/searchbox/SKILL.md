---
name: searchbox
description: Answer a question using only a local, read-only corpus folder, exploring it thoroughly before answering. Use when the task is to answer a prompt grounded in a provided folder of files.
---

# Searchbox

Answer the question using ONLY the read-only `corpus/` folder in your working directory. No web,
no outside knowledge. Every claim in your answer must come from a file in `corpus/`.

Explore thoroughly before answering. Use `corpus_search` (semantic search) and `corpus_rerank`
(precise reranking) to find passages, then `read` the full files that matter. Vary your queries
and follow leads across files until you have enough evidence.

When done, write `ANSWER.md` in the working directory (NOT in `corpus/`): the direct answer
first, then the reasoning, with every claim citing its source as `[path]`, and a final
`## Sources` list. If the corpus does not answer the question, say so plainly rather than
guessing.
