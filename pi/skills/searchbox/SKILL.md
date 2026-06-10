---
name: searchbox
description: Answer a question using a local folder of files as the only source. Use when the task is to answer a prompt from a provided corpus folder.
---

# Searchbox

Answer the question below using the `corpus/` folder in your working directory as your source.
You have no network access. When done, write your answer to `ANSWER.md` in the working directory
(not inside `corpus/`).

## Tools available over the corpus

Besides reading files directly and `grep`-style keyword search, you have two retrieval tools
over `corpus/`:

- `semantic_search` — find passages by meaning rather than exact wording. When `grep` is too
  literal (you don't know the precise phrasing, or the relevant text uses different words than
  the question), search for it semantically instead. Returns the top matching chunks with their
  source paths.
- `passage_rerank` — given a query and a set of candidate passages, re-order them by how well
  each one actually answers the query (more accurate than raw similarity). Use it to narrow a
  larger candidate set down to the best few.

These work directly over the corpus as-is, alongside `grep` and direct file reads.
