/**
 * corpus-search: local-only retrieval tools over the uploaded corpus, backed by
 * jina-embeddings-v5-text-small (semantic search) and jina-reranker-v3 (cross-encoder),
 * served by server/corpus_service.py. NO web access - this is the searchbox inversion of
 * dataroom's open-web jina CLI: the agent may only look inside the corpus it was given.
 *
 * Tools (each gated by an env flag so the ablation harness can disable individual tools
 * WITHOUT touching this file - flip SEARCHBOX_TOOLS in the orchestrator):
 *   corpus_search { query, k }              -> top-k chunks by v5-text-small cosine
 *   corpus_rerank { query, k?, top_n }      -> reranker-v3 over search hits (or given docs)
 *
 * Env:
 *   CORPUS_INDEX_URL   default http://127.0.0.1:8078
 *   SEARCHBOX_TOOLS    comma list of enabled tool names (default: all). Ablation knob.
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

const BASE = process.env.CORPUS_INDEX_URL || "http://127.0.0.1:8078";

// Ablation gate: which tools this run is allowed to register. Empty/unset = all.
const ENABLED = (() => {
  const raw = (process.env.SEARCHBOX_TOOLS || "").trim();
  if (!raw) return null; // null => allow all
  return new Set(raw.split(",").map((s) => s.trim()).filter(Boolean));
})();
const enabled = (name: string) => ENABLED === null || ENABLED.has(name);

async function call(op: string, body: Record<string, unknown>): Promise<string> {
  const res = await fetch(`${BASE}/${op}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  const text = await res.text();
  if (!res.ok) throw new Error(`corpus service ${op} -> ${res.status}: ${text}`);
  return text;
}

function ok(text: string) {
  return { content: [{ type: "text" as const, text }], details: {} };
}
function err(text: string) {
  return { content: [{ type: "text" as const, text }], isError: true, details: {} };
}

export default function (pi: ExtensionAPI) {
  if (enabled("corpus_search")) {
    pi.registerTool({
      name: "corpus_search",
      label: "Corpus Search",
      // Neutral: state what the model is and what the call returns. No guidance on when or
      // how to use it (the experiment observes whether the model reaches for it on its own).
      description:
        "Embed `query` with jina-embeddings-v5-text-small and return the corpus text chunks " +
        "with the highest cosine similarity. Returns {path, chunk, score, text} for the top k.",
      parameters: Type.Object({
        query: Type.String({ description: "Query text to embed." }),
        k: Type.Optional(Type.Number({ description: "Number of chunks to return (default 8)." })),
      }),
      async execute(_id, params) {
        const p: any = params;
        const query = String(p.query ?? "");
        if (!query) return err("query is required");
        const k = Number.isFinite(p.k) ? Number(p.k) : 8;
        try {
          return ok(await call("search", { query, k }));
        } catch (e: any) {
          return err(String(e?.message || e));
        }
      },
    });
  }

  if (enabled("corpus_rerank")) {
    pi.registerTool({
      name: "corpus_rerank",
      label: "Corpus Rerank",
      // Pure reranker: query + a list of documents -> relevance scores. It does NOT fetch its
      // own candidates (no hidden embedding search) - the caller supplies the documents. Neutral
      // description, basic model usage only.
      description:
        "Score each document in `documents` for relevance to `query` with jina-reranker-v3 " +
        "(a cross-encoder), and return them sorted by relevance_score (highest first).",
      parameters: Type.Object({
        query: Type.String({ description: "Query text." }),
        documents: Type.Array(Type.String(), { description: "Documents to score against the query." }),
        top_n: Type.Optional(Type.Number({ description: "Return only the top N (default: all)." })),
      }),
      async execute(_id, params) {
        const p: any = params;
        const query = String(p.query ?? "");
        if (!query) return err("query is required");
        if (!Array.isArray(p.documents) || !p.documents.length) return err("documents is required");
        const body: Record<string, unknown> = { query, documents: p.documents };
        if (Number.isFinite(p.top_n)) body.top_n = Number(p.top_n);
        try {
          return ok(await call("rerank", body));
        } catch (e: any) {
          return err(String(e?.message || e));
        }
      },
    });
  }
}
