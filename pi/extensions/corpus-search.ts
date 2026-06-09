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
      description:
        "Semantic search over the UPLOADED corpus only (jina-embeddings-v5-text-small). " +
        "No web access. Returns top-k chunks with their file path, chunk index, cosine " +
        "score, and full chunk text. Use this to locate relevant passages before reading " +
        "whole files. Call repeatedly with varied queries to widen coverage.",
      parameters: Type.Object({
        query: Type.String({ description: "What to look for in the corpus." }),
        k: Type.Optional(Type.Number({ description: "How many chunks to return (default 8)." })),
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
      description:
        "Cross-encoder rerank (jina-reranker-v3) over the corpus. Pass just {query, k, top_n} " +
        "to pull k candidates by embedding search and rerank them down to the top_n most " +
        "relevant, OR pass {query, documents:[...], top_n} to rerank a specific candidate set. " +
        "Higher precision than corpus_search for picking the single best passages.",
      parameters: Type.Object({
        query: Type.String({ description: "The relevance criterion / question." }),
        k: Type.Optional(Type.Number({ description: "Candidates to pull via search before reranking (default 30)." })),
        top_n: Type.Optional(Type.Number({ description: "How many to return after rerank (default 8)." })),
        documents: Type.Optional(Type.Array(Type.String(), {
          description: "Optional explicit candidate strings to rerank instead of searching.",
        })),
      }),
      async execute(_id, params) {
        const p: any = params;
        const query = String(p.query ?? "");
        if (!query) return err("query is required");
        const body: Record<string, unknown> = { query };
        if (Array.isArray(p.documents) && p.documents.length) body.documents = p.documents;
        if (Number.isFinite(p.k)) body.k = Number(p.k);
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
