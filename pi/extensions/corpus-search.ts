/**
 * Two local-only retrieval tools over the uploaded corpus, served by server/corpus_service.py:
 *   semantic_search  -> jina-embeddings-v5-text-small
 *   passage_rerank   -> jina-reranker-v3
 *
 * Each tool is gated by SEARCHBOX_TOOLS so the ablation harness can disable it without editing
 * this file. CORPUS_INDEX_URL defaults to http://127.0.0.1:8078.
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
// Backward-compatible aliases so older SEARCHBOX_TOOLS values keep working.
const ALIAS: Record<string, string> = {
  corpus_search: "semantic_search",
  corpus_rerank: "passage_rerank",
};
const enabled = (name: string) =>
  ENABLED === null || ENABLED.has(name) || [...ENABLED].some((n) => ALIAS[n] === name);

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
  if (enabled("semantic_search")) {
    pi.registerTool({
      name: "semantic_search",
      label: "Semantic Search",
      description:
        "Find passages in the corpus by meaning, not keywords (embedding similarity). " +
        "Use it to locate relevant text when you do not know the exact wording. " +
        "Embeds on the fly over the scope you choose: by default the whole corpus, or pass " +
        "`paths` to search only specific files. Returns the top-k chunks as {path, chunk, score, text}.",
      parameters: Type.Object({
        query: Type.String({ description: "What you are looking for." }),
        k: Type.Optional(Type.Number({ description: "Number of chunks to return (default 8)." })),
        paths: Type.Optional(Type.Array(Type.String(), {
          description: "Restrict the search to these corpus files (relative paths). Omit to search all.",
        })),
        chunk_size: Type.Optional(Type.Number({ description: "Characters per chunk (default 1400)." })),
      }),
      async execute(_id, params) {
        const p: any = params;
        const query = String(p.query ?? "");
        if (!query) return err("query is required");
        const k = Number.isFinite(p.k) ? Number(p.k) : 8;
        const body: Record<string, unknown> = { query, k };
        if (Array.isArray(p.paths) && p.paths.length) body.paths = p.paths;
        if (Number.isFinite(p.chunk_size)) body.chunk_size = Number(p.chunk_size);
        try {
          return ok(await call("search", body));
        } catch (e: any) {
          return err(String(e?.message || e));
        }
      },
    });
  }

  if (enabled("passage_rerank")) {
    pi.registerTool({
      name: "passage_rerank",
      label: "Passage Rerank",
      description:
        "Re-order a set of passages by how well each answers a query (cross-encoder, more " +
        "accurate than similarity). Use it to pick the best few from a larger candidate set. " +
        "You supply the passages; returns them sorted by relevance_score.",
      parameters: Type.Object({
        query: Type.String({ description: "What the passages should be relevant to." }),
        documents: Type.Array(Type.String(), { description: "Passages to score." }),
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
