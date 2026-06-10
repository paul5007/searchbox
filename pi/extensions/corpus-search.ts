/**
 * Two local-only ATOMIC model primitives over the uploaded corpus, served by
 * server/corpus_service.py:
 *   sentence_embed  -> jina-embeddings-v5-text-small  (text -> vector, written to jsonl)
 *   passage_rerank  -> jina-reranker-v3               (query + docs -> relevance scores)
 *
 * These are deliberately raw single-model operations: the model itself decides where passages
 * come from (grep/read/chunking) and how to use the embeddings (similarity/clustering). There is
 * no high-level "semantic_search" that hides the embed->score->rank pipeline.
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
  corpus_search: "sentence_embed",
  corpus_rerank: "passage_rerank",
  semantic_search: "sentence_embed",
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
  if (enabled("sentence_embed")) {
    pi.registerTool({
      name: "sentence_embed",
      label: "Sentence Embed",
      description:
        "Embed one or more text strings with jina-embeddings-v5-text-small and APPEND the " +
        "resulting vectors to a jsonl file in your working directory. It does NOT return the " +
        "vectors in the response (they are large); you read them back yourself from the file. " +
        "You decide what to embed (e.g. chunks you got via grep/read) and how to use the " +
        "vectors (cosine similarity, clustering, nearest-neighbour search via your own python). " +
        "Vectors are L2-normalized, so a plain dot product equals cosine similarity. " +
        "INPUT: {texts: string[], role?: \"query\"|\"passage\" (default \"passage\"; use \"query\" " +
        "for the question, \"passage\" for corpus text), out?: filename (default \"embeddings.jsonl\")}. " +
        "OUTPUT (json): {path, relpath, count, dim, role}. " +
        "FILE FORMAT: one json object per line: " +
        "{\"i\": int, \"text\": str, \"role\": str, \"dim\": int, \"embedding\": float[dim]}.",
      parameters: Type.Object({
        texts: Type.Array(Type.String(), {
          description: "The text strings to embed (one vector is produced per string).",
        }),
        role: Type.Optional(Type.String({
          description: "\"query\" or \"passage\" (default \"passage\"). v5 retrieval is asymmetric.",
        })),
        out: Type.Optional(Type.String({
          description: "Output jsonl filename in the working dir (default \"embeddings.jsonl\"). " +
            "Vectors are APPENDED, so reuse one file to accumulate or pick a fresh name to separate.",
        })),
      }),
      async execute(_id, params) {
        const p: any = params;
        const texts = Array.isArray(p.texts) ? p.texts.filter((t: any) => typeof t === "string") : [];
        if (!texts.length) return err("texts must be a non-empty array of strings");
        const body: Record<string, unknown> = { texts };
        if (typeof p.role === "string" && p.role) body.role = p.role;
        if (typeof p.out === "string" && p.out) body.out = p.out;
        try {
          return ok(await call("embed", body));
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
        "Score a set of passages by how well each answers a query, with jina-reranker-v3 (a " +
        "cross-encoder: reads query+passage together, more accurate than embedding similarity). " +
        "You supply the candidate passages (e.g. ones you found via grep/read or ranked via " +
        "sentence_embed); this returns them scored and sorted. " +
        "INPUT: {query: string, documents: string[], top_n?: int (default: all)}. " +
        "OUTPUT (json): {results: [{index: int (position in your documents array), " +
        "score: float (relevance, higher=better), text: string}]} sorted by score desc.",
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
