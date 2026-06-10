/**
 * Local-only tools over the uploaded dataroom, served by server/dataroom_service.py.
 *
 * DEFAULT (loaded when SEARCHBOX_TOOLS is unset) - two ATOMIC model primitives:
 *   sentence_embed  -> jina-embeddings-v5-text-small  (text -> vector, written to jsonl)
 *   passage_rerank  -> jina-reranker-v3               (query + docs -> relevance scores)
 * These are deliberately raw single-model operations: the model itself decides where passages
 * come from (grep/read/chunking) and how to use the embeddings (similarity/clustering).
 *
 * OPT-IN (only loaded when SEARCHBOX_TOOLS explicitly lists it):
 *   semantic_search -> the high-level embed->rank->top-k pipeline (kept in the repo but NOT
 *                      loaded by default, so we can observe whether the model assembles its own
 *                      retrieval from the atomic sentence_embed primitive).
 *
 * Tools are gated by SEARCHBOX_TOOLS so the ablation harness can pick the set without editing
 * this file. DATAROOM_INDEX_URL defaults to http://127.0.0.1:8078.
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

const BASE = process.env.DATAROOM_INDEX_URL || "http://127.0.0.1:8078";

// Ablation gate: which tools this run is allowed to register.
//   SEARCHBOX_TOOLS unset/empty => the DEFAULT set (see DEFAULT_TOOLS below).
//   SEARCHBOX_TOOLS="a,b"      => exactly those tools.
const ENABLED = (() => {
  const raw = (process.env.SEARCHBOX_TOOLS || "").trim();
  if (!raw) return null; // null => use DEFAULT_TOOLS
  return new Set(raw.split(",").map((s) => s.trim()).filter(Boolean));
})();
// Backward-compatible aliases so older SEARCHBOX_TOOLS values keep working.
const ALIAS: Record<string, string> = {
  dataroom_search: "sentence_embed",
  dataroom_rerank: "passage_rerank",
};
// The default tool set when SEARCHBOX_TOOLS is unset. NOTE: semantic_search is intentionally
// NOT here - it is the high-level "do the whole pipeline" tool we keep in the repo but do NOT
// load by default, so we can observe whether the model assembles its own retrieval from the
// atomic sentence_embed primitive. To load it, set SEARCHBOX_TOOLS to include "semantic_search".
const DEFAULT_TOOLS = new Set(["sentence_embed", "passage_rerank"]);
const enabled = (name: string) => {
  if (ENABLED === null) return DEFAULT_TOOLS.has(name);
  return ENABLED.has(name) || [...ENABLED].some((n) => ALIAS[n] === name);
};

async function call(op: string, body: Record<string, unknown>): Promise<string> {
  const res = await fetch(`${BASE}/${op}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  const text = await res.text();
  if (!res.ok) throw new Error(`dataroom service ${op} -> ${res.status}: ${text}`);
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
        "for the question, \"passage\" for dataroom text), out?: filename (default \"embeddings.jsonl\")}. " +
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

  // High-level pipeline tool. KEPT in the repo but NOT in DEFAULT_TOOLS, so it only loads when
  // SEARCHBOX_TOOLS explicitly lists "semantic_search". By default the model does not see it -
  // we want to observe whether it builds its own retrieval from sentence_embed. Backed by the
  // /search endpoint in dataroom_service.py (still live).
  if (enabled("semantic_search")) {
    pi.registerTool({
      name: "semantic_search",
      label: "Semantic Search",
      description:
        "Find passages in the dataroom by meaning, not keywords (embedding similarity). " +
        "Use it to locate relevant text when you do not know the exact wording. " +
        "Embeds on the fly over the scope you choose: by default the whole dataroom, or pass " +
        "`paths` to search only specific files. Returns the top-k chunks as {path, chunk, score, text}.",
      parameters: Type.Object({
        query: Type.String({ description: "What you are looking for." }),
        k: Type.Optional(Type.Number({ description: "Number of chunks to return (default 8)." })),
        paths: Type.Optional(Type.Array(Type.String(), {
          description: "Restrict the search to these dataroom files (relative paths). Omit to search all.",
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
