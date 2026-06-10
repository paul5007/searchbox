#!/usr/bin/env python3
"""Corpus sidecar: local jina models over the UNZIPPED input corpus (read-only).

Two local Jina models, no network/web tools:
  - jina-embeddings-v5-text-small  -> semantic search over the corpus  (/search)
  - jina-reranker-v3               -> cross-encoder reranking          (/rerank)

NOTHING is indexed at boot. There is no precomputed index. The model decides, during the run,
whether it wants embedding search at all - and if so, over what scope and with what chunking.
/search embeds ON THE FLY: it reads the requested files (all corpus files, or only the `paths`
the model names), chunks them (default size, or a `chunk_size` the model picks), embeds, and
returns the top matches. Within one run we memoize an embedded scope so a repeated identical
search does not re-embed, but that index is built lazily on first use, never ahead of time.

jina-embeddings-v5 retrieval is ASYMMETRIC: queries use the "query" prompt and passages
the "document" prompt, so cosine scores are calibrated.

Endpoints (POST JSON):
  /search  {query, k=8, paths?[], chunk_size?, chunk_overlap?}  -> top-k chunks (on-the-fly embed)
  /rerank  {query, documents[], top_n?}                          -> reranker-v3 ordering
  /stats   {}                                                    -> {files, file_count, models}
"""
import os, json, glob, hashlib, time
import numpy as np
from fastapi import FastAPI, Request
import urllib.request
import uvicorn

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Backend per model: "local" (self-hosted weights) or "api" (Jina cloud, https://api.jina.ai).
# This is invisible to the agent - the tools and their outputs are identical either way; only
# this env decides where the embed/rerank actually runs. Clean ablation axis: local vs api.
EMBED_BACKEND = os.environ.get("EMBED_BACKEND", "local").lower()
RERANK_BACKEND = os.environ.get("RERANK_BACKEND", "local").lower()
JINA_API_KEY = os.environ.get("JINA_API_KEY", "")
JINA_API_BASE = os.environ.get("JINA_API_BASE", "https://api.jina.ai/v1")
# Model ids differ by backend: local uses HF repo ids, api uses Jina model names.
API_EMBED_MODEL = os.environ.get("API_EMBED_MODEL", "jina-embeddings-v5-text-small")
API_RERANK_MODEL = os.environ.get("API_RERANK_MODEL", "jina-reranker-v3")

# Keep the LOCAL embedder/reranker off the GPU by default so the LLM owns VRAM (jina-v5
# base+LoRA otherwise OOM a tight card). Set EMBED_DEVICE=cuda when there is headroom.
EMBED_DEVICE = os.environ.get("EMBED_DEVICE", "cpu")
if EMBED_DEVICE.startswith("cpu"):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

CORPUS_DIR = os.path.abspath(os.environ.get("CORPUS_DIR", "corpus"))
EMBED_MODEL = os.environ.get("EMBED_MODEL", "jinaai/jina-embeddings-v5-text-small")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "jinaai/jina-reranker-v3")
EMBED_TASK = os.environ.get("EMBED_TASK", "retrieval")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1400"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "180"))


def _api_post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"{JINA_API_BASE}/{path}",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {JINA_API_KEY}",
                 "Content-Type": "application/json", "Accept": "application/json",
                 "User-Agent": "Mozilla/5.0"})  # CF blocks the default Python-urllib UA
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)

# Text-like files we will index. Binary/asset files are skipped (the agent can still `read`
# or `bash` them directly). Extendable via CORPUS_GLOBS (comma-separated globs).
DEFAULT_GLOBS = (
    "**/*.md", "**/*.txt", "**/*.rst", "**/*.py", "**/*.js", "**/*.ts", "**/*.tsx",
    "**/*.json", "**/*.jsonl", "**/*.yaml", "**/*.yml", "**/*.toml", "**/*.csv",
    "**/*.tsv", "**/*.html", "**/*.htm", "**/*.xml", "**/*.tex", "**/*.c", "**/*.h",
    "**/*.cpp", "**/*.cc", "**/*.go", "**/*.rs", "**/*.java", "**/*.sql", "**/*.sh",
    "**/*.ipynb", "**/*.log", "**/*.cfg", "**/*.ini", "**/*.env",
)
_globs_env = os.environ.get("CORPUS_GLOBS", "").strip()
INDEX_GLOBS = tuple(g.strip() for g in _globs_env.split(",") if g.strip()) or DEFAULT_GLOBS
MAX_FILE_BYTES = int(os.environ.get("MAX_FILE_BYTES", str(4 * 1024 * 1024)))  # skip huge blobs

app = FastAPI()

_embed_model = None
_rerank_model = None
# Lazy per-scope index cache: key (paths tuple, chunk_size, overlap) -> {embs, meta}. Built only
# when /search is first called for that scope; nothing exists until the model asks. Cleared via
# nothing - it just lives for the process. No boot index, no on-disk precompute.
_scope_cache: dict = {}


# --- models ------------------------------------------------------------------
def embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        dev = EMBED_DEVICE
        if dev.startswith("cuda"):
            try:
                import torch
                if not torch.cuda.is_available():
                    dev = "cpu"
            except Exception:
                dev = "cpu"
        print(f"[corpus] loading embed model {EMBED_MODEL} on {dev}", flush=True)
        _embed_model = SentenceTransformer(EMBED_MODEL, device=dev, trust_remote_code=True)
    return _embed_model


def rerank_model():
    global _rerank_model
    if _rerank_model is None:
        from transformers import AutoModel
        print(f"[corpus] loading rerank model {RERANK_MODEL}", flush=True)
        m = AutoModel.from_pretrained(RERANK_MODEL, dtype="auto", trust_remote_code=True)
        m.eval()
        if EMBED_DEVICE.startswith("cuda"):
            try:
                import torch
                if torch.cuda.is_available():
                    m = m.to("cuda")
            except Exception:
                pass
        _rerank_model = m
    return _rerank_model


def _encode_api(texts, role: str) -> np.ndarray:
    """Encode via Jina cloud embeddings API. Same v5 retrieval task, asymmetric query/passage."""
    task = "retrieval.query" if role == "query" else "retrieval.passage"
    out = []
    B = 128
    for i in range(0, len(texts), B):
        d = _api_post("embeddings", {"model": API_EMBED_MODEL, "task": task,
                                     "input": list(texts[i:i + B])})
        rows = sorted(d["data"], key=lambda r: r.get("index", 0))
        out.extend(r["embedding"] for r in rows)
    arr = np.asarray(out, dtype=np.float32)
    # API returns unnormalized; L2-normalize so cosine == dot, matching the local path.
    if arr.size:
        arr /= (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
    return arr


def _encode(texts, role: str) -> np.ndarray:
    """Encode with the retrieval adapter + role-specific prompt (query vs document).

    Routes to the Jina cloud API when EMBED_BACKEND=api; otherwise the local model.
    Degrades gracefully if a build lacks the named prompts (older/odd v5 packaging)."""
    if EMBED_BACKEND == "api":
        return _encode_api(texts, role)
    m = embed_model()
    wants = ["query"] if role == "query" else ["document", "passage"]
    available = set((getattr(m, "prompts", None) or {}).keys())
    names = [n for n in wants if not available or n in available] or wants[:1]
    attempts = []
    for n in names:
        attempts.append({"task": EMBED_TASK, "prompt_name": n})
        attempts.append({"prompt_name": n})
    attempts.append({"task": EMBED_TASK})
    for kwargs in attempts:
        try:
            e = m.encode(texts, normalize_embeddings=True, **kwargs)
            return np.asarray(e, dtype=np.float32)
        except (TypeError, ValueError, KeyError):
            continue
    return np.asarray(m.encode(texts, normalize_embeddings=True), dtype=np.float32)


# --- corpus indexing ---------------------------------------------------------
def chunk(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return out


def _corpus_files() -> list:
    found = []
    seen = set()
    for g in INDEX_GLOBS:
        for ap in glob.glob(os.path.join(CORPUS_DIR, g), recursive=True):
            if not os.path.isfile(ap) or ap in seen:
                continue
            try:
                if os.path.getsize(ap) > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            seen.add(ap)
            found.append(ap)
    return sorted(found)


def _resolve_paths(paths) -> list:
    """Map caller-supplied paths to absolute corpus files, guarding against escape.
    If no paths given, use every text-like corpus file.

    Robust to how the model actually calls it (verified against real traces):
      - ABSOLUTE paths: the model echoes the absolute paths it saw in tool results
        (e.g. /.../work/corpus/foo.md). os.path.join would discard CORPUS_DIR for an
        absolute arg, so we strip a leading CORPUS_DIR / "corpus/" prefix and treat the
        remainder as corpus-relative.
      - DIRECTORIES (incl. the corpus root itself, e.g. paths=["."] or the abs corpus dir):
        expand to every text-like file under that directory, instead of returning nothing.
      - Anything that still resolves to a single file is kept as-is.
    Any path that matches nothing is skipped; if NOTHING matches, fall back to the whole
    corpus so a clumsy `paths` arg degrades to a full search rather than empty results."""
    if not paths:
        return _corpus_files()
    base = os.path.realpath(CORPUS_DIR)
    allowed = set(_corpus_files())  # the canonical text-like file set (abspath, glob-based)
    allowed_real = {os.path.realpath(f): f for f in allowed}  # realpath -> canonical form
    out = []

    def _add(canonical):
        if canonical not in out:
            out.append(canonical)

    for p in paths:
        p = str(p).strip()
        if not p:
            continue
        # Normalize an absolute path that points inside (or at) the corpus to corpus-relative.
        if os.path.isabs(p):
            rp = os.path.realpath(p)
            if rp == base or rp.startswith(base + os.sep):
                p = os.path.relpath(rp, base)
            # else: leave as-is; the guard below rejects escapes.
        ap = os.path.realpath(os.path.join(CORPUS_DIR, p))
        if not (ap == base or ap.startswith(base + os.sep)):
            continue  # escape attempt
        if os.path.isfile(ap):
            # Keep only files in the indexable set (canonical form), matched via realpath.
            canon = allowed_real.get(ap)
            if canon:
                _add(canon)
        elif os.path.isdir(ap):
            # Expand to every indexable file under this directory (or the whole corpus if it
            # IS the corpus root). Match via realpath to survive abspath/symlink differences.
            for fr, canon in allowed_real.items():
                if ap == base or fr == ap or fr.startswith(ap + os.sep):
                    _add(canon)
    # Clumsy/unmatched paths degrade to a full-corpus search instead of empty results.
    if not out:
        return _corpus_files()
    return out


def _build_scope(files: list, size: int, overlap: int):
    """Read+chunk+embed the given files NOW. Returns (embs (N,D) normalized, meta list)."""
    texts, meta = [], []
    for ap in files:
        try:
            raw = open(ap, errors="ignore").read()
        except Exception:
            continue
        rel = os.path.relpath(ap, CORPUS_DIR)
        for ci, c in enumerate(chunk(raw, size, overlap)):
            texts.append(c)
            meta.append({"path": rel, "chunk": ci, "text": c})
    embs = _encode(texts, role="passage") if texts else np.zeros((0, 0), dtype=np.float32)
    return embs, meta


# --- endpoints ---------------------------------------------------------------
@app.post("/search")
async def search(req: Request):
    """On-the-fly semantic search. Embeds the requested scope at call time (no boot index).

    body: {query, k=8, paths?[], chunk_size?, chunk_overlap?}
      - paths omitted  -> search the whole corpus (model chose to search everything)
      - paths given    -> only those files (model scoped it down to a part)
      - chunk_size/overlap -> model controls granularity (defaults otherwise)
    """
    body = await req.json()
    q = body.get("query", "")
    if not q:
        return {"results": [], "scope_chunks": 0}
    k = int(body.get("k", 8))
    size = int(body.get("chunk_size", CHUNK_SIZE))
    overlap = int(body.get("chunk_overlap", CHUNK_OVERLAP))
    files = _resolve_paths(body.get("paths"))
    if not files:
        return {"results": [], "scope_chunks": 0}

    # Memoize this exact scope for the run so repeated identical searches don't re-embed.
    # Built lazily here on first use - never at boot.
    key = (tuple(sorted(os.path.relpath(f, CORPUS_DIR) for f in files)), size, overlap)
    cached = _scope_cache.get(key)
    if cached is None:
        t0 = time.time()
        embs, meta = _build_scope(files, size, overlap)
        _scope_cache[key] = cached = {"embs": embs, "meta": meta}
        print(f"[corpus] on-the-fly embed: {len(files)} files -> {len(meta)} chunks "
              f"(size={size}, overlap={overlap}) in {time.time()-t0:.1f}s", flush=True)
    embs, meta = cached["embs"], cached["meta"]
    if embs is None or embs.shape[0] == 0:
        return {"results": [], "scope_chunks": 0}

    qe = _encode([q], role="query")[0]
    sims = embs @ qe
    order = np.argsort(-sims)[:k]
    results = [{
        "path": meta[i]["path"],
        "chunk": int(meta[i]["chunk"]),
        "score": round(float(sims[i]), 4),
        "text": meta[i]["text"],
    } for i in order]
    return {"results": results, "scope_chunks": int(embs.shape[0]), "scope_files": len(files)}


@app.post("/rerank")
async def rerank(req: Request):
    """Pure cross-encoder rerank with jina-reranker-v3: score caller-supplied documents.

    Deliberately does NOT fetch its own candidates (no hidden embedding search). The tool stays
    a single-model primitive (basic reranker usage) - the model itself decides what to feed it.
    """
    body = await req.json()
    q = body.get("query", "")
    docs = body.get("documents")
    top_n = body.get("top_n")
    if not q or not docs:
        return {"results": []}
    docs = list(docs)
    if RERANK_BACKEND == "api":
        payload = {"model": API_RERANK_MODEL, "query": q, "documents": docs}
        if top_n is not None:
            payload["top_n"] = int(top_n)
        d = _api_post("rerank", payload)
        out = [{"score": round(float(r.get("relevance_score", 0.0)), 4),
                "text": (r.get("document") or {}).get("text", "") if isinstance(r.get("document"), dict)
                        else r.get("document", ""),
                "index": int(r.get("index", -1))} for r in d.get("results", [])]
        return {"results": out}
    m = rerank_model()
    kwargs = {"top_n": int(top_n)} if top_n is not None else {}
    ranked = m.rerank(q, docs, **kwargs)
    out = [{"score": round(float(r.get("relevance_score", 0.0)), 4),
            "text": r.get("document", ""), "index": int(r.get("index", -1))} for r in ranked]
    return {"results": out}


@app.post("/stats")
async def stats(_: Request):
    """List the corpus files. Does NOT embed anything - there is no boot index."""
    files = sorted(os.path.relpath(f, CORPUS_DIR) for f in _corpus_files())
    return {"files": files, "file_count": len(files),
            "embedded_scopes": len(_scope_cache),
            "embed_backend": EMBED_BACKEND, "rerank_backend": RERANK_BACKEND,
            "embed_model": API_EMBED_MODEL if EMBED_BACKEND == "api" else EMBED_MODEL,
            "rerank_model": API_RERANK_MODEL if RERANK_BACKEND == "api" else RERANK_MODEL,
            "corpus_dir": CORPUS_DIR}


if __name__ == "__main__":
    # No index is built at boot. Models load lazily on first /search or /rerank.
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("CORPUS_PORT", "8078")))
