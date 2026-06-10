#!/usr/bin/env python3
"""Corpus sidecar: local jina models over the UNZIPPED input corpus (read-only).

Two local Jina models, no network/web tools:
  - jina-embeddings-v5-text-small  -> semantic search over the corpus  (/search)
  - jina-reranker-v3               -> cross-encoder reranking          (/rerank)

This is the searchbox analogue of dataroom's index sidecar, but inverted: dataroom
indexed the OUTPUT it was building; searchbox indexes the fixed INPUT corpus the user
handed it, and never writes to it. The corpus is chunked + embedded once at boot and the
embeddings are cached on disk keyed by a content hash, so repeated ablation runs over the
same zip reuse the index instead of re-embedding.

jina-embeddings-v5 retrieval is ASYMMETRIC: queries use the "query" prompt and passages
the "document" prompt, so cosine scores are calibrated.

Endpoints (POST JSON):
  /search  {query, k=8}                              -> top-k corpus chunks w/ cosine score
  /rerank  {query, documents?[], k?, top_n=8}        -> reranker-v3 ordering
                                                         (documents given, OR pull k via search)
  /stats   {}                                         -> {chunks, files, embed_model, rerank_model}
"""
import os, json, glob, hashlib, time
import numpy as np
from fastapi import FastAPI, Request
import uvicorn

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Keep the embedder/reranker off the GPU by default so the LLM owns VRAM (mirrors dataroom's
# EMBED_DEVICE=cpu rationale; jina-v5 base+LoRA adapters otherwise OOM a tight card). Set
# EMBED_DEVICE=cuda to move them onto the GPU when there is headroom.
EMBED_DEVICE = os.environ.get("EMBED_DEVICE", "cpu")
if EMBED_DEVICE.startswith("cpu"):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

CORPUS_DIR = os.path.abspath(os.environ.get("CORPUS_DIR", "corpus"))
CACHE_DIR = os.path.abspath(os.environ.get("CORPUS_CACHE_DIR",
                                           os.path.join(os.path.dirname(CORPUS_DIR), ".corpus_cache")))
EMBED_MODEL = os.environ.get("EMBED_MODEL", "jinaai/jina-embeddings-v5-text-small")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "jinaai/jina-reranker-v3")
EMBED_TASK = os.environ.get("EMBED_TASK", "retrieval")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1400"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "180"))

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
_embs = None          # (N, D) float32, L2-normalized
_meta = []            # [{path, chunk, text}]


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


def _encode(texts, role: str) -> np.ndarray:
    """Encode with the retrieval adapter + role-specific prompt (query vs document).

    Degrades gracefully if a build lacks the named prompts (older/odd v5 packaging)."""
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


def _corpus_hash(files: list) -> str:
    h = hashlib.sha1()
    h.update(f"{EMBED_MODEL}|{CHUNK_SIZE}|{CHUNK_OVERLAP}".encode())
    for ap in files:
        try:
            st = os.stat(ap)
            h.update(os.path.relpath(ap, CORPUS_DIR).encode())
            h.update(str(st.st_size).encode())
            h.update(str(int(st.st_mtime)).encode())
        except OSError:
            continue
    return h.hexdigest()[:16]


def build_index():
    """Chunk + embed the whole corpus once. Cached on disk by corpus content hash."""
    global _embs, _meta
    files = _corpus_files()
    key = _corpus_hash(files)
    os.makedirs(CACHE_DIR, exist_ok=True)
    npz = os.path.join(CACHE_DIR, f"{key}.npz")
    jsonl = os.path.join(CACHE_DIR, f"{key}.jsonl")
    if os.path.exists(npz) and os.path.exists(jsonl):
        _embs = np.load(npz)["embs"]
        _meta = [json.loads(l) for l in open(jsonl) if l.strip()]
        print(f"[corpus] loaded cached index {key}: {len(_meta)} chunks", flush=True)
        return

    t0 = time.time()
    texts, meta = [], []
    for ap in files:
        try:
            raw = open(ap, errors="ignore").read()
        except Exception:
            continue
        rel = os.path.relpath(ap, CORPUS_DIR)
        for ci, c in enumerate(chunk(raw)):
            texts.append(c)
            meta.append({"path": rel, "chunk": ci, "text": c})
    if texts:
        embs = _encode(texts, role="passage")
    else:
        embs = np.zeros((0, 0), dtype=np.float32)
    _embs, _meta = embs, meta
    np.savez_compressed(npz, embs=embs)
    with open(jsonl, "w") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"[corpus] indexed {len(files)} files -> {len(meta)} chunks in "
          f"{time.time()-t0:.1f}s (cache {key})", flush=True)


# --- endpoints ---------------------------------------------------------------
@app.post("/search")
async def search(req: Request):
    body = await req.json()
    q = body.get("query", "")
    k = int(body.get("k", 8))
    if _embs is None or _embs.shape[0] == 0 or not q:
        return {"results": [], "count": 0}
    qe = _encode([q], role="query")[0]
    sims = _embs @ qe
    order = np.argsort(-sims)[:k]
    results = [{
        "path": _meta[i]["path"],
        "chunk": int(_meta[i]["chunk"]),
        "score": round(float(sims[i]), 4),
        "text": _meta[i]["text"],
    } for i in order]
    return {"results": results, "count": int(_embs.shape[0])}


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
    m = rerank_model()
    kwargs = {"top_n": int(top_n)} if top_n is not None else {}
    ranked = m.rerank(q, list(docs), **kwargs)
    out = [{"score": round(float(r.get("relevance_score", 0.0)), 4),
            "text": r.get("document", ""), "index": int(r.get("index", -1))} for r in ranked]
    return {"results": out}


@app.post("/stats")
async def stats(_: Request):
    files = sorted({m["path"] for m in _meta})
    return {"chunks": int(_embs.shape[0]) if _embs is not None and _embs.size else 0,
            "files": files, "file_count": len(files),
            "embed_model": EMBED_MODEL, "rerank_model": RERANK_MODEL,
            "corpus_dir": CORPUS_DIR}


if __name__ == "__main__":
    build_index()
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("CORPUS_PORT", "8078")))
