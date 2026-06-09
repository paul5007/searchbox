#!/usr/bin/env python3
"""Searchbox API + UI.

POST /jobs        (multipart: prompt, budget, corpus=<zipfile>)  -> {job_id}
GET  /jobs                         -> list
GET  /jobs/{id}                    -> status
GET  /jobs/{id}/stats              -> live metrics (drives the dashboard)
GET  /jobs/{id}/answer             -> ANSWER.md (text)
GET  /jobs/{id}/file?path=...      -> a corpus/output file
GET  /jobs/{id}/dashboard          -> the dashboard HTML
GET  /                             -> submit page
GET  /health
"""
import json, os, signal, subprocess, sys, threading, time, uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, PlainTextResponse, HTMLResponse
import uvicorn

from server.stats import job_stats

HERE = Path(__file__).resolve().parent
WEB = HERE.parent / "web"
JOBS = Path(os.environ.get("JOBS_DIR", str(HERE.parent / "data" / "jobs"))).resolve()
JOBS.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Searchbox")
_jobs: dict[str, dict] = {}
_lock = threading.Lock()
_queue: list[str] = []
_cond = threading.Condition(_lock)
_current = {"job_id": None, "proc": None}


def _save_meta(job_id: str):
    try:
        (JOBS / job_id / "meta.json").write_text(json.dumps(_jobs.get(job_id, {})))
    except Exception:
        pass


def _load_meta(job_id: str) -> dict:
    mp = JOBS / job_id / "meta.json"
    if mp.exists():
        try:
            return json.loads(mp.read_text())
        except Exception:
            return {}
    return {}


def _run_one(job_id: str):
    job_dir = JOBS / job_id
    with _lock:
        meta = dict(_jobs.get(job_id, {}))
        _jobs[job_id]["status"] = "running"
        _jobs[job_id]["started"] = time.time()
    _save_meta(job_id)
    cmd = [sys.executable, "-m", "server.run_searchbox",
           "--query", meta["query"], "--corpus", str(job_dir / "input.zip"),
           "--out", str(job_dir), "--budget", str(meta["budget"])]
    env = dict(os.environ)
    for k in ("SEARCHBOX_TOOLS", "LLAMA_URL", "MODEL_ID", "CONTEXT_WINDOW",
              "EMBED_MODEL", "RERANK_MODEL", "BUDGET_METRIC"):
        if meta.get(k):
            env[k] = str(meta[k])
    log = open(job_dir / "orchestrator.log", "a")
    proc = subprocess.Popen(cmd, cwd=str(HERE.parent), env=env, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)
    with _lock:
        _current["job_id"], _current["proc"] = job_id, proc
    rc = proc.wait()
    with _lock:
        _current["job_id"], _current["proc"] = None, None
        meta2 = {}
        rm = job_dir / "run_meta.json"
        if rm.exists():
            try:
                meta2 = json.loads(rm.read_text())
            except Exception:
                pass
        _jobs[job_id]["status"] = "done" if meta2.get("done") else ("stopped" if rm.exists() else "failed")
        _jobs[job_id]["stop_reason"] = meta2.get("stop_reason")
        _jobs[job_id]["rc"] = rc
        _jobs[job_id]["finished"] = time.time()
    _save_meta(job_id)


def _worker():
    while True:
        with _cond:
            while not _queue:
                _cond.wait()
            job_id = _queue.pop(0)
        try:
            _run_one(job_id)
        except Exception as e:
            with _lock:
                _jobs.setdefault(job_id, {})["status"] = "failed"
                _jobs[job_id]["error"] = str(e)[:300]
            _save_meta(job_id)


threading.Thread(target=_worker, daemon=True).start()


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/jobs")
async def create(prompt: str = Form(...), budget: int = Form(...), corpus: UploadFile = File(...)):
    prompt = (prompt or "").strip()
    if len(prompt) < 5:
        raise HTTPException(400, "prompt too short")
    if budget < 1:
        raise HTTPException(400, "budget must be >= 1")
    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    data = await corpus.read()
    (job_dir / "input.zip").write_bytes(data)
    with _cond:
        _jobs[job_id] = {"status": "queued", "query": prompt, "budget": budget,
                         "corpus_name": corpus.filename, "corpus_bytes": len(data),
                         "submitted": time.time()}
        _queue.append(job_id)
        _cond.notify_all()
    _save_meta(job_id)
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs")
def list_jobs():
    ids = set(_jobs.keys())
    if JOBS.exists():
        for p in JOBS.iterdir():
            if p.is_dir():
                ids.add(p.name)
    rows = []
    for jid in ids:
        with _lock:
            meta = dict(_jobs.get(jid, {})) or _load_meta(jid)
        job_dir = JOBS / jid
        rm = job_dir / "run_meta.json"
        spent = pct = 0
        bdg = meta.get("budget")
        if rm.exists():
            try:
                m = json.loads(rm.read_text())
                spent = m.get("input_tokens_spent", 0)
                bdg = bdg or m.get("budget")
                pct = m.get("budget_pct") or 0
            except Exception:
                pass
        rows.append({"job_id": jid, "status": meta.get("status", "unknown"),
                     "stop_reason": meta.get("stop_reason"),
                     "query": (meta.get("query") or "")[:200],
                     "budget": bdg, "spent": spent, "percent": pct,
                     "started": meta.get("started"), "finished": meta.get("finished")})
    rows.sort(key=lambda r: (r.get("started") or 0), reverse=True)
    return {"jobs": rows}


@app.get("/jobs/{job_id}")
def status(job_id: str):
    with _lock:
        j = dict(_jobs.get(job_id, {}))
    if not j:
        j = _load_meta(job_id)
        if not j:
            raise HTTPException(404, "unknown job")
    return j


@app.get("/jobs/{job_id}/stats")
def stats_ep(job_id: str):
    job_dir = JOBS / job_id
    if not job_dir.exists():
        raise HTTPException(404, "unknown job")
    with _lock:
        meta = dict(_jobs.get(job_id, {})) or _load_meta(job_id)
    s = job_stats(job_dir, meta.get("budget"))
    s["job_id"] = job_id
    s["job_status"] = meta.get("status") or "unknown"
    s["query"] = meta.get("query", "")
    s["model_id"] = meta.get("MODEL_ID") or os.environ.get("MODEL_ID", "qwen3.6")
    s["tools_enabled"] = meta.get("SEARCHBOX_TOOLS") or os.environ.get("SEARCHBOX_TOOLS", "all")
    return s


@app.get("/jobs/{job_id}/answer")
def answer(job_id: str):
    p = JOBS / job_id / "ANSWER.md"
    if not p.exists():
        raise HTTPException(404, "no answer yet")
    return PlainTextResponse(p.read_text(errors="ignore"))


_IMG = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".svg": "image/svg+xml", ".webp": "image/webp"}


# Tree paths from /stats are relative to the job dir (root labeled 'output'), so resolve
# there. Block the internal plumbing files from being fetched even by direct URL.
_FILE_HIDE = {"input.zip", "pi.log", "corpus.log", "orchestrator.log", "meta.json",
              "run_meta.json", "control", "query.txt"}


@app.get("/jobs/{job_id}/file")
def get_file(job_id: str, path: str):
    base = (JOBS / job_id).resolve()
    target = (base / path).resolve()
    try:
        rel = target.relative_to(base)
    except ValueError:
        raise HTTPException(403, "outside job dir")
    parts = rel.parts
    if parts and (parts[0] in (".pi-agent", ".corpus_cache") or parts[-1] in _FILE_HIDE):
        raise HTTPException(404, "not found")
    if not target.is_file():
        raise HTTPException(404, "not found")
    ext = target.suffix.lower()
    if ext in _IMG:
        return FileResponse(str(target), media_type=_IMG[ext])
    try:
        return PlainTextResponse(target.read_text(errors="ignore")[:500000])
    except Exception:
        raise HTTPException(415, "not previewable")


@app.get("/jobs/{job_id}/dashboard", response_class=HTMLResponse)
def dashboard(job_id: str):
    return HTMLResponse((WEB / "dashboard.html").read_text().replace("__JOB_ID__", job_id))


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse((WEB / "index.html").read_text())


@app.get("/favicon.svg")
def favicon():
    f = WEB / "favicon.svg"
    if f.exists():
        return FileResponse(str(f), media_type="image/svg+xml")
    raise HTTPException(404, "no favicon")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
