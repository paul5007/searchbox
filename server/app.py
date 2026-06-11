#!/usr/bin/env python3
"""Searchbox API + UI.

POST /jobs        (multipart: prompt, budget, dataroom=<zipfile>)  -> {job_id}
GET  /jobs                         -> list
GET  /jobs/{id}                    -> status
GET  /jobs/{id}/stats              -> live metrics (drives the dashboard)
GET  /jobs/{id}/answer             -> ANSWER.md (text)
GET  /jobs/{id}/file?path=...      -> a dataroom/output file
GET  /jobs/{id}/dashboard          -> the dashboard HTML
GET  /                             -> submit page
GET  /health
"""
import json, os, signal, subprocess, sys, threading, time, uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, PlainTextResponse, HTMLResponse
import uvicorn

from server.stats import job_stats, session_usage, BUDGET_METRIC

HERE = Path(__file__).resolve().parent
WEB = HERE.parent / "web"
JOBS = Path(os.environ.get("JOBS_DIR", str(HERE.parent / "data" / "jobs"))).resolve()
JOBS.mkdir(parents=True, exist_ok=True)
CATALOG_PATH = HERE.parent / "pi" / "tools-catalog.json"


def _catalog() -> list:
    """The external-tool catalog (single source of truth, pi/tools-catalog.json)."""
    try:
        return json.loads(CATALOG_PATH.read_text()).get("tools", [])
    except Exception:
        return []


def _catalog_names() -> set:
    return {t.get("name") for t in _catalog() if t.get("name")}

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
           "--query", meta["query"], "--dataroom", str(job_dir / "input.zip"),
           "--out", str(job_dir), "--budget", str(meta["budget"])]
    # force_budget defaults ON; only add the off flag when explicitly disabled.
    if meta.get("force_budget") is False:
        cmd.append("--no-force-budget")
    env = dict(os.environ)
    for k in ("LLAMA_URL", "MODEL_ID", "CONTEXT_WINDOW",
              "EMBED_MODEL", "RERANK_MODEL", "BUDGET_METRIC",
              "EMBED_BACKEND", "RERANK_BACKEND", "API_EMBED_MODEL", "API_RERANK_MODEL"):
        if meta.get(k):
            env[k] = str(meta[k])
    # SEARCHBOX_TOOLS: pass through even when "" (explicit "no external tools"). Only None/missing
    # means "unset" -> extension falls back to its DEFAULT_TOOLS.
    if meta.get("SEARCHBOX_TOOLS") is not None:
        env["SEARCHBOX_TOOLS"] = str(meta["SEARCHBOX_TOOLS"])
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


def _reconcile_on_startup():
    """A fresh app process owns no running jobs. Any job left marked running/queued/pausing by a
    previous (killed) process is stale: if it finished it has run_meta (trust that), otherwise it
    was interrupted. Prevents zombie 'running' rows after an app restart/crash."""
    if not JOBS.exists():
        return
    for p in JOBS.iterdir():
        if not p.is_dir():
            continue
        meta = _load_meta(p.name)
        if meta.get("status") not in ("running", "queued", "pausing"):
            continue
        rm = p / "run_meta.json"
        if rm.exists():
            try:
                m = json.loads(rm.read_text())
                meta["status"] = "done" if m.get("done") else "stopped"
                meta["stop_reason"] = m.get("stop_reason")
            except Exception:
                meta["status"] = "interrupted"
        else:
            meta["status"] = "interrupted"
        meta["finished"] = meta.get("finished") or time.time()
        _jobs[p.name] = meta
        try:
            (p / "meta.json").write_text(json.dumps(meta))
        except Exception:
            pass


_reconcile_on_startup()
threading.Thread(target=_worker, daemon=True).start()


@app.get("/health")
def health():
    return {"ok": True}


# Default dataroom used when the frontend submits no zip. It is treated exactly like an uploaded
# zip (copied to input.zip, NOT pre-extracted) so the default path is end-to-end identical.
DEFAULT_DATAROOM = Path(os.environ.get(
    "DEFAULT_DATAROOM", str(HERE.parent / "data" / "default-dataroom.zip")))


@app.post("/jobs")
async def create(prompt: str = Form(...), budget: int = Form(...),
                 force_budget: bool = Form(True),
                 tools: str = Form(None),
                 dataroom: UploadFile | None = File(None)):
    prompt = (prompt or "").strip()
    if len(prompt) < 5:
        raise HTTPException(400, "prompt too short")
    if budget < 1:
        raise HTTPException(400, "budget must be >= 1")
    # `tools` is a comma-separated list of external tools to register for this job (subset of
    # sentence_embed, passage_rerank, semantic_search). Empty string => no external tools (pi
    # built-ins only). None/missing => leave unset (extension uses its DEFAULT_TOOLS).
    tools_sel = None
    if tools is not None:
        allowed = _catalog_names()
        picked = [t.strip() for t in tools.split(",") if t.strip() in allowed]
        tools_sel = ",".join(picked)  # may be "" meaning none
    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    if dataroom is not None and dataroom.filename:
        data = await dataroom.read()
        dataroom_name = dataroom.filename
    else:
        if not DEFAULT_DATAROOM.exists():
            raise HTTPException(400, "no dataroom uploaded and no default dataroom configured")
        data = DEFAULT_DATAROOM.read_bytes()
        dataroom_name = DEFAULT_DATAROOM.name
    (job_dir / "input.zip").write_bytes(data)
    with _cond:
        _jobs[job_id] = {"status": "queued", "query": prompt, "budget": budget,
                         "force_budget": bool(force_budget),
                         "SEARCHBOX_TOOLS": tools_sel,
                         "dataroom_name": dataroom_name, "dataroom_bytes": len(data),
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
            # finished job: authoritative final numbers from run_meta
            try:
                m = json.loads(rm.read_text())
                spent = m.get("input_tokens_spent", 0)
                bdg = bdg or m.get("budget")
                pct = m.get("budget_pct") or 0
            except Exception:
                pass
        else:
            # running/queued: read live spend from the append-only session file so the homepage
            # progress bar tracks in real time (run_meta only exists once the job ends).
            try:
                spent = session_usage(job_dir).get(BUDGET_METRIC, 0)
                if bdg:
                    pct = round(min(100, 100 * spent / bdg), 1)
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
    s = job_stats(job_dir, meta.get("budget"), live=(meta.get("status") == "running"))
    s["job_id"] = job_id
    s["job_status"] = meta.get("status") or "unknown"
    s["query"] = meta.get("query", "")
    s["model_id"] = meta.get("MODEL_ID") or os.environ.get("MODEL_ID", "qwen3.6")
    s["tools_enabled"] = meta.get("SEARCHBOX_TOOLS") or os.environ.get("SEARCHBOX_TOOLS", "all")
    return s


@app.get("/jobs/{job_id}/answer")
def answer(job_id: str):
    p = JOBS / job_id / "work" / "ANSWER.md"
    if not p.exists():
        raise HTTPException(404, "no answer yet")
    return PlainTextResponse(p.read_text(errors="ignore"))


@app.get("/jobs/{job_id}/answer/download")
def answer_download(job_id: str):
    p = JOBS / job_id / "work" / "ANSWER.md"
    if not p.exists():
        raise HTTPException(404, "no answer yet")
    return FileResponse(str(p), media_type="text/markdown", filename=f"ANSWER-{job_id}.md")


def _session_file(job_id: str) -> Path | None:
    sd = JOBS / job_id / ".pi-agent" / "sessions"
    if not sd.exists():
        return None
    files = sorted(sd.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


@app.get("/jobs/{job_id}/snapshots.zip")
def snapshots_zip(job_id: str):
    """Per-turn experiment data: every ANSWER-t{n}.md snapshot + turns.jsonl (token usage per
    turn). Built on the fly into a zip. These live in job_dir/snapshots (outside work/), so the
    pi sandbox never sees them."""
    import io, zipfile
    snap_dir = JOBS / job_id / "snapshots"
    if not snap_dir.exists() or not any(snap_dir.iterdir()):
        raise HTTPException(404, "no snapshots yet")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(snap_dir.iterdir()):
            if f.is_file():
                z.write(str(f), arcname=f"snapshots-{job_id}/{f.name}")
    buf.seek(0)
    from fastapi.responses import Response
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="snapshots-{job_id}.zip"'},
    )


@app.get("/jobs/{job_id}/trace.jsonl")
def trace_jsonl(job_id: str):
    """The complete native Pi trace: the session JSONL (every message, tool call, and thinking
    block). This is Pi's own append-only record - no custom serialization."""
    sf = _session_file(job_id)
    if not sf:
        raise HTTPException(404, "no trace yet")
    return FileResponse(str(sf), media_type="application/x-ndjson", filename=f"trace-{job_id}.jsonl")


@app.get("/jobs/{job_id}/trace.html")
def trace_html(job_id: str):
    """Rendered trace incl. thinking, via Pi's NATIVE `pi --export` (no custom renderer)."""
    sf = _session_file(job_id)
    if not sf:
        raise HTTPException(404, "no trace yet")
    out = JOBS / job_id / "trace.html"
    pi_bin = os.environ.get("PI_BIN", "pi")
    env = dict(os.environ); env["PI_SKIP_VERSION_CHECK"] = "1"
    try:
        subprocess.run([pi_bin, "--export", str(sf), str(out)],
                       env=env, capture_output=True, timeout=120, check=True)
    except Exception as e:
        raise HTTPException(500, f"pi --export failed: {e}")
    # No filename= -> Content-Disposition: inline, so the browser opens it in a new tab
    # instead of downloading it. no-store so each click shows the freshly re-exported trace
    # (we regenerate it above on every request), never a stale browser-cached copy.
    return FileResponse(str(out), media_type="text/html",
                        headers={"Cache-Control": "no-store, must-revalidate"})


_IMG = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".svg": "image/svg+xml", ".webp": "image/webp"}


# Tree paths from /stats are relative to the job dir (root labeled 'output'), so resolve
@app.get("/jobs/{job_id}/file")
def get_file(job_id: str, path: str):
    # Files are served from work/ only (dataroom + model outputs). Plumbing lives in the parent
    # job_dir, physically outside this base, so it cannot be fetched.
    base = (JOBS / job_id / "work").resolve()
    target = (base / path).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        raise HTTPException(403, "outside work dir")
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


@app.get("/tools")
def tools_catalog():
    """External tool catalog for the homepage toggle list: name, one-line desc, default flag."""
    out = []
    for t in _catalog():
        desc = str(t.get("desc", ""))
        short = desc.split(". INPUT")[0].split(". ")[0][:140]
        out.append({"name": t.get("name"), "op": t.get("op"),
                    "default": bool(t.get("default")), "desc": short})
    return {"tools": out}


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
