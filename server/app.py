#!/usr/bin/env python3
"""Searchbox API + UI.

POST /jobs        (multipart: prompt, budget=<turns>, dataroom=<zipfile>)  -> {job_id}
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

from server.stats import job_stats, session_usage, parse_pi_log, BUDGET_METRIC

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
# Single-slot serial queue (one llama slot). _queue is FIFO of freshly-submitted/resumed job_ids
# waiting; _current holds the live orchestrator Popen so a new submission can preempt it.
#
# Job states:
#   queued   - foreground, waiting for the slot (fresh submit or explicit resume)
#   running  - on the slot now
#   pausing  - a cooperative stop is in flight (-> 'held' for a user pause, 'paused' for a preempt)
#   held     - USER-paused; sticky, NOT auto-backfilled. Only an explicit resume revives it.
#   paused   - preemptible idle pool; auto-backfill may resume it. A preempted FOREGROUND job
#              carries preempted=True so it outranks bulk paused jobs when backfilling.
#   done / stopped / failed - terminal
_queue: list[str] = []
_cond = threading.Condition(_lock)
_current = {"job_id": None, "proc": None}
# Auto-backfill: when no fresh job is queued, keep the slot busy by resuming the oldest paused
# job. A backfill run is preemptible - a new submission pauses it and takes the slot.
AUTO_BACKFILL = os.environ.get("AUTO_BACKFILL", "1") != "0"
# Foreground preemption: a fresh submit / explicit resume pauses ANY running job (incl. another
# foreground job) and takes the slot. Set to 0 to make a new job wait for the running one.
PREEMPT_FOREGROUND = os.environ.get("PREEMPT_FOREGROUND", "1") != "0"


def _save_meta(job_id: str):
    try:
        (JOBS / job_id / "meta.json").write_text(json.dumps(_jobs.get(job_id, {})))
    except Exception:
        pass


def _load_meta(job_id: str) -> dict:
    """Recover job meta from disk. A control flag of pause is authoritative; a run left 'running'
    by a killed app, or any interrupted/cancelled state, collapses to the single resumable
    'paused' state (so it joins the backfill pool / can be resumed)."""
    job_dir = JOBS / job_id
    mp = job_dir / "meta.json"
    meta = {}
    if mp.exists():
        try:
            meta = json.loads(mp.read_text())
        except Exception:
            meta = {}
    ctl = ""
    try:
        ctl = (job_dir / "control").read_text(errors="ignore").strip()
    except Exception:
        ctl = ""
    rm = job_dir / "run_meta.json"
    if rm.exists() and ctl not in ("pause", "cancel", "hold"):
        try:
            m = json.loads(rm.read_text())
            # A finished run: done or stopped (terminal). Not resumable.
            if meta.get("status") not in ("paused", "pausing", "held"):
                meta["status"] = "done" if m.get("done") else "stopped"
                meta["stop_reason"] = m.get("stop_reason")
        except Exception:
            pass
    # 'hold' = sticky user pause; 'pause'/'cancel' = preemptible idle.
    if ctl == "hold":
        meta["status"] = "held"
    elif ctl in ("pause", "cancel"):
        meta["status"] = "paused"
    elif meta.get("status") in ("running", "pausing", "interrupted", "cancelled"):
        # mid-flight when the app died, or legacy terminal-stop -> resumable paused
        meta["status"] = "paused"
    return meta


def _run_one(job_id: str, resume: bool = False):
    job_dir = JOBS / job_id
    # Commit to running under the lock. If a pause raced in during the dequeue window, honor it.
    with _lock:
        meta = dict(_jobs.get(job_id, {}))
        if meta.get("status") in ("paused", "held"):
            _save_meta(job_id)
            return
        _jobs[job_id]["status"] = "running"
        _jobs[job_id]["started"] = time.time()
        _jobs[job_id]["finished"] = None
        # Register as current UNDER THE SAME LOCK as the status flip, so a preempt landing before
        # the subprocess launches is not lost (it sees _current, writes the cooperative control
        # flag; run_searchbox honors it at the next cycle boundary).
        _current["job_id"], _current["proc"] = job_id, None
        try:
            (job_dir / "control").unlink()
        except FileNotFoundError:
            pass
    _save_meta(job_id)
    cmd = [sys.executable, "-m", "server.run_searchbox",
           "--query", meta["query"], "--dataroom", str(job_dir / "input.zip"),
           "--out", str(job_dir), "--budget", str(meta["budget"])]
    # force_budget defaults ON; only add the off flag when explicitly disabled.
    if meta.get("force_budget") is False:
        cmd.append("--no-force-budget")
    # Resume (continue prior pi session + on-disk dataroom) whenever a previous run exists for this
    # job - i.e. it was paused/preempted before. True for backfill AND for an explicit foreground
    # resume. Detected by an existing pi session dir, not just the backfill flag.
    sess_dir = job_dir / ".pi-agent" / "sessions"
    has_prior = resume or (sess_dir.exists() and any(sess_dir.rglob("*.jsonl")))
    if has_prior:
        cmd.append("--resume")
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
        # A preempt may have already flipped status to 'pausing' (and written control) in the launch
        # window; keep that, just attach the proc so the SIGTERM path can reach it.
        _current["job_id"], _current["proc"] = job_id, proc
    rc = proc.wait()
    # The control flag (if we wrote one) is authoritative over zip/meta for paused state.
    ctl = ""
    try:
        ctl = (job_dir / "control").read_text(errors="ignore").strip()
    except Exception:
        pass
    with _lock:
        _current["job_id"], _current["proc"] = None, None
        if ctl == "hold":
            _jobs[job_id]["status"], _jobs[job_id]["stop_reason"] = "held", "held"
        elif ctl in ("pause", "cancel"):
            _jobs[job_id]["status"], _jobs[job_id]["stop_reason"] = "paused", "paused"
        else:
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
        _jobs[job_id]["auto"] = False
    _save_meta(job_id)


def _next_foreground_locked():
    """Oldest 'queued' (freshly submitted or explicitly resumed) job, FIFO, or None. Holds _cond."""
    for j in _queue:
        if _jobs.get(j, {}).get("status") == "queued":
            return j
    return None


def _paused_on_disk() -> list:
    """The backfill pool in resume-priority order. USER-held jobs are excluded (sticky; explicit
    resume only). A preempted FOREGROUND job (preempted=True) outranks bulk paused jobs so it is not
    starved at the back of a large backlog; then oldest-active first within each tier."""
    out = []
    if JOBS.exists():
        for p in sorted(JOBS.iterdir()):
            if not p.is_dir():
                continue
            m = _load_meta(p.name)
            if m.get("status") == "paused":
                tier = 0 if m.get("preempted") else 1
                out.append((p.name, tier, m.get("finished") or 0))
    out.sort(key=lambda t: (t[1], t[2]))
    return [jid for jid, _, _ in out]


def _select_next():
    """Pick next job: (job_id, is_backfill). Tier 1 (foreground 'queued') wins. Tier 2 (backfill)
    only when AUTO_BACKFILL and no foreground work: oldest paused job, resumed from disk."""
    with _cond:
        fg = _next_foreground_locked()
        if fg is not None:
            _queue.remove(fg)
            return fg, False
    if not AUTO_BACKFILL:
        return None, False
    for cand in _paused_on_disk():
        with _cond:
            if _next_foreground_locked() is not None:
                return None, False
            if _jobs.get(cand, {}).get("status") in ("queued", "running", "pausing"):
                continue
            m = _load_meta(cand)
            if m.get("status") != "paused":
                continue
            # getting the slot now -> clear the preempted marker (no longer a waiting victim).
            m.update({"status": "queued", "auto": True, "stop_reason": None,
                      "preempted": False, "started": None, "finished": None})
            _jobs[cand] = m
            _save_meta(cand)
            return cand, True
    return None, False


def _preempt_running_locked():
    """Flag the currently running job to pause so the slot frees for fresh foreground work; it
    returns to the paused pool and can resume later. Caller holds _cond; returns proc to SIGTERM
    OUTSIDE the lock.

    Always preempts a backfill (auto) run. Preempts a FOREGROUND run only when PREEMPT_FOREGROUND.
    A preempted foreground job is tagged preempted=True so it outranks bulk backfill on the way back."""
    jid = _current["job_id"]
    if not jid:
        return None
    cur = _jobs.get(jid, {})
    st = cur.get("status")
    if st == "pausing":
        return None                              # a preempt is already in flight for this job
    if st != "running":
        return None
    is_auto = bool(cur.get("auto"))
    if not is_auto and not PREEMPT_FOREGROUND:
        return None                              # leave the running foreground job alone
    try:
        (JOBS / jid / "control").write_text("pause")
    except Exception:
        return None
    _jobs[jid]["status"] = "pausing"
    if not is_auto:
        _jobs[jid]["preempted"] = True
    _save_meta(jid)
    return _current["proc"]                       # may be None if the subprocess is mid-launch; the
                                                 # cooperative control flag still stops it cleanly


def _worker():
    while True:
        job_id, backfill = _select_next()
        if job_id is None:
            with _cond:
                if _next_foreground_locked() is None:
                    _cond.wait()
            continue
        try:
            _run_one(job_id, resume=backfill)
        except Exception as e:
            with _lock:
                _jobs.setdefault(job_id, {})["status"] = "failed"
                _jobs[job_id]["error"] = str(e)[:300]
            _save_meta(job_id)
        with _cond:
            _cond.notify_all()


def _reconcile_on_startup():
    """A fresh app process owns no running jobs. Load all jobs from disk; _load_meta collapses any
    mid-flight (running/pausing/interrupted) job into the resumable 'paused' state. Those join the
    backfill pool and resume automatically when the slot idles - nothing is lost on restart."""
    if not JOBS.exists():
        return
    for p in sorted(JOBS.iterdir()):
        if not p.is_dir():
            continue
        meta = _load_meta(p.name)
        if meta:
            _jobs[p.name] = meta
            try:
                (p / "meta.json").write_text(json.dumps(meta))
            except Exception:
                pass
    with _cond:
        _cond.notify_all()


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
        raise HTTPException(400, "budget (turns) must be >= 1")
    # `tools` is a comma-separated list of external tools to register for this job (subset of
    # the catalog: high-level search_dataroom/answer_question + low-level primitives). Empty
    # string => no external tools (pi built-ins only). None/missing => extension DEFAULT_TOOLS.
    tools_sel = None
    if tools is not None:
        allowed = _catalog_names()
        picked = [t.strip() for t in tools.split(",") if t.strip() in allowed]
        tools_sel = ",".join(picked)  # may be "" meaning none
    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    if dataroom is not None and dataroom.filename:
        if not dataroom.filename.lower().endswith(".zip"):
            raise HTTPException(400, "dataroom must be a .zip file")
        data = await dataroom.read()
        if data[:2] != b"PK":
            raise HTTPException(400, "dataroom is not a valid zip")
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
        proc = _preempt_running_locked()   # a fresh submission takes the slot from any running job
        _cond.notify_all()
    _save_meta(job_id)
    # SIGTERM the preempted orchestrator OUTSIDE the lock (a long agent cycle would else block).
    if proc is not None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
    return {"job_id": job_id, "status": "queued"}


@app.post("/jobs/{job_id}/pause")
def pause(job_id: str):
    """USER pause an unfinished job. A user pause is STICKY: the job goes to 'held' (control=hold)
    and is NOT auto-backfilled - only an explicit resume revives it. (System preemption uses a
    separate, preemptible 'paused' state with control=pause.) SIGTERM if it is the running one."""
    job_dir = JOBS / job_id
    if not job_dir.exists():
        raise HTTPException(404, "no such job")
    with _cond:
        meta = dict(_jobs.get(job_id, {})) or _load_meta(job_id)
        if meta.get("status") in ("done", "stopped"):
            return {"job_id": job_id, "status": meta.get("status")}
        try:
            (job_dir / "control").write_text("hold")
        except Exception:
            pass
        proc = _current["proc"] if _current["job_id"] == job_id else None
        running = _current["job_id"] == job_id
        m = _jobs.setdefault(job_id, meta)
        m["status"] = "pausing" if running else "held"
        m["preempted"] = False                   # a user hold is not a preempt; drop priority tag
        if job_id in _queue:
            _queue.remove(job_id)
        _save_meta(job_id)
        _cond.notify_all()
    if proc is not None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
    return {"job_id": job_id, "status": "pausing" if running else "held"}


@app.post("/jobs/{job_id}/resume")
def resume(job_id: str):
    """Re-enqueue a held/paused job as foreground work (preempts any running job). The on-disk
    pi session + dataroom are continued (--resume)."""
    job_dir = JOBS / job_id
    if not job_dir.exists():
        raise HTTPException(404, "no such job")
    with _cond:
        meta = _load_meta(job_id)
        if meta.get("status") in ("done", "stopped"):
            return {"job_id": job_id, "status": meta.get("status")}
        # Drop any stale control flag (hold/pause) so _run_one does not see it and bail.
        try:
            (job_dir / "control").unlink()
        except FileNotFoundError:
            pass
        meta.update({"status": "queued", "auto": False, "preempted": False, "stop_reason": None})
        _jobs[job_id] = meta
        if job_id not in _queue:
            _queue.append(job_id)
        proc = _preempt_running_locked()
        _save_meta(job_id)
        _cond.notify_all()
    if proc is not None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
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
        # budget + progress are TURN-based now. spent = turns done, target = turn budget.
        spent = pct = 0
        bdg = meta.get("budget")
        if rm.exists():
            # finished job: authoritative final numbers from run_meta
            try:
                m = json.loads(rm.read_text())
                spent = m.get("turns", 0)
                bdg = bdg or m.get("budget")
                pct = m.get("budget_pct") or 0
            except Exception:
                pass
        else:
            # running/queued: count turns live from pi.log so the homepage bar tracks in real time
            # (run_meta only exists once the job ends).
            try:
                spent = parse_pi_log(job_dir / "pi.log").get("turns", 0)
                if bdg:
                    pct = round(min(100, 100 * spent / bdg), 1)
            except Exception:
                pass
        rows.append({"job_id": jid, "status": meta.get("status", "unknown"),
                     "stop_reason": meta.get("stop_reason"),
                     "auto": bool(meta.get("auto")), "preempted": bool(meta.get("preempted")),
                     "query": (meta.get("query") or "")[:200],
                     "budget": bdg, "spent": spent, "percent": pct, "unit": "turns",
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
                    "group": t.get("group", "low"),
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


@app.get("/og-image.jpg")
def og_image():
    f = WEB / "og-image.jpg"
    if f.exists():
        return FileResponse(str(f), media_type="image/jpeg")
    raise HTTPException(404, "no og image")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
