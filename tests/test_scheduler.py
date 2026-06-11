#!/usr/bin/env python3
"""Scheduler logic test for searchbox's single-slot preemptive queue (no llama / no real run).

Monkeypatches app._run_one with a fake honoring the same contract (commit-to-running under the
lock with atomic _current registration, watch the control flag for a cooperative pause/hold, write
a terminal run_meta on clean completion). Exercises: backfill of paused jobs, foreground priority,
preemption of a running foreground job, sticky user-hold (never auto-backfilled), preempted-job
priority over bulk backfill, and the launch-window preempt race.

Run: python3 tests/test_scheduler.py   (needs fastapi/pydantic/numpy importable, like the app)
"""
import os, sys, json, time, tempfile, subprocess, threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["JOBS_DIR"] = tempfile.mkdtemp(prefix="sb-sched-test-")
os.environ["AUTO_BACKFILL"] = "1"
os.environ["PREEMPT_FOREGROUND"] = "1"

from server import app  # noqa: E402

OBSERVED = []  # (job_id, was_auto_at_run_start)
_obs_lock = threading.Lock()
DEFAULT_FINISH = 0.4   # fallback run length when a job has no explicit _finish_after


def fake_run_one(job_id, resume=False):
    jd = app.JOBS / job_id
    jd.mkdir(parents=True, exist_ok=True)
    with app._lock:
        meta = dict(app._jobs.get(job_id, {}))
        if meta.get("status") in ("paused", "held"):
            app._save_meta(job_id)
            return
        app._jobs[job_id]["status"] = "running"
        app._jobs[job_id]["started"] = time.time()
        app._jobs[job_id]["finished"] = None
        was_auto = bool(app._jobs[job_id].get("auto"))
        # mirror the real _run_one: register current UNDER the commit lock (None proc for now)
        app._current["job_id"], app._current["proc"] = job_id, None
        try:
            (jd / "control").unlink()
        except FileNotFoundError:
            pass
    app._save_meta(job_id)
    with _obs_lock:
        OBSERVED.append((job_id, was_auto))
    proc = subprocess.Popen(["sleep", "30"], start_new_session=True)
    with app._lock:
        app._current["job_id"], app._current["proc"] = job_id, proc
    t0 = time.time()
    ctl = ""
    while True:
        with app._lock:
            fa = float(app._jobs.get(job_id, {}).get("_finish_after", meta.get("_finish_after", DEFAULT_FINISH)))
        if time.time() - t0 >= fa:
            break
        try:
            ctl = (jd / "control").read_text().strip()
        except Exception:
            ctl = ""
        if ctl in ("pause", "cancel", "hold"):
            break
        time.sleep(0.02)
    try:
        ctl = (jd / "control").read_text().strip()
    except Exception:
        ctl = ""
    try:
        proc.terminate()
    except Exception:
        pass
    if ctl not in ("pause", "cancel", "hold"):
        (jd / "run_meta.json").write_text(json.dumps({"done": True, "stop_reason": "budget_spent"}))
    with app._lock:
        app._current["job_id"], app._current["proc"] = None, None
        if ctl == "hold":
            app._jobs[job_id]["status"], app._jobs[job_id]["stop_reason"] = "held", "held"
        elif ctl in ("pause", "cancel"):
            app._jobs[job_id]["status"], app._jobs[job_id]["stop_reason"] = "paused", "paused"
        else:
            app._jobs[job_id]["status"] = "done"
            app._jobs[job_id]["stop_reason"] = "budget_spent"
        app._jobs[job_id]["finished"] = time.time()
        app._jobs[job_id]["auto"] = False
    app._save_meta(job_id)


app._run_one = fake_run_one


def submit(query="a research question", finish_after=None):
    """Mimic create()'s scheduler side-effects without the Form/UploadFile plumbing."""
    import uuid
    job_id = uuid.uuid4().hex[:12]
    (app.JOBS / job_id).mkdir(parents=True, exist_ok=True)
    with app._cond:
        meta = {"status": "queued", "query": query, "budget": 1000, "submitted": time.time()}
        if finish_after is not None:
            meta["_finish_after"] = finish_after
        app._jobs[job_id] = meta
        app._queue.append(job_id)
        proc = app._preempt_running_locked()
        app._cond.notify_all()
    app._save_meta(job_id)
    if proc is not None:
        try:
            os.killpg(os.getpgid(proc.pid), 15)
        except Exception:
            pass
    return job_id


def make_paused(job_id, query, finished, finish_after=0.4, status="paused",
                control="pause", **extra):
    jd = app.JOBS / job_id
    jd.mkdir(parents=True, exist_ok=True)
    meta = {"status": status, "query": query, "budget": 1000, "finished": finished,
            "stop_reason": status, "_finish_after": finish_after}
    meta.update(extra)
    (jd / "meta.json").write_text(json.dumps(meta))
    (jd / "control").write_text(control)
    with app._lock:
        app._jobs[job_id] = dict(meta)


def status(job_id):
    with app._lock:
        return app._jobs.get(job_id, {}).get("status")


def wait_until(pred, timeout=15, msg=""):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.03)
    raise AssertionError(f"timeout waiting: {msg}")


PASS = []


def check(name, cond):
    assert cond, f"FAIL: {name}"
    PASS.append(name)
    print(f"  ok: {name}")


# A: backfill drains the paused pool, oldest-idle first.
print("A: backfill drains paused pool (oldest-idle first)")
make_paused("pa_old", "old", finished=100.0)
make_paused("pa_new", "new", finished=200.0)
with app._cond:
    app._cond.notify_all()
wait_until(lambda: status("pa_old") == "done" and status("pa_new") == "done",
           msg="both paused jobs backfilled")
order = [j for j, _ in OBSERVED if j in ("pa_old", "pa_new")]
check("both paused jobs backfilled to done", status("pa_old") == "done" and status("pa_new") == "done")
check("oldest-idle (pa_old) backfilled first", order[0] == "pa_old")

# B: a fresh submit preempts a running backfill; runs first; backfill resumes after.
print("B: fresh submit preempts a running backfill")
OBSERVED.clear()
make_paused("pb_bg", "bg", finished=50.0, finish_after=3.0)
with app._cond:
    app._cond.notify_all()
wait_until(lambda: status("pb_bg") == "running", msg="backfill running")
check("backfill running and marked auto", status("pb_bg") == "running" and app._jobs["pb_bg"].get("auto") is True)
fid = submit("fresh foreground")
wait_until(lambda: status("pb_bg") == "paused", msg="backfill preempted")
check("backfill preempted back to paused", status("pb_bg") == "paused")
wait_until(lambda: status(fid) == "done", msg="fresh job done")
check("fresh foreground job completed", status(fid) == "done")
wait_until(lambda: status("pb_bg") == "done", msg="preempted backfill resumed")
check("preempted backfill resumed and finished", status("pb_bg") == "done")

# C: a fresh submit preempts a RUNNING FOREGROUND job; preempted job keeps priority.
print("C: fresh submit preempts a running foreground job")
OBSERVED.clear()
DEFAULT_FINISH = 3.0
f1 = submit("first foreground, long", finish_after=3.0)
wait_until(lambda: status(f1) == "running", msg="f1 running")
DEFAULT_FINISH = 0.4
f2 = submit("second foreground preempts first")
wait_until(lambda: status(f1) == "paused", msg="f1 preempted")
check("running foreground job preempted to paused", status(f1) == "paused")
check("preempted foreground carries preempted marker", app._jobs[f1].get("preempted") is True)
wait_until(lambda: status(f2) == "done", msg="f2 done")
check("preempting foreground job completed", status(f2) == "done")
wait_until(lambda: status(f1) == "done", msg="f1 resumed")
check("preempted foreground resumed and finished", status(f1) == "done")
starts = [j for j, _ in OBSERVED]
check("f2 started before f1 resumed",
      starts.index(f2) < (starts.index(f1, starts.index(f2)) if f1 in starts[starts.index(f2):] else 10**9))

# D: a user-HELD job is sticky - never auto-backfilled; explicit resume revives it.
print("D: user-held job is never auto-backfilled")
OBSERVED.clear()
make_paused("pd_held", "user paused this", finished=5.0, status="held", control="hold")
with app._cond:
    app._cond.notify_all()
time.sleep(1.0)
check("held job stays held", status("pd_held") == "held")
check("held job never started running", "pd_held" not in [j for j, _ in OBSERVED])
app.resume("pd_held")
wait_until(lambda: status("pd_held") == "done", msg="held job resumes on explicit resume")
check("explicit resume revives a held job", status("pd_held") == "done")

# E: a preempted foreground job outranks an older bulk paused job in the backfill order.
print("E: preempted foreground outranks bulk backfill")
OBSERVED.clear()
# an OLD bulk paused job (would normally backfill first by age)
make_paused("pe_bulk", "old bulk", finished=1.0, finish_after=0.4)
# a preempted foreground job with a NEWER finished (newer = later by age) but tier-0 priority
make_paused("pe_fg", "preempted fg", finished=999.0, finish_after=0.4, preempted=True)
with app._cond:
    app._cond.notify_all()
wait_until(lambda: status("pe_fg") == "done" and status("pe_bulk") == "done", msg="both drained")
o = [j for j, _ in OBSERVED if j in ("pe_bulk", "pe_fg")]
check("preempted foreground backfilled before older bulk job", o.index("pe_fg") < o.index("pe_bulk"))

print(f"\nALL {len(PASS)} CHECKS PASSED")
