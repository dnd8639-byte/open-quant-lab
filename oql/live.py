"""Live computation for the dashboard: run a function in the background and stream what it produces.

A job function has the signature  fn(params: dict, emit: callable, stop: threading.Event) -> None.
It calls emit(event) whenever it has something to show; the dashboard draws each event as it arrives, so charts
build up while the work runs (the same way the permutation lab does). Event shapes the dashboard understands:

    {"stat": "key", "label": "Hill tail index", "value": 3.1, "fmt": ".2f"}       a headline number
    {"chart": "id", "kind": "line" | "bar" | "scatter" | "heatmap", "title": ..., "series": [...], ...}
    {"table": "id", "title": ..., "columns": [...], "rows": [[...], ...]}
    {"check": "name", "passed": True, "detail": "recovers 3.0 within 0.2"}         a known-answer self-check
    {"log": "text"}                                                              a line of progress
    {"progress": 0.42}                                                           fraction done

The stream ends with {"done": True} or {"error": "..."}. Charts are sent whole each time (small, downsampled arrays),
so a client that connects late still ends up with the latest picture.
"""
from __future__ import annotations

import json
import threading
import time
import traceback
import uuid

MAX_JOBS = 30


class Stopped(Exception):
    """Raised inside a job when the person pressed Stop."""


class Job:
    def __init__(self, kind: str, params: dict, fn):
        self.id = uuid.uuid4().hex[:10]
        self.kind, self.params, self.fn = kind, params, fn
        self.events: list[dict] = []
        self.done = False
        self.started = time.time()
        self.stop = threading.Event()
        self._cond = threading.Condition()

    def emit(self, event: dict) -> None:
        if self.stop.is_set():
            raise Stopped()
        with self._cond:
            self.events.append(_clean(event))
            self._cond.notify_all()

    def run(self) -> None:
        try:
            self.fn(self.params, self.emit, self.stop)
            self._finish({"done": True, "seconds": round(time.time() - self.started, 2)})
        except Stopped:
            self._finish({"done": True, "stopped": True})
        except Exception as e:                               # shown on the page, never kills the server
            traceback.print_exc()
            self._finish({"error": f"{type(e).__name__}: {e}"})

    def _finish(self, event):
        with self._cond:
            self.events.append(event)
            self.done = True
            self._cond.notify_all()

    def wait(self, after: int, timeout: float = 15.0) -> list[dict]:
        """Events after index `after`; blocks until there are some, the job ends, or the timeout passes."""
        with self._cond:
            if len(self.events) <= after and not self.done:
                self._cond.wait(timeout)
            return self.events[after:]


_jobs: dict[str, Job] = {}
_lock = threading.Lock()


def start(kind: str, params: dict, fn) -> Job:
    job = Job(kind, params, fn)
    with _lock:
        _jobs[job.id] = job
        for old in sorted(_jobs.values(), key=lambda j: j.started)[:-MAX_JOBS]:
            if old.done:
                _jobs.pop(old.id, None)
    threading.Thread(target=job.run, daemon=True, name=f"job-{kind}").start()
    return job


def get(job_id: str) -> Job:
    with _lock:
        if job_id not in _jobs:
            raise KeyError(f"no job {job_id}")
        return _jobs[job_id]


def sse(write, job: Job) -> None:
    """Stream a job to a browser as Server-Sent Events until it ends (or the browser goes away)."""
    i = 0
    while True:
        new = job.wait(i)
        if not new:
            write(": keep-alive\n\n")                         # comment line keeps proxies and browsers patient
            continue
        for ev in new:
            write(f"data: {json.dumps(ev, default=_default)}\n\n")
        i += len(new)
        if job.done and i >= len(job.events):
            write("event: end\ndata: {}\n\n")
            return


def _default(x):
    try:
        import numpy as np
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            return float(x) if np.isfinite(x) else None
        if isinstance(x, np.ndarray):
            return _clean(x.tolist())
    except ImportError:
        pass
    return str(x)


def _clean(x):
    """JSON-safe: NaN and infinities become null, numpy values become plain numbers."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, float):
        return x if x == x and abs(x) != float("inf") else None
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return _clean(x.tolist())
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            return _clean(float(x))
        if isinstance(x, np.bool_):
            return bool(x)
    except ImportError:
        pass
    return x


def downsample(x, y=None, n: int = 600):
    """Evenly thin a series for drawing (keeps the first and last points)."""
    import numpy as np
    y_arr = np.asarray(x if y is None else y, float)
    idx = np.unique(np.linspace(0, len(y_arr) - 1, min(n, len(y_arr))).round().astype(int)) if len(y_arr) else np.array([], int)
    if y is None:
        return y_arr[idx].tolist()
    xs = list(x)
    return [xs[i] for i in idx], y_arr[idx].tolist()
