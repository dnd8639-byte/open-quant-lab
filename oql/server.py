"""`oql serve`: a local dashboard for the lab. Standard library only; nothing leaves your machine.

Pages: the research board (with live gate runs), a permutation lab that shuffles a market in front of you and
builds the null distribution one permutation at a time, the data catalog, and the ledger.

The permutation lab only ever uses the first 70% of each dataset (the development part), the same rule as the
gate, so exploring here never touches a holdout. Holdouts can only be run from the command line, on purpose.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

from . import board, catalog, ledger
from .engine import prepare
from .experiment import Experiment
from .permute import permute_book
from .search import Scorer, search, sharpe
from .strategies import RULES, valid

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
_dev_cache: dict = {}
_lock = threading.Lock()
_running: dict = {}                                      # card id -> subprocess


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def _downsample(values, n=400):
    v = np.asarray(values, float)
    if len(v) <= n:
        return v.tolist()
    idx = np.linspace(0, len(v) - 1, n).round().astype(int)
    return v[idx].tolist()


def _dev(ds: str, frac: float = 0.7):
    """Development part of one dataset (first 70% of its bars), cached."""
    with _lock:
        if ds not in _dev_cache:
            df = catalog.read(ds)
            _dev_cache[ds] = df.iloc[: int(len(df) * frac)]
        return _dev_cache[ds]


def explore(ds: str, mode: str, genome: dict, seed: int) -> dict:
    """One draw for the permutation lab. seed < 0 = the real data. mode 'single' scores one strategy;
    mode 'search' runs the whole default search and returns its best (to show selection bias)."""
    dev = _dev(ds)
    book = {ds: dev} if seed < 0 else permute_book({ds: dev}, seed)
    exp = Experiment(datasets=[ds], warmup=min(250, max(20, len(dev) // 10)))
    sc = Scorer(prepare(book, {"default": 1.0}, warmup=exp.warmup), exp.rule_table(), 0.0)
    if mode == "search":
        res = search(sc, exp)
        value, best = res["best_score"], res["best"]
    else:
        port, _, _ = sc.returns(genome, delay=exp.delay)
        value, best = sharpe(port.to_numpy(), sc.prep.ppy), genome
    c = book[ds]["close"].to_numpy()
    return {"sharpe": value, "best": best, "path": _downsample(c / c[0]), "bars": len(c),
            "start": str(dev.index[0])[:10], "end": str(dev.index[-1])[:10]}


def _card_payload(cid: str) -> dict:
    bd = board.load()
    c = board.get(bd, cid)
    d = board.card_dir(c)
    out = {"card": c, "running": cid in _running and _running[cid].poll() is None}
    for name in ("gate", "holdout"):
        p = os.path.join(d, f"{name}.json")
        if os.path.exists(p):
            with open(p) as f:
                out[name] = json.load(f)
    p = os.path.join(d, "trials.json")
    if os.path.exists(p):
        with open(p) as f:
            out["trials"] = json.load(f)
    p = os.path.join(d, "gate_progress.jsonl")
    if os.path.exists(p):
        with open(p) as f:
            out["progress"] = [json.loads(l) for l in f if l.strip()]
    for name in ("PREREGISTRATION.md", "gate.md"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            with open(p) as f:
                out[name] = f.read()
    return out


def _start_gate(cid: str) -> dict:
    bd = board.load()
    c = board.get(bd, cid)
    if c["stage"] != "preregistered":
        raise board.BoardError(f"the gate needs the card in 'preregistered'; it is in '{c['stage']}'")
    if cid in _running and _running[cid].poll() is None:
        raise board.BoardError("this card's gate is already running")
    d = board.card_dir(c)
    log = open(os.path.join(d, "gate_stdout.log"), "w")
    _running[c["id"]] = subprocess.Popen([sys.executable, "-m", "oql", "card", "gate", c["id"]], cwd=board.lab_dir(),
                                         stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
    return {"started": c["id"]}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):                           # keep the terminal quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(_jsonable(body)).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                with open(os.path.join(WEB, "index.html"), "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            if u.path == "/api/board":
                return self._send(200, {"board": board.load(), "ledger": ledger.rows(board.lab_dir()),
                                        "running": [k for k, p in _running.items() if p.poll() is None]})
            if u.path.startswith("/api/card/"):
                return self._send(200, _card_payload(unquote(u.path.split("/")[3])))
            if u.path == "/api/datasets":
                return self._send(200, [m.__dict__ for m in catalog.list_all()])
            if u.path == "/api/check":
                return self._send(200, catalog.check(q["ds"]))
            if u.path == "/api/bars":
                df = catalog.read(q["ds"])
                nums = df.select_dtypes("number").columns
                if "close" in df.columns:
                    col, v = "close", df["close"].to_numpy(float)
                elif len(nums):
                    col, v = nums[0], df[nums[0]].to_numpy(float)
                else:                                     # a table of events: show how many per month
                    counts = df.groupby(df.index.to_period("M")).size()
                    col, v = "rows per month", counts.to_numpy(float)
                return self._send(200, {"values": _downsample(v, 600), "start": str(df.index[0])[:10],
                                        "end": str(df.index[-1])[:10], "column": col, "rows": len(df)})
            if u.path == "/api/paper":
                from . import paper
                return self._send(200, {"status": paper.status(), "deployments": paper.deployments()})
            if u.path == "/api/collect":
                from . import collect
                return self._send(200, {"status": collect.status()})
            if u.path == "/api/rules":
                return self._send(200, {k: {"params": g} for k, (_, g) in RULES.items()})
            if u.path == "/api/explore":
                genome = {"rule": q.get("rule", "momentum"), "params": json.loads(q.get("params", "{}")),
                          "side": q.get("side", "both")}
                if genome["rule"] in RULES and not valid(genome["rule"], genome["params"]):
                    return self._send(400, {"error": "fast must be shorter than slow"})
                return self._send(200, explore(q["ds"], q.get("mode", "single"), genome, int(q.get("seed", -1))))
            return self._send(404, {"error": "not found"})
        except (board.BoardError, KeyError, ValueError) as e:
            return self._send(400, {"error": str(e)})

    def do_POST(self):
        u = urlparse(self.path)
        try:
            if u.path.startswith("/api/gate/"):
                return self._send(200, _start_gate(unquote(u.path.split("/")[3])))
            return self._send(404, {"error": "not found"})
        except (board.BoardError, KeyError, ValueError) as e:
            return self._send(400, {"error": str(e)})


def serve(port: int = 8765, open_browser: bool = True):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Open Quant Lab dashboard: {url}   (lab: {board.lab_dir()}; Control + C stops it)", flush=True)
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
