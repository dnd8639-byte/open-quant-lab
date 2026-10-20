"""Does the gate's p-value mean what it says? A calibration study of the lab's own test.

A test at alpha = 0.05 should PASS about 5% of searches on markets with no edge at all (its false-positive rate)
and should PASS most searches on markets with a real edge (its power). This module measures both, for each null
model, by running the full gate many times on synthetic books whose truth is known:

  noise    random walks with volatility clustering and fat tails (no edge): pass rate should be about alpha,
           and the p-values roughly uniform between 0 and 1
  wild     the same with much stronger volatility clustering: the case where a null that scrambles volatility
           (the bar shuffle) could be miscalibrated
  trend    persistent regimes (a real edge): pass rate = power

    python -m oql.calibration --books 60 --perm 39          # prints a table; writes calibration.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd

from .engine import prepare
from .experiment import Experiment
from .permute import permute_book
from .search import Scorer, search
from .synthetic import _bars, trend_market

KINDS = ("noise", "wild", "trend")
NULLS = ("bars", "vol_matched")


def clustered_market(n=2000, seed=0, vol=0.01, persistence=0.97, shock=0.1) -> pd.DataFrame:
    """Random walk whose log-volatility follows an AR(1): no edge, adjustable volatility clustering."""
    rng = np.random.default_rng(seed)
    h = np.zeros(n)
    for t in range(1, n):
        h[t] = persistence * h[t - 1] + rng.normal(0, shock)
    lr = rng.standard_t(5, n) * vol * np.exp(h) / np.sqrt(5 / 3)
    return _bars(lr, rng, vol=vol)


def book(kind: str, seed: int, n_markets=3, n=2000) -> dict:
    out = {}
    for i in range(n_markets):
        s = seed * 100 + i
        if kind == "noise":
            out[f"m{i}"] = clustered_market(n, s)
        elif kind == "wild":
            out[f"m{i}"] = clustered_market(n, s, persistence=0.99, shock=0.15)
        else:
            out[f"m{i}"] = trend_market(n, s, drift=0.002)
    return out


def experiment() -> Experiment:
    """A smaller search than the default (4 rounds), so hundreds of gates fit in minutes; same structure."""
    return Experiment(datasets=[], rounds=4, pop_init=16, children=10, fresh=4, elite=4, warmup=250)


def best(b: dict, exp: Experiment) -> float:
    sc = Scorer(prepare(b, {"default": 1.0}, warmup=exp.warmup), exp.rule_table(), exp.min_trades_per_year)
    return search(sc, exp)["best_score"]


def _one(args):
    kind, seed, n_perm = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    exp = experiment()
    b = book(kind, seed)
    split = sorted(set().union(*[set(df.index) for df in b.values()]))
    cut = split[int(len(split) * 0.7)]
    dev = {k: df[df.index < cut] for k, df in b.items()}
    real = best(dev, exp)
    out = {"kind": kind, "seed": seed, "real": real}
    for method in NULLS:
        null = np.array([best(permute_book(dev, 10_000 + k, method=method), exp) for k in range(n_perm)])
        out[method] = float((1 + (null >= real).sum()) / (n_perm + 1))
        out[method + "_median"] = float(np.median(null))
    return out


def run(books=60, n_perm=39, procs=None, kinds=KINDS, progress=print) -> list[dict]:
    jobs = [(k, s, n_perm) for k in kinds for s in range(books)]
    res, t0 = [], time.time()
    with mp.get_context("spawn").Pool(procs or max(1, (os.cpu_count() or 2) - 1)) as pool:
        for i, r in enumerate(pool.imap_unordered(_one, jobs)):
            res.append(r)
            if progress and (i + 1) % 10 == 0:
                progress(f"{i + 1}/{len(jobs)} gates ({time.time() - t0:.0f} s)")
    return res


def summarize(res: list[dict], alpha=0.05) -> pd.DataFrame:
    d = pd.DataFrame(res)
    rows = []
    for kind in [k for k in KINDS if k in set(d["kind"])]:
        x = d[d["kind"] == kind]
        for m in NULLS:
            p = x[m].to_numpy()
            se = np.sqrt(max((p <= alpha).mean() * (1 - (p <= alpha).mean()), 1e-9) / len(p))
            rows.append({"markets": kind, "null": m, "gates": len(p), f"pass rate (p <= {alpha})": (p <= alpha).mean(),
                         "+/- 1 s.e.": se, "mean p": p.mean(), "median shuffled best": x[m + "_median"].median(),
                         "median real best": x["real"].median()})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--books", type=int, default=60)
    a.add_argument("--perm", type=int, default=39)
    a.add_argument("--procs", type=int, default=None)
    a.add_argument("--out", default="calibration.json")
    args = a.parse_args()
    res = run(args.books, args.perm, args.procs)
    with open(args.out, "w") as f:
        json.dump(res, f)
    pd.set_option("display.width", 160)
    print(summarize(res).round(3).to_string(index=False))
