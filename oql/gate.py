"""The validation gate and the one-shot holdout.

GATE (development data only; the last 30% of dates is cut off BEFORE anything is computed)
  1. Run the search loop declared in the experiment. Every strategy tried is counted.
  2. Rerun the SAME loop, every round, on `n_perm` shuffled copies of the development data (one joint shuffle
     across markets). p = share of shuffled runs whose best beats the real best. This answers the right
     question: "is the best of everything I tried better than the best that luck produces with this search?"
  3. Report the deflated Sharpe, a cost/delay Monte Carlo, long vs short, and beta to the market.
  PASS = p < alpha AND the cost Monte Carlo's median Sharpe stays above zero.

HOLDOUT (one-shot; the lock is written before the result is computed)
  The frozen finalist runs once on the last 30%. Its Sharpe is compared with the same strategy on shuffled
  holdout bars. PASS = holdout Sharpe > 0 AND p < alpha.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time

import numpy as np

from . import experiment as X
from .engine import prepare
from .permute import permute_book
from .search import Scorer, search, sharpe
from .stats import cost_monte_carlo, deflated_sharpe, side_and_beta


def _scorer(book, exp, **kw):
    costs = {"default": exp.cost_bps.get("default", 1.0), **{k: v for k, v in exp.cost_bps.items()}}
    return Scorer(prepare(book, costs, warmup=exp.warmup, band=getattr(exp, "rebalance_band", 0.0), **kw),
                  exp.rule_table(), exp.min_trades_per_year,
                  getattr(exp, "gate_features", ["price"]))


def _exp(ref):
    return X.load(ref) if isinstance(ref, str) else ref


def _null_search(args):
    ref, dev, seed, *method = args
    exp = _exp(ref)
    return search(_scorer(permute_book(dev, seed, method=method[0] if method else "bars"), exp), exp)["best_score"]


def _null_holdout(args):
    ref, book, split, g, seed, *method = args
    exp = _exp(ref)
    sc = _scorer(permute_book(book, seed, from_date=split, method=method[0] if method else "bars"), exp, score_from=split)
    return sharpe(sc.returns(g, delay=exp.delay)[0].to_numpy(), sc.prep.ppy)


def _pool(procs):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    return mp.get_context("spawn").Pool(procs or max(1, (os.cpu_count() or 2) - 1))


def run_gate(exp, procs=None, n_perm=None, verbose=True, progress=None) -> dict:
    """`progress(event)` (optional) receives dicts as the test runs, for live displays:
    {"status": "searching"}, {"status": "permuting", "real": best, "n_perm": N, "finalist": g},
    {"null": value, "done": k} per shuffled rerun as it finishes, and {"status": "done", "verdict": ..., "p": ...}."""
    t0 = time.time()
    n_perm = n_perm or exp.n_perm
    emit = progress or (lambda e: None)
    emit({"status": "searching"})
    book = X.load_book(exp)
    split = X.split_date(book, exp.dev_frac)
    dev = X.truncate(book, split)                         # the holdout is never loaded past this line
    sc = _scorer(dev, exp)
    log = []
    res = search(sc, exp, log)
    g = res["best"]
    port = sc.returns(g, delay=exp.delay)[0]
    if verbose:
        print(f"search: {res['n_trials']} strategies, best dev Sharpe {res['best_score']:.3f}: {g}", flush=True)
        print(f"permutation test: rerunning the whole search on {n_perm} shuffled copies...", flush=True)
    ref = exp.path or exp
    emit({"status": "permuting", "real": res["best_score"], "n_perm": n_perm, "finalist": g, "n_trials": res["n_trials"]})
    nulls = list(getattr(exp, "nulls", ["bars"]))
    null, others = [], {}
    with _pool(procs) as pool:
        for k, v in enumerate(pool.imap_unordered(_null_search, [(ref, dev, 1000 + k, nulls[0]) for k in range(n_perm)])):
            null.append(v)
            emit({"null": float(v), "done": k + 1})
        for method in nulls[1:]:                           # further declared nulls: same search, same seeds
            emit({"status": "permuting", "null_model": method})
            x = np.sort(np.array(pool.map(_null_search, [(ref, dev, 1000 + k, method) for k in range(n_perm)])))
            others[method] = dict(p=float((1 + (x >= res["best_score"]).sum()) / (n_perm + 1)), median=float(np.median(x)),
                                  p95=float(np.percentile(x, 95)))
    null = np.sort(np.array(null))                        # order of arrival does not matter; sorted = reproducible
    p = float((1 + (null >= res["best_score"]).sum()) / (n_perm + 1))
    mc = cost_monte_carlo(sc, g)
    passed = bool(max([p] + [o["p"] for o in others.values()]) < exp.alpha and mc["median"] > 0)
    emit({"status": "done", "verdict": "PASS" if passed else "FAIL", "p": p})
    return dict(kind="gate", verdict="PASS" if passed else "FAIL", p=p, alpha=exp.alpha, n_perm=n_perm,
                null_model=nulls[0], other_nulls=others,
                split=str(split), finalist=g, dev_sharpe=res["best_score"], n_trials=res["n_trials"],
                history=res["history"], null_median=float(np.median(null)), null_p95=float(np.percentile(null, 95)),
                null=null.tolist(), deflated_sharpe=deflated_sharpe(port.to_numpy(), res["n_trials"]),
                cost_mc=mc, side_beta=side_and_beta(sc, g, exp.delay), trials=log,
                seconds=round(time.time() - t0, 1))


def run_holdout(exp, finalist: dict, split: str, procs=None, n_perm=199) -> dict:
    import pandas as pd
    book = X.load_book(exp)
    split = pd.Timestamp(split)
    sc = _scorer(book, exp, score_from=split)
    port = sc.returns(finalist, delay=exp.delay)[0]
    real = sharpe(port.to_numpy(), sc.prep.ppy)
    ref = exp.path or exp
    nulls = list(getattr(exp, "nulls", ["bars"]))
    ps = {}
    with _pool(procs) as pool:
        for method in nulls:
            x = np.array(pool.map(_null_holdout, [(ref, book, split, finalist, 5000 + k, method) for k in range(n_perm)]))
            ps[method] = float((1 + (x >= real).sum()) / (n_perm + 1))
            if method == nulls[0]:
                null = x
    p = ps[nulls[0]]
    passed = bool(real > 0 and max(ps.values()) < exp.alpha)
    return dict(kind="holdout", verdict="PASS" if passed else "FAIL", holdout_sharpe=real, p=p, n_perm=n_perm,
                start=str(split), end=str(port.index[-1]) if len(port) else "", bars=int(port.notna().sum()),
                side_beta=side_and_beta(sc, finalist, exp.delay), null_median=float(np.median(null)),
                null_model=nulls[0], p_by_null=ps)
