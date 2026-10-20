"""Project 11: does the gate hold when thousands of strategies are tried?

Overfitting bites hardest in big searches: try enough rules and the best one looks excellent on pure noise. This
runs the lab's own search loop at growing sizes (dense grids: every momentum and breakout lookback from 2 to 250
days, about 3,000 moving-average pairs, short reversals) on a synthetic book with a known truth, and for each size
compares three verdicts on the winner:
  naive      a t-test of the winner's returns as if it were the only strategy tried
  deflated   the Deflated Sharpe Ratio (Bailey & Lopez de Prado), which corrects for the number tried
  gate       the lab's test: the SAME search rerun on shuffled copies (volatility-matched null)
On a market with no edge, the naive test should start "finding" edges as the search grows, while the gate keeps its
5% promise. On a market with a planted edge, the gate should still pass at every size.
"""
from __future__ import annotations

import numpy as np

from ..engine import prepare
from ..experiment import Experiment
from ..permute import permute_book
from ..search import Scorer, search, sharpe
from ..stats import deflated_sharpe
from ._util import N, check, line, stat, table

TITLE = "Large search: overfitting at scale"
QUESTION = "Try thousands of strategies on pure noise: does the naive test get fooled, and does the gate hold?"
PARAMS = [{"name": "market", "label": "Synthetic market", "type": "choice",
           "options": ["no edge (random walks)", "planted edge (persistent trends)"], "default": "no edge (random walks)"},
          {"name": "largest", "label": "Largest search", "type": "choice", "options": [1000, 3000], "default": 1000},
          {"name": "shuffles", "label": "Shuffles per size", "type": "choice", "options": [39, 99], "default": 39}]

DENSE = {"momentum": {"lookback": list(range(2, 251))}, "breakout": {"lookback": list(range(5, 251))},
         "ma_cross": {"fast": list(range(2, 60)), "slow": list(range(20, 301, 5))}, "reversal": {"lookback": list(range(1, 21))}}


def experiment(size: int) -> Experiment:
    """The lab's search loop with a budget of about `size` strategies (fewer distinct ones: repeats are not rescored)."""
    return Experiment(datasets=[], grid=DENSE, pop_init=max(2, size // 2), rounds=5, children=max(1, size // 10),
                      fresh=0, elite=max(2, size // 50), warmup=250)


def book(planted: bool, seed: int = 7) -> dict:
    from ..calibration import clustered_market
    from ..synthetic import trend_market
    if planted:
        return {f"m{i}": trend_market(2000, seed * 100 + i, drift=0.0015) for i in range(3)}
    return {f"m{i}": clustered_market(2000, seed * 100 + i) for i in range(3)}


def run(p, emit, stop):
    planted = p["market"].startswith("planted")
    largest, n_perm = int(p["largest"]), int(p["shuffles"])
    sizes = [s for s in (10, 30, 100, 300, 1000, 3000) if s <= largest]
    b = book(planted)
    cut = sorted(set().union(*[set(df.index) for df in b.values()]))
    cut = cut[int(len(cut) * 0.7)]
    dev = {k: df[df.index < cut] for k, df in b.items()}
    emit({"log": f"{'planted edge' if planted else 'no edge'}: 3 synthetic markets, development part only; {n_perm} shuffles per size"})
    xs, real, med, p95, rows = [], [], [], [], []
    for k, size in enumerate(sizes):
        if stop.is_set():
            return
        exp = experiment(size)
        sc = Scorer(prepare(dev, {"default": 1.0}, warmup=exp.warmup), exp.rule_table(), exp.min_trades_per_year)
        res = search(sc, exp)
        port = sc.returns(res["best"], delay=exp.delay)[0].to_numpy()
        r = port[np.isfinite(port)]
        naive_p = 1 - N.cdf(r.mean() / r.std(ddof=1) * np.sqrt(len(r)))       # one-sided t-test, as if alone
        dsr = deflated_sharpe(r, res["n_trials"])
        null = []
        for j in range(n_perm):
            if stop.is_set():
                return
            nb = permute_book(dev, 20_000 + 97 * k + j, method="vol_matched")
            ns = Scorer(prepare(nb, {"default": 1.0}, warmup=exp.warmup), exp.rule_table(), exp.min_trades_per_year)
            null.append(search(ns, exp)["best_score"])
            emit({"progress": (k + (j + 1) / n_perm) / len(sizes)})
        null = np.array(null)
        gate_p = float((1 + (null >= res["best_score"]).sum()) / (n_perm + 1))
        xs.append(res["n_trials"]); real.append(res["best_score"]); med.append(float(np.median(null))); p95.append(float(np.percentile(null, 95)))
        rows.append([res["n_trials"], round(res["best_score"], 2), round(float(np.median(null)), 2), f"{naive_p:.4f}", f"{dsr:.2f}",
                     f"{gate_p:.3f}", "PASS" if gate_p < 0.05 else "FAIL"])
        emit(line("best", "Best Sharpe found vs strategies tried (log scale)", [
            {"name": "best on the real market", "x": xs, "y": real, "points": True},
            {"name": "best on shuffled copies: median", "x": xs, "y": med, "dash": True},
            {"name": "best on shuffled copies: 95th percentile", "x": xs, "y": p95, "dash": True}],
            x="num", x_log=True, xlabel="distinct strategies tried", ylabel="Sharpe of the best"))
        emit(table("verdicts", "Three verdicts on the winner at each search size", ["strategies tried", "best Sharpe",
                   "shuffled best (median)", "naive p", "deflated Sharpe", "gate p", "gate"], rows))
        emit(stat("tried", "Largest search so far (distinct strategies)", res["n_trials"], ",.0f"))
    naive_fooled = any(float(r[3]) < 0.05 for r in rows)
    gate_passes = [r[6] == "PASS" for r in rows]
    if planted:
        emit(check("The gate finds the planted edge at every search size", all(gate_passes),
                   f"gate p by size: {', '.join(r[5] for r in rows)}"))
    else:
        emit(check("The gate is not fooled by the largest search", not gate_passes[-1],
                   f"gate p {rows[-1][5]} with {rows[-1][0]} strategies; naive p {rows[-1][3]}"))
        emit({"log": ("the naive t-test called the winner significant at some size: that is overfitting, and the gate "
                      "did not fall for it") if naive_fooled else "the naive test was not fooled at these sizes on this book"})
