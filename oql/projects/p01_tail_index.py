"""Project 1: how fat are the tails? Hill estimator of the tail index, and what it does to Gaussian VaR.

Far-out losses follow a power law, P(loss > x) ~ c x^-alpha; smaller alpha = fatter tail (alpha < 4: infinite
kurtosis). Hill: sort losses largest first; with the k largest, alpha = 1 / mean(log(x_i / x_(k+1))). Read alpha
where the curve against k is flat (here the median over k = 2%..5% of losses). Checks first on simulated data with
a known alpha, then real markets: a 99% Gaussian VaR from trailing volatility should be breached on 1% of days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import catalog
from ._util import N, bars, check, daily_log_returns, line, stat, table

TITLE = "Tail index: how fat are the tails?"
QUESTION = "How often do big losses really happen, and how badly does a bell-curve risk model understate them?"
FUTURES = [("ES", "S&P 500"), ("CL", "Crude oil"), ("GC", "Gold"), ("ZN", "10y Treasury"), ("6E", "Euro")]
PARAMS = [{"name": "crypto", "label": "Crypto market", "type": "dataset", "kind": "bars", "prefix": "binance/",
           "default": "binance/BTCUSDT/1h"},
          {"name": "futures", "label": "Add CME futures (licensed, stays local)", "type": "choice", "options": ["yes", "no"], "default": "yes"}]


def hill_curve(losses, kmax=None):
    """alpha_hat for every k = 1..kmax (losses = positive numbers)."""
    x = np.sort(losses[losses > 0])[::-1]
    kmax = kmax or len(x) // 5
    lx = np.log(x)
    k = np.arange(1, kmax + 1)
    return k, 1.0 / (np.cumsum(lx)[:kmax] / k - lx[k])


def hill(losses, lo=0.02, hi=0.05):
    n = int((losses > 0).sum())
    k, a = hill_curve(losses, int(n * hi) + 1)
    return float(np.median(a[int(n * lo):int(n * hi)]))


def var_breaches(r, level=0.99, window=250):
    """Share of days with loss beyond the Gaussian VaR, and P(breach tomorrow | breach today)."""
    vol = r.rolling(window).std().shift(1)                     # yesterday's estimate: no look-ahead
    b = (r < -N.inv_cdf(level) * vol)[vol.notna()]
    return float(b.mean()), float((b & b.shift(1, fill_value=False)).sum() / max(b.sum(), 1)), int(b.sum())


def run(p, emit, stop):
    rng = np.random.default_rng(0)
    rows = []
    for name, true, x, ok in [("Pareto, alpha 2.5", 2.5, -(rng.pareto(2.5, 20000) + 1), lambda a: abs(a - 2.5) < 0.25),
                              ("Student-t, 3 dof", 3, rng.standard_t(3, 20000), lambda a: 2.0 < a < 3.6),
                              ("Student-t, 5 dof", 5, rng.standard_t(5, 20000), lambda a: 2.8 < a < 5.8),
                              ("Normal (no power-law tail)", np.inf, rng.standard_normal(20000), lambda a: a > 4.0)]:
        a = hill(-x)
        rows.append([name, "infinite" if np.isinf(true) else true, round(a, 2)])
        emit(table("checks", "Known-answer check: 20,000 simulated draws each", ["distribution", "true alpha", "Hill estimate"], rows))
        emit(check(name, ok(a), f"estimate {a:.2f} (exact for Pareto; biased low for Student-t tails, as theory says)"))
    assets = {p["crypto"].split("/")[1]: daily_log_returns(p["crypto"])}
    if p.get("futures") == "yes":
        have = {m.id for m in catalog.list_all()}
        for root, label in FUTURES:
            if f"databento/{root}/1d" in have:
                assets[label] = daily_log_returns(f"databento/{root}/1d")
    curves, real = [], []
    for name, r in assets.items():
        losses = -r.to_numpy()
        k, a = hill_curve(losses)
        pct = k / (losses > 0).sum() * 100
        keep = pct <= 20
        curves.append({"name": name, "x": pct[keep].tolist(), "y": np.clip(a[keep], 1, 7).tolist()})
        alpha = hill(losses)
        rate, clus, nb = var_breaches(r)
        real.append([name, len(r), round(alpha, 2), round(float(r.kurt() + 3), 1), f"{rate:.2%}", f"{clus:.0%}"])
        emit(line("hill", "Hill plot: read alpha where the curve is flat (shaded: 2-5% of losses)", curves,
                  x="num", xlabel="k, as % of losses used", ylabel="tail index alpha", y_range=[1, 7], x_range=[0, 20],
                  bands=[{"x0": 2, "x1": 5, "label": "read here"}]))
        emit(table("real", "Real daily returns, left tail", ["market", "days", "alpha", "kurtosis", "99% VaR breached", "breach after breach"], real))
        emit(bars("breaches", "Gaussian 99% VaR is breached more often than the 1% it promises", [x[0] for x in real],
                  [{"name": "share of days", "y": [float(x[4].rstrip("%")) / 100 for x in real]}], y_format=".1%",
                  hlines=[{"y": 0.01, "label": "promised 1%"}]))
        if len(real) == 1:
            emit(stat("alpha", f"{name} tail index", alpha))
            emit(stat("breach", f"{name} VaR breaches (promised 1%)", rate, ".2%"))
