"""Project 9: do trades trigger more trades? A Hawkes process fitted to trade arrival times.

Independent arrivals would be a Poisson process; a Hawkes process lets each trade raise the chance of the next. The
branching ratio n = alpha / beta is the share of trades that are reactions to earlier trades. Exponential kernel,
maximum likelihood with the O(N) recursion. Checks: simulated processes with known n must be recovered; Ogata's
residual test (gaps in model time should be Exponential(1)) on every fit, which on real trades FAILS: one
exponential kernel is too simple for real markets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import catalog
from ._util import bars, check, line, stat, table
from .hawkes import fit, ks_exponential, residuals, simulate

TITLE = "Hawkes process: do trades trigger trades?"
QUESTION = "What share of trades are reactions to earlier trades rather than to outside news?"
PARAMS = [{"name": "block_hours", "label": "Hours per block", "type": "choice", "options": [3, 6, 12], "default": 6}]
NEEDS = ["binanceus/BTCUSDT/trades"]
HOW_TO_GET = "oql data import-projects ~/Desktop/ahl-node/quant-projects/data"


def run(p, emit, stop):
    rng = np.random.default_rng(3)
    rows = []
    for mu, alpha, beta in [(0.5, 0.0, 1.0), (0.5, 0.6, 2.0), (0.2, 1.4, 2.0), (0.05, 4.5, 5.0)]:
        if stop.is_set():
            return
        H = 4000 / (mu / (1 - alpha / beta))
        t = simulate(mu, max(alpha, 1e-12), beta, H, rng)
        f = fit(t, H)
        d, pk = ks_exponential(residuals(t, f["mu"], f["alpha"], f["beta"]))
        rows.append([f"mu {mu}, alpha {alpha}, beta {beta}", len(t), round(alpha / beta, 2), round(f["n"], 3), f"{pk:.2f}"])
        emit(table("sim", "Known-answer check: simulated Hawkes processes", ["truth", "events", "true n", "fitted n", "residual test p"], rows))
        emit(check(f"Recovers n = {alpha / beta:.2f}", abs(f["n"] - alpha / beta) < 0.08, f"fitted {f['n']:.3f}; residual test p = {pk:.2f}"))
    t = simulate(0.2, 1.4, 2.0, 300, rng)
    grid = np.linspace(0, 120, 2400)
    lam = 0.2 + 1.4 * np.array([np.exp(-2.0 * (g - t[t < g])).sum() for g in grid])
    emit(line("intensity", "Simulated Hawkes process (n = 0.7): events arrive in bursts", [
        {"name": "intensity (events per second)", "x": grid.tolist(), "y": lam.tolist()}], x="num", xlabel="seconds"))
    try:
        tr = catalog.read("binanceus/BTCUSDT/trades")
    except KeyError:
        emit({"log": f"Real trades not in the catalog. To add them: {HOW_TO_GET}"})
        return
    ms = np.asarray((tr.index - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1))   # any time unit
    ts = np.unique(ms) / 1000.0                                                # fills of one order share a timestamp
    ts = np.sort(ts + np.random.default_rng(0).uniform(0, 0.001, len(ts)))
    t0, hours = ts[0], (ts[-1] - ts[0]) / 3600
    emit(stat("trades", f"Distinct trade times over {hours:.1f} hours", len(ts), ",.0f"))
    f = fit(ts - t0, ts[-1] - t0)
    d, pk = ks_exponential(residuals(ts - t0, f["mu"], f["alpha"], f["beta"]))
    emit(stat("n_all", "Branching ratio, whole sample", f["n"], ".2f"))
    emit(stat("half", "Half-life of a trade's influence (seconds)", np.log(2) / f["beta"], ".2f"))
    emit(stat("ks", "Residual test p (low = model too simple)", pk, ".3f"))
    labels, ns = ["all"], [f["n"]]
    B = int(p["block_hours"])
    for h in range(0, int(round(hours)), B):
        if stop.is_set():
            return
        w = ts[(ts >= t0 + 3600 * h) & (ts < t0 + 3600 * (h + B))] - (t0 + 3600 * h)
        if len(w) >= 200:
            labels.append(f"h {h}-{h + B}"); ns.append(fit(w, 3600.0 * B)["n"])
            emit(bars("blocks", "Real BTC trades: share of trades triggered by other trades, block by block", labels,
                      [{"name": "branching ratio n", "y": ns}], y_range=[0, 1]))
    res = np.sort(residuals(ts - t0, f["mu"], f["alpha"], f["beta"]))
    q = -np.log(1 - (np.arange(1, len(res) + 1) - 0.5) / len(res))
    pois = np.sort(np.diff(ts) * len(ts) / (ts[-1] - ts[0]))
    step = max(1, len(q) // 400)
    emit(line("qq", "Residual check on real trades: on the dashed line = good fit", [
        {"name": "if you assume Poisson", "x": q[::step].tolist(), "y": pois[:len(q)][::step].tolist()},
        {"name": "after the Hawkes fit", "x": q[::step].tolist(), "y": res[::step].tolist()},
        {"name": "perfect fit", "x": [0, 8], "y": [0, 8], "dash": True}], x="num", x_range=[0, 8], y_range=[0, 12],
        xlabel="expected gap (Exponential(1) quantiles)", ylabel="observed gap in model time"))
