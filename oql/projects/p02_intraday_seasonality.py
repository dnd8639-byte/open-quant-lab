"""Project 2: intraday seasonality. Does the hour of the week matter?

Two questions with different answers: is the market reliably more ACTIVE at certain hours (size of moves), and does
it reliably go UP or DOWN at certain hours (sign of moves)? Build a 7 x 24 map (weekday x hour, UTC). A heatmap of
168 cells of pure noise also shows a pattern, so the statistic is the spread of the 168 cell averages, tested
against shuffles of the hour-of-week labels WITHIN each week (keeps each week's moves, destroys only when in the
week they happened). Stability: the map fitted on the years before 2022 vs the map from 2022 on.
Checks: the test must find a pattern planted in fake data, and must not find one in plain noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import catalog
from ._util import check, heatmap, hist, stat

TITLE = "Intraday seasonality: does the hour of the week matter?"
QUESTION = "Is a market reliably busier, or reliably up or down, at certain hours of the week?"
PARAMS = [{"name": "dataset", "label": "Hourly market", "type": "dataset", "kind": "bars", "freq": "1h", "default": "binance/BTCUSDT/1h"},
          {"name": "shuffles", "label": "Shuffles per test", "type": "choice", "options": [200, 500, 1000], "default": 500}]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def cell_means(values, slot):
    return np.bincount(slot, weights=values, minlength=168) / np.maximum(np.bincount(slot, minlength=168), 1)


def shuffled_spreads(v, slot, week, n, rng, every=None, on_progress=None, stop=None):
    """Spread of the 168 cell means after shuffling slot labels within each week, n times."""
    out = []
    for i in range(n):
        if stop is not None and stop.is_set():
            break
        perm = np.lexsort((rng.random(len(week)), week))           # a random order inside each week
        out.append(cell_means(v, slot[perm]).std())
        if on_progress and ((i + 1) % every == 0 or i + 1 == n):
            on_progress(out)
    return np.array(out)


def p_value(real, null):
    return (1 + int((np.asarray(null) >= real).sum())) / (len(null) + 1)


def run(p, emit, stop):
    rng = np.random.default_rng(0)
    # Known-answer checks on fake data with the same calendar shape: 4 years of hours
    t = pd.date_range("2018-01-01", periods=4 * 365 * 24, freq="1h", tz="UTC")
    s_fake = (t.dayofweek * 24 + t.hour).to_numpy(); w_fake = ((t - t[0]).days // 7).to_numpy()
    noise = rng.standard_normal(len(t))
    planted = np.abs(noise * np.where(np.isin(t.hour, [13, 14, 15]), 1.5, 1.0))
    for label, v, want in [("Finds a planted busy-hours pattern", planted, True), ("Finds nothing in plain noise", np.abs(noise), False)]:
        pv = p_value(cell_means(v, s_fake).std(), shuffled_spreads(v, s_fake, w_fake, 99, rng))
        emit(check(label, (pv < 0.05) == want, f"p = {pv:.3f} with 99 within-week shuffles"))

    c = catalog.read(p["dataset"])["close"].astype(float)
    r = np.log(c).diff().dropna()
    r = r[r.index == r.index.floor("1h")]                        # bars off the hour grid cannot be placed in a cell
    slot = (r.index.dayofweek * 24 + r.index.hour).to_numpy()
    week = ((r.index - pd.Timestamp("2018-01-01", tz="UTC")).days // 7).to_numpy()
    order = np.argsort(week, kind="stable")
    slot, week, vals = slot[order], week[order], r.to_numpy()[order]
    split = np.asarray(r.index < pd.Timestamp("2022-01-01", tz="UTC"))[order]
    name = p["dataset"].split("/")[1]
    emit(stat("bars", f"{name} hourly bars", len(r), ",.0f"))
    n = int(p["shuffles"])
    for key, word, label, v, div in [("vol", "Volatility", "VOLATILITY (size of the move)", np.abs(vals), False),
                                     ("dir", "Direction", "DIRECTION (signed move)", vals, True)]:
        emit(heatmap(f"map_{key}", f"{label}: average per hour of the week, basis points", cell_means(v, slot).reshape(7, 24) * 1e4,
                     [f"{h:02d}" for h in range(24)], DAYS, diverging=div, unit="bp", xlabel="hour (UTC)"))
        real = cell_means(v, slot).std()

        def show(null, key=key, word=word, label=label, real=real):
            emit(hist(f"null_{key}", f"{label}: spread of the 168 cells, real vs {len(null)} within-week shuffles",
                      np.array(null) * 1e4, 40, vlines=[{"x": real * 1e4, "label": "real"}], xlabel="spread of cell means, bp"))
            emit(stat(f"p_{key}", f"{word}: p {'so far' if len(null) < n else 'value'}", p_value(real, null), ".3f"))
        shuffled_spreads(v, slot, week, n, rng, every=20, on_progress=show, stop=stop)
        if split.any() and (~split).any():
            oos = float(np.corrcoef(cell_means(v[split], slot[split]), cell_means(v[~split], slot[~split]))[0, 1])
            emit(stat(f"oos_{key}", f"{word}: map before 2022 vs from 2022 (correlation)", oos, "+.2f"))
