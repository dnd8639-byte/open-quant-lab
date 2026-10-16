"""Project 4: regime-dependent momentum ("momentum crashes", Daniel & Moskowitz 2016).

Momentum earns a premium on average but crashes when a beaten-down market rebounds: the losers it is short bounce
hardest. BEAR = the market's return over the previous 24 months is negative (known at the start of the month).
(1) momentum by regime; (2) the "written call" regression: does momentum's beta turn sharply negative when the
market rises in a bear market? (3) scaling exposure by 1 / trailing volatility (Barroso & Santa-Clara 2015), and
whether that still helps after it was published (2016 on).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._util import check, french, line, ols, scatter, stat, table

TITLE = "Momentum crashes: momentum by market regime"
QUESTION = "Does momentum fail in rebounding bear markets, and does volatility scaling still fix it after publication?"
PARAMS = []
NEEDS = ["french/F-F_Momentum_Factor/1mo", "french/F-F_Research_Data_Factors/1mo"]
HOW_TO_GET = "oql data pull-french"


def stats(r):
    cum = (1 + r).cumprod()
    return [f"{r.mean() * 12:.1%}", f"{r.std() * 12 ** 0.5:.1%}", round(float(r.mean() / r.std() * 12 ** 0.5), 2),
            round(float(r.skew()), 2), f"{r.min():.1%}", f"{(cum / cum.cummax() - 1).min():.1%}"]


def run(p, emit, stop):
    emit(check("Regression recovers a planted rebound beta", _self_test(), "fake momentum with beta -0.9 only in rebounding bear months"))
    d = french("F-F_Momentum_Factor").join(french("F-F_Research_Data_Factors"), how="inner")
    d.columns = [c.strip() for c in d.columns]
    d["mkt"] = d["Mkt-RF"] + d["RF"]
    past24 = (1 + d["mkt"]).rolling(24).apply(np.prod, raw=True).shift(1) - 1
    d = d[past24.notna()].assign(bear=(past24 < 0).astype(float)[past24.notna()])
    d["up"] = (d["Mkt-RF"] > 0).astype(float)
    mom = d["Mom"]
    cols = ["months", "return/yr", "vol/yr", "Sharpe", "skew", "worst month", "max drawdown"]
    emit(table("regime", "(1) Momentum by market regime", ["regime"] + cols,
               [["normal markets", int((d.bear == 0).sum())] + stats(mom[d.bear == 0]),
                ["bear markets", int((d.bear == 1).sum())] + stats(mom[d.bear == 1])]))
    X = np.column_stack([d.bear, d["Mkt-RF"], d.bear * d["Mkt-RF"], d.bear * d.up * d["Mkt-RF"]])
    b, t, _ = ols(mom, X, lags=6)
    emit(table("call", "(2) Written-call regression (Newey-West t-stats)", ["term", "estimate", "t"],
               [[nm, round(float(bb), 3), round(float(tt), 2)] for nm, bb, tt in
                zip(["alpha (monthly)", "extra alpha in bear", "market beta", "extra beta in bear", "extra beta, bear AND market up"], b, t)]))
    emit(stat("beta_rebound", "Momentum's beta in a rebounding bear market", float(b[2] + b[3] + b[4]), "+.2f"))
    pts = lambda m: {"x": (d["Mkt-RF"][m] * 100).tolist(), "y": (mom[m] * 100).tolist()}
    xs = np.linspace(-30, 40, 60)
    emit(scatter("scatter", "In bear markets, momentum loses when the market rebounds",
                 [{"name": "normal market", **pts(d.bear == 0)}, {"name": "bear market", **pts(d.bear == 1)},
                  {"name": "fitted, bear market", "x": xs.tolist(), "y": (100 * (b[0] + b[1]) + (b[2] + b[3]) * xs + b[4] * np.maximum(xs, 0)).tolist(), "line": True}],
                 xlabel="market excess return, % per month", ylabel="momentum return, %"))
    vol = mom.rolling(6).std().shift(1)
    scaled = (mom * (mom.std() / vol).clip(upper=3)).dropna()
    raw = mom.loc[scaled.index]
    scaled *= raw.std() / scaled.std()                           # same overall volatility, for a fair picture
    rows = []
    for label, sl in [("full sample", slice(None)), ("before 2016", slice(None, "2015-12")), ("2016 on (published)", slice("2016-01", None))]:
        rows.append([label, "plain"] + stats(raw.loc[sl])[1:4])
        rows.append(["", "volatility-scaled"] + stats(scaled.loc[sl])[1:4])
    emit(table("scaled", "(3) Volatility-scaled momentum vs plain", ["period", "version", "return/yr", "vol/yr", "Sharpe"], rows))
    cr, cs = np.log((1 + raw).cumprod()), np.log((1 + scaled).cumprod())
    tm = [int(x.value // 1_000_000) for x in raw.index]
    for end in list(range(120, len(raw), 120)) + [len(raw)]:      # draw the history decade by decade
        if stop.is_set():
            return
        emit(line("cum", "Cumulative log return: scaling by trailing volatility helps, but 1932 still hurts",
                  [{"name": "plain momentum", "x": tm[:end], "y": cr.iloc[:end].tolist()},
                   {"name": "volatility-scaled (past data only)", "x": tm[:end], "y": cs.iloc[:end].tolist()}],
                  x="time", vlines=[{"x": int(pd.Timestamp("2016-01-01", tz="UTC").value // 1_000_000), "label": "published"}]))


def _self_test() -> bool:
    """Known answer: build fake momentum whose beta is -0.9 only when the market rises in a bear month."""
    rng = np.random.default_rng(4)
    n = 1200
    mkt = rng.normal(0.006, 0.045, n); bear = (rng.random(n) < 0.2).astype(float); up = (mkt > 0).astype(float)
    mom = 0.008 - 0.9 * bear * up * mkt + rng.normal(0, 0.03, n)
    b, t, _ = ols(mom, np.column_stack([bear, mkt, bear * mkt, bear * up * mkt]), lags=0)
    return abs(b[4] + 0.9) < 0.2 and abs(b[2]) < 0.1
