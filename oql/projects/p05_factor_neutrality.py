"""Project 5: factor neutrality. Is a strategy's return really its own, or a known factor in disguise?

Regress the strategy's returns on known factors. The betas show what it is secretly exposed to; the intercept
(alpha) is what is left. If alpha vanishes once the factors are included, the strategy was a repackaged factor.
(0) Self-test: 500 fake strategies = 0.5 x market + noise with zero alpha; the test must find beta 0.5 and raise a
false alarm about 5% of the time. (1) US momentum vs the five Fama-French factors. (2) A lab strategy vs its own
market: is it a hidden bet on the market going up, or on the market moving a lot?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import catalog
from ..engine import prepare
from ..search import Scorer
from ..strategies import RULES
from ._util import bars, check, french, hist, ols, stat, table

TITLE = "Factor neutrality: is the return its own?"
QUESTION = "How much of a strategy's return is just exposure to known factors or to its market?"
PARAMS = [{"name": "dataset", "label": "Market for part 2", "type": "dataset", "kind": "bars", "default": "binance/BTCUSDT/1h"},
          {"name": "rule", "label": "Lab rule", "type": "choice", "options": list(RULES), "default": "momentum"},
          {"name": "lookback", "label": "Lookback / fast (bars)", "type": "choice", "options": [5, 10, 20, 50, 100, 250], "default": 20}]
NEEDS = ["french/F-F_Research_Data_5_Factors_2x3/1mo", "french/F-F_Momentum_Factor/1mo"]
HOW_TO_GET = "oql data pull-french"


def neutrality(y, factors, per_year, lags=6):
    d = pd.concat([y.rename("y"), factors], axis=1).dropna()
    b, t, r2 = ols(d["y"], d[factors.columns], lags)
    raw_t = d["y"].mean() / d["y"].std() * np.sqrt(len(d))
    return dict(n=len(d), raw=d["y"].mean() * per_year, raw_t=raw_t, alpha=b[0] * per_year, t_alpha=t[0], r2=r2,
                betas=dict(zip(factors.columns, b[1:])), t=dict(zip(factors.columns, t[1:])))


def run(p, emit, stop):
    ff5 = french("F-F_Research_Data_5_Factors_2x3")
    ff5.columns = [c.strip() for c in ff5.columns]
    ff5 = ff5.drop(columns="RF")
    rng = np.random.default_rng(1)
    bs, ts = [], []
    for i in range(500):                                          # (0) fake strategies with NO alpha
        if stop.is_set():
            return
        fake = 0.5 * ff5["Mkt-RF"].to_numpy() + rng.normal(0, 0.02, len(ff5))
        b, t, _ = ols(fake, ff5, 6)
        bs.append(b[1]); ts.append(t[0])
        if (i + 1) % 25 == 0:
            emit(hist("fake", f"(0) Self-test: alpha t-statistics of {i + 1} fake strategies with zero alpha", ts, 40, (-4, 4),
                      vlines=[{"x": -2, "label": "-2"}, {"x": 2, "label": "+2"}], xlabel="t-statistic of alpha"))
            emit(stat("false", "False alarms so far (|t| > 2; should be about 5%)", float(np.mean(np.abs(ts) > 2)), ".1%"))
    fa = float(np.mean(np.abs(ts) > 2))
    emit(check("Finds the planted market beta", abs(np.mean(bs) - 0.5) < 0.02, f"average beta {np.mean(bs):.3f} (true 0.5)"))
    emit(check("False alarms near 5%", 0.02 <= fa <= 0.09, f"{fa:.1%} of 500 zero-alpha strategies flagged"))
    mom = french("F-F_Momentum_Factor")
    mom.columns = [c.strip() for c in mom.columns]
    m = neutrality(mom["Mom"], ff5, 12)
    emit(bars("mom_betas", f"(1) US momentum vs the five Fama-French factors: alpha {m['alpha']:.1%}/yr (t = {m['t_alpha']:.1f})",
              list(m["betas"]), [{"name": "beta", "y": [float(v) for v in m["betas"].values()]}],
              notes=[f"{k}: t = {m['t'][k]:+.1f}" for k in m["betas"]]))
    emit(stat("mom_alpha", "Momentum alpha after five factors (per year)", float(m["alpha"]), ".1%"))
    # (2) a lab strategy against its own market
    ds = p["dataset"]
    df = catalog.read(ds)
    g = {"rule": p["rule"], "params": _params(p["rule"], int(p["lookback"])), "side": "both"}
    sc = Scorer(prepare({ds: df}, {"default": 1.0}, warmup=min(250, len(df) // 10)), RULES, 0.0)
    port = sc.returns(g)[0]
    strat = port.resample("1D").sum()
    mkt = np.log(df["close"].astype(float)).diff().resample("1D").sum()
    per_year = 365 if ds.startswith("binance") else 252
    fac = pd.DataFrame({"market": mkt, "size of move": mkt.abs() - mkt.abs().mean()})
    a = neutrality(strat, fac[["market"]], per_year, lags=5)
    b2 = neutrality(strat, fac, per_year, lags=5)
    emit(table("lab", f"(2) {p['rule']} {g['params']} on {ds}, daily, 1 bp costs", ["model", "raw return/yr", "alpha/yr", "t(alpha)", "beta to market", "beta to size of move"],
               [["direction only", f"{a['raw']:.1%}", f"{a['alpha']:.1%}", round(float(a['t_alpha']), 2), round(float(a['betas']['market']), 2), "-"],
                ["direction + size", f"{b2['raw']:.1%}", f"{b2['alpha']:.1%}", round(float(b2['t_alpha']), 2),
                 round(float(b2['betas']['market']), 2), round(float(b2['betas']['size of move']), 2)]]))
    emit(stat("lab_beta", "Lab strategy's beta to its market", float(a["betas"]["market"]), "+.2f"))


def _params(rule, lb):
    if rule == "ma_cross":
        return {"fast": min(lb, 50), "slow": max(lb * 4, 100)}
    return {"lookback": lb}
