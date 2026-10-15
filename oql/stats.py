"""Statistics the gate reports beside the permutation p-value."""
from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

from .search import Scorer, sharpe

ND = NormalDist()


def deflated_sharpe(r, n_trials: int) -> float:
    """Probability that the true per-bar Sharpe beats the best one expected from n_trials no-skill strategies
    (Bailey & Lopez de Prado 2014). The no-skill spread is taken from the track length (sd ~ 1/sqrt(T)),
    adjusted for skew and fat tails."""
    r = np.asarray(r, float); r = r[np.isfinite(r)]
    if len(r) < 10 or r.std() == 0:
        return float("nan")
    sr = r.mean() / r.std(ddof=1)
    z = (r - r.mean()) / r.std()
    skew, kurt = (z ** 3).mean(), (z ** 4).mean()
    sd = 1 / np.sqrt(len(r))
    g = 0.5772156649
    sr0 = 0.0 if n_trials < 2 else sd * ((1 - g) * ND.inv_cdf(1 - 1 / n_trials) + g * ND.inv_cdf(1 - 1 / (n_trials * np.e)))
    return ND.cdf((sr - sr0) * np.sqrt(len(r) - 1) / np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12)))


def cost_monte_carlo(scorer: Scorer, g: dict, n: int = 200, seed: int = 0, max_delay: float = 3 / 23) -> dict:
    """Rerun with random costs (lognormal, median x1.3, often x2-3) and random fill delays."""
    rng = np.random.default_rng(seed)
    s = []
    for _ in range(n):
        port, _, _ = scorer.returns(g, cost_mult=float(np.exp(rng.normal(0.25, 0.45))), delay=float(rng.uniform(0, max_delay)))
        s.append(sharpe(port.to_numpy(), scorer.prep.ppy))
    s = np.array(s)
    return dict(median=float(np.median(s)), p5=float(np.percentile(s, 5)), share_positive=float((s > 0).mean()))


def side_and_beta(scorer: Scorer, g: dict, delay: float) -> dict:
    """Does it make money on both sides, and how much of it is just being long the market?"""
    port, _, _ = scorer.returns(g, delay=delay)
    long_, _, _ = scorer.returns({**g, "side": "long"}, delay=delay)
    short, _, _ = scorer.returns({**g, "side": "short"}, delay=delay)
    always_long = {n: np.ones(len(scorer.prep.index[n])) for n in scorer.prep.names}
    from .engine import portfolio
    mkt, _, _ = portfolio(scorer.prep, always_long, cost_mult=0.0, delay=delay)
    j = pd.concat([port.rename("s"), mkt.rename("m")], axis=1).dropna()
    beta = float(np.cov(j.s, j.m)[0, 1] / j.m.var()) if j.m.var() > 0 else float("nan")
    ppy = scorer.prep.ppy
    return dict(sharpe=sharpe(port, ppy), sharpe_long=sharpe(long_, ppy), sharpe_short=sharpe(short, ppy),
                beta_to_market=beta, corr_to_market=float(j.s.corr(j.m)), market_sharpe=sharpe(mkt, ppy),
                ann_vol=float(port.std() * np.sqrt(ppy)))
