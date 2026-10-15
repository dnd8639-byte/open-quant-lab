"""Fake markets for testing the lab itself: one with a real, planted edge and one with nothing to find.
A validation system is only trustworthy if it passes the first and fails the second."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _bars(lr: np.ndarray, rng, start="2010-01-04", vol=0.01) -> pd.DataFrame:
    n = len(lr)
    days = pd.bdate_range(start, periods=n, tz="UTC", name="ts")
    close = 100 * np.exp(np.cumsum(lr))
    gap = rng.normal(0, 0.15 * vol, n)
    open_ = np.concatenate([[100.0], close[:-1]]) * np.exp(gap)
    wig = np.abs(rng.normal(0, 0.6 * vol, (n, 2)))
    high = np.maximum(open_, close) * np.exp(wig[:, 0])
    low = np.minimum(open_, close) * np.exp(-wig[:, 1])
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": rng.integers(1000, 5000, n).astype(float)}, index=days)


def trend_market(n=3000, seed=0, drift=0.002, vol=0.01, stay=0.985) -> pd.DataFrame:
    """Persistent up/down regimes: a trend-following rule has a real edge here."""
    rng = np.random.default_rng(seed)
    d = np.empty(n); d[0] = 1
    for t in range(1, n):
        d[t] = d[t - 1] if rng.random() < stay else -d[t - 1]
    lr = d * drift + rng.standard_t(5, n) * vol / np.sqrt(5 / 3)
    return _bars(lr, rng, vol=vol)


def noise_market(n=3000, seed=0, vol=0.01) -> pd.DataFrame:
    """Random walk with clustered volatility and fat tails: nothing to find."""
    rng = np.random.default_rng(seed)
    h = np.zeros(n)
    for t in range(1, n):
        h[t] = 0.97 * h[t - 1] + rng.normal(0, 0.1)
    lr = rng.standard_t(5, n) * vol * np.exp(h) / np.sqrt(5 / 3)
    return _bars(lr, rng, vol=vol)
