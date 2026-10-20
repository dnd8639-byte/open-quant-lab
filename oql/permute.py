"""Price permutations for the permutation tests.

`permute_ohlc` is adapted from neurotrader888/mcpt `bar_permute.py` (MIT License, (c) 2025 neurotrader888)
via mcpt-lab. It keeps the distribution of each bar's moves (gap, high, low, close relative to the open) and
destroys their ORDER, so any edge that depends on order (trends, reversals, patterns) should vanish.

`permute_book` applies ONE shuffle to every market at once, so crisis days stay crisis days everywhere and the
real cross-market correlation survives in the null.

Two nulls (`method`):
  "bars"         the classic shuffle above. It also scrambles WHEN volatility was high, so calm and wild bars end up
                 mixed together; strategies whose risk depends on volatility clustering see a slightly different
                 market in the null than in reality.
  "vol_matched"  every bar is first divided by the market's volatility at its time (a smooth two-sided estimate),
                 the standardized bars are shuffled, and each is multiplied back by the volatility of the slot it
                 lands in. The null keeps the real volatility path (calm periods stay calm, crises stay at their
                 dates) and destroys only the order of the standardized moves.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def smooth_vol(close: np.ndarray, halflife: float = 20.0) -> np.ndarray:
    """Two-sided exponentially weighted volatility of close-to-close log returns: the average of a forward and a
    backward pass, so each bar's value reflects the volatility around it. Used only to build the null."""
    r = np.diff(np.log(np.asarray(close, float)), prepend=np.log(close[0]))
    r2 = pd.Series(r ** 2)
    fwd = r2.ewm(halflife=halflife, min_periods=1).mean().to_numpy()
    bwd = r2[::-1].ewm(halflife=halflife, min_periods=1).mean().to_numpy()[::-1]
    v = np.sqrt(0.5 * (fwd + bwd))
    floor = np.nanmedian(v[v > 0]) * 0.05 if np.any(v > 0) else 1e-8
    return np.maximum(v, floor)


def permute_ohlc(ohlc, start_index=0, seed=None, method="bars"):
    if method not in ("bars", "vol_matched"):
        raise ValueError(f"unknown null {method!r}: bars or vol_matched")
    rng = np.random.default_rng(seed)
    single = not isinstance(ohlc, list)
    mkts = [ohlc] if single else ohlc
    idx0 = mkts[0].index
    for m in mkts:
        assert (m.index == idx0).all(), "indexes do not match"
    n = len(idx0); p0 = start_index + 1; pn = n - p0
    rel = []
    for m in mkts:
        lb = np.log(m[["open", "high", "low", "close"]])
        s = smooth_vol(m["close"].to_numpy(float))[p0:] if method == "vol_matched" else np.ones(pn)
        rel.append(dict(o=(lb["open"] - lb["close"].shift()).to_numpy()[p0:] / s, h=(lb["high"] - lb["open"]).to_numpy()[p0:] / s,
                        l=(lb["low"] - lb["open"]).to_numpy()[p0:] / s, c=(lb["close"] - lb["open"]).to_numpy()[p0:] / s,
                        s=s, start=lb.iloc[start_index].to_numpy(), real=lb.to_numpy()))
    perm1, perm2 = rng.permutation(pn), rng.permutation(pn)          # intrabar moves and gaps shuffled separately
    out = []
    for r in rel:
        s = r["s"]                                                   # each slot keeps its own volatility
        o, h, l, c = r["o"][perm2] * s, r["h"][perm1] * s, r["l"][perm1] * s, r["c"][perm1] * s
        close_path = r["start"][3] + np.cumsum(o + c)
        open_path = np.concatenate([[r["start"][3]], close_path[:-1]]) + o
        b = np.zeros((n, 4)); b[:start_index] = r["real"][:start_index]; b[start_index] = r["start"]
        b[p0:, 0], b[p0:, 1], b[p0:, 2], b[p0:, 3] = open_path, open_path + h, open_path + l, close_path
        out.append(pd.DataFrame(np.exp(b), index=idx0, columns=["open", "high", "low", "close"]))
    return out[0] if single else out


def _fold(df: pd.DataFrame, common: pd.DatetimeIndex) -> pd.DataFrame:
    """Merge a market's own bars onto a shared calendar: a bar missing from `common` joins the next shared bar."""
    pos = common.searchsorted(df.index)
    keep = pos < len(common)
    g = pd.Index(common[pos[keep]], name="ts")
    x = df[keep]
    return pd.DataFrame({"open": x["open"].groupby(g).first(), "high": x["high"].groupby(g).max(),
                         "low": x["low"].groupby(g).min(), "close": x["close"].groupby(g).last(),
                         "volume": x["volume"].groupby(g).sum()})


def permute_book(book: dict, seed: int, from_date=None, method: str = "bars") -> dict:
    """Joint permutation of every market. Bars before `from_date` stay real (used for the holdout test)."""
    start = min(df.index[0] for df in book.values())
    early = [k for k, df in book.items() if df.index[0] <= start + pd.Timedelta(days=7)]
    common = None
    for k in early:
        common = book[k].index if common is None else common.intersection(book[k].index)
    out = {}
    folded = {k: _fold(book[k], common) for k in early}
    s0 = 0 if from_date is None else max(0, int(common.searchsorted(from_date)) - 1)
    perm = permute_ohlc([f[["open", "high", "low", "close"]] for f in folded.values()], start_index=s0, seed=seed, method=method)
    for (k, f), p in zip(folded.items(), perm):
        p["volume"] = f["volume"].values
        out[k] = p
    for i, k in enumerate(x for x in book if x not in early):         # late starters: shuffled alone
        f = _fold(book[k], common)
        s = 0 if from_date is None else max(0, int(f.index.searchsorted(from_date)) - 1)
        p = permute_ohlc(f[["open", "high", "low", "close"]], start_index=s, seed=seed * 1000 + i + 1, method=method)
        p["volume"] = f["volume"].values
        out[k] = p
    return {k: out[k] for k in book}
