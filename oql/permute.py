"""Price permutations for the permutation tests.

`permute_ohlc` is adapted from neurotrader888/mcpt `bar_permute.py` (MIT License, (c) 2025 neurotrader888)
via mcpt-lab. It keeps the distribution of each bar's moves (gap, high, low, close relative to the open) and
destroys their ORDER, so any edge that depends on order (trends, reversals, patterns) should vanish.

`permute_book` applies ONE shuffle to every market at once, so crisis days stay crisis days everywhere and the
real cross-market correlation survives in the null.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def permute_ohlc(ohlc, start_index=0, seed=None):
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
        rel.append(dict(o=(lb["open"] - lb["close"].shift()).to_numpy()[p0:], h=(lb["high"] - lb["open"]).to_numpy()[p0:],
                        l=(lb["low"] - lb["open"]).to_numpy()[p0:], c=(lb["close"] - lb["open"]).to_numpy()[p0:],
                        start=lb.iloc[start_index].to_numpy(), real=lb.to_numpy()))
    perm1, perm2 = rng.permutation(pn), rng.permutation(pn)          # intrabar moves and gaps shuffled separately
    out = []
    for r in rel:
        o, h, l, c = r["o"][perm2], r["h"][perm1], r["l"][perm1], r["c"][perm1]
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


def permute_book(book: dict, seed: int, from_date=None) -> dict:
    """Joint permutation of every market. Bars before `from_date` stay real (used for the holdout test)."""
    start = min(df.index[0] for df in book.values())
    early = [k for k, df in book.items() if df.index[0] <= start + pd.Timedelta(days=7)]
    common = None
    for k in early:
        common = book[k].index if common is None else common.intersection(book[k].index)
    out = {}
    folded = {k: _fold(book[k], common) for k in early}
    s0 = 0 if from_date is None else max(0, int(common.searchsorted(from_date)) - 1)
    perm = permute_ohlc([f[["open", "high", "low", "close"]] for f in folded.values()], start_index=s0, seed=seed)
    for (k, f), p in zip(folded.items(), perm):
        p["volume"] = f["volume"].values
        out[k] = p
    for i, k in enumerate(x for x in book if x not in early):         # late starters: shuffled alone
        f = _fold(book[k], common)
        s = 0 if from_date is None else max(0, int(f.index.searchsorted(from_date)) - 1)
        p = permute_ohlc(f[["open", "high", "low", "close"]], start_index=s, seed=seed * 1000 + i + 1)
        p["volume"] = f["volume"].values
        out[k] = p
    return {k: out[k] for k in book}
