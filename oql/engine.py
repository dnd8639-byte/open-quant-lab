"""Bar engine: turns per-bar target positions into net returns, with trading costs and a fill delay.

No look-ahead, by construction:
  * a position is decided at the CLOSE of bar t, using data up to and including bar t;
  * it is filled a fraction `delay` of the way into bar t+1 (default 1/23: one hour of a 23-hour day);
    the old position earns the gap and the first `delay` of bar t+1, the new one earns the rest;
  * every change in position pays `cost_bps` per unit (half-spread + commission + slippage).
Each market is sized to the same expected volatility (trailing 20-bar standard deviation), so a portfolio of
many markets is not dominated by the wildest one. The portfolio return is the average of the markets trading.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TARGET_ANNUAL_VOL = 0.15
MAX_LEVERAGE = 10.0


def bars_per_year(index: pd.DatetimeIndex) -> float:
    """Median number of bars in the complete calendar years of the sample (365 x 24 for crypto hours,
    ~252 for futures days). Falls back to the bar spacing when there is no complete year."""
    years = pd.Series(1, index=index).groupby(index.year).size()
    full = years.iloc[1:-1] if len(years) > 2 else pd.Series(dtype=float)
    if len(full):
        return float(full.median())
    step = pd.Series(index).diff().median()
    return float(pd.Timedelta(days=365) / step) if pd.notna(step) and step > pd.Timedelta(0) else 252.0


@dataclass
class Prepared:
    """Per-market numpy arrays the scorer needs, built once per dataset (and once per permutation)."""
    names: list
    index: dict
    arrays: dict
    ppy: float


def prepare(book: dict, cost_bps: dict, warmup: int = 250, score_from=None, score_to=None, sd_window: int = 20,
            ppy: float | None = None) -> Prepared:
    """`ppy` (bars per year) is measured from the data unless given; paper trading passes the backtest's value so
    a short live sample sizes positions exactly as the backtest did."""
    arrays, index = {}, {}
    ppy = ppy or np.median([bars_per_year(df.index) for df in book.values()])
    per_bar = TARGET_ANNUAL_VOL / np.sqrt(ppy)
    for name, df in book.items():
        c = df["close"].to_numpy(float)
        lc = np.log(c)
        ret = np.diff(lc, prepend=lc[0])
        intra = np.log(df["close"] / df["open"]).to_numpy(float)
        sd = pd.Series(ret).rolling(sd_window).std().to_numpy()
        size = np.clip(per_bar / sd, 0, MAX_LEVERAGE)
        win = np.zeros(len(df), bool)
        win[warmup:] = True
        if score_from is not None:
            win &= (df.index >= score_from)
        if score_to is not None:
            win &= (df.index < score_to)
        arrays[name] = dict(close=c, ret=ret, intra=intra, gap=ret - intra, size=size, win=win,
                            cost=cost_bps.get(name, cost_bps.get("default", 1.0)) / 1e4)
        index[name] = df.index
    return Prepared(list(book), index, arrays, float(ppy))


def market_returns(a: dict, signal: np.ndarray, cost_mult: float = 1.0, delay: float = 1 / 23):
    """Net return per bar of following `signal` (+1/-1/0, decided at each bar's close) in one market."""
    pos = np.where(a["win"] & np.isfinite(a["size"]), signal * a["size"], 0.0)
    held = np.concatenate([[0.0], pos[:-1]])            # in force during bar t
    prev = np.concatenate([[0.0], held[:-1]])           # in force during bar t-1
    pnl = prev * (a["gap"] + delay * a["intra"]) + held * (1 - delay) * a["intra"] - np.abs(held - prev) * a["cost"] * cost_mult
    live = np.concatenate([[False], a["win"][:-1]])
    # a trade = a change of direction (enter, exit or flip). Resizing to hold volatility steady is paid for in
    # costs above but is not a new trade; counting it made every bar a "trade" and the minimum-trades rule a no-op.
    trades = int((np.diff(np.sign(pos[a["win"]]), prepend=0.0) != 0).sum())
    return np.where(live, pnl, np.nan), trades


def portfolio(prep: Prepared, signals: dict, **kw):
    cols, trades = {}, 0
    for name in prep.names:
        x, n = market_returns(prep.arrays[name], signals[name], **kw)
        cols[name] = pd.Series(x, index=prep.index[name])
        trades += n
    R = pd.DataFrame(cols)
    R = R[R.notna().any(axis=1)]
    return R.mean(axis=1), R, trades
