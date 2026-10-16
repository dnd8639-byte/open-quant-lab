"""Strategy rules. A rule maps a close-price array and parameters to a signal (+1 long, -1 short, 0 flat)
decided at each bar's close from data up to that close. Add your own in an experiment file:

    CUSTOM_RULES = {"my_rule": (my_function, {"lookback": [10, 20, 50]})}

A rule that needs the bar times (e.g. "the day's first two hours") sets `fn.needs_index = True` and receives
`index=`. A strategy = rule + parameters + side (+ an optional efficiency gate, see oql.gates).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def breakout(c, lookback):
    s = pd.Series(c)
    hi, lo = s.rolling(lookback).max().shift(1), s.rolling(lookback).min().shift(1)
    return pd.Series(np.where(s > hi, 1.0, np.where(s < lo, -1.0, np.nan))).ffill().fillna(0.0).to_numpy()


def ma_cross(c, fast, slow):
    s = pd.Series(c)
    return np.sign(s.rolling(fast).mean() - s.rolling(slow).mean()).fillna(0.0).to_numpy()


def momentum(c, lookback):
    s = pd.Series(c)
    return np.sign(s - s.shift(lookback)).fillna(0.0).to_numpy()


def reversal(c, lookback):
    return -momentum(c, lookback)


def orb(c, hours, index=None):
    """Opening-range breakout on intraday bars: at the close of the bar that ends `hours` hours into the UTC day,
    take the direction of the move since the day began (the last close before the day) and hold it to the day's
    last bar. Bar times decide everything, so a missing bar skips that day and a day in progress works the same."""
    idx = pd.DatetimeIndex(index)
    if len(idx) < 3:
        return np.zeros(len(c))
    step = pd.Series(idx).diff().median()
    day = idx.floor("1D")
    elapsed = ((idx + step) - day) / pd.Timedelta(hours=1)               # hours into the day at each bar's close
    s = pd.Series(np.asarray(c, float))
    codes = pd.Series(pd.factorize(day)[0])
    first = pd.Series(idx == idx).groupby(codes).cumcount().to_numpy() == 0
    ref = pd.Series(np.where(first, s.shift(1), np.nan)).groupby(codes).ffill().to_numpy()   # close before the day
    decide = np.isclose(elapsed, hours)
    d = pd.Series(np.where(decide, np.sign(s.to_numpy() - ref), np.nan)).groupby(codes).ffill().to_numpy()
    last = (idx + step).floor("1D") != day
    return np.nan_to_num(np.where((elapsed >= hours - 1e-9) & ~last, d, 0.0))


orb.needs_index = True

RULES = {
    "breakout": (breakout, {"lookback": [10, 20, 50, 100, 200]}),
    "ma_cross": (ma_cross, {"fast": [5, 10, 20, 50], "slow": [50, 100, 200]}),
    "momentum": (momentum, {"lookback": [5, 20, 60, 120, 250]}),
    "reversal": (reversal, {"lookback": [1, 2, 3, 5]}),
}


EXTRA_RULES = {                     # available by name; not in the default search (keeps old experiments reproducible)
    "orb": (orb, {"hours": [1, 2, 4, 8]}),
}
ALL_RULES = {**RULES, **EXTRA_RULES}


def valid(rule: str, params: dict) -> bool:
    return not (rule == "ma_cross" and params["fast"] >= params["slow"])


def signal(rules: dict, close: np.ndarray, genome: dict, index=None) -> np.ndarray:
    """The rule's signal with its side applied. No gate: see positions()."""
    fn, _ = rules[genome["rule"]]
    if getattr(fn, "needs_index", False):
        if index is None:
            raise ValueError(f"rule {genome['rule']} needs bar times")
        s = fn(close, **genome["params"], index=index)
    else:
        s = fn(close, **genome["params"])
    if genome.get("side") == "long":
        s = np.maximum(s, 0)
    elif genome.get("side") == "short":
        s = np.minimum(s, 0)
    return s


def positions(rules: dict, frame: pd.DataFrame, genome: dict, ds: str = "", cache: dict | None = None,
              gate_features=("price",)) -> np.ndarray:
    """Everything a strategy decides at each bar's close: rule signal x side x efficiency gate (if any)."""
    from .gates import mask
    s = signal(rules, frame["close"].to_numpy(float), genome, index=frame.index)
    m = mask(genome.get("gate"), frame, ds, cache, tuple(gate_features))
    return s if m is None else s * m
