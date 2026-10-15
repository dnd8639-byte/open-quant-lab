"""Strategy rules. A rule maps a close-price array and parameters to a signal (+1 long, -1 short, 0 flat)
decided at each bar's close from data up to that close. Add your own in an experiment file:

    CUSTOM_RULES = {"my_rule": (my_function, {"lookback": [10, 20, 50]})}
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


RULES = {
    "breakout": (breakout, {"lookback": [10, 20, 50, 100, 200]}),
    "ma_cross": (ma_cross, {"fast": [5, 10, 20, 50], "slow": [50, 100, 200]}),
    "momentum": (momentum, {"lookback": [5, 20, 60, 120, 250]}),
    "reversal": (reversal, {"lookback": [1, 2, 3, 5]}),
}


def valid(rule: str, params: dict) -> bool:
    return not (rule == "ma_cross" and params["fast"] >= params["slow"])


def signal(rules: dict, close: np.ndarray, genome: dict) -> np.ndarray:
    fn, _ = rules[genome["rule"]]
    s = fn(close, **genome["params"])
    if genome.get("side") == "long":
        s = np.maximum(s, 0)
    elif genome.get("side") == "short":
        s = np.minimum(s, 0)
    return s
