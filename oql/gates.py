"""Efficiency gates: let a strategy trade only when the market is (expected to be) moving in straight lines, or
only when it is expected to chop back and forth.

A gate is part of the strategy: it is computed from the same bars the strategy trades, decided at each bar's close
from data up to that close, and multiplies the rule's position by 1 (on) or 0 (off). Because the search computes
it from the bars themselves, the gate test reruns it on every shuffled copy too, so a gate that only helps by luck
is caught like any other choice. Switching a gate on and off is a position change and pays trading costs.

  none          always on
  er_trailing   efficiency of the last `lookback` bars, |close - close L bars ago| / sum of |bar moves|, compared
                with its own recent history (rolling percentile over max(500, 10 x lookback) bars)
  er_forecast   the efficiency lab's walk-forward forecast of the coming day's efficiency (hourly or daily bars) or
                hour's (minute bars), refitted every year (month) on everything before, from price features plus the
                experiment's `gate_features` groups; the first year (three months) has no forecast, so the gate is off

`side` "high" trades only when the value is in the top `share` percent; "low" only in the bottom `share` percent.

Non-price features (funding, basis, macro, positioning) keep their real dates when the bars are shuffled, so in
the null they lose their link to the market, which is what the test needs: luck with them is measured too.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

GATES = {
    "er_trailing": {"lookback": [12, 24, 72, 168], "side": ["high", "low"], "share": [33, 50]},
    "er_forecast": {"side": ["high", "low"], "share": [33, 50]},
}
NONE = {"kind": "none"}


def describe(gate: dict | None) -> str:
    g = gate or NONE
    if g["kind"] == "none":
        return "no gate"
    where = f"{'top' if g['side'] == 'high' else 'bottom'} {g['share']}%"
    if g["kind"] == "er_trailing":
        return f"on when the last {g['lookback']} bars' efficiency is in its {where}"
    return f"on when the forecast of the coming period's efficiency is in its {where}"


def trailing_er(close: np.ndarray, lookback: int) -> np.ndarray:
    c = pd.Series(close, dtype=float)
    return ((c - c.shift(lookback)).abs() / c.diff().abs().rolling(lookback).sum()).to_numpy()


def _on(ranks: np.ndarray, side: str, share: int) -> np.ndarray:
    v = np.asarray(ranks, float)
    on = v >= 1 - share / 100 if side == "high" else v <= share / 100
    return np.where(np.isfinite(v) & on, 1.0, 0.0)


def rolling_rank(x: np.ndarray, window: int) -> np.ndarray:
    """Share of the last `window` values (including this one) that are <= this one. Causal."""
    s = pd.Series(x, dtype=float)
    return s.rolling(window, min_periods=window // 2).rank(pct=True).to_numpy()


def forecast_by_bar(frame: pd.DataFrame, ds: str, groups: list) -> np.ndarray:
    """Forecast rank (0-1) known at each bar's close for the period the NEXT bar belongs to."""
    from . import efficiency as E
    per, horizon = E.periods_from_bars(frame)
    X, _ = E.features(ds, per, horizon, ["price"] + [g for g in groups if g != "price"])
    rank = E.forecast_rank(per, X, horizon)
    period = "1h" if horizon == "hourly" else "1D"
    idx = frame.index
    nxt = idx[1:].append(pd.DatetimeIndex([idx[-1] + E.bar_step(idx)])) if len(idx) else idx
    return rank.reindex(nxt.floor(period)).to_numpy()


def mask(gate: dict | None, frame: pd.DataFrame, ds: str, cache: dict | None = None, groups=("price",)) -> np.ndarray | None:
    """1/0 per bar (None = no gate). `cache` holds per-market work shared by every gate of the same kind."""
    g = gate or NONE
    if g["kind"] == "none":
        return None
    cache = {} if cache is None else cache
    if g["kind"] == "er_trailing":
        L = int(g["lookback"])
        k = ("trail", ds, L)
        if k not in cache:
            cache[k] = rolling_rank(trailing_er(frame["close"].to_numpy(float), L), max(500, 10 * L))
        return _on(cache[k], g["side"], int(g["share"]))
    if g["kind"] == "er_forecast":
        k = ("fc", ds, tuple(groups))
        if k not in cache:
            cache[k] = forecast_by_bar(frame, ds, list(groups))
        return _on(cache[k], g["side"], int(g["share"]))
    raise KeyError(f"unknown gate {g['kind']!r}; known: none, {', '.join(GATES)}")


def validate(gate: dict, grid: dict | None = None) -> dict:
    """Check a gate given by a person or an assistant; returns it normalized."""
    if not gate or gate.get("kind", "none") == "none":
        return dict(NONE)
    kind = gate["kind"]
    table = grid or GATES
    if kind not in table:
        raise ValueError(f"unknown gate {kind!r}; known: none, {', '.join(table)}")
    want = set(table[kind])
    got = {k: v for k, v in gate.items() if k != "kind"}
    if set(got) != want:
        raise ValueError(f"gate {kind} needs exactly {sorted(want)}")
    out = {"kind": kind}
    for k, v in got.items():
        out[k] = str(v) if k == "side" else int(v)
    if out["side"] not in ("high", "low") or not 1 <= out["share"] <= 99:
        raise ValueError("side must be 'high' or 'low' and share a percent between 1 and 99")
    return out


def random_gate(rng, table: dict) -> dict:
    kind = str(rng.choice(list(table)))
    if kind == "none":
        return dict(NONE)
    return {"kind": kind, **{k: (str(rng.choice(v)) if k == "side" else int(rng.choice(v))) for k, v in table[kind].items()}}


def key(gate: dict | None) -> str:
    return json.dumps(gate or NONE, sort_keys=True)
