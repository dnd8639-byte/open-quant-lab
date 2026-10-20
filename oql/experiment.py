"""An experiment = which data, which strategy space, which costs, and the rules of the test.

It lives in a card's experiment.py and is frozen (fingerprinted) together with the pre-registration, so the
search that the gate tests is exactly the search that was declared in advance:

    from oql.experiment import Experiment
    EXPERIMENT = Experiment(datasets=["databento/ES/1d", "databento/CL/1d"], cost_bps={"default": 1.0})
"""
from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field

import pandas as pd

from . import catalog
from .gates import GATES
from .strategies import ALL_RULES, RULES


@dataclass
class Experiment:
    datasets: list
    rules: list = field(default_factory=lambda: list(RULES))
    grid: dict = field(default_factory=dict)          # per-rule parameter grids that replace the defaults
    cost_bps: dict = field(default_factory=lambda: {"default": 1.0})   # by dataset id, or "default"
    delay: float = 1 / 23                             # fraction of the next bar before the fill
    dev_frac: float = 0.70                            # first 70% of dates = development, last 30% = holdout
    alpha: float = 0.05                               # pass threshold for the permutation tests
    n_perm: int = 99                                  # permutations of the WHOLE search
    both_sides: bool = True                           # only strategies that trade long and short
    rounds: int = 12
    pop_init: int = 24
    children: int = 18
    fresh: int = 6
    elite: int = 6
    min_trades_per_year: float = 1.0                  # direction changes per year, per market (fewer = never selected)
    warmup: int = 250                                 # bars ignored at the start (longest lookback)
    rebalance_band: float = 0.0                       # resize only in steps of (1 + band), e.g. 0.25; 0 = every bar
    start: str | None = None
    end: str | None = None
    seed: int = 0
    gates: list = field(default_factory=lambda: ["none"])   # efficiency gates the search may add: none, er_trailing, er_forecast
    gate_grid: dict = field(default_factory=dict)     # per-gate settings that replace the defaults in oql.gates.GATES
    gate_features: list = field(default_factory=lambda: ["price"])   # feature groups for er_forecast (oql.efficiency.GROUPS)
    custom_rules: dict = field(default_factory=dict)
    path: str = ""

    def rule_table(self) -> dict:
        table = {**ALL_RULES, **self.custom_rules}
        out = {}
        for r in self.rules:
            if r not in table:
                raise KeyError(f"unknown rule {r!r}; known: {sorted(table)}")
            fn, grid = table[r]
            out[r] = (fn, self.grid.get(r, grid))
        return out

    def gate_table(self) -> dict | None:
        """None when the search has no gates (the default: genomes and results are exactly as before gates existed)."""
        if list(self.gates) in ([], ["none"]):
            return None
        out = {}
        for g in self.gates:
            if g != "none" and g not in GATES:
                raise KeyError(f"unknown gate {g!r}; known: none, {', '.join(GATES)}")
            out[g] = {} if g == "none" else self.gate_grid.get(g, GATES[g])
        return out


def load(path: str) -> Experiment:
    spec = importlib.util.spec_from_file_location(f"oql_experiment_{abs(hash(os.path.abspath(path)))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    exp = getattr(mod, "EXPERIMENT", None)
    if not isinstance(exp, Experiment):
        raise ValueError(f"{path} must define EXPERIMENT = Experiment(...)")
    exp.custom_rules = dict(getattr(mod, "CUSTOM_RULES", {}))
    exp.path = os.path.abspath(path)
    return exp


def load_book(exp: Experiment) -> dict:
    book = {}
    for ds in exp.datasets:
        m = catalog.meta(ds)
        if m.kind != "bars":
            raise ValueError(f"{ds} is a table, not price bars")
        book[ds] = catalog.read(ds, exp.start, exp.end)
    return book


def split_date(book: dict, frac: float) -> pd.Timestamp:
    """The date at `frac` of all dates in the book: before it = development, from it on = holdout."""
    days = pd.DatetimeIndex(sorted(set().union(*[set(df.index) for df in book.values()])))
    return days[int(len(days) * frac)]


def truncate(book: dict, end: pd.Timestamp) -> dict:
    return {k: df[df.index < end] for k, df in book.items()}
