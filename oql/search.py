"""The safe search loop: a fixed number of rounds, every strategy tried is logged, elites survive.

Round 0 tries `pop_init` random strategies. Each later round mutates the best `elite` found so far
(`children` times) and adds `fresh` random ones. The number of rounds is fixed in advance, so there is no
"keep going until something works": the gate reruns this exact loop on shuffled data to measure how good a
best result luck alone produces.
"""
from __future__ import annotations

import copy
import json

import numpy as np

from .engine import Prepared, portfolio
from .gates import random_gate
from .strategies import positions, valid


def random_genome(rng, rules: dict, both_sides: bool, gates: dict | None = None) -> dict:
    rule = str(rng.choice(list(rules)))
    grid = rules[rule][1]
    for _ in range(50):
        params = {k: int(rng.choice(v)) for k, v in grid.items()}
        if valid(rule, params):
            break
    g = dict(rule=rule, params=params, side="both" if both_sides else str(rng.choice(["both", "long", "short"])))
    if gates:                                              # only when the experiment declares gates: old searches unchanged
        g["gate"] = random_gate(rng, gates)
    return g


def mutate(g: dict, rng, rules: dict, both_sides: bool, gates: dict | None = None) -> dict:
    g = copy.deepcopy(g)
    if gates:
        what = rng.choice(["param", "side", "rule", "gate"], p=[0.55, 0.1, 0.15, 0.2])
    else:
        what = rng.choice(["param", "side", "rule"], p=[0.7, 0.1, 0.2])
    if what == "side" and both_sides:
        what = "param"
    if what == "param" and g["params"]:
        k = str(rng.choice(list(g["params"])))
        g["params"][k] = int(rng.choice(rules[g["rule"]][1][k]))
        if not valid(g["rule"], g["params"]):
            return mutate(g, rng, rules, both_sides, gates)
    elif what == "side":
        g["side"] = str(rng.choice(["both", "long", "short"]))
    elif what == "gate":
        g["gate"] = random_gate(rng, gates)
    else:
        keep = {"side": g["side"], **({"gate": g["gate"]} if gates else {})}
        g = random_genome(rng, rules, both_sides, gates) | keep
    return g


def key(g: dict) -> str:
    return json.dumps(g, sort_keys=True)


class Scorer:
    def __init__(self, prep: Prepared, rules: dict, min_trades_per_year: float = 1.0, gate_features=("price",)):
        self.prep, self.rules, self.mtpy, self.gate_features = prep, rules, min_trades_per_year, tuple(gate_features)
        self._sig, self._gate_cache = {}, {}

    def signals(self, g: dict) -> dict:
        out = {}
        for name in self.prep.names:
            k = (name, key(g))
            if k not in self._sig:
                self._sig[k] = positions(self.rules, self.prep.frames[name], g, name, self._gate_cache, self.gate_features)
            out[name] = self._sig[k]
        return out

    def returns(self, g: dict, **kw):
        return portfolio(self.prep, self.signals(g), **kw)

    def score(self, g: dict, **kw) -> float:
        port, _, trades = self.returns(g, **kw)
        years = len(port) / self.prep.ppy
        if years <= 0 or trades / years < self.mtpy * len(self.prep.names):
            return -9.0                                    # almost never changes direction: never selected
        return sharpe(port.to_numpy(), self.prep.ppy)


def sharpe(x, ppy: float) -> float:
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(x.mean() / x.std() * np.sqrt(ppy)) if len(x) > 2 and x.std() > 0 else 0.0


def search(scorer: Scorer, exp, log: list | None = None) -> dict:
    rules = scorer.rules
    gates = exp.gate_table() if hasattr(exp, "gate_table") else None
    rng = np.random.default_rng(exp.seed)
    tried = {}

    def ev(g, rnd):
        k = key(g)
        if k not in tried:
            tried[k] = (scorer.score(g, delay=exp.delay), g)
            if log is not None:
                log.append(dict(round=rnd, score=round(tried[k][0], 4), genome=k))
        return tried[k][0]

    for _ in range(exp.pop_init):
        ev(random_genome(rng, rules, exp.both_sides, gates), 0)
    history = []
    for rnd in range(1, exp.rounds + 1):
        elite = sorted(tried.values(), key=lambda x: -x[0])[:exp.elite]
        for _ in range(exp.children):
            ev(mutate(elite[rng.integers(len(elite))][1], rng, rules, exp.both_sides, gates), rnd)
        for _ in range(exp.fresh):
            ev(random_genome(rng, rules, exp.both_sides, gates), rnd)
        history.append(max(v[0] for v in tried.values()))
    best_score, best = max(tried.values(), key=lambda x: x[0])
    return dict(best=best, best_score=best_score, history=history, n_trials=len(tried),
                trial_scores=[v[0] for v in tried.values() if v[0] > -9])
