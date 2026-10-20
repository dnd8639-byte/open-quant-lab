"""What an AI assistant is allowed to do in the lab, and nothing more.

The agent works like a research assistant: it can read everything, propose ideas, draft pre-registrations and
experiments, and explore DEVELOPMENT data. It cannot freeze a card, run the gate or the holdout, deploy paper
trading, or touch a holdout's dates. Those are commitments, and they stay with a human.

Two design rules make this safe:
  1. Structured inputs only. The agent never writes Python: experiment.py is rendered from validated fields, and
     the pre-registration's pass rule is fixed text the agent cannot edit.
  2. Every exploration is counted. Each strategy the agent evaluates is logged to the card and the ledger, and the
     gate report shows how many exploratory evaluations happened before the freeze, so an AI's trial and error is
     visible to anyone judging the result.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from . import board, catalog, ledger
from .engine import prepare
from .experiment import Experiment, load as load_exp, split_date
from .permute import permute_book
from .search import Scorer, search, sharpe
from .gates import GATES, validate as validate_gate
from .strategies import ALL_RULES, RULES, valid

EDITABLE = ("idea", "research")
MAX_PERM = 199


class AgentError(RuntimeError):
    pass


def _card(cid):
    return board.get(board.load(), cid)


def _log_exploration(card, entry):
    d = board.card_dir(card)
    with open(os.path.join(d, "explorations.jsonl"), "a") as f:
        f.write(json.dumps({"utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), **entry}) + "\n")
    check = entry.get("kind") == "permutation_check"
    ledger.append(board.lab_dir(), card=card["id"], family=card["family"], kind="explore", verdict="-",
                  p=round(entry["p"], 4) if check else "", sharpe=round(entry.get("sharpe", float("nan")), 3), n_trials=1,
                  finalist=entry.get("genome", ""), data=entry.get("dataset", ""),
                  note=f"by {entry.get('actor', 'agent')}" + (" · single-strategy permutation check" if check else "")
                       + f" · {entry.get('cost_bps', 1.0):g} bp")


def explorations(card) -> list[dict]:
    p = os.path.join(board.card_dir(card), "explorations.jsonl")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def dev_cutoff(card, ds: str) -> pd.Timestamp:
    """Latest date an exploration may see: the dataset's own 70% point, and never past the split date of the
    card's experiment (which is set by ALL its datasets), so exploring can never reach into a holdout."""
    df = catalog.read(ds)
    cut = df.index[int(len(df) * 0.7)]
    exp_path = os.path.join(board.card_dir(card), "experiment.py")
    try:
        exp = load_exp(exp_path)
        if exp.datasets and all(catalog.meta(x).kind == "bars" for x in exp.datasets):
            book = {x: catalog.read(x) for x in exp.datasets}
            cut = min(cut, split_date(book, exp.dev_frac))
    except Exception:
        pass
    return cut


# ---------------------------------------------------------------------------------------------- read
def lab_overview() -> dict:
    bd = board.load()
    rows = ledger.rows(board.lab_dir())
    return {"cards": [{"id": c["id"], "title": c["title"], "stage": c["stage"],
                       "gate": c["results"].get("gate", {}).get("verdict"), "review": c.get("review", {}).get("summary")}
                      for c in bd["cards"]],
            "tests_in_ledger": sum(r["kind"] in ("gate", "holdout") for r in rows),
            "explorations_in_ledger": sum(int(r["n_trials"] or 1) for r in rows if r["kind"] == "explore"),
            "datasets": [m.id for m in catalog.list_all()],
            "rules": {k: g for k, (_, g) in ALL_RULES.items()},
            "rules_note": "orb (opening-range breakout) needs intraday bars and is not in the default search; name it in rules to use it",
            "gates": {"none": "always on", **{k: g for k, g in GATES.items()}},
            "gates_note": ("An efficiency gate switches a strategy on only when the market is (expected to be) trending ('high') or "
                           "choppy ('low'). er_trailing uses the last `lookback` bars; er_forecast uses the efficiency lab's walk-forward "
                           "forecast. Declare gates in write_experiment to let the search use them; the gate test reruns them on shuffled data."),
            "stages": board.STAGES + ["archive"],
            "agent_may": ["propose_idea", "start_research", "write_preregistration", "write_experiment", "explore",
                          "permutation_check", "efficiency_study", "request_review"],
            "human_only": ["freeze", "gate", "holdout", "paper deploy"]}


def list_datasets() -> list[dict]:
    return [{"id": m.id, "kind": m.kind, "license": m.license, "rows": m.rows, "start": m.start[:10], "end": m.end[:10],
             "notes": m.notes[:200]} for m in catalog.list_all()]


def dataset_summary(dataset: str) -> dict:
    m = catalog.meta(dataset)
    rep = catalog.check(dataset)
    out = {"id": m.id, "kind": m.kind, "license": m.license, "rows": m.rows, "start": m.start, "end": m.end,
           "columns": m.columns, "issues": rep["issues"], "warnings": rep.get("warnings", []), "notes": m.notes}
    if m.kind == "bars":
        df = catalog.read(dataset)
        r = np.log(df["close"]).diff().dropna()
        out.update(bar_spacing=str(pd.Series(df.index).diff().median()), daily_vol_estimate=float(r.std()))
    return out


def get_card(card_id: str) -> dict:
    c = _card(card_id)
    d = board.card_dir(c)
    out = {"card": c, "explorations": len(explorations(c))}
    for name in ("PREREGISTRATION.md", "experiment.py", "gate.md"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            with open(p) as f:
                out[name] = f.read()
    return out


# ---------------------------------------------------------------------------------------------- write (idea/research only)
def propose_idea(title: str, mechanism: str, hypothesis: str) -> dict:
    if len(mechanism.strip()) < 40:
        raise AgentError("explain the mechanism: why should this make money, and who is on the other side? (40+ characters)")
    c = board.new_card(title, mechanism)
    bd = board.load()
    card = board.get(bd, c["id"])
    card["author"] = "agent"
    board.save(bd)
    write_preregistration(c["id"], hypothesis=hypothesis, mechanism=mechanism)
    return {"created": c["id"], "stage": "idea", "next": "start_research, then write_experiment and explore"}


def start_research(card_id: str) -> dict:
    return {"stage": board.start(card_id)["stage"]}


PREREG = """# Pre-registration: {title}

Card {id} · family {family} · drafted {utc} UTC{agent}

## Hypothesis
{hypothesis}

## Mechanism
{mechanism}

## Data
{data}

## Strategy space
{space}

## Test and pass rule
The gate reruns the whole search on `n_perm` shuffled copies of the development data (set in experiment.py).
PASS if p < `alpha` and the cost/delay Monte Carlo median Sharpe > 0. Otherwise the card is archived.
The holdout runs once. PASS if holdout Sharpe > 0 and its permutation p < `alpha`.
Exploratory evaluations made before the freeze are counted and reported with the result.

## What would change my mind
{falsify}
"""


def write_preregistration(card_id: str, hypothesis: str = "TODO", mechanism: str | None = None, data: str = "TODO",
                          strategy_space: str = "TODO", what_would_change_my_mind: str = "TODO") -> dict:
    c = _card(card_id)
    if c["stage"] not in EDITABLE:
        raise AgentError(f"card is '{c['stage']}': pre-registrations can only be edited before the freeze")
    text = PREREG.format(title=c["title"], id=c["id"], family=c["family"], utc=time.strftime("%Y-%m-%d %H:%M", time.gmtime()),
                         agent=" · drafted by an AI assistant; frozen only by a person" if c.get("author") == "agent" else "",
                         hypothesis=hypothesis.strip(), mechanism=(mechanism or c["mechanism"]).strip(), data=data.strip(),
                         space=strategy_space.strip(), falsify=what_would_change_my_mind.strip())
    with open(os.path.join(board.card_dir(c), "PREREGISTRATION.md"), "w") as f:
        f.write(text)
    stale = _mark_review_outdated(c["id"], "PREREGISTRATION.md")
    return {"written": "PREREGISTRATION.md", "sections_still_TODO": text.count("TODO"),
            **({"review": stale} if stale else {})}


def _mark_review_outdated(card_id: str, what: str) -> str | None:
    """A review summary describes the card as it was. Editing the card afterwards marks the summary as out of date,
    so the person reviewing knows to read the files, not the summary, or to ask for a fresh one."""
    bd = board.load()
    c = board.get(bd, card_id)
    if not c.get("review"):
        return None
    c["review"]["edited_after"] = f"{what} changed {time.strftime('%Y-%m-%d %H:%M', time.gmtime())} UTC"
    board.save(bd)
    return "your earlier review summary is now out of date; call request_review again with a corrected summary"


EXPERIMENT = '''"""Experiment for card {id}: {title}. Rendered from structured fields{agent}. Frozen with PREREGISTRATION.md."""
from oql.experiment import Experiment

EXPERIMENT = Experiment(
    datasets={datasets!r},
    rules={rules!r},
    grid={grid!r},
    cost_bps={cost!r},
    alpha={alpha!r},
    n_perm={n_perm!r},{gates}{band}{nulls}
)
'''


def write_experiment(card_id: str, datasets: list, rules: list | None = None, grid: dict | None = None,
                     cost_bps: dict | None = None, alpha: float = 0.05, n_perm: int = 99, gates: list | None = None,
                     gate_features: list | None = None, rebalance_band: float = 0.0, nulls: list | None = None) -> dict:
    c = _card(card_id)
    if c["stage"] not in EDITABLE:
        raise AgentError(f"card is '{c['stage']}': experiments can only be edited before the freeze")
    rules = rules or list(RULES)
    for ds in datasets:
        if catalog.meta(ds).kind != "bars":
            raise AgentError(f"{ds} is a table, not price bars")
    bad = [r for r in rules if r not in ALL_RULES]
    if bad:
        raise AgentError(f"unknown rules {bad}; library rules: {sorted(ALL_RULES)} (custom code rules need a person)")
    grid = {r: {k: [int(v) for v in vals] for k, vals in g.items()} for r, g in (grid or {}).items()}
    for r, g in grid.items():
        if r not in rules or set(g) != set(ALL_RULES[r][1]):
            raise AgentError(f"grid for {r} must give exactly the parameters {sorted(ALL_RULES[r][1])}")
    from .efficiency import GROUPS
    gates = list(gates or ["none"])
    if [g for g in gates if g != "none" and g not in GATES]:
        raise AgentError(f"gates must come from: none, {', '.join(GATES)}")
    gate_features = list(gate_features or ["price"])
    if [g for g in gate_features if g not in GROUPS]:
        raise AgentError(f"gate_features must come from: {', '.join(GROUPS)}")
    if "price" not in gate_features:
        gate_features = ["price"] + gate_features
    if not 0 <= float(rebalance_band) <= 0.5:
        raise AgentError("rebalance_band must be between 0 (resize every bar) and 0.5")
    band_text = f"\n    rebalance_band={float(rebalance_band)!r}," if rebalance_band else ""
    nulls = list(nulls or ["bars"])
    if not nulls or [m for m in nulls if m not in ("bars", "vol_matched")]:
        raise AgentError("nulls must be a list of 'bars' and/or 'vol_matched'")
    nulls_text = "" if nulls == ["bars"] else f"\n    nulls={nulls!r},"
    gate_text = "" if gates == ["none"] else f"\n    gates={gates!r},\n    gate_features={gate_features!r},"
    cost_bps = {k: float(v) for k, v in (cost_bps or {"default": 1.0}).items()}
    if not 0.001 <= alpha <= 0.1:
        raise AgentError("alpha must be between 0.001 and 0.1")
    if not 39 <= int(n_perm) <= 999:
        raise AgentError("n_perm must be between 39 and 999 (fewer cannot reach p < 0.05)")
    text = EXPERIMENT.format(id=c["id"], title=c["title"], datasets=list(datasets), rules=list(rules), grid=grid,
                             cost=cost_bps, alpha=float(alpha), n_perm=int(n_perm), gates=gate_text, band=band_text, nulls=nulls_text,
                             agent=" by an AI assistant" if c.get("author") == "agent" else "")
    path = os.path.join(board.card_dir(c), "experiment.py")
    with open(path, "w") as f:
        f.write(text)
    exp = load_exp(path)                                   # proves it loads
    exp.rule_table()
    n, k = _space_size(exp), len(exp.datasets)
    stale = _mark_review_outdated(c["id"], "experiment.py")
    return {"written": "experiment.py", "datasets": exp.datasets, "strategies_in_search_space": n,
            "how_the_search_works": (f"Each of the {n} strategies trades all {k} datasets together as one equal-risk portfolio, "
                                     f"and the gate picks the best portfolio. The count is {n}, not {n} x {k}.") if k > 1 else
                                    f"Each of the {n} strategies trades {exp.datasets[0]}; the gate picks the best one.",
            **({"review": stale} if stale else {})}


def _space_size(exp: Experiment) -> int:
    import itertools
    n = 0
    for r, (_, g) in exp.rule_table().items():
        for combo in itertools.product(*g.values()):
            n += valid(r, dict(zip(g.keys(), combo)))
    gates = exp.gate_table()
    k = 1 if gates is None else sum(1 if not g else int(np.prod([len(v) for v in g.values()])) for g in gates.values())
    return n * k * (1 if exp.both_sides else 3)


# ---------------------------------------------------------------------------------------------- explore (development data only)
def _card_costs(c, ds) -> tuple[float, float]:
    """The cost (bps) and fill delay the card's experiment declares for this dataset, so exploration sees what the
    gate will see. Before an experiment is written: 1 bp and the default delay."""
    path = os.path.join(board.card_dir(c), "experiment.py")
    if os.path.exists(path):
        try:
            exp = load_exp(path)
            return float(exp.cost_bps.get(ds, exp.cost_bps.get("default", 1.0))), float(exp.delay)
        except Exception:                                   # an unfinished template: fall back to the defaults
            pass
    return 1.0, 1 / 23


def _dev_scorer(book, ds, warmup, cost_bps=1.0, gate_features=("price",), band=0.0):
    exp = Experiment(datasets=[ds])
    return Scorer(prepare(book, {"default": cost_bps}, warmup=warmup, band=band), exp.rule_table(), 0.0, gate_features), exp


def _card_band(c) -> float:
    path = os.path.join(board.card_dir(c), "experiment.py")
    try:
        return float(load_exp(path).rebalance_band) if os.path.exists(path) else 0.0
    except Exception:
        return 0.0


def _card_gate_features(c) -> list:
    path = os.path.join(board.card_dir(c), "experiment.py")
    try:
        return list(load_exp(path).gate_features) if os.path.exists(path) else ["price"]
    except Exception:
        return ["price"]


def _genome(rule, params, side, gate):
    if rule not in ALL_RULES or not valid(rule, params) or set(params) != set(ALL_RULES[rule][1]):
        raise AgentError(f"{rule} needs parameters {sorted(ALL_RULES[rule][1]) if rule in ALL_RULES else '?'} (fast < slow for ma_cross)")
    g = {"rule": rule, "params": {k: int(v) for k, v in params.items()}, "side": side}
    if gate and gate.get("kind", "none") != "none":
        try:
            g["gate"] = validate_gate(gate)
        except ValueError as e:
            raise AgentError(str(e))
    return g


def explore(card_id: str, dataset: str, rule: str, params: dict, side: str = "both", gate: dict | None = None,
            actor: str = "agent") -> dict:
    """Score ONE strategy on the development part of one dataset. Logged and counted."""
    c = _card(card_id)
    if c["stage"] != "research":
        raise AgentError(f"exploration happens in 'research'; this card is in '{c['stage']}'")
    g = _genome(rule, params, side, gate)
    cut = dev_cutoff(c, dataset)
    dev = catalog.read(dataset)
    dev = dev[dev.index < cut]
    cost, delay = _card_costs(c, dataset)
    sc, exp = _dev_scorer({dataset: dev}, dataset, min(250, max(20, len(dev) // 10)), cost, _card_gate_features(c), _card_band(c))
    port, _, trades = sc.returns(g, delay=delay)
    s = sharpe(port.to_numpy(), sc.prep.ppy)
    entry = dict(actor=actor, kind="explore", dataset=dataset, genome=json.dumps(g, sort_keys=True), sharpe=s, trades=trades,
                 cost_bps=cost)
    _log_exploration(c, entry)
    n = len(explorations(c))
    years = len(port) / sc.prep.ppy
    on = None
    if "gate" in g:
        from .gates import mask
        on = float(np.nanmean(mask(g["gate"], dev, dataset, sc._gate_cache, sc.gate_features)))
    return {"sharpe_dev": s, "cost_bps": cost, "direction_changes": trades,
            **({"gate": g["gate"], "share_of_time_gate_is_on": round(on, 3)} if on is not None else {}),
            "direction_changes_per_year": round(trades / years, 1) if years > 0 else None,
            "dev_data": f"{str(dev.index[0])[:10]} to {str(dev.index[-1])[:10]}",
            "explorations_on_this_card": n,
            "reminder": f"This is exploration #{n} on this card. Every one is reported next to the gate result; "
                        "the more strategies you look at before freezing, the higher the real bar.",
            **({"scope": f"This scored {dataset} alone. The gate scores each strategy on all {len(others)} datasets of the "
                         "experiment together, as one portfolio, so a single-market result is only indirect evidence."}
               if len(others := _card_datasets(c)) > 1 else {})}


def _card_datasets(c) -> list:
    path = os.path.join(board.card_dir(c), "experiment.py")
    try:
        return list(load_exp(path).datasets) if os.path.exists(path) else []
    except Exception:
        return []


def permutation_check(card_id: str, dataset: str, rule: str, params: dict, side: str = "both", n: int = 99,
                      gate: dict | None = None, actor: str = "agent") -> dict:
    """A quick permutation test of ONE strategy on development data. Exploratory: it is logged and does not
    replace the gate, which tests the whole declared search."""
    n = int(min(max(n, 19), MAX_PERM))
    c = _card(card_id)
    if c["stage"] != "research":
        raise AgentError(f"exploration happens in 'research'; this card is in '{c['stage']}'")
    g = _genome(rule, params, side, gate)
    cut = dev_cutoff(c, dataset)
    dev = catalog.read(dataset)
    dev = dev[dev.index < cut]
    warm = min(250, max(20, len(dev) // 10))
    cost, delay = _card_costs(c, dataset)
    feats, band = _card_gate_features(c), _card_band(c)

    def one(book):
        sc, exp = _dev_scorer(book, dataset, warm, cost, feats, band)
        return sharpe(sc.returns(g, delay=delay)[0].to_numpy(), sc.prep.ppy)
    real = one({dataset: dev})
    null = np.array([one(permute_book({dataset: dev}, 7000 + k)) for k in range(n)])
    p = float((1 + (null >= real).sum()) / (n + 1))
    _log_exploration(c, dict(actor=actor, kind="permutation_check", dataset=dataset, genome=json.dumps(g, sort_keys=True),
                             sharpe=real, p=p, n=n, cost_bps=cost))
    return {"sharpe_dev": real, "cost_bps": cost, "p_single_strategy": p, "permutations": n, "null_median": float(np.median(null)),
            "explorations_on_this_card": len(explorations(c)),
            "warning": "p for ONE strategy you chose after looking is optimistic; only the gate's whole-search test counts."}


def efficiency_study(dataset: str, groups: list | None = None) -> dict:
    """Can the next period's efficiency be forecast? Walk-forward ridge on the dataset's development data. Logged."""
    from . import efficiency as E
    groups = ["price"] + [g for g in (groups or []) if g in E.GROUPS and g != "price"]
    if catalog.meta(dataset).kind != "bars":
        raise AgentError(f"{dataset} is a table, not price bars")
    s = E.study(dataset, groups, label=None)
    return {"dataset": dataset, "horizon": s["horizon"], "groups": groups, "features": s["features"], "notes": s["notes"],
            "oos_r2": round(s["r2"], 4), "correlation": round(s["corr"], 3), "high_minus_low_third_er": round(s["spread"], 4),
            "t_stat": round(s["t_spread"], 2), "own_past_efficiency_only": {"oos_r2": round(s["own_r2"], 4), "correlation": round(s["own_corr"], 3)},
            "development_data_ends": s["dev_end"],
            "context": f"An opening-range breakout needed a correlation of about {E.NEEDED_CORR}; momentum rules can gain from less. "
                       "Logged in the ledger as an exploration. Whether a gate helps a strategy is only settled by a card's gate test."}


def request_review(card_id: str, summary: str) -> dict:
    """Hand the card to a person: they read it, and only they can freeze it and run the gate."""
    bd = board.load()
    c = board.get(bd, card_id)
    if c["stage"] != "research":
        raise AgentError("only a card in research can be sent for review")
    pre = os.path.join(board.card_dir(c), "PREREGISTRATION.md")
    with open(pre) as f:
        if "TODO" in f.read():
            raise AgentError("PREREGISTRATION.md still has TODO sections")
    c["review"] = {"by": "agent", "utc": time.strftime("%Y-%m-%d %H:%M", time.gmtime()), "summary": summary.strip()[:2000],
                   "explorations": len(explorations(c))}
    c["history"].append({"utc": c["review"]["utc"], "from": "research", "to": "research", "note": "AI asked for review"})
    board.save(bd)
    return {"status": "waiting for a person", "next_steps_for_the_person": [f"read cards/{c['id']}/PREREGISTRATION.md",
            f"oql card freeze {c['id'].split('-')[0]}", f"oql card gate {c['id'].split('-')[0]}"]}


# ---------------------------------------------------------------------------------------------- registry
S = lambda props, req: {"type": "object", "properties": props, "required": req, "additionalProperties": False}
STR, INT, NUM = {"type": "string"}, {"type": "integer"}, {"type": "number"}
PARAMS = {"type": "object", "description": "parameter name -> integer, e.g. {\"lookback\": 60}", "additionalProperties": {"type": "integer"}}
SIDE = {"type": "string", "enum": ["both", "long", "short"]}
GATE = {"type": "object", "description": "optional efficiency gate, e.g. {\"kind\": \"er_forecast\", \"side\": \"high\", \"share\": 50} or "
        "{\"kind\": \"er_trailing\", \"lookback\": 24, \"side\": \"high\", \"share\": 33}; share = percent of the time it may be on",
        "properties": {"kind": {"type": "string", "enum": ["none", *GATES]}, "lookback": INT, "side": {"type": "string", "enum": ["high", "low"]},
                       "share": INT}, "required": ["kind"], "additionalProperties": False}
RULE = {"type": "string", "enum": list(ALL_RULES)}

TOOLS = {
    "lab_overview": (lab_overview, "Cards by stage, ledger counts, datasets, strategy rules, and what the agent may and may not do. Call this first.", S({}, [])),
    "list_datasets": (list_datasets, "Every dataset in the catalog with its licence, rows and dates.", S({}, [])),
    "dataset_summary": (dataset_summary, "One dataset's metadata, quality check, bar spacing and volatility.", S({"dataset": STR}, ["dataset"])),
    "get_card": (get_card, "A card with its pre-registration, experiment, gate report and exploration count.", S({"card_id": STR}, ["card_id"])),
    "propose_idea": (propose_idea, "Create a new idea card. The mechanism must say why it should make money and who is on the other side.",
                     S({"title": STR, "mechanism": STR, "hypothesis": STR}, ["title", "mechanism", "hypothesis"])),
    "start_research": (start_research, "Move an idea card to research so it can be explored.", S({"card_id": STR}, ["card_id"])),
    "write_preregistration": (write_preregistration, "Fill the pre-registration sections (only before the freeze). The pass rule is fixed and cannot be edited.",
                              S({"card_id": STR, "hypothesis": STR, "mechanism": STR, "data": STR, "strategy_space": STR,
                                 "what_would_change_my_mind": STR}, ["card_id", "hypothesis", "data", "strategy_space", "what_would_change_my_mind"])),
    "write_experiment": (write_experiment, "Declare the experiment from structured fields: datasets, library rules, optional parameter grids, costs (bps per side), alpha, n_perm. "
                         "Every strategy trades ALL the listed datasets together as one equal-risk portfolio, so the search size is the number of "
                         "rule and parameter combinations, not that number times the datasets.",
                         S({"card_id": STR, "datasets": {"type": "array", "items": STR},
                            "rules": {"type": "array", "items": RULE},
                            "grid": {"type": "object", "description": "rule -> {param -> [integers]}"},
                            "cost_bps": {"type": "object", "additionalProperties": NUM}, "alpha": NUM, "n_perm": INT,
                            "gates": {"type": "array", "items": {"type": "string", "enum": ["none", *GATES]},
                                      "description": "efficiency gates the search may add to strategies (default ['none'] = no gates). Each gate setting multiplies the search size."},
                            "nulls": {"type": "array", "items": {"type": "string", "enum": ["bars", "vol_matched"]},
                                      "description": "null models for the gate; PASS needs p < alpha under every one (default ['bars'])"},
                            "rebalance_band": {"type": "number", "description": "resize positions only in steps of (1 + band), e.g. 0.25, "
                                               "so hourly strategies stop paying to resize every bar; 0 (default) = every bar"},
                            "gate_features": {"type": "array", "items": {"type": "string"},
                                              "description": "feature groups for er_forecast: price (always), calendar, funding, basis, macro, positioning, orderbook"}},
                           ["card_id", "datasets"])),
    "explore": (explore, "Score one strategy on ONE dataset's DEVELOPMENT data only (never the holdout), at the card's declared costs. Logged and "
                "counted against the card. The gate scores strategies on all the experiment's datasets together, so this is indirect evidence when there are several.",
                S({"card_id": STR, "dataset": STR, "rule": RULE, "params": PARAMS, "side": SIDE, "gate": GATE},
                  ["card_id", "dataset", "rule", "params"])),
    "permutation_check": (permutation_check, "Exploratory permutation test of one strategy on development data (19-199 shuffles). Logged; does not replace the gate.",
                          S({"card_id": STR, "dataset": STR, "rule": RULE, "params": PARAMS, "side": SIDE, "n": INT, "gate": GATE},
                            ["card_id", "dataset", "rule", "params"])),
    "efficiency_study": (efficiency_study, "Is the next period's efficiency (|net move| / distance travelled) predictable on this dataset? Walk-forward "
                         "forecast on development data from price features plus optional groups (calendar, funding, basis, macro, positioning, "
                         "orderbook). Returns out-of-sample R², correlation and the realized-efficiency spread between predicted thirds. Logged.",
                         S({"dataset": STR, "groups": {"type": "array", "items": {"type": "string"}}}, ["dataset"])),
    "request_review": (request_review, "Hand a finished research card to a person, with a summary. Only a person can freeze it and run the gate.",
                       S({"card_id": STR, "summary": STR}, ["card_id", "summary"])),
}


def call(name: str, arguments: dict):
    if name not in TOOLS:
        raise AgentError(f"no tool {name!r}. Freezing, the gate, the holdout and paper deployment are for people only.")
    return TOOLS[name][0](**(arguments or {}))
