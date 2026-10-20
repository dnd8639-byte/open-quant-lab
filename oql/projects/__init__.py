"""The ten quant mini-projects, running live inside `oql serve`.

Each project is a module with TITLE, QUESTION, PARAMS and run(params, emit, stop). run() streams its work as it
goes (see oql.live for the event shapes): charts that fill in, headline numbers, and its built-in known-answer
checks. Data comes from the catalog, so a project runs on whatever markets the lab has collected.
"""
from __future__ import annotations

import importlib

IDS = ["p01_tail_index", "p02_intraday_seasonality", "p03_overfitting_detector", "p04_momentum_regimes",
       "p05_factor_neutrality", "p06_american_put", "p07_autodiff_greeks", "p08_implied_distribution",
       "p09_hawkes_trades", "p10_news_diffusion", "p11_large_search"]


def module(pid: str):
    if pid not in IDS:
        raise KeyError(f"no project {pid}")
    return importlib.import_module(f"{__name__}.{pid}")


def listing() -> list[dict]:
    """What the dashboard needs to draw each project's card and controls."""
    from .. import catalog
    have = {m.id: m for m in catalog.list_all()}
    out = []
    for pid in IDS:
        m = module(pid)
        params = []
        for p in m.PARAMS:
            p = dict(p)
            if p.get("type") == "dataset":                        # offer only datasets the catalog actually has
                p["options"] = [d for d in have if _fits(have[d], p)]
                if p.get("default") not in p["options"]:
                    p["default"] = p["options"][0] if p["options"] else None
            params.append(p)
        needs = [d for d in getattr(m, "NEEDS", []) if d not in have]
        out.append({"id": pid, "title": m.TITLE, "question": m.QUESTION, "params": params,
                    "missing": needs, "how_to_get": getattr(m, "HOW_TO_GET", "")})
    return out


def _fits(meta, p) -> bool:
    if p.get("kind") and meta.kind != p["kind"]:
        return False
    if p.get("freq") and not meta.id.endswith("/" + p["freq"]):
        return False
    if p.get("prefix") and not meta.id.startswith(p["prefix"]):
        return False
    return True


def run(pid: str, params: dict, emit, stop):
    m = module(pid)
    defaults = {p["name"]: p.get("default") for p in m.PARAMS}
    try:
        m.run({**defaults, **(params or {})}, emit, stop)
    except KeyError as e:                                      # a dataset the project needs is not in the catalog yet
        if "not in the catalog" not in str(e):
            raise
        how = getattr(m, "HOW_TO_GET", "") or "oql data pull-french (factor tables) or oql data import-projects DIR"
        emit({"log": f"Missing data: {str(e).strip(chr(39))}. To add it: {how}"})
