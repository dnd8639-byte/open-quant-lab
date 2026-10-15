"""The ledger: one row per test ever run, pass or fail. Nothing is ever removed from it.
Judging a result means knowing how many things were tried before it; this file is that count."""
from __future__ import annotations

import csv
import json
import os
import time

FIELDS = ["utc", "card", "family", "kind", "verdict", "p", "sharpe", "n_trials", "finalist", "data", "note"]


def path(lab: str) -> str:
    return os.path.join(lab, "LEDGER.csv")


def append(lab: str, **row) -> None:
    p = path(lab)
    new = not os.path.exists(p)
    row = {k: row.get(k, "") for k in FIELDS} | {"utc": time.strftime("%Y-%m-%d %H:%M", time.gmtime())}
    if isinstance(row["finalist"], dict):
        row["finalist"] = json.dumps(row["finalist"], sort_keys=True)
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def rows(lab: str) -> list[dict]:
    p = path(lab)
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def trials_in_family(lab: str, family: str) -> int:
    """Strategies searched across every card in one idea family (a card and all its revisions)."""
    return sum(int(r["n_trials"] or 0) for r in rows(lab) if r["family"] == family and r["kind"] == "gate")
