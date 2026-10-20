"""The research pipeline board. Each card is one strategy idea moving through fixed stages:

    idea -> research -> preregistered -> gate -> holdout -> paper        (and archive, from anywhere)

The rules are enforced here, in code:
  * no skipping and no going back: a card only moves to the next stage, or to the archive;
  * `freeze` fingerprints PREREGISTRATION.md + experiment.py; the gate and the holdout refuse if either changed;
  * the gate runs the declared search plus a permutation test of the whole search; a FAIL archives the card;
  * the holdout is one-shot (its lock is written before the result exists); a FAIL archives the card;
  * changing a tested idea means `revise`: a NEW card in the same family, and the family's trial count keeps
    adding up across revisions, so a fifth attempt is judged as a fifth attempt;
  * every gate and holdout run is appended to LEDGER.csv, pass or fail.

A lab is a folder (default: the current folder, or $OQL_LAB) holding board.json, LEDGER.csv and cards/.
"""
from __future__ import annotations

import html
import json
import os
import re
import shutil
import time

from . import catalog, ledger, locks

STAGES = ["idea", "research", "preregistered", "gate", "holdout", "paper"]
ALL_STAGES = STAGES + ["archive"]


class BoardError(RuntimeError):
    pass


def lab_dir() -> str:
    return os.environ.get("OQL_LAB", os.getcwd())


def _board_path(lab):
    return os.path.join(lab, "board.json")


def load(lab=None) -> dict:
    lab = lab or lab_dir()
    p = _board_path(lab)
    if not os.path.exists(p):
        return {"cards": []}
    with open(p) as f:
        return json.load(f)


def save(board: dict, lab=None) -> None:
    lab = lab or lab_dir()
    tmp = _board_path(lab) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(board, f, indent=1)
    os.replace(tmp, _board_path(lab))


def card_dir(card: dict, lab=None) -> str:
    return os.path.join(lab or lab_dir(), "cards", card["id"])


def get(board: dict, cid: str) -> dict:
    hits = [c for c in board["cards"] if c["id"] == cid or c["id"].split("-")[0] == cid.zfill(4)]
    if not hits:
        raise BoardError(f"no card {cid!r}")
    return hits[0]


def _now():
    return time.strftime("%Y-%m-%d %H:%M", time.gmtime())


def _move(card: dict, to: str, note: str = ""):
    card["history"].append({"utc": _now(), "from": card["stage"], "to": to, "note": note})
    card["stage"] = to


def _require(card: dict, stage: str):
    if card["stage"] != stage:
        raise BoardError(f"card {card['id']} is in '{card['stage']}'; this step needs it in '{stage}'. "
                         f"Cards move one stage at a time: {' -> '.join(STAGES)}.")


PREREG_TEMPLATE = """# Pre-registration: {title}

Card {id} · family {family} · written {utc} UTC

## Hypothesis
TODO: one sentence that could turn out false.

## Mechanism
{mechanism}

## Data
TODO: which datasets (ids from `oql data list`), which dates. The last {holdout:.0%} of dates is the holdout.

## Strategy space
TODO: which rules and parameter grids the search may use (they are in experiment.py).

## Test and pass rule
The gate reruns the whole search on `n_perm` shuffled copies of the development data (set in experiment.py;
default {n_perm}). PASS if p < `alpha` (default {alpha}) and the cost/delay Monte Carlo median Sharpe > 0.
Otherwise the card is archived. The holdout runs once. PASS if holdout Sharpe > 0 and its permutation p < `alpha`.

## What would change my mind
TODO
"""

EXPERIMENT_TEMPLATE = '''"""Experiment for card {id}: {title}. Frozen together with PREREGISTRATION.md by `oql card freeze`."""
from oql.experiment import Experiment

EXPERIMENT = Experiment(
    datasets=[],                       # TODO: ids from `oql data list`, e.g. "databento/ES/1d"
    rules=["breakout", "ma_cross", "momentum", "reversal"],
    cost_bps={{"default": 1.0}},          # per side, by dataset id or "default"
    alpha=0.05,
    n_perm=99,
)

# Optional: your own rules. Each maps (close array, **params) -> +1 / -1 / 0 per bar, using data up to that bar.
# CUSTOM_RULES = {{"my_rule": (my_function, {{"lookback": [10, 20, 50]}})}}
'''


def new_card(title: str, mechanism: str = "", lab=None, family: str | None = None) -> dict:
    lab = lab or lab_dir()
    board = load(lab)
    n = 1 + max([int(c["id"].split("-")[0]) for c in board["cards"]] or [0])
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40] or "idea"
    cid = f"{n:04d}-{slug}"
    card = dict(id=cid, title=title, mechanism=mechanism, family=family or cid, stage="idea", created=_now(),
                history=[], results={})
    d = card_dir(card, lab)
    os.makedirs(d)
    from .experiment import Experiment
    e = Experiment(datasets=[])
    with open(os.path.join(d, "PREREGISTRATION.md"), "w") as f:
        f.write(PREREG_TEMPLATE.format(title=title, id=cid, family=card["family"], utc=_now(),
                                       mechanism=mechanism or "TODO: why should this make money, and who is on the other side?",
                                       holdout=1 - e.dev_frac, n_perm=e.n_perm, alpha=e.alpha))
    with open(os.path.join(d, "experiment.py"), "w") as f:
        f.write(EXPERIMENT_TEMPLATE.format(id=cid, title=title))
    board["cards"].append(card)
    save(board, lab)
    return card


def start(cid: str, lab=None) -> dict:
    board = load(lab); card = get(board, cid)
    _require(card, "idea")
    _move(card, "research", "exploring development data")
    save(board, lab)
    return card


def freeze(cid: str, lab=None) -> dict:
    from .experiment import load as load_exp
    lab = lab or lab_dir()
    board = load(lab); card = get(board, cid)
    _require(card, "research")
    d = card_dir(card, lab)
    pre, expf = os.path.join(d, "PREREGISTRATION.md"), os.path.join(d, "experiment.py")
    with open(pre) as f:
        text = f.read()
    if "TODO" in text:
        raise BoardError("PREREGISTRATION.md still has TODO sections. Fill every one before freezing.")
    exp = load_exp(expf)
    if not exp.datasets:
        raise BoardError("experiment.py lists no datasets.")
    for ds in exp.datasets:
        catalog.meta(ds)                                       # raises if a dataset is missing
    exp.rule_table()                                           # raises on an unknown rule
    bad = [m for m in getattr(exp, "nulls", ["bars"]) if m not in ("bars", "vol_matched")]
    if bad or not getattr(exp, "nulls", ["bars"]):
        raise BoardError(f"nulls must be a non-empty list of 'bars' and/or 'vol_matched', got {getattr(exp, 'nulls', None)}")
    frozen = locks.freeze([pre, expf])
    frozen["data"] = {ds: catalog.meta(ds).sha256 for ds in exp.datasets}
    from . import __version__
    frozen["oql_version"] = __version__                       # the lab's own code can change what a frozen card means
    exp_log = os.path.join(d, "explorations.jsonl")
    if os.path.exists(exp_log):                            # exploration done before the freeze is part of the record
        with open(exp_log) as f:
            frozen["explorations"] = sum(1 for l in f if l.strip())
    # efficiency studies and gate checks on the same data count too: they shape which strategies get proposed
    frozen["efficiency_explorations"] = sum(int(r["n_trials"] or 1) for r in ledger.rows(lab or lab_dir())
                                            if r["card"] == "efficiency-lab" and r["data"] in exp.datasets)
    with open(os.path.join(d, "frozen.json"), "x") as f:
        json.dump(frozen, f, indent=1)
    _move(card, "preregistered", "rules and experiment fingerprinted")
    save(board, lab)
    return card


def _verify(card, lab):
    d = card_dir(card, lab)
    with open(os.path.join(d, "frozen.json")) as f:
        frozen = json.load(f)
    locks.verify(frozen, d)
    for ds, h in frozen.get("data", {}).items():
        if catalog.meta(ds).sha256 != h:
            raise locks.LockError(f"dataset {ds} changed since the freeze. A test on different data is a new test: revise the card.")
    return frozen


def run_gate(cid: str, lab=None, procs=None) -> dict:
    """The gate is deterministic given the frozen files (fixed seeds), so a run that crashed can simply be
    rerun; a run that finished cannot, because its report already exists."""
    from .experiment import load as load_exp
    from .gate import run_gate as gate
    lab = lab or lab_dir()
    board = load(lab); card = get(board, cid)
    _require(card, "preregistered")
    frozen = _verify(card, lab)
    d = card_dir(card, lab)
    if os.path.exists(os.path.join(d, "gate.json")):
        raise locks.LockError("this card's gate already ran; its report is gate.md")
    exp = load_exp(os.path.join(d, "experiment.py"))
    prog_path = os.path.join(d, "gate_progress.jsonl")
    prog = open(prog_path, "w")

    def progress(event):                                   # one JSON line per event, for the live dashboard
        prog.write(json.dumps({"utc": time.time(), **event}) + "\n")
        prog.flush()
    try:
        rep = gate(exp, procs=procs, progress=progress)
    finally:
        prog.close()
    rep["family_trials_before"] = ledger.trials_in_family(lab, card["family"])
    rep["explorations_before_freeze"] = frozen.get("explorations", 0)
    rep["efficiency_explorations_before_freeze"] = frozen.get("efficiency_explorations", "not recorded")
    from . import __version__
    rep["oql_version"], rep["oql_version_at_freeze"] = __version__, frozen.get("oql_version", "not recorded")
    trials = rep.pop("trials")
    with open(os.path.join(d, "gate.json"), "w") as f:
        json.dump(rep, f, indent=1, default=float)
    with open(os.path.join(d, "trials.json"), "w") as f:
        json.dump(trials, f)
    with open(os.path.join(d, "gate.md"), "w") as f:
        f.write(gate_markdown(card, rep))
    ledger.append(lab, card=card["id"], family=card["family"], kind="gate", verdict=rep["verdict"], p=round(rep["p"], 4),
                  sharpe=round(rep["dev_sharpe"], 3), n_trials=rep["n_trials"], finalist=rep["finalist"],
                  data=";".join(exp.datasets))
    card["results"]["gate"] = {k: rep[k] for k in ("verdict", "p", "dev_sharpe", "n_trials", "finalist", "split")}
    if rep["verdict"] == "PASS":
        _move(card, "gate", f"gate PASS, p={rep['p']:.3f}")
    else:
        _move(card, "archive", f"gate FAIL, p={rep['p']:.3f}")
    save(board, lab)
    return rep


def run_holdout(cid: str, lab=None, procs=None) -> dict:
    from .experiment import load as load_exp
    from .gate import run_holdout as holdout
    lab = lab or lab_dir()
    board = load(lab); card = get(board, cid)
    _require(card, "gate")
    _verify(card, lab)
    d = card_dir(card, lab)
    locks.claim(os.path.join(d, "holdout.lock"), "the holdout for this card")    # lock FIRST, then look
    exp = load_exp(os.path.join(d, "experiment.py"))
    g = card["results"]["gate"]
    rep = holdout(exp, g["finalist"], g["split"], procs=procs)
    with open(os.path.join(d, "holdout.json"), "w") as f:
        json.dump(rep, f, indent=1, default=float)
    ledger.append(lab, card=card["id"], family=card["family"], kind="holdout", verdict=rep["verdict"], p=round(rep["p"], 4),
                  sharpe=round(rep["holdout_sharpe"], 3), finalist=g["finalist"], data=";".join(exp.datasets))
    card["results"]["holdout"] = {k: rep[k] for k in ("verdict", "p", "holdout_sharpe", "start", "end")}
    _move(card, "holdout" if rep["verdict"] == "PASS" else "archive", f"holdout {rep['verdict']}, Sharpe {rep['holdout_sharpe']:.2f}")
    save(board, lab)
    return rep


def to_paper(cid: str, lab=None, note: str = "") -> dict:
    board = load(lab); card = get(board, cid)
    _require(card, "holdout")
    _move(card, "paper", note or "deployed to paper trading")
    save(board, lab)
    return card


def archive(cid: str, reason: str, lab=None) -> dict:
    board = load(lab); card = get(board, cid)
    if card["stage"] == "archive":
        raise BoardError("already archived")
    _move(card, "archive", reason)
    save(board, lab)
    return card


def revise(cid: str, lab=None) -> dict:
    """A changed version of a tested idea: a new card in the same family, back at research, files copied, unfrozen."""
    lab = lab or lab_dir()
    board = load(lab); old = get(board, cid)
    if old["stage"] in ("idea", "research"):
        raise BoardError("this card has not been frozen yet; just edit it.")
    new = new_card(old["title"] + " (revised)", old["mechanism"], lab, family=old["family"])
    for f in ("PREREGISTRATION.md", "experiment.py"):
        shutil.copy(os.path.join(card_dir(old, lab), f), os.path.join(card_dir(new, lab), f))
    board = load(lab)
    o, n = get(board, old["id"]), get(board, new["id"])
    if o["stage"] != "archive":
        _move(o, "archive", f"superseded by {n['id']}")
    _move(n, "research", f"revision of {o['id']}")
    save(board, lab)
    return n


# ---------------------------------------------------------------------------------------------- reports
def _other_nulls(r: dict) -> str:
    """Rows for further null models the experiment declared (PASS needs p < alpha under each)."""
    return "".join(f"\n| Second null: {m} | p = {o['p']:.3f}; shuffled best median {o['median']:.2f}, 95th percentile {o['p95']:.2f} |"
                   for m, o in (r.get("other_nulls") or {}).items())


def gate_markdown(card: dict, r: dict) -> str:
    sb, mc = r["side_beta"], r["cost_mc"]
    return f"""# Gate report: {card['title']}

**{r['verdict']}** · whole-search permutation p = {r['p']:.3f} (pass needs < {r['alpha']}) · card {card['id']}

| | |
|---|---|
| Strategies tried in this search | {r['n_trials']} |
| Strategies tried earlier in this family | {r['family_trials_before']} |
| Exploratory evaluations before the freeze | {r.get('explorations_before_freeze', 0)} |
| Efficiency-lab explorations on these datasets before the freeze | {r.get('efficiency_explorations_before_freeze', 'not recorded')} |
| Best (finalist) | `{json.dumps(r['finalist'])}` |
| Development Sharpe | {r['dev_sharpe']:.2f} (data before {r['split'][:10]}) |
| Best Sharpe on shuffled data | median {r['null_median']:.2f}, 95th percentile {r['null_p95']:.2f} ({r['n_perm']} reruns, null: {r.get('null_model', 'bars')}) |{_other_nulls(r)}
| Deflated Sharpe | {r['deflated_sharpe']:.3f} |
| Cost/delay Monte Carlo | median {mc['median']:.2f}, 5th percentile {mc['p5']:.2f}, positive {mc['share_positive']:.0%} |
| Long side / short side | {sb['sharpe_long']:.2f} / {sb['sharpe_short']:.2f} |
| Beta to the market / correlation | {sb['beta_to_market']:+.2f} / {sb['corr_to_market']:+.2f} |
| Lab version at the freeze / at the gate | {r.get('oql_version_at_freeze', 'not recorded')} / {r.get('oql_version', '?')} |
{_version_note(r)}{_below_all_note(r)}
The holdout (dates from {r['split'][:10]}) has not been touched.
"""


def _below_all_note(r: dict) -> str:
    null = r.get("null") or []
    if null and r["dev_sharpe"] < min(null):
        return (f"\n**Note:** the real result ({r['dev_sharpe']:.2f}) is worse than every one of the {len(null)} shuffled runs. In this "
                "data the rule's signal pointed the wrong way more often than chance would. If the opposite idea is worth testing, "
                "that is a new card and a new counted test, not a conclusion from this one.\n")
    return ""


def _version_note(r: dict) -> str:
    a, b = r.get("oql_version_at_freeze"), r.get("oql_version")
    if a and b and a != b:
        return f"\n**Note:** the lab's code changed between the freeze ({a}) and the gate ({b}). Check the changelog for anything that alters this test.\n"
    return ""


def render_text(board: dict) -> str:
    lines = []
    for st in ALL_STAGES:
        cs = [c for c in board["cards"] if c["stage"] == st]
        lines.append(f"{st.upper()} ({len(cs)})")
        for c in cs:
            g = c["results"].get("gate"); h = c["results"].get("holdout")
            extra = (f"  gate {g['verdict']} p={g['p']:.3f}" if g else "") + (f"  holdout {h['verdict']} SR={h['holdout_sharpe']:.2f}" if h else "")
            lines.append(f"  {c['id']}  {c['title']}{extra}")
    return "\n".join(lines)


def render_html(board: dict, lab=None) -> str:
    lab = lab or lab_dir()
    rows = ledger.rows(lab)
    col = []
    for st in ALL_STAGES:
        cards = []
        for c in [c for c in board["cards"] if c["stage"] == st]:
            g = c["results"].get("gate"); h = c["results"].get("holdout")
            badges = ""
            if g:
                badges += f'<span class="b {g["verdict"].lower()}">gate {g["verdict"]} · p {g["p"]:.3f}</span>'
            if h:
                badges += f'<span class="b {h["verdict"].lower()}">holdout {h["verdict"]} · SR {h["holdout_sharpe"]:.2f}</span>'
            last = c["history"][-1]["note"] if c["history"] else "new"
            cards.append(f'<div class="card"><div class="id">{html.escape(c["id"].split("-")[0])}</div>'
                         f'<div class="t">{html.escape(c["title"])}</div><div class="m">{html.escape(c["mechanism"][:160])}</div>'
                         f'{badges}<div class="h">{html.escape(last)}</div></div>')
        col.append(f'<section><h2>{st}<span>{len(cards)}</span></h2>{"".join(cards) or "<p class=e>empty</p>"}</section>')
    n_tests = len(rows); n_pass = sum(r["verdict"] == "PASS" for r in rows)
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Research board</title><style>
:root{{--bg:#f7f7f5;--fg:#1d2329;--mut:#66707a;--card:#fff;--line:#dfe2e5;--pass:#1f7a4d;--fail:#b3412f}}
@media (prefers-color-scheme:dark){{:root{{--bg:#16191c;--fg:#e8eaec;--mut:#9aa3ab;--card:#1f2327;--line:#30363c;--pass:#5cc493;--fail:#ef8a78}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}} header{{padding:20px 16px 8px}}
h1{{font-size:20px;margin:0}} .sub{{color:var(--mut)}} main{{display:grid;grid-template-columns:repeat(7,minmax(180px,1fr));gap:10px;padding:12px 16px;overflow-x:auto}}
section h2{{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);display:flex;justify-content:space-between}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px;margin-bottom:8px}}
.id{{font:11px ui-monospace,monospace;color:var(--mut)}} .t{{font-weight:600}} .m{{color:var(--mut);font-size:12px;margin:4px 0}}
.h{{font-size:11px;color:var(--mut);margin-top:6px}} .b{{display:inline-block;font-size:11px;border-radius:4px;padding:1px 6px;margin:2px 4px 0 0;border:1px solid}}
.pass{{color:var(--pass)}} .fail{{color:var(--fail)}} .e{{color:var(--mut);font-size:12px}}
@media (max-width:700px){{main{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>Research board</h1>
<div class="sub">{len(board["cards"])} cards · {n_tests} tests in the ledger · {n_pass} passed · generated {_now()} UTC</div></header>
<main>{"".join(col)}</main></body></html>"""
