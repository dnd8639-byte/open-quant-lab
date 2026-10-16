"""oql: the Open Quant Lab command line.   Run `python -m oql --help`, or `oql --help` after `pip install -e .`

DATA
  oql data list                                  every dataset, its licence, rows and dates
  oql data check [ID ...]                        quality report (all datasets if none given)
  oql data pull-fred DGS10 CPIAUCSL ...          FRED series (free, no key)
  oql data pull-binance BTCUSDT --interval 1h --start 2018-01
  oql data import-csv FILE NAME FREQ [--license own|public|licensed]
  oql data import-databento DIR [--roots ES,CL]  licensed: stays on this machine
  oql data import-calendar FILE                  release calendar (date,event)
  oql data import-pi FILE                        the Pi's hourly order-book features
  oql data pull-french [--dir DIR]               Fama-French factor tables (free; or from local zips)
  oql data import-projects DIR                   the mini-projects' option chain and trades files

PIPELINE
  oql card new "TITLE" --mechanism "WHY IT SHOULD WORK"
  oql card start ID          idea -> research (explore development data freely)
  oql card freeze ID         research -> preregistered (fingerprints the rules; no TODOs allowed)
  oql card gate ID           runs the search + whole-search permutation test; FAIL archives the card
  oql card holdout ID        ONE-SHOT test on the last 30%; FAIL archives the card
  oql card paper ID          holdout -> paper trading
  oql card archive ID --reason "..."
  oql card revise ID         new card in the same family (trial counts keep adding up)
  oql card show ID
  oql board [--html board.html]
  oql ledger

PAPER TRADING (no orders are ever sent)
  oql paper deploy ID --feed DATASET=EXCHANGE:SYMBOL:TIMEFRAME   e.g. binance/BTCUSDT/1h=binanceus:BTC/USDT:1h
  oql paper step             fetch newly closed bars, decide, journal (hourly timer on the Pi)
  oql paper status
  oql paper reconcile ID     live vs backtest: replay, data and returns checks

COLLECTOR (run daily on the Pi)
  oql collect init           write collect.json with the default jobs
  oql collect run [--job NAME] [--force]
  oql collect status

EFFICIENCY (development data only; every run is logged in the ledger; see docs/EFFICIENCY.md)
  oql efficiency study DATASET [--groups calendar,funding,basis,macro,positioning,orderbook] [--model ridge|trees]
  oql efficiency gates DATASET "STRATEGY" [--cost 2]   every efficiency gate on one strategy, e.g. "momentum lookback=20"

AI ASSISTANT (Model Context Protocol server; see docs/AGENTS.md)
  oql mcp                    run the lab's tools for an AI assistant over stdio

DASHBOARD
  oql serve [--port 8765]    live board, gate runs as they happen, the permutation lab, data, ledger
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _data(a):
    from . import catalog, sources
    if a.cmd == "list":
        ms = catalog.list_all()
        if not ms:
            print(f"catalog at {catalog.root()} is empty"); return
        print(f"{'dataset':34s} {'licence':9s} {'rows':>9s}  {'from':10s}  {'to':10s}  fmt")
        for m in ms:
            print(f"{m.id:34s} {m.license:9s} {m.rows:9d}  {m.start[:10]:10s}  {m.end[:10]:10s}  {m.fmt}")
    elif a.cmd == "check":
        ids = a.ids or [m.id for m in catalog.list_all()]
        bad = 0
        for i in ids:
            r = catalog.check(i)
            flag = "PROBLEM" if r["issues"] else ("look" if r.get("warnings") else "ok")
            bad += bool(r["issues"])
            print(f"{flag:8s} {i}  ({r['rows']} rows, {r['license']})")
            for x in r["issues"]:
                print(f"           issue:   {x}")
            for x in r.get("warnings", []):
                print(f"           warning: {x}")
        sys.exit(1 if bad else 0)
    elif a.cmd == "pull-fred":
        for s in a.series:
            m = sources.fred(s); print(f"{m.id}: {m.rows} rows {m.start[:10]} -> {m.end[:10]}")
    elif a.cmd == "pull-binance":
        m = sources.binance(a.symbol, a.interval, a.start, a.end)
        print(f"{m.id}: {m.rows} rows {m.start[:16]} -> {m.end[:16]}" + (f"\n  note: {m.notes}" if m.notes else ""))
    elif a.cmd == "import-csv":
        m = sources.csv_bars(a.file, a.name, a.freq, license=a.license); print(f"{m.id}: {m.rows} rows")
    elif a.cmd == "import-databento":
        for m in sources.databento_front(a.dir, a.roots.split(",") if a.roots else None):
            print(f"{m.id}: {m.rows} rows {m.start[:10]} -> {m.end[:10]} (licensed)")
    elif a.cmd == "import-calendar":
        m = sources.calendar(a.file); print(f"{m.id}: {m.rows} events")
    elif a.cmd == "import-pi":
        m = sources.pi_hourly(a.file); print(f"{m.id}: {m.rows} hours")
    elif a.cmd == "pull-french":
        for name in sources.FRENCH:
            m = sources.french(name, a.dir); print(f"{m.id}: {m.rows} months {m.start[:7]} -> {m.end[:7]}")
    elif a.cmd == "import-projects":
        for m in sources.import_projects(a.dir):
            print(f"{m.id}: {m.rows} rows ({m.license})")


def _card(a):
    from . import board
    if a.cmd == "new":
        c = board.new_card(a.title, a.mechanism or "")
        d = board.card_dir(c)
        print(f"created {c['id']}\n  fill in {os.path.join(d, 'PREREGISTRATION.md')}\n  and     {os.path.join(d, 'experiment.py')}")
    elif a.cmd == "start":
        print(board.start(a.id)["stage"])
    elif a.cmd == "freeze":
        c = board.freeze(a.id)
        print(f"{c['id']} frozen. Commit and push now: the commit time is public proof the rules came first.")
    elif a.cmd == "gate":
        r = board.run_gate(a.id, procs=a.procs)
        print(f"\nGATE {r['verdict']}: p = {r['p']:.3f}, dev Sharpe {r['dev_sharpe']:.2f}, {r['n_trials']} strategies; "
              f"report: {os.path.join(board.card_dir(board.get(board.load(), a.id)), 'gate.md')}")
    elif a.cmd == "holdout":
        r = board.run_holdout(a.id, procs=a.procs)
        print(f"HOLDOUT {r['verdict']}: Sharpe {r['holdout_sharpe']:.2f}, p = {r['p']:.3f} ({r['start'][:10]} -> {r['end'][:10]})")
    elif a.cmd == "paper":
        print(board.to_paper(a.id)["stage"])
    elif a.cmd == "archive":
        print(board.archive(a.id, a.reason)["stage"])
    elif a.cmd == "revise":
        c = board.revise(a.id); print(f"created {c['id']} in family {c['family']}")
    elif a.cmd == "show":
        print(json.dumps(board.get(board.load(), a.id), indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(prog="oql", description="Open Quant Lab", formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    sub = p.add_subparsers(dest="area", required=True)

    d = sub.add_parser("data").add_subparsers(dest="cmd", required=True)
    d.add_parser("list")
    x = d.add_parser("check"); x.add_argument("ids", nargs="*")
    x = d.add_parser("pull-fred"); x.add_argument("series", nargs="+")
    x = d.add_parser("pull-binance"); x.add_argument("symbol"); x.add_argument("--interval", default="1h")
    x.add_argument("--start", default="2018-01"); x.add_argument("--end", default=None)
    x = d.add_parser("import-csv"); x.add_argument("file"); x.add_argument("name"); x.add_argument("freq")
    x.add_argument("--license", default="own", choices=["own", "public", "licensed"])
    x = d.add_parser("import-databento"); x.add_argument("dir"); x.add_argument("--roots", default="")
    x = d.add_parser("import-calendar"); x.add_argument("file")
    x = d.add_parser("import-pi"); x.add_argument("file")
    x = d.add_parser("pull-french"); x.add_argument("--dir", default=None)
    x = d.add_parser("import-projects"); x.add_argument("dir")

    c = sub.add_parser("card").add_subparsers(dest="cmd", required=True)
    x = c.add_parser("new"); x.add_argument("title"); x.add_argument("--mechanism", default="")
    for name in ("start", "freeze", "paper", "revise", "show"):
        c.add_parser(name).add_argument("id")
    for name in ("gate", "holdout"):
        x = c.add_parser(name); x.add_argument("id"); x.add_argument("--procs", type=int, default=None)
    x = c.add_parser("archive"); x.add_argument("id"); x.add_argument("--reason", required=True)

    b = sub.add_parser("board"); b.add_argument("--html", default=None)
    sub.add_parser("ledger")
    pp = sub.add_parser("paper").add_subparsers(dest="cmd", required=True)
    x = pp.add_parser("deploy"); x.add_argument("id"); x.add_argument("--feed", action="append", required=True)
    pp.add_parser("step"); pp.add_parser("status")
    x = pp.add_parser("reconcile"); x.add_argument("id")
    k = sub.add_parser("collect").add_subparsers(dest="cmd", required=True)
    k.add_parser("init"); k.add_parser("status")
    x = k.add_parser("run"); x.add_argument("--job", default=None); x.add_argument("--force", action="store_true")
    e = sub.add_parser("efficiency").add_subparsers(dest="cmd", required=True)
    x = e.add_parser("study"); x.add_argument("dataset"); x.add_argument("--groups", default=""); x.add_argument("--model", default="ridge")
    x = e.add_parser("gates"); x.add_argument("dataset"); x.add_argument("strategy"); x.add_argument("--cost", type=float, default=2.0)
    sub.add_parser("mcp")
    s = sub.add_parser("serve"); s.add_argument("--port", type=int, default=8765); s.add_argument("--no-browser", action="store_true")

    a = p.parse_args(argv)
    from .board import BoardError
    from .locks import LockError
    try:
        if a.area == "data":
            _data(a)
        elif a.area == "card":
            _card(a)
        elif a.area == "board":
            from . import board
            bd = board.load()
            if a.html:
                with open(a.html, "w") as f:
                    f.write(board.render_html(bd))
                print(f"wrote {a.html}")
            else:
                print(board.render_text(bd))
        elif a.area == "paper":
            from . import paper
            if a.cmd == "deploy":
                feeds = dict(f.split("=", 1) for f in a.feed)
                d = paper.deploy(a.id, feeds)
                print(f"{d['card']} is paper trading. Run `oql paper step` (or install the timer) to start.")
            elif a.cmd == "step":
                paper.step()
            elif a.cmd == "status":
                print(paper.status())
            else:
                print(json.dumps(paper.reconcile(a.id), indent=1))
        elif a.area == "collect":
            from . import collect
            if a.cmd == "init":
                print(f"wrote {collect.init()}; edit it, then run `oql collect run`")
            elif a.cmd == "run":
                collect.run(only=a.job, force=a.force)
            else:
                print(collect.status())
        elif a.area == "efficiency":
            _efficiency(a)
        elif a.area == "mcp":
            from .mcp_server import serve as mcp_serve
            mcp_serve()
        elif a.area == "serve":
            from .server import serve
            serve(a.port, not a.no_browser)
        elif a.area == "ledger":
            from . import board, ledger
            rows = ledger.rows(board.lab_dir())
            for r in rows:
                print(f"{r['utc']}  {r['kind']:8s} {r['verdict']:5s} p={r['p']:7s} SR={r['sharpe']:7s} trials={r['n_trials']:4s} {r['card']}")
            tests = [r for r in rows if r["kind"] in ("gate", "holdout")]
            n = len(tests)
            x = sum(int(r["n_trials"] or 1) for r in rows if r["kind"] == "explore")   # a gate check row stands for many settings
            print(f"{n} test{'' if n == 1 else 's'} ({sum(r['verdict'] == 'PASS' for r in tests)} passed) · "
                  f"{x} exploration{'' if x == 1 else 's'} before freezes")
    except (BoardError, LockError, KeyError, ValueError) as e:
        sys.exit(f"refused: {e}")
    except OSError as e:                                   # network problems: one clear line, not a traceback
        sys.exit(f"network error: {e}. Check the connection; if it mentions certificates, run `pip install -e .` again.")


def _efficiency(a):
    import threading
    from . import efficiency as E
    last, logs = {}, []

    def emit(ev):
        if "log" in ev:
            print(" ", ev["log"], flush=True)
        for k in ("stat", "check", "table"):
            if k in ev:
                last[(k, ev[k])] = ev
    if a.cmd == "study":
        params = {"dataset": a.dataset, "model": a.model, **{g: True for g in a.groups.split(",") if g}}
        E.run(params, emit, threading.Event())
    else:
        E.gate_check({"dataset": a.dataset, "strategy": a.strategy, "cost_bps": a.cost}, emit, threading.Event())
    for (k, _), ev in last.items():
        if k == "check":
            print(f"  [{'pass' if ev['passed'] else 'FAIL'}] {ev['check']}: {ev['detail']}")
    for (k, _), ev in last.items():
        if k == "stat":
            v = ev["value"]
            print(f"  {ev['label']}: {format(v, ev['fmt'].replace(',', '')) if isinstance(v, (int, float)) else v}")
    for (k, _), ev in last.items():
        if k == "table":
            print(f"\n  {ev['title']}\n  " + " | ".join(ev["columns"]))
            for r in ev["rows"]:
                print("  " + " | ".join(str(x) for x in r))


if __name__ == "__main__":
    main()
