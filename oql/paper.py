"""Phase 2: paper trading. A card that passed its holdout runs live, with the SAME strategy code, on live bars.
No orders are ever sent anywhere: positions are computed, journaled and marked to market.

    oql paper deploy ID --feed DATASET=EXCHANGE:SYMBOL:TIMEFRAME [--feed ...]
        holdout -> paper. Maps each dataset of the experiment to a live market (ccxt), e.g.
        --feed binance/BTCUSDT/1h=binanceus:BTC/USDT:1h
    oql paper step          fetch newly CLOSED bars for every deployment, decide, journal (run hourly by a timer)
    oql paper status        bars, current positions, paper return and Sharpe per deployment
    oql paper reconcile ID  does live behave like the backtest?  three checks:
        1. replay:   recompute every decision from the journaled bars with the backtest code; must match exactly
        2. data:     live bars vs the catalog's copy of the same market and hours (price differences in bps)
        3. returns:  paper P&L vs the backtest's P&L on the catalog data over the same hours

The journal is one SQLite file per deployment (paper/<card>.db): bars as first seen, every decision, and every
time an exchange later changed a bar it had already reported (revisions), which is itself a data-quality record.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time

import numpy as np
import pandas as pd

from . import board, catalog
from .engine import bars_per_year, market_returns, prepare
from .experiment import load as load_exp
from .search import sharpe
from .strategies import signal

TF = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}
SCHEMA = """
CREATE TABLE IF NOT EXISTS bars(ds TEXT, ts INTEGER, open REAL, high REAL, low REAL, close REAL, volume REAL, seen_utc REAL,
                                PRIMARY KEY(ds, ts));
CREATE TABLE IF NOT EXISTS decisions(ds TEXT, ts INTEGER, signal REAL, size REAL, target REAL, decided_utc REAL, PRIMARY KEY(ds, ts));
CREATE TABLE IF NOT EXISTS revisions(ds TEXT, ts INTEGER, field TEXT, old REAL, new REAL, seen_utc REAL);
"""


def _dir(lab=None):
    d = os.path.join(lab or board.lab_dir(), "paper")
    os.makedirs(d, exist_ok=True)
    return d


def _db(cid, lab=None):
    db = sqlite3.connect(os.path.join(_dir(lab), f"{cid}.db"))
    db.executescript(SCHEMA)
    return db


# ---------------------------------------------------------------------------------------------- feeds
class CcxtFeed:
    """Closed bars from an exchange's public API. No account, no keys."""

    def __init__(self):
        self._ex = {}

    def bars(self, exchange: str, symbol: str, timeframe: str, limit: int, now=None) -> pd.DataFrame:
        import ccxt
        if exchange not in self._ex:
            self._ex[exchange] = getattr(ccxt, exchange)({"enableRateLimit": True, "timeout": 15000})
        rows = self._ex[exchange].fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df.index = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
        return _closed(df, timeframe, now)


class ReplayFeed:
    """Serves bars from a DataFrame as if time were `now` (for tests and dry runs)."""

    def __init__(self, frames: dict):
        self.frames = frames                               # key "exchange:symbol:timeframe" -> DataFrame
        self.now = None

    def bars(self, exchange, symbol, timeframe, limit, now=None):
        df = self.frames[f"{exchange}:{symbol}:{timeframe}"]
        return _closed(df, timeframe, now or self.now).iloc[-limit:]


def _closed(df: pd.DataFrame, timeframe: str, now=None) -> pd.DataFrame:
    """Only bars that have finished: a bar labelled by its open time t is closed once t + timeframe <= now."""
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.utcnow()
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    return df[df.index + pd.Timedelta(TF[timeframe]) <= now]


# ---------------------------------------------------------------------------------------------- deploy
def deploy(cid: str, feeds: dict, lab=None, start_utc: float | None = None) -> dict:
    lab = lab or board.lab_dir()
    bd = board.load(lab)
    card = board.get(bd, cid)
    if card["stage"] != "holdout":
        raise board.BoardError(f"only a card that passed its holdout can paper-trade; this one is in '{card['stage']}'")
    exp = load_exp(os.path.join(board.card_dir(card, lab), "experiment.py"))
    missing = [ds for ds in exp.datasets if ds not in feeds]
    if missing:
        raise board.BoardError(f"give a live feed for every dataset: missing {missing} "
                               f"(--feed DATASET=EXCHANGE:SYMBOL:TIMEFRAME)")
    g = card["results"]["gate"]["finalist"]
    ppy = float(np.median([bars_per_year(catalog.read(ds).index) for ds in exp.datasets]))   # as the backtest measured it
    lookback = max([v for v in g["params"].values()] + [1])
    dep = dict(card=card["id"], finalist=g, datasets=exp.datasets, feeds=feeds, cost_bps=exp.cost_bps, delay=exp.delay, ppy=ppy,
               warmup=int(lookback + 25), started_utc=start_utc or time.time(),
               started=time.strftime("%Y-%m-%d %H:%M", time.gmtime(start_utc or time.time())),
               experiment=os.path.join(board.card_dir(card, lab), "experiment.py"))
    with open(os.path.join(_dir(lab), f"{card['id']}.json"), "x") as f:
        json.dump(dep, f, indent=1)
    board.to_paper(card["id"], lab, note="paper trading: " + ", ".join(f"{k} <- {v}" for k, v in feeds.items()))
    return dep


def deployments(lab=None) -> list[dict]:
    d = _dir(lab)
    out = []
    for f in sorted(os.listdir(d)):
        if f.endswith(".json"):
            with open(os.path.join(d, f)) as fh:
                out.append(json.load(fh))
    return out


def _journal_bars(db, ds) -> pd.DataFrame:
    df = pd.read_sql_query("SELECT ts, open, high, low, close, volume FROM bars WHERE ds=? ORDER BY ts", db, params=(ds,))
    df.index = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
    return df


# ---------------------------------------------------------------------------------------------- step
def step(feed=None, lab=None, now=None, verbose=True) -> list[str]:
    """One cycle for every deployment. Safe to run as often as you like: it only acts on newly closed bars."""
    feed = feed or CcxtFeed()
    msgs = []
    for dep in deployments(lab):
        exp = load_exp(dep["experiment"])
        rules = exp.rule_table()
        db = _db(dep["card"], lab)
        now_s = time.time()
        for ds in dep["datasets"]:
            exch, symbol, tf = dep["feeds"][ds].split(":")
            have = db.execute("SELECT COUNT(*) FROM bars WHERE ds=?", (ds,)).fetchone()[0]
            live = feed.bars(exch, symbol, tf, limit=max(dep["warmup"] * 2, 300) if have == 0 else 50, now=now)
            new = 0
            for ts, r in live.iterrows():
                key = int(ts.value // 1_000_000)
                old = db.execute("SELECT open, high, low, close, volume FROM bars WHERE ds=? AND ts=?", (ds, key)).fetchone()
                vals = (float(r.open), float(r.high), float(r.low), float(r.close), float(r.volume))
                if old is None:
                    db.execute("INSERT INTO bars VALUES(?,?,?,?,?,?,?,?)", (ds, key, *vals, now_s))
                    new += 1
                else:
                    for name, a, b in zip(("open", "high", "low", "close"), old[:4], vals[:4]):
                        if abs(a - b) > 1e-9 * max(1.0, abs(a)):
                            db.execute("INSERT INTO revisions VALUES(?,?,?,?,?,?)", (ds, key, name, a, b, now_s))
            bars = _journal_bars(db, ds)
            if len(bars) < dep["warmup"]:
                msgs.append(f"{dep['card']} {ds}: {len(bars)} bars, warming up (need {dep['warmup']})"); continue
            sig = signal(rules, bars["close"].to_numpy(float), dep["finalist"])
            prep = prepare({ds: bars}, dep["cost_bps"], warmup=0, ppy=dep["ppy"])
            size = prep.arrays[ds]["size"]
            decided = 0
            for i, ts in enumerate(bars.index):
                key = int(ts.value // 1_000_000)
                if key / 1000 < dep["started_utc"] - pd.Timedelta(TF[tf]).total_seconds():
                    continue                               # decisions only from deployment on
                if db.execute("SELECT 1 FROM decisions WHERE ds=? AND ts=?", (ds, key)).fetchone():
                    continue
                s = float(sig[i]); z = float(size[i]) if np.isfinite(size[i]) else 0.0
                db.execute("INSERT INTO decisions VALUES(?,?,?,?,?,?)", (ds, key, s, z, s * z, now_s))
                decided += 1
            last = bars.index[-1]
            msgs.append(f"{dep['card']} {ds}: +{new} bars (last closed {last:%Y-%m-%d %H:%M}), {decided} decisions, "
                        f"position {sig[-1]:+.0f} x {size[-1]:.2f}")
        db.commit()
        db.close()
    if verbose:
        print("\n".join(msgs) if msgs else "no deployments (oql paper deploy ...)")
    return msgs


# ---------------------------------------------------------------------------------------------- results
def _paper_returns(dep, lab=None):
    """Paper P&L per market from the journal: recorded decisions, journaled bars, the backtest engine's fill model."""
    db = _db(dep["card"], lab)
    out = {}
    for ds in dep["datasets"]:
        bars = _journal_bars(db, ds)
        dec = pd.read_sql_query("SELECT ts, signal FROM decisions WHERE ds=? ORDER BY ts", db, params=(ds,))
        if len(bars) < 3 or dec.empty:
            continue
        dec.index = pd.to_datetime(dec.pop("ts"), unit="ms", utc=True)
        prep = prepare({ds: bars}, dep["cost_bps"], warmup=0, score_from=dec.index[0], ppy=dep["ppy"])
        sig = dec["signal"].reindex(bars.index).fillna(0.0).to_numpy()
        pnl, trades = market_returns(prep.arrays[ds], sig, delay=dep["delay"])
        out[ds] = (pd.Series(pnl, index=bars.index).dropna(), trades, prep.ppy, dec.index[0])
    db.close()
    return out


def status(lab=None) -> str:
    lines = []
    for dep in deployments(lab):
        db = _db(dep["card"], lab)
        lines.append(f"{dep['card']}  since {dep['started']} UTC  finalist {json.dumps(dep['finalist'])}")
        for ds in dep["datasets"]:
            n = db.execute("SELECT COUNT(*), MAX(ts) FROM bars WHERE ds=?", (ds,)).fetchone()
            d = db.execute("SELECT COUNT(*) FROM decisions WHERE ds=?", (ds,)).fetchone()[0]
            r = db.execute("SELECT COUNT(*) FROM revisions WHERE ds=?", (ds,)).fetchone()[0]
            last = pd.Timestamp(n[1], unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M") if n[1] else "-"
            lines.append(f"  {ds:28s} <- {dep['feeds'][ds]:28s} bars {n[0]:6d}  last {last}  decisions {d:5d}  revisions {r}")
        db.close()
        for ds, (pnl, trades, ppy, _) in _paper_returns(dep, lab).items():
            lines.append(f"  {ds:28s} paper return {pnl.sum() * 100:+.2f}% over {len(pnl)} bars, {trades} trades, "
                         f"Sharpe {sharpe(pnl.to_numpy(), ppy):+.2f}")
    return "\n".join(lines) or "no deployments"


def reconcile(cid: str, lab=None) -> dict:
    dep = next((d for d in deployments(lab) if d["card"] == cid or d["card"].startswith(cid.zfill(4))), None)
    if dep is None:
        raise board.BoardError(f"no deployment for card {cid}")
    exp = load_exp(dep["experiment"])
    rules = exp.rule_table()
    db = _db(dep["card"], lab)
    report = {"card": dep["card"], "markets": {}}
    paper = _paper_returns(dep, lab)
    for ds in dep["datasets"]:
        m = {}
        bars = _journal_bars(db, ds)
        dec = pd.read_sql_query("SELECT ts, signal FROM decisions WHERE ds=? ORDER BY ts", db, params=(ds,))
        dec.index = pd.to_datetime(dec.pop("ts"), unit="ms", utc=True)
        # 1. replay: same bars + same code -> same decisions
        if len(bars) and len(dec):
            replay = pd.Series(signal(rules, bars["close"].to_numpy(float), dep["finalist"]), index=bars.index).reindex(dec.index)
            mism = int((replay.to_numpy() != dec["signal"].to_numpy()).sum())
            m["replay"] = {"decisions": len(dec), "mismatches": mism,
                           "note": "decisions differ from a replay; usually an exchange revised a bar after it was used"
                           if mism else "every decision reproduces exactly"}
        m["revisions"] = db.execute("SELECT COUNT(*) FROM revisions WHERE ds=?", (ds,)).fetchone()[0]
        # 2. data: live vs catalog
        try:
            cat = catalog.read(ds)
            j = pd.concat([bars["close"].rename("live"), cat["close"].rename("catalog")], axis=1, join="inner")
            if len(j):
                diff = (j.live / j.catalog - 1) * 1e4
                m["data"] = {"overlap_bars": len(j), "median_abs_bps": float(diff.abs().median()), "max_abs_bps": float(diff.abs().max())}
            # 3. returns: paper vs backtest on catalog data over the same hours
            if ds in paper and len(j):
                pnl, _, ppy, first = paper[ds]
                prep = prepare({ds: cat}, dep["cost_bps"], warmup=0, score_from=first, ppy=dep["ppy"])
                bt, _ = market_returns(prep.arrays[ds], signal(rules, cat["close"].to_numpy(float), dep["finalist"]), delay=dep["delay"])
                bt = pd.Series(bt, index=cat.index).dropna()
                both = pd.concat([pnl.rename("paper"), bt.rename("backtest")], axis=1, join="inner")
                if len(both) > 2:
                    m["returns"] = {"bars": len(both), "paper_total_pct": float(both.paper.sum() * 100),
                                    "backtest_total_pct": float(both.backtest.sum() * 100),
                                    "correlation": float(both.paper.corr(both.backtest))}
        except KeyError:
            m["data"] = "dataset not in the catalog: run the collector to compare live with downloaded history"
        report["markets"][ds] = m
    db.close()
    return report
