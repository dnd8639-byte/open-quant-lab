"""Loaders that put outside data into the catalog. Each returns the catalog Meta of what it wrote.

Free, no account:
  fred(series)                 Federal Reserve Economic Data, any series id (DGS10, CPIAUCSL, ...)
  binance(symbol, interval)    Binance public bulk files (data.binance.vision), monthly zips of klines
Your own files:
  csv_bars(path, ...)          any OHLCV csv (Yahoo export, the mcpt-lab BTC file, ...)
  calendar(path)               release calendar csv with columns date,event (edge-engine/futures/events_full.csv)
  pi_hourly(path)              the Pi's order-book features (micro/data/hourly.csv)
Licensed (stays on your machine):
  databento_front(dir, roots)  per-contract daily files -> front-contract continuous bars, roll-safe

Network calls go through `fetch(url) -> bytes`, so tests can pass a fake.
"""
from __future__ import annotations

import glob
import io
import os
import zipfile

import numpy as np
import pandas as pd

from . import catalog

UA = {"User-Agent": "open-quant-lab/0.1 (research; contact via GitHub)"}


def _ssl_context():
    """Python from python.org on macOS ships without trusted certificates; certifi supplies them."""
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def fetch(url: str, timeout: int = 60) -> bytes:
    import urllib.request
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as r:
        return r.read()


# ---------------------------------------------------------------------------------------------- FRED
def parse_fred(text: str, series: str) -> pd.DataFrame:
    d = pd.read_csv(io.StringIO(text))
    tcol = next(c for c in d.columns if c.lower() in ("date", "observation_date"))
    vcol = next(c for c in d.columns if c != tcol)
    out = pd.DataFrame({"value": pd.to_numeric(d[vcol].replace(".", np.nan), errors="coerce").to_numpy()},
                       index=pd.DatetimeIndex(pd.to_datetime(d[tcol], utc=True), name="ts"))
    return out.dropna()


def fred(series: str, getter=fetch):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    df = parse_fred(getter(url).decode(), series)
    return catalog.write(f"fred/{series}/1d", df, license="public", origin=url, kind="table",
                         notes="FRED observation dates; values as first published may differ (no vintages)")


# ---------------------------------------------------------------------------------------------- Binance public files
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades",
              "taker_buy_base", "taker_buy_quote", "ignore"]


def parse_binance_zip(blob: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        raw = z.read(z.namelist()[0]).decode()
    first = raw.lstrip().split("\n", 1)[0].split(",", 1)[0].strip()
    has_header = not first.lstrip("-").isdigit()                    # futures files carry a header row; spot files do not
    d = pd.read_csv(io.StringIO(raw), header=0 if has_header else None)
    d = d.iloc[:, :12]
    d.columns = KLINE_COLS
    d = d.apply(pd.to_numeric, errors="coerce").dropna(subset=["open_time"])
    d = d.astype({"open_time": "int64"})
    return catalog.normalize_bars(d[["open_time", "open", "high", "low", "close", "volume", "trades",
                                     "taker_buy_base"]].rename(columns={"open_time": "ts"}))


def binance(symbol: str, interval: str = "1h", start: str = "2018-01", end: str | None = None, getter=fetch,
            market: str = "spot", verbose: bool = True):
    """Monthly kline files from data.binance.vision. Timestamps switch from ms to microseconds in 2025;
    both are handled. Months that do not exist yet (or are not published) are skipped and reported."""
    end = end or (pd.Timestamp.utcnow() - pd.offsets.MonthBegin(1)).strftime("%Y-%m")
    parts, missing = [], []
    for m in pd.period_range(start, end, freq="M"):
        url = (f"https://data.binance.vision/data/{market}/monthly/klines/{symbol}/{interval}/"
               f"{symbol}-{interval}-{m.year}-{m.month:02d}.zip")
        try:
            parts.append(parse_binance_zip(getter(url)))
            if verbose:
                print(f"  {symbol} {m}: {len(parts[-1])} bars", flush=True)
        except Exception as e:                                        # a missing month is reported, not fatal
            missing.append(f"{m} ({type(e).__name__})")
    if not parts:
        raise RuntimeError(f"no files downloaded for {symbol}; missing: {missing[:5]}")
    df = pd.concat(parts)
    return catalog.write(f"binance/{symbol}/{interval}", df, license="public",
                         origin=f"https://data.binance.vision ({market}, {symbol}, {interval})",
                         notes=("missing months: " + ", ".join(missing)) if missing else "")


# ---------------------------------------------------------------------------------------------- Kenneth French library
FRENCH = ["F-F_Research_Data_Factors", "F-F_Momentum_Factor", "F-F_Research_Data_5_Factors_2x3"]
FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}_CSV.zip"


def french(name: str, directory: str | None = None, getter=fetch):
    """A monthly factor table from the Kenneth French Data Library (free), as decimals. From a local zip in
    `directory` if given, otherwise downloaded."""
    from .projects._util import parse_french_zip
    if directory:
        with open(os.path.join(os.path.expanduser(directory), f"{name}_CSV.zip"), "rb") as f:
            blob = f.read()
    else:
        blob = getter(FRENCH_URL.format(name=name))
    df = parse_french_zip(blob)
    return catalog.write(f"french/{name}/1mo", df, license="public", origin=FRENCH_URL.format(name=name), kind="table",
                         notes="Kenneth R. French Data Library, monthly, percent converted to decimals; index = month end")


def import_projects(directory: str) -> list:
    """The mini-projects' own data files: a SPY option chain (Yahoo Finance: licensed, stays local) and a day of
    Binance.US BTC trades (public)."""
    d = os.path.expanduser(directory)
    out = []
    p = os.path.join(d, "spy_chain.csv")
    if os.path.exists(p):
        ch = pd.read_csv(p)
        ch.index = pd.DatetimeIndex(pd.to_datetime(ch["downloaded"].str.replace(" UTC", ""), utc=True), name="ts")
        out.append(catalog.write("options/SPY/chain", ch, license="licensed", origin=os.path.abspath(p), kind="table",
                                 notes="SPY option chain snapshot via Yahoo Finance (yfinance); not for redistribution"))
    p = os.path.join(d, "btc_trades.csv")
    if os.path.exists(p):
        tr = pd.read_csv(p)
        tr.index = pd.DatetimeIndex(pd.to_datetime(tr.pop("timestamp"), unit="ms", utc=True), name="ts")
        out.append(catalog.write("binanceus/BTCUSDT/trades", tr, license="public", origin=os.path.abspath(p), kind="table",
                                 notes="Binance.US BTC/USDT public trades (one day)"))
    if not out:
        raise FileNotFoundError(f"no spy_chain.csv or btc_trades.csv in {d}")
    return out


# ---------------------------------------------------------------------------------------------- local files
def csv_bars(path: str, name: str, freq: str, source: str = "csv", license: str = "own"):
    df = pd.read_csv(path)
    return catalog.write(f"{source}/{name}/{freq}", df, license=license, origin=os.path.abspath(path))


def calendar(path: str, name: str = "us_releases"):
    d = pd.read_csv(path, parse_dates=["date"])
    d = d.set_index(pd.DatetimeIndex(d.pop("date"), tz="UTC").rename("ts")).sort_index()
    return catalog.write(f"calendar/{name}/1d", d, license="public", origin=os.path.abspath(path), kind="table",
                         notes="scheduled release dates known in advance (FOMC, CPI, payrolls, EIA)")


def pi_hourly(path: str):
    d = pd.read_csv(path, parse_dates=["hour"])
    d = d.set_index(pd.DatetimeIndex(d.pop("hour"), tz="UTC").rename("ts")).sort_index()
    return catalog.write("pi/book_features/1h", d, license="own", origin=os.path.abspath(path), kind="table",
                         notes="hourly order-book and trade-flow features from the Raspberry Pi recorder")


# ---------------------------------------------------------------------------------------------- Databento (licensed)
def front_contract(per_contract: pd.DataFrame, buffer: int = 5) -> pd.DataFrame:
    """Daily bars of the contract with the most volume YESTERDAY, excluding contracts in their last `buffer`
    trading days. All of a day's numbers come from that one contract; `ret` compares that contract's close with
    its OWN previous close, so a roll never enters a return. The returned OHLC is back-adjusted from those
    returns (ratio-adjusted), so it is continuous and signals see no roll jumps."""
    df = per_contract.copy()
    df["date"] = pd.to_datetime(df["ts_event"].astype(str).str[:10])
    df["key"] = df["symbol"].astype(str) + "_" + df["instrument_id"].astype(str)   # CME reuses 1-digit years
    wd = df["date"].dt.dayofweek
    df.loc[wd == 6, "date"] += pd.Timedelta(days=1)                                 # Sunday evening -> Monday
    df.loc[wd == 5, "date"] += pd.Timedelta(days=2)
    df = df.sort_values("ts_event")
    g = df.groupby(["date", "key"])
    bars = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                         "close": g["close"].last(), "volume": g["volume"].sum()}).reset_index()
    P = {c: bars.pivot(index="date", columns="key", values=c).sort_index() for c in ("open", "high", "low", "close", "volume")}
    close = P["close"]
    pos = pd.Series(np.arange(len(close)), index=close.index)
    last_pos = close.apply(lambda c: pos[c.last_valid_index()])
    elig = pd.DataFrame(pos.values[:, None] <= (last_pos.values[None, :] - buffer), index=close.index, columns=close.columns)
    v = P["volume"].fillna(0).where(elig & close.notna(), -1.0)
    held = v.idxmax(axis=1).where(v.max(axis=1) > 0).shift(1)
    ok = held.notna()
    dates, keys = held.index[ok], held[ok].to_numpy()
    cols = pd.Index(pd.unique(keys))
    rows_i, cols_i = close.index.get_indexer(dates), cols.get_indexer(keys)
    take = lambda M: M.reindex(columns=cols).to_numpy()[rows_i, cols_i]
    o, h, l, c, vol, pc = take(P["open"]), take(P["high"]), take(P["low"]), take(close), take(P["volume"]), take(close.shift(1))
    raw = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": vol, "ret": np.log(c / pc)}, index=dates)
    raw = raw.dropna(subset=["close"])
    raw = raw[(raw[["open", "high", "low", "close"]] > 0).all(axis=1)]
    r = raw["ret"].fillna(0.0)
    adj_c = raw["close"].iloc[0] * np.exp(r.cumsum())
    k = adj_c / raw["close"]
    out = pd.DataFrame({"open": raw["open"] * k, "high": raw["high"] * k, "low": raw["low"] * k, "close": adj_c,
                        "volume": raw["volume"]}, index=raw.index)
    return out


def databento_front(directory: str, roots: list[str] | None = None):
    metas = []
    for p in sorted(glob.glob(os.path.join(directory, "*.csv"))):
        root_sym = os.path.basename(p)[:-4]
        if roots and root_sym not in roots:
            continue
        df = front_contract(pd.read_csv(p))
        metas.append(catalog.write(f"databento/{root_sym}/1d", df, license="licensed", origin=os.path.abspath(p),
                                   notes="front contract by previous-day volume, 5-day expiry buffer, ratio back-adjusted; "
                                         "Sunday sessions merged into Monday. LICENSED: never publish."))
    return metas
