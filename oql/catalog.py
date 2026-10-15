"""The data catalog: every dataset the lab uses, in one place, with a record of where it came from.

Layout (all under one root folder, default ./catalog, or $OQL_CATALOG):

    catalog/<source>/<name>/<freq>.parquet      the data (or .csv.gz when pyarrow is not installed)
    catalog/<source>/<name>/<freq>.meta.json    where it came from, its licence, when, how many rows, a SHA-256

A dataset id is "<source>/<name>/<freq>", e.g. "binance/BTCUSDT/1h", "fred/DGS10/1d", "databento/ES/1d".

Bars share one schema: a UTC DatetimeIndex named "ts" (the bar's OPEN time) and float columns
open, high, low, close, volume. Other tables (release calendars, order-book features) keep their own columns
and declare kind="table" in their metadata.

Every dataset carries a licence tag: "public" (free to redistribute), "own" (you made it), or "licensed"
(paid data you may use but never publish). `oql data list` shows it; the repo's .gitignore keeps the whole
catalog out of git, so licensed data cannot be committed by accident.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

BAR_COLUMNS = ["open", "high", "low", "close", "volume"]
LICENSES = ("public", "own", "licensed")
FREQS = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D", "1w": "7D"}


def root() -> str:
    return os.environ.get("OQL_CATALOG", os.path.join(os.getcwd(), "catalog"))


def _has_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


@dataclass
class Meta:
    id: str
    kind: str                 # "bars" or "table"
    license: str              # public | own | licensed
    origin: str               # URL or local path it was built from
    fetched_utc: str
    rows: int
    start: str
    end: str
    columns: list = field(default_factory=list)
    sha256: str = ""
    fmt: str = "parquet"
    notes: str = ""


def _paths(ds_id: str):
    source, name, freq = split_id(ds_id)
    base = os.path.join(root(), source, name)
    return base, os.path.join(base, f"{freq}.meta.json")


def split_id(ds_id: str):
    parts = ds_id.strip("/").split("/")
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"dataset id must look like source/name/freq, got {ds_id!r}")
    return parts


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------- write / read
def normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce any OHLCV frame to the catalog schema: UTC index 'ts', float columns, sorted, no duplicates."""
    d = df.copy()
    d.columns = [str(c).strip().lower() for c in d.columns]
    if not isinstance(d.index, pd.DatetimeIndex):
        tcol = next((c for c in ("ts", "date", "datetime", "time", "timestamp", "open_time") if c in d.columns), None)
        if tcol is None:
            raise ValueError("no time column found (expected ts/date/datetime/time/timestamp/open_time)")
        t = d.pop(tcol)
        if pd.api.types.is_numeric_dtype(t):
            v = float(t.iloc[0])
            unit = "us" if v > 1e14 else "ms" if v > 1e11 else "s"
            t = pd.to_datetime(t, unit=unit, utc=True)
        else:
            t = pd.to_datetime(t, utc=True)
        d.index = t
    idx = d.index
    d.index = (idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize("UTC")).rename("ts")
    if "volume" not in d.columns:
        d["volume"] = 0.0
    missing = [c for c in BAR_COLUMNS if c not in d.columns]
    if missing:
        raise ValueError(f"missing columns {missing}")
    extra = [c for c in d.columns if c not in BAR_COLUMNS]
    d = d[BAR_COLUMNS + extra].sort_index()
    d = d[~d.index.duplicated(keep="last")]
    for c in BAR_COLUMNS:
        d[c] = pd.to_numeric(d[c], errors="coerce").astype(float)
    return d.dropna(subset=["close"])


def write(ds_id: str, df: pd.DataFrame, *, license: str, origin: str, kind: str = "bars", notes: str = "") -> Meta:
    if license not in LICENSES:
        raise ValueError(f"license must be one of {LICENSES}")
    source, name, freq = split_id(ds_id)
    if kind == "bars":
        df = normalize_bars(df)
    base, meta_path = _paths(ds_id)
    os.makedirs(base, exist_ok=True)
    fmt = "parquet" if _has_parquet() else "csv.gz"
    data_path = os.path.join(base, f"{freq}.{fmt}")
    tmp = data_path + ".tmp"
    if fmt == "parquet":
        df.to_parquet(tmp)
    else:
        df.to_csv(tmp, compression="gzip")
    os.replace(tmp, data_path)                         # never leave a half-written file behind
    for other in ("parquet", "csv.gz"):                # one format per dataset
        p = os.path.join(base, f"{freq}.{other}")
        if other != fmt and os.path.exists(p):
            os.remove(p)
    ix = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.DatetimeIndex([])
    meta = Meta(id=ds_id, kind=kind, license=license, origin=origin,
                fetched_utc=time.strftime("%Y-%m-%d %H:%M", time.gmtime()), rows=len(df),
                start=str(ix.min()) if len(ix) else "", end=str(ix.max()) if len(ix) else "",
                columns=[str(c) for c in df.columns], sha256=_sha(data_path), fmt=fmt, notes=notes)
    with open(meta_path, "w") as f:
        json.dump(asdict(meta), f, indent=1)
    return meta


def meta(ds_id: str) -> Meta:
    _, meta_path = _paths(ds_id)
    if not os.path.exists(meta_path):
        raise KeyError(f"{ds_id} is not in the catalog at {root()} (see `oql data list`)")
    with open(meta_path) as f:
        return Meta(**json.load(f))


def read(ds_id: str, start=None, end=None) -> pd.DataFrame:
    m = meta(ds_id)
    base, _ = _paths(ds_id)
    path = os.path.join(base, f"{split_id(ds_id)[2]}.{m.fmt}")
    if m.fmt == "parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, compression="gzip", index_col=0)
        if m.kind == "bars" or str(df.index.name) in ("ts", "date", "hour"):
            df.index = pd.to_datetime(df.index, utc=True)
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        if start is not None:
            df = df[df.index >= pd.Timestamp(start, tz="UTC")]
        if end is not None:
            df = df[df.index < pd.Timestamp(end, tz="UTC")]
    return df


def list_all() -> list[Meta]:
    out = []
    for dirpath, _, files in os.walk(root()):
        for f in sorted(files):
            if f.endswith(".meta.json"):
                with open(os.path.join(dirpath, f)) as fh:
                    out.append(Meta(**json.load(fh)))
    return sorted(out, key=lambda m: m.id)


# ---------------------------------------------------------------------------------------------- checks
def check(ds_id: str) -> dict:
    """Data-quality report. Problems are reported, never silently fixed."""
    m = meta(ds_id)
    base, _ = _paths(ds_id)
    path = os.path.join(base, f"{split_id(ds_id)[2]}.{m.fmt}")
    issues, warnings = [], []                          # issues = broken data; warnings = worth a look
    if _sha(path) != m.sha256:
        issues.append("file changed since it was written (SHA-256 mismatch)")
    df = read(ds_id)
    rep = dict(id=ds_id, rows=len(df), license=m.license, issues=issues, warnings=warnings)
    if m.kind != "bars":
        return rep
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        issues.append("time index not strictly increasing")
    bad_px = int((df[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())
    if bad_px:
        issues.append(f"{bad_px} bars with a non-positive price")
    incons = int(((df["low"] > df[["open", "close"]].min(axis=1) + 1e-12) |
                  (df["high"] < df[["open", "close"]].max(axis=1) - 1e-12)).sum())
    if incons:
        issues.append(f"{incons} bars where open/close fall outside low-high")
    nan = int(df[BAR_COLUMNS].isna().any(axis=1).sum())
    if nan:
        issues.append(f"{nan} bars with missing values")
    freq = split_id(ds_id)[2]
    if freq in FREQS and len(df) > 2:
        step = pd.Timedelta(FREQS[freq])
        gaps = df.index.to_series().diff().dropna()
        big = gaps[gaps > step * (3 if freq != "1d" else 4)]      # a weekend is not a gap for daily bars
        rep["gaps"] = int(len(big))
        rep["largest_gap"] = str(big.max()) if len(big) else "-"
        if len(big):
            warnings.append(f"{len(big)} gaps longer than expected (largest {big.max()})")
    with np.errstate(invalid="ignore", divide="ignore"):       # bad prices are already reported above
        r = np.log(df["close"].where(df["close"] > 0)).diff().dropna()
    if len(r) > 50:
        z = (r - r.median()).abs() / (1.4826 * (r - r.median()).abs().median() + 1e-12)
        jumps = int((z > 25).sum())
        rep["jumps"] = jumps
        if jumps:
            warnings.append(f"{jumps} one-bar moves over 25 robust standard deviations (check for bad ticks or unadjusted rolls)")
    return rep
