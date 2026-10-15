"""`oql collect`: keep a local copy of free market data current, incrementally. Built to run unattended on a
Raspberry Pi (a systemd timer runs `oql collect run` once a day) and to be synced to the research machine.

    oql collect init            write collect.json with a sensible starting set of jobs
    oql collect run [--job N]   run every job that is due (or one job); --force ignores the schedule
    oql collect status          last run, rows and errors per job

Every downloaded file is cached under catalog/_raw/ and never fetched twice; a job only downloads what is new,
then rebuilds its datasets from the cache. Binance files are checked against Binance's published SHA-256.

Job types
  binance_klines   candles from data.binance.vision (spot or futures/um), monthly files + daily for the current month
  binance_funding  USD-M perpetual funding-rate history (futures/um monthly files)
  fred             FRED series (CSV endpoint, no key)
  cftc_cot         CFTC Commitments of Traders yearly files (legacy, disaggregated or financial futures)
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import time
import traceback
import zipfile

import pandas as pd

from . import catalog, sources

DEFAULT_JOBS = {"jobs": [
    {"name": "binance-spot-1h", "type": "binance_klines", "market": "spot", "interval": "1h", "start": "2018-01",
     "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"], "every_hours": 24},
    {"name": "binance-spot-1m", "type": "binance_klines", "market": "spot", "interval": "1m", "start": "2022-01",
     "symbols": ["BTCUSDT", "ETHUSDT"], "every_hours": 24},
    {"name": "binance-perp-1h", "type": "binance_klines", "market": "futures/um", "interval": "1h", "start": "2020-01",
     "symbols": ["BTCUSDT", "ETHUSDT"], "every_hours": 24},
    {"name": "binance-funding", "type": "binance_funding", "start": "2020-01", "symbols": ["BTCUSDT", "ETHUSDT"],
     "every_hours": 24},
    {"name": "fred-daily", "type": "fred", "every_hours": 24,
     "series": ["DGS10", "DGS2", "DGS3MO", "T10Y2Y", "DFF", "VIXCLS", "BAMLH0A0HYM2", "DTWEXBGS", "DCOILWTICO"]},
    {"name": "fred-monthly", "type": "fred", "every_hours": 168, "series": ["CPIAUCSL", "PAYEMS", "UNRATE", "INDPRO", "M2SL"]},
    {"name": "cftc-financial", "type": "cftc_cot", "report": "financial", "start_year": 2010, "every_hours": 168},
    {"name": "cftc-legacy", "type": "cftc_cot", "report": "legacy", "start_year": 2010, "every_hours": 168},
]}

CFTC_FILES = {   # yearly zip names on cftc.gov/files/dea/history/
    "legacy": "deacot{y}.zip", "disaggregated": "fut_disagg_txt_{y}.zip", "financial": "fut_fin_txt_{y}.zip"}


def lab_paths(lab=None):
    from .board import lab_dir
    lab = lab or lab_dir()
    return os.path.join(lab, "collect.json"), os.path.join(lab, "collect_state.json")


def init(lab=None) -> str:
    cfg, _ = lab_paths(lab)
    if os.path.exists(cfg):
        raise FileExistsError(f"{cfg} already exists")
    with open(cfg, "w") as f:
        json.dump(DEFAULT_JOBS, f, indent=1)
    return cfg


def _raw(*parts) -> str:
    p = os.path.join(catalog.root(), "_raw", *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def _cached(url: str, path: str, getter, checksum: bool = False) -> bytes | None:
    """Return the file at `path`, downloading it from `url` the first time. None if it does not exist upstream."""
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    import urllib.error
    try:
        blob = getter(url)
    except urllib.error.HTTPError as e:
        if e.code in (403, 404):                          # not there (S3 answers 403 for a missing key)
            return None
        raise                                             # anything else: fail the job, retry next run
    except FileNotFoundError:
        return None
    if checksum:
        try:
            expected = getter(url + ".CHECKSUM").decode().split()[0].lower()
            if hashlib.sha256(blob).hexdigest() != expected:
                raise IOError(f"checksum mismatch for {url}")
        except IOError:
            raise
        except Exception:
            pass                                          # no checksum published: keep the file anyway
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(blob)
    os.replace(tmp, path)
    return blob


# ---------------------------------------------------------------------------------------------- Binance
def _binance_files(market, kind, symbol, interval, start, getter, today=None):
    """All monthly files from `start`, plus daily files for months whose monthly file is not out yet."""
    today = pd.Timestamp(today or pd.Timestamp.utcnow().date())
    base = f"https://data.binance.vision/data/{market}"
    tag = f"{symbol}-{interval}" if interval else f"{symbol}-{kind}"
    sub = f"{kind}/{symbol}/{interval}" if interval else f"{kind}/{symbol}"
    blobs, missing = [], []
    recent = today.to_period("M") - 1                     # months before this one have their monthly file by now
    for m in pd.period_range(start, today.to_period("M"), freq="M"):
        name = f"{tag}-{m.year}-{m.month:02d}.zip"
        path = _raw("binance", market, kind, symbol, interval or "-", name)
        if os.path.exists(path + ".404"):                 # known not to exist (e.g. before the coin was listed)
            continue
        b = None
        if m < today.to_period("M"):
            b = _cached(f"{base}/monthly/{sub}/{name}", path, getter, True)
        if b is not None:
            blobs.append(b); continue
        if m < recent:                                     # old and absent: remember, never ask again
            open(path + ".404", "w").close()
            missing.append(str(m)); continue
        if kind != "klines":                              # funding files are monthly only
            missing.append(str(m)); continue
        for d in pd.date_range(m.start_time, min(m.end_time.normalize(), today - pd.Timedelta(days=1)), freq="D"):
            dn = f"{tag}-{d:%Y-%m-%d}.zip"
            db = _cached(f"{base}/daily/{sub}/{dn}", _raw("binance", market, kind, symbol, interval, "daily", dn), getter, True)
            if db is not None:
                blobs.append(db)
            elif d < today - pd.Timedelta(days=2):
                missing.append(str(d.date()))
    return blobs, missing


def parse_funding_zip(blob: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        d = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])))
    d.columns = [c.strip().lower() for c in d.columns]
    tcol = next(c for c in d.columns if "time" in c)
    rcol = next(c for c in d.columns if "rate" in c)
    t = pd.to_numeric(d[tcol])
    unit = "us" if t.iloc[0] > 1e14 else "ms"
    return pd.DataFrame({"funding_rate": pd.to_numeric(d[rcol]).to_numpy()},
                        index=pd.DatetimeIndex(pd.to_datetime(t, unit=unit, utc=True), name="ts"))


def job_binance_klines(job, getter):
    out = []
    for sym in job["symbols"]:
        blobs, missing = _binance_files(job["market"], "klines", sym, job["interval"], job.get("start", "2018-01"), getter)
        if not blobs:
            out.append(f"{sym}: nothing downloaded"); continue
        df = pd.concat([sources.parse_binance_zip(b) for b in blobs])
        src = "binance" if job["market"] == "spot" else "binance-" + job["market"].replace("/", "-")
        m = catalog.write(f"{src}/{sym}/{job['interval']}", df, license="public",
                          origin=f"https://data.binance.vision/data/{job['market']} klines",
                          notes=("missing: " + ", ".join(missing[:20])) if missing else "")
        out.append(f"{m.id}: {m.rows} rows to {m.end[:16]}" + (f" ({len(missing)} missing)" if missing else ""))
    return out


def job_binance_funding(job, getter):
    out = []
    for sym in job["symbols"]:
        blobs, missing = _binance_files("futures/um", "fundingRate", sym, None, job.get("start", "2020-01"), getter)
        if not blobs:
            out.append(f"{sym}: nothing downloaded"); continue
        df = pd.concat([parse_funding_zip(b) for b in blobs]).sort_index()
        df = df[~df.index.duplicated()]
        m = catalog.write(f"binance-funding/{sym}/8h", df, license="public", kind="table",
                          origin="https://data.binance.vision/data/futures/um fundingRate")
        out.append(f"{m.id}: {m.rows} rows to {m.end[:16]}")
    return out


def job_fred(job, getter):
    out = []
    for s in job["series"]:
        m = sources.fred(s, getter=getter)
        out.append(f"{m.id}: {m.rows} rows to {m.end[:10]}")
    return out


def parse_cot(blob: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        d = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), low_memory=False)
    d.columns = [c.strip() for c in d.columns]
    dcol = next((c for c in d.columns if "YYYY-MM-DD" in c or c.lower().startswith("report_date")), None)
    if dcol is None:
        raise ValueError(f"no report-date column in {list(d.columns)[:6]}")
    asof = pd.to_datetime(d[dcol], utc=True)
    d = d.drop(columns=[dcol])
    d["released"] = (asof + pd.Timedelta(days=3)).dt.strftime("%Y-%m-%d")   # positions as of Tuesday, published Friday
    return d.set_index(pd.DatetimeIndex(asof, name="ts")).sort_index()


def job_cftc_cot(job, getter):
    report = job.get("report", "legacy")
    pattern = CFTC_FILES[report]
    this_year = pd.Timestamp.utcnow().year
    parts, missing = [], []
    for y in range(job.get("start_year", 2010), this_year + 1):
        name = pattern.format(y=y)
        path = _raw("cftc", report, name)
        if y == this_year and os.path.exists(path) and time.time() - os.path.getmtime(path) > 6 * 86400:
            os.remove(path)                               # the current year's file grows every week: refresh it
        b = _cached(f"https://www.cftc.gov/files/dea/history/{name}", path, getter)
        if b is None:
            missing.append(str(y)); continue
        parts.append(parse_cot(b))
    if not parts:
        return [f"cftc {report}: nothing downloaded (missing {missing})"]
    df = pd.concat(parts).sort_index()
    m = catalog.write(f"cftc/{report}/1w", df, license="public", kind="table", origin="https://www.cftc.gov/files/dea/history/",
                      notes="index = positions' as-of date (Tuesday); column 'released' = publication date (Friday). "
                            "Use 'released' to avoid look-ahead.")
    return [f"{m.id}: {m.rows} rows to {m.end[:10]}" + (f" (missing {missing})" if missing else "")]


JOBS = {"binance_klines": job_binance_klines, "binance_funding": job_binance_funding, "fred": job_fred, "cftc_cot": job_cftc_cot}


def run(lab=None, only: str | None = None, force: bool = False, getter=None, verbose=True) -> dict:
    getter = getter or sources.fetch
    cfg_path, state_path = lab_paths(lab)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError("no collect.json here: run `oql collect init` first")
    with open(cfg_path) as f:
        jobs = json.load(f)["jobs"]
    state = {}
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
    for job in jobs:
        name = job["name"]
        if only and name != only:
            continue
        last = state.get(name, {}).get("ok_utc_epoch", 0)
        if not force and not only and time.time() - last < job.get("every_hours", 24) * 3600 - 300:
            continue
        t0 = time.time()
        if verbose:
            print(f"[{name}] running", flush=True)
        try:
            lines = JOBS[job["type"]](job, getter)
            state[name] = {"ok_utc_epoch": time.time(), "ok_utc": time.strftime("%Y-%m-%d %H:%M", time.gmtime()),
                           "seconds": round(time.time() - t0, 1), "result": lines, "error": ""}
        except Exception as e:
            state[name] = {**state.get(name, {}), "error": f"{type(e).__name__}: {e}",
                           "error_utc": time.strftime("%Y-%m-%d %H:%M", time.gmtime()), "trace": traceback.format_exc()[-1500:]}
        if verbose:
            for l in state[name].get("result", []) if not state[name]["error"] else [state[name]["error"]]:
                print(f"    {l}", flush=True)
        with open(state_path + ".tmp", "w") as f:
            json.dump(state, f, indent=1)
        os.replace(state_path + ".tmp", state_path)
    return state


def status(lab=None) -> str:
    cfg_path, state_path = lab_paths(lab)
    if not os.path.exists(state_path):
        return "nothing collected yet"
    with open(state_path) as f:
        state = json.load(f)
    lines = []
    for name, s in state.items():
        flag = "ERROR" if s.get("error") else "ok"
        lines.append(f"{flag:5s} {name:18s} last ok {s.get('ok_utc', '-'):16s} {s.get('seconds', '')}s")
        if s.get("error"):
            lines.append(f"      {s['error']}")
    return "\n".join(lines)
