"""Shared helpers for the mini-projects: statistics, a small optimizer, catalog loaders, and chart-event builders."""
from __future__ import annotations

import io
import zipfile
from statistics import NormalDist

import numpy as np
import pandas as pd

from .. import catalog

N = NormalDist()


# ---------------------------------------------------------------------------------------------- data from the catalog
def closes(ds: str) -> pd.Series:
    return catalog.read(ds)["close"].astype(float)


def daily_log_returns(ds: str) -> pd.Series:
    """Daily log returns of any bars dataset (intraday bars are combined into UTC days)."""
    c = closes(ds)
    if len(c) > 2 and (c.index[1] - c.index[0]) < pd.Timedelta(hours=20):
        c = c.resample("1D").last().dropna()
    return np.log(c).diff().dropna()


def french(name: str) -> pd.DataFrame:
    """A Kenneth French Data Library table from the catalog (monthly, decimals). Import with `oql data pull-french`."""
    return catalog.read(f"french/{name}/1mo").astype(float)


def parse_french_zip(blob: bytes) -> pd.DataFrame:
    """The monthly block of a French library CSV zip, percent -> decimals, indexed by month end (UTC)."""
    z = zipfile.ZipFile(io.BytesIO(blob))
    lines = z.read(z.namelist()[0]).decode("latin-1").splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(","))
    rows = []
    for l in lines[start + 1:]:
        parts = [x.strip() for x in l.split(",")]
        if len(parts[0]) != 6 or not parts[0].isdigit():          # the monthly block ends at the annual table
            break
        rows.append(parts)
    cols = [c.strip() for c in lines[start].split(",")[1:]]
    d = pd.DataFrame(rows, columns=["m"] + cols)
    d.index = pd.DatetimeIndex(pd.to_datetime(d.pop("m"), format="%Y%m") + pd.offsets.MonthEnd(0)).tz_localize("UTC").rename("ts")
    d = d.astype(float)
    return d[(d > -99).all(axis=1)] / 100


# ---------------------------------------------------------------------------------------------- statistics
def ols(y, X, lags=0):
    """OLS with an intercept. Returns (coefficients, t-stats, R^2); coefficient 0 is the intercept.
    lags > 0 uses Newey-West standard errors (robust to autocorrelation and changing volatility)."""
    y = np.asarray(y, float); X = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for L in range(1, lags + 1):
        G = Xe[L:].T @ Xe[:-L]
        S += (1 - L / (lags + 1)) * (G + G.T)
    se = np.sqrt(np.diag(XtXi @ S @ XtXi) * len(y) / (len(y) - X.shape[1]))
    return b, b / se, 1 - e.var() / y.var()


def nelder_mead(f, x0, iters=2000, tol=1e-10, step=0.25):
    """Small derivative-free minimizer (so the projects need only numpy)."""
    x0 = np.asarray(x0, float); n = len(x0)
    sim = np.vstack([x0] + [x0 + step * np.abs(x0[i] if x0[i] else 1) * np.eye(n)[i] for i in range(n)])
    fs = np.array([f(x) for x in sim])
    for _ in range(iters):
        o = np.argsort(fs); sim, fs = sim[o], fs[o]
        if abs(fs[-1] - fs[0]) < tol * (abs(fs[0]) + tol):
            break
        c = sim[:-1].mean(0)
        xr = c + (c - sim[-1]); fr = f(xr)
        if fr < fs[0]:
            xe = c + 2 * (c - sim[-1]); fe = f(xe)
            sim[-1], fs[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < fs[-2]:
            sim[-1], fs[-1] = xr, fr
        else:
            xc = c + 0.5 * (sim[-1] - c); fc = f(xc)
            if fc < fs[-1]:
                sim[-1], fs[-1] = xc, fc
            else:
                sim[1:] = sim[0] + 0.5 * (sim[1:] - sim[0]); fs[1:] = [f(x) for x in sim[1:]]
    return sim[np.argmin(fs)], fs.min()


def bs_call(S, K, r, T, sigma):
    """Black-Scholes call price (no dividends). Works on arrays."""
    S, K, sigma = np.asarray(S, float), np.asarray(K, float), np.asarray(sigma, float)
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    cdf = np.vectorize(N.cdf)
    return S * cdf(d1) - K * np.exp(-r * T) * cdf(d1 - sigma * np.sqrt(T))


# ---------------------------------------------------------------------------------------------- chart events
def line(cid, title, series, **kw):
    """series: [{"name", "x", "y", "dash": bool, "points": bool}]. kw: xlabel, ylabel, x ("time"|"num"),
    x_log, y_log, x_range, y_range, y_format, vlines [{x, label}], hlines [{y, label}], bands [{x0, x1, label}]."""
    return {"chart": cid, "kind": "line", "title": title, "series": series, **kw}


def scatter(cid, title, series, **kw):
    return {"chart": cid, "kind": "scatter", "title": title, "series": series, **kw}


def bars(cid, title, categories, series, **kw):
    """Grouped bars: categories on x, series [{"name", "y", "err"?}]."""
    return {"chart": cid, "kind": "bar", "title": title, "categories": list(categories), "series": series, **kw}


def hist(cid, title, values, bins=40, range_=None, **kw):
    """A histogram, binned here so the page only draws. kw: vlines, xlabel, overlay [{"name", "values"}]."""
    v = np.asarray(values, float); v = v[np.isfinite(v)]
    lo, hi = range_ or ((float(v.min()), float(v.max())) if len(v) else (0.0, 1.0))
    marks = [float(m["x"]) for m in kw.get("vlines", []) if np.isfinite(float(m["x"]))]
    if marks and not range_:                                    # a marker (the real result) is always on the chart
        lo, hi = min([lo] + marks), max([hi] + marks)
        pad = (hi - lo) * 0.04; lo, hi = lo - pad, hi + pad
    if hi <= lo:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, bins + 1)
    out = {"chart": cid, "kind": "hist", "title": title, "edges": edges.tolist(),
           "series": [{"name": kw.pop("name", "count"), "counts": np.histogram(np.clip(v, lo, hi), edges)[0].tolist()}], **kw}
    for o in kw.get("overlay", []):
        vv = np.asarray(o["values"], float); vv = vv[np.isfinite(vv)]
        out["series"].append({"name": o["name"], "counts": np.histogram(np.clip(vv, lo, hi), edges)[0].tolist()})
    out.pop("overlay", None)
    return out


def heatmap(cid, title, z, xcats, ycats, diverging=False, unit="", **kw):
    return {"chart": cid, "kind": "heatmap", "title": title, "z": np.asarray(z, float).tolist(), "xcats": list(xcats),
            "ycats": list(ycats), "diverging": diverging, "unit": unit, **kw}


def table(tid, title, columns, rows):
    return {"table": tid, "title": title, "columns": columns, "rows": rows}


def stat(key, label, value, fmt=".2f", unit=""):
    return {"stat": key, "label": label, "value": value, "fmt": fmt, "unit": unit}


def check(name, passed, detail):
    return {"check": name, "passed": bool(passed), "detail": detail}


def times_ms(index) -> list:
    return [int(t.value // 1_000_000) for t in pd.DatetimeIndex(index)]
