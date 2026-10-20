"""The efficiency lab: can the NEXT period's efficiency be predicted from what is known before it starts?

Efficiency ratio (ER) of a period = |net move| / total distance travelled inside it. Near 1 = price went one way
(trend rules are paid); near 0 = back and forth (trend rules pay costs for nothing). The earlier futures research found
next-day ER unpredictable from price features (OOS R^2 0.0014) while volatility is predictable; an opening-range
breakout on BTC would be paid if a forecast reached a correlation of about 0.3 with realized ER. This module reruns
that question inside the lab, with every data source the collector now has, and keeps the lab's rules:

  * development data only: periods before each target dataset's 70% split (the holdout stays untouched);
  * walk-forward: the model is refit on all earlier periods and predicts the next block, never the past;
  * every feature is known before the period it predicts starts (as-of joins, publication lags);
  * the order-book study's confirmation sample (Pi features from its lock date on) is sealed;
  * known-answer checks first (planted predictability must be found, noise must not), and every study is counted
    in the ledger.

Targets:  daily  = path ER over the UTC day from hourly bars (or |C-O|/(H-L) on daily bars, the validated proxy)
          hourly = path ER over each hour from 1-minute bars.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import catalog

SEALED = {"pi/book_features/1h": "2026-10-20"}      # micro study: confirmation data, unusable until its test has run
NEEDED_CORR = 0.30                                   # correlation an opening-range breakout needed (research/orb_feasibility.py)
GROUPS = {
    "price": "past efficiency, volatility, range, volume, time of day/week",
    "calendar": "scheduled US releases (FOMC, CPI, payrolls, EIA), known in advance",
    "funding": "perpetual-futures funding rate (as of the period start)",
    "basis": "perpetual-futures price vs spot at the previous close",
    "macro": "VIX and the 10y-2y yield slope (previous US day)",
    "positioning": "CFTC leveraged-fund positioning in CME bitcoin futures (only once published)",
    "orderbook": "the Pi's hourly order-book and trade-flow features (before the study's lock date)",
}


# ---------------------------------------------------------------------------------------------- targets
def path_er(bars: pd.DataFrame, period: str, min_bars: int) -> pd.DataFrame:
    """Per period: path ER, realized volatility, log range, |return|, volume. Index = period start (UTC)."""
    b = bars[["open", "high", "low", "close", "volume"]].astype(float)
    key = b.index.floor(period)
    c, o = b["close"], b["open"]
    first = pd.Series(key, index=b.index).ne(pd.Series(key, index=b.index).shift())
    step = (c - c.shift()).abs().where(~first, (c - o).abs())
    lr = np.log(c / c.shift()).where(~first, np.log(c / o))
    g = pd.DataFrame({"step": step, "lr2": lr ** 2, "k": key, "o": o, "c": c, "h": b["high"], "l": b["low"], "v": b["volume"]}).groupby("k")
    out = pd.DataFrame({"path": g["step"].sum(), "open": g["o"].first(), "close": g["c"].last(), "high": g["h"].max(),
                        "low": g["l"].min(), "volume": g["v"].sum(), "rv": np.sqrt(g["lr2"].sum()), "n": g["step"].size()})
    out = out[out["n"] >= min_bars]
    out["er"] = ((out["close"] - out["open"]).abs() / out["path"]).where(out["path"] > 0)
    out["log_range"] = np.log(out["high"] / out["low"])
    out["abs_ret"] = np.log(out["close"] / out["open"]).abs()
    return out.dropna(subset=["er"])


def bar_step(index: pd.DatetimeIndex) -> pd.Timedelta:
    d = pd.Series(index).diff().median()
    return d if pd.notna(d) else pd.Timedelta(days=1)


def periods_from_bars(bars: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """(per-period frame with 'er', horizon) from any bars: minute bars -> hourly ER, hourly bars -> daily path ER,
    daily bars -> |C-O|/(H-L), the proxy that ranks days like the hourly-path ER (rank correlation 0.94)."""
    step = bar_step(bars.index)
    if step <= pd.Timedelta(minutes=1):
        return path_er(bars, "1h", 50), "hourly"
    if step <= pd.Timedelta(hours=1):
        return path_er(bars, "1D", 20), "daily"
    d = bars.astype(float)
    out = pd.DataFrame({"open": d["open"], "close": d["close"], "high": d["high"], "low": d["low"], "volume": d["volume"]})
    out["er"] = ((d["close"] - d["open"]).abs() / (d["high"] - d["low"])).where(d["high"] > d["low"])
    out["rv"] = np.log(d["high"] / d["low"]) / 2                       # range-based volatility stand-in
    out["log_range"] = np.log(d["high"] / d["low"]); out["abs_ret"] = np.log(d["close"] / d["open"]).abs()
    out.index = out.index.floor("1D")
    out = out[~out.index.duplicated(keep="last")]
    return out.dropna(subset=["er"]), "daily"


def target(ds: str) -> tuple[pd.DataFrame, str]:
    """(per-period frame with 'er', horizon name) for a bars dataset in the catalog."""
    return periods_from_bars(catalog.read(ds))


def dev_cut(ds: str, frac: float = 0.7) -> pd.Timestamp:
    """The first date of the holdout of this dataset: everything from here on is out of bounds."""
    idx = catalog.read(ds).index
    return idx[int(len(idx) * frac)]


def dev_bars(ds: str, frac: float = 0.7) -> pd.DataFrame:
    """The development part of a bars dataset (its first 70% of bars). Nothing after it is ever used here."""
    b = catalog.read(ds)
    return b[b.index < b.index[int(len(b) * frac)]]


# ---------------------------------------------------------------------------------------------- features
def _symbol(ds: str) -> str:
    parts = ds.strip("/").split("/")
    return parts[1] if len(parts) == 3 else ds


def _read(ds: str) -> pd.DataFrame:
    """A catalog table with a UTC DatetimeIndex, however it was stored."""
    d = catalog.read(ds)
    if not isinstance(d.index, pd.DatetimeIndex):
        d.index = pd.to_datetime(d.index, utc=True)
    return d


def _asof(index: pd.DatetimeIndex, series: pd.Series, lag: pd.Timedelta = pd.Timedelta(0), name: str = "x") -> pd.Series:
    """Value of `series` known strictly before each index time minus `lag`."""
    s = series.dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    left = pd.DataFrame({"t": (index - lag).astype("datetime64[ns, UTC]")}).sort_values("t")
    right = pd.DataFrame({"t": pd.DatetimeIndex(s.index).astype("datetime64[ns, UTC]"), name: s.to_numpy()})
    m = pd.merge_asof(left, right, on="t", allow_exact_matches=False)
    return pd.Series(m[name].to_numpy(), index=index[left.index], name=name).reindex(index)


def features(ds: str, per: pd.DataFrame, horizon: str, groups: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Feature table aligned to `per`'s index (period starts). Returns (X, notes about groups that were unavailable)."""
    idx, notes, cols = per.index, [], {}
    step = pd.Timedelta(hours=1) if horizon == "hourly" else pd.Timedelta(days=1)
    have = {m.id for m in catalog.list_all()}
    if "price" in groups:
        prev = per.shift(1)                                           # the period that just ended
        cols["er_lag1"] = prev["er"]
        cols["er_mean5"] = per["er"].shift(1).rolling(5).mean()
        cols["er_mean20"] = per["er"].shift(1).rolling(20).mean()
        cols["log_rv_lag1"] = np.log(prev["rv"].replace(0, np.nan))
        cols["log_rv_mean20"] = np.log(per["rv"].shift(1).rolling(20).mean().replace(0, np.nan))
        cols["log_range_lag1"] = prev["log_range"]
        cols["abs_ret_lag1"] = prev["abs_ret"]
        vol = per["volume"].replace(0, np.nan)
        cols["log_volume_rel"] = np.log(vol.shift(1) / vol.shift(1).rolling(20).mean())
        if horizon == "hourly":
            h = idx.hour.to_numpy()
            for k in (1, 2):
                cols[f"hour_sin{k}"], cols[f"hour_cos{k}"] = np.sin(2 * np.pi * k * h / 24), np.cos(2 * np.pi * k * h / 24)
            slot = idx.hour                                           # the same hour on the last 20 days
        else:
            slot = idx.dayofweek                                      # the same weekday in the last 20 weeks
        cols["er_same_slot_mean20"] = per["er"].groupby(slot).transform(lambda x: x.shift(1).rolling(20, min_periods=5).mean())
        cols["weekend"] = (idx.dayofweek >= 5).astype(float)
        cols["monday"] = (idx.dayofweek == 0).astype(float)
    if "calendar" in groups:
        if "calendar/us_releases/1d" in have:
            cal = _read("calendar/us_releases/1d")
            ecol = next((c for c in cal.columns if "event" in c.lower()), cal.columns[0])
            labels = cal[ecol].astype(str).str.lower()
            covered = (idx >= cal.index.min().normalize()) & (idx <= cal.index.max().normalize() + pd.Timedelta(days=1))
            for key, words, hhmm in [("fomc", ["fomc"], "14:00"), ("cpi", ["cpi"], "08:30"),
                                     ("payrolls", ["nfp", "payroll", "employment"], "08:30"), ("eia", ["eia"], "10:30")]:
                days = cal.index[labels.str.contains("|".join(words))]
                if horizon == "daily":
                    hit = idx.normalize().isin(days.normalize())
                else:
                    when = pd.DatetimeIndex([pd.Timestamp(f"{d.strftime('%Y-%m-%d')} {hhmm}", tz="America/New_York").tz_convert("UTC")
                                             for d in days])
                    hit = idx.isin(when.floor("1h"))
                cols[f"release_{key}"] = np.where(covered, hit.astype(float), np.nan)   # outside the calendar: unknown, not "no release"
        else:
            notes.append("calendar: no calendar/us_releases/1d in the catalog")
    sym = _symbol(ds)
    if "funding" in groups:
        fid = f"binance-funding/{sym}/8h"
        if fid in have:
            f = _read(fid)["funding_rate"].astype(float)
            cols["funding"] = _asof(idx, f, name="funding")
            cols["abs_funding"] = cols["funding"].abs()
            cols["funding_change"] = cols["funding"] - _asof(idx, f.shift(1), name="fprev")
        else:
            notes.append(f"funding: no {fid}")
    if "basis" in groups:
        pid, sid = f"binance-futures-um/{sym}/1h", f"binance/{sym}/1h"
        if pid in have and sid in have:
            perp, spot = catalog.read(pid)["close"].astype(float), catalog.read(sid)["close"].astype(float)
            basis = np.log(perp / spot.reindex(perp.index)).dropna()
            close_time = basis.copy(); close_time.index = close_time.index + pd.Timedelta(hours=1)   # known at the bar's close
            cols["basis"] = _asof(idx + pd.Timedelta(seconds=1), close_time, name="basis").to_numpy()
            cols["basis_change_24h"] = cols["basis"] - _asof(idx + pd.Timedelta(seconds=1) - pd.Timedelta(hours=24), close_time, name="b24").to_numpy()
        else:
            notes.append(f"basis: needs {pid} and {sid}")
    if "macro" in groups:
        for sid, name in [("fred/VIXCLS/1d", "vix"), ("fred/T10Y2Y/1d", "slope_10y2y")]:
            if sid in have:
                s = _read(sid)
                s = s[s.columns[0]].astype(float)
                cols[name] = _asof(idx, s, lag=pd.Timedelta(days=1), name=name).to_numpy()   # previous US day's close
            else:
                notes.append(f"macro: no {sid}")
    if "positioning" in groups:
        if "cftc/financial/1w" in have and sym.startswith("BTC"):
            try:
                cols["levfund_net"] = _cftc_btc(idx)
            except (KeyError, StopIteration, ValueError) as e:
                notes.append(f"positioning: {e}")
        else:
            notes.append("positioning: needs cftc/financial/1w and a bitcoin market")
    if "orderbook" in groups:
        oid = "pi/book_features/1h"
        if oid in have and horizon == "hourly":
            ob = _read(oid)
            ob = ob[ob.index < pd.Timestamp(SEALED[oid], tz="UTC")]          # the confirmation sample stays sealed
            for c in ob.select_dtypes("number").columns:
                s = ob[c].astype(float).copy(); s.index = s.index + pd.Timedelta(hours=1)   # an hour's features are known at its end
                cols[f"ob_{c}"] = _asof(idx + pd.Timedelta(seconds=1), s, name=c).to_numpy()
        else:
            notes.append("orderbook: hourly target and pi/book_features/1h needed")
    X = pd.DataFrame(cols, index=idx).replace([np.inf, -np.inf], np.nan)
    return X, notes


def _cftc_btc(idx: pd.DatetimeIndex) -> np.ndarray:
    d = _read("cftc/financial/1w")
    name = next(c for c in d.columns if "market" in c.lower() and "name" in c.lower())
    names = d[name].astype(str).str.upper()
    btc = names[names.str.contains("BITCOIN") & ~names.str.contains("MICRO|NANO|ETHER")]
    if btc.empty:
        raise ValueError("no bitcoin rows in cftc/financial/1w")
    rows = d[names == btc.value_counts().index[0]]                     # the main CME contract: the longest history

    def col(side):
        bad = ("pct", "chang", "trader", "conc")
        return next(c for c in rows.columns if all(w in c.lower() for w in ("lev", side, "all")) and not any(b in c.lower() for b in bad))
    lo, sh = col("long"), col("short")
    oi = next(c for c in rows.columns if c.lower().startswith("open_interest_all"))
    val = (pd.to_numeric(rows[lo], errors="coerce") - pd.to_numeric(rows[sh], errors="coerce")) / pd.to_numeric(rows[oi], errors="coerce")
    rel = pd.DatetimeIndex(pd.to_datetime(rows["released"], utc=True))
    s = pd.Series(val.to_numpy(), index=rel).groupby(level=0).last()
    return _asof(idx, s, lag=pd.Timedelta(days=1), name="cftc").to_numpy()        # published the day before, at the latest


# ---------------------------------------------------------------------------------------------- models
def fit_predict(Xtr, ytr, Xte, model="ridge", lam=10.0):
    """Standardize on the training rows only, fit, predict. Missing values are filled with training medians."""
    with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)        # a column with no values yet: filled with 0
        med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xtr = np.where(np.isfinite(Xtr), Xtr, med); Xte = np.where(np.isfinite(Xte), Xte, med)
    if model == "trees":
        from sklearn.ensemble import HistGradientBoostingRegressor
        m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=50, random_state=0)
        m.fit(Xtr, ytr)
        return m.predict(Xte), None
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    Z, Zt = (Xtr - mu) / sd, (Xte - mu) / sd
    b = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (ytr - ytr.mean()))
    return Zt @ b + ytr.mean(), b


def trees_available() -> bool:
    try:
        import sklearn  # noqa: F401
        return True
    except ImportError:
        return False


def folds(index: pd.DatetimeIndex, horizon: str):
    """Expanding-window walk-forward blocks: (train_end, test_end) pairs."""
    start = index[0] + (pd.Timedelta(days=730) if horizon == "daily" else pd.Timedelta(days=180))
    step = pd.DateOffset(years=1) if horizon == "daily" else pd.DateOffset(days=30)
    t = start
    while t < index[-1]:
        yield t, min(t + step, index[-1] + pd.Timedelta(seconds=1))
        t = t + step


def forecast_rank(per: pd.DataFrame, X: pd.DataFrame, horizon: str, lam: float = 10.0) -> pd.Series:
    """Walk-forward forecast of each period's efficiency, as a rank (0-1) among the forecasts the model made for its
    own training periods. A period's forecast uses a model fitted only on periods that ended before it started.
    Refit yearly (daily periods) or monthly (hourly periods) on an expanding window; the first year (daily) or
    three months (hourly) get no forecast. Used by the efficiency gates inside strategies."""
    y = per["er"]
    ok = X.notna().mean() > 0.5
    X = X.loc[:, ok]
    out = pd.Series(np.nan, index=per.index)
    if X.shape[1] == 0 or len(y) < 50:
        return out
    first = pd.Timedelta(days=365) if horizon == "daily" else pd.Timedelta(days=90)
    step = pd.DateOffset(years=1) if horizon == "daily" else pd.DateOffset(days=30)
    t = y.index[0] + first
    Xv, yv = X.to_numpy(float), y.to_numpy(float)
    while t <= y.index[-1]:
        tr, te = y.index < t, (y.index >= t) & (y.index < t + step)
        if tr.sum() >= 100 and te.any():
            fitted, _ = fit_predict(Xv[tr], yv[tr], Xv[tr], lam=lam)
            pred, _ = fit_predict(Xv[tr], yv[tr], Xv[te], lam=lam)
            srt = np.sort(fitted)
            out[te] = np.searchsorted(srt, pred, side="right") / len(srt)
        t = t + step
    return out


# ---------------------------------------------------------------------------------------------- statistics
def newey_west_t(x, lags=5):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return float("nan")
    e = x - x.mean()
    s = (e @ e) / n
    for L in range(1, lags + 1):
        s += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return float(x.mean() / np.sqrt(max(s, 1e-18) / n))


def evaluate(y, pred, train_means):
    """OOS R^2 against the training-mean forecast, correlation, and the realized ER by predicted tier."""
    y, pred, base = map(lambda a: np.asarray(a, float), (y, pred, train_means))
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - base) ** 2)
    corr = float(np.corrcoef(y, pred)[0, 1]) if len(y) > 2 and np.std(pred) > 0 else 0.0
    q1, q2 = np.quantile(pred, [1 / 3, 2 / 3])
    tiers = [y[pred <= q1], y[(pred > q1) & (pred <= q2)], y[pred > q2]]
    hi, lo = (pred > q2).astype(float), (pred <= q1).astype(float)
    spread_series = y * hi / max(hi.mean(), 1e-9) - y * lo / max(lo.mean(), 1e-9)   # per-period contributions
    return dict(r2=float(r2), corr=corr, tiers=[float(t.mean()) for t in tiers], spread=float(tiers[2].mean() - tiers[0].mean()),
                t_spread=newey_west_t(spread_series), n=int(len(y)))


# ---------------------------------------------------------------------------------------------- the study
def study(ds, groups, model="ridge", emit=None, stop=None, log_to_ledger=True, per=None, horizon=None, label=None):
    """Walk-forward on development data. Streams progress through `emit` (dashboard events). Returns the summary."""
    from .projects._util import bars as bar_chart, line, scatter, stat, table
    emit = emit or (lambda e: None)
    if per is None:
        per, horizon = periods_from_bars(dev_bars(ds))
    X, notes = features(ds, per, horizon, groups)
    y = per["er"]
    cover = X.notna().mean()
    ok = cover > 0.5                                                  # drop features missing most of the time
    dropped = [c for c in X.columns if not ok[c]]
    if dropped:
        notes.append("dropped (known for under half of the development periods): "
                     + ", ".join(f"{c} {cover[c]:.0%}" for c in dropped[:8]) + (" ..." if len(dropped) > 8 else ""))
    X = X.loc[:, ok]
    if X.shape[1] == 0:
        raise ValueError("no usable features: " + "; ".join(notes + [f"dropped {dropped}"]))
    emit(table("groups", "Feature groups", ["group", "status"],
               [[g, next((n.split(": ", 1)[1] for n in notes if n.startswith(g + ":")), "used")] for g in groups]
               + [["dropped", n.split(": ", 1)[1]] for n in notes if n.startswith("dropped")]))
    emit(stat("periods", f"{horizon.title()} periods of development data", len(y), ",.0f"))
    emit(stat("features", "Features used", X.shape[1], ",.0f"))
    preds, ys, bases, own, fold_rows, idx_out = [], [], [], [], [], []
    for k, (tr_end, te_end) in enumerate(folds(y.index, horizon)):
        if stop is not None and stop.is_set():
            return None
        tr, te = y.index < tr_end, (y.index >= tr_end) & (y.index < te_end)
        if te.sum() < 5 or tr.sum() < 100:
            continue
        p, b = fit_predict(X.values[tr], y.values[tr], X.values[te], model)
        lag_cols = [c for c in X.columns if c.startswith("er_")]
        p_own = fit_predict(X[lag_cols].values[tr], y.values[tr], X[lag_cols].values[te])[0] if lag_cols else np.full(te.sum(), y.values[tr].mean())
        preds.append(p); ys.append(y.values[te]); bases.append(np.full(te.sum(), y.values[tr].mean())); own.append(p_own)
        idx_out.append(y.index[te])
        ev = evaluate(y.values[te], p, bases[-1])
        fold_rows.append([str(tr_end)[:10], int(te.sum()), round(ev["r2"], 4), round(ev["corr"], 3), round(ev["spread"], 3)])
        allp, ally, allb = np.concatenate(preds), np.concatenate(ys), np.concatenate(bases)
        tot = evaluate(ally, allp, allb)
        emit(table("folds", "Walk-forward blocks (each block predicted by a model fitted only on earlier periods)",
                   ["test block starts", "periods", "OOS R²", "correlation", "high minus low tier ER"], fold_rows))
        emit(line("corr", f"Correlation of forecast with realized efficiency, block by block (an opening-range breakout needed about {NEEDED_CORR})",
                  [{"name": "this block", "x": list(range(1, len(fold_rows) + 1)), "y": [r[3] for r in fold_rows], "points": True},
                   {"name": "needed", "x": [1, max(2, len(fold_rows))], "y": [NEEDED_CORR, NEEDED_CORR], "dash": True}],
                  x="num", xlabel="walk-forward block", hlines=[{"y": 0, "label": ""}]))
        emit(stat("r2", "Out-of-sample R² so far", tot["r2"], ".4f"))
        emit(stat("corr", "Correlation, forecast vs realized", tot["corr"], ".3f"))
        emit(stat("spread", "Realized ER: predicted-high minus predicted-low thirds", tot["spread"], "+.3f"))
        emit(stat("t", "t-statistic of that spread (Newey-West)", tot["t_spread"], ".1f"))
        emit({"progress": min(1.0, (k + 1) / max(1, len(list(folds(y.index, horizon)))))})
    if not preds:
        raise ValueError("not enough development data for a walk-forward test")
    allp, ally, allb, allo = np.concatenate(preds), np.concatenate(ys), np.concatenate(bases), np.concatenate(own)
    tot, own_ev = evaluate(ally, allp, allb), evaluate(ally, allo, allb)
    emit(bar_chart("tiers", "Realized efficiency by predicted third (development data, out of sample)",
                   ["predicted low", "predicted middle", "predicted high"], [{"name": "realized mean ER", "y": tot["tiers"]}],
                   y_format=".3f"))
    dec = pd.qcut(pd.Series(allp).rank(method="first"), 10, labels=False)
    cal = pd.DataFrame({"p": allp, "y": ally, "d": dec}).groupby("d").mean()
    emit(scatter("calibration", "Calibration: forecast decile vs realized efficiency", [
        {"name": "decile", "x": cal["p"].tolist(), "y": cal["y"].tolist()},
        {"name": "perfect", "x": [float(cal["p"].min()), float(cal["p"].max())], "y": [float(cal["p"].min()), float(cal["p"].max())], "line": True}],
        xlabel="mean forecast", ylabel="mean realized ER"))
    _, b_full = fit_predict(X.values, y.values, X.values[:1])                  # weights of a model fitted on all dev data
    if b_full is not None:
        order = np.argsort(-np.abs(b_full))[:15]
        emit(bar_chart("weights", "Feature weights (standardized; fitted on all development data, for reading only)",
                       [X.columns[i] for i in order], [{"name": "weight", "y": [float(b_full[i]) for i in order]}], y_format=".3f",
                       horizontal=True))
    emit(table("summary", "Summary", ["model", "OOS R²", "correlation", "high - low ER", "t"],
               [[f"{model}, groups: {', '.join(groups)}", round(tot["r2"], 4), round(tot["corr"], 3), round(tot["spread"], 3), round(tot["t_spread"], 1)],
                ["own past efficiency only", round(own_ev["r2"], 4), round(own_ev["corr"], 3), round(own_ev["spread"], 3), round(own_ev["t_spread"], 1)]]))
    summary = dict(dataset=ds, horizon=horizon, groups=groups, model=model, features=list(X.columns), dropped=dropped, notes=notes,
                   **{k: v for k, v in tot.items() if k != "tiers"}, tiers=tot["tiers"], own_r2=own_ev["r2"], own_corr=own_ev["corr"],
                   dev_end=str(per.index[-1])[:10])
    if log_to_ledger:
        _log(summary, label)
    return summary


def _log(s, label):
    from . import board, ledger
    ledger.append(board.lab_dir(), card="efficiency-lab", family="efficiency-lab", kind="explore", verdict="-",
                  p="", sharpe="", n_trials=1, finalist=f"{s['model']} on {'+'.join(s['groups'])}", data=s["dataset"],
                  note=label or f"efficiency study ({s['horizon']}): OOS R2 {s['r2']:.4f}, corr {s['corr']:.3f}, spread {s['spread']:+.3f} (t {s['t_spread']:.1f})")


def studies_so_far(ds: str) -> int:
    from . import board, ledger
    return sum(int(r["n_trials"] or 1) for r in ledger.rows(board.lab_dir()) if r["card"] == "efficiency-lab" and r["data"] == ds)


def history() -> list[dict]:
    """Every efficiency study and gate check in the ledger, newest first."""
    from . import board, ledger
    return [r for r in ledger.rows(board.lab_dir()) if r["card"] == "efficiency-lab"][::-1]


# ---------------------------------------------------------------------------------------------- synthetic checks
def synthetic_bars(predictable: bool, n_days: int = 1500, seed: int = 0, strength: float = 0.7) -> pd.DataFrame:
    """Hourly bars of a synthetic market. Predictable: trend days and choppy days come in regimes that persist (each
    day keeps yesterday's regime with probability 0.8), and `strength` sets how much a regime changes the day, so
    yesterday's efficiency says something, but not much, about today's (correlation about 0.3, the level an
    opening-range breakout needed). A trend day drifts one way all day (direction random per day); on a choppy day
    each hour partly reverses the last. Noise: every day is the same random walk."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n_days * 24, freq="1h", tz="UTC")
    regime, reg = np.empty(n_days), 1.0
    for d in range(n_days):
        if rng.random() > 0.8:
            reg = -reg
        regime[d] = reg
    r = rng.standard_normal(n_days * 24) * 0.005
    if predictable:
        for d in range(n_days):
            sl = slice(d * 24, (d + 1) * 24)
            if regime[d] > 0:
                r[sl] = r[sl] + strength * 0.005 * 0.35 * rng.choice([-1, 1])
            else:
                r[sl] = r[sl] - np.r_[0, r[sl][:-1]] * strength
    c = 100 * np.exp(np.cumsum(r))
    o = np.r_[100, c[:-1]]
    wig = np.abs(rng.standard_normal(len(c))) * 0.001
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * (1 + wig), "low": np.minimum(o, c) * (1 - wig), "close": c,
                         "volume": 1.0 + rng.random(len(c))}, index=idx)


def synthetic_periods(predictable: bool, n_days: int = 1500, seed: int = 0, strength: float = 0.7) -> pd.DataFrame:
    return path_er(synthetic_bars(predictable, n_days, seed, strength), "1D", 20)


def self_checks(emit, stop=None) -> bool:
    from .projects._util import check
    ok = True
    for label, predictable in [("Finds planted, persistent efficiency regimes", True), ("Finds nothing in a plain random walk", False)]:
        per = synthetic_periods(predictable)
        s = study("synth/known-answer/1h", ["price"], emit=lambda e: None, stop=stop, log_to_ledger=False, per=per, horizon="daily")
        good = (s["corr"] > 0.2 and s["t_spread"] > 3) if predictable else (abs(s["corr"]) < 0.1 and abs(s["t_spread"]) < 2.5)
        ok &= good
        emit(check(label, good, f"correlation {s['corr']:.3f}, high-low spread {s['spread']:+.3f} (t {s['t_spread']:.1f})"))
    return ok


# ---------------------------------------------------------------------------------------------- dashboard entry point
def run(p, emit, stop):
    """Job for the dashboard's Efficiency tab: checks first, then the study on development data."""
    from .projects._util import stat
    ds, model = p["dataset"], p.get("model", "ridge")
    groups = ["price"] + [g for g in GROUPS if g != "price" and p.get(g) in (True, "yes", "on", 1)]
    if model == "trees" and not trees_available():
        raise ValueError("boosted trees need scikit-learn: pip install scikit-learn")
    if not self_checks(emit, stop):
        emit({"log": "a self-check failed: the study below should not be trusted until it is fixed"})
    k = studies_so_far(ds) + 1
    emit(stat("study_no", f"Efficiency explorations on {ds} so far, this one included", k, ",.0f"))
    emit({"log": f"walk-forward on development data before {str(dev_cut(ds))[:10]}; nothing from the holdout is used"})
    t0 = time.time()
    s = study(ds, groups, model, emit, stop)
    if s:
        emit({"log": f"done in {time.time() - t0:.0f} s; logged in the ledger (exploration #{k} on {ds})"})


# ---------------------------------------------------------------------------------------------- what each group needs
def availability(ds: str) -> dict:
    """{group: [usable, note]} for a target dataset, from what the catalog holds."""
    have = {m.id for m in catalog.list_all()}
    sym = _symbol(ds)
    freq = ds.rsplit("/", 1)[-1]
    need = {
        "price": ([], "always"),
        "calendar": (["calendar/us_releases/1d"], "oql data import-calendar ~/Desktop/ahl-node/edge-engine/futures/events_full.csv"),
        "funding": ([f"binance-funding/{sym}/8h"], "collector job binance-funding"),
        "basis": ([f"binance-futures-um/{sym}/1h", f"binance/{sym}/1h"], "collector jobs binance-perp-1h and binance-spot-1h"),
        "macro": (["fred/VIXCLS/1d", "fred/T10Y2Y/1d"], "collector job fred-daily"),
        "positioning": (["cftc/financial/1w"], "collector job cftc-financial (bitcoin markets only)"),
        "orderbook": (["pi/book_features/1h"], "the Pi recorder (hourly targets from minute bars only)"),
    }
    out = {}
    for g, (ids, how) in need.items():
        missing = [i for i in ids if i not in have]
        ok = not missing
        if g == "positioning" and not sym.startswith("BTC"):
            ok, how = False, "CFTC positioning is wired for bitcoin only"
        if g == "orderbook" and freq != "1m":
            ok, how = False, "order-book features are hourly: pick a minute-bar dataset"
        out[g] = [ok, "" if ok else (f"missing {', '.join(missing)}: {how}" if missing else how)]
    return out


# ---------------------------------------------------------------------------------------------- do gates help a strategy?
def strategy_choices() -> list[str]:
    """Every library strategy as 'rule a=1 b=2' (both sides), for the dashboard."""
    import itertools
    from .strategies import ALL_RULES, valid
    out = []
    for r, (_, grid) in ALL_RULES.items():
        for combo in itertools.product(*grid.values()):
            prm = dict(zip(grid, combo))
            if valid(r, prm):
                out.append(r + " " + " ".join(f"{k}={v}" for k, v in prm.items()))
    return out


def parse_strategy(text: str) -> dict:
    from .strategies import ALL_RULES, valid
    parts = text.split()
    rule, prm = parts[0], {k: int(v) for k, v in (x.split("=") for x in parts[1:])}
    if rule not in ALL_RULES or set(prm) != set(ALL_RULES[rule][1]) or not valid(rule, prm):
        raise ValueError(f"not a library strategy: {text!r}")
    return {"rule": rule, "params": prm, "side": "both"}


def all_gate_settings(grid: dict | None = None) -> list[dict]:
    import itertools
    from .gates import GATES
    out = [{"kind": "none"}]
    for kind, g in (grid or GATES).items():
        for combo in itertools.product(*g.values()):
            out.append({"kind": kind, **dict(zip(g, combo))})
    return out


def _gate_label(g: dict) -> str:
    if g["kind"] == "none":
        return "no gate"
    tail = f"{'top' if g['side'] == 'high' else 'bottom'} {g['share']}%"
    return f"trailing {g['lookback']} bars, {tail}" if g["kind"] == "er_trailing" else f"forecast, {tail}"


def gate_check(p, emit, stop):
    """Dashboard job: on development data, how does each efficiency gate change one strategy, and how strong would a
    forecast have to be to make the gate worth it? Exploratory and logged: only a card's gate test decides."""
    from .engine import market_returns, prepare
    from .gates import mask
    from .projects._util import bars as bar_chart, check, line, stat, table
    from .search import Scorer, sharpe
    from .strategies import ALL_RULES
    ds, g0 = p["dataset"], parse_strategy(p["strategy"])
    cost = float(p.get("cost_bps", 2))
    band = float(p.get("band", 0.0))
    rules = {g0["rule"]: ALL_RULES[g0["rule"]]}

    def scorer(b):
        return Scorer(prepare(b, {"default": cost}, warmup=min(250, len(next(iter(b.values()))) // 10), band=band), rules, 0.0)

    # known-answer check: on a market with planted, persistent efficiency regimes, an opening-range breakout traded
    # only on forecast-high days must clearly beat the same rule on forecast-low days; on a random walk it must not
    for planted in (True, False):
        sb = synthetic_bars(planted, n_days=900, seed=11 if planted else 12)
        sc = Scorer(prepare({"synth": sb}, {"default": 1.0}, warmup=250), {"orb": ALL_RULES["orb"]}, 0.0)
        base = {"rule": "orb", "params": {"hours": 2}, "side": "both"}
        hi = sc.score({**base, "gate": {"kind": "er_forecast", "side": "high", "share": 33}}, delay=1 / 23)
        lo = sc.score({**base, "gate": {"kind": "er_forecast", "side": "low", "share": 33}}, delay=1 / 23)
        ok = (hi - lo > 1.0) if planted else (abs(hi - lo) < 1.0)
        emit(check("The forecast gate separates trend days from choppy days where that is planted" if planted
                   else "The forecast gate finds nothing in a random walk", ok,
                   f"opening-range breakout on forecast-high days {hi:+.2f}, on forecast-low days {lo:+.2f}"))
    bars_ = dev_bars(ds)
    emit({"log": f"{ds} development data {str(bars_.index[0])[:10]} to {str(bars_.index[-1])[:10]}, {cost:g} bp per trade"
                 + (f", resizing in steps of {band:.0%}" if band else ", resizing every bar")})
    sc = scorer({ds: bars_})
    settings = all_gate_settings()
    rows, labels, vals = [], [], []
    for k, gate in enumerate(settings):
        if stop.is_set():
            return
        g = {**g0, **({} if gate["kind"] == "none" else {"gate": gate})}
        port, _, trades = sc.returns(g, delay=1 / 23)
        s = sharpe(port.to_numpy(), sc.prep.ppy)
        on = 1.0 if gate["kind"] == "none" else float(np.nanmean(mask(gate, bars_, ds, sc._gate_cache)))
        rows.append([_gate_label(gate), round(s, 3), f"{on:.0%}", trades])
        labels.append(_gate_label(gate)); vals.append(s)
        order = np.argsort(vals)[::-1]
        emit(bar_chart("by_gate", f"{p['strategy']}: development Sharpe with each gate setting, best first",
                       [labels[i] for i in order], [{"name": "Sharpe", "y": [vals[i] for i in order]}],
                       y_format=".2f", horizontal=True, ref={"value": vals[0], "label": "no gate"}, wide=True))
        emit({"progress": (k + 1) / (len(settings) + 8)})
    emit(table("gates", "Every gate setting tried (each one is counted in the ledger)", ["gate", "Sharpe", "time on", "direction changes"],
               sorted(rows, key=lambda r: -r[1])))
    best = int(np.argmax(vals[1:])) + 1
    curves = []
    for i in (0, best):
        g = {**g0, **({} if i == 0 else {"gate": settings[i]})}
        port = sc.returns(g, delay=1 / 23)[0].fillna(0.0)
        cum = port.cumsum()
        step = max(1, len(cum) // 600)
        curves.append({"name": labels[i], "x": [int(t.value // 1_000_000) for t in cum.index[::step]], "y": (cum.iloc[::step] * 100).tolist()})
    emit(line("equity", "Cumulative return, % (development data)", curves, x="time"))
    emit(stat("sharpe0", "Sharpe without a gate", vals[0], "+.2f"))
    emit(stat("sharpe_best", f"Best gate: {labels[best]}", vals[best], "+.2f"))
    emit(stat("tried", "Gate settings tried", len(settings), ",.0f"))
    # forecast tiers: realized efficiency and the strategy's P&L per period, by the forecast made before the period
    per, horizon = periods_from_bars(bars_)
    X, _ = features(ds, per, horizon, ["price"])
    rank = forecast_rank(per, X, horizon)
    period = "1h" if horizon == "hourly" else "1D"
    pnl = sc.returns(g0, delay=1 / 23)[0].fillna(0.0)
    per_pnl = pnl.groupby(pnl.index.floor(period)).sum().reindex(per.index)
    d = pd.DataFrame({"rank": rank, "er": per["er"], "pnl": per_pnl}).dropna()
    if len(d) > 60:
        tier = pd.cut(d["rank"], [-0.01, 1 / 3, 2 / 3, 1.01], labels=["predicted low", "predicted middle", "predicted high"])
        tab = d.groupby(tier, observed=False).agg(er=("er", "mean"), pnl=("pnl", "mean"), n=("er", "size"))
        emit(bar_chart("tiers", f"{p['strategy']}: P&L by forecast third ({len(d)} {'days' if horizon == 'daily' else 'hours'} with a forecast)",
                       list(tab.index.astype(str)), [{"name": "mean P&L per period, bps", "y": (tab["pnl"] * 1e4).round(2).tolist()}], y_format=".1f"))
        corr = float(np.corrcoef(d["rank"], d["er"])[0, 1])
        emit(stat("fc_corr", "Our forecast's correlation with realized efficiency", corr, ".3f"))
        # how strong would a forecast need to be? blend the realized efficiency with noise, gate on the top third
        z = (per["er"] - per["er"].mean()) / per["er"].std()
        nxt = bars_.index[1:].append(pd.DatetimeIndex([bars_.index[-1] + bar_step(bars_.index)])).floor(period)
        sig = sc.signals(g0)[ds]
        rhos, curve = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0], []
        for rho in rhos:
            if stop.is_set():
                return
            ss = []
            for seed in range(4):
                rng = np.random.default_rng(seed)
                f = rho * z + np.sqrt(1 - rho ** 2) * pd.Series(rng.standard_normal(len(z)), index=z.index)
                on = (f >= f.quantile(2 / 3)).astype(float).reindex(nxt).fillna(0.0).to_numpy()
                r, _ = market_returns(sc.prep.arrays[ds], sig * on, delay=1 / 23)
                ss.append(sharpe(r, sc.prep.ppy))
            curve.append(float(np.mean(ss)))
            emit(line("needed", "If a forecast had this correlation with realized efficiency, trading only its top third would give...",
                      [{"name": "Sharpe, top-third gate (average of 4 simulated forecasts)", "x": rhos[:len(curve)], "y": curve, "points": True},
                       {"name": "no gate", "x": [0, 1], "y": [vals[0], vals[0]], "dash": True}],
                      x="num", x_range=[0, 1], xlabel="forecast correlation with realized efficiency", ylabel="Sharpe",
                      vlines=[{"x": max(corr, 0.0), "label": "ours"}]))
            emit({"progress": (len(settings) + len(curve)) / (len(settings) + 8)})
    _log_gate_check(ds, p["strategy"], len(settings), vals[0], vals[best], labels[best], cost, band)
    emit({"log": f"logged in the ledger as {len(settings)} explorations on {ds}. To test a gate for real, put it in a card's "
                 "experiment (gates=[...]) and run the gate test: it reruns the gates on shuffled data too."})


def _log_gate_check(ds, strategy, n, s0, s1, label, cost, band=0.0):
    band_note = f", resize band {band:.0%}" if band else ""
    from . import board, ledger
    ledger.append(board.lab_dir(), card="efficiency-lab", family="efficiency-lab", kind="explore", verdict="-", p="",
                  sharpe=round(s1, 3), n_trials=n, finalist=f"{strategy} | {label}", data=ds,
                  note=f"gate check: {n} gate settings on {strategy} at {cost:g} bp{band_note}; no gate {s0:+.2f}, best {s1:+.2f}")
