"""Project 10: information diffusion. How hard does scheduled news hit the market, and how fast is it absorbed?

Two streams of timestamps: NEWS (scheduled US releases: FOMC at 2:00 pm and CPI at 8:30 am New York) and PRICE EVENTS
(minutes whose move is over 4x the median size of the 12 hours BEFORE the news, so the threshold never uses
post-news data). The rate of price events is a Hawkes process excited by both: a jump when news lands, decaying
with a half-life, plus the market reacting to itself ("echo"). Checks: simulated windows with known parameters are
recovered; moving the news 6 hours early (placebo) must make the effect vanish. Limits: both releases land at the
start of the busiest US hours, so the raised activity AFTER the spike may be time-of-day seasonality (project 2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import catalog
from ._util import bars, check, line, nelder_mead, stat, table

TITLE = "News diffusion: how fast does news get priced?"
QUESTION = "How much does a Fed or CPI release jolt the market, and how quickly does the jolt fade?"
PARAMS = [{"name": "dataset", "label": "Minute bars", "type": "dataset", "kind": "bars", "freq": "1m", "default": "binance/BTCUSDT/1m"},
          {"name": "events", "label": "Releases", "type": "choice", "options": ["FOMC + CPI (built-in list)", "FOMC + CPI from the release calendar"],
           "default": "FOMC + CPI (built-in list)"}]
FOMC = ["2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
        "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
        "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16"]
CPI = ["2025-12-18", "2026-01-13", "2026-02-13", "2026-03-11", "2026-04-10", "2026-05-12", "2026-06-10", "2026-07-14",
       "2026-08-12", "2026-09-11"]
HALF = 12 * 3600.0


def _ts(day, clock):
    return pd.Timestamp(f"{day} {clock}", tz="America/New_York").tz_convert("UTC")


def events(source):
    if source.startswith("FOMC + CPI from"):
        cal = catalog.read("calendar/us_releases/1d")
        col = next(c for c in cal.columns if "event" in c.lower())
        out = []
        for d, e in zip(cal.index, cal[col].astype(str)):
            if "fomc" in e.lower():
                out.append((_ts(d.strftime("%Y-%m-%d"), "14:00"), "FOMC"))
            elif "cpi" in e.lower():
                out.append((_ts(d.strftime("%Y-%m-%d"), "08:30"), "CPI"))
        return sorted(out)
    return sorted([(_ts(d, "14:00"), "FOMC") for d in FOMC] + [(_ts(d, "08:30"), "CPI") for d in CPI])


def window_ll(params, price, news, horizon, use_news=True):
    mu, aN, bN, aP, bP = params
    lam = np.full(len(price), mu)
    comp = mu * horizon
    if use_news:
        dt = price - news
        lam += np.where(dt > 0, aN * np.exp(-bN * np.clip(dt, 0, None)), 0.0)
        comp += aN / bN * (1 - np.exp(-bN * (horizon - news)))
    A, decay = np.zeros(len(price)), np.exp(-bP * np.diff(price))
    for i in range(1, len(price)):
        A[i] = decay[i - 1] * (1 + A[i - 1])
    lam += aP * A
    comp += aP / bP * np.sum(1 - np.exp(-bP * (horizon - price)))
    return np.sum(np.log(lam)) - comp


def fit(windows, use_news=True):
    rate = sum(len(w[0]) for w in windows) / (len(windows) * 2 * HALF)

    def nll(x):
        p = np.exp(x)
        if p[3] >= p[4]:
            return 1e12
        return -sum(window_ll(p, pr, nw, 2 * HALF, use_news) for pr, nw in windows)
    best = None
    for hl in (300.0, 1800.0):
        x, f = nelder_mead(nll, np.log([rate / 2, rate * 20, np.log(2) / hl, 0.005, 0.01]), iters=1500)
        if best is None or f < best[1]:
            best = (x, f)
    mu, aN, bN, aP, bP = np.exp(best[0])
    n = sum(len(w[0]) for w in windows)
    bg, news = mu * 2 * HALF * len(windows), (aN / bN * len(windows) if use_news else 0.0)
    return dict(mu=mu, bN=bN, aN=aN, ll=-best[1], half_life_min=np.log(2) / bN / 60, jump=aN / mu, echo=aP / bP,
                share_bg=bg / n, share_news=news / n, share_echo=1 - (bg + news) / n)


def simulate_window(mu, aN, bN, aP, bP, rng):
    news, t, out = HALF, 0.0, []
    while True:
        exc = sum(aP * np.exp(-bP * (t - s)) for s in out[-50:])
        lam_bar = mu + exc + aN
        t += rng.exponential(1 / lam_bar)
        if t > 2 * HALF:
            return np.array(out), news
        lam = mu + (aN * np.exp(-bN * (t - news)) if t > news else 0.0) + sum(aP * np.exp(-bP * (t - s)) for s in out[-50:])
        if rng.random() * lam_bar <= lam:
            out.append(t)


def price_events(closes, news_ts, rng, mult=4.0):
    w = closes.loc[news_ts - pd.Timedelta(seconds=HALF):news_ts + pd.Timedelta(seconds=HALF)]
    if len(w) < 1200:
        return None
    r = np.log(w).diff().dropna()
    base = r.loc[:news_ts - pd.Timedelta(minutes=1)].abs()
    thr = mult * base[base > 0].median()
    ev = r[r.abs() > thr]
    secs = (ev.index - (news_ts - pd.Timedelta(seconds=HALF))).total_seconds().to_numpy() - rng.uniform(0, 60, len(ev))
    offs = ((r.index - news_ts).total_seconds() // 60).astype(int)
    return np.sort(np.clip(secs, 1e-3, 2 * HALF - 1e-3)), pd.Series(r.abs().to_numpy() / base.mean(), index=offs)


def row(label, f):
    none = f["half_life_min"] < 0.5 or f["share_news"] < 0.005        # an effect that lasts no time is no effect
    return [label, "none" if none else f"{f['jump']:.1f}x", "–" if none else f"{f['half_life_min']:.1f}", f"{f['echo']:.2f}", f"{f['share_bg']:.0%}", f"{f['share_news']:.0%}", f"{f['share_echo']:.0%}"]


COLS = ["model", "news jump", "half-life (min)", "echo n", "background", "news", "echo"]


def run(p, emit, stop):
    rng = np.random.default_rng(5)
    true = dict(mu=0.0008, aN=0.02, bN=np.log(2) / 900, aP=0.004, bP=0.01)
    emit({"log": "simulating 30 news windows with known parameters (jump 25x, half-life 15 min, echo 0.40)"})
    sim = [simulate_window(**true, rng=rng) for _ in range(30)]
    fs = fit(sim)
    rows = [row("check: simulated (truth 25x, 15 min, 0.40)", fs)]
    emit(table("fits", "Fits (each price event is attributed to background, news or echo)", COLS, rows))
    emit(check("Recovers the planted news jump and half-life", abs(fs["jump"] / 25 - 1) < 0.35 and abs(fs["half_life_min"] / 15 - 1) < 0.35,
               f"jump {fs['jump']:.1f}x (25x), half-life {fs['half_life_min']:.1f} min (15)"))
    fp = fit([(pp, n - 6 * 3600) for pp, n in sim])
    rows.append(row("check: placebo, news moved 6 h early", fp))
    emit(table("fits", "Fits (each price event is attributed to background, news or echo)", COLS, rows))
    emit(check("Placebo: the effect vanishes when the news time is wrong", fp["share_news"] < 0.1 * max(fs["share_news"], 1e-9) + 0.02,
               f"news share {fp['share_news']:.1%} vs {fs['share_news']:.1%} at the true time"))
    closes = catalog.read(p["dataset"])["close"].astype(float)
    closes = closes[~closes.index.duplicated()]
    evs = [(t, k) for t, k in events(p["events"]) if closes.index[0] + pd.Timedelta(hours=13) < t < closes.index[-1] - pd.Timedelta(hours=13)]
    emit(stat("events", "Releases inside the data", len(evs), ",.0f"))
    wins, profile, kinds = [], [], []
    for i, (ts, kind) in enumerate(evs):
        if stop.is_set():
            return
        got = price_events(closes, ts, rng)
        if got is None:
            continue
        wins.append((got[0], HALF)); profile.append(got[1]); kinds.append(kind)
        prof = pd.concat(profile, axis=1).mean(axis=1).sort_index().loc[-120:240]
        emit(bars("profile", f"Average size of 1-minute moves around {len(wins)} releases (1 = normal)",
                  [int(x) for x in prof.index], [{"name": "average |1-minute move|", "y": prof.tolist()}],
                  vlines=[{"x": 0, "label": "release"}], xlabel="minutes from the release", dense=True, label_multiple=30))
        emit({"progress": (i + 1) / len(evs)})
    if len(wins) < 5:
        emit({"log": "fewer than 5 release windows inside this dataset: not enough to fit"})
        return
    emit({"log": f"fitting the model to {len(wins)} real windows ({kinds.count('FOMC')} FOMC, {kinds.count('CPI')} CPI)"})
    f1, f0 = fit(wins), fit(wins, use_news=False)
    rows.append(row(f"real: all {len(wins)} releases", f1))
    emit(table("fits", "Fits (each price event is attributed to background, news or echo)", COLS, rows))
    lr = 2 * (f1["ll"] - f0["ll"])
    emit(stat("jump", "News jump in activity (x background)", f1["jump"], ".1f"))
    emit(stat("half", "Half-life of the news effect (minutes)", f1["half_life_min"], ".1f"))
    emit(stat("lr", "Likelihood ratio, news vs no news (>9.2 = p < 0.01)", lr, ".1f"))
    rows.append(row("real: placebo, 6 h early", fit([(pp, n - 6 * 3600) for pp, n in wins])))
    emit(table("fits", "Fits (each price event is attributed to background, news or echo)", COLS, rows))
    for k in ("FOMC", "CPI"):
        if stop.is_set():
            return
        sub = [w for w, kk in zip(wins, kinds) if kk == k]
        if len(sub) >= 5:
            rows.append(row(f"real: {k} only ({len(sub)})", fit(sub)))
            emit(table("fits", "Fits (each price event is attributed to background, news or echo)", COLS, rows))
