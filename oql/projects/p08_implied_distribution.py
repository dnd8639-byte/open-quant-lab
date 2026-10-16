"""Project 8: the market's implied probability distribution (Breeden-Litzenberger 1978).

The risk-neutral density is the second derivative of the call price in the strike: q(K) = e^(rT) d2C/dK2. A second
derivative amplifies noise, so prices rounded to cents give nonsense (even negative probabilities). Method: convert
prices to implied volatilities, fit a smooth curve through them, turn it back into smooth prices, differentiate.
Checks: a fake chain with one volatility must give the lognormal curve; rounding to cents must break the naive method
but not the smooth one; a fake skew must show up as a fat left tail. Then real SPY options, if imported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import catalog
from ._util import N, check, line, stat

TITLE = "Implied distribution: what option prices say about the future"
QUESTION = "What probabilities do SPY option prices assign to big moves, compared with a bell curve?"
PARAMS = []
NEEDS = ["options/SPY/chain"]
HOW_TO_GET = "oql data import-projects ~/Desktop/ahl-node/quant-projects/data  (licensed: Yahoo data stays local)"
cdf = np.vectorize(N.cdf)
_trapz = getattr(np, "trapezoid", None) or np.trapz


def black_call(F, K, T, sigma, df=1.0):
    sd = sigma * np.sqrt(T)
    d1 = np.log(F / K) / sd + sd / 2
    return df * (F * cdf(d1) - K * cdf(d1 - sd))


def implied_vol(price, F, K, T, df=1.0):
    price, K = np.asarray(price, float), np.asarray(K, float)
    lo, hi = np.full(K.shape, 1e-4), np.full(K.shape, 5.0)
    bad = (price <= df * np.maximum(F - K, 0) + 1e-10) | (price >= df * F)
    for _ in range(60):
        mid = (lo + hi) / 2
        too_low = black_call(F, K, T, mid, df) < price
        lo, hi = np.where(too_low, mid, lo), np.where(too_low, hi, mid)
    return np.where(bad, np.nan, (lo + hi) / 2)


def density_raw(K, C, df=1.0):
    h1, h2 = K[1:-1] - K[:-2], K[2:] - K[1:-1]
    return K[1:-1], 2 * (h1 * C[2:] - (h1 + h2) * C[1:-1] + h2 * C[:-2]) / (h1 * h2 * (h1 + h2)) / df


def density_smooth(K, C, F, T, df=1.0, degree=4, n_grid=400, min_tv=0.05):
    iv = implied_vol(C, F, K, T, df)
    ok = np.isfinite(iv) & ((C - df * np.maximum(F - K, 0)) >= min_tv)
    x = np.log(K[ok] / F)
    sd = iv[ok] * np.sqrt(T)
    coef = np.polyfit(x, iv[ok], degree, w=np.exp(-(x / sd - sd / 2) ** 2 / 2))
    grid = np.linspace(K[ok].min(), K[ok].max(), n_grid)
    vol = np.polyval(coef, np.log(grid / F))
    q = np.gradient(np.gradient(black_call(F, grid, T, vol, df), grid), grid) / df
    return grid[2:-2], q[2:-2], vol[2:-2]


def moments(k, q, F):
    w = _trapz(q, k)
    mean = _trapz(q * k, k) / w
    sd = np.sqrt(_trapz(q * (k - mean) ** 2, k) / w)
    return dict(mass=w, skew=_trapz(q * ((k - mean) / sd) ** 3, k) / w, p15=_trapz(q[k < 0.85 * F], k[k < 0.85 * F]))


def run(p, emit, stop):
    F, T, SIG = 100.0, 0.25, 0.20
    K = np.arange(60.0, 141.0, 1.0)
    kf = np.linspace(60, 140, 400)
    logn = np.exp(-(np.log(kf / F) + SIG**2 * T / 2) ** 2 / (2 * SIG**2 * T)) / (kf * SIG * np.sqrt(2 * np.pi * T))
    C = black_call(F, K, T, SIG)
    g, q, _ = density_smooth(K, C, F, T)
    e_a = float(np.abs(q - np.interp(g, kf, logn)).max())
    emit(check("(a) Recovers the lognormal density from a flat-volatility chain", e_a < 0.002, f"max error {e_a:.5f} (peak {logn.max():.3f})"))
    kr, qr = density_raw(K, np.round(C, 2))
    g2, q2, _ = density_smooth(K, np.round(C, 2), F, T)
    e_raw, e_sm = float(np.abs(qr - np.interp(kr, kf, logn)).max()), float(np.abs(q2 - np.interp(g2, kf, logn)).max())
    emit(check("(b) Rounding to cents breaks raw differencing but not the smooth method", e_sm < 0.002 < e_raw,
               f"raw error {e_raw:.4f} with {(qr < 0).sum()} negative probabilities; smooth error {e_sm:.5f}"))
    emit(line("rounding", "Check: rounding prices to cents breaks the naive method", [
        {"name": "exact lognormal", "x": kf.tolist(), "y": logn.tolist()},
        {"name": "raw 2nd difference of rounded prices", "x": kr.tolist(), "y": qr.tolist(), "points": True},
        {"name": "smoothed in implied-vol space", "x": g2.tolist(), "y": q2.tolist(), "dash": True}],
        x="num", xlabel="price at expiry", ylabel="probability density"))
    skew_vol = 0.20 - 0.35 * np.log(K / F) + 0.4 * np.log(K / F) ** 2
    gs, qs, _ = density_smooth(K, black_call(F, K, T, skew_vol), F, T)
    m_flat, m_skew = moments(g, q, F), moments(gs, qs, F)
    # Thresholds: skew down by more than 0.5 and the 15%-fall probability up by more than half. (An earlier, arbitrary
    # "more than double" was too strict: the planted skew takes it from 3.6% to 7.2%.)
    emit(check("(c) A volatility skew shows up as a fat left tail", m_skew["skew"] < m_flat["skew"] - 0.5 and m_skew["p15"] > 1.5 * m_flat["p15"],
               f"skew {m_flat['skew']:+.2f} -> {m_skew['skew']:+.2f}; P(15% drop) {m_flat['p15']:.2%} -> {m_skew['p15']:.2%}"))
    emit(line("skew", "Check: a skew in volatility makes the left tail fat", [
        {"name": "flat volatility (bell)", "x": g.tolist(), "y": q.tolist(), "dash": True},
        {"name": "with a volatility skew", "x": gs.tolist(), "y": qs.tolist()}], x="num", xlabel="price at expiry",
        bands=[{"x0": 60, "x1": 85, "label": "15%+ fall"}]))
    try:
        ch = catalog.read("options/SPY/chain").reset_index(drop=True)
    except KeyError:
        emit({"log": f"Real SPY data not in the catalog. To add it: {HOW_TO_GET}"})
        return
    meta = ch.iloc[0]
    T_r, spot, df = float(meta["T_years"]), float(meta["spot"]), np.exp(-0.04 * float(meta["T_years"]))
    ch["mid"] = np.where((ch.bid > 0) & (ch.ask > 0), (ch.bid + ch.ask) / 2, ch.lastPrice)
    c = ch[ch.type == "call"].set_index("strike")["mid"]; pu = ch[ch.type == "put"].set_index("strike")["mid"]
    both = pd.concat([c, pu], axis=1, keys=["c", "p"]).dropna()
    both = both[(both.index > 0.75 * spot) & (both.index < 1.2 * spot) & (both.c > 0.05) & (both.p > 0.05)]
    k_atm = (both.c - both.p).abs().idxmin()
    F_r = k_atm + (both.c[k_atm] - both.p[k_atm]) / df                       # put-call parity gives the forward
    Kr = both.index.to_numpy(float)
    call = np.where(Kr < F_r, both.p + df * (F_r - Kr), both.c)               # out-of-the-money side only
    gr, qr_, vol_r = density_smooth(Kr, call, F_r, T_r, df)
    atm = float(np.interp(F_r, gr, vol_r))
    ln = np.exp(-(np.log(gr / F_r) + atm**2 * T_r / 2) ** 2 / (2 * atm**2 * T_r)) / (gr * atm * np.sqrt(2 * np.pi * T_r))
    mr, ml = moments(gr, qr_, F_r), moments(gr, ln, F_r)
    emit(stat("atm", f"SPY at-the-money volatility (expiry {meta['expiry']})", atm, ".1%"))
    emit(stat("skew_real", "Skew implied by SPY options", float(mr["skew"]), "+.2f"))
    emit(stat("p15", "Priced chance of a 15%+ fall", float(mr["p15"]), ".2%"))
    emit(stat("p15_bell", "Same chance under a bell curve", float(ml["p15"]), ".3%"))
    emit(line("real", f"SPY options, expiry {meta['expiry']} (downloaded {meta['downloaded']})", [
        {"name": "implied by SPY options", "x": gr.tolist(), "y": qr_.tolist()},
        {"name": "lognormal, same at-the-money vol", "x": gr.tolist(), "y": ln.tolist(), "dash": True}],
        x="num", xlabel="SPY price at expiry", vlines=[{"x": float(F_r), "label": "forward"}]))
