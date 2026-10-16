"""Project 7: Monte Carlo Greeks by automatic differentiation, written from scratch.

"Bump and reprice" divides simulation noise by a tiny number (small bump) or is biased (big bump). Forward-mode
automatic differentiation carries each value's derivative through every arithmetic step ("dual numbers"), so one run
gives the exact derivative of what was simulated. Checks against Black-Scholes, which gives exact Greeks for a
European call: delta and vega by autodiff vs bumping; gamma by differentiating twice, which FAILS (exactly 0: the
payoff's kink has zero second derivative everywhere a path can land), and the likelihood-ratio fix that recovers it.
"""
from __future__ import annotations

import numpy as np

from ._util import N, bars, check, hist, stat, table

TITLE = "Autodiff Greeks: derivatives without bumping"
QUESTION = "Can a Monte Carlo pricer give exact, low-noise sensitivities, and where does automatic differentiation fail?"
PARAMS = [{"name": "paths", "label": "Paths per run", "type": "choice", "options": [5000, 20000], "default": 20000},
          {"name": "repeats", "label": "Repeats", "type": "choice", "options": [100, 200, 400], "default": 200}]
S0, K, R, SIGMA, T = 100.0, 100.0, 0.05, 0.20, 1.0


class Dual:
    """A value and its derivative. Works on numbers, numpy arrays, and other Duals (for 2nd derivatives)."""
    def __init__(self, v, d=0.0):
        self.v, self.d = v, d

    def _w(self, o):
        return o if isinstance(o, Dual) else Dual(o, 0.0)

    def __add__(self, o): o = self._w(o); return Dual(self.v + o.v, self.d + o.d)
    def __sub__(self, o): o = self._w(o); return Dual(self.v - o.v, self.d - o.d)
    def __mul__(self, o): o = self._w(o); return Dual(self.v * o.v, self.d * o.v + self.v * o.d)
    def __truediv__(self, o): o = self._w(o); return Dual(self.v / o.v, (self.d * o.v - self.v * o.d) / (o.v * o.v))
    def __neg__(self): return Dual(-self.v, -self.d)
    __radd__ = __add__; __rmul__ = __mul__
    def __rsub__(self, o): return self._w(o) - self


def exp(x):
    if isinstance(x, Dual):
        e = exp(x.v); return Dual(e, e * x.d)
    return np.exp(x)


def sqrt(x):
    if isinstance(x, Dual):
        s = sqrt(x.v); return Dual(s, x.d / (2 * s))
    return np.sqrt(x)


def val(x):
    return val(x.v) if isinstance(x, Dual) else x


def relu(x):
    if isinstance(x, Dual):
        return Dual(relu(x.v), x.d * ((val(x.v) > 0) * 1.0))
    return np.maximum(x, 0.0)


def mean(x):
    return Dual(mean(x.v), mean(x.d)) if isinstance(x, Dual) else np.mean(x)


def price(s0, sigma, z):
    st = s0 * exp((R - sigma * sigma / 2) * T + sigma * sqrt(T) * z)
    return float(np.exp(-R * T)) * mean(relu(st - K))


def black_scholes():
    d1 = (np.log(S0 / K) + (R + SIGMA**2 / 2) * T) / (SIGMA * np.sqrt(T))
    pdf = np.exp(-d1**2 / 2) / np.sqrt(2 * np.pi)
    return N.cdf(d1), S0 * pdf * np.sqrt(T), pdf / (S0 * SIGMA * np.sqrt(T))


def run(p, emit, stop):
    delta_bs, vega_bs, gamma_bs = black_scholes()
    emit(stat("exact", "Exact delta (Black-Scholes)", delta_bs, ".4f"))
    rng = np.random.default_rng(0)
    NP, REP = int(p["paths"]), int(p["repeats"])
    res = {k: [] for k in ["autodiff", "bump 1.0, same randoms", "bump 1.0, fresh randoms"]}
    vega_ad, g_ad, g_lr, g_bump = [], [], [], []
    for i in range(REP):
        if stop.is_set():
            return
        z, z2 = rng.standard_normal(NP), rng.standard_normal(NP)
        res["autodiff"].append(price(Dual(S0, 1.0), SIGMA, z).d)
        res["bump 1.0, same randoms"].append(price(S0 + 1, SIGMA, z) - price(S0, SIGMA, z))
        res["bump 1.0, fresh randoms"].append(price(S0 + 1, SIGMA, z2) - price(S0, SIGMA, z))
        vega_ad.append(price(S0, Dual(SIGMA, 1.0), z).d)
        g_ad.append(price(Dual(Dual(S0, 1.0), Dual(1.0, 0.0)), SIGMA, z).d.d)
        st = S0 * np.exp((R - SIGMA**2 / 2) * T + SIGMA * np.sqrt(T) * z)
        h = np.where(st > K, st, 0.0)
        g_lr.append(np.exp(-R * T) * (np.mean(h * z) / (S0**2 * SIGMA * np.sqrt(T)) - np.mean(h) / S0**2))
        g_bump.append(price(S0 + 1, SIGMA, z) - 2 * price(S0, SIGMA, z) + price(S0 - 1, SIGMA, z))
        if (i + 1) % 10 == 0 or i + 1 == REP:
            names = list(res)
            emit(hist("delta", f"Delta estimates over {i + 1} repeats ({NP:,} paths each); exact {delta_bs:.4f}",
                      res[names[0]], 50, (delta_bs - 0.12, delta_bs + 0.12), name=names[0],
                      overlay=[{"name": n, "values": res[n]} for n in names[1:]], vlines=[{"x": delta_bs, "label": "exact"}],
                      xlabel="delta estimate"))
            emit(table("delta_t", "Delta: bias and noise", ["method", "mean", "bias", "noise (std)"],
                       [[n, round(float(np.mean(v)), 4), f"{np.mean(v) - delta_bs:+.4f}", round(float(np.std(v)), 4)] for n, v in res.items()] +
                       [["vega by autodiff", round(float(np.mean(vega_ad)), 3), f"{np.mean(vega_ad) - vega_bs:+.3f}", round(float(np.std(vega_ad)), 3)]]))
            emit(bars("gamma", f"Gamma: naive second-order autodiff returns exactly zero (exact {gamma_bs:.5f})",
                      ["autodiff twice (naive)", "bump and reprice", "likelihood-ratio fix"],
                      [{"name": "mean estimate", "y": [float(np.mean(g_ad)), float(np.mean(g_bump)), float(np.mean(g_lr))],
                        "err": [float(np.std(g_ad)), float(np.std(g_bump)), float(np.std(g_lr))]}],
                      hlines=[{"y": gamma_bs, "label": "exact"}]))
            emit({"progress": (i + 1) / REP})
    se = np.std(res["autodiff"]) / np.sqrt(REP)
    emit(check("Autodiff delta is unbiased", abs(np.mean(res["autodiff"]) - delta_bs) < 4 * se + 1e-4,
               f"bias {np.mean(res['autodiff']) - delta_bs:+.5f}, standard error {se:.5f}"))
    emit(check("Autodiff is far less noisy than fresh-random bumping", np.std(res["autodiff"]) * 5 < np.std(res["bump 1.0, fresh randoms"]),
               f"noise {np.std(res['autodiff']):.4f} vs {np.std(res['bump 1.0, fresh randoms']):.4f}"))
    emit(check("Naive gamma fails exactly as theory says", float(np.max(np.abs(g_ad))) == 0.0, "second derivative of the kinked payoff is 0 on every path"))
    emit(check("Likelihood-ratio gamma recovers the true value", abs(np.mean(g_lr) - gamma_bs) / gamma_bs < 0.05,
               f"{np.mean(g_lr):.5f} vs exact {gamma_bs:.5f}"))
