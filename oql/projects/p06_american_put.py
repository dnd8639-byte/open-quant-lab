"""Project 6: the American put and its free boundary (Crank-Nicolson + projected SOR).

An American put can be exercised any time, so at every moment you must ask whether waiting is worth more than
exercising now: no formula, and the exercise boundary must be found together with the value. The Black-Scholes PDE
is stepped backwards from expiry with Crank-Nicolson; each step is solved by projected SOR (iterate, and after each
sweep clamp the value up to the payoff), red-black ordered so numpy does each sweep. Checks: the European put from
the same grid vs the formula; the American put vs a 4,000-step binomial tree; and a convergence study.
"""
from __future__ import annotations

import numpy as np

from ._util import bs_call, check, line, stat, table

TITLE = "American put: the free boundary"
QUESTION = "What is an option worth when you can exercise it any time, and when should you?"
PARAMS = [{"name": "finest", "label": "Finest grid (price points)", "type": "choice", "options": [300, 600, 1200], "default": 600},
          {"name": "sigma", "label": "Volatility", "type": "choice", "options": [0.1, 0.2, 0.4], "default": 0.2}]
K, R, T, S0 = 100.0, 0.05, 1.0, 100.0


def solve(n_s, n_t, sigma, american=True, rannacher=True, s_max=3 * K, omega=1.3, tol=1e-9, stop=None):
    """Returns (S grid, value at t=0, exercise boundary per time step, total PSOR sweeps)."""
    i = np.arange(n_s + 1); S = i * s_max / n_s
    payoff = np.maximum(K - S, 0.0)
    V = payoff.copy()
    al, be, ga = 0.5 * (sigma**2 * i**2 - R * i), -(sigma**2 * i**2 + R), 0.5 * (sigma**2 * i**2 + R * i)
    steps = [(T / n_t / 2, 1.0)] * 4 + [(T / n_t, 0.5)] * (n_t - 2) if rannacher else [(T / n_t, 0.5)] * n_t
    tau, boundary, sweeps = 0.0, [], 0
    red, black = np.arange(1, n_s, 2), np.arange(2, n_s, 2)
    for dt, th in steps:
        if stop is not None and stop.is_set():
            break
        tau += dt
        rhs = V.copy()
        rhs[1:-1] += (1 - th) * dt * (al[1:-1] * V[:-2] + be[1:-1] * V[1:-1] + ga[1:-1] * V[2:])
        lo = K if american else K * np.exp(-R * tau)
        V[0], V[-1] = lo, 0.0
        diag = 1 - th * dt * be
        if american:
            for _ in range(20000):
                change = 0.0
                for idx in (red, black):
                    gs = (rhs[idx] + th * dt * (al[idx] * V[idx - 1] + ga[idx] * V[idx + 1])) / diag[idx]
                    new = np.maximum(payoff[idx], V[idx] + omega * (gs - V[idx]))
                    change = max(change, np.abs(new - V[idx]).max()); V[idx] = new
                sweeps += 1
                if change < tol:
                    break
            ex = np.flatnonzero((V[1:-1] - payoff[1:-1] < 1e-7) & (payoff[1:-1] > 0))
            boundary.append((tau, S[ex.max() + 1] if len(ex) else np.nan))
        else:
            M = np.diag(diag[1:-1]) - th * dt * (np.diag(al[2:-1], -1) + np.diag(ga[1:-2], 1))
            b = rhs[1:-1].copy(); b[0] += th * dt * al[1] * lo
            V[1:-1] = np.linalg.solve(M, b)
    return S, V, boundary, sweeps


def binomial_american_put(n, sigma):
    dt = T / n; u = np.exp(sigma * np.sqrt(dt)); d = 1 / u; q = (np.exp(R * dt) - d) / (u - d); disc = np.exp(-R * dt)
    j = np.arange(n + 1); V = np.maximum(K - S0 * u ** j * d ** (n - j), 0)
    for m in range(n - 1, -1, -1):
        j = np.arange(m + 1)
        V = np.maximum(disc * (q * V[1:] + (1 - q) * V[:-1]), K - S0 * u ** j * d ** (m - j))
    return V[0]


def at_s0(S, V):
    return float(np.interp(S0, S, V))


def run(p, emit, stop):
    sigma = float(p["sigma"])
    bs_put = float(bs_call(S0, K, R, T, sigma)) - S0 + K * np.exp(-R * T)
    S, V, _, _ = solve(300, 150, sigma, american=False)
    err = at_s0(S, V) - bs_put
    emit(check("European put from the grid matches Black-Scholes", abs(err) < 0.01, f"grid {at_s0(S, V):.4f} vs formula {bs_put:.4f}"))
    ref = 2 * binomial_american_put(4000, sigma) - binomial_american_put(2000, sigma)     # Richardson-extrapolated tree
    emit(stat("ref", "American put (binomial tree reference)", ref, ".4f"))
    emit(stat("premium", "Early-exercise premium over the European put", ref - bs_put, ".4f"))
    rows, errs, n = [], [], 75
    while n <= int(p["finest"]):
        if stop.is_set():
            return
        e0 = at_s0(*solve(n, n // 2, sigma, rannacher=False, stop=stop)[:2]) - ref
        S, V, bnd, sw = solve(n, n // 2, sigma, stop=stop)
        e1 = at_s0(S, V) - ref
        ratio = f"{errs[-1][2] / abs(e1):.1f}" if errs else ""
        errs.append((n, abs(e0), abs(e1)))
        rows.append([f"{n} x {n // 2}", f"{e0:+.5f}", f"{e1:+.5f}", ratio, sw])
        emit(table("conv", "Convergence: error vs the tree as the grid is refined (ratio 4 = second order)",
                   ["grid (S x t)", "Crank-Nicolson", "with Rannacher smoothing", "error ratio", "PSOR sweeps"], rows))
        ns = [e[0] for e in errs]
        emit(line("conv_plot", "Convergence study (log-log)", [
            {"name": "Crank-Nicolson", "x": ns, "y": [e[1] for e in errs], "points": True},
            {"name": "with Rannacher smoothing", "x": ns, "y": [e[2] for e in errs], "points": True},
            {"name": "slope of 2nd order", "x": ns, "y": [errs[0][2] * (ns[0] / x) ** 2 for x in ns], "dash": True}],
            x="num", x_log=True, y_log=True, xlabel="price points in the grid", ylabel="|error|"))
        keep = (S >= 50) & (S <= 150)
        Se, Ve, _, _ = solve(300, 150, sigma, american=False)
        ke = (Se >= 50) & (Se <= 150)
        emit(line("value", "The American put never drops below its payoff", [
            {"name": "American put", "x": S[keep].tolist(), "y": V[keep].tolist()},
            {"name": "European put", "x": Se[ke].tolist(), "y": Ve[ke].tolist(), "dash": True},
            {"name": "exercise now (payoff)", "x": S[keep].tolist(), "y": np.maximum(K - S[keep], 0).tolist()}],
            x="num", xlabel="stock price", ylabel="option value", y_range=[0, 50]))
        tau, sb = np.array(bnd).T
        emit(line("boundary", "Free boundary: exercise when the price is below this line", [
            {"name": "exercise boundary", "x": (T - tau).tolist(), "y": sb.tolist()}],
            x="num", xlabel="time (years; expiry at 1.0)", ylabel="stock price"))
        emit(stat("boundary_now", "Exercise now if the price is below (one year left)", float(sb[-1]), ".1f"))
        n *= 2
    if len(errs) >= 2:
        order = np.log(errs[-2][2] / errs[-1][2]) / np.log(2)
        emit(check("Second-order convergence", order > 1.5, f"error shrinks {errs[-2][2] / errs[-1][2]:.1f}x per halving (order {order:.1f})"))
    emit(check("American price matches the binomial tree", abs(errs[-1][2]) < 0.01, f"error {errs[-1][2]:.5f} on the finest grid"))
