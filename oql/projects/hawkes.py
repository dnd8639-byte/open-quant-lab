"""Hawkes process tools shared by projects 9 and 10 (numpy only).

Intensity (expected events per second) with an exponential kernel:
    lambda(t) = mu + sum over past events of  alpha * exp(-beta * (t - t_i))
mu = events that arrive for outside reasons; each event raises the rate by alpha, decaying at beta.
Branching ratio n = alpha / beta = average number of follow-on events each event triggers directly.
n = 0: pure Poisson.  n near 1: almost every event is a reaction to an earlier one.
"""
import numpy as np
from ._util import nelder_mead


def simulate(mu, alpha, beta, horizon, rng):
    """Exact simulation by thinning (Ogata 1981)."""
    t, lam_excess, out = 0.0, 0.0, []
    while True:
        lam_bar = mu + lam_excess
        w = rng.exponential(1 / lam_bar)
        t += w
        if t > horizon:
            return np.array(out)
        lam_excess *= np.exp(-beta * w)
        if rng.random() * lam_bar <= mu + lam_excess:
            out.append(t); lam_excess += alpha


def _excitation(times, beta):
    """A_i = sum_{j<i} exp(-beta (t_i - t_j)), by the O(N) recursion A_i = e^{-beta dt} (1 + A_{i-1})."""
    A = np.zeros(len(times)); decay = np.exp(-beta * np.diff(times))
    for i in range(1, len(times)):
        A[i] = decay[i - 1] * (1 + A[i - 1])
    return A


def neg_loglik(params, times, horizon):
    mu, alpha, beta = np.exp(params)                          # optimize in logs so all three stay positive
    if alpha >= beta:                                          # n >= 1: explosive, not allowed
        return 1e12
    A = _excitation(times, beta)
    comp = mu * horizon + alpha / beta * np.sum(1 - np.exp(-beta * (horizon - times)))
    return -(np.sum(np.log(mu + alpha * A)) - comp)


def fit(times, horizon):
    """Maximum likelihood. Returns dict(mu, alpha, beta, n, loglik, loglik_poisson)."""
    rate = len(times) / horizon
    best = None
    for b0 in (0.1, 1.0, 10.0):                                # a few starting points: the surface is bumpy
        x, f = nelder_mead(lambda p: neg_loglik(p, times, horizon), np.log([rate / 2, b0 / 2, b0]), iters=600)
        if best is None or f < best[1]:
            best = (x, f)
    mu, alpha, beta = np.exp(best[0])
    return dict(mu=mu, alpha=alpha, beta=beta, n=alpha / beta, loglik=-best[1],
                loglik_poisson=len(times) * np.log(rate) - len(times))


def residuals(times, mu, alpha, beta):
    """Ogata's test: gaps between events measured in 'model time' should be Exponential(1)."""
    A = _excitation(times, beta)
    dt = np.diff(times)
    return mu * dt + alpha / beta * (1 + A[:-1]) * (1 - np.exp(-beta * dt))


def ks_exponential(x):
    """Kolmogorov-Smirnov distance from Exponential(1) and its approximate p-value."""
    x = np.sort(x); n = len(x); F = 1 - np.exp(-x)
    d = max(np.max(np.arange(1, n + 1) / n - F), np.max(F - np.arange(n) / n))
    lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * d
    p = 2 * sum((-1) ** (k - 1) * np.exp(-2 * k * k * lam * lam) for k in range(1, 101))
    return d, float(min(max(p, 0.0), 1.0))
