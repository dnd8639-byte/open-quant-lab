"""Project 3: backtest overfitting detector (Deflated Sharpe Ratio, Bailey & Lopez de Prado 2014).

Try enough strategies on any data and the best looks excellent by luck alone. Generate random trading rules (random
long/short, random holding periods: they know nothing), backtest them all on real daily prices, and keep the best.
Compare its Sharpe with the formula for the best-of-N expected from pure noise:
    E[max SR] ~ sd(SR) * [ (1 - g) Z(1 - 1/N) + g Z(1 - 1/(N e)) ],  g = 0.5772...
Then the Deflated Sharpe Ratio: the probability that the winner's true Sharpe beats that noise benchmark.
"""
from __future__ import annotations

import numpy as np

from ._util import N, check, daily_log_returns, hist, line, stat

TITLE = "Overfitting detector: the best of many random rules"
QUESTION = "How good does the best of N coin-flip strategies look, and does the Deflated Sharpe Ratio see through it?"
PARAMS = [{"name": "dataset", "label": "Market", "type": "dataset", "kind": "bars", "default": "binance/BTCUSDT/1h"},
          {"name": "rules", "label": "Random rules", "type": "choice", "options": [200, 1000, 5000], "default": 1000}]
EULER = 0.5772156649


def expected_max_sharpe(n_trials, sd_sharpe):
    if n_trials < 2:
        return 0.0
    return sd_sharpe * ((1 - EULER) * N.inv_cdf(1 - 1 / n_trials) + EULER * N.inv_cdf(1 - 1 / (n_trials * np.e)))


def deflated_sharpe(returns, n_trials, sd_sharpe):
    r = np.asarray(returns, float); r = r[np.isfinite(r)]
    sr = r.mean() / r.std(ddof=1)
    z = (r - r.mean()) / r.std()
    skew, kurt = (z ** 3).mean(), (z ** 4).mean()
    sr0 = expected_max_sharpe(n_trials, sd_sharpe)
    return N.cdf((sr - sr0) * np.sqrt(len(r) - 1) / np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2))


def random_rules(n_rules, n_days, rng):
    """Each rule: start long or short at random, flip with probability 1/h each day (h = 1..40 days)."""
    pos = np.empty((n_rules, n_days))
    for i in range(n_rules):
        flips = rng.random(n_days) < 1 / rng.integers(1, 41)
        pos[i] = rng.choice([-1, 1]) * np.where(np.cumsum(flips) % 2 == 0, 1, -1)
    return pos


def run(p, emit, stop):
    rng = np.random.default_rng(42)
    daily = daily_log_returns(p["dataset"]).to_numpy()
    ppy = 365 if p["dataset"].startswith("binance") else 252
    T, NR = len(daily), int(p["rules"])
    emit(stat("days", "Days of real prices", T, ",.0f"))
    sr_all, best_r, curve = [], None, []
    for start in range(0, NR, 50):
        if stop.is_set():
            return
        rets = random_rules(min(50, NR - start), T, rng) * daily      # no costs: generous to the rules
        sr = rets.mean(1) / rets.std(1, ddof=1)
        i = int(sr.argmax())
        if not sr_all or sr[i] > max(sr_all):
            best_r = rets[i]
        sr_all.extend(sr.tolist())
        s = np.array(sr_all)
        ann = s * np.sqrt(ppy)
        em = expected_max_sharpe(len(s), s.std(ddof=1)) * np.sqrt(ppy)
        curve.append((len(s), float(ann.max()), em))
        emit(hist("rules", f"{len(s):,} random rules on real prices: annualized Sharpe of each", ann, 50, (-2.5, 2.5),
                  vlines=[{"x": float(ann.max()), "label": f"best {ann.max():.2f}"}, {"x": em, "label": f"formula {em:.2f}"}],
                  xlabel="annualized Sharpe ratio"))
        emit(line("growth", "The more you try, the better the best looks: observed best vs the formula",
                  [{"name": "best found", "x": [c[0] for c in curve], "y": [c[1] for c in curve]},
                   {"name": "formula for pure noise", "x": [c[0] for c in curve], "y": [c[2] for c in curve], "dash": True}],
                  x="num", xlabel="rules tried", ylabel="annualized Sharpe"))
        emit(stat("best", "Best random rule (annualized Sharpe)", float(ann.max())))
        emit(stat("formula", "Formula: expected best from noise", em))
        emit({"progress": len(s) / NR})
    s = np.array(sr_all)
    best, em = float(s.max() * np.sqrt(ppy)), expected_max_sharpe(NR, s.std(ddof=1)) * np.sqrt(ppy)
    d1, dn = deflated_sharpe(best_r, 1, s.std(ddof=1)), deflated_sharpe(best_r, NR, s.std(ddof=1))
    emit(stat("dsr1", "Deflated Sharpe if you hide the other rules (N = 1)", d1, ".3f"))
    emit(stat("dsrn", f"Deflated Sharpe counting all {NR:,} rules", dn, ".3f"))
    emit(check("The noise formula predicts the best of N", abs(best - em) / em < 0.3,
               f"observed best {best:.2f} vs formula {em:.2f} for {NR:,} rules that know nothing"))
    emit(check("Counting the search removes the false discovery", dn < 0.95,
               f"DSR {d1:.3f} when the search is hidden, {dn:.3f} when it is counted (want > 0.95 to believe it)"))
