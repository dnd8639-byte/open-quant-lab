"""The efficiency lab and efficiency gates.   python -m unittest tests.test_efficiency   (about a minute)"""
import os
import tempfile
import threading
import unittest

import numpy as np
import pandas as pd


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["OQL_CATALOG"] = os.path.join(self.tmp.name, "catalog")
        os.environ["OQL_LAB"] = self.tmp.name
        os.environ["OMP_NUM_THREADS"] = "1"

    def tearDown(self):
        self.tmp.cleanup()


def hourly(c, start="2024-01-01"):
    idx = pd.date_range(start, periods=len(c), freq="1h", tz="UTC")
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.0001, "low": np.minimum(o, c) * 0.9999, "close": c,
                         "volume": 1.0}, index=idx)


class TestEfficiencyLab(Base):
    def test_path_efficiency(self):
        from oql.efficiency import path_er
        straight = hourly(100 + np.arange(48.0))                       # one way all day: ER 1
        zigzag = hourly(100 + np.tile([0.0, 1.0], 24))                 # back and forth: ER near 0
        self.assertTrue(np.allclose(path_er(straight, "1D", 20)["er"], 1.0))
        self.assertLess(path_er(zigzag, "1D", 20)["er"].max(), 0.05)

    def test_self_checks_find_planted_and_ignore_noise(self):
        from oql import efficiency as E
        ev = []
        self.assertTrue(E.self_checks(ev.append))
        self.assertEqual(len([e for e in ev if e.get("passed")]), 2)

    def test_features_never_use_the_future(self):
        """Changing everything after a date must not change any feature, or the forecast, for periods before it."""
        from oql import catalog, efficiency as E
        bars = E.synthetic_bars(True, n_days=900, seed=1)
        catalog.write("binance/SYN/1h", bars, license="own", origin="test")
        funding = pd.DataFrame({"funding_rate": np.random.default_rng(0).normal(0, 1e-4, 1500)},
                               index=pd.date_range("2018-01-01", periods=1500, freq="8h", tz="UTC"))
        catalog.write("binance-funding/SYN/8h", funding, license="own", origin="test", kind="table")
        cut = bars.index[600 * 24]
        later = bars.copy()
        later.loc[later.index >= cut, ["open", "high", "low", "close"]] *= np.exp(np.cumsum(
            np.random.default_rng(9).normal(0, 0.02, (later.index >= cut).sum())))[:, None]
        per_a, h = E.periods_from_bars(bars)
        per_b, _ = E.periods_from_bars(later)
        Xa, _ = E.features("binance/SYN/1h", per_a, h, ["price", "funding"])
        Xb, _ = E.features("binance/SYN/1h", per_b, h, ["price", "funding"])
        before = Xa.index <= cut                                         # the period starting at the cut uses only earlier days
        pd.testing.assert_frame_equal(Xa[before], Xb[before])
        ra, rb = E.forecast_rank(per_a, Xa, h), E.forecast_rank(per_b, Xb, h)
        pd.testing.assert_series_equal(ra[before], rb[before])
        self.assertGreater(ra[before].notna().sum(), 200)                 # the comparison covers real forecasts

    def test_order_book_confirmation_data_stays_sealed(self):
        from oql import catalog, efficiency as E
        idx = pd.date_range("2026-09-01", "2026-11-30", freq="1h", tz="UTC")
        ob = pd.DataFrame({"imbalance": np.where(idx >= pd.Timestamp(E.SEALED["pi/book_features/1h"], tz="UTC"), 1e6, 0.1)}, index=idx)
        catalog.write("pi/book_features/1h", ob, license="own", origin="test", kind="table")
        per = pd.DataFrame({"er": 0.5, "rv": 0.01, "log_range": 0.01, "abs_ret": 0.0, "volume": 1.0}, index=idx)
        X, _ = E.features("binance/SYN/1m", per, "hourly", ["orderbook"])
        self.assertLess(np.nanmax(X["ob_imbalance"].to_numpy()), 1.0)     # nothing from the lock date on

    def test_study_uses_development_data_only_and_is_logged(self):
        from oql import catalog, efficiency as E, ledger
        bars = E.synthetic_bars(False, n_days=1200, seed=2)
        catalog.write("binance/SYN/1h", bars, license="own", origin="test")
        s = E.study("binance/SYN/1h", ["price"])
        self.assertLess(pd.Timestamp(s["dev_end"], tz="UTC"), bars.index[int(len(bars) * 0.7)])
        rows = ledger.rows(os.environ["OQL_LAB"])
        self.assertEqual((rows[-1]["card"], rows[-1]["kind"]), ("efficiency-lab", "explore"))
        self.assertEqual(E.studies_so_far("binance/SYN/1h"), 1)

    def test_dashboard_jobs(self):
        from oql import catalog, efficiency as E
        catalog.write("binance/SYN/1h", E.synthetic_bars(True, n_days=1100, seed=4), license="own", origin="test")
        for fn, params in [(E.run, {"dataset": "binance/SYN/1h", "funding": True}),
                           (E.gate_check, {"dataset": "binance/SYN/1h", "strategy": "orb hours=2", "cost_bps": 1})]:
            ev = []
            fn(params, ev.append, threading.Event())
            checks = [e for e in ev if "check" in e]
            self.assertTrue(checks and all(c["passed"] for c in checks), checks)
            self.assertTrue(any(e.get("chart") for e in ev))
        self.assertEqual(E.studies_so_far("binance/SYN/1h"), 1 + 21)          # one study + 21 gate settings, all counted
        av = E.availability("binance/SYN/1h")
        self.assertTrue(av["price"][0])
        self.assertFalse(av["funding"][0])


class TestGates(Base):
    def test_opening_range_breakout(self):
        from oql.strategies import orb
        idx = pd.date_range("2024-01-01", periods=72, freq="1h", tz="UTC")
        c = np.r_[100 + np.arange(24.0), 124 - np.arange(24) * 0.5, 112 + np.sin(np.arange(24))]
        s = orb(c, 2, index=idx)
        self.assertTrue((s[:25] == 0).all())                               # day 1 has no close before it
        self.assertTrue((s[25:47] == 1).all() and s[47] == 0)             # decided at 02:00, out at the day's last bar
        keep = np.ones(72, bool); keep[49] = False                         # the deciding bar of day 3 is missing
        self.assertTrue((orb(c[keep], 2, index=idx[keep])[47:] == 0).all())

    def test_gate_is_decided_from_the_past_only(self):
        from oql import efficiency as E
        from oql.gates import mask
        bars = E.synthetic_bars(True, n_days=600, seed=5)
        cut = 400 * 24 + 7
        later = bars.copy()
        later.iloc[cut + 1:, :4] *= 1.3
        for gate in ({"kind": "er_trailing", "lookback": 24, "side": "high", "share": 33},
                     {"kind": "er_forecast", "side": "high", "share": 50}):
            a, b = mask(gate, bars, "x"), mask(gate, later, "x")
            self.assertTrue(np.array_equal(a[:cut + 1], b[:cut + 1]), gate)
            self.assertGreater(a.mean(), 0.15)

    def test_no_gates_means_the_search_is_unchanged(self):
        from oql import efficiency as E
        from oql.engine import prepare
        from oql.experiment import Experiment
        from oql.search import Scorer, random_genome, search
        bars = E.synthetic_bars(True, n_days=400, seed=6)
        exp = Experiment(datasets=["x"])
        self.assertIsNone(exp.gate_table())
        g = random_genome(np.random.default_rng(0), exp.rule_table(), True)
        self.assertNotIn("gate", g)
        res = search(Scorer(prepare({"x": bars}, {"default": 1.0}), exp.rule_table()), exp)
        self.assertTrue(all("gate" not in json for json in [str(res["best"])]))
        gated = Experiment(datasets=["x"], gates=["none", "er_trailing"])
        self.assertIn("gate", random_genome(np.random.default_rng(0), gated.rule_table(), True, gated.gate_table()))

    def test_gated_search_finds_planted_efficiency_and_is_not_fooled_by_noise(self):
        from oql import catalog, efficiency as E
        from oql.experiment import Experiment
        from oql.gate import run_gate
        out = {}
        for planted, seed in ((True, 3), (False, 4)):
            ds = f"synth/{'p' if planted else 'n'}/1h"
            catalog.write(ds, E.synthetic_bars(planted, n_days=1000, seed=seed), license="own", origin="test")
            exp = Experiment(datasets=[ds], rules=["orb", "momentum", "breakout"], gates=["none", "er_trailing", "er_forecast"])
            out[planted] = run_gate(exp, procs=2, n_perm=39, verbose=False)
        self.assertEqual(out[True]["verdict"], "PASS")
        self.assertNotEqual(out[True]["finalist"].get("gate", {}).get("kind", "none"), "none")
        self.assertEqual(out[False]["verdict"], "FAIL")

    def test_paper_trading_applies_the_gate_with_history(self):
        """Live decisions on a short journal must equal the backtest's on the full history (the gate needs years)."""
        from oql import catalog, efficiency as E
        from oql.paper import _decide
        from oql.experiment import Experiment
        from oql.strategies import positions
        bars = E.synthetic_bars(True, n_days=700, seed=7)
        catalog.write("synth/p/1h", bars, license="own", origin="test")
        exp = Experiment(datasets=["synth/p/1h"], rules=["momentum"], gates=["er_forecast"])
        g = {"rule": "momentum", "params": {"lookback": 20}, "side": "both", "gate": {"kind": "er_forecast", "side": "high", "share": 50}}
        journal = bars.iloc[-300:]
        live = _decide(exp.rule_table(), journal, {"finalist": g}, "synth/p/1h", exp)
        full = positions(exp.rule_table(), bars, g, "synth/p/1h")[-300:]
        self.assertTrue(np.array_equal(live, full))
        self.assertGreater(np.abs(live).sum(), 0)


class TestRebalanceBand(Base):
    def test_band_sizes_stay_close_and_change_rarely(self):
        from oql.engine import band_sizes
        rng = np.random.default_rng(0)
        size = np.exp(np.cumsum(rng.normal(0, 0.02, 5000))) * 0.5
        b = band_sizes(size, 0.25)
        self.assertLessEqual(np.max(np.abs(np.log(b / size))), np.log(1.25) + 1e-12)     # never outside the band
        self.assertLess((np.diff(b) != 0).sum(), 0.1 * (np.diff(size) != 0).sum())     # far fewer resizes
        self.assertIs(band_sizes(size, 0.0), size)

    def test_paper_sizes_match_the_backtest_with_a_band(self):
        from oql import catalog, efficiency as E
        from oql.engine import prepare
        from oql.paper import _sizes
        bars = E.synthetic_bars(False, n_days=200, seed=12)
        catalog.write("synth/x/1h", bars, license="own", origin="test")
        dep = {"finalist": {"rule": "momentum", "params": {"lookback": 20}, "side": "both"}, "cost_bps": {"default": 5.0},
               "ppy": 8760.0, "rebalance_band": 0.25}
        full = prepare({"synth/x/1h": bars}, dep["cost_bps"], warmup=0, ppy=8760.0, band=0.25).arrays["synth/x/1h"]["size"]
        self.assertTrue(np.allclose(_sizes(bars.iloc[-300:], dep, "synth/x/1h"), full[-300:], equal_nan=True))

    def test_band_cuts_costs_and_is_declared_in_the_experiment(self):
        from oql import agent_tools as T, catalog, efficiency as E
        from oql.engine import prepare
        from oql.search import Scorer
        from oql.strategies import RULES
        bars = E.synthetic_bars(False, n_days=400, seed=10)
        g = {"rule": "momentum", "params": {"lookback": 250}, "side": "both"}
        paid = {}
        for band in (0.0, 0.25):
            sc = Scorer(prepare({"x": bars}, {"default": 5.0}, band=band), RULES)
            gross, _, _ = sc.returns(g, cost_mult=0.0)
            net, _, _ = sc.returns(g)
            paid[band] = float((gross - net).sum())
        self.assertLess(paid[0.25], 0.8 * paid[0.0])                                   # resizing is part of the bill, not all
        catalog.write("binance/SYN/1h", bars, license="own", origin="test")
        T.propose_idea("Cheap resizing", "Resizing every hour pays costs for nothing; stepwise sizing keeps the risk target.", "x")
        T.start_research("1")
        T.write_experiment("1", ["binance/SYN/1h"], rules=["momentum"], rebalance_band=0.25, n_perm=39)
        from oql import board
        from oql.experiment import load
        self.assertEqual(load(os.path.join(board.card_dir(board.get(board.load(), "1")), "experiment.py")).rebalance_band, 0.25)
        with self.assertRaises(T.AgentError):
            T.write_experiment("1", ["binance/SYN/1h"], rebalance_band=0.9)


class TestReporting(Base):
    def test_efficiency_explorations_are_counted_at_the_freeze(self):
        from oql import board, catalog, efficiency as E, ledger
        catalog.write("binance/SYN/1h", E.synthetic_bars(True, n_days=600, seed=9), license="own", origin="test")
        ledger.append(os.environ["OQL_LAB"], card="efficiency-lab", family="efficiency-lab", kind="explore", n_trials=21,
                      data="binance/SYN/1h", note="gate check")
        c = board.new_card("gated", "Trend rules are paid only on efficient days, so skip forecast-choppy days.")
        d = board.card_dir(c)
        with open(os.path.join(d, "PREREGISTRATION.md"), "w") as f:
            f.write("# done\n")
        with open(os.path.join(d, "experiment.py"), "w") as f:
            f.write('from oql.experiment import Experiment\nEXPERIMENT = Experiment(datasets=["binance/SYN/1h"])\n')
        board.start(c["id"]); board.freeze(c["id"])
        import json
        with open(os.path.join(d, "frozen.json")) as f:
            self.assertEqual(json.load(f)["efficiency_explorations"], 21)

    def test_missing_project_data_is_explained_not_an_error(self):
        from oql import projects
        ev = []
        projects.run("p04_momentum_regimes", {}, ev.append, threading.Event())
        self.assertTrue(any("pull-french" in e.get("log", "") for e in ev))


class TestAgentGates(Base):
    def test_assistant_can_declare_and_explore_gates(self):
        from oql import agent_tools as T, catalog, efficiency as E
        catalog.write("binance/SYN/1h", E.synthetic_bars(True, n_days=1500, seed=8), license="own", origin="test")
        T.propose_idea("Trend days", "Trend-following pays only on days that move one way; skip the choppy ones.", "Gated momentum.")
        T.start_research("1")
        plain = T.write_experiment("1", ["binance/SYN/1h"], rules=["momentum"], n_perm=39)
        gated = T.write_experiment("1", ["binance/SYN/1h"], rules=["momentum", "orb"], gates=["none", "er_forecast"],
                                   gate_features=["funding"], n_perm=39)
        self.assertEqual(gated["strategies_in_search_space"], (5 + 4) * (1 + 4))
        self.assertEqual(plain["strategies_in_search_space"], 5)
        x = T.explore("1", "binance/SYN/1h", "momentum", {"lookback": 20}, gate={"kind": "er_forecast", "side": "high", "share": 50})
        self.assertTrue(0.15 < x["share_of_time_gate_is_on"] < 0.6, x)
        with self.assertRaises(T.AgentError):
            T.explore("1", "binance/SYN/1h", "momentum", {"lookback": 20}, gate={"kind": "er_forecast", "side": "up", "share": 50})
        st = T.efficiency_study("binance/SYN/1h")
        self.assertGreater(st["correlation"], 0.15)
        self.assertIn("efficiency_study", T.TOOLS)


if __name__ == "__main__":
    unittest.main()
