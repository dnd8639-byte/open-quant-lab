"""The second null model and the calibration study.   python -m unittest tests.test_nulls"""
import os
import tempfile
import unittest

import numpy as np


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["OQL_CATALOG"] = os.path.join(self.tmp.name, "catalog")
        os.environ["OQL_LAB"] = self.tmp.name
        os.environ["OMP_NUM_THREADS"] = "1"

    def tearDown(self):
        self.tmp.cleanup()


def abs_autocorr(close):
    a = np.abs(np.diff(np.log(close)))
    return np.corrcoef(a[1:], a[:-1])[0, 1]


class TestVolMatchedNull(Base):
    def test_keeps_the_volatility_path_and_destroys_order(self):
        from oql.calibration import clustered_market
        from oql.permute import permute_book, smooth_vol
        df = clustered_market(3000, seed=1, persistence=0.99, shock=0.15)
        bars = permute_book({"x": df}, 3, method="bars")["x"]
        vm = permute_book({"x": df}, 3, method="vol_matched")["x"]
        self.assertLess(abs(abs_autocorr(bars.close.values)), 0.05)                    # bar shuffle: clustering gone
        self.assertGreater(abs_autocorr(vm.close.values), 0.5 * abs_autocorr(df.close.values))   # kept
        rv, rr = smooth_vol(vm.close.values), smooth_vol(df.close.values)
        self.assertGreater(np.corrcoef(np.log(rv), np.log(rr))[0, 1], 0.8)             # calm stays calm, wild stays wild
        r = np.diff(np.log(df.close.values)); rp = np.diff(np.log(vm.close.values))
        self.assertLess(abs(np.corrcoef(r[1:], rp[1:])[0, 1]), 0.1)                     # the moves themselves are reordered
        ok = (vm.high >= vm[["open", "close"]].max(axis=1) - 1e-9) & (vm.low <= vm[["open", "close"]].min(axis=1) + 1e-9)
        self.assertTrue(ok.all())
        with self.assertRaises(ValueError):
            permute_book({"x": df}, 3, method="nope")

    def test_gate_with_both_nulls_passes_planted_and_fails_noise(self):
        from oql import catalog, synthetic
        from oql.experiment import Experiment
        from oql.gate import run_gate
        out = {}
        for kind in ("trend", "noise"):
            ids = []
            for i in range(4):
                df = synthetic.trend_market(seed=i) if kind == "trend" else synthetic.noise_market(seed=50 + i)
                ids.append(catalog.write(f"{kind}/m{i}/1d", df, license="own", origin="synthetic").id)
            out[kind] = run_gate(Experiment(datasets=ids, nulls=["bars", "vol_matched"]), procs=2, n_perm=39, verbose=False)
        self.assertEqual(out["trend"]["verdict"], "PASS")
        self.assertIn("vol_matched", out["trend"]["other_nulls"])
        self.assertLess(out["trend"]["other_nulls"]["vol_matched"]["p"], 0.05)
        self.assertEqual(out["noise"]["verdict"], "FAIL")

    def test_unknown_null_is_refused_at_the_freeze(self):
        from oql import board, catalog, synthetic
        catalog.write("noise/m0/1d", synthetic.noise_market(seed=1), license="own", origin="synthetic")
        c = board.new_card("x", "A mechanism long enough to be accepted by the board here.")
        d = board.card_dir(c)
        with open(os.path.join(d, "PREREGISTRATION.md"), "w") as f:
            f.write("# done\n")
        with open(os.path.join(d, "experiment.py"), "w") as f:
            f.write('from oql.experiment import Experiment\nEXPERIMENT = Experiment(datasets=["noise/m0/1d"], nulls=["block"])\n')
        board.start(c["id"])
        with self.assertRaises(board.BoardError):
            board.freeze(c["id"])


class TestLongHistory(unittest.TestCase):
    def test_series_become_tradable_bars(self):
        import pandas as pd
        from oql.sources import series_to_bars
        idx = pd.date_range("1971-01-04", periods=4, freq="B")
        jpy = series_to_bars(pd.Series([360.0, 300.0, "", 240.0], index=idx), "fx_inv")       # yen per dollar
        self.assertAlmostEqual(jpy.close.iloc[-1] / jpy.close.iloc[0], 1.5)                   # yen worth 50% more
        bond = series_to_bars(pd.Series([5.0, 6.0, 6.0, 5.0], index=idx), "bond:8")
        r = np.diff(np.log(bond.close.values))
        self.assertLess(r[0], -0.07); self.assertGreater(r[2], 0.07)                         # yields up = price down
        self.assertTrue((bond.high >= bond.close).all() and (bond.low <= bond.open).all())


class TestExcessReturns(unittest.TestCase):
    def test_financing_is_subtracted_and_carry_added(self):
        import pandas as pd
        from oql.sources import series_to_excess_bars
        idx = pd.bdate_range("1980-01-01", periods=600)
        tbill = pd.Series(8.0, index=pd.bdate_range("1979-01-01", periods=900))
        flat = series_to_excess_bars(pd.Series(8.0, index=idx), "bond:8", tbill)            # yield = T-bill, unchanged
        self.assertAlmostEqual(flat.close.iloc[-1] / flat.close.iloc[0], 1.0, places=9)    # earns nothing in excess
        fx = series_to_excess_bars(pd.Series(2.0, index=idx), "fx", tbill,
                                   pd.Series(12.0, index=pd.date_range("1979-01-01", periods=40, freq="MS")))
        r = np.log(fx.close).diff().dropna()
        self.assertAlmostEqual(r.mean() * 252, 0.04, places=4)                              # 12% - 8% carry, flat spot


class TestLargeSearch(unittest.TestCase):
    def test_small_version_runs_and_finds_the_planted_edge(self):
        import threading
        from oql.projects import p11_large_search as P
        P_sizes = P.experiment(30)
        self.assertGreater(len(P_sizes.rule_table()["ma_cross"][1]["fast"]), 50)            # the dense grid is in use
        ev = []
        P.run({"market": "planted edge (persistent trends)", "largest": 30, "shuffles": 39}, ev.append, threading.Event())
        checks = [e for e in ev if "check" in e]
        self.assertTrue(checks and all(c["passed"] for c in checks), checks)


class TestCalibration(unittest.TestCase):
    def test_one_calibration_gate(self):
        from oql.calibration import _one, summarize
        r = [_one(("noise", 0, 9)), _one(("trend", 1, 9))]
        self.assertTrue(all(0 < x["bars"] <= 1 and 0 < x["vol_matched"] <= 1 for x in r))
        self.assertEqual(r[1]["bars"], 0.1)                                              # a real edge beats all 9
        self.assertEqual(len(summarize(r)), 4)


if __name__ == "__main__":
    unittest.main()
