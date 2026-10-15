"""python -m unittest discover tests     (about a minute)"""
import io
import os
import tempfile
import unittest
import zipfile

import numpy as np
import pandas as pd


def _edit(path, *pairs):
    with open(path) as f:
        text = f.read()
    for old, new in pairs:
        text = text.replace(old, new)
    with open(path, "w") as f:
        f.write(text)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["OQL_CATALOG"] = os.path.join(self.tmp.name, "catalog")
        os.environ["OQL_LAB"] = self.tmp.name
        os.environ["OMP_NUM_THREADS"] = "1"

    def tearDown(self):
        self.tmp.cleanup()


class TestCatalog(Base):
    def test_roundtrip_and_metadata(self):
        from oql import catalog, synthetic
        df = synthetic.noise_market(n=300)
        m = catalog.write("synth/x/1d", df, license="own", origin="test")
        back = catalog.read("synth/x/1d")
        self.assertEqual(m.rows, 300)
        self.assertTrue(np.allclose(back["close"].to_numpy(), df["close"].to_numpy()))
        self.assertEqual(str(back.index.tz), "UTC")
        self.assertEqual(catalog.check("synth/x/1d")["issues"], [])

    def test_check_finds_broken_bars(self):
        from oql import catalog, synthetic
        df = synthetic.noise_market(n=300)
        df.iloc[10, df.columns.get_loc("low")] = df["high"].iloc[10] * 1.1   # low above high
        df.iloc[20, df.columns.get_loc("close")] = -1.0
        catalog.write("synth/bad/1d", df, license="own", origin="test")
        issues = " ".join(catalog.check("synth/bad/1d")["issues"])
        self.assertIn("non-positive", issues)
        self.assertIn("outside low-high", issues)

    def test_check_finds_bars_off_the_grid(self):
        """Seen in real Binance data: after the February 2018 outage, 43 hourly bars opened 28 minutes past the hour."""
        from oql import catalog, synthetic
        df = synthetic.noise_market(n=300)
        df.index = pd.date_range("2018-01-01", periods=300, freq="1h", tz="UTC", name="ts")
        df.index = df.index.where(~((df.index >= df.index[100]) & (df.index < df.index[143])), df.index + pd.Timedelta("28min"))
        catalog.write("synth/offgrid/1h", df, license="own", origin="test")
        r = catalog.check("synth/offgrid/1h")
        self.assertEqual(r["off_grid"], 43)
        self.assertIn("not on the 1h grid", " ".join(r["warnings"]))
        self.assertEqual(catalog.check("synth/offgrid/1h")["issues"], [])            # a warning to look at, not an error

    def test_licence_required(self):
        from oql import catalog, synthetic
        with self.assertRaises(ValueError):
            catalog.write("a/b/1d", synthetic.noise_market(n=50), license="free-for-all", origin="x")


class TestParsers(unittest.TestCase):
    def test_binance_microsecond_timestamps(self):
        from oql.sources import parse_binance_zip
        rows = ("1735689600000000,94000,94100,93900,94050,12.5,1735693199999999,1.1e6,900,6.1,5.7e5,0\n"
                "1735693200000000,94050,94200,94000,94150,10.0,1735696799999999,9e5,800,5,4.7e5,0\n")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("x.csv", rows)
        d = parse_binance_zip(buf.getvalue())
        self.assertEqual(str(d.index[0]), "2025-01-01 00:00:00+00:00")
        self.assertEqual(d["close"].iloc[1], 94150)

    def test_binance_futures_header_row(self):
        """Futures files start with a header row; mixed with headerless months they must still save as numbers."""
        import pandas as pd
        from oql.sources import parse_binance_zip

        def z(text):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as f:
                f.writestr("x.csv", text)
            return buf.getvalue()
        head = ("open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,"
                "taker_buy_quote_volume,ignore\n")
        a = parse_binance_zip(z(head + "1704067200000,42000,42100,41900,42050,120.5,1704070799999,5e6,91267,60,2.5e6,0\n"))
        b = parse_binance_zip(z("1704070800000,42050,42200,42000,42150,100.0,1704074399999,4e6,80000,50,2e6,0\n"))
        self.assertEqual(len(a), 1)
        df = pd.concat([a, b])
        self.assertTrue(pd.api.types.is_numeric_dtype(df["trades"]))
        self.assertEqual(int(df["trades"].iloc[0]), 91267)
        from oql import catalog
        before = os.environ.get("OQL_CATALOG")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["OQL_CATALOG"] = tmp
            try:
                m = catalog.write("binance-futures-um/BTCUSDT/1h", df, license="public", origin="test")
                self.assertEqual(m.rows, 2)
                self.assertEqual(int(catalog.read(m.id)["trades"].iloc[0]), 91267)
            finally:
                if before is None:
                    os.environ.pop("OQL_CATALOG", None)
                else:
                    os.environ["OQL_CATALOG"] = before

    def test_fred_missing_values(self):
        from oql.sources import parse_fred
        d = parse_fred("observation_date,DGS10\n2024-01-02,3.95\n2024-01-03,.\n2024-01-04,3.99\n", "DGS10")
        self.assertEqual(list(d["value"]), [3.95, 3.99])


class TestEngine(unittest.TestCase):
    def _arrays(self, lc, intra):
        ret = np.diff(lc, prepend=lc[0])
        n = len(lc)
        return dict(close=np.exp(lc), ret=ret, intra=intra, gap=ret - intra, size=np.ones(n), win=np.ones(n, bool), cost=0.0)

    def test_no_lookahead_buy_and_hold(self):
        """Long from the first close, no delay, no cost: earns everything after the NEXT open (the decision is made
        at the close, so the overnight gap into the first bar belongs to the old, flat position)."""
        from oql.engine import market_returns
        rng = np.random.default_rng(0)
        lc = np.cumsum(rng.normal(0, 0.01, 500)); intra = rng.normal(0, 0.005, 500)
        pnl, trades = market_returns(self._arrays(lc, intra), np.ones(500), delay=0.0)
        a = self._arrays(lc, intra)
        self.assertAlmostEqual(np.nansum(pnl), lc[-1] - lc[0] - a["gap"][1], places=10)
        self.assertEqual(trades, 1)

    def test_knowing_today_is_not_enough(self):
        """A signal equal to TODAY's return sign must not capture today's return (it is traded tomorrow)."""
        from oql.engine import market_returns
        rng = np.random.default_rng(1)
        lc = np.cumsum(rng.normal(0, 0.01, 4000)); intra = np.diff(lc, prepend=lc[0])
        a = self._arrays(lc, intra)
        pnl, _ = market_returns(a, np.sign(a["ret"]), delay=0.0)
        self.assertLess(abs(np.nanmean(pnl)) / np.nanstd(pnl) * np.sqrt(len(pnl)), 3.5)

    def test_costs_charged_per_change(self):
        from oql.engine import market_returns
        lc = np.zeros(10); a = self._arrays(lc, np.zeros(10)); a["cost"] = 1e-4
        pnl, trades = market_returns(a, np.array([0, 1, 1, -1, -1, 0, 0, 0, 0, 0.0]), delay=0.0)
        self.assertAlmostEqual(np.nansum(pnl), -(1 + 2 + 1) * 1e-4)
        self.assertEqual(trades, 3)

    def test_resizing_is_not_a_trade(self):
        """Positions are resized every bar to hold volatility steady. That is paid for in costs, but it is not a new
        trade, so the minimum-trades rule still measures how often a strategy changes direction."""
        from oql.engine import market_returns
        n = 50
        a = self._arrays(np.zeros(n), np.zeros(n)); a["cost"] = 1e-4; a["size"] = 1 + 0.1 * np.sin(np.arange(n))
        pnl, trades = market_returns(a, np.ones(n), delay=0.0)
        self.assertEqual(trades, 1)
        self.assertLess(np.nansum(pnl), -1.5e-4)          # entry plus every resize is charged


class TestPermutation(unittest.TestCase):
    def test_keeps_distribution_and_correlation(self):
        from oql import synthetic
        from oql.permute import permute_book
        a = synthetic.noise_market(n=1500, seed=1)
        b = a.copy(); b[["open", "high", "low", "close"]] *= 1.0          # perfectly correlated twin
        p = permute_book({"a": a, "b": b}, seed=3)
        ra, rp = np.log(a.close).diff().dropna(), np.log(p["a"].close).diff().dropna()
        self.assertAlmostEqual(ra.std(), rp.std(), delta=0.15 * ra.std())
        self.assertGreater(np.corrcoef(np.log(p["a"].close).diff().dropna(), np.log(p["b"].close).diff().dropna())[0, 1], 0.99)
        self.assertFalse(np.allclose(a.close.to_numpy(), p["a"].close.to_numpy()))


class TestGate(Base):
    def _book(self, kind, k=4):
        from oql import catalog, synthetic
        ids = []
        for i in range(k):
            df = synthetic.trend_market(seed=i) if kind == "trend" else synthetic.noise_market(seed=50 + i)
            ids.append(catalog.write(f"{kind}/m{i}/1d", df, license="own", origin="synthetic").id)
        return ids

    def test_planted_edge_passes_and_noise_fails(self):
        from oql.experiment import Experiment
        from oql.gate import run_gate
        t = run_gate(Experiment(datasets=self._book("trend")), procs=2, n_perm=39, verbose=False)
        n = run_gate(Experiment(datasets=self._book("noise")), procs=2, n_perm=39, verbose=False)
        self.assertEqual(t["verdict"], "PASS")
        self.assertEqual(n["verdict"], "FAIL")


class TestBoard(Base):
    def _ready_card(self):
        from oql import board, catalog, synthetic
        for i in range(3):
            catalog.write(f"trend/m{i}/1d", synthetic.trend_market(seed=i), license="own", origin="synthetic")
        c = board.new_card("trend test", "planted regimes")
        d = board.card_dir(c)
        _edit(os.path.join(d, "PREREGISTRATION.md"), ("TODO", "done"))
        _edit(os.path.join(d, "experiment.py"), ("datasets=[],", 'datasets=["trend/m0/1d", "trend/m1/1d", "trend/m2/1d"],'),
              ("n_perm=99", "n_perm=39"))
        board.start(c["id"])
        return c, d

    def test_no_skipping(self):
        from oql import board
        c = board.new_card("x")
        with self.assertRaises(board.BoardError):
            board.freeze(c["id"])                    # idea -> preregistered skips research

    def test_todo_blocks_freeze(self):
        from oql import board
        c = board.new_card("x"); board.start(c["id"])
        with self.assertRaises(board.BoardError):
            board.freeze(c["id"])

    def test_tampering_after_freeze_is_refused(self):
        from oql import board, locks
        c, d = self._ready_card()
        board.freeze(c["id"])
        with open(os.path.join(d, "experiment.py"), "a") as f:
            f.write("\n# quietly widened the search\n")
        with self.assertRaises(locks.LockError):
            board.run_gate(c["id"], procs=2)

    def test_full_path_holdout_is_one_shot_and_revisions_count(self):
        from oql import board, ledger, locks
        c, d = self._ready_card()
        board.freeze(c["id"])
        g = board.run_gate(c["id"], procs=2)
        self.assertEqual(g["verdict"], "PASS")
        h = board.run_holdout(c["id"], procs=2)
        self.assertIn(h["verdict"], ("PASS", "FAIL"))
        from oql.board import get, load, save
        bd = load(); get(bd, c["id"])["stage"] = "gate"; save(bd)             # even if someone edits board.json...
        with self.assertRaises(locks.LockError):
            board.run_holdout(c["id"], procs=2)                               # ...the lock file still refuses
        n = board.revise(c["id"])
        self.assertEqual(n["family"], c["id"])
        self.assertEqual(ledger.trials_in_family(os.environ["OQL_LAB"], c["id"]), g["n_trials"])
        html = board.render_html(board.load())
        self.assertIn("trend test", html)


if __name__ == "__main__":
    unittest.main()


class TestCollector(Base):
    def _zip(self, text, name="x.csv"):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr(name, text)
        return buf.getvalue()

    def test_incremental_download_with_checksums(self):
        import hashlib
        import json as _json
        from oql import catalog, collect
        calls = []
        month = self._zip("1704067200000,42000,42500,41900,42400,10,1704070799999,4e5,100,5,2e5,0\n"
                          "1704070800000,42400,42600,42300,42500,8,1704074399999,3e5,90,4,1.7e5,0\n")
        funding = self._zip("calc_time,funding_interval_hours,last_funding_rate\n1704067200000,8,0.0001\n1704096000000,8,0.0002\n")
        cot = self._zip("Market_and_Exchange_Names,Report_Date_as_YYYY-MM-DD,Open_Interest_All\nE-MINI S&P,2024-01-02,100\n")

        def getter(url):
            calls.append(url)
            if url.endswith(".CHECKSUM"):
                body = getter(url[:-9]); calls.pop()
                return (hashlib.sha256(body).hexdigest() + "  f.zip").encode()
            if "monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2024-01.zip" in url:
                return month
            if "fundingRate/BTCUSDT/BTCUSDT-fundingRate-2024-01.zip" in url:
                return funding
            if "fredgraph" in url:
                return b"observation_date,DGS10\n2024-01-02,3.95\n"
            if url.endswith("deacot2024.zip"):
                return cot
            raise FileNotFoundError(url)

        with open(os.path.join(self.tmp.name, "collect.json"), "w") as f:
            _json.dump({"jobs": [
                {"name": "k", "type": "binance_klines", "market": "spot", "interval": "1h", "start": "2024-01", "symbols": ["BTCUSDT"]},
                {"name": "f", "type": "binance_funding", "start": "2024-01", "symbols": ["BTCUSDT"]},
                {"name": "r", "type": "fred", "series": ["DGS10"]},
                {"name": "c", "type": "cftc_cot", "report": "legacy", "start_year": 2024}]}, f)
        collect.run(force=True, getter=getter, verbose=False)
        self.assertEqual(catalog.meta("binance/BTCUSDT/1h").rows, 2)
        self.assertEqual(catalog.meta("binance-funding/BTCUSDT/8h").rows, 2)
        self.assertEqual(catalog.meta("fred/DGS10/1d").rows, 1)
        cot_df = catalog.read("cftc/legacy/1w")
        self.assertEqual(cot_df["released"].iloc[0], "2024-01-05")       # Tuesday as-of -> Friday release
        n = len([u for u in calls if "2024-01.zip" in u and "monthly/klines" in u])
        collect.run(force=True, getter=getter, verbose=False)            # second run: cached, no re-download
        self.assertEqual(len([u for u in calls if "2024-01.zip" in u and "monthly/klines" in u]), n)

    def test_bad_checksum_is_an_error(self):
        import json as _json
        from oql import collect
        blob = self._zip("1704067200000,1,1,1,1,1,1,1,1,1,1,0\n")

        def getter(url):
            if url.endswith(".CHECKSUM"):
                return b"0" * 64 + b"  f.zip"
            if "BTCUSDT-1h-2024-01.zip" in url:
                return blob
            raise FileNotFoundError(url)
        with open(os.path.join(self.tmp.name, "collect.json"), "w") as f:
            _json.dump({"jobs": [{"name": "k", "type": "binance_klines", "market": "spot", "interval": "1h", "start": "2024-01",
                                  "symbols": ["BTCUSDT"]}]}, f)
        st = collect.run(force=True, getter=getter, verbose=False)
        self.assertIn("checksum mismatch", st["k"]["error"])


class TestPaper(Base):
    def test_live_matches_backtest(self):
        """Deploy a card that passed its holdout, feed it bars one day at a time, and reconcile."""
        from oql import board, catalog, paper, synthetic
        frames, ids = {}, []
        for i in range(4):
            df = synthetic.trend_market(seed=i)
            ids.append(catalog.write(f"trend/m{i}/1d", df, license="own", origin="synthetic").id)
            frames[f"replay:SYN{i}:1d"] = df
        c = board.new_card("paper test", "planted regimes")
        d = board.card_dir(c)
        _edit(os.path.join(d, "PREREGISTRATION.md"), ("TODO", "done"))
        _edit(os.path.join(d, "experiment.py"), ("datasets=[],", f"datasets={ids!r},"), ("n_perm=99", "n_perm=39"))
        board.start(c["id"]); board.freeze(c["id"])
        self.assertEqual(board.run_gate(c["id"], procs=2)["verdict"], "PASS")
        self.assertEqual(board.run_holdout(c["id"], procs=2)["verdict"], "PASS")
        days = frames["replay:SYN0:1d"].index
        paper.deploy(c["id"], {ds: f"replay:SYN{i}:1d" for i, ds in enumerate(ids)}, start_utc=days[-80].timestamp())
        feed = paper.ReplayFeed(frames)
        for day in days[-80:]:
            paper.step(feed=feed, now=day + pd.Timedelta(days=1, hours=1), verbose=False)
        for ds in ids:
            rep = paper.reconcile(c["id"])["markets"][ds]
            self.assertEqual(rep["replay"]["mismatches"], 0)
            self.assertGreater(rep["replay"]["decisions"], 70)
            self.assertLess(rep["data"]["max_abs_bps"], 1e-6)
            self.assertGreater(rep["returns"]["correlation"], 0.999)
            self.assertAlmostEqual(rep["returns"]["paper_total_pct"], rep["returns"]["backtest_total_pct"], places=6)
        self.assertIn("paper return", paper.status())
        self.assertEqual(board.get(board.load(), c["id"])["stage"], "paper")


class TestAgent(Base):
    def _rpc(self, msgs):
        import json as _json
        from oql.mcp_server import serve
        out = io.StringIO()
        serve(io.StringIO("\n".join(_json.dumps(m) for m in msgs) + "\n"), out)
        return [_json.loads(l) for l in out.getvalue().splitlines()]

    def test_protocol_and_permissions(self):
        import json as _json
        from oql import catalog, synthetic
        for i in range(3):
            catalog.write(f"trend/m{i}/1d", synthetic.trend_market(seed=i), license="own", origin="synthetic")
        call = lambda i, name, args: {"jsonrpc": "2.0", "id": i, "method": "tools/call", "params": {"name": name, "arguments": args}}
        r = self._rpc([
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            call(3, "propose_idea", {"title": "Trend", "hypothesis": "Trends persist.",
                                     "mechanism": "Regimes persist by construction in these synthetic markets, so momentum is paid."}),
            call(4, "start_research", {"card_id": "1"}),
            call(5, "write_experiment", {"card_id": "1", "datasets": ["trend/m0/1d", "trend/m1/1d", "trend/m2/1d"], "n_perm": 39}),
            call(6, "explore", {"card_id": "1", "dataset": "trend/m0/1d", "rule": "momentum", "params": {"lookback": 20}}),
            call(7, "freeze", {"card_id": "1"}),
            call(8, "write_experiment", {"card_id": "1", "datasets": ["trend/m0/1d"], "n_perm": 5}),
            {"jsonrpc": "2.0", "id": 9, "method": "nonsense"},
        ])
        self.assertEqual(r[0]["result"]["serverInfo"]["name"], "open-quant-lab")
        names = {t["name"] for t in r[1]["result"]["tools"]}
        self.assertIn("explore", names)
        self.assertFalse(names & {"freeze", "gate", "holdout", "paper_deploy"})          # never exposed
        self.assertFalse(r[2]["result"]["isError"])
        self.assertFalse(r[4]["result"]["isError"], r[4]["result"]["content"][0]["text"])
        ex = _json.loads(r[5]["result"]["content"][0]["text"])
        self.assertEqual(ex["explorations_on_this_card"], 1)
        self.assertTrue(r[6]["result"]["isError"])                                       # no freeze tool
        self.assertTrue(r[7]["result"]["isError"])                                       # n_perm 5 refused
        self.assertEqual(r[8]["error"]["code"], -32601)

    def test_explorations_are_reported_at_the_gate(self):
        from oql import agent_tools as T, board, catalog, synthetic
        for i in range(3):
            catalog.write(f"trend/m{i}/1d", synthetic.trend_market(seed=i), license="own", origin="synthetic")
        T.propose_idea("Trend", "Regimes persist by construction in these synthetic markets, so momentum is paid.", "Trends persist.")
        T.start_research("1")
        T.write_experiment("1", ["trend/m0/1d", "trend/m1/1d", "trend/m2/1d"], n_perm=39)
        for lb in (5, 20, 60):
            T.explore("1", "trend/m0/1d", "momentum", {"lookback": lb})
        out = T.explore("1", "trend/m0/1d", "momentum", {"lookback": 120})
        self.assertLess(out["dev_data"][-10:], "2021-01-01")                            # never past the dev split
        T.write_preregistration("1", hypothesis="Trends persist.", data="three synthetic markets", strategy_space="library rules",
                                what_would_change_my_mind="gate p above 0.05")
        T.request_review("1", "Momentum looks strong on development data.")
        board.freeze("1")                                                               # the human step
        with self.assertRaises(T.AgentError):
            T.write_experiment("1", ["trend/m0/1d"])                                    # frozen: agent locked out
        rep = board.run_gate("1", procs=2)
        self.assertEqual(rep["explorations_before_freeze"], 4)
        from oql import __version__
        with open(os.path.join(board.card_dir(board.get(board.load(), "1")), "gate.md")) as f:
            text = f.read()
        self.assertIn("Exploratory evaluations before the freeze | 4", text)
        self.assertIn(f"Lab version at the freeze / at the gate | {__version__} / {__version__}", text)

    def test_exploration_uses_the_cards_costs(self):
        """The assistant must see what the gate will see: the experiment's costs, not a cheaper default."""
        from oql import agent_tools as T, catalog, synthetic
        catalog.write("noise/m0/1d", synthetic.noise_market(seed=3), license="own", origin="synthetic")
        T.propose_idea("Reversal", "Liquidity providers are paid to absorb one-sided flow, so short moves partly revert.", "Moves revert.")
        T.start_research("1")
        cheap = T.explore("1", "noise/m0/1d", "reversal", {"lookback": 1})              # no experiment yet: 1 bp
        T.write_experiment("1", ["noise/m0/1d"], rules=["reversal"], cost_bps={"default": 25}, n_perm=39)
        dear = T.explore("1", "noise/m0/1d", "reversal", {"lookback": 1})
        self.assertEqual((cheap["cost_bps"], dear["cost_bps"]), (1.0, 25.0))
        self.assertLess(dear["sharpe_dev"], cheap["sharpe_dev"] - 1.0)                # daily reversal pays heavily for costs
        self.assertGreater(dear["direction_changes_per_year"], 50)
