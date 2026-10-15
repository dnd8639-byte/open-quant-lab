# Open Quant Lab

A fully open research pipeline for trading strategies: a data catalog, a backtest engine with costs and delays, and a validation gate that the strategy has to pass before anything else happens. The rules of every test are enforced by code, not by good intentions.

Designed and built by David DeConti. MIT licence. Inspired by the shape of QuantConnect's research pipeline (Ideas → Research → Backtest → Paper → Live), rebuilt so that no part of it is paid, and so that a strategy cannot be promoted by trying things until one looks good.

```
idea ──> research ──> preregistered ──> gate ──> holdout ──> paper
  explore freely   rules fingerprinted   whole-search      one-shot     Raspberry Pi
                   (SHA-256)             permutation test  (locked)     paper trader
                                         FAIL ─> archive   FAIL ─> archive
```

## What makes it different

| Rule | How the code enforces it |
|---|---|
| Decide the test before seeing the result | `oql card freeze` fingerprints `PREREGISTRATION.md`, `experiment.py` and every dataset (SHA-256). The gate and the holdout refuse to run if any of them changed. |
| Judge the best of many tries as the best of many tries | The gate reruns the **entire search**, every round, on shuffled copies of the data. p = how often luck alone finds something as good. |
| Count every attempt | Every gate and holdout run goes into `LEDGER.csv`, pass or fail. Changing a tested idea means `oql card revise`: a new card in the same family, and the family's trial count keeps adding up. |
| Look at the holdout once | `oql card holdout` writes its lock file before it computes anything. A second run is refused, even if someone edits `board.json`. |
| No skipping | Cards move one stage at a time; a failed gate or holdout sends the card to the archive. |
| Licensed data never leaks | Every dataset carries a licence tag (`public`, `own`, `licensed`); the whole catalog is git-ignored. |

The lab checks itself: on fake markets with a planted, real edge the gate passes (p = 0.025, the smallest possible with 39 permutations, in 5 of 5 draws); on pure noise it fails (p between 0.30 and 0.85 in 5 of 5 draws). Run `python examples/quickstart.py` to watch both happen.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .            # numpy, pandas, pyarrow
python -m unittest discover tests
python examples/quickstart.py
```

## Use

```bash
oql data pull-fred DGS10 CPIAUCSL                 # free, no key
oql data pull-binance BTCUSDT --interval 1h --start 2018-01
oql data import-databento ../state/v3/contracts --roots ES,CL,GC   # licensed: stays local
oql data list
oql data check

oql card new "Time-series momentum on CME futures" --mechanism "Slow-moving capital and hedging demand make trends persist."
#   edit cards/0001-.../PREREGISTRATION.md and experiment.py
oql card start 1
oql card freeze 1           # then commit and push: the commit time proves the rules came first
oql card gate 1             # search + whole-search permutation test -> cards/0001-.../gate.md
oql card holdout 1          # one shot
oql board --html board.html
oql ledger
oql serve                   # the dashboard: live gate runs, the permutation lab, data, ledger
oql paper deploy 1 --feed binance/BTCUSDT/1h=binanceus:BTC/USDT:1h   # after a holdout PASS
oql collect init && oql collect run                                  # keep free data current (Pi)
```

## The dashboard

`oql serve` opens a local web app (nothing leaves your machine):

- **Research board**: every card by stage. Run a preregistered card's gate from the page and watch the shuffled
  reruns arrive one by one, the null histogram fill in, and the p-value settle.
- **Permutation lab**: pick a market and a strategy; the lab shuffles the market in front of you and builds the
  distribution of luck one permutation at a time. Switch to "best of the whole search" to see selection bias.
  Uses only the first 70% of each dataset, so it never touches a holdout.
- **Data** and **Ledger**: the catalog with quality checks, and every test ever run.

Holdouts are deliberately not runnable from the page: they are one-shot, so they need a typed command.

## Layout

| Path | What it is |
|---|---|
| `oql/catalog.py` | Parquet catalog with metadata, licence tags, SHA-256 and quality checks |
| `oql/sources.py` | FRED, Binance public files, CSV, release calendar, Pi order-book features, Databento front contracts (roll-safe) |
| `oql/engine.py` | Bar engine: decisions at the close, fills a fraction into the next bar, costs per change, equal-risk sizing |
| `oql/strategies.py` | Rule library (breakout, moving-average cross, momentum, reversal); add your own per experiment |
| `oql/search.py` | The safe search loop: fixed rounds, elites survive, every strategy logged |
| `oql/permute.py` | Joint bar permutation across markets (adapted from neurotrader888's MIT-licensed method) |
| `oql/gate.py` | The gate and the holdout |
| `oql/stats.py` | Deflated Sharpe, cost and delay Monte Carlo, long/short split, beta |
| `oql/board.py` | Cards, stages, enforced moves, reports, the HTML board |
| `oql/server.py`, `oql/web/` | The `oql serve` dashboard (standard library only) |
| `oql/collect.py` | Scheduled, incremental, checksum-verified downloads |
| `oql/paper.py` | Paper trading: journal, revisions, reconciliation |
| `oql/ledger.py`, `oql/locks.py` | Every test counted; freeze fingerprints and one-shot locks |

## Roadmap

- [x] Phase 1: data catalog and loaders, plus `oql collect` for scheduled downloads on a Raspberry Pi
- [x] Phase 2: paper trading with the same code as the backtest (`oql paper`), with replay, data and returns
      reconciliation; a NautilusTrader adapter remains optional
- [x] Phase 3: validation gate
- [x] Phase 4: pipeline board (command line + HTML)
- [ ] Phase 5: optional LLM agents through an MCP server that can propose cards and write experiment code, and cannot freeze, run a holdout or move a card past a failed gate

Building on it: [docs/DEVELOPING.md](docs/DEVELOPING.md). Running it on a Pi: [deploy/README.md](deploy/README.md).
See [NOTICE.md](NOTICE.md) for credits.
