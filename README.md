# Open Quant Lab

A fully open research pipeline for trading strategies: a data catalog, a backtest engine with costs and delays, and a validation gate that the strategy has to pass before anything else happens. The rules of every test are enforced by code, not by good intentions.

Designed and built by David DeConti. MIT licence. Inspired by the shape of QuantConnect's research pipeline (Ideas → Research → Backtest → Paper → Live), rebuilt so that no part of it is paid, and so that a strategy cannot be promoted by trying things until one looks good.

**Live demo:** the validation gate runs in your browser at [dnd8639-byte.github.io/open-quant-lab](https://dnd8639-byte.github.io/open-quant-lab/).
It is a JavaScript port of the engine, rules and search; `python demo/check_parity.py` confirms that it computes the same
numbers as this code.

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
oql data pull-french                              # Fama-French factors, for the projects
oql data import-projects ../quant-projects/data   # the projects' option chain (licensed) and trades
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
oql efficiency study binance/BTCUSDT/1h --groups funding,basis,macro   # is efficiency predictable? (dev data, logged)
oql efficiency gates binance/BTCUSDT/1h "momentum lookback=20"          # does an efficiency gate help this strategy?
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
- **Projects**: ten quant mini-projects (tail risk, seasonality, overfitting, momentum crashes, factor neutrality,
  American options, autodiff Greeks, implied distributions, Hawkes processes, news diffusion) running live on the
  catalog's data. Each runs its known-answer checks first; charts fill in as the work happens.
- **Live market**: price, the rolling one-hour efficiency ratio, volatility and the spread from public exchange data
  (Kraken, Coinbase, Binance.US), updating every few seconds. No account or key.
- **Efficiency**: can the next day's (or hour's) efficiency be forecast from price, calendar, funding, basis, macro,
  positioning and order-book data, walk-forward on development data? And does switching a strategy on and off by
  efficiency help it, and how strong would the forecast need to be? Every run checks itself on synthetic markets
  first and is counted in the ledger. Gates found here are tested for real by declaring them in a card's experiment
  (`gates=[...]`), where the gate test reruns them on shuffled data. See [docs/EFFICIENCY.md](docs/EFFICIENCY.md).

Holdouts are deliberately not runnable from the page: they are one-shot, so they need a typed command.

## Layout

| Path | What it is |
|---|---|
| `oql/catalog.py` | Parquet catalog with metadata, licence tags, SHA-256 and quality checks |
| `oql/sources.py` | FRED, Binance public files, CSV, release calendar, Pi order-book features, Databento front contracts (roll-safe) |
| `oql/engine.py` | Bar engine: decisions at the close, fills a fraction into the next bar, costs per change, equal-risk sizing |
| `oql/strategies.py` | Rule library (breakout, moving-average cross, momentum, reversal, opening-range breakout); add your own per experiment |
| `oql/gates.py` | Efficiency gates a search can add to any strategy (trailing efficiency, or the efficiency forecast) |
| `oql/efficiency.py` | The efficiency lab: targets, look-ahead-free features, walk-forward forecasts, gate checks |
| `oql/search.py` | The safe search loop: fixed rounds, elites survive, every strategy logged |
| `oql/permute.py` | Joint bar permutation across markets (adapted from neurotrader888's MIT-licensed method), plus a volatility-matched null |
| `oql/calibration.py` | Measures the gate's own false-positive rate and power on synthetic books (docs/CALIBRATION.md) |
| `oql/gate.py` | The gate and the holdout |
| `oql/stats.py` | Deflated Sharpe, cost and delay Monte Carlo, long/short split, beta |
| `oql/board.py` | Cards, stages, enforced moves, reports, the HTML board |
| `oql/server.py`, `oql/web/` | The `oql serve` dashboard (standard library only) |
| `oql/live.py`, `oql/web/live.js` | Background jobs streamed to the page; the shared live chart renderer |
| `oql/projects/` | The ten mini-projects, each with its known-answer checks |
| `oql/market.py` | Live public exchange data for the Live market tab |
| `oql/collect.py` | Scheduled, incremental, checksum-verified downloads |
| `oql/paper.py` | Paper trading: journal, revisions, reconciliation |
| `oql/agent_tools.py`, `oql/mcp_server.py` | What an AI assistant may do, served over MCP |
| `oql/ledger.py`, `oql/locks.py` | Every test counted; freeze fingerprints and one-shot locks |
| `demo/check_parity.py` | Checks that the live demo page computes the same numbers as the Python lab |

## Roadmap

- [x] Phase 1: data catalog and loaders, plus `oql collect` for scheduled downloads on a Raspberry Pi
- [x] Phase 2: paper trading with the same code as the backtest (`oql paper`), with replay, data and returns
      reconciliation; a NautilusTrader adapter remains optional
- [x] Phase 3: validation gate
- [x] Phase 4: pipeline board (command line + HTML)
- [x] Phase 5: an MCP server (`oql mcp`) so an AI assistant can propose cards, write pre-registrations and
      experiments, and explore development data; it cannot freeze, run the gate or holdout, or deploy, and every
      exploration it makes is counted in the gate report. See [docs/AGENTS.md](docs/AGENTS.md)

Building on it: [docs/DEVELOPING.md](docs/DEVELOPING.md). Running it on a Pi: [deploy/README.md](deploy/README.md).
See [NOTICE.md](NOTICE.md) for credits.
