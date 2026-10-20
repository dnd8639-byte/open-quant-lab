# Changelog

Changes that can alter what a test measures are marked **(affects results)**. A frozen card records the lab
version it was frozen with, and its gate report shows both versions if they differ.

## 0.5.2
- **Rebalance band** (opt-in, `Experiment(rebalance_band=0.25)`; `write_experiment(rebalance_band=...)` for the
  assistant; "Resizing" in the gate check). A position keeps its size until the volatility-based target has moved more
  than the band away, then jumps to the target, so a strategy stops paying to resize every bar while staying within
  the band of its risk target. Paper trading puts the catalog's history in front of the live bars so its sizes match
  the backtest's. On synthetic hourly data a 25% band cut the cost bill of a slow rule (250-hour momentum) by about
  30% and of a fast one (20-hour) by about 8%: for fast rules most of the cost is direction changes, not resizing.
  Experiments without it are unchanged.

## 0.5.1
- The gate report counts efficiency studies and gate checks on the card's datasets made before its freeze
  ("Efficiency-lab explorations on these datasets before the freeze"). They shape which ideas get proposed, so they
  belong next to the result. Recorded at the freeze; cards frozen earlier say "not recorded".
- A project whose data is not in the catalog says what is missing and how to add it, instead of failing.
- `oql serve` on a port already in use says so (it was reported as a network error).
- No more "All-NaN slice" warning when a feature has no values yet early in a study.

## 0.5.0
- **Efficiency lab** (`oql/efficiency.py`, `oql efficiency study`, the dashboard's Efficiency tab, the assistant's
  `efficiency_study` tool). Is the next day's (hourly bars) or hour's (minute bars) efficiency ratio predictable?
  Walk-forward ridge (or boosted trees) on development data only, from price features plus optional calendar,
  funding, perp-spot basis, VIX and yield slope, CFTC positioning and Pi order-book groups, all joined as of the
  period start. The order-book data from the micro study's lock date on stays sealed. Each run first checks itself on
  a synthetic market with planted efficiency regimes and on a random walk, and is counted in the ledger.
- **Efficiency gates in the search** (`oql/gates.py`). An experiment can declare `gates=["none", "er_trailing",
  "er_forecast"]`; the search may then switch any strategy on only in the top (or bottom) third or half of trailing
  efficiency or of the efficiency forecast. Gates are computed from the bars inside the search, so the gate test
  reruns them on every shuffled copy. Switching pays costs. Paper trading computes the gate the same way, with the
  catalog's history in front of the live journal. Experiments without `gates` are unchanged.
- **Gate check** (`oql efficiency gates`, the Efficiency tab): all 21 gate settings on one strategy, P&L by forecast
  third, and a simulation of how strong a forecast would need to be. Counted in the ledger as 21 explorations.
- New rule `orb` (opening-range breakout on intraday bars), available by name; not in the default search.
- Assistant tools: `explore` and `permutation_check` take an optional gate; `write_experiment` takes `gates` and
  `gate_features` and counts gate settings in the search size; new `efficiency_study`.
- Ledger counts (command line, dashboard, assistant) add up the settings a row stands for, so a gate check counts 21.
- Charts: ranked horizontal bars for long category names.

## 0.4.0
- **Live dashboard.** `oql serve` can run any computation in the background and stream its results to the page as they
  arrive (`oql/live.py`, Server-Sent Events); charts redraw as the work happens, with hover readouts and a validated
  colour palette for light and dark mode (`oql/web/live.js`).
- **Projects tab.** The ten quant mini-projects (tail index, intraday seasonality, overfitting detector, momentum
  crashes, factor neutrality, American put, autodiff Greeks, implied distribution, Hawkes trades, news diffusion) run
  live on the lab's catalog data (`oql/projects/`). Each runs its known-answer checks first and shows them passing or
  failing; the four projects with published real-data results reproduce them.
- **Live market tab.** Public exchange data (Kraken, Coinbase, Binance.US; no account) polled by the server: price,
  the rolling one-hour efficiency ratio, realized volatility and the bid-ask spread, updating on their own
  (`oql/market.py`).
- New data commands: `oql data pull-french` (Fama-French factor tables) and `oql data import-projects DIR` (the
  projects' SPY option chain, licensed and kept local, and a day of Binance.US trades).
- The ledger tab counts tests (gate and holdout runs) separately from explorations.
- Packaging includes the `oql.projects` package and the page script, so a regular `pip install` works.

## 0.3.4
Found in the first dry run with an AI assistant connected over MCP:
- `write_experiment` and `explore` now say that every strategy trades all of an experiment's datasets together as one
  equal-risk portfolio. The assistant had counted 4 lookbacks on 2 coins as 8 strategies.
- Editing a card after asking for review marks the review summary as out of date, on the card and on the dashboard,
  until the assistant sends a fresh one.
- The gate report explains a result worse than every shuffled run: the signal pointed the wrong way more often than
  chance, and testing the opposite idea would be a new card and a new counted test.

## 0.3.3
- `oql data check` compares prices with a relative tolerance. Prices stored as decimals round slightly, and at large
  levels (Nikkei futures in the tens of thousands) that rounding was being reported as broken bars.
- Large one-bar moves are split in two: spikes that reverse on the next bar (the mark of a bad price) and moves that
  stick (real events, or an unadjusted futures roll). Each warning names a date to look at, as does the gap warning.
  This does not change any test result.

## 0.3.2
- **(affects results)** A trade is now a change of direction (enter, exit or flip). Positions are resized every
  bar to hold volatility steady; those resizes are still charged as costs, but they were also being counted as
  trades, so every strategy "traded" every bar and the minimum-trades rule never rejected anything. The minimum
  is now 1 direction change per year per market (`min_trades_per_year=1.0`): it rejects strategies that almost
  never change position, and keeps slow trend systems such as 12-month momentum on daily data.
- Exploration (`explore`, `permutation_check`) now uses the costs and fill delay declared in the card's
  `experiment.py`, so an AI assistant sees what the gate will see. Results report `cost_bps` and
  `direction_changes_per_year`.
- `oql card freeze` records the lab version; the gate report shows the version at the freeze and at the gate.
- `oql ledger` separates tests (gate, holdout) from explorations in its summary.
- `oql data check` warns about bars that are not on the dataset's time grid (found in real Binance data: 43 hourly
  bars after the February 2018 outage opened 28 minutes past the hour).
- The code repository ignores lab state (`cards/`, `board.json`, `LEDGER.csv`): run a lab in its own folder.
- `demo/check_parity.py` checks that the live demo page computes the same numbers as this code.

## 0.3.1
- Binance futures files start with a header row; the parser now reads them, and number columns stored as text
  are converted before saving.

## 0.3.0
- Phase 5: `oql mcp`, an MCP server for AI research assistants. Every exploration is counted and reported in the
  gate report.

## 0.2.0
- Data catalog, validation gate, research board, dashboard (`oql serve`), collector, paper trading.
