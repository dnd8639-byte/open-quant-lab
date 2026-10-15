# Changelog

Changes that can alter what a test measures are marked **(affects results)**. A frozen card records the lab
version it was frozen with, and its gate report shows both versions if they differ.

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
