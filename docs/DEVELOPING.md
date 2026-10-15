# Developing Open Quant Lab

How the code fits together, the rules every change must keep, and step-by-step recipes for the changes you will
make most often. Run the tests before and after every change: `python -m unittest discover tests` (about 30 s).

## How it fits together

```
sources.py / collect.py ──> catalog.py ──> experiment.py ──> engine.py + strategies.py ──> search.py
   (get data)              (store + check)  (what to test)    (positions -> net returns)    (the loop)
                                                                                               │
board.py ── freeze (locks.py) ── gate.py (permute.py, stats.py) ── holdout ── paper.py ───────┘
   │                                   │
   └── ledger.py (every test)          └── server.py + web/index.html (dashboard)
```

| Module | Owns | Touch it when |
|---|---|---|
| `catalog.py` | Dataset files, metadata, licence tags, quality checks | Changing how data is stored or checked |
| `sources.py` | One-off importers and downloaders | Adding a data source |
| `collect.py` | Scheduled, incremental downloads (the Pi) | Adding a source that should stay current |
| `strategies.py` | Rule library | Adding a rule everyone can use |
| `engine.py` | Fills, costs, sizing, portfolio | Changing how trading is simulated (rare; affects every result) |
| `search.py` | The search loop and scoring | Changing how strategies are proposed |
| `gate.py`, `stats.py` | The tests and their statistics | Adding a check to the gate report |
| `board.py`, `locks.py`, `ledger.py` | Stages, fingerprints, one-shot locks, the record | Changing pipeline rules (think twice) |
| `paper.py` | Live paper trading and reconciliation | New live feeds or checks |
| `server.py`, `web/index.html` | The dashboard | New pages and endpoints |

## Rules every change must keep

1. **No look-ahead.** Anything computed for bar t uses data up to the close of bar t. A position decided at that
   close earns returns from the next bar on. `tests/test_lab.py::TestEngine` guards this; add a test when you add
   a feature that could leak.
2. **The holdout is untouchable.** Development code paths cut data at the split date before computing anything
   (`experiment.truncate`). The dashboard's permutation lab uses only the first 70%.
3. **Every test is counted.** Anything that runs a gate or holdout writes to the ledger.
4. **Frozen means frozen.** Anything that can change a test's outcome must be inside the fingerprinted files
   (PREREGISTRATION.md, experiment.py, the datasets' SHA-256).
5. **Same code live and in backtest.** Paper trading calls `strategies.signal` and `engine.prepare`; never fork them.
6. **Licences.** New data enters the catalog tagged `public`, `own` or `licensed`. Licensed data never enters git.
7. **Prove it on fake data first.** A new test or statistic must pass a planted edge and fail pure noise
   (`synthetic.trend_market`, `synthetic.noise_market`).

## Recipes

### Add a data source
1. In `sources.py`, write a parser that takes raw bytes and returns a DataFrame (UTC index named `ts`), and a
   function that downloads through `getter` and calls `catalog.write(id, df, license=..., origin=...)`.
2. Add a test that feeds the parser a small fake file (see `TestParsers`).
3. One-off source: add a `data` subcommand in `cli.py`. Recurring source: add a `job_<name>` function in
   `collect.py`, register it in `JOBS`, and add a default entry to `DEFAULT_JOBS`.

### Add a strategy rule
- For one experiment: define a function `(close_array, **params) -> array of +1/-1/0` in that card's
  experiment.py and add it to `CUSTOM_RULES = {"name": (function, {"param": [values]})}`.
- For everyone: add it to `RULES` in `strategies.py` with its parameter grid. The signal at index t may use
  `close[:t+1]` only.

### Add a statistic to the gate report
1. Compute it in `stats.py` from a `Scorer` and the finalist.
2. Call it in `gate.run_gate` and add the value to the returned dict.
3. Show it in `board.gate_markdown` and in the card detail of `web/index.html`.
4. Test it on synthetic data.

### Add a dashboard page
1. In `server.py`, add a route in `Handler.do_GET` that returns JSON (`self._send(200, {...})`).
2. In `web/index.html`, add a `<button data-tab="name">` to the nav, a `<section id="t-name" class="hide">`, and a
   `loadName()` function that calls `api("/api/...")` and renders with `lineChart`, `histogram` or a table.
3. Keep it standard-library and offline: no external scripts, so the dashboard works on the Pi with no internet.

### Add a live feed for paper trading
Write a class with `bars(exchange, symbol, timeframe, limit, now) -> DataFrame` that returns only CLOSED bars
(use `_closed`), then pass it to `paper.step(feed=...)`. `ReplayFeed` is the model.

## Workflow
```bash
git checkout -b my-change
python -m unittest discover tests        # green before
# ...edit...
python -m unittest discover tests        # green after
git commit -am "what changed and why" && git push -u origin my-change
```
