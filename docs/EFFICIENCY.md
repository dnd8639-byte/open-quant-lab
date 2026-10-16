# Efficiency: forecasting it, and gating strategies with it

The **efficiency ratio** of a period is |net move| ÷ total distance travelled inside it. Near 1, price went one way;
near 0, it went back and forth. Trend rules make money on efficient periods and pay costs for nothing on choppy ones,
so a good forecast of efficiency could switch them on and off. This page covers the two tools the lab has for that
idea, and the rules they follow.

## 1. Is efficiency predictable?

`oql efficiency study DATASET`, the dashboard's **Efficiency** tab, or the assistant's `efficiency_study` tool.

* **Target.** Minute bars → each hour's efficiency (path of 60 one-minute moves). Hourly bars → each UTC day's
  (24 hourly moves). Daily bars → |close − open| ÷ (high − low), which ranks days like the path version (rank
  correlation 0.94 on seven markets in the earlier futures research).
* **Development data only.** The first 70% of the dataset's bars. The holdout is cut off before anything is computed.
* **Walk-forward.** An expanding window: the model is refitted every year (daily) or 30 days (hourly) on everything
  before, and predicts the next block. Out-of-sample R² is measured against the training mean.
* **No look-ahead.** Every feature is known before the period it predicts starts: the previous period's numbers,
  as-of joins for other data, publication lags. A test changes every price after a date and checks that no feature,
  and no forecast, before that date moves.
* **Known answer first.** Each run first checks itself on two synthetic markets: one with planted, persistent
  efficiency regimes (forecast correlation about 0.3 must be found) and a random walk (nothing must be found).
* **Counted.** Every study is a row in the ledger (card `efficiency-lab`).

| feature group | what | needs |
|---|---|---|
| price | last period's efficiency, volatility, range, \|return\|, relative volume; 5- and 20-period averages; the average efficiency of the same hour (or weekday) over the last 20 days (weeks); hour of day, weekend, Monday | the bars |
| calendar | FOMC, CPI, payrolls and EIA days (hours), known in advance | `calendar/us_releases/1d` |
| funding | perpetual funding rate as of the period start, its size and change | `binance-funding/SYMBOL/8h` |
| basis | perpetual vs spot price at the previous hour's close, and its 24-hour change | `binance-futures-um/SYMBOL/1h` and `binance/SYMBOL/1h` |
| macro | VIX and the 10y-2y slope, previous US day | `fred/VIXCLS/1d`, `fred/T10Y2Y/1d` |
| positioning | leveraged funds' net position in CME bitcoin futures, from its publication date | `cftc/financial/1w` |
| orderbook | the Pi's hourly order-book and trade-flow features | `pi/book_features/1h`, hourly targets only |

**Sealed:** the order-book features from the micro study's lock date (2026-10-20) on are its confirmation sample.
They are never read here, whatever the date.

What the study reports: out-of-sample R², the correlation between forecast and realized efficiency (an opening-range
breakout needed about 0.3 to be worth trading), the realized efficiency in the predicted-high third minus the
predicted-low third with a Newey-West t-statistic, the same for a model that only uses the market's own past
efficiency, calibration by forecast decile, and the feature weights.

## 2. Do efficiency gates help a strategy?

A **gate** is part of a strategy. It is computed from the same bars, decided at each bar's close, and multiplies the
rule's position by 1 (on) or 0 (off). Switching on and off is a position change and pays costs.

| gate | on when |
|---|---|
| `none` | always |
| `er_trailing` (lookback 12, 24, 72, 168 bars) | the efficiency of the last `lookback` bars is in its top (`high`) or bottom (`low`) `share` percent of the recent past (rolling percentile over max(500, 10 × lookback) bars) |
| `er_forecast` | the walk-forward forecast of the coming day's (or hour's) efficiency is in the top or bottom `share` percent of the forecasts the model made on its own training data; no forecast in the first year (three months), so off |

`share` is 33 or 50. The opening-range breakout rule `orb` (take the direction of the day's first `hours` hours, hold
to the day's last bar) is in the library for intraday bars but not in the default search.

**Exploring:** `oql efficiency gates DATASET "momentum lookback=20"` (or the tab) tries all 21 gate settings on one
strategy, shows P&L by forecast third, and simulates how strong a forecast would have to be: it blends the realized
efficiency with noise until the forecast has a given correlation and reruns the strategy gated on the top third.
All 21 settings are counted in the ledger.

**Testing:** declare gates in a card's experiment. The search may then add any of them to any strategy, and the gate
test reruns that whole search, gates included, on every shuffled copy:

```python
EXPERIMENT = Experiment(
    datasets=["binance/BTCUSDT/1h"],
    rules=["momentum", "breakout", "orb"],
    gates=["none", "er_trailing", "er_forecast"],
    gate_features=["price", "funding"],        # groups the er_forecast gate may use
    cost_bps={"default": 2.0},
)
```

Why the null is fair: price-based gates are recomputed from the shuffled bars, so a gate that only helped by luck
helps the shuffled searches just as much. Non-price features (funding, macro, ...) keep their real dates on shuffled
bars, so their link to the market is broken there too. Paper trading computes the gate the same way, putting the
catalog's history in front of the live journal so the forecast has its years of data.

Experiments that do not mention gates behave exactly as before (same genomes, same random draws, same results).

## What earlier work found (edge-engine/futures/FINDINGS.md)

On 35 CME futures (100,000 market-days) next-day efficiency was not predictable from price features (out-of-sample
R² 0.0014, below the market's own average). Every gate setting made 12-month momentum worse. An opening-range
breakout on BTC is paid by efficiency, but its gate needed a forecast correlation of about 0.3; the futures model
reached 0.045. The tools here rerun that question per market, with the extra data the collector now has.
