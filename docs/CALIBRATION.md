# Does the gate's p-value mean what it says?

The gate claims: "on a market with no edge, a search passes only about 5% of the time (alpha = 0.05)". That is a
claim about the test itself, so it can be measured. `python -m oql.calibration` runs the full gate (search, then the
same search on 39 shuffled copies) hundreds of times on synthetic books whose truth is known, under each null model.

Books: 3 markets of 2,000 daily bars each, development part = first 70%. A smaller search than the default (4 rounds)
so 300 gates fit in about 20 minutes on two cores. Raw results: `calibration_results.json`.

| markets | null | gates | pass rate (p ≤ 0.05) | mean p | shuffled best (median) | real best (median) |
|---|---|---|---|---|---|---|
| no edge, volatility clustering | bars | 100 | 3% (± 2) | 0.57 | 0.64 | 0.62 |
| no edge, volatility clustering | vol_matched | 100 | 4% (± 2) | 0.56 | 0.63 | 0.62 |
| no edge, strong clustering | bars | 100 | 1% (± 1) | 0.60 | 0.69 | 0.60 |
| no edge, strong clustering | vol_matched | 100 | 1% (± 1) | 0.55 | 0.64 | 0.60 |
| real edge (persistent trends) | bars | 100 | 96% (± 2) | 0.03 | 0.80 | 2.93 |
| real edge (persistent trends) | vol_matched | 100 | 98% (± 1) | 0.03 | 0.81 | 2.93 |

What this shows:

1. **Neither null lets luck through.** On markets with no edge both pass at or below the 5% they promise.
2. **The classic bar shuffle is too strict when volatility clusters strongly.** Scrambling *when* volatility was
   high produces shuffled markets on which the best strategy looks better (median 0.69) than on the real no-edge
   market (0.60), so real results are judged against an inflated bar: p-values pile up near 1 (mean 0.60, 90th
   percentile 0.98). A test that is too strict does not produce false discoveries, but it loses real ones.
3. **The volatility-matched null is better calibrated there** (mean p 0.55, shuffled best 0.64) and at least as
   powerful on a real edge (98% vs 96%; the difference is within the noise of 100 gates).
4. Both nulls give mean p slightly above 0.5 even with mild clustering: the whole test leans a little conservative.

What it does not show: how either null behaves on real markets (unknowable without a known truth), or for intraday
strategies. Experiments choose with `nulls=[...]`; declaring both requires p < alpha under each, the stricter test.
