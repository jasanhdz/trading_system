# Aegis W3 Intrabar Wave State and Optimal Timing Result

## Status

`AEGIS_INTRABAR_WAVE_W3_NO_ECONOMIC_EDGE`

```text
W3A_ENTRY_EDGE_FOUND = FALSE
W3A_MODELING_JUSTIFIED = FALSE
W3A_READY_FOR_SHADOW = FALSE
W3A_READY_FOR_LIVE = FALSE

W3B_EXIT_EDGE_FOUND = FALSE
W3B_MODELING_JUSTIFIED = FALSE
W3B_READY_FOR_SHADOW = FALSE
W3B_READY_FOR_LIVE = FALSE

W3_INTRABAR_WAVE_EDGE_FOUND = FALSE
W3_READY_FOR_SHADOW = FALSE
W3_READY_FOR_LIVE = FALSE
```

W3 refutes the tested claim that closed one-minute wave state creates an
economically defendable WAIT/ENTER/HOLD/EXIT policy. The W3 holdout remains
sealed. No production, TypeScript, WebSocket, PM2, credential, or exchange
state was changed.

## Data Integrity

- Complete 1m USD-M klines: 2024-01-01 through 2026-07-31, all 11 symbols,
  with OHLC, volume, trade count, and Binance taker-buy volume.
- Tick-level aggTrades: available from 2025-08 through 2026-07, but not used
  because that interval overlaps W1 selection and sealed evidence.
- Historical order book: unavailable; none was reconstructed.
- TRAIN_W3: 85,004 independent wave episodes.
- VALIDATION_W3: 45,175 independent wave episodes.
- Total: 130,179 episodes, 911,222 W3A decisions, and 2,266,193 W3B decisions.
- A 180-minute purge is enforced at temporal boundaries.
- Dataset audit: `PASS` for identities, finite causal features, one-minute
  execution delay, uniqueness, partition boundaries, and holdout isolation.
- FINAL_HOLDOUT_W3: `SEALED`; outcomes not read.

## W3A Entry Timing

The primary contract asks whether +0.50 ATR occurs before -0.25 ATR within ten
minutes. A decision is made only from a closed minute and is executed at the
next minute open. Logistic and shallow gradient-boosting models were calibrated
on temporal TRAIN data.

| Side | Validation episodes | Model ROC-AUC | Selected policy | Net expectancy | Best baseline | Gate |
|---|---:|---:|---|---:|---|---|
| LONG | 22,354 | 0.491 | ABANDON/no trade | 0.00 bps | No trade | FAIL |
| SHORT | 22,821 | 0.510 | ABANDON/no trade | 0.00 bps | No trade | FAIL |

Every fixed timing offset was economically negative after the registered
14 bps cost:

| Offset after impulse | LONG net | SHORT net |
|---:|---:|---:|
| 0m | -13.432 bps | -14.115 bps |
| 1m | -13.795 bps | -13.455 bps |
| 2m | -13.920 bps | -13.664 bps |
| 3m | -14.234 bps | -13.695 bps |
| 5m | -13.962 bps | -14.009 bps |
| 8m | -13.651 bps | -13.666 bps |
| 10m | -13.414 bps | -14.000 bps |

Immediate, wait-one-minute, pullback, and impulse-extreme-break baselines all
lost money. Pullback/break rules traded fewer episodes, which reduced aggregate
loss, but did not create positive per-opportunity utility. No probability at
the frozen thresholds justified entering, so the model abstained rather than
manufacturing trades.

## W3B Exit Timing

W3B is conditional research: it includes positions that already reached 0.50
ATR MFE. Its positive returns do not establish a profitable entry strategy.

| Side | N | Frozen model | TRAIN-selected baseline | Model net | Baseline net | Improvement | Capture model/base | Gate |
|---|---:|---|---|---:|---:|---:|---:|---|
| LONG | 16,698 | Logistic, 0.80 | Time exit 10m | +2.378 bps | +1.350 bps | +1.028 bps | 34.0% / 30.1% | FAIL |
| SHORT | 17,021 | HGB, 0.80 | Time exit 5m | +1.776 bps | +1.571 bps | +0.204 bps | 32.2% / 29.6% | FAIL |

LONG is suggestive but insufficient. Its paired net-improvement 95% interval
is approximately +0.12 to +2.31 bps, and 9/11 symbols improved, but:

- the frozen minimum improvement was 2 bps;
- median capture improvement was about 4.0 points, below the frozen 5-point
  hurdle;
- bounded hold produced +2.252 bps in VALIDATION, leaving only about 0.13 bps
  between it and W3B LONG;
- at 20 bps cost the W3B LONG expectancy became negative;
- its FDR-adjusted result did not pass.

SHORT improved only 0.20 bps. Its net and capture confidence intervals crossed
zero and it failed cost and FDR gates.

The W3B classifiers predicted the registered near-term giveback label well
(ROC-AUC 0.945 LONG and 0.952 SHORT), but the target prevalence was 93%-94%.
More importantly, prediction did not translate into a sufficiently superior
exit policy. This is another example of classification quality not being equal
to trading utility.

## Required Questions

### Entry

1. Entering at the impulse remained too late economically.
2. No optimal profitable post-impulse window appeared from 0-10 minutes.
3. Registered pullbacks did not produce positive future MFE/MAE economics.
4. No stable, economically useful healthy-pullback signature was identified.
5. The joint model containing taker reacceleration had no entry ranking power;
   incremental taker value is therefore not established.
6. The joint 1m microstructure features did not create entry edge.
7. Waiting one minute slightly reduced SHORT MAE but increased LONG MAE; it was
   not a stable improvement.
8. Waiting sacrificed or delayed MFE without improving net expectancy.
9. No point in the registered 0-10m window changed the conclusion.
10. No entry policy covered 14 bps, much less 20 bps stress.

### Exit

11. Recovery-related state helped classify giveback, but did not establish a
    profitable HOLD rule.
12. Failure-related state was predictive, but EXIT improvement was too small.
13. W3 distinguished the target statistically better than W2, but not with
    enough economic benefit to call pauses and reversals operably distinct.
14. Taker flow was present in the joint model; standalone incremental value was
    not preregistered and is therefore `INCONCLUSIVE`, not post-hoc optimized.
15. Velocity has the same `INCONCLUSIVE` incremental status.
16. Microstructure has the same `INCONCLUSIVE` incremental status.
17. LONG reduced giveback modestly but missed the frozen net/capture hurdles;
    SHORT did not improve robustly.
18. W3B did not robustly beat simple trailing/hold/time/giveback alternatives.
19. W3B has stronger minute-level classification than W2, but no accepted
    economic improvement over W2's negative conclusion.
20. It did not reach holdout; out-of-sample holdout survival is unproven.

## Statistical Controls

- 10,000 UTC-day clustered episode bootstraps.
- Four frozen temporal validation folds.
- Bayesian bootstrap superiority probabilities.
- Benjamini-Hochberg FDR across W3A/W3B and LONG/SHORT.
- 14 bps base cost and 20/30 bps stress.
- No leverage optimization, reentry, or intraminute fill assumptions.
- Expanding walk-forward was not run because no policy passed the prior gates.

## Conclusion

One-minute aggregation recovered information that predicts a common giveback
label, but it did not recover a defensible trading edge. W3A strongly supports
abstention. W3B LONG is a useful diagnostic lead, not a promotable policy.

W1, W2, and W3 now consistently show that further subdivision of the same
volume-wave family risks data mining. This family should be closed unless a
genuinely new hypothesis is preregistered using new information, such as a
future untouched tick/order-book dataset with executable latency and spread,
not another threshold search over these episodes.

## Artifact Integrity

- Config SHA-256: `517d2e07aea306503ddcf3adc58a8ab29ff06f2af06330268009612a839fe618`
- Dataset manifest SHA-256: `b65fba2209ae7b73ff9d2d00c477f616c3478345e04be6155d4c363388c45889`
- Dataset audit SHA-256: `8c18352f52b721ae2cce3ac43ccf91979179aee0bf87abbdb511bc304e533cd4`
- Evaluation SHA-256: `fcdf2fa6088217e7efc05293f40e1d35ba209301b4f87ea12f2628d3e16863ed`
- Authenticated requests: `0`
- Exchange mutations: `0`
- Production changes: `0`
