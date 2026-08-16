# AEGIS W15 Result

## Verdict

`AEGIS_W15_NO_RECENT_SHORT_SIGNAL_EDGE`

- `W15_RECENT_SHORT_SIGNAL_EDGE_FOUND = FALSE`
- `W15_MODELING_JUSTIFIED = FALSE`
- `W15_READY_FOR_SHADOW = FALSE`
- `W15_READY_FOR_LIVE = FALSE`

## Actual recent Aegis SHORT performance

The prospective journal contains 1,634 completed `ENTER_NOW` SHORT episodes.
The recorded cost is 10 bps per round trip.

| Period | Episodes | Gross bps/trade | Net bps/trade |
| --- | ---: | ---: | ---: |
| Complete journal | 1,634 | +2.23 | -7.77 |
| Last 7 days | 471 | +4.47 | -5.53 |
| Last 3 days | 218 | +3.90 | -6.10 |

The recent market was somewhat more favorable to SHORT than the complete
journal, but average gross movement still did not pay the recorded 10 bps cost.

## Richer TAKE/SKIP challenger

W15 used only actual Aegis SHORT entries. Causal inputs included existing
direction/quality/tail-risk probabilities, expected return, QMAE, regime,
component scores, symbol, and time. Calibration froze shallow HGB regression at
its 80th-percentile score threshold.

| Metric | Result |
| --- | ---: |
| TRAIN signals | 1,106 |
| VALIDATION signals | 286 |
| Selected trades | 24 (8.39%) |
| Gross expectancy | -17.32 bps/trade |
| Net expectancy, recorded 10 bps | -27.32 bps/trade |
| Net expectancy, 14 bps | -31.32 bps/trade |
| Net expectancy, 20 bps | -37.32 bps/trade |
| Profit factor at 14 bps | 0.35 |
| Positive symbols | 2 of 11 |
| Bootstrap 95% CI at 14 bps | [-73.35, +24.84] bps |

The ungated current policy over the same validation interval produced -1.20
bps/trade at its recorded 10 bps cost. W15 therefore made selection materially
worse despite operating much less.

## Interpretation

The richer model fitted calibration-specific relationships that reversed in
the next period. Increasing capacity or allowing the model to memorize the
past does not solve the current directional/economic problem. Selecting only
ADA or SOL after observing their isolated winners would be post-result symbol
selection and is prohibited.

No threshold retuning, production change, or live activation is justified. A
genuinely new model would require new information or a future untouched period,
not another search over the same recent outcomes.
