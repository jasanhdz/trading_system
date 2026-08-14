# Aegis Adaptive Profit Guard W6 - Result

## Verdict

`AEGIS_ADAPTIVE_PROFIT_GUARD_W6_NO_ECONOMIC_EDGE`

- W6_ADAPTIVE_GUARD_EDGE_FOUND: `FALSE`
- W6_MODELING_JUSTIFIED: `FALSE`
- W6_READY_FOR_SHADOW: `FALSE`
- W6_READY_FOR_LIVE: `FALSE`
- FINAL_HOLDOUT_W6: `SEALED_NOT_OPENED`

## Population

- TRAIN episodes: 5459
- VALIDATION episodes: 7364
- Activated in TRAIN: 3790
- Activated in VALIDATION: 4679
- Symbols: 11
- Source accessed: W2 TRAIN only; prohibited partitions accessed: none.

## Primary Comparison

- Selected model: `MULTINOMIAL_LOGISTIC_L2`
- Validation states: NORMAL=5160, EXPANSION=1120, DEFENSIVE=1084
- CURRENT_GUARD expectancy: -11.3717 bps/episode
- Adaptive expectancy: -11.3287 bps/episode
- Improvement: 0.0430 bps/episode
- Paired 95% CI: [-0.3781, 0.5198]
- Side improvement: LONG=0.2816 bps, SHORT=-0.1971 bps
- Positive symbols: 4/11
- Positive temporal folds: 2/4
- Failed gates: minimum_effect, paired_ci_positive, positive_symbols, positive_folds, stress_cost_positive

## Profit And Risk

| Metric | CURRENT_GUARD | Adaptive |
|---|---:|---:|
| Profit factor | 0.6724 | 0.6751 |
| Median capture ratio | 0.5376 | 0.5382 |
| Median giveback | 32.3726 bps | 30.8719 bps |
| P95 giveback | 219.8723 bps | 220.2323 bps |
| Median early-exit regret | 111.4825 bps | 111.4526 bps |
| Median hold-too-long regret | 32.3726 bps | 30.8719 bps |

## Interpretation

Wave/regime state did not improve the frozen current guard by the preregistered economic margin with stable risk. The holdout remained sealed and no Shadow or production change is justified. The classifier assigned different states, but the economic effect was effectively zero and unstable across sides, symbols, and time.

This is a conservative closed-5m reconstruction. Production evaluates mark price more frequently, so even a positive result would require a new independent holdout and Shadow parity before any runtime use.
