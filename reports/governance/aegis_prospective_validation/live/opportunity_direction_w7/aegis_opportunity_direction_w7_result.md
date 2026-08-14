# Aegis W7 Opportunity x Direction - Result

## Verdict

`AEGIS_W7_DECOMPOSITION_NO_ECONOMIC_EDGE`

- W7A_OPPORTUNITY_VALUE_FOUND: `TRUE`
- W7A_META_LABEL_EDGE_FOUND: `FALSE`
- W7B_DIRECTIONAL_ALPHA_FOUND: `FALSE`
- W7B_CROSS_SECTIONAL_ALPHA_FOUND: `FALSE`
- W7_COMBINED_EDGE_FOUND: `FALSE`
- W7_READY_FOR_SHADOW: `FALSE`
- W7_READY_FOR_LIVE: `FALSE`
- FINAL_HOLDOUT_W7: `SEALED_NOT_OPENED`

## Population

- TRAIN: 510 frozen Aegis SHORT signals
- VALIDATION: 358 frozen Aegis SHORT signals
- LONG: NOT_AVAILABLE; the current historical brain was SHORT-only.
- HOLD rows were excluded rather than relabeled as entries.

## Opportunity

- Horizons passing magnitude gates: 2/3
- Selected policy: `H60:LOGISTIC_L2:P_GTE_0.7`
- Validation take/skip: 89/358 taken; 75.14% skipped

| Horizon | Magnitude Spearman | Decisive AUC | Frozen SHORT baseline |
|---:|---:|---:|---:|
| 15m | 0.2367 | 0.6125 | -12.8835 bps |
| 30m | 0.1916 | 0.6054 | -13.9873 bps |
| 60m | 0.2126 | 0.5375 | -13.6491 bps |

The selected opportunities really were larger on average (83.47 vs 77.40 bps), confirming that magnitude and direction are different questions.

## Economics

- Baseline gross/net expectancy: 0.3509 / -13.6491 bps per signal.
- Selected gross/net expectancy: 2.2224 / -11.7776 bps per trade.
- Skipped signals counterfactual expectancy: -14.2683 bps.
- Selected-trade 95% CI: [-24.6372, 0.6162]
- Portfolio improvement: 10.7211 bps/signal, caused mostly by abstaining from a negative baseline.
- Profit factor: 0.5770; maximum additive drawdown: 1216.80 bps; Sortino: -0.2769.
- Stress expectancy at 20/30 bps cost: -17.7776 / -27.7776 bps.
- Positive symbols: 1/11; positive temporal folds: 1/4.

## W7B Data Audit

Funding and mark/spot basis are available, but were already tested without validated edge in M1B. Relative strength and breadth were already negative in B1/B2. Open interest, positioning/crowding ratios and liquidations are absent. W7B was therefore not trained: doing so would reuse the same failed information rather than test genuinely new directional alpha.

## Conclusion

Opportunity magnitude was learnable, but it did not convert the frozen Aegis direction into robust positive net expectancy. The remaining bottleneck is direction, and the local repository lacks a genuinely new complete OI/crowding/liquidation dataset for a defensible W7B test.

No production, TypeScript, guards, leverage, PM2, Shadow or exchange state changed.
