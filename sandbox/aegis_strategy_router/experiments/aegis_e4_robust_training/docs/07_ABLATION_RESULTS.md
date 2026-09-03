# E4 Ablation Results

Validation favorable-first results:

| Ablation | Features | AUC | Brier |
|---|---:|---:|---:|
| E4_BASE | 88 | 0.5098 | 0.249871 |
| + FLOW | 104 | 0.5098 | 0.249870 |
| + CROSS_MARKET | 102 | 0.5127 | 0.249823 |
| + REMAINING_MOVE | 104 | 0.5114 | 0.249855 |
| E4_FULL | 146 | 0.5141 | 0.249803 |

No family meets the preregistered incremental threshold of +0.005 AUC plus a
Brier improvement over BASE. Cross-market is the largest incremental change,
but remains below that gate.

Other heads contain real information:

- Tail-risk AUC: 0.7319.
- Late-entry-risk AUC: 0.8294.
- Entry-quality AUC: 0.5198.
- MFE MAE improves over the constant baseline (44.05 vs 50.60 bps).
- MAE MAE improves similarly (44.03 vs 50.60 bps).
- Expected return and event-time regressions do not beat their constant
  baselines.

This means E4 recognizes volatility/tail and late-state risk better than it
recognizes which side will be favorable.
