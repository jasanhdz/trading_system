# Aegis SHORT Profitability Semantics V1 Validation

## Scope

This report closes the semantic ambiguity around `short_prob`.

- `short_prob` remains a directional side-authority output.
- It is not interpreted as probability of winning or probability of profit.
- `terminal_net_positive_h12_after_costs` is a separate calibrated research
  estimate for `P(net SHORT return at H12 > 0 after frozen costs)`.
- `clean_entry_low_mae_h12` remains a separate path-quality estimate.
- The experiment is Shadow-only and has no selection or exchange authority.

## Historical Validation

- Dataset rows: 172,480.
- Feature schema: `aegis-features-v2` (83 features).
- Validation: four purged temporal folds.
- Candidate families: logistic regression, random forest and histogram
  gradient boosting.
- Frozen round-trip cost fraction: 0.001.
- Winner by the preregistered ordering:
  `eqm_random_forest_clean|MODEL_ONLY`.
- Scoring rows in the final artifact: 11,847.
- Positive terminal-net label prevalence: 0.440956.
- Average precision: 0.456468.
- Expected calibration error: 0.046454.
- Brier score: 0.248136.

The winning selection did not establish economic edge:

- Mean selected expectancy: -0.000640.
- Worst-fold selected expectancy: -0.001373.
- Positive folds: 1 of 4.
- Passing folds: 0 of 4.

The offline verdict is `FAILED`; promotion readiness is false.

## Runtime Disposition

The exact artifact may run only as a prospective Shadow observer. It records,
for every configured symbol:

1. SHORT side-authority probability;
2. terminal net-positive probability at H12 after costs;
3. clean-entry/low-MAE probability at H12;
4. the realized terminal return, MFE and MAE after maturity.

The observer:

- does not return or override `selected`;
- does not change `would_execute`;
- does not alter committee decisions;
- does not alter guards, capital, sizing or leverage;
- has no Binance client or exchange authority;
- cannot promote itself automatically.
- deduplicates repeated HTTP evaluations by canonical market timestamp, so
  twelve API calls cannot be mistaken for twelve completed 5-minute bars.

## Superseded Initial Runtime Evidence

The initial `signals.jsonl` and `outcomes.jsonl` journals counted distinct HTTP
decision cycles within the same market bar. They are preserved for audit but
are invalid for scientific evaluation and MUST NOT be used. The corrected
authority starts with `signals_v2.jsonl` and `outcomes_v2.jsonl`, which
deduplicate by canonical market timestamp.

## Decision

Use the observer to collect calibrated prospective evidence and diagnose
whether terminal profitability adds information beyond direction and clean
path quality. Do not use this artifact to authorize or block Live entries
unless a future preregistered evaluation passes and the owner separately
authorizes promotion.
