# Joint Path V12 Validation Summary

## Verdict

`RESEARCH_ONLY_NOT_PROMOTABLE`

V12 proves that a joint direction/path state is learnable and well calibrated,
but its ranking does not produce stable economic or MAE improvement across
untouched periods. No model was exported and neither Shadow nor Live changed.

## Frozen Evidence

- Source: the immutable V11 dataset and all V10 barrier outcomes.
- Evidence interval: 2025-08-09 through 2026-08-09.
- Independent episodes: 48,191.
- Sides: LONG and SHORT.
- Symbols: all 11 canonical symbols.
- Contracts: 5/10/20% ROE over 30/60/120 minutes.
- Severe round-trip cost: 0.20% of price.
- Test method: four purged expanding walk-forward folds with separate model,
  probability-calibration, contract-assignment, threshold-policy and test
  windows.

## Component Result

| Side | Joint-state skilled folds | Economic folds | Required |
|---|---:|---:|---:|
| LONG | 4/4 | 0/4 | 3/4 |
| SHORT | 4/4 | 0/4 | 3/4 |

Every joint estimator beats its training prior log loss, matches or exceeds
majority accuracy and has multiclass ECE below 0.10. The model therefore learns
the registered joint states. It does not, however, rank enough positive-utility
entries to form an eligible policy of at least 30 candidates.

## Untouched-Test Ranking Diagnostic

The following top-30 diagnostic has no selection authority. It answers whether
the model's highest rankings improve path quality even when no policy passes.

| Side | Fold | Mean net utility | Clean | Adverse first | Mean MAE | Control MAE |
|---|---:|---:|---:|---:|---:|---:|
| LONG | 1 | -0.0014% | 56.7% | 20.0% | 0.596% | 0.781% |
| LONG | 2 | -0.4287% | 10.0% | 33.3% | 0.745% | 0.587% |
| LONG | 3 | -0.3040% | 33.3% | 20.0% | 1.003% | 0.543% |
| LONG | 4 | -0.3111% | 23.3% | 63.3% | 1.446% | 0.792% |
| SHORT | 1 | -0.9734% | 3.3% | 60.0% | 1.616% | 0.809% |
| SHORT | 2 | -0.2310% | 6.7% | 26.7% | 0.827% | 0.610% |
| SHORT | 3 | -0.4556% | 16.7% | 66.7% | 1.472% | 0.483% |
| SHORT | 4 | 0.1292% | 33.3% | 26.7% | 1.201% | 0.723% |

LONG fold 1 demonstrates the desired behavior: cleaner paths, fewer
adverse-first outcomes and lower MAE. SHORT fold 4 has positive net utility,
but does not improve MAE. These effects do not persist across time and cannot
support promotion.

## What Changed Relative To V11

1. Direction, cleanliness and barrier outcome are represented by one
   multiclass target instead of intersecting independently trained heads.
2. One estimator per side shares all barrier/horizon evidence.
3. Contract choice is frozen by side and causal regime in an assignment-only
   policy window; V12 never maximizes nine utilities per live candidate.
4. Thresholds are learned in a later policy-only window.
5. MAE, time to positive and adverse-first rates are first-class gates.
6. The existing 40% ROI stop remains catastrophe context and is not changed.

## Conclusion And Next Experiment

Post-hoc intersection was not the only cause of V11 failure. V12's joint model
is statistically valid, but the relationship between features and clean net
paths changes materially between periods. The next experiment should be V13,
a preregistered temporal-stability challenger:

- predict whether the current feature-to-outcome relationship is in-distribution;
- train regime-conditional experts only where each regime has adequate temporal
  support;
- require agreement between a long-history model and a recent-window model;
- abstain on disagreement rather than relaxing utility or quality thresholds;
- compare directly with V12 using unchanged labels, costs, folds and minimum
  candidate counts;
- retain LONG/SHORT separation because their unstable periods differ.

An exit-policy optimization remains premature. Entry ranking must first show
stable net utility and MAE improvement. Manual pyramiding or discretionary
closures must be evaluated as separate interventions and must not contaminate
the autonomous policy labels.

## Safety

- Runtime effect: `NONE`.
- Model exported: `false`.
- Shadow activated: `false`.
- Live activated: `false`.
- Exchange calls: `0`.
- Exchange mutations: `0`.

## Artifact Hashes

- Configuration: `4cbcdab4cf69154851e8bc3585234128e4cead95e006fa9afdf2561fe2298403`.
- Source dataset: `2c270a7b38b2f05c2b4ab78960b788c387e5d98bfae29da16a167f4f54548747`.
- Validation: `964db032993681eea4386c2416e346fa37128293f059d3d6637474c16cd3df62`.
