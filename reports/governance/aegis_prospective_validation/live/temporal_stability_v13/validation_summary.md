# Temporal Stability V13 Validation Summary

## Verdict

`RESEARCH_ONLY_NOT_PROMOTABLE`

V13 implements historical/recent consensus, supported causal-regime experts,
an out-of-distribution gate and conservative q90 MAE prediction. Both temporal
models are statistically skilled in every fold, but their agreement does not
create positive economic utility or reduce MAE consistently. No model was
exported and no runtime changed.

## Evidence And Method

- Source: immutable V11 episodes and V12 joint-state definitions.
- Evidence interval: 2025-08-09 through 2026-08-09.
- Independent episodes: 48,191.
- Symbols: all 11 canonical symbols.
- Sides: LONG and SHORT.
- Historical model: expanding training window.
- Recent model: trailing 120-day window, with a preregistered minimum sample.
- Regime experts: trained only above fixed train/calibration support minima.
- Risk estimate: maximum of historical and recent q90 MAE predictions.
- Validation: four purged expanding walk-forward folds with five separated
  temporal windows and no test tuning.

## Model Skill And Economics

| Side | Historical skilled | Recent skilled | Economic folds | Required |
|---|---:|---:|---:|---:|
| LONG | 4/4 | 4/4 | 0/4 | 3/4 |
| SHORT | 4/4 | 4/4 | 0/4 | 3/4 |

Both temporal models beat their training-prior log loss, preserve majority
accuracy and remain below the registered ECE ceiling. Nevertheless, every
policy window contains zero positive predicted-utility candidates after the
frozen severe cost and joint-state penalties. No threshold policy is eligible.

## Temporal Gate Behavior

Historical/recent/regime consensus rates remain between 85.2% and 99.9% on
untouched test windows. In-distribution rates remain between 85.7% and 97.9%.
This level of agreement is not selective enough to identify economic edge:
historical and recent models learn similar dominant states from the same causal
feature family, even when their training windows differ.

## Top-30 Report-Only Diagnostic

| Side | Fold | Net utility | Clean | Adverse first | Mean MAE | Control MAE |
|---|---:|---:|---:|---:|---:|---:|
| LONG | 1 | -0.1236% | 26.7% | 33.3% | 0.879% | 0.781% |
| LONG | 2 | -0.4170% | 16.7% | 30.0% | 0.811% | 0.587% |
| LONG | 3 | -0.3121% | 20.0% | 13.3% | 0.904% | 0.543% |
| LONG | 4 | -0.1477% | 30.0% | 36.7% | 1.155% | 0.792% |
| SHORT | 1 | -0.1487% | 6.7% | 30.0% | 0.857% | 0.809% |
| SHORT | 2 | -0.5150% | 6.7% | 66.7% | 1.424% | 0.610% |
| SHORT | 3 | -0.1653% | 13.3% | 36.7% | 0.784% | 0.483% |
| SHORT | 4 | -0.2807% | 16.7% | 33.3% | 1.092% | 0.722% |

The top rankings have negative net utility in all eight side-folds and higher
mean MAE than unfiltered control in all eight. Agreement therefore filters
model instability but cannot manufacture information absent from the inputs.

## What V13 Establishes

1. Temporal calibration and dual-window inference are reproducible.
2. Conservative q90 MAE prediction can be integrated without leakage.
3. Regime experts can be trained for three common low-volatility regimes and,
   in the last fold, RANGE_LOW_VOL. High-volatility regimes lack the registered
   support and correctly receive no expert.
4. The current causal feature family makes historical and recent predictions
   too similar for agreement to be a useful edge discriminator.
5. Architecture and stricter voting are no longer the primary bottleneck.

## Recommended Next Work

Do not create V14 by adding another voter. First perform a preregistered feature
information audit against the same outcomes:

- measure incremental information from market-wide BTC context, cross-symbol
  breadth, realized-volatility acceleration, volume/order-flow imbalance,
  open-interest change and funding/basis when causally available;
- verify timestamp alignment, publication delay and missingness before modeling;
- test each feature family alone and incrementally against the frozen V13 base;
- require stable mutual information or out-of-sample log-loss improvement in at
  least three folds before admitting a feature family;
- rebuild causal regimes from validated features rather than fixed return and
  volatility thresholds;
- only then repeat the joint path and MAE experiment.

Exit optimization remains deferred because entry rankings still lack stable
economic value. Manual pyramiding and manual closes remain separate causal
interventions and are not used as autonomous labels.

## Safety

- Runtime effect: `NONE`.
- Model exported: `false`.
- Shadow activated: `false`.
- Live activated: `false`.
- Exchange calls: `0`.
- Exchange mutations: `0`.

## Artifact Hashes

- Configuration: `2130c2bedeee647ac61954cab51d4839bab82366ac55157e96de411b7f2ca2be`.
- Source dataset: `2c270a7b38b2f05c2b4ab78960b788c387e5d98bfae29da16a167f4f54548747`.
- V12 validation: `964db032993681eea4386c2416e346fa37128293f059d3d6637474c16cd3df62`.
- V13 validation: `748b3826acb6bf81f128060089e03c2f05a68e83b50edbe5308c9401eab666ca`.
