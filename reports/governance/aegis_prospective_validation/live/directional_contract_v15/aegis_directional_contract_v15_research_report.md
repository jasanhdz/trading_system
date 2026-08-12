# Aegis Directional Contract V15 Research Report

## Verdict

`RESEARCH_ONLY_NOT_PROMOTABLE`

V15 tested simpler direction-specific feature contracts against the unchanged
176-position V9 control. It exported no model and changed neither Shadow nor
Live. Exchange calls and mutations were zero.

## Design

- Evidence: 96,382 directional rows / 48,191 paired episodes.
- Period: 2025-08-09 through 2026-08-09.
- Control: all 176 V9 feature positions.
- LONG candidate: 129 positions after exact deduplication and removal of the
  V14-flagged `LOCAL_MOMENTUM_TREND` and `ROLLING_CONTEXT` families.
- SHORT candidate: 168 positions after exact deduplication only.
- Models per side and contract: danger-first classifier, clean-entry classifier,
  and q90 MAE estimator.
- Policy score: clean probability minus danger probability minus normalized
  q90 MAE.
- Validation: four purged expanding train/calibration/test folds. Thresholds
  were selected on calibration only, with at most one symbol per timestamp.
- Fold 4 covers 2026-07-17 through 2026-08-09 and is new relative to V14.

## Fold Stability

| Side | Danger | Clean | q90 MAE | Mean utility | CVaR | Adverse-first | Selected MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LONG | 2/4 | 0/4 | 3/4 | 3/4 | 3/4 | 2/4 | 3/4 |
| SHORT | 2/4 | 1/4 | 0/4 | 0/4 | 4/4 | 4/4 | 3/4 |

No side satisfied the preregistered requirement of improvement in at least
three folds on every metric plus no regression on the post-V14 holdout.

## Post-V14 Holdout

### LONG

The simpler LONG contract selected 205 entries versus 207 for the control.
Mean utility improved from -0.1964% to -0.1897%, CVaR improved from -0.5797% to
negative 0.5639%, and mean MAE improved from 0.2474% to 0.2370%. These are directionally
useful changes, but expected utility remained negative. Danger log loss and q90
MAE pinball loss regressed, and clean-entry probability did not improve on both
required metrics.

### SHORT

The deduplicated SHORT contract selected the same 223 entries as the control.
Mean utility changed from -0.2162% to -0.2170% and mean MAE worsened from
0.2808% to 0.2847%. Predictive changes were negligible and did not satisfy the
joint metric gates. Deduplication reduced representation size but did not
improve decisions.

## Interpretation

V15 confirms three points:

1. Exact duplicate removal is structurally sound but does not create predictive
   edge by itself.
2. LONG benefits modestly from removing unstable families, particularly in
   realized path quality, but the remaining ranking still selects negative
   expected utility.
3. The principal bottleneck is now target and ranking design rather than adding
   or deleting conventional features.

The clean-entry label is very sparse among selected candidates, and a score
assembled from independently trained clean, danger, and MAE models does not
learn realized economic ordering reliably. Further feature manipulation against
the same target is unlikely to solve this.

## Recommended Next Experiment

V16 should be preregistered as a target-and-ranking study, not another feature
expansion:

1. retain the V15 deduplicated contracts as controls;
2. train a direct ordinal or pairwise ranker for realized utility after costs;
3. represent adverse-first, time-underwater, and time-to-protectable advantage
   jointly rather than combining independent probabilities after training;
4. use regime and symbol only for hierarchical calibration, not ad hoc
   thresholds;
5. require positive utility and improved CVaR on a future untouched holdout;
6. add no runtime behavior until separately authorized Shadow evidence exists.

V15 does not authorize Shadow or Live deployment.
