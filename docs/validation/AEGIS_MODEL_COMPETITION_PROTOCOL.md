# Aegis Model Competition Protocol V1

This protocol was frozen before running the Phase C model competition. The machine-readable
source is `config/scientific_competition_v1.yaml`.

## Scope

- Side: SHORT only. LONG cannot contribute to any promotion metric.
- Data: hash-pinned finalized D3 canonical series, read-only.
- Features/labels: `aegis-features-v2` and `aegis-labels-short-v4`.
- Four expanding temporal folds with a 120-minute embargo.
- Final test is a lockbox and cannot fit normalizers, calibrators, thresholds, or models.

## Selection

TRRM compares a linear logistic baseline, Random Forest, and HistGradientBoosting. QMAE
compares unconditional quantiles with HGB pinball q50/q90 plus one-sided split conformal.
EQM keeps `clean_entry` classification separate from `net_quality_after_costs` regression.

Every probabilistic candidate compares raw, Platt, and isotonic calibration fitted outside
the scored fold. Mean ECE is primary and mean Brier is the tie-breaker. Model ranking is
worst fold, mean, standard deviation, then compute cost. Top-decile symbol concentration
cannot exceed 30%.

H1 requires TRRM lift of at least 1.10 over prevalence in every fold. H2 requires the worst
fold to retain at least 75% of mean quality. H3 requires positive incremental expectancy in
the independent ECON replay. QMAE requires every fold to cover in `[0.87, 0.93]` and pinball
loss no greater than 90% of its unconditional baseline.

Smoke runs validate infrastructure only. They cannot select a winner, derive a productive
threshold, create a CANDIDATE bundle, or count as parity evidence. Published artifacts are
inspectable JSON evaluated by Aegis code; pickle is prohibited.
