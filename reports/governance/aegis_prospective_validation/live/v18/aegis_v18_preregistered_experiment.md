# Aegis V18 Preregistered Experiment

## Purpose

V18 is a new research experiment, not a repair or retuning of V17. V17 remains
frozen as negative evidence. V18 tests one explicit hypothesis: entry selection
may improve when safety probability, adverse-excursion magnitude, and expected
economic utility are estimated separately and combined only by a frozen policy.

The hypothesis is not assumed true. Failure is a valid result.

## Data Separation

- TRAIN: 2025-08-09 06:55 UTC through 2026-05-31 23:55 UTC.
- VALIDATION: 2026-06-01 00:00 UTC through 2026-08-09 06:55 UTC.
- FINAL HOLDOUT: 2026-08-10 00:00 UTC through 2026-09-30 23:55 UTC.

The final holdout begins after every row used by V17 and is unavailable at
registration time. It must remain sealed until 2026-10-01 UTC. It may be opened
once, and only if the frozen validation gate passes without changing features,
models, seeds, calibration, or thresholds.

## Frozen Candidate

LONG and SHORT are evaluated independently. Each direction uses its already
causal, runtime-reproducible directional feature contract. Four separate heads
answer four different questions:

1. probability of a clean path;
2. probability of adverse-first or ambiguous behavior;
3. conditional q90 MAE magnitude;
4. expected utility after frozen costs.

A candidate survives only if all four preregistered absolute conditions pass.
Among survivors, at most one symbol per timestamp and direction is selected by
predicted utility. No validation quantile or favorable seed may be selected.

## Controls

V18 is compared on identical timestamps with frozen V15, frozen V17, a simple
causal V15 score, and a deterministic random control. Better performance than
V17 alone is insufficient.

## Economic Gate

The primary outcome is net expectancy after the dataset's frozen costs. The
gate also requires uncertainty bounds, profit factor, CVaR, MAE, drawdown,
frequency, and temporal stability. LONG and SHORT must each pass. Accuracy is
diagnostic, not the promotion objective.

## Anti-Overfitting Rules

There is one seed and no hyperparameter search. Feature mining, threshold
mining, selective fold reporting, repeated holdout access, and deleting failed
runs are prohibited. Validation is not training data. Final-holdout outcomes
cannot choose a model or policy.

## Gates

`V18_READY_FOR_SHADOW` requires the complete offline gate, including the final
holdout when it becomes mature. `V18_READY_FOR_LIVE` is a separate future gate
that additionally requires at least 300 non-overlapping new Shadow
opportunities per direction over at least 30 days with frozen policy and no
integrity incident.

At preregistration both gates are false. V18 has no exchange authority and does
not modify Live or Shadow.
