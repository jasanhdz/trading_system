# Aegis Safety-Gated Economic Ranker V17 Research Plan

## Status and Scope

V17 is a preregistered, offline-only experiment. It does not modify Shadow,
Live, PM2, exchange configuration, capital management, guards, models, or
runtime decisions. Model export and automatic promotion are prohibited.

The evidence is retrospective after V16. No result from this experiment can
authorize Shadow or Live because the canonical dataset contains no untouched
post-V16 holdout.

## Motivation

V16 separated relative opportunity ranking from V15's safety composite. Its
pairwise ordering exceeded random in every fold and selected more positive
outcomes, but worsened MAE, adverse-first rate, and tail utility. The result
supports a narrower hypothesis: ranking may add value only after an explicit
safety layer has removed candidates with dangerous expected paths.

## Frozen Candidate

V17 uses one independent pipeline per direction and the unchanged V15 feature
contracts: 129 LONG features and 168 SHORT features.

1. V15 clean, danger, and q90-MAE heads are fitted on the training partition.
2. The first chronological half of calibration selects a feasibility gate from
   a fixed 27-policy grid.
3. The gate requires minimum clean probability, maximum danger probability,
   and maximum predicted q90 MAE simultaneously.
4. The V16 pairwise ranker orders only candidates that survive the gate.
5. The second chronological half of calibration selects the ranking threshold.
6. At most one candidate per timestamp is selected.

The two calibration roles cannot exchange outcomes. Test data cannot train a
model or choose a threshold.

## Controls

V17 is compared against both existing approaches on the same test partitions:

- V15 independent-head safety composite;
- V16 direct pairwise ranker without a feasibility gate.

This prevents a favorable comparison caused by choosing only the weaker
control.

## Gate Calibration

The frozen grid combines clean-probability quantiles 0, 0.25, and 0.50 with
danger and MAE quantiles 0.25, 0.50, and 0.75. A policy is eligible only when
it retains at least 100 rows and 10% of gate-calibration candidates.

Eligible policies are selected lexicographically by lower adverse-first rate,
lower mean MAE, higher mean utility, and higher survivor rate. This explicitly
assigns safety to the gate rather than the ranker.

## Ranking Calibration

Among gate survivors in the second calibration half, V16 score quantiles 0.80,
0.90, and 0.95 are evaluated. Policies must select at least 20 candidates.
Selection favors mean utility, then CVaR, then lower MAE.

## Success Criteria

For each direction, at least three of four test folds must have positive mean
utility and improve over both V15 and V16 in mean utility, CVaR, positive rate,
adverse-first rate, mean MAE, and clean rate. Coverage must remain viable:
selection count must be at least half of V16 and the p95 opportunity gap may
not exceed twice V16's.

Even if every retrospective criterion passes, the result remains research
only. A genuinely new post-V16 temporal holdout is mandatory before a separate
Shadow proposal.

## Safety

- Exchange calls: 0
- Exchange mutations: 0
- Runtime model export: prohibited
- Shadow changes: prohibited
- Live changes: prohibited
- Automatic promotion: prohibited
