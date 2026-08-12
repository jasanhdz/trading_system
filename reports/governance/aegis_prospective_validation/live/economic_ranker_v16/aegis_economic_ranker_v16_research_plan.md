# Aegis Economic Ranker V16 Research Plan

## Objective

V16 tests the hypothesis produced by V15: the bottleneck is target and ranking
design, not another feature expansion. It compares V15's three independent
heads with a direct pairwise ranker trained to order candidates that existed at
the same market timestamp.

## Pairwise Target

The candidate uses the unchanged V15 directional feature contracts. Every
within-timestamp pair is ordered by canonical evidence:

1. clean, positive, non-adverse trajectories;
2. positive, non-adverse trajectories;
3. other non-adverse trajectories;
4. adverse-first or same-bar-ambiguous trajectories.

Within the same tier, higher frozen `ROE_10_H12` realized utility is preferred,
then lower MAE and less time underwater. Both pair orientations are included.
No pair crosses timestamps, so the model learns opportunity ranking rather than
predicting one market period from another.

## Comparison

The control reproduces V15's clean-minus-danger-minus-q90-MAE policy on the same
LONG and SHORT contracts. The candidate uses the pairwise decision score.
Thresholds are selected only on calibration data. Test metrics include pairwise
accuracy, mean utility after frozen costs, CVaR, positive rate, adverse-first
rate, clean rate, MAE, and opportunity frequency.

Regime and symbol metrics are report-only. Subgroup thresholds are prohibited
until the global ranker proves skill, preventing sparse groups from creating ad
hoc policies.

## Evidence Limitation

All currently available data was already visible when V15 concluded. V16 is
therefore retrospective hypothesis testing. Even a successful result requires a
future untouched holdout and separate Shadow authorization; this experiment
cannot promote itself.

## Safety

V16 is offline and read-only. It exports no model, changes no feature contract,
does not modify Shadow or Live, and has no network or exchange authority.
