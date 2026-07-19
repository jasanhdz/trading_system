# E3 formal closure

## Final disposition

E3 is permanently closed as `E3_REJECTED_PRE_LOCKBOX` under direct owner
authorization and the binding independent scientific audit. The validation was
technically valid and deterministic; the rejection is predictive and economic,
not technical.

## What was validated

Two independent pre-lockbox validation runs used the frozen E3 preregistration,
competition v2, canonical dev data, fixed seeds, literal folds, hourly sampling,
full historical capacities, rank-based TRRM veto, EQM survivor populations,
calibration, conformal QMAE, frozen baselines, and the authoritative runtime.
Their eight scientific artifacts were byte-identical.

The dataset contained 15,680 coordinated cycles and 172,480 rows with no skipped
history cycles or quarantined labels. Calibration passed its ECE limit and QMAE
coverage remained inside the frozen band in all four folds.

## Why E3 is rejected

The B_BASE result contained 1,292 trades, profit factor `0.5370443650824593`,
and expectancy `-0.0015341790376828055`. Zero of four folds had positive
expectancy. The optimistic scenario also remained negative at
`-0.0010841790376828054`. The no-trade directional baseline, with expectancy
`0.0`, dominated the full system.

The frozen positive-fold, profit-factor, net-expectancy, worst-fold,
directional-baseline, model-beaten, and robust-ECON checks failed. E3 therefore
does not continue and must not be repeated.

## Model gate terminology

`model_not_beaten=true` means that at least one complex model did not beat its
corresponding baseline. The promotion field is its logical inverse:
`models_beaten = not model_not_beaten`. Consequently `models_beaten` failed.
The naming is confusing, but the fields are logically consistent and do not
change the disposition.

## Maximum drawdown semantics

Max DD is the drawdown of the non-compounded cumulative sum of
`net_return_fraction` using fixed notional per trade. It is not a direct
percentage drawdown of a compounded account. B_BASE Max DD of approximately
`2.008` represents cumulative drawdown of about two times the fixed trade
notional. This clarification reinforces the rejection and does not alter any
metric.

## Lockbox and publication

The semi-blind lockbox remains `NOT_CONSUMED`, with `consumed_queries=[]` and
one query remaining. No lease or semi-blind query is justified because dev
evidence already fails the mandatory criteria.

No Candidate, Selection Policy, System Freeze, shadow authorization, paper
authorization, or live authorization exists for E3. The experimental bundle
remains unapproved and is retained only as validation evidence.

## Permanence

The frozen E3 preregistration and all scientific artifacts remain unchanged.
This disposition is the authoritative E3 status because the repository has no
central experiment-disposition registry. Any future E4 would be a separate new
hypothesis; this closure does not define or propose it.
