# Aegis Feature Information V14 Research Plan

## Objective

V14 determines whether the information feeding V13 can distinguish clean entry
paths from adverse-first paths. It audits all 176 V9 research features, tests
existing families by removal, and evaluates one genuinely new causal family:
historical taker buy/sell imbalance from closed candles.

## Data Authority

The immutable V11 episodes and outcomes remain the target authority. The
existing candle database is opened read-only. Taker-flow features use only the
24 closed bars preceding next-bar-open entry and require a complete synchronized
eleven-symbol timestamp. Rows without this evidence are excluded, never filled.

Funding and open-interest columns exist but contain zero rows. Order-book and
liquidation histories are absent. They are documented as evidence gaps and are
not modeled or reconstructed.

## Evaluation

For LONG and SHORT independently, four purged expanding walk-forward folds
measure danger-first classification, clean-entry classification and q90 MAE.
The 176-feature baseline is compared with:

1. each existing family alone, report-only;
2. baseline minus each existing family;
3. baseline plus the ten taker-flow candidates.

A new family is admissible only if danger log loss and average precision, clean
log loss, and MAE pinball loss improve in at least three folds without test
threshold tuning. Aggregate improvement cannot hide temporally unstable folds.

## Safety

V14 has no selection authority. It cannot change the feature contract, export a
model, activate Shadow/Live, access private exchange endpoints or mutate the
exchange. Manual pyramiding and discretionary closes remain excluded.

## Completion

The preregistered experiment completed with verdict
`RESEARCH_ONLY_NOT_PROMOTABLE`. The taker-flow family failed the stability gate
and no model or runtime change was authorized. See
`aegis_feature_information_v14_research_report.md` for the results.
