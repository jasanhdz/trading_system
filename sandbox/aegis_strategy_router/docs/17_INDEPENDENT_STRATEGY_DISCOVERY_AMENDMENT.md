# Independent Strategy Discovery Amendment

Recorded: `2026-08-18` UTC

## Governing scope

- `INITIAL_EXPERIMENT_MODE = INDEPENDENT_STRATEGY_DISCOVERY`
- `AEGIS_INCLUDED_IN_INITIAL_EXPERIMENT = FALSE`

This amendment overrides the signal-conditioned initial-scope language in
Phase 0 and the design-review documents. It does not alter the accepted Phase
1 snapshot infrastructure, the Phase 2 rule freeze, candidate substates, data
quality requirements, fresh partitions, sample minimums, or sealed holdouts.

Aegis is excluded from initial discovery, candidate generation, proposal side,
training, calibration, specialist validation, and initial routing. Historical
or prospective Aegis signals cannot select anchors, symbols, sides, thresholds,
models, or candidate populations in this experiment.

Combining a frozen independent system with Aegis is a later, separately
preregistered transfer experiment. It is permitted only if the independent
system first demonstrates edge under its own fresh validation and holdout.

## General-market anchor contract

1. Generate one side-neutral market snapshot per symbol at every fully closed,
   UTC-aligned 15-minute boundary inside the authorized fresh partition.
2. The reference price is the close of the final fully closed one-minute bar
   at that boundary. No open or later bar may contribute.
3. The snapshot has `proposed_side = None` and `signal_id = None`.
4. Evaluate the same immutable snapshot once for LONG and once for SHORT using
   identical frozen rules and separate replay state.
5. Failure, invalidation, or ineligibility of one side supplies no positive
   evidence to the other side. Both can be `UNKNOWN`/`INELIGIBLE`; that is the
   valid `NONE` outcome.
6. Candidate identity remains content-addressed by snapshot, strategy, side,
   decision time, and generator version. Related substates retain their frozen
   `setup_episode_id`.

The 15-minute cadence is methodological, not performance-selected. It matches
the frozen operational setup and breakout timeframe, prevents evaluation on
every one-minute row, and remains identical for every symbol and side.

## Overlap control

All raw label-free evaluations are retained. Effective independent candidate
support is counted conservatively:

- only `CANDIDATE` or `ENTERABLE` dispositions can enter the independent
  candidate population;
- repeated substates with one `setup_episode_id` count once;
- within each `(strategy, symbol, side)`, retain the earliest candidate and
  suppress later candidates whose decision time is less than 60 minutes after
  the retained episode;
- ties use `(decision_at, candidate_episode_id)` only;
- suppressed evaluations remain auditable and never become negative labels.

The 60-minute embargo is the already frozen common candidate horizon. It was
not selected from event rates or outcomes.

## Permitted Phase 2 reporting

Only snapshot coverage, source gaps, status/substate counts, candidate event
rate, side/symbol distribution, overlap suppression, and effective independent
episode count may be reported. PnL, win rate, future MFE/MAE, barriers, labels,
edge, and strategy ranking remain prohibited.

## Governance

- `W1_W14_HOLDOUTS = SEALED`
- `EDGE_VALIDATION_PERFORMED = FALSE`
- `READY_TO_IMPLEMENT_SPECIALISTS = FALSE`
- `READY_FOR_SHADOW = FALSE`
- `READY_FOR_LIVE = FALSE`

