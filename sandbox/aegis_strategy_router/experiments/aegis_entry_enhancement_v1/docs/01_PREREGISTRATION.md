# Aegis Entry Enhancement V1 Preregistration

## Frozen question

Given an immutable historical Aegis side, can frozen Opportunity and
Directional Alpha diagnostics improve entry selection using only
`ACCEPT/REJECT`? A side can never be changed. `WAIT` is not evaluated in V1:
the frozen static modules do not define a causal confirmation timestamp, so a
delayed fill would require a new sequential policy.

## Evidence boundary

May-July 2026 contains 613 closed Live entries, but these outcomes were already
used by W11/W12 and prior entry audits. They may support falsification and
engineering only. July 15-31 is called VALIDATION for temporal reporting, but
is not clean confirmation. August 1 onward remains the sealed W11 holdout and
will receive no outcomes or scores in this experiment.

Every candidate is frozen from its recorded OPEN event. The policy-decision
timestamp is decoded from the immutable trade ID and the event's side cannot
change. Because Aegis evolved during the interval, the baseline version is a
per-signal policy metadata hash rather than a falsely claimed single binary.
An OPEN more than 60 seconds after the timestamp encoded in its trade ID is a
reconciliation/restoration record rather than a contemporaneous signal and
fails the timestamp-integrity gate.

## Frozen policies

- `AEGIS_ONLY`: accept every eligible Aegis signal.
- `AEGIS_OPPORTUNITY_GATE`: accept only above the previously frozen Opportunity
  TRAIN-p90 threshold `0.9999066987326859`.
- `AEGIS_CROSS_MARKET_CONFIRMATION`: preserve Aegis unless the opposite side's
  frozen predicted net utility exceeds the Aegis side by at least 20 bps.
- `AEGIS_OPPORTUNITY_CONFLICT_GATE`: require Opportunity, positive predicted
  utility for the Aegis side and at least 20 bps predicted advantage over the
  opposite side. This is the primary composed policy.

No threshold is selected from these signals. Ranking uses the product of the
frozen Opportunity probability and frozen directional favorable-first
probability for the original Aegis side. All coverage points are reported.

## Outcome and gates

The entry-quality outcome is the symmetric 0.5 ATR/60-minute common path target
with adverse-first same-bar resolution and 20 bps conservative cost. Skipped
signals contribute zero per original signal. Promotion would require a clean,
sufficiently supported validation set, positive and stable delta versus Aegis,
better path geometry and tail risk, useful coverage, and rejection of BAD at a
higher rate than destruction of GOOD.

Since no clean independent historical validation population exists, this V1
cannot authorize FINAL_HOLDOUT, prospective observation, Shadow or Live even
if a diagnostic subset looks favorable.
