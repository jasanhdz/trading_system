# Phase 2 Governance Amendment

Status: `PHASE_2_IMPLEMENTATION_AUTHORIZED_VALIDATION_BLOCKED`

Recorded: `2026-08-17 UTC`

This amendment separates technical implementation from empirical validation. It
overrides only the Phase 1/Phase 2 readiness wording and the initial
`FRESH_TRAIN` boundary in earlier documents. All other Phase 0 decisions remain
frozen.

## Governance flags

- `PHASE_1_TECHNICAL_ACCEPTANCE = MET`
- `READY_TO_IMPLEMENT_PHASE_2 = TRUE`
- `FRESH_DATA_COLLECTION = ACTIVE`
- `FRESH_DATA_SUFFICIENCY = NOT_YET_MET`
- `READY_TO_VALIDATE_PHASE_2 = FALSE`
- `READY_TO_IMPLEMENT_SPECIALISTS = FALSE`
- `READY_FOR_SHADOW = FALSE`
- `READY_FOR_LIVE = FALSE`

Implementation authorization does not imply validation authorization. Phase 2
may implement deterministic, label-free candidate logic and coverage auditing.
It may not inspect outcomes or select/promote a strategy.

## Freeze checkpoint and fresh start

The governing implementation checkpoint is:

```text
checkpoint = dcd445cb293d661cc6c184a75cd39df054447ab1
checkpoint_commit_timestamp = 2026-08-17T20:59:31Z
```

There is no observed collection coverage at the exact checkpoint timestamp.
The first public market event found after it is:

```text
FRESH_TRAIN_START = 2026-08-17T21:14:26.093000Z
source = W13-P public BOOK event, SUIUSDT
```

The first contemporaneous signal snapshot after the freeze is:

```text
FIRST_FRESH_SIGNAL_SNAPSHOT = 2026-08-17T21:15:00Z
symbol = SUIUSDT
side = SHORT
quality_eligible = TRUE under the independent W13-P collection contract
```

The raw market-event start establishes acquisition coverage only. It does not
claim complete Phase 1 market snapshots, candidate coverage, sample
sufficiency, or model eligibility at that instant.

The remainder of the previously frozen calendar is unchanged:

| Partition | UTC interval |
|---|---|
| `FRESH_TRAIN` | 2026-08-17 21:14:26.093000 through 2026-10-31 23:59:59.999999 |
| `FRESH_CALIBRATION` | 2026-11-01 through 2026-11-15 |
| `SPECIALIST_VALIDATION` | 2026-11-16 through 2026-12-15 |
| `ROUTER_VALIDATION` | 2026-12-16 through 2027-01-15 |
| `FINAL_SYSTEM_HOLDOUT` | 2027-01-16 through 2027-02-28 |

## Discovery quarantine

Historical data and any data that influenced W1-W14 or the architecture may be
used only for:

- development and debugging;
- deterministic fixtures and replay;
- causal and schema validation;
- label-free coverage, gap, snapshot, and event-rate audits.

It may not be used to select thresholds, change frozen rules, promote a
strategy, compare strategy performance, or claim edge. W1-W14 holdouts remain
sealed.

Fresh Phase 2 collection is likewise label-free until
`READY_TO_VALIDATE_PHASE_2 = TRUE`. Prohibited fields and calculations include
PnL, win rate, future MFE/MAE, future barriers, realized outcomes, and any
ranking of which strategy wins.

## Frozen decision gap policy

A narrative rule is not a computable threshold. If the frozen documents do not
define a causal predicate precisely enough to produce the same result in two
independent implementations, the generator must emit `FrozenDecisionGap` and
fail closed. The gap may not be resolved from discovery outcomes or event-rate
convenience.

This policy permits generator contracts and substates to be implemented while
preventing an unspecified rule from silently becoming an optimized rule.
