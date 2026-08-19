# Design Review Gate

## Decisions already frozen for the first implementation proposal

- Scope begins as signal-conditioned: Aegis side is immutable.
- Five strategy specialists only.
- Critics are separate from specialists.
- Initial router is deterministic and supports abstention.
- Entry routing is separate from position management.
- No direct side flip or reentry in the entry experiment.
- Direction/path safety is primary during specialist discovery.
- Costs, latency, and execution feasibility are mandatory before deployment.
- Leverage is excluded from all learning and promotion criteria.
- W14 transition evidence is discovery only.
- New code remains entirely inside the sandbox until a separately reviewed
  prospective event adapter is required.
- Existing W1-W14 holdouts remain sealed.

## Resolved blockers

All prior blockers are frozen in
`10_PHASE0_FROZEN_DECISIONS.md`: fresh nested timeline, causal structural
levels, common targets, probability simplex, substate timestamps, static WAIT,
critic enforcement, equivalent-coverage routing, economic plausibility, and
signal-conditioned opposite-side semantics, general-to-Aegis transfer, pooled
side modeling, and the exact deterministic router equation.

## Phase 1 acceptance gates

The methodological choices above are no longer open implementation decisions.
Phase 1 must verify that they can be represented faithfully in data and code:

- fresh-window data exists with causal warmup and stable source identifiers;
- the frozen structural-level algorithm passes availability-time and
  future-injection tests;
- snapshot serialization is deterministic and immutable;
- effective episode counts can be audited without weakening frozen minima;
- signal-conditioned side immutability is enforced by types and tests;
- no imported adapter exposes authenticated exchange or financial mutation
  capability.

Failure of one of these checks blocks Phase 2. It does not reopen thresholds,
splits, targets, critic policy, or router metrics.

Current result: Phase 1 code and tests exist, but the full acceptance gate is
not met because the fresh-data gate is not yet available. The prior structural
clustering gap is closed by `COMPLETE_LINKAGE_DETERMINISTIC`. See
`11_PHASE1_IMPLEMENTATION_REPORT.md`. Phase 2 remains blocked.


## Approval checklist before Phase 1 code

- [x] Specialist definitions approved.
- [x] Critic definitions approved.
- [x] Signal-conditioned scope approved.
- [x] Fresh split/data acquisition approved.
- [x] Structural-level algorithm selected.
- [x] Candidate event-rate audit design approved.
- [x] Snapshot and feature contracts approved.
- [x] Router abstention and conflict behavior approved.
- [x] Primary metrics and negative-result policy approved.
- [x] Sandbox dependency and deletion boundaries approved.
- [x] No production integration authorized.

The checklist authorizes Phase 1 design implementation only. Specialists,
models, router replay, prospective observation, Shadow, and Live remain blocked
by their later phase gates.
