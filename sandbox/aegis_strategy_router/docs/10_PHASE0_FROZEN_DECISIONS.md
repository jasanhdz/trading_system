# Phase 0 Frozen Decisions

Status: `PHASE_0_METHODOLOGY_FROZEN`

This document resolves the blockers identified during adversarial design review.
It overrides any conflicting earlier wording in the sandbox documentation.

## 1. Discovery quarantine and fresh timeline

All market data, operations, reports, screenshots, thresholds, observations, and
ideas that influenced W1-W14 or this architecture through
`2026-08-17T23:59:59.999999Z` belong to `DISCOVERY_QUARANTINE`.

They may be used to:

- test plumbing and deterministic replay;
- audit candidate event rates;
- detect schema/data defects;
- create synthetic/unit fixtures;
- document prior negative evidence.

They may not provide confirmatory performance, choose final thresholds, approve
specialists, approve critics, approve the router, or open old holdouts.

Fresh prospective timeline:

| Partition | UTC interval | Purpose |
|---|---|---|
| `FRESH_TRAIN` | 2026-08-18 00:00 through 2026-10-31 23:59 | Candidate/rule development, model fitting, feature ablation |
| `FRESH_CALIBRATION` | 2026-11-01 00:00 through 2026-11-15 23:59 | Probability calibration only |
| `SPECIALIST_VALIDATION` | 2026-11-16 00:00 through 2026-12-15 23:59 | Specialist selection, critic validation, router design/freeze |
| `ROUTER_VALIDATION` | 2026-12-16 00:00 through 2027-01-15 23:59 | Frozen router versus frozen specialists |
| `FINAL_SYSTEM_HOLDOUT` | 2027-01-16 00:00 through 2027-02-28 23:59 | One final system test after complete freeze |

Boundaries are based on event timestamps. Candidate horizons crossing a boundary
are purged. Episodes/snapshots and all specialist evaluations derived from them
remain in one partition.

No partition is extended, merged, or borrowed from after outcomes are inspected.
Insufficient effective sample produces `BLOCKED_INSUFFICIENT_SAMPLE`. A later
experiment may preregister a new future schedule before opening labels.

Minimum independent episodes:

- specialist TRAIN: 2,000 candidates per fitted specialist;
- calibration: 500 candidates per fitted specialist;
- specialist validation: 500 candidates per promoted specialist;
- router validation: 300 original Aegis signal episodes;
- final system holdout: 300 original Aegis signal episodes;
- side-specific claims: at least 150 validation episodes for that side;
- temporal support: at least four non-overlapping weekly blocks;
- symbol support: at least six symbols, with no symbol above 35% of episodes.

Rows are not sample size. Minimums refer to effective candidate/signal episodes
after overlap purging.

## 2. Router-development separation

Specialists are fitted on `FRESH_TRAIN`, calibrated on `FRESH_CALIBRATION`, and
selected on `SPECIALIST_VALIDATION`.

Router rules, dominance margins, compatible-specialist handling, validated
critic behavior, and abstention thresholds may use only out-of-fold TRAIN
predictions plus `SPECIALIST_VALIDATION`. They are frozen before
`ROUTER_VALIDATION`.

`ROUTER_VALIDATION` is never used to repair specialists or router policy.

`FINAL_SYSTEM_HOLDOUT` opens once only when all of the following are frozen:

- candidate rules and substates;
- feature schemas;
- specialist models and calibrators;
- critics and enforcement modes;
- router equation, thresholds, margins, and coverage policy;
- data-quality/OOD policy;
- primary metrics and gates;
- cost/latency plausibility assumptions.

Failure at `ROUTER_VALIDATION` leaves final holdout sealed.

## 3. Causal support/resistance algorithm

Primary structural levels use confirmed fractal pivots and causal clustering.

Pivot definition:

- pivot high at bar i: `high[i]` is strictly greater than highs at i-2, i-1,
  i+1, and i+2;
- pivot low is symmetric;
- the pivot becomes available only when bar i+2 closes;
- ties do not create a pivot;
- no pivot may be timestamped at bar i for inference before confirmation.

Clustering and strength:

- levels cluster separately for highs and lows;
- cluster tolerance: 0.20 ATR14 of the level's timeframe at evaluation time;
- at least two confirmed touches;
- touches must be separated by at least three bars;
- cluster price is the median confirmed pivot price;
- level age and last-touch time are retained continuously;
- broken levels remain historical and may become retest candidates, but are not
  silently deleted.

Lookbacks:

- 15m: 96 closed bars;
- 1h: 120 closed bars;
- 4h: 90 closed bars;
- 1d: 60 closed bars.

The context map emits nearest causal support/resistance above and below price,
touch count, age, distance in ATR/bps, and next-level space. Rolling prior
high/low is diagnostic only and not a second selectable level family.

## 4. Candidate substates and availability

Candidate state belongs to a timestamp. A later confirmation creates a new
candidate decision boundary and cannot inherit the earlier entry price.

### Trend continuation

- `TREND_CANDIDATE`
- `TREND_CONFIRMED`
- `TREND_INVALIDATED`

### Pullback continuation

- `PULLBACK_FORMING`: terminal `WAIT` during static routing;
- `PULLBACK_CONFIRMED`: new candidate timestamp after causal realignment;
- `PULLBACK_INVALIDATED`;
- `PULLBACK_TOO_LATE`.

### Breakout/retest

- `BREAKOUT_CANDIDATE`: close beyond level by at least 0.10 ATR;
- `BREAKOUT_CONFIRMED`: breakout close also satisfies specialist eligibility;
- `RETEST_PENDING`: terminal `WAIT` during static routing;
- `RETEST_CONFIRMED`: price touches within 0.20 ATR of broken level and closes
  back on the breakout side;
- `FALSE_BREAKOUT`: closed back inside before confirmation;
- `BREAKOUT_TOO_LATE`: remaining common-target space below one target barrier.

### Range mean reversion

- `RANGE_EDGE_CANDIDATE`
- `RANGE_REJECTION_CONFIRMED`
- `RANGE_BROKEN`

### Regime transition/reversal

- `OLD_REGIME_DETERIORATING`: terminal `WAIT` in static routing;
- `TRANSITION_CANDIDATE`;
- `NEW_REGIME_CONFIRMED`;
- `TRANSITION_FAILED`.

Phase 6 static routing never advances substates. It outputs terminal `WAIT`.
Only Phase 7 may create `pending_episode_id`, consume new snapshots, and advance
or cancel a candidate.

## 5. Common label and probability contract

All specialists must predict one common router target in addition to optional
strategy diagnostics.

Reference volatility:

- ATR14 from the last fully closed 15m candle at candidate timestamp;
- frozen for that candidate episode.

Common triple barrier:

- favorable: +0.50 reference ATR in proposed direction;
- adverse: -0.50 reference ATR;
- horizon: 60 minutes from actual candidate/confirmation timestamp;
- classes: `FAVORABLE_FIRST`, `ADVERSE_FIRST`, `NEITHER`;
- same lower-resolution bar ambiguity resolves `ADVERSE_FIRST`;
- higher-resolution ordering may replace ambiguity only when causally complete.

Probability estimation:

- one multinomial model/softmax head per specialist;
- probabilities are non-negative and sum to 1 within numerical tolerance
  `1e-6`;
- three independent binary classifiers are prohibited for the common class;
- calibration preserves simplex coherence using a multiclass calibration map.

Magnitude semantics are explicit:

- `expected_mfe_unconditional = E[MFE from now]`;
- `expected_mae_unconditional = E[MAE from now]`;
- `expected_mfe_given_favorable = E[MFE | FAVORABLE_FIRST]`;
- `expected_mae_given_adverse = E[MAE | ADVERSE_FIRST]`;
- `expected_terminal_return_given_neither = E[R_60m | NEITHER]`.

No field named only `expected_mfe` or `expected_mae` is allowed.

Common expected barrier payoff, before costs:

```text
expected_barrier_payoff_bps =
    P(FAVORABLE_FIRST) × favorable_barrier_bps
  - P(ADVERSE_FIRST) × adverse_barrier_bps
  + P(NEITHER) × E[R_60m_bps | NEITHER]
```

MFE/MAE expectations are path-risk constraints and explanations. They are not
multiplied into this payoff, preventing probability double counting.

Strategy-specific secondary labels:

- continuation: structure survives through common horizon;
- pullback: realignment before invalidation and remaining common payoff from
  confirmation;
- breakout/retest: level holds versus false break;
- mean reversion: range midpoint before range break;
- transition: new-regime structure persists through 120 minutes.

Secondary labels cannot replace the common target in router comparison.

## 6. Signal-conditioned side contract

Initial scope remains signal-conditioned.

Every specialist assessment has:

- `proposal_side`;
- `aegis_side`;
- `routing_role = PROPOSER | CONFLICT_ONLY`.

When sides match, role may be `PROPOSER`. When sides oppose, role is always
`CONFLICT_ONLY` and the type contract forbids selection as an entry winner.

Opposing evidence may produce `WAIT` or `SKIP`; it can never transform SHORT to
LONG or LONG to SHORT. Independent-direction routing is a separate experiment.

## 7. Population transfer and side modeling

Specialists fit on the general candidate population generated across the frozen
fresh market timeline. They are calibrated on the corresponding general
candidate population in `FRESH_CALIBRATION`. The initial experiment does not
refit or recalibrate a specialist on Aegis-selected signals.

Transfer to Aegis is evaluated explicitly:

- report feature support and prediction reliability on Aegis signals;
- reject individual assessments outside calibrated support through the OOD
  contract;
- report general-candidate and Aegis-conditioned metrics separately;
- router approval uses only original Aegis signal episodes;
- performance on the general population cannot approve the router.

Each initial specialist uses one pooled, direction-normalized model. Features
are transformed so positive means favorable to `proposal_side`, and LONG/SHORT
share identical labels and thresholds. Side is retained as a diagnostic field,
not a tunable interaction. A separate side-specific model is prohibited in the
initial experiment. Any side-specific claim still requires the frozen minimum
of 150 independent validation episodes for that side.

## 8. Critic enforcement policy

Hard abstentions by architecture:

- invalid/stale/incoherent data;
- critical out-of-distribution state;
- model/feature/calibration version mismatch.

Market critics, including Shock, Exhaustion, Space, and Conflict, begin as
`DIAGNOSTIC_ONLY`. Their values are logged but cannot veto or penalize specialist
utility during specialist validation.

A market critic may become a router soft penalty or veto only if an independent
preregistered test on `SPECIALIST_VALIDATION` proves incremental value and its
mode is frozen before `ROUTER_VALIDATION`. Shock CRITICAL is therefore not an
initial veto.

## 9. Static WAIT versus sequential WAIT

Phase 6 `WAIT` is terminal for that replay row:

- no future snapshot is read;
- no candidate is advanced;
- no delayed entry is simulated;
- realized utility for that original signal is zero;
- it is logged separately from `SKIP` for coverage diagnostics.

Phase 7 is the first phase allowed to create pending episodes, reevaluate new
snapshots, measure consumed movement, and produce `ENTER_CONFIRMED/CANCEL_*`.

Phase 7 cannot be used to rescue a failed static router.

## 10. Frozen router equation and equivalent coverage

For each specialist, create 1,000 temporal-block bootstrap refits using only
its authorized TRAIN rows. Apply the frozen calibration map to every refit.
For an assessment, compute `expected_barrier_payoff_bps` for every refit and
define:

```text
router_value_bps = percentile_10(expected_barrier_payoff_bps_refits)
```

This is the sole numeric utility of the initial deterministic router. There are
no fitted utility weights and market critics contribute no numeric penalty
while diagnostic-only.

Aligned `PROPOSER` assessments compete with `NONE`, whose value is zero.
Opposing `CONFLICT_ONLY` assessments cannot win but participate in the conflict
test. `ENTER` requires all of:

- winner `router_value_bps > 20`;
- winner exceeds the next aligned proposer, or `NONE`, by at least 5 bps;
- no opposing assessment has `router_value_bps` within 5 bps of the winner;
- `p_favorable_first > p_adverse_first`;
- `expected_mfe_unconditional > expected_mae_unconditional`;
- no hard data-quality, version, or critical-OOD abstention;
- candidate substate is enterable rather than pending or invalidated;
- at least one full common barrier remains before the nearest favorable
  structural obstruction.

`WAIT` is returned only for a valid pending substate or conflict inside the
5-bps ambiguity band. All other failures return a deterministic `SKIP_*` reason.
No threshold in this decision rule is fitted on validation.

Realized common payoff for an executed episode:

```text
+favorable_barrier_bps when FAVORABLE_FIRST
-adverse_barrier_bps when ADVERSE_FIRST
terminal directional return at 60m when NEITHER
```

Skipped or terminal-WAIT signals contribute zero to per-original-signal payoff.

Primary router metrics:

- mean realized common payoff per original Aegis signal;
- mean realized common payoff per executed candidate;
- favorable-first and adverse-first rates;
- MFE > MAE rate;
- median, P90, and P95 MAE;
- execution coverage.

Equivalent-coverage comparison:

- frozen coverage grid: 10%, 20%, 35%, 50%, 75%, 100%;
- at router realized coverage k, compare against every validated specialist's
  top-k candidates ranked by its frozen conservative expected payoff;
- interpolate only between adjacent grid points and report the exact unrounded
  count;
- compare against random eligible selection with 10,000 episode bootstraps;
- report the full risk/coverage curve and area under that curve.

Router promotion requires:

- at least +5 bps common payoff per original signal over the best specialist at
  equivalent coverage;
- at least +5 percentage points favorable-first rate at equivalent coverage;
- 95% paired episode-bootstrap lower bound above zero for payoff improvement;
- executed common payoff > 0;
- MFE > MAE in at least 55% of executed episodes;
- P95 MAE no worse than the best specialist at equivalent coverage;
- router coverage at least 20% and at most 80%;
- no symbol above 35% and required temporal/side support.

These gates are evaluated first on `ROUTER_VALIDATION`, then unchanged on the
single final holdout.

## 11. Economic plausibility kill gate

Directional/path information remains the training objective. Economic
plausibility prevents progression of statistically interesting but unusable
specialists.

Frozen conservative stress cost: 20 bps round trip, covering fee, spread,
slippage, and modest latency uncertainty for plausibility screening.

After `SPECIALIST_VALIDATION`:

- compute episode-bootstrap 95% CI for gross realized common payoff per
  executed candidate;
- if the CI upper bound is <= 20 bps, set
  `ECONOMIC_PLAUSIBILITY = FALSE`;
- such a specialist may remain scientifically documented but cannot enter the
  router candidate set;
- costs are not subtracted from training labels;
- leverage is never used to pass this gate.

At least two specialists must pass directional/path gates and economic
plausibility before router evaluation is meaningful.

## 12. Phase 0 verdict

With this document and the consistency amendments in the other design files:

- `DESIGN_CONCEPT_APPROVED = TRUE`
- `ARCHITECTURE_REWRITE_NEEDED = FALSE`
- `PHASE_0_APPROVED = TRUE`
- `READY_TO_IMPLEMENT_PHASE_1 = TRUE`
- `READY_TO_IMPLEMENT_SPECIALISTS = FALSE`
- `READY_FOR_SHADOW = FALSE`
- `READY_FOR_LIVE = FALSE`

Phase 1 authorization covers contracts, snapshots, causal feature/level
adapters, data audit, leakage tests, and deterministic fixtures only. Phase 2
candidate generators require Phase 1 acceptance. Model training remains gated
by the event-rate audit.
