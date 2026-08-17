# Router Specification

## 1. Router purpose

The router compares already-evaluated hypotheses. It does not generate
candidates, calculate indicators, train models, or manage exchange orders.

Initial router type: deterministic, versioned, fully replayable.

Learned meta-routing is prohibited until specialists generate sufficient
out-of-fold predictions and the deterministic router is a stable baseline.

## 2. Inputs

The router receives:

- one immutable snapshot identity;
- optional frozen Aegis signal and side;
- zero or more specialist assessments;
- all critic assessments;
- calibration metadata;
- strategy compatibility graph;
- research policy version;
- no future outcome fields.

Assessments from different snapshot IDs are rejected.

Every assessment follows the probability and magnitude contract frozen in
`10_PHASE0_FROZEN_DECISIONS.md`: the three barrier probabilities come from one
coherent multinomial distribution and sum to one within tolerance. Conditional
and unconditional excursion fields are distinct and may not be substituted.

## 3. Decision stages

### Stage A: integrity and hard abstention

Return `SKIP` immediately when:

- snapshot/data quality is invalid;
- feature schema/model contract mismatches;
- required timeframe has not warmed up;
- timestamp ordering cannot be proven;
- critical OOD is detected;
- critical data quality or OOD policy requires abstention.

This stage never chooses a strategy.

### Stage B: eligibility

Remove specialists whose deterministic candidate rules are false. A specialist
cannot overcome ineligibility with a high model probability.

If no specialist remains, return `SKIP_NO_ELIGIBLE_HYPOTHESIS`.

### Stage C: calibration and support

For each eligible specialist verify:

- probability is calibrated with the correct specialist/side/horizon map;
- current feature region has sufficient support;
- uncertainty bounds are available;
- model version is approved for the dataset version.

Unsupported assessments remain logged but cannot win routing.

### Stage D: evidence de-duplication

Specialists often share evidence. The router groups evidence into root families:

- price/structure;
- trend/location;
- volatility;
- volume/flow;
- order book/microstructure;
- cross-market context;
- timing/remaining move.

Two specialists supported by the same EMA and return features do not become two
independent votes. Correlated support modifies explanation, not confidence by
simple addition.

### Stage E: comparable utility

Probability alone is insufficient because strategies have different gains,
losses, and horizons. Each assessment is transformed into a comparable research
utility.

Frozen barrier-payoff core:

```text
expected_barrier_payoff
  = p_favorable_first × favorable_barrier
  - p_adverse_first × adverse_barrier
  + p_neither × expected_terminal_return_given_neither

router_value
  = lower_90_bound(expected_barrier_payoff)
```

The lower bound is estimated by the frozen block-bootstrap model ensemble with
the frozen calibration map defined in the Phase 0 record. No learned utility weights, critic
penalties, or path-quality coefficients exist in the initial router. Path,
conflict, OOD, lateness, and economic requirements are explicit gates. The
formula does not multiply unconditional MFE/MAE by event probabilities, which
would count event likelihood twice. Leverage is never part of this formula.

### Stage F: dominance and conflict

Rank eligible hypotheses by conservative value, using the lower uncertainty
bound rather than point estimate where possible.

Required conditions for `ENTER`:

- winner conservative value exceeds the frozen 20 bps plausibility hurdle;
- winner exceeds `NONE/NO_TRADE`;
- winner exceeds the runner-up by at least 5 bps;
- no opposed `CONFLICT_ONLY` strategy remains within 5 bps of the winner;
- `p_favorable_first > p_adverse_first`;
- `expected_mfe_unconditional > expected_mae_unconditional`;
- no hard data-quality, version, or critical-OOD abstention exists;
- diagnostic market critics are recorded but do not alter initial utility;
- expected remaining move has not already been consumed;
- intended horizon is compatible with timeframe context.

Otherwise return terminal `WAIT` during static router replay when evidence may
resolve later, or `SKIP` when it cannot. Static `WAIT` does not read another
snapshot; sequential reevaluation begins only in the later pending experiment.

### Stage G: action contract

Every action contains:

- selected strategy or explicit absence;
- side and horizon;
- `ENTER`, `WAIT`, or `SKIP`;
- winner and runner-up values;
- dominance margin;
- critic severities;
- positive and negative evidence;
- invalidation level;
- expiry for pending decisions;
- reason codes for every rejected specialist;
- complete version/hash references.

## 4. Signal-conditioned versus independent mode

Two modes must not be mixed in one experiment.

### Signal-conditioned router

Aegis provides frozen LONG/SHORT. Opposite-side specialists may identify
conflict but cannot create an opposite order. Actions are:

- accept original side;
- wait;
- skip.

### Independent strategy router

Specialists may propose either side. This is a new directional brain and needs
separate data, labels, governance, and authorization.

Initial work uses signal-conditioned mode unless explicitly changed.

Every assessment also declares `routing_role = PROPOSER | CONFLICT_ONLY`.
Only specialists aligned with the frozen Aegis side may be `PROPOSER`.
Opposite-side specialists are `CONFLICT_ONLY`: they may cause `WAIT` or `SKIP`
but cannot win, create an order, or flip the side.

## 5. Timeframe-horizon compatibility

Higher timeframes are not universal vetoes. The router uses an explicit matrix:

| Intended horizon | Primary context | Timing | Higher context effect |
|---|---|---|---|
| 1-5 minutes | 5m/15m | 1m/micro | 1h/4h raises confirmation requirement |
| 5-30 minutes | 15m/1h | 1m/5m | 4h affects direction/location |
| 30-120 minutes | 1h/4h | 5m/15m | 1d affects regime/location |

An opposing daily trend may be acceptable for a short mean-reversion scalp but
not for a two-hour continuation thesis. Specialists must declare their horizon.

## 6. Pending decisions

In Phase 6, `WAIT` is a terminal classification and realized per-signal utility
is zero. It stores an explanation but does not advance time.

In Phase 7, `WAIT` may create a pending hypothesis set, not a promise to enter.

Only in Phase 7, at every pending evaluation:

- rebuild a new causal snapshot;
- reevaluate the same eligible hypotheses;
- recompute remaining excursion from the current price;
- cancel expired or invalidated hypotheses;
- prohibit retroactive capture of price movement during waiting.

Pending terminates with exactly one of:

- `ENTER_CONFIRMED`;
- `CANCEL_INVALIDATED`;
- `CANCEL_TOO_LATE`;
- `CANCEL_AMBIGUOUS`;
- `CANCEL_TIMEOUT`;
- `CANCEL_DATA_QUALITY`.

## 7. Existing-position behavior

Entry routing does not manage open positions in the first experiment.

A future lifecycle router must preserve the current thesis identity. A new
opposite hypothesis may increase invalidation evidence but cannot immediately
flip the position.

Required sequence:

```text
OPEN(current thesis)
→ INVALIDATING
→ EXIT
→ COOLDOWN/FLAT
→ new independent confirmation
→ optional new entry
```

## 8. Meta-router prerequisites

A learned router may be considered only when:

- every included specialist passed its own validation gate;
- predictions are generated out-of-fold for all router training rows;
- calibration is performed without router-test leakage;
- `NONE` is represented explicitly;
- sample size supports interactions;
- deterministic router is a frozen baseline;
- ablation proves the meta-router adds stable value.

Training a router on in-sample specialist predictions is prohibited because it
would learn unrealistically clean confidence patterns.
