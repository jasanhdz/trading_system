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

## 3. Decision stages

### Stage A: integrity and hard abstention

Return `SKIP` immediately when:

- snapshot/data quality is invalid;
- feature schema/model contract mismatches;
- required timeframe has not warmed up;
- timestamp ordering cannot be proven;
- critical OOD is detected;
- a preregistered safety critic is CRITICAL.

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

Conceptual form:

```text
directional_value
  = P(favorable first) × expected favorable excursion
  - P(adverse first) × expected adverse excursion

router_value
  = directional_value
  + path-quality adjustment
  - uncertainty penalty
  - OOD penalty
  - conflict penalty
  - lateness/consumed-move penalty
```

During early directional research, costs are reported separately. Before any
deployment gate, transaction costs and latency must be added to router value.
Leverage is never part of this formula.

The exact coefficients are not chosen in this design document. They must be
preregistered or learned on TRAIN only and frozen before validation.

### Stage F: dominance and conflict

Rank eligible hypotheses by conservative value, using the lower uncertainty
bound rather than point estimate where possible.

Required conditions for `ENTER`:

- winner exceeds its absolute minimum quality;
- winner exceeds `NONE/NO_TRADE`;
- winner exceeds the runner-up by a frozen dominance margin;
- no opposed strategy remains within the ambiguity band;
- risk critics remain below their allowed severity;
- expected remaining move has not already been consumed;
- intended horizon is compatible with timeframe context.

Otherwise return `WAIT` if evidence may resolve within a frozen expiry, or
`SKIP` when it cannot.

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

`WAIT` stores a hypothesis set, not a promise to enter later.

At every pending evaluation:

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

