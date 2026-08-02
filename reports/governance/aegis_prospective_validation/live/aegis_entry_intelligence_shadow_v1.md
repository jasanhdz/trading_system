# Aegis Entry Intelligence Shadow V1

## Purpose

This contract implements observational corrections for conceptual problems 4
through 7 without changing the canonical Python selection or TypeScript
execution semantics.

The observer records, for every closed 5-minute candle and all eleven symbols:

- single-estimator uncertainty as `NOT_APPLICABLE`;
- factorized global and symbol direction;
- volatility, structure, extension, alignment and stability;
- causal entry-timing setup state;
- causal directional acceleration, pressure, transition and short-chase evidence;
- complete canonical and timing-counterfactual rankings;
- TypeScript operational disposition under the same decision-cycle identity;
- mature net return, MAE, MFE and time-underwater evidence.

## Counterfactual Alternatives

The unchanged current decision remains `CONTROL_IMMEDIATE`. Shadow compares it
with:

1. `CONTEXT_FILTERED`;
2. `TIMING_RANKED`;
3. `WAIT_RETEST`;
4. `EXHAUSTION_AVOID`.

The timing lifecycle is bounded to `CANDIDATE_SEEN`,
`WAITING_FOR_RETEST`, `TIMING_CONFIRMED`, `INVALIDATED` and `EXPIRED`.
Every transition uses only information available at the associated closed
candle. A repeated HTTP request for the same market timestamp cannot create a
new scientific observation.

## Authority Boundary

- Runtime mode: `SHADOW`.
- Exchange authority: `NONE`.
- Selection effect: `NONE`.
- Automatic training: `PROHIBITED`.
- Automatic promotion: `PROHIBITED`.
- Fallback to a second-ranked candidate: `NOT_IMPLEMENTED_OBSERVATION_ONLY`.
- Existing `selected`, `side`, guards, capital, sizing, leverage, brackets,
  trailing, callback and position management remain unchanged.

The observer contains no Binance adapter and cannot submit, cancel, modify or
close an order.

## Regime V3 Semantics

Regime V3 wraps the existing stateful Regime V2 axes while separating market
direction from symbol direction. Volatility and structure remain independent
axes. Extension uses the existing causal feature contract. Liquidity is
reported as `NOT_PRESENT_NO_CAUSAL_FEATURE`; no liquidity value is fabricated.

The output is context evidence, not entry authorization.

## Directional Acceleration Semantics

The observer separately records upward and downward pressure from nine causal
closed-candle components. It never authorizes averaging a position and never
learns online. Every observation receives a matured directional outcome after
the frozen horizon so a later, separately validated hazard model can be tested
without contaminating current Live behavior.

## Promotion Evidence

Any future proposal must use non-overlapping or embargoed outcomes and include
at least:

- 300 independent selected episodes overall;
- 50 independent episodes per included symbol;
- seven temporal blocks;
- profit factor above one;
- positive lower bound of the 95% expectancy interval;
- positive performance in both temporal halves;
- MAE improvement over `CONTROL_IMMEDIATE`;
- bounded symbol concentration;
- avoided-loss and missed-winner accounting;
- separate owner authorization tied to exact hashes.

Passing these criteria permits a promotion review only. It does not activate
Live automatically.

## Independent Committee Boundary

This work does not manufacture additional directional votes. Committee V2 and
V2.1 remain non-promotable Shadow experiments. A future directional ensemble
must demonstrate incremental out-of-sample information and measured error
diversity before its outputs can be described as consensus.
