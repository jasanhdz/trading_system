# Aegis SHORT Reversal Exit Compatibility X1 Preregistration

## Question

V21 found that its unchanged SHORT extreme-reversal rule selected relatively
better paths, but the assigned `LOCK_AT_5_ROE` exit remained unprofitable. A
preregistered control showed that `CURRENT_TS` would have been positive on the
same V21 holdout events. That observation generated this hypothesis; those V21
events cannot confirm it.

X1 therefore asks one specific question on evidence strictly after the V21
holdout: does the unchanged V21 SHORT extreme-reversal entry produce stable net
value when paired with the existing current TypeScript protection profile?

## Frozen Design

- The entry definition, thresholds, side, cross-sectional ranking, flow
  confirmation, and 60-minute symbol spacing are imported exactly from the
  committed V21 configuration.
- `CURRENT_TS` is the only candidate exit.
- `LOCK_AT_5_ROE` is the frozen V21 same-event control.
- `LOCK_AT_10_ROE` and `LOCK_AT_20_ROE` may be reported diagnostically but can
  neither win X1 nor replace its candidate.
- The evidence begins at 2026-07-17 19:00 UTC, after the last V21 holdout row,
  and ends at the existing canonical source boundary on 2026-08-09 06:55 UTC.
- Taker flow is reconstructed from closed five-minute candles using the same
  causal V14 implementation and complete eleven-symbol requirement.

No threshold, feature, symbol, exit, cost, gate, or seed may be changed after
the future evidence is opened and still be called X1.

## Economic Gate

X1 requires at least 20 independent candidate events, positive mean protected
net return, positive return after an additional five basis points of
round-trip cost, profit factor of at least 1.10, mean MAE no greater than
0.60%, at least two positive temporal thirds, outperformance of both the V21
exit and a matched random SHORT control, and a non-negative lower 95% daily
block-bootstrap bound.

The gate emphasizes uncertainty and path quality, not win rate. A high win
rate with negative expectancy fails.

## Boundaries

X1 does not train or export a model. It does not change Live, Shadow,
TypeScript, PM2, guards, sizing, capital, leverage, exits, or exchange state.
It performs local read-only research and has zero exchange authority. A
positive X1 result would justify a separate prospective Shadow protocol; it
would not authorize Live promotion.
