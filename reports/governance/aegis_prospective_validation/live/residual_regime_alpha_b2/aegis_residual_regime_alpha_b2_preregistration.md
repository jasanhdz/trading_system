# Aegis Residual Regime Alpha B2 - Preregistration

## Purpose

B2 tests whether the useful path-risk signal found in B1 can be combined with
an independently measurable source of directional and cross-sectional alpha.
It replaces absolute future winners with beta-neutral residual returns and
conditions transparent directional mechanisms on causal market regimes.

B2 is an architecture diagnostic on previously observed history. It cannot
authorize Shadow or Live and cannot establish a production edge. A positive
result may justify a separately preregistered forward experiment only.

## Frozen Hypotheses

1. A symbol's return after removing its causal BTC beta and the common altcoin
   component is more rankable than its absolute return.
2. Directional momentum or reversal may work in specific causal regimes even
   though a universal LONG/SHORT classifier failed.
3. B1's MAE/MFE predictability can improve abstention, but cannot supply
   direction by itself.
4. A competitive favorable-versus-adverse barrier is a more useful quality
   label than selecting the best ex-post member of 22 alternatives.

## Frozen Design

- Events occur on an independent four-hour grid and enter at the next 15-minute
  open.
- Horizons are 60 and 240 minutes.
- BTC beta is estimated from the preceding 180 four-hour events, requiring at
  least 84 observations, and clipped to `[-3, 3]`.
- The common alt factor is the cross-sectional median BTC-neutral return of
  non-BTC symbols.
- Regime thresholds are fit on TRAIN only. Trend, range, transition,
  compression and expansion are causal labels, not future-known states.
- Direction must first be demonstrated by a transparent momentum, reversal or
  relative-strength baseline in TRAIN and confirmed in CALIBRATION.
- Ranking uses timestamp-grouped pairwise comparisons of residual utility.
- Path risk predicts MAE and MFE independently. It may reject a candidate but
  may not reverse or create direction.
- Favorable and adverse barriers are symmetric at 42 bps. A same-candle hit is
  classified adverse-first because OHLC data cannot establish intrabar order.
- Primary costs are 14 bps and stress costs are 20 bps.

The exact partitions, features, thresholds, controls and gates are frozen in
`config/experiments/aegis_residual_regime_alpha_b2.yaml`.

## Anti-Overfitting Rules

There is no hyperparameter search, seed search, validation threshold tuning,
best-symbol exclusion, best-regime cherry-picking or repeated gate repair.
Validation and pseudo-forward results are reported for every side, horizon and
regime, including failures. A component cannot pass on pooled performance while
hiding a failed required partition.

## Safety

B2 performs no network or exchange calls. It cannot alter PM2, Live, Shadow,
TypeScript, credentials, orders, positions, capital or runtime configuration.
All deployment flags remain false regardless of the result.
