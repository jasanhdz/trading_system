# Aegis V20 Causal Opportunity Dataset

## Purpose

V20 tests whether Aegis should stop treating every closed candle as a potential
entry and instead evaluate sparse, causally recognizable market events. It is a
dataset and feasibility experiment before it is a model experiment.

The five frozen families are trend continuation, breakout expansion, pullback
reclaim, confirmed reversal, and volatility expansion. Every condition uses
features available at the signal close. LONG and SHORT use the same directional
semantics through side-adjusted returns, trend, candle structure, and taker
flow.

## Economic Target

The primary outcome is the worst deterministic intrabar result under the
already frozen TypeScript protection proxy: hard stop, take profit, break-even,
ATR trailing with callback fallback, and round-trip costs. Secondary evidence
includes frozen ROE-10/H12 utility, MAE, MFE, time underwater, holding time,
exit reason, break-even, and trailing activation.

This aligns research more closely with how a position is actually managed. It
does not claim perfect tick-level parity because historical tick ordering and
historical exchange filters are unavailable. Both admissible OHLC paths are
replayed and the worse result is authoritative.

## Model Prohibition

No model may be trained for a family merely because its pattern sounds
plausible. Each side/family pair must first have at least 50 events, positive
mean protected return, positive mean frozen utility, positive results in at
least two of three temporal blocks, mean MAE no greater than 0.6%, and protected
win rate of at least 55%.

Families that fail remain negative evidence. Their thresholds may not be tuned
on the same observations and re-described as independent validation.

## Data Boundary

The source contains 176 causal V9 features and ten real taker-flow features for
all eleven symbols at complete timestamps. Funding, open interest, historical
order book, and liquidations are absent from this hash-bound source and will be
reported as gaps rather than fabricated.

V20 has no exchange authority. It does not modify Live, Shadow, PM2, runtime
configuration, thresholds, capital, orders, or positions.
