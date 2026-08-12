# Aegis Market Event Fast Track M1A

## Purpose

M1A avoids waiting for a new 60-day microstructure window by downloading
checksum-verified public Binance archives. It tests whether a small,
preregistered set of causal micro-patterns has economic utility when conditioned
on multi-timeframe regime and Spot/Futures flow.

M1A is research only. It cannot modify Live, Shadow, TypeScript execution,
capital, guards, models, orders, positions, or PM2.

## Honest evidence boundary

The historical prices through July 2026 have been indirectly examined during
earlier Aegis experiments. Therefore M1A's latest retrospective interval is a
**pseudo-holdout**, not untouched promotion evidence. It can reject weak ideas
and identify candidates, but it cannot independently authorize Live.

Fresh confirmation begins after this preregistration and requires at least 30
days and 200 independent events per direction before any later Shadow verdict.

## Data

M1A uses official public archives for Spot and USD-M Futures klines,
aggregate trades, funding and mark-price candles where available. Every archive
must have a verified checksum, an immutable raw copy and a source manifest.
Missing periods are quarantined; they are never zero-filled.

For the preregistered one-minute first pass, aggressive flow is reconstructed
directly from the physical kline fields `taker buy quote volume` and total quote
volume. Tick-level aggTrades are optional enrichment for a later subminute
experiment; they are not required to duplicate the same one-minute aggregate.

One-minute observations are the physical base. The 5m, 15m, 1h, 4h and 1d
views are resampled from closed one-minute observations so all features share a
single causal clock. Partial higher-timeframe candles are prohibited.

## Regime

Regime has separate direction, volatility and liquidity axes. It is context,
not an entry signal. Thresholds are fitted on the discovery partition only,
with hysteresis and minimum state duration to avoid changing regime on every
small fluctuation.

## Frozen pattern families

1. trend pullback continuation;
2. compression breakout;
3. liquidity sweep rejection;
4. aggressive-flow absorption reversal;
5. exhaustion reversal;
6. Spot/Futures divergence followed by convergence;
7. multi-timeframe reclaim;
8. session/funding dislocation with flow confirmation.

Day of week, session and regime may describe context but cannot independently
authorize an event. Symbol-specific thresholds are prohibited in the first
pass.

## Evaluation

Entries occur at the next one-minute open after confirmation. Outcomes include
net return, MAE, MFE, target-before-stop and time-to-profit over 15–240 minutes.
Fees, slippage, funding, correlated events, overlapping symbols and capital are
included. Controls are time-matched and regime-matched random events, simple
momentum, simple mean reversion and candle-only matched events.

The first pass is rules-only. Models are permitted only for a pattern family
that first shows positive validation edge. Direction, MAE risk and net utility
remain separate questions.

## Excluded sources

News and whale labels are deferred to M1B. They require immutable publication
timestamps, revision history and defensible entity/transfer classification.
Using them now would create a larger leakage and attribution problem than the
information they might add.

## Gate

A pattern needs at least 100 independent validation events per direction,
positive 95% lower bounds for net expectancy and profit factor, temporal
stability, controlled concentration, superiority to time/regime-matched random
events and survival under base and stress costs. Negative results are retained.

No M1A result automatically activates Shadow or Live.
