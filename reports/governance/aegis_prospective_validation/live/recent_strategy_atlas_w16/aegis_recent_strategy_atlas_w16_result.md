# AEGIS W16 Result

## Verdict

`AEGIS_W16_NO_RECENT_CLASSIC_STRATEGY_EDGE`

- `W16_RECENT_STRATEGY_EDGE_FOUND = FALSE`
- `W16_READY_FOR_PROSPECTIVE_OBSERVATION = FALSE`
- `W16_READY_FOR_SHADOW = FALSE`
- `W16_READY_FOR_LIVE = FALSE`

## Walk-forward result

Thirty-six frozen variants across mean reversion, trend pullback, and breakout
were evaluated. For every test day from 2026-08-01 through 2026-08-15, model
selection used only the preceding 30 days.

No variant simultaneously achieved:

- positive net training expectancy;
- at least 60 training trades;
- an average frequency of 3-8 trades/day.

Consequently the governed policy correctly selected `NO_TRADE` on all 15 test
days. This is not a profitable strategy result; it is a refusal to deploy a
negative candidate.

## Closest failed candidates

The sparse mean-reversion candidate based on approximately two ATR of extension
and a flat EMA25 produced roughly 2.0-2.4 trades/day. Its rolling 30-day net
expectancy ranged from approximately -9.27 to -5.67 bps/trade after the 10 bps
cost. Its implied gross expectancy was therefore only about +0.73 to +4.33 bps.

The least-negative early-period breakout candidate generated approximately 155
trades/day and still lost around -9.4 bps net/trade. It was economically weak
before imposing the requested frequency cap and would create unacceptable
turnover.

Trend pullback variants did not outperform these failed candidates under the
frozen selection contract.

## Interpretation

Recent mean reversion exists descriptively, but the captured movement is too
small relative to costs. Forcing three to eight trades every day would require
lowering entry quality or selecting negative-expectancy setups. A daily trade
quota is therefore incompatible with the evidence.

W16 did not use leverage, send orders, alter production, or enable autonomous
execution. The available candle history ended on 2026-08-15; this result does
not represent a tradable signal for the current day.
