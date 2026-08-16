# AEGIS W16 - Recent Classic Strategy Atlas

W16 compares three independent, simple strategy families on recent public
closed candles: mean reversion, trend pullback, and volume-confirmed breakout.

For each test day from 2026-08-01 through 2026-08-15, the variant is selected
using only the preceding 30 days. A variant must have positive training net
expectancy, at least 60 trades, and an average frequency between three and eight
trades/day. The next day is then evaluated without parameter changes. If no
variant qualifies, the required action is `NO_TRADE`.

Execution uses the next 5-minute open, at most one open trade per symbol, a
one-ATR stop, and a frozen family-specific causal exit. At most eight trades
may be selected on a test day. Costs are 10 bps baseline, 14 bps stress, and 20
bps severe stress.

Approval requires at least 45 walk-forward trades, at least eight positive test
days, more than +2 bps/trade at 14 bps cost, and positive expectancy at 20 bps.
W16 cannot place orders, alter production, or authorize autonomous trading.
