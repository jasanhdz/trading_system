# AEGIS W14 Result

## Verdict

`AEGIS_W14_NO_RECENT_ECONOMIC_EDGE`

- `W14A_RECENT_SHORT_EDGE_FOUND = FALSE`
- `W14B_VOLUME_NAVIGATION_EDGE_FOUND = FALSE`
- `W14_EDGE_FOUND = FALSE`
- `W14_READY_FOR_SHADOW = FALSE`
- `W14_READY_FOR_LIVE = FALSE`

No production configuration, trading logic, leverage, or order path changed.

## W14A: recent 30-day SHORT selector

The calibration procedure froze regularized logistic regression, a 30-minute
horizon, and the 95th probability percentile. It selected 52 of 869 validation
anchors (5.98%).

| Metric | Result |
| --- | ---: |
| Gross expectancy/trade | +6.92 bps |
| Net expectancy/trade, 14 bps cost | -7.08 bps |
| Net expectancy/original signal | -0.42 bps |
| Net expectancy/trade, 20 bps stress | -13.08 bps |
| Profit factor | 0.70 |
| Win rate after 14 bps | 38.46% |
| Positive symbols | 1 of 11 |
| Bootstrap 95% CI | [-21.57, +6.54] bps |
| Bootstrap P(net > 0) | 15.78% |

Entering SHORT at every hourly anchor was also negative: +1.10 bps gross and
-12.90 bps net per trade. The selector reduced the loss substantially but did
not identify trades whose movement paid the cost.

Daily selected-trade expectancy was unstable:

- 2026-08-12: -17.46 bps net/trade.
- 2026-08-13: +2.64 bps net/trade.
- 2026-08-14: -26.03 bps net/trade.
- 2026-08-15 partial day: +16.08 bps net/trade.

The positive partial final day cannot be promoted because the frozen aggregate
validation failed and selecting that day afterward would be temporal overfit.

## W14B: isolated volume navigation

Calibration froze two aligned candles, volume ratio >=1.5, next-open entry,
and the causal decay exit. Validation contained 515 non-overlapping episodes.

| Metric | Result |
| --- | ---: |
| Gross expectancy/trade | +1.16 bps |
| Net expectancy/trade, 14 bps cost | -12.84 bps |
| Net expectancy/trade, 20 bps stress | -18.84 bps |
| Profit factor | 0.29 |
| Win rate after 14 bps | 18.45% |
| Positive symbols | 0 of 11 |
| Bootstrap 95% CI | [-15.74, -9.94] bps |

LONG episodes produced -12.01 bps net/trade and SHORT episodes -13.60 bps.
Every validation day was negative after costs. Two or three volume-aligned
candles did not leave enough remaining movement to navigate economically.

## Interpretation

The recent market did contain profitable SHORT intervals, but the frozen
causal features did not identify them consistently before entry. A 30-day
window adapts faster, but it also has much higher regime and estimation risk.
The observed gross movement was below the 14 bps hurdle in both studies.

This experiment does not justify using either candidate today or tomorrow.
Testing more recent sub-days or retaining only the positive dates after seeing
validation would turn the experiment into retrospective selection.
