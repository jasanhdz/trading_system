# Aegis V20 Opportunity Feasibility Report

## Verdict

`V20_READY_FOR_MODELING = FALSE`

`V20_READY_FOR_SHADOW = FALSE`

`V20_READY_FOR_LIVE = FALSE`

The preregistered builder evaluated 90,442 rows with complete eleven-symbol
taker flow and produced 7,172 side/family opportunity rows. No side/family pair
passed every frozen economic and stability gate. Model training was therefore
not executed.

## Results

| Side | Family | Events | Mean protected net | Frozen utility | Win rate | Mean MAE | Positive thirds |
|---|---|---:|---:|---:|---:|---:|---:|
| LONG | Breakout expansion | 21 | -0.2270% | +0.0004% | 81.0% | 1.43% | 1/3 |
| LONG | Confirmed reversal | 29 | -0.1608% | -0.1122% | 51.7% | 0.51% | 0/3 |
| LONG | Pullback reclaim | 45 | -0.0811% | -0.1715% | 53.3% | 0.63% | 1/3 |
| LONG | Trend continuation | 3,142 | -0.1788% | -0.2127% | 53.5% | 0.74% | 0/3 |
| LONG | Volatility expansion | 138 | -0.2438% | -0.1466% | 73.9% | 1.28% | 1/3 |
| SHORT | Breakout expansion | 7 | -0.2261% | -0.4219% | 71.4% | 1.26% | 1/3 |
| SHORT | Confirmed reversal | 51 | -0.0815% | -0.1576% | 49.0% | 0.44% | 0/3 |
| SHORT | Pullback reclaim | 38 | -0.2662% | -0.2107% | 52.6% | 0.69% | 0/3 |
| SHORT | Trend continuation | 3,584 | -0.1799% | -0.2051% | 57.1% | 0.81% | 0/3 |
| SHORT | Volatility expansion | 117 | -0.3179% | -0.2828% | 63.2% | 1.01% | 1/3 |

Win rate did not imply profitability. Several breakout and expansion families
closed positively often, but their less frequent losses were larger. The
protection layer improved many paths but did not create positive expectancy.

Trend continuation had sufficient sample size and the tightest uncertainty
intervals. Its monthly-block 95% intervals were entirely negative for both
LONG and SHORT, providing strong evidence against training a ranker on that
population as currently defined.

The smaller families remain statistically uncertain, but uncertainty is not
evidence of edge. Their thresholds are frozen and were not relaxed after these
results.

## Controls And Interpretation

Most family rules did not outperform a matched random sample reliably. Small
positive regime slices existed, but typically contained only one to twenty-five
events. They are exploratory diagnostics and cannot be promoted or used to
retroactively redefine V20.

The result supports the architectural diagnosis: sparse events are a better
research unit than every candle, but conventional candle-pattern names plus
taker direction do not automatically produce an economic advantage. A richer
state representation or a different economic hypothesis is needed.

## Data Readiness

- Five-minute kline microstructure: 1,710,710 rows, 11 symbols, from
  2025-02-15 through 2026-08-09.
- Funding history: 17,820 rows, 11 symbols, over the same broad period.
- Open interest: 5,501 rows covering only about two days.
- Global taker ratio: 5,501 rows covering only about two days.
- Depth: 11 snapshots, one per symbol.
- Historical liquidations: absent.

Funding is mature enough to become a preregistered input in a later experiment.
Open interest, depth, and liquidation context require prospective collection
before they can support a credible model.

## Safety

No model was exported. Live and Shadow were unchanged. The builder made no
network or exchange call and no exchange mutation.
