# Aegis Volume Wave W1 Result

## Verdict

`AEGIS_VOLUME_WAVE_W1_NO_OPERABLE_EDGE`

W1 does not justify a model, a real-time detector, Shadow deployment, or Live
integration. The final holdout remains sealed. No production source, process,
configuration, credential, or exchange state was touched.

The visual hypothesis is plausible as a description of some market episodes,
but the preregistered conditions did not identify a stable, cost-surviving
continuation edge in TRAIN and VALIDATION.

## Scope

- Data: checksum-verified Binance USD-M 1m klines and aggTrades already held
  locally, aggregated causally to closed 5m and 15m bars.
- Period: August 2025 through July 2026.
- Symbols: BTC, ETH, SOL, BNB, XRP, DOGE, ADA, AVAX, LINK, SUI, and LTC USDT.
- Directions: LONG and SHORT evaluated separately.
- Context: asset 5m, asset 15m, and BTC 5m/15m.
- Entry variants: impulse close, one-bar confirmation, causal pullback, and
  extreme-break confirmation.
- Labels: 30 registered ATR triple-barrier contracts, resolved on the future
  1m path for horizons of one through six 5m bars.
- Economics: 14 bps base round trip, with 20 and 30 bps stress tests.
- Inference controls: day-clustered 10,000-sample bootstrap, Bayesian
  continuation posterior, matched price-only controls, walk-forward folds,
  and Benjamini-Hochberg false-discovery control.

The supplied XRP screenshots were used only to test the feature design. They
show both continuation and a high-volume selloff followed by reversal, which
supports the need to distinguish persistence from terminal volume. They were
not used as statistical evidence or threshold inputs.

## Causal Features

The dataset records continuous volume ratios and log-volume z-scores, body
ratio, close location, taker imbalance, ATR-normalized velocity and
acceleration, RSI 6/12/24, MA 7/25/99 relationships and slopes, market
structure, path efficiency, BTC alignment, rolling BTC correlation, 15m
context, extension, and future-only labels. Features at decision time never
read future data. Only complete 1m, 5m, and 15m bars are used.

The 1,112,204-row research dataset contains 556,102 wave candidates and the
same number of deterministic matched controls. TRAIN contains 555,434 rows and
VALIDATION contains 273,308 rows. The May-July 2026 FINAL HOLDOUT was not read.

## Main Results

Fifty-two hypotheses had enough TRAIN data to evaluate; four preregistered
high-specificity combinations were insufficient. Passing hypotheses: zero.
FDR-significant hypotheses: zero.

| Best validation configuration | N | Gross diagnostic | Net expectancy | PF | Positive symbols | Positive WF folds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LONG, space remaining, immediate, H3/F1.0/A0.6 ATR | 426 | +1.386 bps | -12.614 bps | 0.254 | 0/11 | 0/3 |
| SHORT, BTC not opposing, one-bar, H2/F1.0/A0.6 ATR | 1,020 | +1.354 bps | -12.646 bps | 0.332 | 0/11 | 0/4 |

The LONG result is also below the preregistered minimum of 1,000 validation
events. Its 95% expectancy interval is approximately -15.18 to -10.04 bps.
The SHORT interval is approximately -16.08 to -8.87 bps. Both fail the 20 and
30 bps stress tests and neither has a positive walk-forward month.

Best result by entry timing:

| Entry timing | Best net expectancy | Gross diagnostic |
| --- | ---: | ---: |
| Immediate impulse close | -12.614 bps | +1.386 bps |
| One-bar confirmation | -12.646 bps | +1.354 bps |
| Causal pullback | -13.940 bps | +0.060 bps |
| Extreme break | -13.862 bps | +0.138 bps |

Waiting for a pullback or break did not reveal hidden edge. Immediate and
one-bar entries had a very small gross tendency, but it was roughly one tenth
of the preregistered execution cost.

## Variable Findings

No volume-ratio, z-score, body, taker-flow, RSI-space, extension, 5m trend,
15m trend, or BTC-context bin produced positive net expectancy under the fixed
descriptive contract.

- Volume around 1.25x-4x sometimes improved MFE relative to MAE, but did not
  create positive net EV.
- Volume above 4x was generally worse, consistent with the terminal-volume or
  capitulation concern, but this is diagnostic rather than a promoted rule.
- Strong aligned taker flow improved path shape in some SHORT samples, but did
  not survive costs or stability gates.
- Adding clean-body, flow, trend, BTC, and remaining-space conditions reduced
  sample size without creating robust economic utility.
- LONG and SHORT behaved differently, but neither direction passed.

## Persistence And Exhaustion

The underlying path is not equivalent to candle color, so W1 measured MFE,
MAE, directional persistence, path efficiency, future flow, velocity, and
giveback. The findings do not support a universal ride-the-wave rule:

- Only about 47%-48% of candidate paths closed favorably at each of the first
  six 5m bars.
- Side-aligned taker imbalance was small after the impulse and generally faded
  toward zero or opposition.
- Among paths with a meaningful favorable excursion, median giveback was about
  84%-86% after one bar and approached the entire peak later.
- This explains why screenshots can look compelling while unconditional
  event-level expectancy remains weak: many impulses contain tradable-looking
  excursions but give them back quickly and inconsistently.

These observations are useful for future exit research, but W1 did not first
establish an entry population with positive net EV. Optimizing an exhaustion
score on these same samples would be post-selection and is therefore not done.

## Reusable Architecture

Existing components that could be reused only after a future hypothesis earns
Shadow admission:

- Python local public archive and causal aggregation for historical research.
- Existing Aegis feature and event-quality utilities for ATR, path quality,
  costs, bootstrap, and governance evidence.
- TypeScript `WebSocketManager` for closed kline and aggTrade streams.
- TypeScript `TradingService`, `BinanceAdapter`, and `FsStateStore` for the
  already established execution, bracket, reconciliation, and persistence
  boundaries.

No parallel execution engine was built. W1 remains isolated in
`src/aegis/research/volume_wave_w1.py` and its two research scripts.

## Integrity

- Preregistered configuration SHA-256:
  `4adb0ee12b74a023412d476d7d21cf5b88e3afb695896c5fca460c1537a16ee3`
- Dataset manifest SHA-256:
  `438b184f85836758543a4dddc96cbec441a924f30e032c06d2391c7924f273af`
- TRAIN/VALIDATION evaluation SHA-256:
  `e3c5b8abd3b2ee6b964d60bd2241cf29dad5f53bfe673ea149d3cced78539460`
- Final holdout: `SEALED`
- Authenticated exchange requests: `0`
- Exchange mutations: `0`
- Production integration: `NONE`

## Decision

`W1_RULE_EDGE_FOUND = FALSE`

`W1_MODELING_JUSTIFIED = FALSE`

`W1_READY_FOR_SHADOW = FALSE`

`W1_READY_FOR_LIVE = FALSE`

A future W2 must be a separately preregistered hypothesis. It should not mine
new thresholds from W1 VALIDATION and then reuse that period as evidence.
