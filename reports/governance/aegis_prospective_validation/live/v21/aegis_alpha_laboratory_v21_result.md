# Aegis Alpha Laboratory V21 Result

## Verdict

`V21_READY_FOR_MODELING = FALSE`

`V21_READY_FOR_SHADOW = FALSE`

`V21_READY_FOR_LIVE = FALSE`

No preregistered side/strategy pair passed every economic and stability gate.
No model was trained or exported. Live, Shadow, TypeScript, PM2, and exchange
state were not changed.

## Evidence

The deterministic runner read 90,442 hash-bound V14 rows grouped into 4,111
cross-sectional timestamps. It produced 3,094 causal candidate events after
the frozen 60-minute symbol/side/strategy spacing rule. Discovery, validation,
and final holdout boundaries were committed before the holdout was opened.

| Side | Strategy | Discovery | Validation | Holdout | Validation net | Holdout net | Holdout win | Profit factor | Mean MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LONG | Cross-sectional momentum | 546 | 257 | 194 | -0.2397% | -0.1261% | 53.1% | 0.59 | 0.728% |
| SHORT | Cross-sectional momentum | 602 | 281 | 207 | -0.2373% | -0.0822% | 62.3% | 0.74 | 0.778% |
| LONG | Extreme reversal | 289 | 108 | 111 | -0.2747% | -0.1610% | 65.8% | 0.34 | 0.736% |
| SHORT | Extreme reversal | 277 | 110 | 85 | -0.1440% | -0.0254% | 74.1% | 0.85 | 0.643% |
| LONG | Breakout flow/funding | 19 | 1 | 1 | -0.7810% | +0.2285% | 100.0% | N/A | 0.328% |
| SHORT | Breakout flow/funding | 5 | 1 | 0 | +0.2333% | N/A | N/A | N/A | N/A |

The breakout rows are far below the frozen sample minimums and do not support
an economic conclusion. Their isolated positive observations are not evidence
of edge.

## What V21 Established

### Momentum

Relative leaders and laggards with aligned volume and taker flow did better
than a matched random control in the final holdout, but both sides remained
negative after the frozen protection and cost contract. More selectivity did
not turn the rule profitable. Mean MAE also exceeded the 0.60% gate.

### Reversal

LONG reversal was negative and did not beat its random control. SHORT reversal
was the strongest diagnostic: it selected a 74.1% holdout win rate and greatly
outperformed the random control, but its assigned `LOCK_AT_5_ROE` exit still
lost 0.0254% per event and validation lost 0.1440% per event.

On the same SHORT holdout events, the current TypeScript protection replay
would have returned +0.0493% per event with profit factor 1.25. This comparison
was a preregistered control, not the V21 candidate policy. It suggests that
entry and exit compatibility deserves a new preregistered study. It does not
permit retroactively replacing the V21 exit, does not establish temporal
stability, and does not authorize model training or deployment.

### Funding And Basis

Funding history is available for all eleven symbols. Historical spot prices
and executable basis are absent from the bound database, so a true
delta-neutral carry return cannot be reconstructed. `FUNDING_BASIS_CARRY` is
classified `DATA_GAP`. No directional proxy, synthetic spot leg, or zero-cost
substitution was used.

## Comparison With Earlier Versions

- V15 simplified direction-specific feature contracts but retained negative
  post-V14 utility: -0.1897% LONG and -0.2170% SHORT.
- V17's safety-gated ranker produced 0/4 successful folds for both directions
  and negative utility in every evaluated fold.
- V20 tested five causal candle/flow opportunity families; none passed its
  viability gate.
- V21 removed model complexity and tested explicit economic hypotheses. The
  large momentum and reversal populations still failed validation and holdout
  economics under their frozen exits.

The repeated result narrows the problem: model complexity is not the current
bottleneck. The available features can identify some relatively better paths,
especially SHORT reversals, but the complete entry/exit/cost policy has not
demonstrated stable positive expectancy.

## Next Evidence

1. Preserve V21 and its one-shot holdout. Do not tune its thresholds or exit
   assignments and call the result V21.
2. Preregister a separate entry/exit compatibility experiment for SHORT
   reversal. Derive candidate exits only from discovery, choose at most one on
   validation, and evaluate once on evidence strictly after the V21 holdout.
3. Continue prospective collection of open interest, depth imbalance,
   aggregate trades, liquidations, and spot/basis history. These inputs must
   mature before supporting new claims.
4. Build models only after an underlying family shows positive validation and
   future-holdout economics. Use the first model as an abstention/path-risk
   filter, not as a source of fabricated edge.
5. Keep no-trade, random, V15, V17, V20, and the unmodeled family rule as
   controls. Preserve every failed experiment.

## Reproducibility And Safety

- Preregistration commit: `9899f23`
- Implementation commit: `55e263b`
- Opportunity dataset SHA-256:
  `d4261a1699e663adb0ac6f4d1daa1f9b7bada1ea5201cf7699316f13e699123e`
- Result SHA-256:
  `8fb414b10d12734d647d14dc54089b1a299c2ef626d4b26e03e1284c3765d052`
- Manifest SHA-256:
  `f7c3cf2f1d2f3a33c17fa6ab658cf399a4968fa6e7034fb3fa59c4f91be23147`
- Deterministic rerun: identical hashes
- Focused tests: 17 passed
- Full Python regression: 744 passed, 5 historical branch-authority failures
- Python compilation: passed
- Git whitespace validation: passed
- Model trained/exported: no/no
- Exchange calls/mutations: 0/0
- Live/Shadow changes: none/none

