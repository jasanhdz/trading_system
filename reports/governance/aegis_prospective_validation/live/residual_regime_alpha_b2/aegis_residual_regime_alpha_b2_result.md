# Aegis Residual Regime Alpha B2 - Result

## Verdict

`B2_PATH_RISK_CONFIRMED_RESIDUAL_DIRECTIONAL_ALPHA_NOT_DEMONSTRATED`

B2 evaluated 5,523 independent four-hour events and approximately 121,500
symbol-side rows at each of the 60- and 240-minute horizons. Neither horizon
passed the preregistered combined gate. B2 does not authorize a forward
experiment, Shadow or Live.

## What Worked

Path-risk estimation transferred again. MAE Spearman correlations were
`0.360-0.386` in 60-minute validation, `0.334-0.348` in pseudo-forward,
`0.332-0.333` in 240-minute validation and `0.301-0.336` in pseudo-forward.
MFE correlations were similarly positive at `0.299-0.386`.

At 60 minutes, the combined candidates reached the favorable 42-bps barrier
before the adverse barrier more often than the full population:

- validation: `47.1%` selected versus `39.0%` population;
- pseudo-forward: `62.9%` selected versus `34.2%` population.

This is evidence that path quality and abstention are measurable. It is not
evidence of direction or positive economic expectancy.

## What Failed

### Regime Direction

Transparent momentum, reversal and relative-strength mechanisms were selected
only when positive in TRAIN and CALIBRATION for a causal regime. Their frozen
aggregate performance was negative in every required partition:

| Horizon | Partition | Events | Net expectancy | Profit factor |
| --- | --- | ---: | ---: | ---: |
| 60m | Validation | 770 | -0.1409% | 0.597 |
| 60m | Pseudo-forward | 948 | -0.0932% | 0.643 |
| 240m | Validation | 418 | -0.1551% | 0.761 |
| 240m | Pseudo-forward | 657 | -0.1393% | 0.761 |

Regime conditioning did not rescue directional momentum or reversal. Several
TRAIN winners survived CALIBRATION but failed later, showing temporal
instability rather than a universal regime edge.

### Residual Ranking

Removing causal BTC beta and the common altcoin factor improved grouped rank
correlation slightly relative to B1, but not enough:

- 60m: `0.0381` validation and `0.0338` pseudo-forward;
- 240m: `0.0331` validation and `0.0288` pseudo-forward.

All values are below the frozen `0.05` gate. Top-ranked rows beat the
deterministic random rank control on residual utility, but their raw net
expectancy remained negative. The ranker found a weak relative ordering, not a
tradeable absolute advantage.

### Combined Policy

At 60 minutes the policy selected 34 validation and 35 pseudo-forward events.
Net expectancy was `-0.0415%` and `-0.0202%`; bootstrap lower bounds were
negative and stress costs worsened both results.

At 240 minutes it selected only 14 and 15 events. Validation showed `+0.7522%`
per event, but the confidence interval crossed zero, the sample was far below
100 and pseudo-forward reversed to `-0.3953%`. This is an unstable small-sample
result, not evidence of edge.

## Interpretation

The principal problem is not model size. With the current causal inputs, the
system can estimate the shape and adversity of a future path more reliably
than it can select its direction or symbol. Risk prediction cannot create
directional alpha, and using it alone would select smoother losses as well as
smoother wins.

The next experiment should not add another committee over these same targets.
It should acquire genuinely different causal information and test whether it
adds incremental directional content before combination. Highest-priority
families are historical open-interest change, liquidation pressure, taker-flow
imbalance at finer resolution, cross-market breadth/correlation transitions
and funding/basis dislocations. Each family should be tested singly against
the frozen B2 residual target and path-risk control, with a new untouched
forward period reserved for promotion evidence.

## Reproducibility And Safety

- Result SHA-256: `853e2d046618b14b04b11e51c978ac10991277ae964206766f20cdb1785a3c33`
- 60m derived dataset SHA-256: `825a160258d743f5a8c187c4c27400412f6a53b9f258341f4fefe516546c2c2e`
- 240m derived dataset SHA-256: `1bb5cea187e40972686fbb86eb3ee1c1d24b3b4f320c059871107d0ebaec0453`
- Repeated evaluation reproduced all three hashes exactly.
- Focused A1/A2/B1/B2 tests: `21 passed`.
- Full unit suite: `785 passed, 5 failed`. The five failures are the unchanged
  historical branch-authority checks that require the literal branch
  `feature/aegis-ts-clean-rebuild`; B2 remains isolated on
  `work/entry-quality-evidence-20260726`.
- Exchange calls and mutations: `0`.
- Runtime, PM2, Live, Shadow and TypeScript changes: `NONE`.
