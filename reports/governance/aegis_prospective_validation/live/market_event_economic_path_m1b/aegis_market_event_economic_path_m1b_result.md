# Aegis Market-Event Economic Path M1B - Retrospective Result

## Verdict

`M1B_RETROSPECTIVE_EDGE_NOT_DEMONSTRATED`

- `M1B_READY_FOR_FORWARD_COLLECTION=true`
- `M1B_READY_FOR_SHADOW=false`
- `M1B_READY_FOR_LIVE=false`
- Complete validation gates passed: `0`
- Runtime, Shadow and Live changes: `NONE`
- Authenticated exchange requests and mutations: `0`

M1B improved candidate ranking and reduced adverse excursion in one LONG
population, but neither evaluated selector produced positive net validation
expectancy. It must not alter the running trading system.

## Frozen Method

- Implementation commit: `f1af935abbc3dfd88a361c6f9e7a16d5f5177907`
- Configuration SHA-256:
  `842a66bd9e32a30745ced4108070a02d457acb053c1a06e45a076f2627c00124`
- Universe: the 11 canonical symbols
- Public source period: 2024-01 through 2026-07
- New checksum-verified archives: 341 funding and 341 mark-price archives
- Base inputs: checksum-verified M1A Spot and USD-M Futures 1-minute archives
- Feature schema: `aegis-m1b-economic-path-features-v1`, 23 ordered causal features
- Models: L2 logistic probability, calibration-only Platt scaling, q90 MAE
  gradient boosting and Huber net-utility gradient boosting
- M1A outcome: worst of two 1-minute intrabar TypeScript protection replays
- V21 outcome: frozen current-TypeScript 5-minute replay
- Round-trip M1A costs: 5 bps fee plus 2 bps slippage per side, with realized
  funding through the replay exit
- Train: 2024-01 through 2025-03
- Calibration: 2025-04 through 2025-09
- Retrospective validation: 2025-10 through 2026-07

The M1A replay is parameter-equivalent, not exact runtime equivalence, because
ATR and protection updates use 1-minute bars rather than the runtime's 5-minute
bars. No result from this contaminated retrospective period has promotion
authority.

## Data Result

The completed dataset contains 12,138 causal rows:

- Compression breakout LONG: 5,726
- Spot/Futures dislocation LONG: 5,940
- Extreme reversal SHORT: 472

Missing official mark-price minutes were not imputed. Only affected feature
rows were rejected. Futures price paths remained unchanged.

Extreme reversal SHORT could not be modeled under the preregistered temporal
protocol: its frozen source starts in August 2025, leaving zero training events
before April 2025. Its status is `INSUFFICIENT_PREDEFINED_HISTORY`. Temporal
boundaries were not moved after discovering this limitation.

## Compression Breakout LONG

Partition counts were 2,913 train, 1,281 calibration and 1,525 validation.
The frozen policy selected 111 validation events.

| Metric | Unfiltered | Selected | Matched control |
|---|---:|---:|---:|
| Net expectancy | -0.20246% | -0.02941% | -0.31455% |
| Profit factor | 0.4747 | 0.8670 | 0.2792 |
| Win rate | 56.52% | 64.86% | 44.14% |
| Mean MAE | 0.69954% | 0.52821% | 0.70893% |
| Mean MFE | 0.52708% | 0.56389% | 0.47236% |

The selector reduced MAE, improved win rate and outperformed both unfiltered
candidates and its symbol/regime/time-matched control. It still lost after
costs. The day-block bootstrap 95% expectancy interval was approximately
[-0.15258%, 0.07239%], profit-factor lower bound was 0.5064, and only one of
three temporal thirds was positive. This is useful ranking information, not an
operable edge.

## Spot/Futures Dislocation LONG

Partition counts were 1,145 train, 472 calibration and 4,320 validation. The
frozen policy selected 471 validation events.

| Metric | Unfiltered | Selected | Matched control |
|---|---:|---:|---:|
| Net expectancy | -0.17928% | -0.14469% | -0.18258% |
| Profit factor | 0.5444 | 0.6110 | 0.5383 |
| Win rate | 64.95% | 62.42% | 63.06% |
| Mean MAE | 0.75189% | 0.70069% | 0.77720% |
| Mean MFE | 0.59551% | 0.58926% | 0.58609% |

All three temporal thirds were negative. The day-block bootstrap 95% interval
was approximately [-0.24008%, -0.04613%]. LINK represented 42.68% of selected
events, violating the frozen concentration gate. This family is decisively not
ready.

## Interpretation

The richer causal context was not useless: it identified a substantially less
bad compression-breakout subset and lower-MAE events. It did not uncover enough
gross information to overcome losses and execution costs. Removing fees would
not repair the statistical instability or the negative dislocation result, and
the experiment is not permission to mine new thresholds on the opened period.

The correct next step is passive fresh-forward collection under the frozen
schema. It may test whether compression ranking transfers to unseen data, but
it must remain non-authoritative until at least 30 new days and a separate
review. The SHORT hypothesis requires a new preregistered experiment with
genuine pre-April-2025 source history; moving M1B's split would be retrospective
tuning.

## Validation

- Focused M1B and protection tests: 13 passed
- Full Python unit regression: 760 passed, 5 failed
- The five failures are pre-existing branch-authority assertions requiring the
  literal branch `feature/aegis-ts-clean-rebuild`; the current research branch
  is `work/entry-quality-evidence-20260726`
- Python compilation: passed
- Git whitespace validation: passed
- Model serialization reload: exact for both trained populations
- `black --check`: unavailable in practice; it hung on the three changed files
  and was terminated without modifying them
- TypeScript repository and runtime: unchanged

## Private Evidence

- Archive manifest SHA-256:
  `1cc559055937f3d2432f0559a6badda6865495fdfd26f52f3f02c0943836f92b`
- Dataset SHA-256:
  `957b0d9903cbfbc930d53827dc349ea2e0ac5df025961e23fc5c4198cb6956bf`
- Private result SHA-256:
  `2223bcf4d9ee36816881528af022ad4d792fb2e59053c6f45f85e423c77fec3d`
- Compression model SHA-256:
  `98b437959d506755e9a320b0383799f478707ce8b22458de1adabd66260dfe76`
- Dislocation model SHA-256:
  `625652936d4c07a859b9c3d898d3d1a176bab7fb9808f92c2833787e368b0628`

Private datasets, reports and research models have permissions `0600`; archive
files are read-only. No credentials, private exchange responses or runtime
journals are present.
