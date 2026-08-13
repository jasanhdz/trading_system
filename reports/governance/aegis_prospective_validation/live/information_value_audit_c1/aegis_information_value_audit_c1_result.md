# Aegis Information Value Audit C1 - Result

## Verdict

`C1_AVAILABLE_INFORMATION_HAS_NO_STABLE_INCREMENTAL_ECONOMIC_EDGE`

C1 evaluated five frozen feature candidates against the same PRICE_STATE
baseline at 60- and 240-minute horizons, separately for LONG and SHORT. No
candidate passed the preregistered incremental-information gate. C1 does not
authorize C2, Shadow or Live.

## Source Availability

| Family | State | Interpretation |
| --- | --- | --- |
| Price state | Available | Complete causal history in the B2 panel |
| Flow activity | Available | Taker-buy and volume data from public klines |
| Derivatives carry | Available | Mark/spot basis and funding archives |
| Cross-market | Available | Causal BTC beta, breadth and altcoin context |
| Calendar | Available | Causal timestamp controls |
| Open interest | Insufficient | Recent collector exists; required aligned history does not |
| Liquidations | Unavailable | No authoritative historical event archive |
| Order book | Unavailable | No historical point-in-time snapshots |
| News | Unavailable | No timestamped point-in-time corpus |

Unavailable sources were not represented by zeros, proxies or retrospective
values.

## Main Findings

### Cross-Market Context

Cross-market context produced the clearest ranking lift. At 60 minutes,
grouped Spearman increased from `0.0256` to `0.0422` in validation and from
`0.0207` to `0.0532` in pseudo-forward. At 240 minutes it increased from
`0.0118` to `0.0324` and from `0.0086` to `0.0415`.

This is useful relative-order information, but it failed the absolute `0.05`
gate in validation and all selected populations remained negative after 14
bps costs. It also degraded barrier calibration in pseudo-forward.

### Flow Activity

Taker flow slightly improved barrier metrics in validation at both horizons,
but the improvement disappeared or reversed in pseudo-forward. Ranking and MAE
lifts were negligible. This public-kline flow representation is not a stable
incremental directional source.

### Derivatives Carry

Basis and funding produced small isolated improvements in pseudo-forward
barrier metrics, but failed validation, ranking, MAE and economics. Funding at
its archive frequency appears too sparse to identify short-horizon entries on
its own.

### Calendar Controls

Calendar controls improved 240-minute MAE Spearman by approximately `0.038` in
both required partitions. They did not improve direction, barrier quality or
net expectancy. Time-of-day helps estimate path adversity, not profitable
direction.

### All Available Families

Combining every non-calendar family improved residual ranking, reaching
`0.0534` in 60-minute pseudo-forward, but validation stayed below the gate,
barrier quality worsened and selected expectancy remained negative. More
features increased relative ordering without creating economic edge.

## Economics

Every candidate's calibration-q90 selections had negative primary and stress
expectancy in validation and pseudo-forward. Samples were ample; the failure
was not caused by a low event count. Day-cluster bootstrap lower bounds were
not positive.

The experiment therefore distinguishes statistical information from trading
utility: cross-market and calendar data contain information, but the available
families do not contain sufficient stable directional content to overcome
costs.

## Recommended Next Step

Do not create another model from the same inputs. Build a point-in-time data
program before C2:

1. continuously archive open-interest changes for all 11 symbols;
2. capture forced-liquidation events with exchange timestamps and side;
3. capture bounded depth imbalance and spread snapshots;
4. retain finer aggregate-trade imbalance rather than only candle summaries;
5. create immutable daily manifests and completeness checks;
6. evaluate each new family singly against this frozen C1 baseline;
7. permit interactions only after one family transfers independently.

Historical third-party acquisition may shorten the wait only if provenance,
coverage and point-in-time semantics can be verified. It must not be mixed
silently with live-collected data.

## Reproducibility And Safety

- Result SHA-256: `261f03a857aad78b7ffc26f447ce3add631ea693729923c2baf2d2949368a385`
- Repeated run reproduced the result hash exactly.
- Focused A1/A2/B1/B2/C1 tests: `27 passed`.
- Full unit suite: `791 passed, 5 failed`. The five failures are the unchanged
  historical branch-authority checks requiring
  `feature/aegis-ts-clean-rebuild`; C1 is isolated on
  `work/entry-quality-evidence-20260726`.
- Exchange calls and mutations: `0`.
- Runtime, PM2, Live, Shadow and TypeScript changes: `NONE`.
