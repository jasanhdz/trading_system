# Aegis Compression Entry M1C - Retrospective Result

## Verdict

`M1C_CAUSAL_PULLBACK_EDGE_NOT_DEMONSTRATED`

- `M1C_READY_FOR_FORWARD_COLLECTION=false`
- `M1C_READY_FOR_SHADOW=false`
- `M1C_READY_FOR_LIVE=false`
- Primary validation gate: `FAIL`
- Runtime, Shadow and Live changes: `NONE`
- Exchange calls and mutations: `0`

The primary pullback-and-reclaim rule improved over unfiltered Compression LONG
and a matched random control, but was inferior to the prior M1B selector and
remained negative after both primary and optimistic costs. This exact rule does
not justify fresh-forward collection.

## Reproducible Scope

- Preregistration commit: `70c4c69`
- Implementation commit: `b6850d6`
- Configuration SHA-256:
  `ce5fdbf98afa29d46dcf7cced23bd5e1890ae73738895d2a87de9cff1735dc7e`
- Universe: 11 canonical symbols, LONG only
- Source population: frozen M1A Compression Breakout LONG
- Feature schema: 38 ordered causal features
- Entry variants: immediate and first pullback/reclaim within five completed
  one-minute candles
- Exit variants: current TypeScript 240-minute replay, 60-minute timebox and
  120-minute early lock
- Cost views: 0, 8, 14 and 20 bps round trip; 14 bps is primary
- Train/calibration/validation: unchanged from M1B
- Dataset rows: 31,120
- Immediate events: 5,727
- Pullback-confirmed events: 4,647

The immediate/current-TypeScript validation baseline exactly reconciled with
M1B at `-0.202459%` net expectancy. This establishes that the challenger was
compared against the same economic path rather than a rewritten baseline.

## Primary Result

`PULLBACK_RECLAIM__CURRENT_TS_240` had 2,368 train, 1,074 calibration and
1,198 validation candidates. The calibration-frozen selector retained 140
validation events.

| Metric | Immediate unfiltered | Pullback selected | Matched control | M1B selected |
|---|---:|---:|---:|---:|
| Net expectancy | -0.20246% | -0.11940% | -0.21335% | -0.02941% |
| Profit factor | 0.4747 | 0.6093 | 0.4451 | 0.8670 |
| Win rate | 56.52% | 57.86% | 53.57% | 64.86% |
| Mean MAE | 0.69954% | 0.56378% | 0.65345% | 0.52821% |
| Mean MFE | 0.52708% | 0.52963% | 0.52238% | 0.56389% |

The pullback rule reduced MAE relative to immediate entry and random control,
but did not improve enough to beat M1B. Its day-block bootstrap 95% expectancy
interval was approximately `[-0.23661%, 0.00468%]`, and the profit-factor lower
bound was `0.3738`.

The temporal thirds were approximately `-0.0333%`, `+0.0308%` and `-0.3538%`.
The last third dominates the failure and shows temporal deterioration rather
than a stable effect.

## Cost Attribution

| Cost | Selected expectancy | Profit factor |
|---|---:|---:|
| Zero-cost diagnostic | +0.02060% | 1.0823 |
| Optimistic 8 bps | -0.05940% | 0.7888 |
| Primary 14 bps | -0.11940% | 0.6093 |
| Stress 20 bps | -0.17940% | 0.4590 |

The gross signal was too small to survive even optimistic costs. Zero-cost
positivity is not sufficient and had no promotion authority.

## Regime Result

All direction regimes were negative. Transition contained 126 of 140 selected
events and returned `-0.10025%` per event. DOWN and UP had only 4 and 10 events
and were also negative.

Compressed volatility returned `-0.14710%`; normal volatility returned
`-0.08804%`. Expanding volatility returned `+0.05626%`, but had only six events
and excessive concentration. It is an underpowered observation, not permission
for a regime exception.

## Exit Diagnostics

The 60-minute timebox reduced MAE to `0.36015%`, but selected expectancy remained
`-0.14391%`, including `-0.00391%` even before costs. The pullback early-lock
variant returned `-0.17848%` and was also negative before costs. The immediate
early-lock variant produced no calibration policy under the frozen minimum;
its status is `CALIBRATION_POLICY_UNAVAILABLE`. No minimum was lowered.

These diagnostics do not support replacing the current exit merely to improve
reported MAE. Shorter exposure reduced excursion but also removed favorable
path opportunity.

## Interpretation

The M1C hypothesis was directionally reasonable but empirically insufficient.
A five-minute pullback/reclaim is common enough to preserve frequency, yet it
does not distinguish durable continuation from a temporary bounce. Adding
multi-timeframe price, flow and regime context improved ranking relative to
random, but did not create enough gross edge to pay realistic costs.

The next experiment should not tune the five-minute window, reclaim threshold
or regime exclusions on this opened validation period. A defensible next
direction requires a distinct mechanism and preregistration, such as measuring
post-breakout acceptance versus failed auction using duration above the level,
relative volume persistence and cross-sectional leadership. It must use a new
untouched evaluation period or blocked nested cross-validation and still clear
costs.

## Validation

- Focused M1C/M1B/protection tests: 17 passed
- Full Python unit regression: 764 passed, 5 failed
- The five failures are the pre-existing branch-authority assertions requiring
  `feature/aegis-ts-clean-rebuild`; this work remains on the isolated research
  branch `work/entry-quality-evidence-20260726`
- Python compilation: passed
- Git whitespace validation: passed
- Baseline reconciliation: exact
- Five trained artifacts: exact prediction reload
- One variant: calibration policy unavailable and retained as negative evidence
- Private artifact permissions: `0600`
- TypeScript repository and runtime: unchanged

## Private Evidence

- Dataset SHA-256:
  `e4e0cdae2bd20e6c60c4bde272d414da3f73b49582650c4e6ce426e6480f1c5b`
- Result SHA-256:
  `274a44e8b8bb872692ef59a7e2a293e0743ff0300c78be080a5fd3baa8404045`
- Regime thresholds SHA-256:
  `c4f7446d94d908d710ba0d75d87c91705b579469e7b001547561c8c3d1b8776a`
- Primary model SHA-256:
  `67c4094c6f8df71a879e690202f7100a13d91ca9e2ee4ee76d1d09a1d77bdf63`

No credentials, private exchange payloads or mutable runtime journals were
stored.
