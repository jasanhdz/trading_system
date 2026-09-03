# MVDR V1 M1.1 Level-To-Level State Report

## Status

`MVDR_V1_M1_1_LEVEL_TO_LEVEL_STATE_READY_FOR_REVIEW`

Interpretation: **LEVEL_TO_LEVEL_STATE_INSUFFICIENT**.

Labels: `RETROSPECTIVE_DISCOVERY_ONLY`, `POST_M1_HYPOTHESIS`, `NO_VALIDATION_AUTHORITY`, `NO_PROFITABILITY_AUTHORITY`.

This phase tests a post-M1 explanation of manual entry decisions. It does not reinterpret M1 as positive, test profitability, study exits, or authorize automation.

## Authority And Causality

- M1 authority: `522ebcd43711ca1bd75770384f5a6026fa46e467`.
- The complete M1 artifact manifest was hash-verified before replay.
- SUIUSDT 1m SHA256: `384019dff980f49f49b28e7ba37655b0f9c599124fd022141cce057a73102b08`.
- BTCUSDT 1m SHA256: `139d8fa801f678c2aa30d0a2693b468b824d6fcc195968c66e043ba4d4b15efb`.
- The same 1,680 decision minutes and nine frozen entries are reused.
- Only completed 1m bars are visible. Partial 3m/5m/15m candles use completed constituent minutes.
- Exits, outcomes, PnL, future returns, MFE and MAE are unused.

The three M1 S/R families remain frozen: recent swings, repeated-touch clusters, and 5m/15m extrema. The combined level chooses the valid `support < resistance` pair with minimum total distance; ties use fixed family order `swing`, `cluster`, `mtf_extrema` independently for support and resistance.

## Frozen State And Safety Grammar

Only these action families exist:

1. `SHORT_TO_SUPPORT`
2. `LONG_FROM_SUPPORT`
3. `LONG_TO_RESISTANCE`
4. `SHORT_FROM_RESISTANCE`
5. `NO_TRADE`

State scores use equal-weight components fixed before the final run. State requires score at least `0.45` and top-two margin at least `0.05`. Safety is an allow/block veto, never a direction changer. It requires aggregate safety at least `0.55` and can block ambiguous levels, insufficient room, inconsistent destination momentum, unconfirmed origin reaction, strongly opposing BTC (at least 30 bps over three completed minutes), or volatility chaos.

Matched controls use the same session, manual-side-compatible state family, distance bin and volatility tercile. When exact matching is sparse, the deterministic relaxation order is: ignore volatility tercile, then retain session/side/family and match nearest approach score. No outcome enters matching.

## Primary Results

| Metric | M1.1 level state | Frozen M1 full | Frozen M1 S/R-only |
|---|---:|---:|---:|
| Top 1% | 0/9 | 0/9 | 0/9 |
| Top 5% | 2/9 | 3/9 | 0/9 |
| Top 10% | 2/9 | 3/9 | 1/9 |
| Median percentile rank | 63.4% | 73.6% | 77.5% |
| Correct side | 1/9 | 7/9 | 4/9 |
| Capture +/-1m | 0/9 | 7/9 | 6/9 |
| Capture +/-3m | 1/9 | 7/9 | 6/9 |
| Capture +/-6m | 1/9 | 7/9 | 6/9 |
| Total emitted across folds | 194 | 16,143 | 12,601 |

M1.1 greatly reduces emissions, but loses the manual decisions and direction. Selectivity without retention is not replication.

Strict leave-session-out is weak because only two sessions exist: top-5%=2/9, correct-side=1/9, capture +/-6m=2/9, median percentile=45.6%, and 1,127 emissions.

## Safety Veto

| View | Emitted signals | Capture +/-6m |
|---|---:|---:|
| Level state without safety | 405 | 1/9 |
| Level state with safety | 194 | 1/9 |

Safety reduces emissions by 52.1% while retaining the already-low aggregate capture count. It does not explain manual selectivity because eight of nine anchors remain uncaptured and most anchors are vetoed or classified `OTHER`.

## Q1. How Many Manual Entries Look Like Destination Trades?

Model inference: **3/9**.

- `LONG_TO_RESISTANCE`: M1-01 and M1-02.
- `SHORT_TO_SUPPORT`: M1-05.

These are model-inferred states, not trader annotations.

## Q2. How Many Look Like Origin/Reaction Trades?

Model inference: **0/9**. Six anchors are `NO_TRADE/OTHER` because state confidence or margin is insufficient.

Manual annotation is separate: only M1-09 has an explicit trader explanation, `TOWARD_SUPPORT`. The other eight remain unannotated.

## Q3. Does Toward-Support/Resistance Explain M1's Poorly Ranked Trades?

Only sporadically. M1-01 improves from rank 2170/2880 in M1 to 527/1440 and is inferred `TOWARD_RESISTANCE`, with the correct potential side, but safety vetoes it. M1-03 improves modestly from approximately the 46th percentile to 58th percentile but remains `OTHER`. There is no broad improvement.

M1-02 was already strong in M1 (123/2880) but M1.1 infers `TOWARD_RESISTANCE/LONG`, opposite the manual SHORT, and degrades to 990/1440.

## Q4. Does From-Support/Resistance Improve Other Trades?

No. No manual anchor is confidently classified into either origin/reaction family. Reaction states therefore cannot explain the nine decisions under the fixed grammar.

## Q5. Does Distance To Level Matter?

It contains ranking information but is insufficient. `LEVEL_ONLY` reaches median percentile 63.7% and top-5%=2/9, nearly the full state's 63.4% and 2/9. Distance does not recover direction or capture.

## Q6. Does Approach Velocity Matter?

Not positively in aggregate. Adding approach to level lowers median percentile from 63.7% to 56.6% and top-5% from 2/9 to 1/9. The level-state representation detects 78 minutes where a proximity rule would buy support while price is classified as traveling toward it, and 44 symmetric resistance cases, but that distinction does not replicate the anchors.

## Q7. Does Momentum Add Selectivity?

No robust improvement. `LEVEL+APPROACH+MOMENTUM` emits 240 signals versus 241 without momentum, has the same 1/9 top-5%, and captures 0/9 at +/-6m.

## Q8. Does Reaction Add Selectivity?

It reduces emissions from 240 to 209 and raises top-5% from 1/9 to 2/9, but median percentile falls to 51.8%, correct-side remains 1/9, and capture remains 0/9 at +/-6m. This is insufficient.

## Q9. How Many Signals Per Day Remain?

Across nine LOTO folds the system emits 194 signals, 21.6 per held-out trade/fold. Full-day folds emit 19-49 signals; the short second-session folds emit zero. This is a 98.8% reduction from M1's 16,143 fold emissions, but still does not preserve anchors.

## Q10. How Many Anchors Reach Top 1%, 5%, And 10%?

- Top 1%: **0/9**.
- Top 5%: **2/9**.
- Top 10%: **2/9**.

## Q11. Does Correct Side Improve Over 7/9?

No. It falls to **1/9**. Six anchors are `OTHER`, and two of the three directional states oppose the manual side.

## Q12. Does Median Rank Improve Over M1?

No. Median percentile rank falls from M1's 73.6% to 63.4%.

## Q13. Does It Beat M1 S/R-Only?

No. M1 S/R-only has a higher median percentile rank (77.5% versus 63.4%) and better correct-side rate (4/9 versus 1/9). M1.1 has better matched-hard-negative percentile (85% versus the reconstructed frozen M1-full baseline's 70%) and far fewer emissions, but this does not offset failure on anchors.

## Q14. Do Label/State Shuffles Reproduce The Result?

No shuffle reaches the real-label top-5% count of 2/9; label and state shuffles range from 0/9 to 1/9. However state-shuffle median ranking reaches as high as 64.1%, similar to the real 63.4%. There is no strong real-label effect to validate, and controls do not change the negative conclusion.

## Q15. Does The Mathematics Explain The Manual Rule?

Not on the single annotated case. M1-09 is manually annotated `TOWARD_SUPPORT`, but the model returns `OTHER/NO_TRADE`; therefore state consistency is 0/1 for available manual annotations.

The grammar can mathematically represent the sentence using distance reduction, velocity, path efficiency, momentum, reaction and remaining room. It does not identify the trader's explicitly described example under the frozen thresholds.

## Additional Safety Question

Safety explains part of emission reduction but not the missing human selectivity. It halves emitted candidates without reducing the already-low aggregate capture count, yet does not retain the annotated anchor and does not restore correct direction. The evidence does not support the claim that safety alone was the missing M1 component.

## Criterion

`LEVEL_STATE_REPLICATION_IMPROVEMENT=true` required all conditions:

| Requirement | Result | Pass |
|---|---:|---|
| Top-5% anchors >=6/9 | 2/9 | No |
| Correct side >=8/9 | 1/9 | No |
| Median percentile beats M1 full | 63.4% vs 73.6% | No |
| Emissions reduced >=90% | 98.8% | Yes |
| Matched discrimination beats M1 full | 85% vs 70% | Yes |
| Shuffles do not reproduce top-5 count | Maximum 1/9 | Yes |

Because three mandatory conditions fail, the flag is false.

## Flags

| Flag | Value |
|---|---|
| CAUSAL_LEVEL_STATE_RECONSTRUCTION_COMPLETE | `true` |
| MANUAL_STATE_ANNOTATION_AVAILABLE | `true` |
| LEVEL_STATE_REPLICATION_IMPROVEMENT | `false` |
| PROSPECTIVE_CAPTURE_RESEARCH_JUSTIFIED | `false` |
| FULL_AUTOMATION_RESEARCH_JUSTIFIED | `false` |

## Final Decision

**LEVEL_TO_LEVEL_STATE_INSUFFICIENT**

Do not add more retrospective rules to these nine trades. The scientifically appropriate next action is to collect new manual decisions prospectively, not to optimize this sample.

No exits, economics, leverage, orders, production code or other AEGIS programs were touched.
