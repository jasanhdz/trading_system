# MVDR V1 M1.2 Compact S/R Corridor Report

## Status

`MVDR_V1_M1_2_COMPACT_SR_CORRIDOR_GRAMMAR_READY_FOR_REVIEW`

Interpretation: **COMPACT_CORRIDOR_GRAMMAR_INSUFFICIENT**.

Labels: `POST_M1_1_HYPOTHESIS`, `RETROSPECTIVE_SANITY_ONLY`, `PROSPECTIVE_CONFIRMATION_REQUIRED`, `NO_PROFITABILITY_AUTHORITY`, `NO_AUTOMATION_AUTHORITY`.

The nine historical trades are a retrospective sanity check, not OOS evidence. No rules were changed after observing results.

## Authority And Frozen Grammar

- M1 authority: `522ebcd43711ca1bd75770384f5a6026fa46e467`.
- M1.1 authority: `9cc915424b9be5985a4a84b851c54c6d670f550d`.
- Every material M1 artifact passed SHA256 verification.
- The same SUIUSDT/BTCUSDT 1m observations and completed-bar convention were reused without downloading replacements.
- Primary system: deterministic rules only; no ML.

Zone width is the maximum of causal median 1m range and per-minute causal median 3m range. Level selection maximizes equal-weight respect evidence, then minimizes distance, then applies frozen family order. Respect has six equally weighted components: contacts, valid closes, penetration returns, post-touch movement, age, and break integrity.

Corridor clarity equally weights support respect, resistance respect, separation, containment, non-ambiguity and break integrity. A clear corridor requires both respect scores >=0.50, width >=25 bps, zero nearby contradictions and clarity >=0.60. Travel/reaction confirmation requires >=0.60. Destination room requires >=25 bps.

BTC only vetoes when at least two of 3m/5m/15m moves are strongly opposing relative to their causal prior 95th percentiles. A 1m range above its causal prior 95th percentile triggers the volatility veto.

## Retrospective Metrics

| Metric | M1.2 | M1 full | M1 S/R-only | M1.1 |
|---|---:|---:|---:|---:|
| Correct direction | 3/9 | 7/9 | 4/9 | 1/9 |
| Capture +/-6m | 7/9 | 7/9 | 6/9 | 1/9 |
| Top 5% | 1/9 | 3/9 | 0/9 | 2/9 |
| Median percentile rank | 24.3% | 73.6% | 77.5% | 63.4% |
| Signals | 485/80 by session | 16,143 fold sum | 12,601 fold sum | 194 fold sum |

The emission units differ: M1/M1.1 aggregate held-out folds, while M1.2 is one deterministic replay per session. M1.2 nevertheless emits far above the preregistered 20/day maximum.

## Q1-Q4. Detected Levels, Respect, Corridor And Position

| Trade | Support center (family/respect) | Resistance center (family/respect) | Clear | Position |
|---|---|---|---|---:|
| M1-01 | 0.790400 (MTF/0.831) | 0.805900 (MTF/0.930) | Yes | 0.490 |
| M1-02 | 0.795100 (MTF/0.990) | 0.827776 (cluster/0.417) | No | 0.924 |
| M1-03 | 0.796700 (swing/0.871) | 0.809800 (MTF/0.912) | Yes | 0.420 |
| M1-04 | 0.786207 (cluster/0.850) | 0.802725 (cluster/0.824) | No | 0.024 |
| M1-05 | 0.786239 (cluster/0.723) | 0.803400 (MTF/0.805) | Yes | 0.161 |
| M1-06 | 0.751200 (swing/0.849) | 0.777300 (MTF/0.850) | Yes | 0.268 |
| M1-07 | 0.743000 (MTF/0.731) | 0.773300 (MTF/0.910) | Yes | 0.749 |
| M1-08 | 0.753700 (swing/0.774) | 0.767600 (MTF/0.903) | No | 0.014 |
| M1-09 | 0.752300 (MTF/0.861) | 0.763000 (swing/0.923) | Yes | 0.748 |

Six of nine anchors have a clear corridor. M1-02 fails because resistance respect is only 0.417. M1-04 and M1-08 fail non-ambiguity despite individually plausible respect scores.

All exact zones, touch counts, break counts, penetration history and bounds are in `level_respect_diagnostics.jsonl.gz` and `anchor_dossiers.jsonl.gz`.

## Q5-Q7. Travel, Reaction, Direction And Safety

| Trade | Compact decision | Manual side | Match | Safety |
|---|---|---|---|---|
| M1-01 | SHORT_TO_SUPPORT | LONG | No | Allow |
| M1-02 | NO_TRADE | SHORT | No | Block: corridor, shock, ambiguity |
| M1-03 | LONG_TO_RESISTANCE | SHORT | No | Allow |
| M1-04 | NO_TRADE | SHORT | No | Block: corridor, ambiguity |
| M1-05 | LONG_FROM_SUPPORT | LONG | Yes | Allow |
| M1-06 | LONG_TO_RESISTANCE | LONG | Yes | Allow |
| M1-07 | LONG_TO_RESISTANCE | SHORT | No | Allow |
| M1-08 | NO_TRADE | LONG | No | Block: corridor, ambiguity |
| M1-09 | SHORT_TO_SUPPORT | SHORT | Yes | Allow |

Safety blocks three anchors and allows three wrong-direction decisions. It therefore does not reconstruct the human safety criterion adequately.

The known grammar authority M1-09 is correctly represented: position 0.748, support respect 0.861, resistance respect 0.923, corridor clarity 0.931, support travel confirmation 0.613, BTC neutral, and final action `SHORT_TO_SUPPORT`. This isolated success was not used for threshold fitting.

## Q8. Signals Per Day

- 2026-08-25: **485**.
- 2026-08-26 partial session: **80**.
- Mean: **282.5/session**.

The <=20/day selectivity criterion fails substantially. Capture +/-6m=7/9 is explained by dense signal generation rather than precise replication.

## Q9. Comparison With M1

M1.2 does not improve M1. Correct direction falls from 7/9 to 3/9, top-5% falls from 3/9 to 1/9, and median percentile falls from 73.6% to 24.3%. Capture +/-6m remains 7/9 but is non-selective.

## Q10. Comparison With M1.1

M1.2 improves correct direction (3/9 versus 1/9) and capture (7/9 versus 1/9), but worsens top-5% (1/9 versus 2/9) and median percentile (24.3% versus 63.4%). It does not clearly improve M1.1 under the frozen criterion.

## Q11. Time-Shift Controls

Both controls lose exact direction agreement:

- Real anchors: 3/9 correct direction, 1/9 top-5%, 7/9 capture +/-6m.
- +60m: 0/9 correct direction, 0/9 top-5%, 2/9 capture +/-6m.
- -60m: 0/7 correct direction, 1/7 top-5%, 2/7 capture +/-6m.

The controls are weaker, but they do not turn the inadequate real-anchor result into evidence of replication.

## Q12. Does The Compact Grammar Represent The Explanation Better?

Conceptually, yes: the implementation explicitly models nearest zones, respect, corridor position, travel, reaction, BTC veto and abstention. It also reconstructs the one explicit `SHORT_TO_SUPPORT` example.

Empirically across all nine anchors, no: direction, rank and selectivity remain inadequate. The explanation is representable but not supported as a general retrospective replication grammar on this sample.

## Q13. Prospective Recorder

`PROSPECTIVE_SHADOW_RECORDER_READY=true`.

`record_corridor_shadow.py` is a one-shot, schedulable, read-only recorder. It downloads only completed public market bars, computes the frozen grammar and appends one immutable daily JSONL record. It writes a daily SHA256, rejects duplicate/out-of-order timestamps, records `outcome_known=false`, and has no order path.

Prospective records live under `artifacts/m1_2_prospective_shadow/`. The prospective criterion remains `NOT_EVALUATED` until at least 30 new manual decisions across at least five days exist.

## Criterion And Flags

| Criterion | Result | Pass |
|---|---:|---|
| Correct direction >=7/9 | 3/9 | No |
| Top-5% >=4/9 | 1/9 | No |
| Capture +/-6m >=6/9 | 7/9 | Yes |
| Signals/day <=20 | 485 maximum | No |
| Clearly improves M1.1 | No | No |

```text
COMPACT_SR_GRAMMAR_IMPLEMENTED=true
LEVEL_RESPECT_RECONSTRUCTION_COMPLETE=true
COMPACT_GRAMMAR_RETROSPECTIVE_FIT=false
PROSPECTIVE_SHADOW_RECORDER_READY=true
PROSPECTIVE_REPLICATION_SIGNAL=NOT_EVALUATED
FULL_AUTOMATION_RESEARCH_JUSTIFIED=false
```

## Final Decision

**COMPACT_CORRIDOR_GRAMMAR_INSUFFICIENT**

Do not adjust this grammar against the same nine trades. No exits, leverage, economics, orders or production code were studied or changed.
