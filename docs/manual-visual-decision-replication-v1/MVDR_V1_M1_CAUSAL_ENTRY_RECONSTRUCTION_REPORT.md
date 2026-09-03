# MVDR V1 M1 Causal Entry Reconstruction Report

## Status

`MVDR_V1_M1_CAUSAL_ENTRY_RECONSTRUCTION_READY_FOR_REVIEW`

Interpretation: **NO_REPLICATION_SIGNAL**.

This is retrospective replication discovery only. It is not evidence of profitability and does not authorize automated trading, exit research, economic backtesting, or production changes.

## Identity And Scope

Nine manual SUIUSDT entries were matched uniquely by side, entry price, exit price, and approximate date in `backtest_results/real_trade_analysis.json`.

- Source SHA256: `6f7ce034ce790b3a806bcb1c0bd2c0d424700ac5eda659d0474f93edcbc9e463`
- Local IDs: JSON records 1, 3, 5, 9, 10, 13, 14, 15, and 16.
- Records 15 and 16 are independently corroborated by SUIUSDT manual-position runtime logs.
- The source is a derived local trade-history artifact, not a raw exchange export. It has no Binance order IDs and naive timestamps; UTC is supported by runtime logs and the linked `hour_utc` analysis.
- Seven records belonging to other strategies are excluded. Their entry windows and all manual anchors receive a +/-6 minute negative-label blackout.

`TRADE_IDENTITY_COMPLETE=true` means the supplied nine anchors have unique local matches. It does not claim raw exchange-order provenance.

## Market Replay

Official Binance USD-M Futures 1m klines were downloaded for SUIUSDT and BTCUSDT from `2026-08-24T20:00:00Z` through `2026-08-26T04:00:00Z`.

| Symbol | Rows | Gaps | Duplicates | SHA256 |
|---|---:|---:|---:|---|
| SUIUSDT | 1,920 | 0 | 0 | `384019dff980f49f49b28e7ba37655b0f9c599124fd022141cce057a73102b08` |
| BTCUSDT | 1,920 | 0 | 0 | `139d8fa801f678c2aa30d0a2693b468b824d6fcc195968c66e043ba4d4b15efb` |

At decision time T, only 1m bars with `open_at + 60s <= T` are available. Open 3m/5m/15m buckets are partial aggregates of completed 1m bars. No future high, low, close, volume, BTC observation, exit, outcome, or manual timestamp is a feature.

The candidate generator is label-blind. Labels and exclusion windows are applied afterward.

The fixed hard-negative gate, defined before the final run, admits structurally difficult non-selections when any of these conditions holds: nearest side-relevant automatic level within 25 bps, side-relevant wick rejection at least 0.40, or absolute 3-candle 3m displacement at least 30 bps. These are labeled `NOT_SELECTED_BY_TRADER`, never `BAD_TRADE`.

## Structural Grammar

The frozen automatic S/R view reports all three causal families without retrospective family selection:

1. Recent swing high/low zones.
2. Repeated-touch price clusters.
3. 5m/15m local extrema.

No recorded manual levels exist. Therefore `MANUAL_SR_ORACLE` is unavailable and `MANUAL_SR_REQUIRED` is not computable.

The visual frame contains candle geometry, 2/3/5/8-candle sequences, path efficiency, displacement, traveled path, compression/expansion, rejection, volume/taker ratio, 3m/5m/15m structure, automatic S/R relations, and BTC context. Indicators and future outcomes are not used.

The primary fixed model family is sparse logistic regression with training-fold-only normalization. No tree ensemble, boosting, deep learning, outcome model, or post-result feature mining is used.

## Q1. What Was The Trader Seeing?

The entry-time dossiers show heterogeneous geometry rather than one repeated setup:

| Trade | Side | Main causal description at entry | LOTO rank |
|---|---|---|---:|
| M1-01 | LONG | Low-volume 3m compression (`range_ratio=0.19`), weak path efficiency (`0.15`), cluster support 5 bps away; 5m displacement still negative. | 2170/2880 |
| M1-02 | SHORT | Strong 3m/5m/15m upside impulse (`3m=+1.83%`, efficiency `1.00`), volume `4.0x`, resistance about 30 bps away; fade hypothesis. | 123/2880 |
| M1-03 | SHORT | Choppy/flat 3m displacement (`+0.02%`, efficiency `0.07`), compressed range, swing resistance 14 bps away. | 1553/2880 |
| M1-04 | SHORT | Efficient downside impulse (`-0.83%`, efficiency `1.00`) aligned with BTC, near cluster support rather than resistance. | 129/2880 |
| M1-05 | LONG | Efficient upside impulse (`+0.96%`), strong lower-wick rejection (`0.89`), three bullish 3m bars, compressed range. | 65/2880 |
| M1-06 | LONG | Efficient upside sequence (`+0.80%`), three bullish 3m bars, resistance 15-20 bps away, compression. | 760/2880 |
| M1-07 | SHORT | Upside approach (`+0.28%` 3m, `+0.79%` 5m) into cluster resistance 5 bps away; lower-wick rather than upper-wick rejection. | 1312/2880 |
| M1-08 | LONG | Negative 5m/15m context, automatic swing/MTF support 3 bps away, low volume (`0.43x`). | 209/480 |
| M1-09 | SHORT | Negative 3m/5m displacement, low volume (`0.16x`), cluster level at current price and swing support 14 bps away. | 100/480 |

This table uses only the entry frame. Outcomes and exits did not participate.

## Q2. Which Characteristics Repeat?

- Compression is common: eight of nine entry frames have 3m short-window range ratio below 0.55.
- BTC and raw SUI 3m displacement share direction in all nine frames, but the BTC ablation does not show robust incremental value.
- Several entries are near at least one automatic structural level, but not consistently near the side-appropriate support/resistance.
- Five entries show highly efficient (`1.00`) short-window paths; the remaining four range from choppy to moderate efficiency.

These are descriptive regularities among nine selected trades, not learned entry rules.

## Q3. Which Characteristics Are Not Consistent?

- Direction of approach: manual LONG and SHORT entries occur after both rising and falling paths.
- Rejection geometry: strong side-consistent wick rejection appears clearly in only one anchor.
- Volume: entry frames range from `0.16x` to `4.0x` recent volume.
- Location: some SHORT entries occur near resistance, while another occurs near support after an impulse.
- Multi-timeframe direction is not uniform.

## Q4. Does S/R Explain An Important Part?

Partially. Removing S/R reduces median percentile rank from `73.6%` to `60.1%`. However S/R-only reaches `77.5%`, higher than the full model's median, while producing 12,601 emitted signals across folds and only 44.4% correct-side discrimination. Automatic S/R is informative but not selective enough.

## Q5. Does Candle Sequence Add Information Beyond S/R?

It changes the top tail: visual sequence obtains 3/9 top-5% anchors versus 0/9 for S/R-only, and full obtains 3/9. But median ranking for visual sequence alone is only `61.5%`, and the full system still does not beat S/R-only on median rank. Evidence is mixed and insufficient.

## Q6. Do 5m/15m Add Information?

No robust positive evidence. Removing MTF improves median percentile rank from `73.6%` to `76.8%` with unchanged capture and direction rates. With N=9 this is descriptive only.

## Q7. Does BTC Add Information?

No robust positive evidence. Removing BTC changes median percentile rank only from `73.6%` to `72.6%`, with identical capture, correct-side rate, and top-5% count.

## Q8-Q9. Can The Model Find The Entries Without Their Timestamps?

Not with adequate selectivity.

LOTO automatic causal S/R:

- Top 1%: 0/9.
- Top 5%: 3/9 (33.3%).
- Top 10%: 3/9 (33.3%).
- Median percentile rank: 73.6%.
- Capture within +/-1m, +/-3m, and +/-6m: 7/9.

The high capture rate is not meaningful by itself because the calibrated thresholds emit thousands of candidates.

Strict leave-session-out, with only two independent sessions:

- Top 5%: 2/9.
- Capture +/-6m: 5/9.
- Correct-side rate: 7/9.
- Median percentile rank: 85.2%.
- Total emitted signals across folds: 3,983.

The session-out result is explicitly weak because only two sessions exist.

## Q10. How Many Extra Entries Are Generated?

LOTO emits 16,143 signals across the nine held-out folds, or 1,793.7 per manual trade. Individual full-day folds emit roughly 2,188-2,430 signals; the shorter second-day folds emit 11-22. This fails the preregistered maximum of five signals/day by orders of magnitude.

The +/-6m replication precision proxy is `0.043%`. This is a selection proxy, not trade precision or profitability.

## Q11. Does It Replicate Direction?

Correct-side discrimination is 7/9 (`77.8%`), below the preregistered `80%`. The same rate appears in the strict session-out view.

## Q12. Does The Explanation Require Manual Levels?

Unknown. No recorded manual levels exist, so View A (`MANUAL_SR_ORACLE`) cannot be evaluated. View B (`AUTOMATIC_CAUSAL_SR`) is insufficiently selective. `MANUAL_SR_REQUIRED=null` rather than guessing.

## Q13. Does The Result Disappear Under Shuffled Labels?

Ten independently retrained label-shuffle controls preserve day and side counts. Their top-5% capture ranges from 0/9 to 1/9, versus 3/9 for real labels. However no real-label replication signal exists under the preregistered criterion, so there is no validated effect to claim as having disappeared. The controls do not rescue the result.

## Q14. Is Automation Research Justified?

No.

The preregistered criterion requires at least 70% top-5%, at least 80% correct side, median timing error at most six minutes, no more than five signals/day, and clear superiority over S/R-only and rejection-only.

| Criterion | Result | Pass |
|---|---:|---|
| Top-5% anchors | 33.3% | No |
| Correct side | 77.8% | No |
| Median timing error | 0.5m | Yes, but threshold is non-selective |
| Signals/day | Hundreds to thousands | No |
| Beats S/R-only and rejection-only | Does not beat S/R-only median rank | No |

## Flags

| Flag | Value |
|---|---|
| TRADE_IDENTITY_COMPLETE | `true` |
| CAUSAL_MARKET_REPLAY_COMPLETE | `true` |
| VISUAL_FEATURE_RECONSTRUCTION_COMPLETE | `true` |
| MANUAL_ENTRY_REPLICATION_SIGNAL_PRESENT | `false` |
| FULL_AUTOMATION_RESEARCH_JUSTIFIED | `false` |
| MANUAL_SR_REQUIRED | `null` (not computable) |

## Final Interpretation

**NO_REPLICATION_SIGNAL**

The fixed causal visual grammar recognizes some anchors, but it does not rank them consistently near the top and cannot reproduce the trader's selectivity. No thresholds or features were changed after observing this result.

No Phase M2, economic analysis, automated trading, orders, production changes, or Range modifications were performed.
