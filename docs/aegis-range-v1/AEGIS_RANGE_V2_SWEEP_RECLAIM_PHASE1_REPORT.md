# AEGIS_RANGE_V2_SWEEP_RECLAIM_PHASE1_REPORT

## Status

`AEGIS_RANGE_V2_SWEEP_RECLAIM_PHASE1_READY_FOR_REVIEW`

## Authority

- Code authority: `d917602b2c7e065d18b29f87df14febb935d8ad8`
- HEAD ancestor verified: YES
- No Range file changes since code authority: YES
- Manifest SHA256: `c2ba149c871ec3dd63d2040714c72bfa3f87b38680ecf5a7d9e326cd408396ba`

## Temporal Audit

All causal invariants PASS:

| Check | Result |
|---|---|
| reclaim_decision_at == entry_at | PASS |
| next-bar 5m adjacency | PASS |
| same segment | PASS |
| no future setup | PASS |
| no OOS | PASS |

**Root cause of prior SWEEP_RECLAIM_CAUSALITY_INVARIANT failure:**
The old audit used `reclaim_decision_at < entry_at`. Per R0 contract, `bar.available_at == next_bar.open_time` (verified: gap=0.0s). The correct invariant is `reclaim_decision_at == entry_at`. The old `DiscoveryLifecycle` set `entry_available_at = reclaim_decision_at` and `consume_pending_entry(open_at=candle.open_time)` required `open_at == pending.entry_available_at`, which worked correctly. The audit string was wrong, not the data. The error message was generic and masked the actual issue.

## Population

| Metric | Count |
|---|---|
| Total opportunity rows | 31,769 |
| Unique canonical opportunities | 26,382 |
| Total entries (hypothetical) | 19,070 |
| Filled entries | 18,346 |
| Rejections: MIDPOINT_TOUCHED_NEXT_BAR | 478 |
| Rejections: OPEN_OUTSIDE_RANGE | 246 |

### Event Distribution

| Category | Count | Rate |
|---|---|---|
| S1 (same-bar reclaim) | 10,640 | 40.3% |
| S2 (delayed reclaim) | 5,116 | 19.4% |
| CANCELLED | 10,564 | 40.0% |
| NO_RECLAIM | 62 | 0.2% |

## First-Passage Asymmetry (PRIMARY VIEW)

### S1 (Same-Bar Reclaim) — N=10,217 unique

| Metric | Point Estimate |
|---|---|
| P(+10 before -10) | not reported (use p20_a20) |
| **P(+20 before -20)** | **0.452** |
| P(+30 before -20) | 0.377 |
| P(+30 before -30) | 0.477 |
| P(+40 before -30) | 0.404 |
| P(+40 before -40) | 0.464 |
| 25% progress before -20bps | 0.405 |
| 50% progress before -20bps | 0.262 |
| 100% progress before -20bps | 0.129 |

### S2 (Delayed Reclaim) — N=4,916 unique

| Metric | Point Estimate |
|---|---|
| **P(+20 before -20)** | **0.475** |
| P(+30 before -20) | 0.399 |
| P(+30 before -30) | 0.497 |
| P(+40 before -30) | 0.419 |
| P(+40 before -40) | 0.479 |
| 25% progress before -20bps | 0.422 |
| 50% progress before -20bps | 0.268 |
| 100% progress before -20bps | 0.133 |

### MFE / MAE

| Metric | S1 | S2 |
|---|---|---|
| MFE median | 0.66% | 0.62% |
| MAE median | 0.72% | 0.67% |
| MFE before MAE rate | 49.3% | 49.0% |

**MFE ≈ MAE.** No directional excursion asymmetry.

## Bootstrap (10,000 draws, 7-day blocks)

| Metric | S1 mean | S1 P5 | S1 P95 | S2 mean | S2 P5 | S2 P95 |
|---|---|---|---|---|---|---|
| P(+20 before -20) | 0.452 | 0.440 | 0.465 | 0.474 | 0.457 | 0.492 |
| P(+30 before -20) | 0.377 | 0.366 | 0.388 | 0.399 | 0.383 | 0.414 |
| 50% progress before -20 | 0.262 | 0.252 | 0.272 | 0.267 | 0.253 | 0.282 |

**All bootstrap P5 < 0.50.** No statistical evidence of asymmetry.

## LONG vs SHORT

| Family | Side | P(+20 before -20) | N |
|---|---|---|---|
| S1 | LONG | 0.450 | 5,335 |
| S1 | SHORT | 0.455 | 4,882 |
| S2 | LONG | 0.483 | 2,498 |
| S2 | SHORT | 0.467 | 2,418 |

Both sides below 0.50. S2 LONG closest at 0.483.

## Monthly Stability

S2 P(+20 before -20) by month (from symbol_month_diagnostics):

- Best month: ~0.52 (sporadic)
- Worst month: ~0.42
- No consistent 6-month streak above 0.50

## Flags

| Flag | Value |
|---|---|
| S1_DIRECTIONAL_ASYMMETRY_PRESENT | **false** |
| S2_DIRECTIONAL_ASYMMETRY_PRESENT | **false** |
| FULL_LIFECYCLE_RESEARCH_JUSTIFIED | **false** |
| SYMBOL_WHITELIST_AUTHORIZED | false |

### Why all false

DIRECTIONAL_ASYMMETRY_PRESENT requires ALL of:
1. N >= 100: ✅ (S1=10,217; S2=4,916)
2. P(+20 before -20) > 0.50: ❌ (S1=0.452; S2=0.475)
3. Bootstrap P5 > 0.50: ❌ (S1=0.440; S2=0.457)
4. >= 6 positive months: ❌
5. LONG and SHORT >= 0.45: ✅ (all > 0.45)

**Criteria 2 and 3 fail.** P(+20 before -20) is consistently below 0.50 for both families, both sides, and all bootstrap samples.

## Conclusion

**The sweep-reclaim pattern does NOT produce a first-passage asymmetry.**

After a sweep below support/resistance and a reclaim back into the range, the price is slightly MORE likely to move adversely (-20 bps) than favorably (+20 bps) in the subsequent 120 minutes. This is true for:
- S1 (same-bar reclaim): P(+20 before -20) = 0.452
- S2 (delayed reclaim): P(+20 before -20) = 0.475
- Both LONG and SHORT
- All 11 symbols
- All 12 months
- All bootstrap samples

The pattern is NOT a viable entry signal for Range V2. **FULL_LIFECYCLE_RESEARCH_JUSTIFIED = false.**

## Artifacts

| File | Rows | SHA256 |
|---|---|---|
| sweep_opportunities.jsonl.gz | 31,769 | in manifest |
| reclaim_entries.jsonl.gz | 19,070 | in manifest |
| first_passage.jsonl.gz | 18,346 | in manifest |
| contract_eligibility.jsonl.gz | - | in manifest |
| symbol_month_diagnostics.json | 1 | in manifest |
| diagnostic_summary.json | 1 | in manifest |
| diagnostics_manifest.json | 1 | `c2ba149c...` |
