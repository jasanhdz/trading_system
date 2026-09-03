# Aegis Version And Signal Audit

## Source

The immutable audit CSV contains 718 historical Live entries from 2026-05-07
through 2026-08-14. Every row matched exactly one `OPEN` record from
`turbo_trades_*.jsonl`; symbol and side agreed and no OPEN event was missing.

The decision timestamp is encoded in each `trade_id` and corresponds to the
contemporaneous `ENTRY_POLICY_DECISION` event. Three records were excluded by
the frozen 60-second integrity gate because their later OPEN represented
reconciliation/restoration rather than a new contemporaneous signal. This left
715 eligible signals: 228 LONG and 487 SHORT.

## Version contract

Aegis changed during the historical interval, so there is no honest single
binary version for all rows. Each signal stores an immutable SHA256 of its
contemporaneous policy metadata, raw/final reason, score, votes and recorded
entry model version. There are 715 distinct policy snapshots. Recorded entry
model versions are `v020`, `aegis-prospective-shadow-candidate-v1`, and 219
older events where the field was absent.

This experiment treats those actual allowed entries as the frozen baseline. It
does not reconstruct rejected Aegis candidates, alter Aegis thresholds or use
Aegis side/confidence as inputs to the frozen Opportunity/Directional models.

`AEGIS_BASELINE_REPRODUCED = TRUE`

## Evidence boundary

- DISCOVERY: 437 signals.
- CALIBRATION: 67 signals.
- VALIDATION: 108 signals, all SHORT.
- FINAL_HOLDOUT: 102 feature-only signals, sealed.
- Technical embargo: 1 signal.

May-July outcomes were already seen by W11/W12. They remain diagnostic discovery
and cannot establish clean validation. August remains unopened.
