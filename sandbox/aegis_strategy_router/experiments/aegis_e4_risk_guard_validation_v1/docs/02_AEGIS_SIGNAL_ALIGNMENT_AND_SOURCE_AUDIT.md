# 02_AEGIS_SIGNAL_ALIGNMENT_AND_SOURCE_AUDIT.md

> **Experiment**: `AEGIS_E4_RISK_GUARD_VALIDATION_V1`  
> **Topic**: Aegis Live Signal Reconciliation, Trade Matching, and Temporal Alignment

---

## 1. Audit Summary

The canonical Aegis live trade dataset from `live_entry_classification.csv` was reconciled against TS live log files (`turbo_trades_*.jsonl`).

- **Total Live Signals in Scope**: 715 trades.
- **Matched Open Events**: 718 events (100% matching, 3 multi-leg adjustments reconciled).
- **Missing Open Events**: 0.
- **Excluded Due to Timestamp Integrity**: 3 trades (trade execution preceded signal log by >60s).
- **Validated Clean Trades**: 715 trades across 11 symbols.

---

## 2. Symbol and Directional Distribution

| Symbol | Total Trades | LONG | SHORT |
|---|---:|---:|---:|
| **ADAUSDT** | 87 | 25 | 62 |
| **AVAXUSDT** | 108 | 36 | 72 |
| **BNBUSDT** | 32 | 11 | 21 |
| **BTCUSDT** | 27 | 9 | 18 |
| **DOGEUSDT** | 64 | 22 | 42 |
| **ETHUSDT** | 54 | 19 | 35 |
| **LINKUSDT** | 49 | 15 | 34 |
| **LTCUSDT** | 49 | 17 | 32 |
| **SOLUSDT** | 83 | 29 | 54 |
| **SUIUSDT** | 90 | 28 | 62 |
| **XRPUSDT** | 72 | 17 | 55 |
| **TOTAL** | **715** | **228 (31.9%)** | **487 (68.1%)** |

---

## 3. Split Distribution of Reconstructed Trades

| Split | Time Window | Trade Count | Role |
|---|---|---:|---|
| **DISCOVERY** | 2026-05-07T00:00:00Z — 2026-07-01T00:00:00Z | 437 | Historical context |
| **EMBARGO_1** | 2026-07-01T00:00:00Z — 2026-07-01T01:00:00Z | 1 | Leakage buffer |
| **CALIBRATION** | 2026-07-01T01:00:00Z — 2026-07-15T00:00:00Z | 67 | Threshold calibration |
| **EMBARGO_2** | 2026-07-15T00:00:00Z — 2026-07-15T01:00:00Z | 0 | Leakage buffer |
| **VALIDATION** | 2026-07-15T01:00:00Z — 2026-08-01T00:00:00Z | 108 | Confirmatory testing |
| **FINAL_HOLDOUT** | 2026-08-01T01:00:00Z — 2026-08-15T00:00:00Z | 102 | Sealed (untouched) |
