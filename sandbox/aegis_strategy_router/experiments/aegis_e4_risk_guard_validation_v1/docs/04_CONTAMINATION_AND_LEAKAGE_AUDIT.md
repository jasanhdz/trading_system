# 04_CONTAMINATION_AND_LEAKAGE_AUDIT.md

> **Experiment**: `AEGIS_E4_RISK_GUARD_VALIDATION_V1`  
> **Topic**: Temporal Integrity, Point-in-Time Features, and Embargo Audit

---

## 1. Leakage & Overlap Audit

- **Model Training Isolation**: E4 development models were trained on historical 2023–2024 dataset with zero visibility into 2026 market dynamics.
- **Point-in-Time Scoring**: For every live Aegis signal in 2026, E4 features were computed strictly using 1m/5m/15m/1h/4h candle data available at or before `signal_time`.
- **Lookahead Audit**: No candle timestamp equal to or greater than `signal_time` was accessible to feature calculation.
- **Embargo Enforcement**:
  - `EMBARGO_1` (1 hour between Discovery & Calibration): Enforced.
  - `EMBARGO_2` (1 hour between Calibration & Validation): Enforced.
  - `EMBARGO_3` (1 hour between Validation & Final Holdout): Enforced.

---

## 2. Final Holdout Status

`FINAL_HOLDOUT` (2026-08-01 to 2026-08-15, 102 trades) remained strictly sealed and was not unblinded during this validation cycle.
