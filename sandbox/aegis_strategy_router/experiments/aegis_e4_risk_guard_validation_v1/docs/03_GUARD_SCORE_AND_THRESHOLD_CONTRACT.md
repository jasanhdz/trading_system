# 03_GUARD_SCORE_AND_THRESHOLD_CONTRACT.md

> **Experiment**: `AEGIS_E4_RISK_GUARD_VALIDATION_V1`  
> **Topic**: Frozen Critic Scores and Calibration Decision Thresholds

---

## 1. Frozen Critic Score Definitions

E4 critic models produce out-of-fold probability estimates based on multi-timeframe feature states (5m, 15m, 1h, 4h), order flow, and BTC cross-market metrics:

1. **`e4_late_entry_score`**: Predicted probability that the trade is entering into an exhausted momentum wave (`target__late_entry_risk`).
2. **`e4_tail_risk_score`**: Predicted probability that Maximum Adverse Excursion exceeds 200 bps (`target__tail_risk`).
3. **`e4_dual_risk_score`**: Joint risk index combining late entry and tail risk.

---

## 2. Calibration on CALIBRATION Split (67 trades)

Thresholds were calibrated strictly on the `CALIBRATION` split (2026-07-01 to 2026-07-15) at the 70th percentile (Q70):

| Guard Head | Frozen Threshold | Calibration Quantile | Calibration Bad Rejection | Calibration Good Destruction |
|---|---|---|---:|---:|
| **Late Entry Guard** | `0.037696937` | Q70 (top 30% risk) | 23.1% | 34.1% |
| **Tail Risk Guard** | `0.452245221` | Q70 (top 30% risk) | 46.2% | 19.5% |
| **Dual Guard** | Late ≥ 0.0377 OR Tail ≥ 0.4522 | Disjunction | 53.8% | 43.9% |

All thresholds were locked into `thresholds_frozen_v1.json` before touching the `VALIDATION` split.
