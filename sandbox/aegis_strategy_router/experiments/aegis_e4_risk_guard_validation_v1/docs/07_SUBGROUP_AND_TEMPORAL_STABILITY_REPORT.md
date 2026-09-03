# 07_SUBGROUP_AND_TEMPORAL_STABILITY_REPORT.md

> **Experiment**: `AEGIS_E4_RISK_GUARD_VALIDATION_V1`  
> **Topic**: Directional Subgroups, Multi-Symbol Consistency, and Temporal Stability

---

## 1. Directional Stability (LONG vs SHORT)

| Subgroup | Policy | Executed | Bad Rejection (%) | Good Destruction (%) | Net Expectancy (bps) | Profit Factor |
|---|---|---:|---:|---:|---:|---:|
| **LONG (33 signals)** | AEGIS_ONLY | 33 | 0.0% | 0.0% | +24.45 | 1.83 |
| | TAIL_RISK_GUARD | 30 | 16.7% | 3.7% | **+33.72** | **2.62** |
| **SHORT (75 signals)** | AEGIS_ONLY | 75 | 0.0% | 0.0% | +13.74 | 1.36 |
| | TAIL_RISK_GUARD | 72 | 14.3% | 3.3% | **+18.64** | **1.52** |

*Tail Risk Guard improved both LONG (+37.9%) and SHORT (+35.7%) subgroups consistently.*

---

## 2. Temporal Stability Across Validation Weeks

| Week Block | Signals | Aegis Net bps | Tail Guard Net bps | Tail Guard Value (bps) |
|---|---:|---:|---:|---:|
| **Week 1 (2026-07-15 to 2026-07-26)** | 46 | -18.71 | **-15.80** | **+181.18** |
| **Week 2 (2026-07-27 to 2026-08-01)** | 62 | +43.52 | **+51.41** | **+335.06** |

*Tail Risk Guard added positive net value in both losing/choppy weeks (+181 bps) and strong trend weeks (+335 bps).*

---

## 3. Cost Sensitivity Stress Test

| Cost Scenario | Policy | Win Rate (%) | Net Expectancy (bps) | Profit Factor | Net Guard Value (bps) |
|---|---|---:|---:|---:|---:|
| **0 bps (Gross)** | AEGIS_ONLY | 81.5% | +31.01 | 1.48 | 0.0 |
| | TAIL_RISK_GUARD | 83.3% | +37.07 | 1.75 | +516.25 |
| **14 bps (Primary)** | AEGIS_ONLY | 81.5% | +17.01 | 1.48 | 0.0 |
| | TAIL_RISK_GUARD | 83.3% | +23.07 | 1.75 | +516.25 |
| **20 bps (Conservative)** | AEGIS_ONLY | 69.4% | +11.01 | 1.30 | 0.0 |
| | TAIL_RISK_GUARD | 70.6% | +17.07 | 1.53 | +552.25 |
| **30 bps (Stress)** | AEGIS_ONLY | 62.0% | +1.01 | 1.02 | 0.0 |
| | TAIL_RISK_GUARD | 62.7% | +7.07 | 1.20 | +612.25 |

*Tail Risk Guard demonstrates robust alpha across all cost assumptions up to 30 bps.*
