# 06_CAUSAL_GUARD_EVALUATION_REPORT.md

> **Experiment**: `AEGIS_E4_RISK_GUARD_VALIDATION_V1`  
> **Topic**: Counterfactual Evaluation of Frozen Guard Policies on VALIDATION Split

---

## 1. Primary Policy Comparison (Cost = 14 bps)

| Metric | Policy A: AEGIS_ONLY | Policy B: LATE_ENTRY_GUARD | Policy C: TAIL_RISK_GUARD | Policy D: DUAL_GUARD |
|---|---:|---:|---:|---:|
| **Total Signals** | 108 | 108 | 108 | 108 |
| **Executed Trades** | 108 | 45 | 102 | 39 |
| **Blocked Trades** | 0 | 63 | 6 | 69 |
| **Coverage (%)** | 100.0% | 41.7% | **94.4%** | 36.1% |
| **Bad Trades Rejected / Total (20)** | 0 / 20 (0.0%) | 10 / 20 (50.0%) | **3 / 20 (15.0%)** | 13 / 20 (65.0%) |
| **Good Trades Destroyed / Total (88)** | 0 / 88 (0.0%) | 53 / 88 (60.2%) | **3 / 88 (3.4%)** | 56 / 88 (63.6%) |
| **Tail Losses Rejected / Total (12)** | 0 / 12 (0.0%) | 8 / 12 (66.7%) | **2 / 12 (16.7%)** | 10 / 12 (83.3%) |
| **Loss Avoided (bps)** | 0.0 | +2,328.97 | **+701.11** | +3,030.09 |
| **Profit Destroyed (bps)** | 0.0 | -3,402.70 | **-184.87** | -3,587.57 |
| **Fees Avoided (bps)** | 0.0 | +882.00 | **+84.00** | +966.00 |
| **Net Guard Value (bps)** | **0.0** | **-1,073.73** | **+516.25** | **-557.48** |
| **Loss Saved / Profit Destroyed Ratio** | 0.00 | 0.68 | **3.79** | 0.84 |
| **Net Expectancy / Executed (bps)** | **+17.01** | **+16.97** | **+23.07 (+35.6%)** | **+32.82** |
| **Win Rate Executed (%)** | 81.48% | 77.78% | **83.33%** | 82.05% |
| **Profit Factor** | 1.48 | 1.50 | **1.75** | 2.55 |
| **MAE Mean (bps)** | 89.51 | 104.02 | **86.67** | 98.83 |
| **MFE / MAE Ratio** | 1.10 | 0.96 | **1.17** | 1.09 |
| **Max Drawdown (bps)** | 2,405.94 | 824.05 | **2,125.92** | 544.04 |

---

## 2. Key Findings by Policy

### Policy C (Tail Risk Guard): CONFIRMED RISK REDUCTION VALUE
- **Loss Saved per Profit Destroyed**: **3.79**. For every 1 bp of winning trade forfeited, the guard saved **3.79 bps** of severe drawdown.
- **Selectivity**: Destroyed only 3.41% of winning trades while eliminating 15.0% of bad trades and 16.7% of tail losses.
- **Economic Improvement**: Increased Net Expectancy from +17.01 bps to **+23.07 bps** and Profit Factor from 1.48 to **1.75**.

### Policy B (Late Entry Guard): REJECTED / DESTRUCTIVE
- Over-filtered the strategy (blocked 58.3% of all signals).
- Destroyed 60.2% of winning trades to save only 50.0% of losing trades.
- Net Guard Value was heavily negative (**-1,073.73 bps**).

### Policy D (Dual Guard): OVER-CONSTRAINED
- Too aggressive (coverage dropped to 36.1%).
- While individual executed trade expectancy was high (+32.82 bps), cumulative net value was negative (**-557.48 bps**) due to excessive trade suppression.
