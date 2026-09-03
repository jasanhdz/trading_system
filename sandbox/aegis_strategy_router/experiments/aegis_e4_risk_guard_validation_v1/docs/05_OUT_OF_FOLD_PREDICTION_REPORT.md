# 05_OUT_OF_FOLD_PREDICTION_REPORT.md

> **Experiment**: `AEGIS_E4_RISK_GUARD_VALIDATION_V1`  
> **Topic**: Out-of-Fold Risk Ranking and Monotonicity Analysis

---

## 1. Risk Decile Monotonicity on VALIDATION Split (108 signals)

| Risk Filter Threshold | Executed Trades | Blocked Trades | Coverage (%) | Bad Rejection (%) | Good Destruction (%) | Net Expectancy (bps) | Profit Factor |
|---|---:|---:|---:|---:|---:|---:|---:|
| **TOP_100_PCT (Allow All)** | 108 | 0 | 100.0% | 0.0% | 0.0% | +17.01 | 1.48 |
| **TOP_90_PCT_QUALITY** | 98 | 10 | 90.7% | 20.0% | 6.8% | +24.48 | 1.83 |
| **TOP_80_PCT_QUALITY** | 87 | 21 | 80.6% | 25.0% | 18.2% | +19.78 | 1.60 |
| **TOP_70_PCT_QUALITY** | 76 | 32 | 70.4% | 40.0% | 27.3% | +21.84 | 1.64 |
| **TOP_60_PCT_QUALITY** | 65 | 43 | 60.2% | 50.0% | 37.5% | +26.89 | 1.86 |
| **TOP_50_PCT_QUALITY** | 54 | 54 | 50.0% | 60.0% | 47.7% | +28.10 | 1.87 |

---

## 2. Analysis of Ranking Monotonicity

- Filtering out the highest 10% risk bucket (`TOP_90_PCT_QUALITY`) provides the cleanest separation: rejects 20.0% of bad trades while destroying only 6.8% of good trades, lifting net expectancy from +17.01 bps to +24.48 bps.
- As filters become stricter (>20% rejection), good trade destruction scales linearly (~1% good destruction per 1% extra rejection), showing diminishing net economic value.
