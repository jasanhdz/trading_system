# 08_HEURISTIC_AND_RANDOM_BENCHMARK_REPORT.md

> **Experiment**: `AEGIS_E4_RISK_GUARD_VALIDATION_V1`  
> **Topic**: Comparison Against 10,000 Monte Carlo Random Baselines and Heuristic Filters

---

## 1. Monte Carlo Random Baseline Benchmark (10,000 iterations)

| Policy | Matched Coverage | Model Bad Rejection (%) | Random Bad Rejection (%) | Model Net Guard Value (bps) | Random Net Guard Value (bps) | Beats Random? |
|---|---:|---:|---:|---:|---:|:---:|
| **Tail Risk Guard** | 94.4% | **15.0%** | 5.6% | **+516.25** | -97.42 | **YES (p < 0.01)** |
| **Late Entry Guard** | 41.7% | 50.0% | 58.5% | -1,073.73 | -1,064.43 | NO |
| **Dual Guard** | 36.1% | 65.0% | 64.1% | -557.48 | -1,168.91 | NO |

---

## 2. Heuristic Filter Comparisons

Simple heuristics (e.g. rejecting when 15m ATR > 90th percentile or 30m return > 2%) achieved lower selectivity:
- Heuristic Tail Filter: Bad rejection 10.0%, Good destruction 8.0% (Loss Saved / Profit Destroyed = 1.25).
- E4 Tail Risk Guard: Bad rejection 15.0%, Good destruction 3.4% (Loss Saved / Profit Destroyed = **3.79**).

E4's multi-timeframe feature model provides significant non-linear discriminative power over raw ATR / volume spikes.
