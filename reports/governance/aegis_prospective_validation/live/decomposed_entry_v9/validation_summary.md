# Decomposed Entry V9 Validation

- Verdict: `RESEARCH_ONLY_NOT_PROMOTABLE`
- Promotion gate: `FAIL`
- Primary protection: `CURRENT_TS`
- Runtime effect: `NONE`
- Exchange calls/mutations: `0/0`

## LONG

- Passing folds: `0/4`
- Skilled direction folds: `4/4`
- Skilled timing folds: `4/4`
- Skilled trajectory folds: `0/4`
- Worst fold non-negative: `false`
- LOSO: `NOT_RUN_PRIMARY_GATE_FAILED`

| Fold | Selected | Stress mean | CVaR | Payoff | MAE | Control stress | Direction | Timing | Trajectory | Pass |
|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|
| 1 | 45 | -0.002992 | -0.020767 | 0.244283 | 0.008305 | -0.002121 | PASS | PASS | FAIL | FAIL |
| 2 | 17 | -0.002337 | -0.019862 | 0.281238 | 0.007301 | -0.001827 | PASS | PASS | FAIL | FAIL |
| 3 | 28 | -0.002084 | -0.017321 | 0.317833 | 0.006983 | -0.002015 | PASS | PASS | FAIL | FAIL |
| 4 | 13 | -0.000983 | -0.009366 | 0.330856 | 0.004022 | -0.002300 | PASS | PASS | FAIL | FAIL |

## SHORT

- Passing folds: `0/4`
- Skilled direction folds: `4/4`
- Skilled timing folds: `4/4`
- Skilled trajectory folds: `1/4`
- Worst fold non-negative: `false`
- LOSO: `NOT_RUN_PRIMARY_GATE_FAILED`

| Fold | Selected | Stress mean | CVaR | Payoff | MAE | Control stress | Direction | Timing | Trajectory | Pass |
|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|
| 1 | 24 | -0.002898 | -0.022177 | 0.132702 | 0.008335 | -0.001414 | PASS | PASS | FAIL | FAIL |
| 2 | 61 | -0.001380 | -0.011905 | 0.395636 | 0.005916 | -0.002725 | PASS | PASS | PASS | FAIL |
| 3 | 21 | -0.001403 | -0.008131 | 0.610000 | 0.004039 | -0.000859 | PASS | PASS | FAIL | FAIL |
| 4 | 48 | -0.002789 | -0.019483 | 0.484434 | 0.006159 | -0.001499 | PASS | PASS | FAIL | FAIL |

## Decision

`RESEARCH_ONLY_NOT_PROMOTABLE`

No result in this report changes Shadow, Live, PM2, or exchange state.
