# Tail-aware Entry V8 Validation

- Verdict: `RESEARCH_ONLY_NOT_PROMOTABLE`
- Promotion gate: `FAIL`
- Runtime effect: `NONE`
- Model exported: `false`
- Exchange calls/mutations: `0/0`

## LONG

- Passing folds: `0/4`
- Skilled regime-router folds: `3/4`
- Skilled late-detector folds: `0/4`
- Worst fold non-negative: `false`
- LOSO: `NOT_RUN_PRIMARY_GATE_FAILED`

| Fold | Selected | Stress mean | Stress CVaR | Payoff | Mean MAE | Control stress | Router | Late detector | Pass |
|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|
| 1 | 96 | -0.001060 | -0.013500 | 0.815471 | 0.006210 | -0.002121 | PASS | FAIL | FAIL |
| 2 | 178 | -0.001644 | -0.011973 | 0.872049 | 0.004709 | -0.001827 | FAIL | FAIL | FAIL |
| 3 | 160 | -0.001387 | -0.011917 | 0.708831 | 0.004735 | -0.002015 | PASS | FAIL | FAIL |
| 4 | 9 | -0.002681 | -0.011500 | 0.483708 | 0.004835 | -0.002300 | PASS | FAIL | FAIL |

## SHORT

- Passing folds: `0/4`
- Skilled regime-router folds: `3/4`
- Skilled late-detector folds: `0/4`
- Worst fold non-negative: `false`
- LOSO: `NOT_RUN_PRIMARY_GATE_FAILED`

| Fold | Selected | Stress mean | Stress CVaR | Payoff | Mean MAE | Control stress | Router | Late detector | Pass |
|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|
| 1 | 58 | -0.001645 | -0.014211 | 0.746608 | 0.005719 | -0.001414 | PASS | FAIL | FAIL |
| 2 | 195 | -0.001057 | -0.012315 | 0.868579 | 0.004978 | -0.002725 | FAIL | FAIL | FAIL |
| 3 | 68 | -0.002292 | -0.011113 | 0.653792 | 0.004861 | -0.000859 | PASS | FAIL | FAIL |
| 4 | 34 | -0.002913 | -0.012212 | 0.553891 | 0.005869 | -0.001499 | PASS | FAIL | FAIL |

## Decision

`RESEARCH_ONLY_NOT_PROMOTABLE`

This result cannot alter Shadow or Live without separate authorization.
