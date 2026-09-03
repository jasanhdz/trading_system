# Aegis Bundle Candidate Report

## Decision

**REJECTED**. The experiment produced a reproducible trained artifact, but it was not published to the registry and was not configured for shadow or live use.

## Pre-registered protocol

The immutable experiment input is `config/candidate_experiment.yaml`. Mandatory criteria were declared before the final run: at least 100 final-test signals, at least three positive walk-forward folds, profit factor at least 1.05, positive net expectancy, expectancy above the best directional baseline, no symbol above 30% of signals, and no known leakage.

Data came only from local `data/binance_candles.db`, opened immutable/read-only. The period is 2026-01-01 through 2026-07-01 UTC, timeframe 5m, all eleven canonical symbols, 60 history bars, H12 outcome, sampled hourly. Friction is 0.0014 per hypothetical round trip and seed is 20260717.

## Data audit

- Rows loaded: 574,200
- Candidate coordinated cycles: 4,343
- Accepted cycles: 4,231
- Skipped incomplete cycles: 112
- Duplicate rows: 65
- Conflicting duplicates excluded: 30
- Dataset hash: `50cdb33782d65bf9e4cadd72c330a1ed62717c3e31499ee3b444682fb1a3c328`
- Feature hash: `9a6e74720e14ce52800033e14979c4309c2a7f3d3c49b0218e727f29fe64248d`

Windows:

- Train: 2026-01-01 00:05 to 2026-04-16 16:05 UTC
- Validation: 2026-04-16 19:05 to 2026-05-25 04:05 UTC
- Test: 2026-05-25 07:05 to 2026-06-30 23:05 UTC
- Embargo: 120 minutes between adjacent partitions

## Artifact

The experimental artifact is `linear-afec30337dc5ee4254c3`, hash `afec30337dc5ee4254c3882cd6f9521e4aa1d08d433448f25fbef3536b6c46b6`. It is a deterministic NumPy ridge-linear multi-head artifact with train-only normalization and held-out validation thresholding. Metadata marks it `REJECTED_EXPERIMENT`; it is not an approved bundle.

## Baselines on final test

| Strategy | Signals | Net expectancy | Profit factor | Macro F1 | Brier | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|
| NO_TRADE | 0 | 0.000000 | 0.000 | 0.1288 | n/a | 0.0000 |
| Seeded random | 846 | -0.000615 | 0.788 | 0.1854 | n/a | 0.6122 |
| Momentum | 846 | -0.002165 | 0.509 | 0.1808 | n/a | 1.8314 |
| Mean reversion | 846 | -0.000635 | 0.818 | 0.1952 | n/a | 0.7103 |
| Last candle | 846 | -0.001619 | 0.593 | 0.1847 | n/a | 1.3693 |
| Model without layers | 606 | -0.001075 | 0.708 | 0.1711 | 0.1941 | 0.7170 |
| Model with layers | 34 | 0.003421 | 2.043 | 0.1323 | 0.1941 | 0.0288 |

The full-layer path beat directional baselines on expectancy and passed final-test expectancy/profit factor, but only emitted 34 signals. SUIUSDT contributed 21 (61.8%), violating the 30% concentration cap.

## Walk-forward folds

| Fold | Signals | Net expectancy | Profit factor | Macro F1 |
|---|---:|---:|---:|---:|
| 1 | 0 | 0.000000 | 0.000 | 0.1534 |
| 2 | 1 | 0.002090 | unbounded (no losing signal) | 0.1785 |
| 3 | 31 | -0.005144 | 0.316 | 0.1551 |
| 4 | 0 | 0.000000 | 0.000 | 0.1204 |

Only one fold was positive and two produced no signals. The apparent aggregate test edge is therefore too sparse and unstable.

## Promotion checks

Passed: positive test expectancy, minimum test profit factor, better expectancy than directional baselines, no known leakage.

Failed: minimum 100 signals, minimum three positive folds, maximum 30% per-symbol concentration.

The classification is `REJECTED`, not `CANDIDATE` and not `APPROVED_FOR_SHADOW`. `aegis-offline-reference-v1` remains clearly reference-only and was not replaced. No live status exists in this evaluation vocabulary.

Reproduction command:

```bash
PYTHONPATH=src /home/jasan/.venv_rocm62/bin/python scripts/training/run_aegis_candidate_experiment.py
```

Runtime JSON reports are ignored artifacts under `reports/aegis_phase2/`; they are not registry entries.
