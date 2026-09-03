# E4 Train-Live Alignment Report

The controlled cadence experiment isolates the population shift:

| Training/evaluation population | AUC | Brier | ECE |
|---|---:|---:|---:|
| E3-like hourly -> hourly | 0.5431 | 0.2489 | 0.0081 |
| E3-like hourly -> every 5m | 0.5042 | 0.2523 | 0.0381 |
| E4 full 5m -> every 5m | 0.5141 | 0.2498 | 0.0008 |

The hourly model's discrimination nearly disappears and its calibration error
increases about 4.7x on the LIVE-like cadence. E4 removes that mechanical
cadence mismatch and is substantially better calibrated, but its directional
discrimination remains weak.

`TRAIN_LIVE_POPULATION_SHIFT_REDUCED = TRUE` does not imply economic edge.
