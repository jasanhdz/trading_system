# AEGIS W14 - Recent SHORT and Volume Navigation

Preregistered: 2026-08-16 UTC, before running validation.

## Purpose

W14A tests whether a deliberately small SHORT-only selector trained on a recent
30-day regime can select economically useful hourly entries. It does not claim
that the current bearish regime will persist. W14B independently tests whether
two or three aligned 5-minute candles with contemporaneous volume can support a
causal trend-navigation entry and exit.

The studies cannot be combined to rescue either result. No production or live
configuration may be changed by this experiment.

## Frozen data and splits

- Eleven operational symbols; no post-result symbol exclusion.
- Source: closed public 1-minute candles resampled causally to 5 minutes.
- TRAIN: 2026-07-09 through 2026-08-07 UTC (30 days).
- CALIBRATION: 2026-08-08 through 2026-08-11 UTC (4 days).
- VALIDATION: 2026-08-12 through 2026-08-15 06:59 UTC (~3.3 days).
- FINAL HOLDOUT: `SEALED_NOT_OPENED`.

The short validation answers only whether the candidate survived this recent
period. It cannot establish durable alpha or authorize use today/tomorrow.

## W14A

- Direction is always SHORT; the model chooses TAKE/SKIP.
- Hourly anchors prevent overlap at the longest 60-minute horizon.
- Horizons: 15, 30, and 60 minutes.
- Models: regularized logistic and shallow histogram gradient boosting.
- Threshold family: calibration probability quantiles 0.80, 0.90, and 0.95.
- Label: fixed-horizon SHORT gross return greater than the 14 bps baseline cost.
- Candidate selection occurs only in CALIBRATION by net bps per original signal.

The validation gate requires at least 40 trades, at least +2 bps net expectancy
per executed trade at 14 bps cost, positive expectancy at 20 bps stress, at
least four positive symbols, and a positive 95% block-bootstrap lower bound.

## W14B

- Separate LONG and SHORT direction follows the observed candle sequence.
- Sequence length: two or three aligned 5-minute candles.
- Current volume ratio minimum: 1.0 or 1.5 versus prior 20-bar median.
- Entries use the next bar open.
- Exits: fixed 15 minutes, fixed 30 minutes, or frozen causal decay.
- Candidate episodes are separated by at least 12 bars per symbol.

The same economic, symbol, stress, and bootstrap gates apply. W14B does not use
W14A predictions and cannot modify its verdict.

## Statistics and governance

- Costs: 14 bps baseline and 20 bps stress.
- Bootstrap: 10,000 resamples of six-hour temporal blocks.
- Validation is inspected once after all families and gates are frozen.
- No leverage optimization, live activation, shadow activation, or production
  modification is permitted.
