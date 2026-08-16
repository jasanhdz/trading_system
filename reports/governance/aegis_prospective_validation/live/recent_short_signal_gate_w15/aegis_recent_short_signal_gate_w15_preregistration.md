# AEGIS W15 - Recent SHORT Signal Gate

W15 is a TAKE/SKIP meta-model over actual contemporary Aegis `ENTER_NOW` SHORT
signals. It cannot create a new direction or trade signals rejected by Aegis.

The causal feature set is frozen to model probabilities, risk estimates,
component outputs, symbol, regime, and time available in each immutable signal
bundle. Outcomes are fixed 60-minute net returns recorded by the existing
prospective journal.

- TRAIN: 2026-07-21 through 2026-08-07 UTC.
- CALIBRATION: 2026-08-08 through 2026-08-11 UTC.
- Historical validation: 2026-08-12 through data available on 2026-08-16.
- True final holdout: future signals after the freeze commit only.
- Recorded cost: 10 bps; baseline stress: 14 bps; severe stress: 20 bps.
- Models: regularized logistic, shallow HGB classifier, shallow HGB regression.
- Threshold quantiles: 0.70, 0.80, 0.90, selected only in calibration.
- Bootstrap: 10,000 resamples of 12-hour blocks.

Historical validation is a challenger screen, not authority for immediate use.
Promotion requires positive validation at 14 and 20 bps, at least four positive
symbols, a positive bootstrap lower bound, and a later untouched prospective
period. Production activation is forbidden.
