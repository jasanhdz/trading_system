# Label Definitions

Labels are computed only for TRAIN, CALIBRATION and VALIDATION. Every label
uses market data strictly after the causal snapshot and is prefixed
`target__`; the model allowlist rejects that prefix.

Primary path label:

- reference: snapshot price at `t`;
- volatility: ATR14 from the latest fully closed 15m bar;
- favorable/adverse barriers: symmetric ±0.50 ATR;
- horizon: 60 one-minute bars;
- same-minute dual touch: `ADVERSE_FIRST`;
- otherwise `FAVORABLE_FIRST`, `ADVERSE_FIRST`, or `NEITHER`.

Continuous labels are MFE, MAE, MFE-MAE, MFE/MAE, directional 60m return,
time-to-barrier, common barrier/terminal payoff, and payoff after frozen 20 bps
cost and next-1m-open latency diagnostic.

Opportunity is direction-independent and true if either absolute excursion
reaches `max(0.50 ATR, 20 bps)` within 60 minutes.

`FINAL_HOLDOUT` contains no columns beginning with `target__`.
