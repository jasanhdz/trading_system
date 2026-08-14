# Aegis W2 Momentum Exhaustion and Profit Retention Result

## Status

`AEGIS_MOMENTUM_EXHAUSTION_W2_NO_ECONOMIC_EDGE`

```text
W2_RULE_EDGE_FOUND = FALSE
W2_MODELING_JUSTIFIED = FALSE
W2_READY_FOR_SHADOW = FALSE
W2_READY_FOR_LIVE = FALSE
```

W2 found modest predictive information about future giveback, but no frozen
HOLD/EXIT policy improved profit capture and preserved economic value with the
required stability. The W2 final holdout remains sealed. No production path,
WebSocket, runtime service, or exchange state was changed.

## Experimental Integrity

- Primary unit: `position_episode`, not individual signals.
- Source: 21,774 non-overlapping simulated episodes and 1,954,090 causal
  decision rows over 11 symbols, with LONG and SHORT evaluated separately.
- TRAIN: 12,856 episodes, 2024-01-08 through 2024-10-01.
- VALIDATION: 8,918 episodes, 2024-10-01 through 2025-04-01.
- W2 FINAL HOLDOUT: 23,022 episode identities, `SEALED`; outcomes not read.
- Actual inventory: 706 complete local episodes, classified `ACTUAL`, retained
  as audit-only final evidence; their outcomes were not read by the builder.
- Simulated episodes use separate `simulated_*` fields and
  `outcome_source=SIMULATED`; actual fields remain empty.
- W1 remains frozen and its final holdout was not opened.
- Bootstrap: 10,000 paired day-cluster resamples. FDR was applied across the
  eight preregistered side/gate hypotheses.

## Baselines And Models

The evaluation includes bounded hold, fixed ATR take-profit, fixed ATR
trailing, percentage giveback, time exits, the current 1.5 ATR trailing
parameterization, and the current break-even/profit-protection parameters.

The TypeScript trailing and break-even thresholds were recovered from the
Position Manager. Their replay is parameter-equivalent at closed 5-minute bar
resolution, not byte-for-byte exchange execution. Historical ExitEye committee
outputs, tick-level paths, fill timing, and intrabar order acknowledgements are
not available. Exact historical ExitEye replay was therefore not fabricated.

Models compared: frozen interpretable score, L2 logistic regression,
histogram gradient boosting, random forest, and discrete-time hazard logistic
regression. Model family and threshold selection used TRAIN only.

## Validation Result

Net expectancy is shown in basis points per episode after the preregistered
14 bps cost. Capture is median Profit Capture Ratio.

| Side / gate | N | Frozen baseline | Baseline capture | W2 capture | Delta | Baseline net | W2 net | Net delta | Positive symbols | Positive temporal folds |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LONG 0.25 ATR | 4,164 | 60% giveback | 0.9% | 13.9% | +13.0 pp | -3.09 | -3.76 | -0.67 | 8 | 3/4 |
| LONG 0.50 ATR | 3,977 | bounded hold | 0.0% | 16.9% | +16.9 pp | +10.27 | +6.04 | -4.23 | 3 | 2/4 |
| LONG 0.75 ATR | 3,781 | bounded hold | 4.0% | 20.1% | +16.1 pp | +21.05 | +15.96 | -5.09 | 3 | 2/4 |
| LONG 1.00 ATR | 3,581 | bounded hold | 14.1% | 23.3% | +9.2 pp | +32.74 | +27.89 | -4.85 | 3 | 1/4 |
| SHORT 0.25 ATR | 4,209 | current ATR trailing | 15.5% | 6.3% | -9.2 pp | -5.25 | -5.27 | -0.02 | 0 | 0/4 |
| SHORT 0.50 ATR | 4,010 | current ATR trailing | 17.7% | 12.5% | -5.2 pp | +4.28 | +4.24 | -0.04 | 0 | 0/4 |
| SHORT 0.75 ATR | 3,804 | 12-bar exit | 12.0% | 18.6% | +6.6 pp | +17.62 | +15.07 | -2.56 | 6 | 2/4 |
| SHORT 1.00 ATR | 3,623 | 12-bar exit | 15.6% | 22.5% | +6.9 pp | +27.33 | +24.54 | -2.79 | 5 | 1/4 |

Every W2 policy reduced net expectancy relative to its frozen baseline. No
net-delta confidence interval established non-inferiority. Low-gate candidates
also failed the 20 bps cost stress. Stability was insufficient across symbols
and temporal folds. Since no candidate passed these preceding gates, an
expanding-refit walk-forward was not run; the frozen-policy temporal test is
reported accurately and is not represented as refit walk-forward evidence.

## Predictive Diagnostics

The best validation classifier, histogram gradient boosting, reached ROC-AUC
0.592 LONG and 0.586 SHORT. The discrete hazard model reached ROC-AUC 0.599
for both sides. This is modest ranking information, not a useful trading policy
by itself. High PR-AUC largely reflects the approximately 76.5% primary target
base rate.

At the fixed 0.50 ATR gate, descriptive diagnostics found:

- `volume_ratio > 4`: future-giveback target rate increased by 6.5 pp LONG and
  4.8 pp SHORT.
- directional RSI extension at least 70: increased it by 4.0 pp LONG and
  5.0 pp SHORT.
- velocity decay and taker-flow decay: only small increases, about 0.6-1.2 pp.
- BTC opposition, opposite-body pressure, and structure: weak or inconsistent.
- failure to make a new extreme for two bars: about 2.4-3.5 pp lower target
  rate, contrary to the preregistered intuition at this horizon.

These are diagnostics, not validated thresholds. Extreme volume also retained
meaningful additional favorable movement, so exiting solely on it would confuse
high activity with terminal exhaustion.

## Questions Answered

1. Winning simulated positions often returned most of their available MFE;
   low median capture confirms a real retention problem, but not a solved one.
2. Results vary materially by symbol; no model policy was broadly stable.
3. LONG and SHORT differ. LONG models sometimes raised capture but exited too
   early; low-gate SHORT models reduced capture.
4. Regime/context did not produce an accepted stable policy.
5. Time from peak is represented causally, but temporal effects were unstable.
6. Failure to make a new extreme did not add the expected predictive power.
7. Velocity decay added small information, insufficient economically.
8. Taker-flow decay added small incremental information.
9. Volume above 4x had the strongest descriptive exhaustion association, but
   was not sufficient as an exit rule.
10. RSI extension had descriptive value; divergence was not validated as a
    standalone economic improvement.
11. BTC context was weak and inconsistent.
12. W2 did not consistently improve the current ATR trailing baseline.
13. W2 did not consistently improve simple percentage-giveback policies.
14. W2 frequently exited too early and sacrificed subsequent favorable move.
15. The sacrificed net expectancy ranged from 0.02 to 5.09 bps per episode
    versus the selected baseline, depending on side and gate.
16. No stable HOLD/EXIT policy survived uncertainty, costs, symbols, and time.

## Verdict

W2 confirms that profit giveback is large enough to study and that exhaustion
features contain some information. It refutes the stronger claim that the
tested combination can currently outperform simple/current exits economically.
Opening the holdout, connecting Shadow, or adding production exit logic is not
justified.

The next defensible experiment should isolate whether the descriptive signal
predicts *regret-adjusted exit utility* at a shorter horizon, with a simpler
predeclared model and intrabar/order-flow data. It must be a new preregistered
experiment, not a retune of W2 VALIDATION.

## Artifact Integrity

- Config SHA-256: `75bf6ffecfccc5cfb7c95b4aa0883ddb94bfe34c562ecbc3644aa8671e6c418e`
- Dataset manifest SHA-256: `d095b3f64548b5470629111e7c57b35aec3411f0b5e464269478b66eb5829956`
- Evaluation SHA-256: `8e996dd7ab59b605cbf2882b7db23e68f4e15eee275341b2c11a6b0005a979da`
- Feature diagnostics SHA-256: `505fa0b411df8de36efcc5deead881410e5be0cd2dff5aacc78215d89231ed52`
- Authenticated exchange requests: `0`
- Exchange mutations: `0`
- Production files changed: `0`
