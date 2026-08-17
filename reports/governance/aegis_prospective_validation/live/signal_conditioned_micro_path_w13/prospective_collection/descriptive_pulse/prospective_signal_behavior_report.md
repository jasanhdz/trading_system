# Prospective Signal Behavior Pulse

Date: 2026-08-17

## Scope

This is an observational, read-only study of the contemporary prospective signal and
outcome journals. It does not train a model, select a threshold, assign a future split,
open a holdout or authorize a trading change. Costs embedded in this journal are 10 bps
(8 bps fees plus 2 bps slippage); W13's later 14/20-bps gates remain unchanged.

## Population

- Signal evaluations: 75,174.
- Mature signal/outcome joins: 75,042.
- Selected `ENTER_NOW`: 1,656.
- Rejected counterfactuals: 73,386.
- Unmatured signals: 132.
- Selected UTC-hour blocks: 432.
- Direction: SHORT only.

The 75,042 evaluations are not independent trades. Symbols evaluated in the same cycle
share market state. Hour-block bootstrap and same-cycle comparisons are therefore more
defensible than treating every row as independent.

## Selected Signal Behavior

- Gross expectancy: approximately +2.17 bps per selected signal.
- Net expectancy after 10 bps: **-7.83 bps**.
- Median net return: **-6.88 bps**.
- Win rate: 44.14%.
- Profit factor: 0.665.
- Mean MFE: 40.97 bps.
- Mean MAE: 40.71 bps.
- Bootstrap 95% interval for net expectancy: **-12.82 to -2.96 bps**.

The median selected signal briefly offered 31.84 bps MFE, suffered 28.81 bps MAE and
finished with only 3.12 bps gross. This is consistent with substantial movement and
giveback, but the journal has only final MFE/MAE and cannot establish the intrahorizon
order of those excursions.

## What Selection Adds

Selected signals were less negative than rejected counterfactuals by about 1.87 bps,
but still materially negative. Within the same evaluation cycle, selected signals were
about 1.25 bps worse than the cycle mean. Among selected signals:

- calibrated score versus realized net return correlation: approximately 0.004;
- predicted expected return versus realized net return correlation: approximately -0.040;
- selected score deciles were all negative and not economically monotonic.

The current selector identifies larger, riskier movement more reliably than realized
economic quality.

## Regime Description

Selected SHORT net expectancy by existing D3 regime:

| Regime | N | Mean net bps |
| --- | ---: | ---: |
| BEAR_TREND | 139 | -3.49 |
| RANGE | 640 | -5.77 |
| TRANSITION | 737 | -8.61 |
| BULL_TREND | 131 | -17.37 |
| HIGH_VOLATILITY | 9 | -17.85 |

The ordering is directionally plausible for SHORT, but no regime is positive. The
high-volatility count is too small for a stable conclusion. D3 regime labels also
collapse almost completely by symbol for ADAUSDT and DOGEUSDT, so regime effects are
partly confounded with symbol identity.

## Symbol Description

SUIUSDT was the only positive selected symbol (+4.86 bps mean net, N=149). This is a
retrospective subgroup and cannot justify trading only SUI. LINKUSDT was weakest at
-18.03 bps (N=327). The cross-symbol heterogeneity is a warning, not a selection rule.

## W13-P Measurement Finding

The first full-L2 diagnostic exposed a post-signal acquisition gap. Signal journal
publication currently arrives approximately 7-18 seconds after T0, while quality-v2
persisted the ring only through T0. Its apparent 1-10 second inactivity was therefore a
measurement artifact.

Quality-v3 backfills all ring events already observed after T0. The 31 quality-v2
bundles remain useful only for delayed endpoint diagnostics and no longer count toward
the 1,000 TRAIN / 500 VALIDATION W13 minimum. Existing rows were not rewritten.

## Conclusion

We can already learn that the current Aegis selection concentrates signals with more
movement, higher MFE and higher MAE, but does not rank net outcomes effectively. Bearish
context is less harmful for SHORT and bullish/high-volatility context is descriptively
more dangerous, yet none is profitable after the journal's 10-bps cost.

We cannot yet answer whether 100ms-30s price/flow/L2 behavior confirms or cancels a
signal. That requires new quality-v3 bundles. No entry guard or production rule is
justified by this pulse.
