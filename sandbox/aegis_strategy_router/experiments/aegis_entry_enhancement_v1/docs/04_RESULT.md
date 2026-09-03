# Aegis Entry Enhancement V1 Final Report

## Verdict

`AEGIS_ENTRY_ENHANCEMENT_NO_VALUE`

This result is `DISCOVERY_CONTAMINATED_DIAGNOSTIC_ONLY`. FINAL_HOLDOUT remains
`SEALED_NOT_OPENED`.

## July baseline

The 108 diagnostic validation signals were all SHORT:

- gross expectancy: -0.64 bps per signal;
- conservative net expectancy: -20.64 bps;
- favorable-first: 48.15%;
- MFE/MAE: 42.19 / 41.75 bps, ratio 1.011;
- tail MAE: 167.34 bps.

Thus this population had no directional gross edge before costs.

## Frozen policies

| Policy | Executed | Coverage | Net/signal | Net/executed | BAD rejected | GOOD destroyed |
|---|---:|---:|---:|---:|---:|---:|
| AEGIS_ONLY | 108 | 100.0% | -20.64 | -20.64 | 0.0% | 0.0% |
| Opportunity gate | 0 | 0.0% | 0.00 | N/A | 100.0% | 100.0% |
| Cross-market confirmation | 86 | 79.6% | -16.92 | -21.25 | 17.4% | 31.8% |
| Combined gate | 0 | 0.0% | 0.00 | N/A | 100.0% | 100.0% |

The apparent per-signal improvement of zero-trade policies is rejected. The
cross-market gate had usable coverage but made executed expectancy and
MFE/MAE worse. Its rejection precision, 68.2%, was below the 79.6% BAD
prevalence, and it destroyed GOOD more rapidly than it rejected BAD.

## Ranking and baselines

Quality ranking was not monotonic (`Spearman = 0.60`). Every reported coverage
from 100% through 10% remained negative per executed trade. Simple random,
volatility, BTC-alignment and Aegis-confidence comparisons did not rescue the
primary zero-coverage gate or establish robust incremental value.

Discovery and calibration diagnostics tell the same qualitative story:
cross-market filtering reduced total loss by trading less but did not improve
executed expectancy consistently. Validation has no LONG sample.

## WAIT and latency

WAIT was not evaluated. The frozen modules are static and do not define a
confirmation event; minute candles cannot honestly infer sub-minute execution
latency. Inventing a delayed fill would mix a new sequential policy into this
entry-filter experiment.

## Decision

Do not continue to prospective observation, Shadow or Live with these frozen
policies. Preserve Opportunity and cross-market scores only as diagnostics.

`POST_EXPERIMENT_HYPOTHESES`: a V2 would require a prospectively frozen Aegis
population and a threshold/calibration contract defined without these outcomes,
or a separately preregistered sequential WAIT model. This history could then be
used only for development, never confirmation.

No production, Aegis, PM2, exchange, order, position, sizing or leverage code
was modified.
