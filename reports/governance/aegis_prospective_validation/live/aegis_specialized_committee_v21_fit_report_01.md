# Aegis Specialized Committee V2.1 Fit Report 01

## Decision

- Candidate: `aegis-specialized-committee-v21-risk-v1`
- Diagnostic verdict: `FAILED`
- V2.1 runtime activation: `PROHIBITED`
- Committee V2 state: `SHADOW_UNCHANGED`
- Live effect: `NONE`
- Exchange mutations: `0`

The preregistered V2.1 implementation was fitted and calibrated successfully,
but it did not demonstrate stable risk ranking outside its training period.
It must not be connected to the running Shadow composite or promoted to Live.

## Frozen Authority

- Original preregistration commit: `23580ac`
- Duplicate-candle amendment commit: `586390f`
- Preregistration SHA-256:
  `a42c6b8627c170b746964201bb2177da20d7ced08bf44462d603fcf07613f2f3`
- Historical database SHA-256:
  `fdc3f3ab88950ca4c217b3b132ba8552a256db25db7ed9cd27402f52b5716021`
- Current feature schema: `aegis-features-v2`
- Feature count: `83`

The historical database contained 65 duplicate candle identities in the fit
source interval, including 15 with conflicting OHLC and 30 with conflicting
volume. The preregistered policy retained the first persisted row by minimum
SQLite row ID. No result was fitted before that policy was committed.

## Population

- Training: 5,983 canonical selected SHORT episodes
- Calibration: 1,924 canonical selected SHORT episodes
- Diagnostic only: 598 canonical selected SHORT episodes
- Total: 8,505 episodes
- Model basis terms: 147
- Logistic iterations: 64
- Training interval:
  `2026-05-01T00:00:00Z` through `2026-06-20T23:55:00Z`
- Calibration interval:
  `2026-06-21T00:00:00Z` through `2026-07-04T23:55:00Z`
- Diagnostic interval:
  `2026-07-05T00:00:00Z` through `2026-07-11T09:20:00Z`

## Calibration

- Platt slope: `-0.0715121179823546`
- Platt intercept: `0.1430662212195647`
- Frozen calibrated-risk threshold: `0.540699399501468`
- Calibration ECE: `0.0000003532`
- Diagnostic ECE: `0.0110015292`

The low calibration error is not sufficient evidence of useful ranking. The
predictions are compressed around the adverse base rate, so they can be
well-calibrated in aggregate while failing to distinguish safer from riskier
entries.

## Ranking Stability

| Split | Base-model AUC | Calibrated AUC |
| --- | ---: | ---: |
| Training | 0.61383 | 0.38617 |
| Calibration | 0.49148 | 0.50852 |
| Diagnostic only | 0.53633 | 0.46367 |

The base interaction model learned an in-sample relationship, but that
relationship was nearly absent in calibration. Platt therefore fitted a
negative slope. The inversion was marginally useful inside calibration but
failed in the separate diagnostic period.

## Diagnostic Economics

- Retained coverage: 78.43%
- Mean paired delta: +0.01077%
- Control mean net return: -0.08878%
- Retained mean net return: -0.09947%
- Waited mean net return: -0.04991%
- Control mean MAE: 0.45032%
- Retained mean MAE: 0.46371%
- Waited mean MAE: 0.40162%
- Control worst-decile mean return: -1.17123%
- Retained worst-decile mean return: -1.19363%
- Waited worst-decile mean return: -1.06960%

The small positive policy delta comes from abstaining on some losing entries,
but the model retained the worse subgroup: retained entries had lower win
rate, worse mean return, higher MAE, and worse tail return than abstentions.
That is not a valid calibrated-risk selector.

## Interpretation

This result rejects the exact V2.1 hypothesis, not the broader idea of using
interactions. The failure indicates temporal instability and excessive
compression around the base rate. Symbol and regime interactions in one
global linear logit did not transfer reliably across the frozen periods.

The model artifact is retained only for reproducibility:

- Path:
  `config/bundles/aegis-specialized-committee-v21-risk-v1.json`
- SHA-256:
  `c83a691c81923a2f4ce3c1637a9f3f48dcedf4c0d57c05d5f5b4efaf1bc762d0`

It is not referenced by `live_api.py`, the composite research observer, PM2,
or any runtime configuration.

## Required Next Step

Committee V2 must continue unchanged in Shadow. Any successor must use a new
versioned preregistration and fresh evidence. A reasonable V2.2 hypothesis
would reduce dimensionality, enforce monotonic calibration, and use
purged temporal model selection before freezing a final artifact. V2.1
diagnostic or prospective outcomes may not be reused as an untouched promotion
holdout for that successor.
