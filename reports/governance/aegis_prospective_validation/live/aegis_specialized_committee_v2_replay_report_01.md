# Aegis Specialized Committee V2 Replay Report 01

## Decision

- Combined verdict: `NO_INCREMENTAL_VALUE_DETECTED`
- Live promotion: `PROHIBITED`
- Shadow collection: `CONTINUE`
- Training performed: `NO`
- Exchange mutations: `0`

The current Committee V2 counterfactual is not demonstrated superior to the
canonical control. The historical post-cutoff replay is encouraging in
isolation, but the newer forward evidence contradicts it. The combined result
does not satisfy the frozen preliminary gates.

## Frozen Contract

- Contract:
  `config/experiments/aegis_committee_v2_replay_v1.yaml`
- Committee configuration SHA-256:
  `a56db29124e44f9b1aa7b63d7982189b9bc1750c724fffd8873b5bdb3c18ec9c`
- Horizon: 12 closed 5-minute bars
- Round-trip cost fraction: `0.001`
- Global and symbol embargo: 60 minutes
- Walk-forward blocks: 4
- Bootstrap resamples: 2,000
- `WAIT_CONFIRMATION` interpretation: abstention only
- Delayed-entry policy: `NOT_DEFINED`

No threshold, flag, symbol exception, or confirmation rule was fitted after
seeing replay outcomes.

## Forward Journal Replay

- Evaluated through: `2026-07-28T06:50:00Z`
- Matured outcomes: 9,317
- Canonical selected episodes: 237
- Globally independent episodes: 43
- Retained `ENTER_NOW`: 29
- `WAIT_CONFIRMATION`: 14
- Retained coverage: 67.44%
- Mean paired delta: -0.04414%
- Paired delta 95% interval: -0.18213% to +0.07367%
- Positive temporal blocks: 2 of 4
- Verdict: `INSUFFICIENT_INDEPENDENT_EVIDENCE`

The forward sample is below the minimum size and points in the wrong
direction. The `WAIT_CONFIRMATION` group had better average net return and
lower average MAE than the retained group. The current OR-of-risk-flags rule
therefore removed useful opportunities in this period.

## Post-Cutoff Historical Replay

- Information cutoff: `2026-07-11T09:20:00Z`
- Replay interval:
  `2026-07-11T09:25:00Z` through `2026-07-17T18:00:00Z`
- Valid decision cycles: 1,832
- Canonical selected episodes: 532
- Globally independent episodes: 105
- Retained `ENTER_NOW`: 74
- `WAIT_CONFIRMATION`: 31
- Retained coverage: 70.48%
- Mean paired delta: +0.02449%
- Paired delta 95% interval: -0.03289% to +0.08227%
- Control mean net return: -0.01889%
- Committee-policy mean net return: +0.00560%
- Control mean MAE: 0.37160%
- Committee-policy mean MAE: 0.24871%
- Positive temporal blocks: 3 of 4
- Verdict: `PRELIMINARY_INCREMENTAL_VALUE_SUPPORTED`

The historical replay passed every preliminary gate, but did not pass the
robust gate because it had fewer than 300 globally independent episodes and
the confidence interval crossed zero.

## Combined Evidence

- Globally independent episodes: 148
- Retained `ENTER_NOW`: 103
- `WAIT_CONFIRMATION`: 45
- Retained coverage: 69.59%
- Mean paired delta: +0.00455%
- Paired delta 95% interval: -0.05462% to +0.05949%
- Control mean net return: -0.06784%
- Committee-policy mean net return: -0.06329%
- Control mean MAE: 0.40472%
- Retained-entry mean MAE: 0.41155%
- Positive temporal blocks: 2 of 4
- Verdict: `NO_INCREMENTAL_VALUE_DETECTED`

The small positive aggregate delta is neither stable nor statistically
distinguishable from zero. Retained entries did not reduce mean MAE, and only
two of four chronological blocks improved.

## Specialist Diagnosis

Several features currently named as reversal risks were associated with
better, not worse, outcomes in the combined sample:

- `failed_breakdown_proxy`;
- `fake_breakdown_risk_proxy`;
- `low_room_to_fall_risk_proxy`;
- `overextended_down_risk_proxy`;
- `squeeze_plus_reclaim_risk_proxy`;
- `squeeze_risk_proxy_causal`.

This does not prove those features are universally favorable. It proves that
combining every positive flag with a Boolean OR and treating the result as a
reason to wait is not supported. Feature meaning depends on magnitude,
interaction, symbol, and regime.

The combined filter helped preliminarily for SOL and SUI and was harmful for
BNB, BTC, LINK, LTC, and XRP. Symbol counts remain too small for promotion or
symbol-specific rules.

## Reproducibility

- Historical database SHA-256:
  `fdc3f3ab88950ca4c217b3b132ba8552a256db25db7ed9cd27402f52b5716021`
- Historical report SHA-256:
  `3e2e4c17d45c6959d20b99db82f82e9491df072fc6818ecd9742de49720e461d`
- Repeated historical replay: `BYTE_IDENTICAL`
- Forward report SHA-256:
  `d14c29a33c25c7de7b8657512e7962ba6a018168ae5e55445e2583a604ccfe2f`
- Combined report SHA-256:
  `f036551e3a5caf9137f4926e06733dd5c9a3d14e568a3312e6ff50872a2607c8`

Private detailed reports are stored under:

`data/committee_v2_replay/committee_v2_replay_01/`

## Required Next Step

Do not promote the current rule. Continue prospective Shadow evidence and
design any Committee V2.1 hypothesis under a new frozen contract. A V2.1
should model interactions and calibrated risk rather than converting every
nonzero proxy into an equal veto. Its rules must be fixed before replaying new
holdout evidence.
