# Aegis SHORT Profitability Semantics V1

## Status

- Experiment: `aegis-short-profitability-semantics-v1`
- Runtime authority: `SHADOW_ONLY`
- Exchange authority: `NONE`
- Automatic Live promotion: `PROHIBITED`
- Configuration SHA-256:
  `03f8ebed96adddd48fe6b608283eff68c243f9ee57d8898d1bcbc3a344446dd0`

## Problem

The current `short_prob` is produced by fixed directional heads. It represents
SHORT side authority and does not estimate whether a trade will finish with a
positive net return. It must remain available only for transport compatibility
and must not be presented as profitability confidence.

## Frozen Probability Meanings

- `short_side_authority`: the model is evaluating the SHORT side. This is not
  an economic probability.
- `terminal_net_positive_probability_h12`: probability that the SHORT terminal
  return after 12 closed 5-minute bars exceeds the frozen round-trip cost.
- `clean_low_mae_probability_h12`: probability of satisfying the existing
  path-aware clean-entry and MAE contract.
- `tail_risk_probability_h12`: probability of the existing adverse tail event.
- `qmae_q90_h12`: the existing conformal adverse-excursion estimate.

The terminal-net-positive label is:

`terminal_short_return - round_trip_cost_fraction > 0`

The frozen round-trip cost is `0.001` and the horizon is 12 bars. It is a
scientific fixed-horizon outcome, not a claim about every possible operational
exit policy.

## Return-Sign Audit

The current E3 bundle stores a model trained on `net_quality_after_costs` in a
legacy field named `expected_return`. Runtime code then applies directional
sign mapping. This experiment must report that legacy value separately and
must not relabel it as expected net profit until its target provenance and sign
are validated end to end.

## Validation

The model is trained and calibrated using purged temporal folds. Thresholds are
derived only from calibration data. Evaluation includes:

- average precision against prevalence;
- ECE and Brier score;
- calibration by probability bucket;
- net expectancy after costs;
- MAE and worst-decile behavior;
- symbol concentration;
- walk-forward fold stability;
- prospective Shadow evidence.

No model can affect Live selection until a separate owner-authorized promotion
shows positive out-of-sample expectancy, adequate calibration, controlled MAE,
and sufficient prospective evidence.

## Prohibitions

This experiment must not:

- create, cancel, modify, or close exchange orders;
- alter the current Python canonical decision;
- alter TypeScript guards, sizing, leverage, or capital;
- use `short_prob` as a win probability;
- silently substitute clean-entry probability for win probability;
- automatically train or promote from runtime observations;
- activate USD100.
