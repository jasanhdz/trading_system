# Aegis Decision Brain v010

## Purpose

Aegis Decision Brain is a research meta-model that evaluates whether an existing Turbo opportunity deserves capital now. It does not predict direction. Turbo still owns direction and signal generation.

Version v010 is integrated only as SHADOW runtime inference.

## What It Does Not Do

- It does not open trades.
- It does not close trades.
- It does not block entries.
- It does not change Turbo action.
- It does not change leverage, sizing, smart leverage, brackets, or marketOpen.
- It does not enable enforcement.

Every runtime response has:

```json
{
  "mode": "SHADOW",
  "execute": false,
  "production_allowed": false,
  "status": "RESEARCH_CANDIDATE_NOT_LIVE"
}
```

## Phase 5 Metrics

Offline OOS report:

- Accuracy: `0.492158`
- Macro F1: `0.509609`
- Recommended policy: `SHADOW_ONLY`

Replay proxy:

- Baseline Turbo actual: `net_proxy_pnl=-16.86`, bad-entry `57.5%`, avg MAE `-29.7%`
- Block `DO_NOT_ENTER`: `net_proxy_pnl=5623.28`, bad-entry `43.9%`, avg MAE `-18.3%`
- Enter only `ENTER_NOW`: `net_proxy_pnl=2174.59`, bad-entry `39.4%`, avg MAE `-14.3%`

The offline model shows filter edge, but it is not ready for live enforcement because EventRiskAuto and News/Sentiment coverage is still immature in the labeled dataset.

## Decisions

- `ENTER_NOW`: the model sees an opportunity profile similar to cleaner historical entries.
- `WAIT_CONFIRMATION`: the opportunity may work, but the model expects confirmation or a slower path to green.
- `MANUAL_ONLY`: conditions look contradictory or context-heavy enough for human review.
- `DO_NOT_ENTER`: the model sees high similarity to historically bad or high-MAE entries.
- `UNKNOWN`: artifacts or features were insufficient.

## Probabilities

Runtime exposes:

- `enter_now_prob`
- `wait_confirmation_prob`
- `manual_only_prob`
- `do_not_enter_prob`

These are SHADOW probabilities for forward validation. They are not trading commands.

## Runtime Location

The block is exposed in `/ml-v2/predict`:

```json
{
  "aegis": {
    "decision_brain": {
      "mode": "SHADOW",
      "decision": "DO_NOT_ENTER",
      "do_not_enter_prob": 0.52,
      "execute": false,
      "production_allowed": false
    }
  }
}
```

Runtime status is available in `/debug/runtime` under `decision_brain_status`.

## Feature Status

Decision Brain was trained with 112 features. Runtime feature building keeps the same column order and fills unavailable values with `NaN`, which the trained preprocessor imputes.

Feature status:

- `ok`: high feature coverage and critical groups present.
- `partial`: enough coverage for research scoring, but some groups are missing.
- `insufficient`: low coverage or critical groups missing.

Even when probabilities are emitted, `feature_status` must be reviewed before trusting the signal.

## Phase 7 Use

Phase 7 Outcome Analyzer should compare Decision Brain SHADOW decisions against actual outcomes:

- Did `DO_NOT_ENTER` avoid high MAE?
- Did `ENTER_NOW` go green faster?
- Did `WAIT_CONFIRMATION` identify slow but profitable entries?
- Did `MANUAL_ONLY` correlate with external or BTC/ETH context risk?

Only after enough forward samples should enforcement be considered.
