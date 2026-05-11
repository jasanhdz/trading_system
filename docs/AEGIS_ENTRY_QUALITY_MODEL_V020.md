# Aegis Entry Quality Model v0.2.0

Entry Quality Model v0.2.0 is a research-only shadow model for Aegis Turbo entries. It estimates whether the current entry candidate is likely to be clean and whether it carries tail-risk characteristics.

It does not replace the rule-based EntryQualityGate and it does not affect live execution.

## Models

- `entry_quality_score`: probability-like score for `label_good_entry_v1`.
- `tail_risk_score`: probability-like score for `label_tail_risk_v1`.

The models were trained from the Phase 1 historical replay dataset with 5m, 15m, 1h, Turbo score, vote, symbol, and side features.

Phase 2 OOS metrics were weak to moderate:

- Entry Quality ROC AUC: 0.5891
- Entry Quality PR AUC: 0.3776
- Tail Risk ROC AUC: 0.5749
- Tail Risk PR AUC: 0.1859

This is enough for forward validation, not enough for enforcement.

## Runtime Mode

The API exposes the model under:

```json
"aegis": {
  "entry_quality_model": {
    "mode": "SHADOW",
    "execute": false,
    "production_allowed": false,
    "status": "RESEARCH_CANDIDATE_NOT_LIVE"
  }
}
```

These fields are hard guarantees. The block is observational only.

## Thresholds

The research thresholds are:

- `quality_min`: 0.60
- `tail_max`: 0.50

Shadow recommendations:

- `ALLOW_SHADOW`: quality is above threshold and tail risk is acceptable.
- `BLOCK_SHADOW`: quality is low, tail risk is high, or both.
- `INSUFFICIENT_DATA`: models or compatible features are unavailable.
- `MODEL_ERROR`: inference failed, but `/ml-v2/predict` still responds.

## Field Guide

- `entry_quality_score`: higher means cleaner entry candidate.
- `tail_risk_score`: higher means more tail-risk probability.
- `recommendation`: what the model would do if it were a shadow gate.
- `reason`: threshold or data reason behind the recommendation.
- `feature_status`: `ok`, `partial`, or `insufficient`.
- `missing_features`: feature names filled by the preprocessor or unavailable at runtime.
- `model_scope`: `symbol`, `global`, or `none`.

## Why Shadow

The historical model shows some separation, but the proxy PnL and AUC are not strong enough to justify live blocking. Phase 3 exists to collect forward evidence:

- Do `BLOCK_SHADOW` candidates produce worse MAE?
- Does `ALLOW_SHADOW` reduce bad-entry rate?
- Is the signal stronger on SHORTs?
- Which symbols benefit without starving trade frequency?

## Future Promotion

Promotion should require Phase 4 forward validation over real live candidates. A reasonable first enforcement experiment, if justified later, would be narrow:

- SHORT-only.
- Specific weak symbols only.
- Conservative threshold.
- Easy rollback.

Do not enable ENFORCE from these research metrics alone.
