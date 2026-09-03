from __future__ import annotations

import json
from pathlib import Path

from aegis.models import CalibrationMethod, CalibratorSpec
from aegis.research.entry_methodology_v2_shadow import (
    ENTRY_PATH_MODEL_FEATURE_NAMES,
)
from aegis.research.entry_path_meta_model_shadow import EntryPathMetaModelShadow
from aegis.training.hybrid_directional import calibrator_mapping
from aegis.tree_models import (
    DecisionTree,
    EnsembleAggregation,
    TreeEnsemble,
    TreeNode,
)
from aegis.utils import Sha256HashProvider, canonical_json


def artifact(path: Path) -> None:
    tree = TreeEnsemble(
        ensemble_id="test-entry-path",
        schema_version="aegis-tree-ensemble-v1",
        feature_names=ENTRY_PATH_MODEL_FEATURE_NAMES,
        aggregation=EnsembleAggregation.ADDITIVE_LOGIT,
        base_value=0.0,
        trees=(DecisionTree((TreeNode(0, 0.0, 0, 0, 0.0, True),)),),
        content_hash="",
    ).to_payload()
    calibrator = calibrator_mapping(
        CalibratorSpec(CalibrationMethod.IDENTITY, 0.0, 0.25, 10)
    )
    payload = {
        "schema_id": "aegis-entry-path-meta-model-shadow-validation-v1",
        "mode": "SHADOW",
        "target": "CLEAN_FAST_SUCCESS",
        "feature_names": list(ENTRY_PATH_MODEL_FEATURE_NAMES),
        "models": {
            side: {
                "model": tree,
                "calibrator": calibrator,
                "selection_threshold": 0.4,
                "validation_pass": side == "SHORT",
            }
            for side in ("LONG", "SHORT")
        },
        "live_selection_effect": "NONE",
        "exchange_authority": False,
        "exchange_mutations": 0,
    }
    payload["content_hash"] = Sha256HashProvider().digest_value(payload)
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def inputs():
    prediction = {
        "opportunity_probability": 0.6,
        "danger_probability": 0.2,
        "mae_q50": 0.001,
        "mae_q90": 0.003,
        "mfe_q50": 0.006,
        "net_return_mean": 0.002,
        "shadow_rank_score": 0.7,
    }
    confirmation = {
        "components_passed": 4,
        "relative_quality": {
            "opportunity_percentile": 0.8,
            "danger_quality_percentile": 0.7,
            "net_return_percentile": 0.6,
            "path_efficiency_percentile": 0.9,
            "path_efficiency": 2.0,
        },
    }
    confirmation_features = {
        name: 0.1 for name in ENTRY_PATH_MODEL_FEATURE_NAMES[13:32]
    }
    intelligence = {
        "regime_v3_shadow": {
            "global_direction": "BEARISH",
            "symbol_direction": "BEARISH",
            "volatility": "NORMAL",
            "structure": "TREND",
            "alignment": "ALIGNED_BEARISH",
            "extension": "NORMAL",
            "evidence_ready": True,
            "global_stability_bars": 4,
            "symbol_stability_bars": 5,
            "volatility_stability_bars": 6,
            "structure_stability_bars": 7,
        },
        "entry_timing_shadow": {"reversal_flags": {}},
        "directional_acceleration_shadow": {
            "component_count": 10,
            "upward_component_count": 2,
            "downward_component_count": 8,
        },
    }
    return prediction, confirmation, confirmation_features, intelligence


def test_only_validated_side_can_emit_shadow_score(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    artifact(path)
    scorer = EntryPathMetaModelShadow(path)
    prediction, confirmation, confirmation_features, intelligence = inputs()
    short = scorer.assess(
        side="SHORT",
        prediction=prediction,
        confirmation=confirmation,
        confirmation_features=confirmation_features,
        entry_intelligence=intelligence,
    )
    long = scorer.assess(
        side="LONG",
        prediction=prediction,
        confirmation=confirmation,
        confirmation_features=confirmation_features,
        entry_intelligence=intelligence,
    )
    assert short["status"] == "VALIDATED_SHADOW_SCORE"
    assert short["probability"] == 0.5
    assert short["counterfactual_pass"] is True
    assert long["status"] == "SIDE_MODEL_NOT_VALIDATED"
    assert long["probability"] is None
    assert short["selection_effect"] == "NONE"
    assert short["exchange_authority"] is False
    assert short["exchange_mutations"] == 0
