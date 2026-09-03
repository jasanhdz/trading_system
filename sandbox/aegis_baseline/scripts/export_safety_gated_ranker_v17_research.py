#!/usr/bin/env python3
"""Export a deterministic, non-authoritative V17 pre-holdout research artifact."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from aegis.training.competition import export_hist_gradient_boosting
from aegis.utils import Sha256HashProvider, canonical_json, sha256_file
from aegis.v17_feature_contract import contract_for_side
from aegis.tree_models import TreeEnsemble
from aegis.v17_research_artifact import FrozenLinearModel, V17ResearchArtifact
from training.train_directional_contract_v15_research import _fit_bundle, _mapping, _partition, _predict
from training.train_economic_ranker_v16_research import _fit_ranker, _load_v16, _ranked
from training.train_safety_gated_ranker_v17_research import _candidate, _verify_authority


def _linear_payload(model: Any, names: tuple[str, ...], *, output: str) -> Mapping[str, Any]:
    scaler, estimator = model[0], model[-1]
    if int(scaler.n_features_in_) != len(names) or int(estimator.n_features_in_) != len(names):
        raise ValueError("V17 fitted linear feature width mismatch")
    coefficients = np.asarray(estimator.coef_, dtype=np.float64).reshape(-1)
    intercept = np.asarray(estimator.intercept_, dtype=np.float64).reshape(-1)
    if len(coefficients) != len(names) or len(intercept) != 1:
        raise ValueError("V17 fitted linear shape mismatch")
    payload = {
        "schema_id": "aegis-v17-frozen-linear-v1",
        "feature_names": list(names),
        "means": np.asarray(scaler.mean_, dtype=np.float64).tolist(),
        "scales": np.asarray(scaler.scale_, dtype=np.float64).tolist(),
        "coefficients": coefficients.tolist(),
        "intercept": float(intercept[0]),
        "output": output,
    }
    payload["content_hash"] = Sha256HashProvider().digest_value(payload)
    return payload


def _serialization_parity(
    *,
    rows: list[Mapping[str, Any]],
    indices: tuple[int, ...],
    clean_model: Any,
    danger_model: Any,
    mae_model: Any,
    ranker: Any,
    models: Mapping[str, Any],
) -> Mapping[str, Any]:
    sample = rows[: min(256, len(rows))]
    matrix64 = np.asarray(
        [[row["features"][index] for index in indices] for row in sample],
        dtype=np.float64,
    )
    matrix32 = matrix64.astype(np.float32)
    frozen_clean = FrozenLinearModel.from_payload(models["clean"])
    frozen_danger = FrozenLinearModel.from_payload(models["danger"])
    frozen_ranker = FrozenLinearModel.from_payload(models["ranker"])
    frozen_mae = TreeEnsemble.from_payload(models["mae_q90"])
    comparisons = {
        "clean_probability": (
            clean_model.predict_proba(matrix64)[:, 1],
            np.asarray([frozen_clean.evaluate(row) for row in matrix64]),
        ),
        "danger_probability": (
            danger_model.predict_proba(matrix64)[:, 1],
            np.asarray([frozen_danger.evaluate(row) for row in matrix64]),
        ),
        "mae_q90": (
            mae_model.predict(matrix32),
            np.asarray([frozen_mae.evaluate(row) for row in matrix32]),
        ),
        "rank_score": (
            ranker.decision_function(matrix32),
            np.asarray([frozen_ranker.evaluate(row) for row in matrix32]),
        ),
    }
    maximum = {
        name: float(np.max(np.abs(expected - actual)))
        for name, (expected, actual) in comparisons.items()
    }
    tolerances = {
        "clean_probability": 1e-12,
        "danger_probability": 1e-12,
        "mae_q90": 1e-12,
        # SGD is fitted and served by the research code with float32 matrices;
        # the inspectable JSON evaluator accumulates the same terms in float64.
        "rank_score": 1e-6,
    }
    if any(maximum[name] > tolerances[name] for name in maximum):
        raise ValueError(f"V17 serialization parity failed: {maximum}")
    return {
        "samples": len(sample),
        "absolute_tolerance_by_output": tolerances,
        "maximum_absolute_difference": maximum,
        "result": "PASS",
    }


def export(root: Path, config: Mapping[str, Any]) -> Mapping[str, Any]:
    _verify_authority(root, config)
    dataset = root / str(config["authority"]["source_dataset"])
    rows = _load_v16(dataset)
    v15_config = _mapping(yaml.safe_load((root / str(config["authority"]["source_v15_config"])).read_text()), "v15_config")
    v16_config = _mapping(yaml.safe_load((root / str(config["authority"]["source_v16_config"])).read_text()), "v16_config")
    fold = config["validation"]["folds"][-1]
    sides = {}
    for side_offset, side in enumerate(("LONG", "SHORT")):
        population = [row for row in rows if row["side"] == side]
        train_rows, calibration, test = _partition(population, fold, int(config["validation"]["embargo_minutes"]))
        contract = contract_for_side(side)
        indices = tuple(int(value) for value in contract["source_indices"])
        names = tuple(str(value) for value in contract["feature_names"])
        fold_id = int(fold["id"])
        safety = _fit_bundle(train_rows, indices, seed=2026081800 + side_offset * 100 + fold_id * 10)
        ranker, pairs = _fit_ranker(train_rows, indices, v16_config, seed=2026081900 + side_offset * 100 + fold_id)
        calibration_ranked = _ranked(_predict(calibration, safety), ranker, indices)
        test_ranked = _ranked(_predict(test, safety), ranker, indices)
        candidate = _candidate(calibration_ranked, test_ranked, config)
        policy = candidate["rank_policy"]
        status = "COMPLETE_RESEARCH_POLICY" if policy is not None else "CALIBRATION_INFEASIBLE_FAIL_CLOSED"
        models = {
            "clean": _linear_payload(safety["clean"], names, output="PROBABILITY"),
            "danger": _linear_payload(safety["danger"], names, output="PROBABILITY"),
            "mae_q90": export_hist_gradient_boosting(safety["mae"], f"v17-{side.lower()}-mae-q90", names, classifier=False).to_payload(),
            "ranker": _linear_payload(ranker, names, output="RAW_SCORE"),
        }
        sides[side] = {
            "status": status,
            "feature_schema": contract["schema_version"],
            "feature_schema_hash": contract["schema_hash"],
            "feature_count": contract["feature_count"],
            "models": models,
            "gate": candidate["gate"],
            "policy": policy,
            "holdout_metrics": candidate["selected"],
            "training_pairs": pairs,
            "serialization_parity": _serialization_parity(
                rows=test,
                indices=indices,
                clean_model=safety["clean"],
                danger_model=safety["danger"],
                mae_model=safety["mae"],
                ranker=ranker,
                models=models,
            ),
        }
    payload = {
        "schema_id": "aegis-v17-research-artifact-v1",
        "artifact_id": "aegis-safety-gated-ranker-v17-preholdout-fold4",
        "mode": "RESEARCH_ONLY",
        "promotion_authority": False,
        "execution_authority": False,
        "source_dataset_sha256": sha256_file(dataset),
        "configuration_content_sha256": Sha256HashProvider().digest_value(config),
        "fit_boundary": {
            "fold": int(fold["id"]),
            "train_end": str(fold["train_end"]),
            "calibration_end": str(fold["calibration_end"]),
            "test_end": str(fold["test_end"]),
            "test_excluded_from_fit_and_calibration": True,
        },
        "sides": sides,
    }
    payload["content_hash"] = Sha256HashProvider().digest_value(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/experiments/aegis_safety_gated_ranker_v17_research.yaml"))
    parser.add_argument("--output", type=Path, default=Path("data/safety_gated_ranker_v17/research_artifact.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    output = args.output if args.output.is_absolute() else root / args.output
    payload = export(root, _mapping(yaml.safe_load(config_path.read_text()), "config"))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    os.chmod(output, 0o600)
    loaded = V17ResearchArtifact.load(output)
    if loaded.content_hash != payload["content_hash"]:
        raise ValueError("V17 artifact reload hash mismatch")
    print(canonical_json({"output": str(output), "content_hash": payload["content_hash"], "side_status": {side: value["status"] for side, value in payload["sides"].items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
