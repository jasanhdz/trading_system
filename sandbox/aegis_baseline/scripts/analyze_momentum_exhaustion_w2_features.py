#!/usr/bin/env python3
"""Produce fixed, non-selective W2 feature diagnostics on VALIDATION."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aegis.utils import sha256_file


def _score(rows: pd.DataFrame) -> np.ndarray:
    return (
        20 * rows["giveback_ratio"].ge(0.30).to_numpy(dtype=int)
        + 15 * rows["bars_since_peak"].ge(2).to_numpy(dtype=int)
        + 15 * rows["directional_velocity_1"].le(0.0).to_numpy(dtype=int)
        + 15 * rows["taker_imbalance_decay"].lt(0.0).to_numpy(dtype=int)
        + 10 * rows["opposite_body_ratio"].ge(0.60).to_numpy(dtype=int)
        + 10 * rows["volume_over_4"].to_numpy(dtype=int)
        + 10 * rows["structure_deterioration"].gt(0.0).to_numpy(dtype=int)
        + 5 * rows["btc_opposes_position"].gt(0.0).to_numpy(dtype=int)
    )


def _weighted_rate(rows: pd.DataFrame, mask: pd.Series) -> dict[str, Any]:
    selected = rows.loc[mask].copy()
    complement = rows.loc[~mask].copy()

    def metrics(frame: pd.DataFrame) -> dict[str, float | int]:
        if frame.empty:
            return {"rows": 0, "episodes": 0, "target_rate": 0.0}
        counts = frame.groupby("position_episode_id")[
            "position_episode_id"
        ].transform("size")
        weights = 1.0 / counts.to_numpy(dtype=float)
        target = frame["target_giveback_before_new_extreme"].to_numpy(dtype=float)
        return {
            "rows": int(len(frame)),
            "episodes": int(frame["position_episode_id"].nunique()),
            "target_rate": float(np.average(target, weights=weights)),
            "mean_additional_mfe_atr": float(np.average(
                frame["target_additional_mfe_atr"], weights=weights
            )),
            "mean_future_giveback_atr": float(np.average(
                frame["target_future_giveback_atr"], weights=weights
            )),
        }

    selected_metrics = metrics(selected)
    complement_metrics = metrics(complement)
    return {
        "condition": selected_metrics,
        "complement": complement_metrics,
        "target_rate_delta": float(
            selected_metrics["target_rate"] - complement_metrics["target_rate"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path,
        default=Path("data/momentum_exhaustion_w2/dataset_train_validation_01"),
    )
    parser.add_argument(
        "--evaluation", type=Path,
        default=Path("data/momentum_exhaustion_w2/evaluation_train_validation_01.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/momentum_exhaustion_w2/feature_diagnostics_validation_01.json"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    def resolve(value: Path) -> Path:
        return value if value.is_absolute() else root / value

    dataset_root = resolve(args.dataset_root)
    evaluation_path = resolve(args.evaluation)
    output = resolve(args.output)
    manifest_path = dataset_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    evaluation = json.loads(evaluation_path.read_text())
    if (
        manifest["final_holdout_state"] != "SEALED"
        or evaluation["final_holdout_outcomes_read"]
    ):
        raise RuntimeError("AEGIS_W2_DIAGNOSTIC_HOLDOUT_VIOLATION")
    columns = [
        "position_episode_id", "partition", "side", "gate_050",
        "target_giveback_before_new_extreme", "target_additional_mfe_atr",
        "target_future_giveback_atr", "giveback_ratio", "bars_since_peak",
        "new_extreme_last_2", "directional_velocity_1", "velocity_decay",
        "directional_taker_imbalance", "taker_imbalance_decay",
        "opposite_body_ratio", "volume_over_4", "structure_deterioration",
        "btc_opposes_position", "directional_rsi_extension",
    ]
    decisions = pd.concat([
        pd.read_parquet(path, columns=columns)
        for path in sorted(dataset_root.glob("*_decisions.parquet"))
    ], ignore_index=True)
    decisions = decisions.loc[
        decisions["partition"].eq("VALIDATION") & decisions["gate_050"]
    ].copy()
    result: dict[str, Any] = {
        "schema_version": "aegis-momentum-exhaustion-w2-feature-diagnostics-v1",
        "classification": "VALIDATION_DIAGNOSTIC_NOT_THRESHOLD_SELECTION",
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "evaluation_sha256": sha256_file(evaluation_path),
        "final_holdout_state": "SEALED",
        "sides": {},
    }
    for side in ("LONG", "SHORT"):
        rows = decisions.loc[decisions["side"].eq(side)].copy()
        score = _score(rows)
        conditions = {
            "NO_NEW_EXTREME_TWO_BARS": ~rows["new_extreme_last_2"],
            "BARS_SINCE_PEAK_GTE_2": rows["bars_since_peak"].ge(2),
            "VELOCITY_NONPOSITIVE": rows["directional_velocity_1"].le(0.0),
            "VELOCITY_DECAY_NEGATIVE": rows["velocity_decay"].lt(0.0),
            "TAKER_FLOW_OPPOSING": rows["directional_taker_imbalance"].lt(0.0),
            "TAKER_FLOW_DECAY_NEGATIVE": rows["taker_imbalance_decay"].lt(0.0),
            "OPPOSITE_BODY_GTE_060": rows["opposite_body_ratio"].ge(0.60),
            "VOLUME_RATIO_OVER_4": rows["volume_over_4"],
            "STRUCTURE_DETERIORATION": rows["structure_deterioration"].gt(0.0),
            "BTC_OPPOSES_POSITION": rows["btc_opposes_position"].gt(0.0),
            "DIRECTIONAL_RSI_EXTREME_GTE_70": rows["directional_rsi_extension"].ge(70.0),
            "FROZEN_SCORE_GTE_50": pd.Series(score >= 50, index=rows.index),
        }
        result["sides"][side] = {
            "rows": int(len(rows)),
            "episodes": int(rows["position_episode_id"].nunique()),
            "conditions": {
                name: _weighted_rate(rows, mask)
                for name, mask in conditions.items()
            },
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.chmod(output, 0o600)
    print(json.dumps({"output": str(output), "holdout": "SEALED"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
