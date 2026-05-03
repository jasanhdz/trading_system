from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

WINDOW_FEATURE_NAMES: tuple[str, ...] = (
    "last",
    "mean_6",
    "mean_12",
    "mean_64",
    "std_12",
    "std_64",
    "delta_6",
    "delta_12",
)


def edge_feature_names(base_columns: list[str] | np.ndarray) -> list[str]:
    columns = [str(col) for col in base_columns]
    return [f"{prefix}_{col}" for prefix in WINDOW_FEATURE_NAMES for col in columns]


def build_edge_feature_matrix(features: np.ndarray, window_size: int) -> np.ndarray:
    rows = len(features) - window_size
    if rows <= 0:
        raise ValueError(f"Not enough features for window_size={window_size}: {len(features)}")

    out = np.empty((rows, features.shape[1] * len(WINDOW_FEATURE_NAMES)), dtype=np.float32)
    for out_idx, step in enumerate(range(window_size, len(features))):
        window = features[step - window_size : step]
        last = features[step]
        mean_6 = window[-6:].mean(axis=0)
        mean_12 = window[-12:].mean(axis=0)
        mean_64 = window.mean(axis=0)
        std_12 = window[-12:].std(axis=0)
        std_64 = window.std(axis=0)
        delta_6 = last - window[-6]
        delta_12 = last - window[-12]
        out[out_idx] = np.concatenate((last, mean_6, mean_12, mean_64, std_12, std_64, delta_6, delta_12))
    return np.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0).clip(-10.0, 10.0)


def profit_factor(returns: np.ndarray) -> float:
    wins = returns[returns > 0.0]
    losses = returns[returns < 0.0]
    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    if gross_loss <= 0.0:
        return 999.0 if gross_win > 0.0 else 0.0
    return gross_win / gross_loss


def safe_float(value: Any) -> float:
    return float(np.asarray(value).item())


def save_model_bundle(path: Path, bundle: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_model_bundle(path: Path) -> dict[str, Any]:
    return joblib.load(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
