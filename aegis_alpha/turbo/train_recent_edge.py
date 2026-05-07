#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.edge.common import safe_float, save_model_bundle  # noqa: E402
from aegis_alpha.turbo.config import DEFAULT_TURBO_CONFIG, TURBO_VERSION  # noqa: E402
from aegis_alpha.turbo.recent_dataset import build_recent_dataset  # noqa: E402
from aegis_alpha.turbo.snapshot_utils import normalize_turbo_symbol, turbo_symbol_model_dir  # noqa: E402


MIN_TRAIN_SAMPLES = 200
VALIDATION_PCT = 0.25


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _model_path(side: str, lookback_days: int, symbol: str) -> Path:
    legacy_path = DEFAULT_TURBO_CONFIG.model_dir / f"turbo_{side}_edge_{lookback_days}d_v010.joblib"
    if normalize_turbo_symbol(symbol) == normalize_turbo_symbol(DEFAULT_TURBO_CONFIG.symbol):
        return legacy_path
    return turbo_symbol_model_dir(symbol) / f"turbo_{side}_edge_{lookback_days}d_v010.joblib"


def _train_one(dataset: dict[str, Any], side: str, lookback_days: int, symbol: str) -> dict[str, Any]:
    x = np.asarray(dataset["X"], dtype=np.float32)
    target_key = "long_net_return_12" if side == "long" else "short_net_return_12"
    y = np.asarray(dataset[target_key], dtype=np.float32)
    path = _model_path(side, lookback_days, symbol)

    if len(x) < MIN_TRAIN_SAMPLES or len(np.unique(y)) < 2:
        return {
            "lookback_days": int(lookback_days),
            "side": side,
            "model_status": "insufficient_data",
            "sample_count": int(len(x)),
            "model_path": str(path),
            "sklearn_version": sklearn_version,
        }

    split = max(1, int(len(x) * (1.0 - VALIDATION_PCT)))
    if split >= len(x):
        split = len(x) - 1
    x_train, x_val = x[:split], x[split:]
    y_train, y_val = y[:split], y[split:]

    estimator = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=120,
        learning_rate=0.055,
        l2_regularization=0.08,
        max_leaf_nodes=15,
        early_stopping=True,
        random_state=7010 + lookback_days + (0 if side == "long" else 100),
    )
    estimator.fit(x_train, y_train)
    train_pred = estimator.predict(x_train)
    val_pred = estimator.predict(x_val)

    bundle = {
        "metadata": {
            "schema_version": "aegis_turbo_recent_edge_model_v1",
            "created_at": _utc_stamp(),
            "turbo_version": TURBO_VERSION,
            "lookback_days": int(lookback_days),
            "side": side,
            "target_key": target_key,
            "model_kind": "regressor",
            "sample_count": int(len(x)),
            "train_samples": int(len(x_train)),
            "validation_samples": int(len(x_val)),
            "sklearn_version": sklearn_version,
        },
        "feature_names": dataset["feature_names"].tolist() if hasattr(dataset["feature_names"], "tolist") else list(dataset["feature_names"]),
        "estimator": estimator,
    }
    save_model_bundle(path, bundle)

    return {
        "lookback_days": int(lookback_days),
        "side": side,
        "model_status": "trained",
        "sample_count": int(len(x)),
        "train_score": safe_float(-mean_absolute_error(y_train, train_pred)),
        "validation_score": safe_float(-mean_absolute_error(y_val, val_pred)),
        "validation_rmse": safe_float(np.sqrt(mean_squared_error(y_val, val_pred))),
        "avg_predicted_return": safe_float(np.mean(val_pred)),
        "sklearn_version": sklearn_version,
        "model_path": str(path),
    }


def train_recent_edge_models(symbol: str = DEFAULT_TURBO_CONFIG.symbol, lookbacks: tuple[int, ...] | None = None) -> dict[str, Any]:
    cfg = DEFAULT_TURBO_CONFIG
    symbol = normalize_turbo_symbol(symbol)
    turbo_symbol_model_dir(symbol).mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    dataset_reports: list[dict[str, Any]] = []
    for lookback_days in lookbacks or cfg.lookback_days:
        built = build_recent_dataset(symbol, int(lookback_days), save=True)
        dataset = built["dataset"]
        dataset_reports.append(built["report"])
        reports.append(_train_one(dataset, "long", int(lookback_days), symbol))
        reports.append(_train_one(dataset, "short", int(lookback_days), symbol))

    report = {
        "schema_version": "aegis_turbo_train_report_v1",
        "created_at": _utc_stamp(),
        "turbo_version": TURBO_VERSION,
        "symbol": symbol,
        "dataset_reports": dataset_reports,
        "models": reports,
    }
    path = cfg.log_dir / f"turbo_train_report_{_utc_stamp()}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=DEFAULT_TURBO_CONFIG.symbol)
    parser.add_argument("--lookback-days", type=int, action="append")
    args = parser.parse_args()
    train_recent_edge_models(args.symbol, tuple(args.lookback_days) if args.lookback_days else None)


if __name__ == "__main__":
    main()
