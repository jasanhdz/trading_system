#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.edge.common import load_model_bundle, safe_float  # noqa: E402
from aegis_alpha.signals.common import drawdown_stats, profit_factor, return_stats  # noqa: E402
from aegis_alpha.signals.signal_registry import SIGNAL_REGISTRY, get_signal_spec  # noqa: E402


DEFAULT_DATASET = Path("aegis_alpha/data/processed/signal_lab_dataset_v050.npz")
DEFAULT_MODEL_DIR = Path("aegis_alpha/models/signals")
DEFAULT_REPORT = Path("aegis_alpha/logs/signals/signal_decile_report_v050.json")
DEFAULT_COMPARISON = Path("aegis_alpha/logs/signals/signal_decile_comparison_v050.json")
DEFAULT_TEST_PCT = 0.20
EDGE_PCTS = (0.10, 0.05, 0.03, 0.02, 0.01, 0.005)
RISK_PCTS = (0.10, 0.20)


def _load_dataset(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    x = data["X"].astype(np.float32) if "X" in data else data["x"].astype(np.float32)
    steps = data["step"].astype(np.int64)
    order = np.argsort(steps, kind="stable")
    loaded: dict[str, Any] = {}
    for key in data.files:
        value = data[key]
        if key in {"feature_names", "horizons", "profit_threshold", "risk_threshold", "window_size", "fee_round_trip", "config"}:
            loaded[key] = value
        elif isinstance(value, np.ndarray) and value.ndim > 0 and len(value) == len(order):
            loaded[key] = value[order]
        else:
            loaded[key] = value
    loaded["X"] = x[order]
    loaded["step"] = steps[order]
    return loaded


def _bucket_metrics(returns: np.ndarray, mfe: np.ndarray, mae: np.ndarray, regimes: np.ndarray) -> dict[str, Any]:
    stats = return_stats(returns)
    dd = drawdown_stats(np.cumsum(returns, dtype=np.float32) + 20.0)
    unique, counts = np.unique(regimes, return_counts=True)
    regime_dist = {str(k): int(v) for k, v in zip(unique, counts)}
    stats.update(
        {
            "avg_return_after_fees": safe_float(np.mean(returns)) if len(returns) else 0.0,
            "median_return": safe_float(np.median(returns)) if len(returns) else 0.0,
            "avg_mfe": safe_float(np.mean(mfe)) if len(mfe) else 0.0,
            "avg_mae": safe_float(np.mean(mae)) if len(mae) else 0.0,
            "regime_distribution": regime_dist,
            "dominance": safe_float(max(np.mean(returns > 0.0), np.mean(returns <= 0.0))) if len(returns) else 0.0,
        }
    )
    stats.update(dd)
    return stats


def _evaluate_edge_model(
    model_name: str,
    scores: np.ndarray,
    returns: np.ndarray,
    mfe: np.ndarray,
    mae: np.ndarray,
    regimes: np.ndarray,
) -> dict[str, Any]:
    pct_metrics: dict[str, Any] = {}
    for pct in EDGE_PCTS:
        threshold = np.quantile(scores, 1.0 - pct)
        mask = scores >= threshold
        pct_metrics[f"top_{int(pct * 1000):03d}bp" if pct < 0.01 else f"top_{int(pct * 100):02d}pct"] = {
            "threshold": safe_float(threshold),
            "count": int(mask.sum()),
            **_bucket_metrics(returns[mask], mfe[mask], mae[mask], regimes[mask]),
        }
    top3 = pct_metrics["top_03pct"]
    top1 = pct_metrics["top_01pct"]
    top05 = pct_metrics["top_005bp"]
    stability = float(np.std([top3["profit_factor"], top1["profit_factor"], top05["profit_factor"]]))
    return {
        "score_direction": "high_better",
        "buckets": pct_metrics,
        "summary": {
            "top_03_pf": top3["profit_factor"],
            "top_03_avg_return": top3["avg_return_after_fees"],
            "top_03_count": top3["count"],
            "stability_pf_std": safe_float(stability),
            "regime_robustness": safe_float(1.0 - max((v / max(top3["count"], 1)) for v in top3["regime_distribution"].values()) if top3["regime_distribution"] else 0.0),
        },
    }


def _evaluate_risk_model(
    scores: np.ndarray,
    y: np.ndarray,
    returns: np.ndarray,
    mfe: np.ndarray,
    mae: np.ndarray,
    regimes: np.ndarray,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "roc_auc": 0.0,
        "average_precision": 0.0,
    }
    if len(np.unique(y)) > 1:
        report["roc_auc"] = safe_float(roc_auc_score(y, scores))
        report["average_precision"] = safe_float(average_precision_score(y, scores))
    buckets: dict[str, Any] = {}
    for pct in RISK_PCTS:
        threshold = np.quantile(scores, pct)
        mask = scores <= threshold
        buckets[f"bottom_{int(pct * 100):02d}pct"] = {
            "threshold": safe_float(threshold),
            "count": int(mask.sum()),
            **_bucket_metrics(returns[mask], mfe[mask], mae[mask], regimes[mask]),
            "failure_rate": safe_float(np.mean(y[mask])) if np.any(mask) else 0.0,
        }
    top_threshold = np.quantile(scores, 0.90)
    top_mask = scores >= top_threshold
    buckets["top_risk_bucket"] = {
        "threshold": safe_float(top_threshold),
        "count": int(top_mask.sum()),
        **_bucket_metrics(returns[top_mask], mfe[top_mask], mae[top_mask], regimes[top_mask]),
        "failure_rate": safe_float(np.mean(y[top_mask])) if np.any(top_mask) else 0.0,
    }
    report["buckets"] = buckets
    report["summary"] = {
        "bottom_10_pf": buckets["bottom_10pct"]["profit_factor"],
        "bottom_20_pf": buckets["bottom_20pct"]["profit_factor"],
        "bottom_10_avg_return": buckets["bottom_10pct"]["avg_return_after_fees"],
        "bottom_20_avg_return": buckets["bottom_20pct"]["avg_return_after_fees"],
    }
    return report


def evaluate_signal_deciles(
    dataset_path: Path,
    model_dir: Path,
    report_path: Path,
    comparison_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    data = _load_dataset(dataset_path)
    x = data["X"].astype(np.float32)
    steps = data["step"].astype(np.int64)
    timestamps = data["timestamp"].astype(str)
    regimes = data["regime"].astype(str)
    split = int(len(x) * (1.0 - DEFAULT_TEST_PCT))
    if split <= 0 or split >= len(x):
        raise ValueError("Invalid holdout split")

    x_test = x[split:]
    regimes_test = regimes[split:]
    timestamps_test = timestamps[split:]
    report_models: dict[str, Any] = {}
    ranking_rows: list[dict[str, Any]] = []

    for spec in SIGNAL_REGISTRY:
        model_path = model_dir / f"aegis_{spec.name}_v050.joblib"
        bundle = load_model_bundle(model_path)
        estimator = bundle.get("estimator") or bundle.get("classifier") or bundle.get("regressor")
        if estimator is None:
            raise RuntimeError(f"Missing estimator in {model_path}")

        if spec.model_type == "classifier":
            scores = estimator.predict_proba(x_test)[:, 1].astype(np.float32)
            y = data[f"h{spec.horizon}_{spec.target_type}"][split:].astype(np.int8)
            actual_returns = data[f"h{spec.horizon}_long_net_return" if spec.side == "LONG" else f"h{spec.horizon}_short_net_return"][split:].astype(np.float32)
            mfe = data[f"h{spec.horizon}_mfe"][split:].astype(np.float32)
            mae = data[f"h{spec.horizon}_mae"][split:].astype(np.float32)
            if spec.side == "SHORT":
                mfe, mae = mae, mfe
            report_models[spec.name] = _evaluate_risk_model(scores, y, actual_returns, mfe, mae, regimes_test)
            ranking_rows.append(
                {
                    "model": spec.name,
                    "type": "risk",
                    "top_03_pf": report_models[spec.name]["summary"]["bottom_10_pf"],
                    "top_03_avg_return": report_models[spec.name]["summary"]["bottom_10_avg_return"],
                    "top_03_count": report_models[spec.name]["buckets"]["bottom_10pct"]["count"],
                    "stability_pf_std": 0.0,
                    "regime_robustness": safe_float(1.0 - max((v / max(report_models[spec.name]["buckets"]["bottom_10pct"]["count"], 1)) for v in report_models[spec.name]["buckets"]["bottom_10pct"]["regime_distribution"].values()) if report_models[spec.name]["buckets"]["bottom_10pct"]["regime_distribution"] else 0.0),
                }
            )
        else:
            scores = estimator.predict(x_test).astype(np.float32)
            actual_returns = data[f"h{spec.horizon}_long_net_return" if spec.side == "LONG" else f"h{spec.horizon}_short_net_return"][split:].astype(np.float32)
            mfe = data[f"h{spec.horizon}_mfe"][split:].astype(np.float32)
            mae = data[f"h{spec.horizon}_mae"][split:].astype(np.float32)
            if spec.side == "SHORT":
                mfe, mae = mae, mfe
            model_report = _evaluate_edge_model(spec.name, scores, actual_returns, mfe, mae, regimes_test)
            report_models[spec.name] = model_report
            ranking_rows.append(
                {
                    "model": spec.name,
                    "type": "edge",
                    **model_report["summary"],
                }
            )

        report_models[spec.name]["holdout_range"] = {
            "start": str(timestamps_test[0]),
            "end": str(timestamps_test[-1]),
        }

    def rank_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
        return (
            float(row.get("top_03_pf", 0.0)),
            float(row.get("top_03_avg_return", 0.0)),
            -float(row.get("stability_pf_std", 0.0)),
            float(row.get("top_03_count", 0.0)),
            float(row.get("regime_robustness", 0.0)),
        )

    comparison = {
        "schema_version": "aegis_signal_decile_comparison_v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "dataset_path": str(dataset_path),
        "holdout_range": {"start": str(timestamps_test[0]), "end": str(timestamps_test[-1])},
        "ranking": sorted(ranking_rows, key=rank_key, reverse=True),
    }
    report = {
        "schema_version": "aegis_signal_decile_report_v1",
        "created_at": comparison["created_at"],
        "dataset_path": str(dataset_path),
        "models": report_models,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json_dumps(report), encoding="utf-8")
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text(json_dumps(comparison), encoding="utf-8")
    print(f"Signal decile report -> {report_path}")
    print(f"Signal decile comparison -> {comparison_path}")
    return report, comparison


def json_dumps(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--comparison", default=str(DEFAULT_COMPARISON))
    args = parser.parse_args()
    evaluate_signal_deciles(
        dataset_path=Path(args.dataset),
        model_dir=Path(args.model_dir),
        report_path=Path(args.report),
        comparison_path=Path(args.comparison),
    )


if __name__ == "__main__":
    main()
