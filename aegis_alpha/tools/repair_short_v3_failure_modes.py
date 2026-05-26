#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.config import REPO_ROOT  # noqa: E402
from aegis_alpha.signals.common import load_signal_market  # noqa: E402
from aegis_alpha.tools.confirm_short_v3_lockbox import (  # noqa: E402
    build_last_block_fold,
    classify_lockbox_candidate,
)
from aegis_alpha.tools.evaluate_ada_short_operable_v2_matrix import finite, validate_research_model_dir  # noqa: E402
from aegis_alpha.turbo.config import DEFAULT_TURBO_CONFIG  # noqa: E402
from aegis_alpha.turbo.operable_feature_builder_v3 import apply_feature_set  # noqa: E402
from aegis_alpha.turbo.recent_dataset import build_recent_dataset  # noqa: E402
from aegis_alpha.turbo.snapshot_utils import normalize_turbo_symbol  # noqa: E402
from aegis_alpha.turbo.train_operable_edge_v2 import (  # noqa: E402
    _classifier,
    _model_seed,
    _regressor,
    _side_arrays,
    classification_metrics,
    safe_corr,
)


MODE = "RESEARCH_ONLY"
SIDE = "SHORT"
DEFAULT_SYMBOLS = ("LINKUSDT", "ADAUSDT", "SOLUSDT", "BTCUSDT", "DOGEUSDT")
OPTIONAL_SYMBOLS = ("BNBUSDT", "XRPUSDT")
DEFAULT_MODEL_DIR = REPO_ROOT / "aegis_alpha" / "models" / "research" / "turbo_v3_failure_repair"
BASE_CONFIGS: dict[str, dict[str, Any]] = {
    "LINKUSDT": {"symbol": "LINKUSDT", "side": SIDE, "feature_set": "operable_v3", "lookback_days": 14, "horizon_candles": 12, "cross_context_enabled": True},
    "ADAUSDT": {"symbol": "ADAUSDT", "side": SIDE, "feature_set": "operable_v2", "lookback_days": 30, "horizon_candles": 24, "cross_context_enabled": True},
    "SOLUSDT": {"symbol": "SOLUSDT", "side": SIDE, "feature_set": "operable_v2", "lookback_days": 14, "horizon_candles": 12, "cross_context_enabled": True},
    "BTCUSDT": {"symbol": "BTCUSDT", "side": SIDE, "feature_set": "combined_v3", "lookback_days": 30, "horizon_candles": 12, "cross_context_enabled": True},
    "DOGEUSDT": {"symbol": "DOGEUSDT", "side": SIDE, "feature_set": "operable_v3", "lookback_days": 30, "horizon_candles": 24, "cross_context_enabled": True},
}
PHASE_I_BASELINE: dict[str, dict[str, str]] = {
    "LINKUSDT": {"original_lockbox_status": "LOCKBOX_WEAK", "failure_mode": "QUALITY_CORR_NEGATIVE_LIFTS_OK"},
    "ADAUSDT": {"original_lockbox_status": "LOCKBOX_WEAK", "failure_mode": "AUC_WEAK_QUALITY_OK"},
    "SOLUSDT": {"original_lockbox_status": "LOCKBOX_WEAK", "failure_mode": "QUALITY_CORR_NEGATIVE_LIFTS_OK"},
    "BTCUSDT": {"original_lockbox_status": "LOCKBOX_WEAK", "failure_mode": "AUC_WEAK_QUALITY_OK"},
    "DOGEUSDT": {"original_lockbox_status": "LOCKBOX_FAILED", "failure_mode": "HIT8_LIFT_NEGATIVE"},
}
MODE_PLANS: dict[str, tuple[str, ...]] = {
    "LINKUSDT": ("top_bucket_only", "top_bucket_only_danger_filtered", "quality_primary_danger_filtered"),
    "ADAUSDT": ("quality_primary", "quality_primary_danger_filtered", "top_bucket_only", "hit8_primary"),
    "SOLUSDT": ("top_bucket_only", "top_bucket_only_danger_filtered", "quality_primary_danger_filtered"),
    "BTCUSDT": ("quality_primary", "quality_primary_danger_filtered", "top_bucket_only", "hit8_primary"),
    "DOGEUSDT": ("hit8_primary", "quality_primary", "quality_primary_danger_filtered", "top_bucket_only", "top_bucket_only_danger_filtered"),
    "BNBUSDT": ("quality_primary", "quality_primary_danger_filtered", "hit8_primary", "top_bucket_only"),
    "XRPUSDT": ("quality_primary", "quality_primary_danger_filtered", "hit8_primary", "top_bucket_only"),
}
STATUS_PRIORITY = {
    "REPAIRED_CONFIRMED": 0,
    "REPAIRED_WEAK": 1,
    "REPAIRED_FAILED": 2,
    "DO_NOT_TRADE_RESEARCH_ONLY": 3,
    "INSUFFICIENT_DATA": 4,
}
MODE_PREFERENCE = {
    "quality_primary_danger_filtered": 0,
    "hit8_primary_danger_filtered": 1,
    "top_bucket_only_danger_filtered": 2,
    "quality_primary": 3,
    "hit8_primary": 4,
    "top_bucket_only": 5,
    "danger_filtered": 6,
}
CSV_COLUMNS = (
    "symbol",
    "side",
    "original_lockbox_status",
    "current_lockbox_status",
    "failure_mode",
    "current_failure_mode",
    "repair_mode",
    "repair_status",
    "repair_reason",
    "feature_set",
    "lookback_days",
    "horizon_candles",
    "cross_context_enabled",
    "test_samples",
    "selected_count",
    "selected_fraction",
    "baseline_hit8",
    "baseline_quality",
    "baseline_mae_danger",
    "baseline_p90_mae",
    "selected_hit8_rate",
    "selected_hit8_lift",
    "selected_quality_mean",
    "selected_quality_lift",
    "selected_net_quality_lift_after_cost",
    "selected_mae_danger_rate",
    "selected_p90_mae",
    "selected_p90_mae_delta",
    "selected_avg_mfe",
    "selected_avg_mae",
    "hit8_auc",
    "quality_corr",
    "danger_auc",
    "danger_filter_usefulness",
    "repair_score",
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_stamp() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%SZ")


def parse_symbols(raw: str | None, include_bnb_xrp: bool) -> list[str]:
    requested = list(DEFAULT_SYMBOLS if raw is None else tuple(value.strip() for value in raw.split(",") if value.strip()))
    if include_bnb_xrp:
        requested.extend(OPTIONAL_SYMBOLS)
    return list(dict.fromkeys(normalize_turbo_symbol(symbol) for symbol in requested))


def base_config(symbol: str) -> dict[str, Any]:
    normalized = normalize_turbo_symbol(symbol)
    if normalized in BASE_CONFIGS:
        return dict(BASE_CONFIGS[normalized])
    return {
        "symbol": normalized,
        "side": SIDE,
        "feature_set": "operable_v3",
        "lookback_days": 30,
        "horizon_candles": 12,
        "cross_context_enabled": True,
    }


def repair_configs(symbol: str, disable_cross_context: bool = False) -> list[dict[str, Any]]:
    original = base_config(symbol)
    if symbol != "DOGEUSDT":
        return [original]
    return [
        {**original, "feature_set": "operable_v2", "lookback_days": 30, "horizon_candles": 12},
        {**original, "feature_set": "operable_v3", "lookback_days": 30, "horizon_candles": 12, "cross_context_enabled": not disable_cross_context},
        {**original, "feature_set": "operable_v3", "lookback_days": 14, "horizon_candles": 12, "cross_context_enabled": not disable_cross_context},
    ]


def classify_failure_mode(lockbox_row: dict[str, Any]) -> str:
    if (
        lockbox_row.get("model_status") != "trained"
        or int(lockbox_row.get("test_samples") or 0) <= 0
        or lockbox_row.get("lockbox_status") == "LOCKBOX_INSUFFICIENT_DATA"
    ):
        return "INSUFFICIENT_DATA"
    if finite(lockbox_row.get("hit8_top_decile_lift")) < 0.0:
        return "HIT8_LIFT_NEGATIVE"
    if (
        finite(lockbox_row.get("quality_top_decile_lift")) < 0.0
        or finite(lockbox_row.get("net_quality_lift_after_cost_proxy")) <= 0.0
    ):
        return "QUALITY_NEGATIVE"
    if (
        lockbox_row.get("top_decile_p90_mae") is not None
        and lockbox_row.get("baseline_p90_mae") is not None
        and finite(lockbox_row["top_decile_p90_mae"]) > finite(lockbox_row["baseline_p90_mae"]) * 1.15
    ) or (
        lockbox_row.get("top_decile_mae_danger_rate") is not None
        and lockbox_row.get("baseline_mae_danger") is not None
        and finite(lockbox_row["top_decile_mae_danger_rate"]) > finite(lockbox_row["baseline_mae_danger"]) * 1.15
    ):
        return "RISK_UNCONTROLLED"
    if (
        finite(lockbox_row.get("hit8_auc")) < 0.53
        and finite(lockbox_row.get("quality_top_decile_lift")) > 0.0
        and finite(lockbox_row.get("net_quality_lift_after_cost_proxy")) > 0.0
    ):
        return "AUC_WEAK_QUALITY_OK"
    if (
        finite(lockbox_row.get("quality_corr")) < 0.0
        and finite(lockbox_row.get("hit8_top_decile_lift")) > 0.0
        and finite(lockbox_row.get("quality_top_decile_lift")) > 0.0
    ):
        return "QUALITY_CORR_NEGATIVE_LIFTS_OK"
    return "MIXED_UNCLEAR"


def _rank_fraction(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(np.asarray(scores, dtype=np.float64))
    rank = np.empty(len(order), dtype=np.float64)
    rank[order] = np.arange(len(order), dtype=np.float64)
    return rank / max(len(order) - 1, 1)


def top_bucket_mask(scores: np.ndarray, fraction: float = 0.10) -> np.ndarray:
    array = np.asarray(scores, dtype=np.float64)
    if not len(array):
        return np.zeros(0, dtype=bool)
    return array >= float(np.quantile(array, 1.0 - fraction))


def selection_mask(predictions: dict[str, np.ndarray], repair_mode: str) -> np.ndarray:
    hit8 = np.asarray(predictions["hit8"], dtype=np.float64)
    quality = np.asarray(predictions["quality"], dtype=np.float64)
    danger = np.asarray(predictions["danger"], dtype=np.float64)
    if repair_mode.startswith("hit8_primary") or repair_mode == "danger_filtered":
        selected = top_bucket_mask(hit8)
    elif repair_mode.startswith("quality_primary"):
        selected = top_bucket_mask(quality)
    elif repair_mode.startswith("top_bucket_only"):
        consensus = (_rank_fraction(hit8) + _rank_fraction(quality)) / 2.0
        selected = top_bucket_mask(consensus)
    else:
        raise ValueError(f"unsupported repair mode: {repair_mode}")
    if "danger_filtered" in repair_mode or repair_mode == "danger_filtered":
        selected = selected & (danger <= float(np.quantile(danger, 0.80)))
    return selected


def _baseline(arrays: dict[str, np.ndarray], test: np.ndarray) -> dict[str, float]:
    return {
        "baseline_hit8": float(np.mean(arrays["hit8"][test])),
        "baseline_quality": float(np.mean(arrays["quality"][test])),
        "baseline_mae_danger": float(np.mean(arrays["danger"][test])),
        "baseline_p90_mae": float(np.quantile(arrays["mae"][test], 0.90)),
    }


def _danger_filter_usefulness(predicted_danger: np.ndarray, actual_danger: np.ndarray) -> float | None:
    predictions = np.asarray(predicted_danger, dtype=np.float64)
    actual = np.asarray(actual_danger, dtype=np.float64)
    if not len(predictions):
        return None
    low = actual[predictions <= float(np.quantile(predictions, 0.10))]
    high = actual[predictions >= float(np.quantile(predictions, 0.90))]
    if not len(low) or not len(high):
        return None
    return float(np.mean(high) - np.mean(low))


def _fit_predictions(
    dataset: dict[str, Any],
    config: dict[str, Any],
    *,
    min_train_samples: int,
    min_test_samples: int,
    lockbox_test_ratio: float,
    fast: bool,
) -> dict[str, Any]:
    x = np.asarray(dataset["X"], dtype=np.float32)
    fold = build_last_block_fold(
        len(x),
        test_ratio=lockbox_test_ratio,
        min_train_samples=min_train_samples,
        min_test_samples=min_test_samples,
    )
    if fold is None:
        return {"model_status": "insufficient_data"}
    arrays = _side_arrays(dataset, SIDE.lower(), int(config["horizon_candles"]))
    arrays["mfe"] = np.asarray(dataset[f"short_mfe_{config['horizon_candles']}"], dtype=np.float32)
    split = {name: np.asarray(fold[name], dtype=np.int64) for name in ("train", "validation", "test")}
    if len(np.unique(arrays["hit8"][split["train"]])) < 2 or len(np.unique(arrays["danger"][split["train"]])) < 2:
        return {"model_status": "insufficient_class_diversity", "split": split}
    max_iter = 60 if fast else 140
    predictions: dict[str, np.ndarray] = {}
    for name, y, constructor in (
        ("hit8", arrays["hit8"], _classifier),
        ("quality", arrays["quality"], _regressor),
        ("danger", arrays["danger"], _classifier),
    ):
        seed = _model_seed(
            config["symbol"],
            SIDE.lower(),
            int(config["lookback_days"]),
            f"repair:{config['feature_set']}:{config['horizon_candles']}:{name}",
        )
        model = constructor(max_iter, seed)
        model.fit(x[split["train"]], y[split["train"]])
        predictions[name] = (
            model.predict(x[split["test"]])
            if name == "quality"
            else model.predict_proba(x[split["test"]])[:, 1]
        )
    test = split["test"]
    hit_metrics = classification_metrics(arrays["hit8"][test], predictions["hit8"])
    danger_metrics = classification_metrics(arrays["danger"][test], predictions["danger"])
    return {
        "model_status": "trained",
        "split": split,
        "arrays": arrays,
        "predictions": predictions,
        "baseline": _baseline(arrays, test),
        "hit8_auc": hit_metrics.get("roc_auc"),
        "quality_corr": safe_corr(predictions["quality"], arrays["quality"][test], method="spearman"),
        "danger_auc": danger_metrics.get("roc_auc"),
        "danger_filter_usefulness": _danger_filter_usefulness(predictions["danger"], arrays["danger"][test]),
        "test_samples": int(len(test)),
    }


def original_lockbox_metrics(trained: dict[str, Any], config: dict[str, Any], *, fee_bps: float, slippage_bps: float) -> dict[str, Any]:
    if trained.get("model_status") != "trained":
        return {**config, "model_status": trained.get("model_status"), "test_samples": 0, "lockbox_status": "LOCKBOX_INSUFFICIENT_DATA"}
    test = trained["split"]["test"]
    arrays = trained["arrays"]
    baseline = trained["baseline"]
    hit_mask = top_bucket_mask(trained["predictions"]["hit8"])
    quality_mask = top_bucket_mask(trained["predictions"]["quality"])
    quality_lift = float(np.mean(arrays["quality"][test][quality_mask])) - baseline["baseline_quality"]
    row = {
        **config,
        "model_status": "trained",
        "test_samples": trained["test_samples"],
        **baseline,
        "hit8_auc": trained["hit8_auc"],
        "quality_corr": trained["quality_corr"],
        "danger_auc": trained["danger_auc"],
        "danger_filter_usefulness": trained["danger_filter_usefulness"],
        "hit8_top_decile_lift": float(np.mean(arrays["hit8"][test][hit_mask])) - baseline["baseline_hit8"],
        "quality_top_decile_lift": quality_lift,
        "net_quality_lift_after_cost_proxy": quality_lift - (fee_bps + slippage_bps) / 10000.0,
        "latest_fold_quality_lift": quality_lift,
        "top_decile_p90_mae": float(np.quantile(arrays["mae"][test][quality_mask], 0.90)),
        "top_decile_mae_danger_rate": float(np.mean(arrays["danger"][test][quality_mask])),
    }
    row["lockbox_status"] = classify_lockbox_candidate(row, min_test_samples=300, strict=False)
    return row


def evaluate_repair_mode(
    trained: dict[str, Any],
    config: dict[str, Any],
    *,
    repair_mode: str,
    original_lockbox_status: str,
    failure_mode: str,
    fee_bps: float,
    slippage_bps: float,
) -> dict[str, Any]:
    if trained.get("model_status") != "trained":
        return {
            **config,
            "repair_mode": repair_mode,
            "original_lockbox_status": original_lockbox_status,
            "failure_mode": failure_mode,
            "repair_status": "INSUFFICIENT_DATA",
            "repair_reason": "insufficient_model_training_data",
            "test_samples": 0,
        }
    arrays = trained["arrays"]
    test = trained["split"]["test"]
    baseline = trained["baseline"]
    selected = selection_mask(trained["predictions"], repair_mode)
    selected_count = int(selected.sum())
    cost = (fee_bps + slippage_bps) / 10000.0
    if selected_count:
        hit8 = float(np.mean(arrays["hit8"][test][selected]))
        quality = float(np.mean(arrays["quality"][test][selected]))
        danger = float(np.mean(arrays["danger"][test][selected]))
        p90_mae = float(np.quantile(arrays["mae"][test][selected], 0.90))
        avg_mae = float(np.mean(arrays["mae"][test][selected]))
        avg_mfe = float(np.mean(arrays["mfe"][test][selected]))
    else:
        hit8 = quality = danger = p90_mae = avg_mae = avg_mfe = float("nan")
    row = {
        **config,
        "original_lockbox_status": original_lockbox_status,
        "failure_mode": failure_mode,
        "repair_mode": repair_mode,
        "test_samples": int(len(test)),
        "selected_count": selected_count,
        "selected_fraction": selected_count / max(len(test), 1),
        **baseline,
        "selected_hit8_rate": hit8,
        "selected_hit8_lift": hit8 - baseline["baseline_hit8"],
        "selected_quality_mean": quality,
        "selected_quality_lift": quality - baseline["baseline_quality"],
        "selected_net_quality_lift_after_cost": quality - baseline["baseline_quality"] - cost,
        "selected_mae_danger_rate": danger,
        "selected_p90_mae": p90_mae,
        "selected_p90_mae_delta": p90_mae - baseline["baseline_p90_mae"],
        "selected_avg_mfe": avg_mfe,
        "selected_avg_mae": avg_mae,
        "hit8_auc": trained["hit8_auc"],
        "quality_corr": trained["quality_corr"],
        "danger_auc": trained["danger_auc"],
        "danger_filter_usefulness": trained["danger_filter_usefulness"],
    }
    row["repair_status"] = classify_repair_candidate(row)
    row["repair_reason"] = repair_reason(row)
    row["repair_score"] = repair_score(row)
    return row


def classify_repair_candidate(row: dict[str, Any]) -> str:
    if int(row.get("test_samples") or 0) <= 0:
        return "INSUFFICIENT_DATA"
    selected_count = int(row.get("selected_count") or 0)
    required_count = min(30, max(1, math.ceil(float(row["test_samples"]) * 0.05)))
    if selected_count < required_count:
        return "REPAIRED_FAILED"
    p90_ratio = finite(row.get("selected_p90_mae")) / max(finite(row.get("baseline_p90_mae")), 1e-12)
    hit_floor = 0.03 if row.get("symbol") == "DOGEUSDT" else 0.0
    confirmed = (
        finite(row.get("selected_hit8_lift")) > hit_floor
        and finite(row.get("selected_quality_lift")) > 0.0
        and finite(row.get("selected_net_quality_lift_after_cost")) > 0.0
        and p90_ratio <= 1.15
    )
    if confirmed:
        return "REPAIRED_CONFIRMED"
    failed = (
        finite(row.get("selected_hit8_lift")) < 0.0
        or finite(row.get("selected_quality_lift")) < 0.0
        or finite(row.get("selected_net_quality_lift_after_cost")) <= 0.0
        or p90_ratio > 1.30
    )
    return "REPAIRED_FAILED" if failed else "REPAIRED_WEAK"


def repair_reason(row: dict[str, Any]) -> str:
    if row["repair_status"] == "REPAIRED_CONFIRMED":
        return "repair_improves_hit8_quality_after_cost_with_controlled_mae"
    if row["repair_status"] == "REPAIRED_WEAK":
        return "repair_has_partial_improvement_without_full_confirmation"
    if row["repair_status"] == "INSUFFICIENT_DATA":
        return "insufficient_repair_evaluation_data"
    return "repair_does_not_restore_operable_edge"


def repair_score(row: dict[str, Any]) -> float:
    return (
        finite(row.get("selected_hit8_lift")) * 2.0
        + finite(row.get("selected_net_quality_lift_after_cost")) * 3.0
        - max(0.0, finite(row.get("selected_p90_mae_delta"))) * 20.0
        + finite(row.get("danger_auc")) * 0.10
    )


def best_by_symbol(rows: list[dict[str, Any]], symbols: list[str]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for symbol in symbols:
        candidates = [row for row in rows if row.get("symbol") == symbol]
        if not candidates:
            selected.append({"symbol": symbol, "side": SIDE, "repair_status": "INSUFFICIENT_DATA", "repair_reason": "no_modes_evaluated"})
            continue
        candidates.sort(
            key=lambda row: (
                STATUS_PRIORITY.get(str(row.get("repair_status")), 99),
                -finite(row.get("repair_score")),
                MODE_PREFERENCE.get(str(row.get("repair_mode")), 99),
            )
        )
        best = dict(candidates[0])
        if all(row.get("repair_status") in {"REPAIRED_FAILED", "INSUFFICIENT_DATA"} for row in candidates):
            best["repair_status"] = "DO_NOT_TRADE_RESEARCH_ONLY"
            best["repair_reason"] = "all_bounded_repair_modes_failed"
        selected.append(best)
    return selected


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _num(value: Any) -> str:
    return "null" if value is None else f"{float(value):.4f}"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        f"# Aegis SHORT V3 Failure-Mode Repair {report['created_at']}",
        "",
        "## Safety",
        "",
        "- Mode: `RESEARCH_ONLY`.",
        "- Repair search is bounded to failure-specific modes and documented DOGE variants.",
        "- No model artifacts are saved; no `active/` models or active manifests are modified.",
        "- No live inference, YAML, thresholds or PM2 action is involved.",
        "",
        "## Best Repair By Symbol",
        "",
        "| Symbol | Original | Failure Mode | Best Repair | Status | Hit8 Lift | Quality Lift | Net Lift | P90 Delta | Fraction |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["best_by_symbol"]:
        lines.append(
            f"| {row['symbol']} | {row.get('original_lockbox_status', '-')} | {row.get('failure_mode', '-')} | "
            f"{row.get('repair_mode', '-')} | {row['repair_status']} | {_num(row.get('selected_hit8_lift'))} | "
            f"{_num(row.get('selected_quality_lift'))} | {_num(row.get('selected_net_quality_lift_after_cost'))} | "
            f"{_num(row.get('selected_p90_mae_delta'))} | {_num(row.get('selected_fraction'))} |"
        )
    for key, title in (
        ("repaired_confirmed", "Repaired Confirmed"),
        ("repaired_weak", "Repaired Weak"),
        ("failed", "Failed Or Do Not Trade"),
    ):
        lines.extend(["", f"## {title}", ""])
        rows = report[key]
        if not rows:
            lines.append("- None.")
        for row in rows:
            lines.append(f"- `{row['symbol']}`: `{row.get('repair_mode', '-')}` -> `{row['repair_status']}`.")
    lines.extend(["", "## Decision", "", f"- Recommendation: `{report['next_recommendation']}`.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _evaluate_dataset(
    dataset: dict[str, Any],
    config: dict[str, Any],
    *,
    modes: tuple[str, ...],
    original_status: str,
    failure_mode: str,
    args: argparse.Namespace,
    trained: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if trained is None:
        trained = _fit_predictions(
            dataset,
            config,
            min_train_samples=int(args.min_train_samples),
            min_test_samples=int(args.min_test_samples),
            lockbox_test_ratio=float(args.lockbox_test_ratio),
            fast=bool(args.fast),
        )
    return [
        evaluate_repair_mode(
            trained,
            config,
            repair_mode=repair_mode,
            original_lockbox_status=original_status,
            failure_mode=failure_mode,
            fee_bps=float(args.fee_bps),
            slippage_bps=float(args.slippage_bps),
        )
        for repair_mode in modes
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    symbols = parse_symbols(args.symbols, bool(args.include_bnb_xrp))
    validate_research_model_dir(Path(args.model_dir))
    context_markets = {
        symbol: load_signal_market(DEFAULT_TURBO_CONFIG.config_path, symbol_override=symbol)
        for symbol in ("BTCUSDT", "ETHUSDT")
    }
    market_cache: dict[str, Any] = {}
    base_dataset_cache: dict[tuple[str, int], dict[str, Any]] = {}
    dataset_cache: dict[tuple[str, str, int, bool], dict[str, Any]] = {}
    all_modes: list[dict[str, Any]] = []
    original_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for symbol in symbols:
        try:
            market_cache[symbol] = load_signal_market(DEFAULT_TURBO_CONFIG.config_path, symbol_override=symbol)
            original = base_config(symbol)
            key = (symbol, original["feature_set"], original["lookback_days"], original["cross_context_enabled"])
            base_key = (symbol, original["lookback_days"])
            if base_key not in base_dataset_cache:
                base_dataset_cache[base_key] = build_recent_dataset(
                    symbol,
                    original["lookback_days"],
                    save=False,
                    market=market_cache[symbol],
                )["dataset"]
            if key not in dataset_cache:
                dataset_cache[key] = apply_feature_set(
                    base_dataset_cache[base_key],
                    market_cache[symbol],
                    original["feature_set"],
                    context_markets=context_markets if original["cross_context_enabled"] else {},
                )
            trained_original = _fit_predictions(
                dataset_cache[key],
                original,
                min_train_samples=int(args.min_train_samples),
                min_test_samples=int(args.min_test_samples),
                lockbox_test_ratio=float(args.lockbox_test_ratio),
                fast=bool(args.fast),
            )
            original_row = original_lockbox_metrics(
                trained_original,
                original,
                fee_bps=float(args.fee_bps),
                slippage_bps=float(args.slippage_bps),
            )
            original_row["current_lockbox_status"] = original_row["lockbox_status"]
            original_row["current_failure_mode"] = classify_failure_mode(original_row)
            phase_i = PHASE_I_BASELINE.get(symbol, {
                "original_lockbox_status": original_row["lockbox_status"],
                "failure_mode": original_row["current_failure_mode"],
            })
            original_row["original_lockbox_status"] = phase_i["original_lockbox_status"]
            original_row["failure_mode"] = phase_i["failure_mode"]
            original_rows.append(original_row)
            modes = MODE_PLANS.get(symbol, ("quality_primary", "quality_primary_danger_filtered", "hit8_primary"))
            for config in repair_configs(symbol, bool(args.disable_cross_context)):
                dataset_key = (symbol, config["feature_set"], config["lookback_days"], config["cross_context_enabled"])
                base_key = (symbol, config["lookback_days"])
                if base_key not in base_dataset_cache:
                    base_dataset_cache[base_key] = build_recent_dataset(
                        symbol,
                        config["lookback_days"],
                        save=False,
                        market=market_cache[symbol],
                    )["dataset"]
                if dataset_key not in dataset_cache:
                    dataset_cache[dataset_key] = apply_feature_set(
                        base_dataset_cache[base_key],
                        market_cache[symbol],
                        config["feature_set"],
                        context_markets=context_markets if config["cross_context_enabled"] else {},
                    )
                all_modes.extend(_evaluate_dataset(
                    dataset_cache[dataset_key],
                    config,
                    modes=modes,
                    original_status=original_row["original_lockbox_status"],
                    failure_mode=original_row["failure_mode"],
                    args=args,
                    trained=trained_original if dataset_key == key and config == original else None,
                ))
        except Exception as exc:
            errors.append({"symbol": symbol, "error": repr(exc)})
    best = best_by_symbol(all_modes, symbols)
    repaired_confirmed = [row for row in best if row["repair_status"] == "REPAIRED_CONFIRMED"]
    repaired_weak = [row for row in best if row["repair_status"] == "REPAIRED_WEAK"]
    failed = [row for row in best if row["repair_status"] in {"REPAIRED_FAILED", "DO_NOT_TRADE_RESEARCH_ONLY", "INSUFFICIENT_DATA"}]
    token = utc_stamp()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "md": out_dir / f"aegis_short_v3_failure_repair_{token}.md",
        "json": out_dir / f"aegis_short_v3_failure_repair_{token}.json",
        "all_modes_csv": out_dir / f"aegis_short_v3_failure_repair_all_modes_{token}.csv",
        "best_by_symbol_csv": out_dir / f"aegis_short_v3_failure_repair_best_by_symbol_{token}.csv",
        "repaired_confirmed_csv": out_dir / f"aegis_short_v3_failure_repair_repaired_confirmed_{token}.csv",
        "repaired_weak_csv": out_dir / f"aegis_short_v3_failure_repair_repaired_weak_{token}.csv",
        "failed_csv": out_dir / f"aegis_short_v3_failure_repair_failed_{token}.csv",
    }
    report = {
        "schema_version": "aegis_short_v3_failure_repair_v1",
        "created_at": utc_now().isoformat(),
        "mode": MODE,
        "side": SIDE,
        "symbols": symbols,
        "bounded_repair_only": True,
        "phase_i_reference_source": "frozen_user_supplied_phase_i_result",
        "disable_cross_context": bool(args.disable_cross_context),
        "original_lockbox_rows": original_rows,
        "all_modes": all_modes,
        "best_by_symbol": best,
        "repaired_confirmed": repaired_confirmed,
        "repaired_weak": repaired_weak,
        "failed": failed,
        "status_counts": dict(Counter(row["repair_status"] for row in best)),
        "errors": errors,
        "save_models": False,
        "shadow_models_generated": False,
        "active_manifest_touched": False,
        "live_inference_changed": False,
        "next_recommendation": (
            "consider_shadow_metadata_only_for_repaired_confirmed_after_review"
            if repaired_confirmed else "do_not_expand_shadow_candidates_continue_research"
        ),
        "paths": {name: str(path) for name, path in paths.items()},
    }
    paths["json"].write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(paths["md"], report)
    write_csv(paths["all_modes_csv"], all_modes)
    write_csv(paths["best_by_symbol_csv"], best)
    write_csv(paths["repaired_confirmed_csv"], repaired_confirmed)
    write_csv(paths["repaired_weak_csv"], repaired_weak)
    write_csv(paths["failed_csv"], failed)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Research-only failure-mode repair for SHORT V3 lockbox weak/failed symbols.")
    parser.add_argument("--symbols")
    parser.add_argument("--include-bnb-xrp", action="store_true")
    parser.add_argument("--disable-cross-context", action="store_true")
    parser.add_argument("--out-dir", default="/home/jasan/Develop")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--lockbox-test-ratio", type=float, default=0.20)
    parser.add_argument("--min-train-samples", type=int, default=1000)
    parser.add_argument("--min-test-samples", type=int, default=300)
    parser.add_argument("--fee-bps", type=float, default=8.0)
    parser.add_argument("--slippage-bps", type=float, default=3.0)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({
        "paths": report["paths"],
        "symbols": report["symbols"],
        "status_counts": report["status_counts"],
        "repaired_confirmed": [row["symbol"] for row in report["repaired_confirmed"]],
        "repaired_weak": [row["symbol"] for row in report["repaired_weak"]],
        "failed": [row["symbol"] for row in report["failed"]],
        "errors": report["errors"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
