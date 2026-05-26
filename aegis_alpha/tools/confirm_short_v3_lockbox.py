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
from aegis_alpha.tools.evaluate_ada_short_operable_v2_matrix import finite, validate_research_model_dir  # noqa: E402
from aegis_alpha.turbo.config import DEFAULT_TURBO_CONFIG  # noqa: E402
from aegis_alpha.turbo.operable_feature_builder_v3 import FEATURE_SETS, apply_feature_set  # noqa: E402
from aegis_alpha.turbo.recent_dataset import build_recent_dataset  # noqa: E402
from aegis_alpha.turbo.snapshot_utils import normalize_turbo_symbol  # noqa: E402
from aegis_alpha.turbo.walk_forward_operable_v2 import run_walk_forward, train_fold_models  # noqa: E402


MODE = "RESEARCH_ONLY"
SIDE = "SHORT"
DEFAULT_MODEL_DIR = REPO_ROOT / "aegis_alpha" / "models" / "research" / "turbo_v3_short_lockbox"
LOCKBOX_MODES = ("last-block", "rolling-forward", "recent-only")
DEFAULT_FROZEN_CONFIGS: tuple[tuple[str, str, int, int], ...] = (
    ("LINKUSDT", "operable_v3", 14, 12),
    ("LTCUSDT", "operable_v3", 14, 12),
    ("ADAUSDT", "operable_v2", 30, 24),
    ("DOGEUSDT", "operable_v3", 30, 24),
    ("SOLUSDT", "operable_v2", 14, 12),
    ("AVAXUSDT", "operable_v3", 14, 12),
    ("BTCUSDT", "combined_v3", 30, 12),
    ("ETHUSDT", "operable_v3", 30, 24),
    ("SUIUSDT", "combined_v3", 7, 12),
)
CSV_COLUMNS = (
    "symbol",
    "side",
    "feature_set",
    "lookback_days",
    "horizon_candles",
    "lockbox_mode",
    "lockbox_status",
    "train_samples",
    "validation_samples",
    "test_samples",
    "valid_fold_count",
    "feature_count",
    "feature_schema_hash",
    "baseline_hit8",
    "baseline_quality",
    "baseline_mae_danger",
    "baseline_p90_mae",
    "hit8_auc",
    "hit8_average_precision",
    "hit8_top_decile_lift",
    "hit8_top_decile_rate",
    "quality_corr",
    "quality_top_decile_lift",
    "quality_top_decile_mean",
    "danger_auc",
    "danger_filter_usefulness",
    "top_decile_p90_mae",
    "p90_mae_delta",
    "top_decile_mae_danger_rate",
    "net_quality_lift_after_cost_proxy",
    "top_decile_net_quality_after_cost_proxy",
    "latest_fold_quality_lift",
    "latest_fold_hit8_lift",
    "latest_fold_p90_mae_delta",
    "fold_consistency_score",
    "lockbox_vs_prior_decay",
    "reason",
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_stamp() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%SZ")


def default_frozen_configs() -> list[dict[str, Any]]:
    return [
        {
            "symbol": symbol,
            "side": SIDE,
            "feature_set": feature_set,
            "lookback_days": lookback_days,
            "horizon_candles": horizon,
        }
        for symbol, feature_set, lookback_days, horizon in DEFAULT_FROZEN_CONFIGS
    ]


def _normalized_config(raw: dict[str, Any]) -> dict[str, Any]:
    feature_set = str(raw.get("feature_set", "")).lower()
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unsupported frozen feature_set: {feature_set}")
    side = str(raw.get("side", SIDE)).upper()
    if side != SIDE:
        raise ValueError(f"lockbox is restricted to SHORT configs: {side}")
    horizon = raw.get("horizon_candles", raw.get("horizon"))
    return {
        "symbol": normalize_turbo_symbol(str(raw["symbol"])),
        "side": side,
        "feature_set": feature_set,
        "lookback_days": int(raw["lookback_days"]),
        "horizon_candles": int(horizon),
    }


def load_frozen_configs(configs_json: str | None, symbols_raw: str | None) -> list[dict[str, Any]]:
    if configs_json:
        payload = json.loads(Path(configs_json).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("strong_best", payload.get("configs", payload.get("best_by_symbol", [])))
        configs = [_normalized_config(dict(item)) for item in payload]
    else:
        configs = default_frozen_configs()
    requested = None
    if symbols_raw:
        requested = {
            normalize_turbo_symbol(value.strip())
            for value in symbols_raw.split(",")
            if value.strip()
        }
    selected = [config for config in configs if requested is None or config["symbol"] in requested]
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for config in selected:
        if config["symbol"] in seen:
            raise ValueError(f"multiple frozen configs for symbol: {config['symbol']}")
        seen.add(config["symbol"])
        unique.append(config)
    return unique


def build_last_block_fold(
    sample_count: int,
    *,
    test_ratio: float,
    min_train_samples: int,
    min_test_samples: int,
    recent_only: bool = False,
) -> dict[str, Any] | None:
    if sample_count <= 0 or test_ratio <= 0.0 or test_ratio >= 1.0:
        return None
    test_size = min_test_samples if recent_only else max(min_test_samples, int(sample_count * test_ratio))
    validation_size = max(1, int(sample_count * 0.15))
    test_start = sample_count - test_size
    validation_start = test_start - validation_size
    if validation_start < min_train_samples or test_start <= validation_start:
        return None
    split = {
        "train": np.arange(0, validation_start, dtype=np.int64),
        "validation": np.arange(validation_start, test_start, dtype=np.int64),
        "test": np.arange(test_start, sample_count, dtype=np.int64),
    }
    return {
        "fold": 1,
        "expanding_window": True,
        "lockbox": True,
        "ranges": {
            name: {"start": int(indices[0]), "end": int(indices[-1]), "count": int(len(indices))}
            for name, indices in split.items()
        },
        **split,
    }


def _family(fold: dict[str, Any], family: str, *keys: str) -> Any:
    current: Any = (fold.get("families") or {}).get(family, {})
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _mean(values: list[Any]) -> float | None:
    numbers = [finite(value, float("nan")) for value in values]
    numbers = [number for number in numbers if math.isfinite(number)]
    return float(np.mean(numbers)) if numbers else None


def lockbox_row(
    config: dict[str, Any],
    folds: list[dict[str, Any]],
    *,
    lockbox_mode: str,
    fee_bps: float,
    slippage_bps: float,
    min_test_samples: int,
    strict: bool,
) -> dict[str, Any]:
    valid = [fold for fold in folds if fold.get("model_status") == "trained"]
    latest = valid[-1] if valid else (folds[-1] if folds else {})
    baseline = latest.get("baseline_test", {}) if latest else {}
    hit_top = _family(latest, "hit8_classifier", "top_decile") or {}
    quality_top = _family(latest, "trade_quality_regressor", "top_decile") or {}
    danger_top = _family(latest, "mae_danger_classifier", "top_decile") or {}
    cost_proxy = (float(fee_bps) + float(slippage_bps)) / 10000.0
    quality_lift = quality_top.get("quality_lift_vs_baseline")
    test_samples = int((latest.get("split_samples") or {}).get("test", 0))
    row = {
        **config,
        "lockbox_mode": lockbox_mode,
        "train_samples": int((latest.get("split_samples") or {}).get("train", 0)),
        "validation_samples": int((latest.get("split_samples") or {}).get("validation", 0)),
        "test_samples": test_samples,
        "valid_fold_count": len(valid),
        "model_status": latest.get("model_status", "not_evaluated"),
        "feature_count": latest.get("feature_count"),
        "feature_schema_hash": latest.get("feature_schema_hash"),
        "feature_diagnostics": latest.get("feature_diagnostics"),
        "baseline_hit8": baseline.get("hit8_rate"),
        "baseline_quality": baseline.get("avg_trade_quality"),
        "baseline_mae_danger": baseline.get("mae_danger_rate"),
        "baseline_p90_mae": baseline.get("p90_mae"),
        "hit8_auc": _family(latest, "hit8_classifier", "test_metrics", "roc_auc"),
        "hit8_average_precision": _family(latest, "hit8_classifier", "test_metrics", "average_precision"),
        "hit8_top_decile_lift": hit_top.get("hit8_lift_vs_baseline"),
        "hit8_top_decile_rate": hit_top.get("hit8_rate"),
        "hit8_baseline_rate": baseline.get("hit8_rate"),
        "quality_corr": _family(latest, "trade_quality_regressor", "test_metrics", "spearman"),
        "quality_top_decile_lift": quality_lift,
        "quality_top_decile_mean": quality_top.get("avg_trade_quality"),
        "quality_baseline_mean": baseline.get("avg_trade_quality"),
        "quality_top_decile_min_if_available": quality_top.get("avg_trade_quality"),
        "danger_auc": _family(latest, "mae_danger_classifier", "test_metrics", "roc_auc"),
        "danger_filter_usefulness": _family(latest, "mae_danger_classifier", "usefulness_as_filter"),
        "danger_top_bucket_actual_danger": danger_top.get("mae_danger_rate"),
        "danger_baseline_rate": baseline.get("mae_danger_rate"),
        "top_decile_p90_mae": quality_top.get("p90_mae"),
        "p90_mae_delta": quality_top.get("p90_mae_delta_vs_baseline"),
        "top_decile_mae_danger_rate": quality_top.get("mae_danger_rate"),
        "estimated_fee_bps": float(fee_bps),
        "estimated_slippage_bps": float(slippage_bps),
        "net_quality_lift_after_cost_proxy": (
            finite(quality_lift) - cost_proxy if quality_lift is not None else None
        ),
        "top_decile_net_quality_after_cost_proxy": (
            finite(quality_top.get("avg_trade_quality")) - cost_proxy
            if quality_top.get("avg_trade_quality") is not None else None
        ),
        "latest_fold_quality_lift": quality_lift,
        "latest_fold_hit8_lift": hit_top.get("hit8_lift_vs_baseline"),
        "latest_fold_p90_mae_delta": quality_top.get("p90_mae_delta_vs_baseline"),
        "fold_consistency_score": _mean([
            (
                float((_family(fold, "hit8_classifier", "top_decile", "hit8_lift_vs_baseline") or 0.0) > 0.0)
                + float((_family(fold, "trade_quality_regressor", "top_decile", "quality_lift_vs_baseline") or 0.0) > 0.0)
            ) / 2.0
            for fold in valid
        ]),
        "lockbox_vs_prior_decay": (
            finite(quality_lift) - finite(_mean([
                _family(fold, "trade_quality_regressor", "top_decile", "quality_lift_vs_baseline")
                for fold in valid[:-1]
            ]))
            if quality_lift is not None and len(valid) > 1 else None
        ),
    }
    row["lockbox_status"] = classify_lockbox_candidate(row, min_test_samples=min_test_samples, strict=strict)
    row["reason"] = lockbox_reason(row)
    return row


def classify_lockbox_candidate(
    row: dict[str, Any],
    *,
    min_test_samples: int = 300,
    strict: bool = False,
) -> str:
    if row.get("config_error") or row.get("model_status") != "trained" or int(row.get("test_samples") or 0) < min_test_samples:
        return "LOCKBOX_INSUFFICIENT_DATA"
    p90_ratio = (
        finite(row.get("top_decile_p90_mae")) / max(finite(row.get("baseline_p90_mae")), 1e-12)
        if row.get("top_decile_p90_mae") is not None and row.get("baseline_p90_mae") is not None
        else float("inf")
    )
    auc_floor = 0.55 if strict else 0.53
    corr_floor = 0.0 if strict else -0.02
    confirmed = (
        finite(row.get("hit8_top_decile_lift")) > 0.0
        and finite(row.get("quality_top_decile_lift")) > 0.0
        and finite(row.get("net_quality_lift_after_cost_proxy")) > 0.0
        and finite(row.get("latest_fold_quality_lift")) > 0.0
        and finite(row.get("hit8_auc")) > auc_floor
        and finite(row.get("quality_corr"), -1.0) >= corr_floor
        and p90_ratio <= (1.10 if strict else 1.15)
        and finite(row.get("danger_filter_usefulness"), 0.0) >= -0.05
    )
    if confirmed:
        return "LOCKBOX_CONFIRMED"
    failed = (
        finite(row.get("hit8_top_decile_lift")) < 0.0
        or finite(row.get("quality_top_decile_lift")) < 0.0
        or finite(row.get("net_quality_lift_after_cost_proxy")) <= 0.0
        or p90_ratio > 1.30
    )
    return "LOCKBOX_FAILED" if failed else "LOCKBOX_WEAK"


def lockbox_reason(row: dict[str, Any]) -> str:
    status = row["lockbox_status"]
    if status == "LOCKBOX_CONFIRMED":
        return "positive_lockbox_lifts_cost_adjusted_and_risk_controlled"
    if status == "LOCKBOX_INSUFFICIENT_DATA":
        return "insufficient_lockbox_samples_or_model_diversity"
    if status == "LOCKBOX_FAILED":
        return "negative_or_risk_uncontrolled_lockbox_outcome"
    return "partial_lockbox_improvement_not_confirmed"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _num(value: Any) -> str:
    return "null" if value is None else f"{float(value):.4f}"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        f"# Aegis SHORT V3 Lockbox Confirmation {report['created_at']}",
        "",
        "## Safety",
        "",
        "- Mode: `RESEARCH_ONLY`.",
        "- Configurations are frozen from Phase H; no alternatives are searched.",
        "- No model artifacts are saved; no `active/` models or active manifests are modified.",
        "- No live inference, YAML, thresholds or PM2 action is involved.",
        "",
        "## Lockbox Results",
        "",
        "| Symbol | Config | Status | Samples | Hit8 AUC | Hit8 Lift | Quality Lift | Net Quality Lift | Quality Corr | Danger AUC | P90 Delta |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["results"]:
        lines.append(
            f"| {row['symbol']} | {row['feature_set']} {row['lookback_days']}d h{row['horizon_candles']} | "
            f"{row['lockbox_status']} | {row['test_samples']} | {_num(row.get('hit8_auc'))} | "
            f"{_num(row.get('hit8_top_decile_lift'))} | {_num(row.get('quality_top_decile_lift'))} | "
            f"{_num(row.get('net_quality_lift_after_cost_proxy'))} | {_num(row.get('quality_corr'))} | "
            f"{_num(row.get('danger_auc'))} | {_num(row.get('p90_mae_delta'))} |"
        )
    for key, title in (("confirmed", "Confirmed Candidates"), ("weak", "Weak Candidates"), ("failed", "Failed Candidates")):
        lines.extend(["", f"## {title}", ""])
        rows = report[key]
        if not rows:
            lines.append("- None.")
        for row in rows:
            lines.append(f"- `{row['symbol']}`: `{row['feature_set']} {row['lookback_days']}d h{row['horizon_candles']}`.")
    lines.extend(["", "## Decision", "", f"- Recommendation: `{report['next_recommendation']}`.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _evaluate_config(
    config: dict[str, Any],
    dataset: dict[str, Any],
    args: argparse.Namespace,
    model_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_dir = model_dir / config["symbol"] / f"{config['feature_set']}_{config['lookback_days']}d_h{config['horizon_candles']}"
    if args.lockbox_mode == "rolling-forward":
        result = run_walk_forward(
            dataset,
            symbol=config["symbol"],
            side=SIDE.lower(),
            lookback_days=config["lookback_days"],
            horizon=config["horizon_candles"],
            fold_count=int(args.fold_count),
            train_ratio=0.50,
            validation_ratio=0.15,
            test_ratio=float(args.lockbox_test_ratio),
            expanding_window=True,
            min_train_samples=int(args.min_train_samples),
            min_test_samples=int(args.min_test_samples),
            run_dir=run_dir,
            save_models=False,
            fast=bool(args.fast),
        )
        return result["folds"], result["summary"]
    fold = build_last_block_fold(
        len(np.asarray(dataset["X"])),
        test_ratio=float(args.lockbox_test_ratio),
        min_train_samples=int(args.min_train_samples),
        min_test_samples=int(args.min_test_samples),
        recent_only=args.lockbox_mode == "recent-only",
    )
    if fold is None:
        return [], {}
    fold_result = train_fold_models(
        dataset,
        symbol=config["symbol"],
        side=SIDE.lower(),
        lookback_days=config["lookback_days"],
        horizon=config["horizon_candles"],
        fold=fold,
        run_dir=run_dir,
        save_models=False,
        fast=bool(args.fast),
    )
    return [fold_result], {}


def run(args: argparse.Namespace) -> dict[str, Any]:
    configs = load_frozen_configs(args.configs_json, args.symbols)
    model_dir = Path(args.model_dir)
    validate_research_model_dir(model_dir)
    context_markets: dict[str, Any] = {}
    if any(config["feature_set"] in {"operable_v3", "combined_v3"} for config in configs):
        context_markets = {
            symbol: load_signal_market(DEFAULT_TURBO_CONFIG.config_path, symbol_override=symbol)
            for symbol in ("BTCUSDT", "ETHUSDT")
        }
    results: list[dict[str, Any]] = []
    folds_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for config in configs:
        try:
            market = load_signal_market(DEFAULT_TURBO_CONFIG.config_path, symbol_override=config["symbol"])
            base = build_recent_dataset(config["symbol"], config["lookback_days"], save=False, market=market)["dataset"]
            dataset = apply_feature_set(base, market, config["feature_set"], context_markets=context_markets)
            folds, _summary = _evaluate_config(config, dataset, args, model_dir)
            row = lockbox_row(
                config,
                folds,
                lockbox_mode=args.lockbox_mode,
                fee_bps=args.fee_bps,
                slippage_bps=args.slippage_bps,
                min_test_samples=args.min_test_samples,
                strict=args.strict,
            )
            results.append(row)
            for fold in folds:
                folds_rows.append({
                    **config,
                    "lockbox_mode": args.lockbox_mode,
                    "fold": fold.get("fold"),
                    "model_status": fold.get("model_status"),
                    "test_samples": (fold.get("split_samples") or {}).get("test"),
                    "baseline_hit8": (fold.get("baseline_test") or {}).get("hit8_rate"),
                    "hit8_lift": _family(fold, "hit8_classifier", "top_decile", "hit8_lift_vs_baseline"),
                    "quality_lift": _family(fold, "trade_quality_regressor", "top_decile", "quality_lift_vs_baseline"),
                    "p90_mae_delta": _family(fold, "trade_quality_regressor", "top_decile", "p90_mae_delta_vs_baseline"),
                })
        except Exception as exc:
            error = {**config, "error": repr(exc)}
            errors.append(error)
            results.append({
                **config,
                "lockbox_mode": args.lockbox_mode,
                "lockbox_status": "LOCKBOX_INSUFFICIENT_DATA",
                "reason": "configuration_evaluation_error",
                "test_samples": 0,
                "config_error": repr(exc),
            })
    confirmed = [row for row in results if row["lockbox_status"] == "LOCKBOX_CONFIRMED"]
    weak = [row for row in results if row["lockbox_status"] == "LOCKBOX_WEAK"]
    failed = [row for row in results if row["lockbox_status"] == "LOCKBOX_FAILED"]
    insufficient = [row for row in results if row["lockbox_status"] == "LOCKBOX_INSUFFICIENT_DATA"]
    token = utc_stamp()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "md": out_dir / f"aegis_short_v3_lockbox_confirmation_{token}.md",
        "json": out_dir / f"aegis_short_v3_lockbox_confirmation_{token}.json",
        "summary_csv": out_dir / f"aegis_short_v3_lockbox_summary_{token}.csv",
        "folds_csv": out_dir / f"aegis_short_v3_lockbox_folds_{token}.csv",
        "confirmed_csv": out_dir / f"aegis_short_v3_lockbox_confirmed_{token}.csv",
        "weak_csv": out_dir / f"aegis_short_v3_lockbox_weak_{token}.csv",
        "failed_csv": out_dir / f"aegis_short_v3_lockbox_failed_{token}.csv",
    }
    report: dict[str, Any] = {
        "schema_version": "aegis_short_v3_lockbox_confirmation_v1",
        "created_at": utc_now().isoformat(),
        "mode": MODE,
        "side": SIDE,
        "lockbox_mode": args.lockbox_mode,
        "strict": bool(args.strict),
        "lockbox_test_ratio": float(args.lockbox_test_ratio),
        "frozen_configs": configs,
        "evaluated_config_count": len(configs),
        "save_models": False,
        "shadow_models_generated": False,
        "active_manifest_touched": False,
        "live_inference_changed": False,
        "results": results,
        "folds": folds_rows,
        "confirmed": confirmed,
        "weak": weak,
        "failed": failed,
        "insufficient_data": insufficient,
        "status_counts": dict(Counter(row["lockbox_status"] for row in results)),
        "errors": errors,
        "next_recommendation": (
            "generate_shadow_metadata_only_artifacts_for_lockbox_confirmed_default_off"
            if confirmed else "do_not_generate_shadow_artifacts_continue_research"
        ),
        "paths": {key: str(value) for key, value in paths.items()},
    }
    paths["json"].write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(paths["md"], report)
    write_csv(paths["summary_csv"], results)
    write_csv(paths["confirmed_csv"], confirmed)
    write_csv(paths["weak_csv"], weak)
    write_csv(paths["failed_csv"], failed)
    fold_fields = sorted({key for row in folds_rows for key in row}) or ["symbol", "side"]
    with paths["folds_csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fold_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(folds_rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Research-only lockbox confirmation for frozen SHORT V3 candidates.")
    parser.add_argument("--configs-json")
    parser.add_argument("--symbols")
    parser.add_argument("--out-dir", default="/home/jasan/Develop")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--fold-count", type=int, default=4)
    parser.add_argument("--lockbox-mode", choices=LOCKBOX_MODES, default="last-block")
    parser.add_argument("--lockbox-test-ratio", type=float, default=0.20)
    parser.add_argument("--min-train-samples", type=int, default=1000)
    parser.add_argument("--min-test-samples", type=int, default=300)
    parser.add_argument("--fee-bps", type=float, default=8.0)
    parser.add_argument("--slippage-bps", type=float, default=3.0)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({
        "paths": report["paths"],
        "evaluated_config_count": report["evaluated_config_count"],
        "status_counts": report["status_counts"],
        "confirmed": [row["symbol"] for row in report["confirmed"]],
        "weak": [row["symbol"] for row in report["weak"]],
        "failed": [row["symbol"] for row in report["failed"]],
        "errors": report["errors"],
        "next_recommendation": report["next_recommendation"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
