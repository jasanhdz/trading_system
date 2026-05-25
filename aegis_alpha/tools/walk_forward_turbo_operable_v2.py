#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.config import REPO_ROOT  # noqa: E402
from aegis_alpha.signals.common import load_signal_market  # noqa: E402
from aegis_alpha.turbo.config import DEFAULT_TURBO_CONFIG  # noqa: E402
from aegis_alpha.turbo.recent_dataset import build_recent_dataset  # noqa: E402
from aegis_alpha.turbo.snapshot_utils import normalize_turbo_symbol  # noqa: E402
from aegis_alpha.turbo.walk_forward_operable_v2 import (  # noqa: E402
    WALK_FORWARD_SCHEMA_VERSION,
    run_walk_forward,
)


DEFAULT_MODEL_DIR = REPO_ROOT / "aegis_alpha" / "models" / "research" / "turbo_v2_walkforward"
SECONDARY_SYMBOLS = ("SOLUSDT", "DOGEUSDT", "LTCUSDT")
SUMMARY_COLUMNS = (
    "symbol",
    "side",
    "lookback_days",
    "horizon_candles",
    "recommendation",
    "fold_count",
    "generated_fold_count",
    "valid_fold_count",
    "baseline_hit8_mean",
    "baseline_quality_mean",
    "baseline_danger_mean",
    "v1_corr_quality_mean",
    "v2_hit8_auc_mean",
    "v2_hit8_auc_min",
    "v2_quality_corr_mean",
    "v2_quality_corr_min",
    "v2_danger_auc_mean",
    "hit8_top_decile_lift_mean",
    "hit8_top_decile_lift_min",
    "quality_top_decile_lift_mean",
    "quality_top_decile_lift_min",
    "danger_filter_usefulness_mean",
    "stability_score",
    "decay_score",
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_stamp() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%SZ")


def parse_symbols(raw: str | None, include_secondary: bool) -> list[str]:
    values = [part.strip() for part in (raw or "ADAUSDT,AVAXUSDT").split(",") if part.strip()]
    if include_secondary:
        values.extend(SECONDARY_SYMBOLS)
    return list(dict.fromkeys(normalize_turbo_symbol(value) for value in values))


def parse_sides(value: str) -> list[str]:
    return ["short", "long"] if value.upper() == "BOTH" else [value.lower()]


def _dig(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _pct(value: Any) -> str:
    return "null" if value is None else f"{float(value) * 100:.2f}%"


def _num(value: Any) -> str:
    return "null" if value is None else f"{float(value):.4f}"


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fold_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        for fold in result.get("folds", []):
            rows.append({
                "symbol": fold.get("symbol"),
                "side": fold.get("side"),
                "lookback_days": fold.get("lookback_days"),
                "horizon_candles": fold.get("horizon_candles"),
                "fold": fold.get("fold"),
                "model_status": fold.get("model_status"),
                "train_count": _dig(fold, "split_samples", "train"),
                "validation_count": _dig(fold, "split_samples", "validation"),
                "test_count": _dig(fold, "split_samples", "test"),
                "baseline_hit8_rate": _dig(fold, "baseline_test", "hit8_rate"),
                "baseline_quality": _dig(fold, "baseline_test", "avg_trade_quality"),
                "baseline_danger": _dig(fold, "baseline_test", "mae_danger_rate"),
                "baseline_p90_mae": _dig(fold, "baseline_test", "p90_mae"),
                "v1_corr_hit8": _dig(fold, "v1_target_reference", "corr_hit8"),
                "v1_corr_quality": _dig(fold, "v1_target_reference", "corr_trade_quality"),
                "hit8_auc": _dig(fold, "families", "hit8_classifier", "test_metrics", "roc_auc"),
                "hit8_top_lift": _dig(fold, "families", "hit8_classifier", "top_decile", "hit8_lift_vs_baseline"),
                "hit8_top_quality": _dig(fold, "families", "hit8_classifier", "top_decile", "avg_trade_quality"),
                "quality_corr": _dig(fold, "families", "trade_quality_regressor", "test_metrics", "spearman"),
                "quality_top_lift": _dig(fold, "families", "trade_quality_regressor", "top_decile", "quality_lift_vs_baseline"),
                "quality_top_quality": _dig(fold, "families", "trade_quality_regressor", "top_decile", "avg_trade_quality"),
                "quality_top_p90_mae": _dig(fold, "families", "trade_quality_regressor", "top_decile", "p90_mae"),
                "danger_auc": _dig(fold, "families", "mae_danger_classifier", "test_metrics", "roc_auc"),
                "danger_filter_usefulness": _dig(fold, "families", "mae_danger_classifier", "usefulness_as_filter"),
            })
    return rows


def bucket_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        for fold in result.get("folds", []):
            for family, payload in (fold.get("families") or {}).items():
                for bucket in payload.get("buckets") or []:
                    rows.append({
                        "symbol": fold.get("symbol"),
                        "side": fold.get("side"),
                        "lookback_days": fold.get("lookback_days"),
                        "fold": fold.get("fold"),
                        "model_family": family,
                        **bucket,
                    })
    return rows


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        f"# Aegis Turbo V2 Walk-Forward {report['created_at']}",
        "",
        "## Safety",
        "",
        "- Mode: `RESEARCH_ONLY`",
        f"- Save fold models: `{report['save_fold_models']}`",
        f"- Model root: `{report['model_dir']}`",
        "- No live inference, `active/` model or `active_manifest.json` is modified.",
        "- V1 figures use the held-out V1 outcome target as a reference, not an out-of-sample V1 model prediction.",
        "",
        "## Summary",
        "",
        f"- Status counts: `{report['status_counts']}`",
        f"- Symbols: `{', '.join(report['symbols'])}`",
        f"- Sides: `{', '.join(report['sides'])}`",
        "",
        "| Symbol / Side | Status | Valid Folds | Base Hit8 | V2 Hit8 AUC | Hit8 Lift | Quality Lift | Quality Corr | Danger AUC | Stability |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["summaries"]:
        lines.append(
            f"| {row['symbol']} {row['side']} | {row['recommendation']} | {row['valid_fold_count']}/{row['fold_count']} | "
            f"{_pct(row['baseline_hit8_mean'])} | {_num(row['v2_hit8_auc_mean'])} | "
            f"{_pct(row['hit8_top_decile_lift_mean'])} | {_num(row['quality_top_decile_lift_mean'])} | "
            f"{_num(row['v2_quality_corr_mean'])} | {_num(row['v2_danger_auc_mean'])} | {_num(row['stability_score'])} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `WALK_FORWARD_PROMISING` requires improvement in multiple temporal folds and a non-adverse latest fold.",
        "- `WALK_FORWARD_MIXED` is research evidence only; it is not sufficient for live shadow integration by itself.",
        "- `WALK_FORWARD_BAD` indicates that V2 selection did not persist out of sample for this slice.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    symbols = parse_symbols(args.symbols, bool(args.include_secondary))
    sides = parse_sides(args.side)
    stamp = utc_stamp()
    model_dir = Path(args.model_dir)
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for symbol in symbols:
        try:
            market = load_signal_market(DEFAULT_TURBO_CONFIG.config_path, symbol_override=symbol)
            dataset = build_recent_dataset(symbol, int(args.lookback_days), save=False, market=market)["dataset"]
            for side in sides:
                results.append(run_walk_forward(
                    dataset,
                    symbol=symbol,
                    side=side,
                    lookback_days=int(args.lookback_days),
                    horizon=int(args.horizon),
                    fold_count=int(args.fold_count),
                    train_ratio=float(args.train_ratio),
                    validation_ratio=float(args.validation_ratio),
                    test_ratio=float(args.test_ratio),
                    expanding_window=bool(args.expanding_window),
                    min_train_samples=int(args.min_train_samples),
                    min_test_samples=int(args.min_test_samples),
                    run_dir=model_dir / symbol / stamp,
                    save_models=bool(args.save_fold_models),
                    fast=bool(args.fast),
                ))
        except Exception as exc:
            errors.append({"symbol": symbol, "error": repr(exc)})
    summaries = [result["summary"] for result in results]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"aegis_turbo_v2_walkforward_{stamp}.md"
    json_path = out_dir / f"aegis_turbo_v2_walkforward_{stamp}.json"
    summary_path = out_dir / f"aegis_turbo_v2_walkforward_summary_{stamp}.csv"
    folds_path = out_dir / f"aegis_turbo_v2_walkforward_folds_{stamp}.csv"
    buckets_path = out_dir / f"aegis_turbo_v2_walkforward_buckets_{stamp}.csv"
    report: dict[str, Any] = {
        "schema_version": WALK_FORWARD_SCHEMA_VERSION,
        "created_at": utc_now().isoformat(),
        "run_stamp": stamp,
        "mode": "RESEARCH_ONLY",
        "symbols": symbols,
        "sides": [side.upper() for side in sides],
        "lookback_days": int(args.lookback_days),
        "horizon_candles": int(args.horizon),
        "fold_count": int(args.fold_count),
        "fold_configuration": {
            "train_ratio": float(args.train_ratio),
            "validation_ratio": float(args.validation_ratio),
            "test_ratio": float(args.test_ratio),
            "expanding_window": bool(args.expanding_window),
            "min_train_samples": int(args.min_train_samples),
            "min_test_samples": int(args.min_test_samples),
        },
        "fast": bool(args.fast),
        "save_fold_models": bool(args.save_fold_models),
        "model_dir": str(model_dir),
        "active_manifest_touched": False,
        "live_inference_changed": False,
        "v1_reference_warning": "outcome_target_reference_not_v1_model_prediction",
        "summaries": summaries,
        "results": results,
        "errors": errors,
        "status_counts": dict(Counter(row.get("recommendation") for row in summaries)),
        "md_path": str(md_path),
        "json_path": str(json_path),
        "summary_csv_path": str(summary_path),
        "folds_csv_path": str(folds_path),
        "buckets_csv_path": str(buckets_path),
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(md_path, report)
    write_csv(summary_path, summaries, SUMMARY_COLUMNS)
    fold_payloads = fold_rows(results)
    fold_columns = sorted({key for row in fold_payloads for key in row}) if fold_payloads else ["symbol", "side"]
    write_csv(folds_path, fold_payloads, fold_columns)
    bucket_payloads = bucket_rows(results)
    bucket_columns = sorted({key for row in bucket_payloads for key in row}) if bucket_payloads else ["symbol", "side", "model_family"]
    write_csv(buckets_path, bucket_payloads, bucket_columns)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run research-only walk-forward validation for Aegis Turbo operable V2 candidates.")
    parser.add_argument("--symbols", default="ADAUSDT,AVAXUSDT")
    parser.add_argument("--side", choices=("LONG", "SHORT", "BOTH"), default="SHORT")
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--fold-count", type=int, default=4)
    parser.add_argument("--train-ratio", type=float, default=0.50)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--min-train-samples", type=int, default=1000)
    parser.add_argument("--min-test-samples", type=int, default=300)
    window = parser.add_mutually_exclusive_group()
    window.add_argument("--expanding-window", dest="expanding_window", action="store_true", default=True)
    window.add_argument("--sliding-window", dest="expanding_window", action="store_false")
    parser.add_argument("--out-dir", default="/home/jasan/Develop")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--save-fold-models", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--include-secondary", action="store_true")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({
        "md": report["md_path"],
        "json": report["json_path"],
        "summary_csv": report["summary_csv_path"],
        "folds_csv": report["folds_csv_path"],
        "buckets_csv": report["buckets_csv_path"],
        "status_counts": report["status_counts"],
        "save_fold_models": report["save_fold_models"],
        "errors": report["errors"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
