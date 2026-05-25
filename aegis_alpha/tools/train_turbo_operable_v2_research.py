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
from aegis_alpha.turbo.operable_feature_builder_v3 import FEATURE_SETS, apply_feature_set  # noqa: E402
from aegis_alpha.turbo.recent_dataset import build_recent_dataset  # noqa: E402
from aegis_alpha.turbo.snapshot_utils import normalize_turbo_symbol  # noqa: E402
from aegis_alpha.turbo.train_operable_edge_v2 import MODEL_SCHEMA_VERSION, train_side_models  # noqa: E402


EXPECTED_SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "SUIUSDT",
    "LTCUSDT",
)
DEFAULT_MODEL_DIR = REPO_ROOT / "aegis_alpha" / "models" / "research" / "turbo_v2"
SUMMARY_COLUMNS = (
    "symbol",
    "side",
    "lookback_days",
    "horizon_candles",
    "research_status",
    "feature_set",
    "feature_count",
    "base_feature_count",
    "new_feature_count",
    "operable_v2_feature_count",
    "operable_v3_feature_count",
    "feature_schema_hash",
    "sample_count",
    "baseline_hit8_rate",
    "baseline_quality",
    "baseline_danger_rate",
    "v1_corr_hit8",
    "v1_corr_quality",
    "hit8_auc",
    "hit8_average_precision",
    "hit8_corr_hit8",
    "hit8_corr_quality",
    "hit8_top_hit_rate",
    "hit8_top_quality",
    "quality_spearman",
    "quality_corr_hit8",
    "quality_corr_quality",
    "quality_top_hit_rate",
    "quality_top_quality",
    "danger_auc",
    "danger_average_precision",
    "danger_corr_danger",
    "danger_corr_quality",
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_stamp() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%SZ")


def parse_symbols(raw: str | None, only_symbol: str | None, limit_symbols: int | None) -> list[str]:
    if only_symbol:
        symbols = [normalize_turbo_symbol(only_symbol)]
    elif raw:
        symbols = [normalize_turbo_symbol(part) for part in raw.split(",") if part.strip()]
    else:
        symbols = list(EXPECTED_SYMBOLS)
    symbols = list(dict.fromkeys(symbols))
    return symbols[:limit_symbols] if limit_symbols and limit_symbols > 0 else symbols


def parse_sides(value: str) -> list[str]:
    return ["long", "short"] if value.upper() == "BOTH" else [value.lower()]


def _dig(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def summary_row(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "symbol": result.get("symbol"),
        "side": result.get("side"),
        "lookback_days": result.get("lookback_days"),
        "horizon_candles": result.get("horizon_candles"),
        "research_status": result.get("research_status"),
        "feature_set": result.get("feature_set", "base"),
        "feature_count": result.get("feature_count"),
        "base_feature_count": result.get("base_feature_count"),
        "new_feature_count": result.get("new_feature_count"),
        "operable_v2_feature_count": result.get("operable_v2_feature_count"),
        "operable_v3_feature_count": result.get("operable_v3_feature_count"),
        "feature_schema_hash": result.get("feature_schema_hash"),
        "sample_count": result.get("sample_count"),
        "baseline_hit8_rate": _dig(result, "baseline_test", "hit8_rate"),
        "baseline_quality": _dig(result, "baseline_test", "avg_trade_quality"),
        "baseline_danger_rate": _dig(result, "baseline_test", "mae_danger_rate"),
        "v1_corr_hit8": _dig(result, "baseline_test", "v1_corr_hit8"),
        "v1_corr_quality": _dig(result, "baseline_test", "v1_corr_trade_quality"),
        "hit8_auc": _dig(result, "families", "hit8_classifier", "test_metrics", "roc_auc"),
        "hit8_average_precision": _dig(result, "families", "hit8_classifier", "test_metrics", "average_precision"),
        "hit8_corr_hit8": _dig(result, "families", "hit8_classifier", "test_comparisons", "prediction_vs_hit8"),
        "hit8_corr_quality": _dig(result, "families", "hit8_classifier", "test_comparisons", "prediction_vs_trade_quality"),
        "hit8_top_hit_rate": _dig(result, "families", "hit8_classifier", "top_decile", "hit8_rate"),
        "hit8_top_quality": _dig(result, "families", "hit8_classifier", "top_decile", "avg_trade_quality"),
        "quality_spearman": _dig(result, "families", "trade_quality_regressor", "test_metrics", "spearman"),
        "quality_corr_hit8": _dig(result, "families", "trade_quality_regressor", "test_comparisons", "prediction_vs_hit8"),
        "quality_corr_quality": _dig(result, "families", "trade_quality_regressor", "test_comparisons", "prediction_vs_trade_quality"),
        "quality_top_hit_rate": _dig(result, "families", "trade_quality_regressor", "top_decile", "hit8_rate"),
        "quality_top_quality": _dig(result, "families", "trade_quality_regressor", "top_decile", "avg_trade_quality"),
        "danger_auc": _dig(result, "families", "mae_danger_classifier", "test_metrics", "roc_auc"),
        "danger_average_precision": _dig(result, "families", "mae_danger_classifier", "test_metrics", "average_precision"),
        "danger_corr_danger": _dig(result, "families", "mae_danger_classifier", "test_comparisons", "prediction_vs_mae_danger"),
        "danger_corr_quality": _dig(result, "families", "mae_danger_classifier", "test_comparisons", "prediction_vs_trade_quality"),
    }


def bucket_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        for family, family_result in (result.get("families") or {}).items():
            for bucket in family_result.get("buckets") or []:
                rows.append({
                    "symbol": result.get("symbol"),
                    "side": result.get("side"),
                    "lookback_days": result.get("lookback_days"),
                    "horizon_candles": result.get("horizon_candles"),
                    "model_family": family,
                    **bucket,
                })
    return rows


def ranked_results(rows: list[dict[str, Any]], reverse: bool) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: float(row.get("quality_top_quality") or -999.0),
        reverse=reverse,
    )[:8]


def weighted_average(rows: list[dict[str, Any]], key: str) -> float | None:
    pairs = [
        (float(row[key]), int(row.get("sample_count") or 0))
        for row in rows
        if row.get(key) is not None and int(row.get("sample_count") or 0) > 0
    ]
    total = sum(weight for _, weight in pairs)
    return sum(value * weight for value, weight in pairs) / total if total else None


def global_metrics(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    return {
        key: weighted_average(rows, key)
        for key in (
            "baseline_hit8_rate",
            "baseline_quality",
            "baseline_danger_rate",
            "v1_corr_hit8",
            "v1_corr_quality",
            "hit8_auc",
            "hit8_corr_hit8",
            "hit8_corr_quality",
            "hit8_top_hit_rate",
            "hit8_top_quality",
            "quality_spearman",
            "quality_corr_hit8",
            "quality_corr_quality",
            "quality_top_hit_rate",
            "quality_top_quality",
            "danger_auc",
            "danger_corr_danger",
            "danger_corr_quality",
        )
    }


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _pct(value: Any) -> str:
    return "null" if value is None else f"{float(value) * 100:.2f}%"


def _num(value: Any) -> str:
    return "null" if value is None else f"{float(value):.4f}"


def write_markdown(path: Path, report: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        f"# Aegis Turbo V2 Research Training {report['created_at']}",
        "",
        "## Safety",
        "",
        "- Mode: `RESEARCH_ONLY`",
        f"- Save models: `{report['save_models']}`",
        f"- Feature set: `{report['feature_set']}`",
        f"- Model root: `{report['model_dir']}`",
        "- Live `active/` models and `active_manifest.json` are not modified.",
        "",
        "## Summary",
        "",
        f"- Trained symbol/side/lookback evaluations: `{len(rows)}`",
        f"- Status counts: `{report['research_status_counts']}`",
        "",
        "## Global Held-Out Metrics",
        "",
        "| V1 Corr Hit8 | V2 Hit8 Corr Hit8 | V1 Corr Quality | V2 Quality Corr Quality | Hit8 AUC | Danger AUC | Quality Top Quality |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {_num(report['global_metrics']['v1_corr_hit8'])} | {_num(report['global_metrics']['hit8_corr_hit8'])} | "
        f"{_num(report['global_metrics']['v1_corr_quality'])} | {_num(report['global_metrics']['quality_corr_quality'])} | "
        f"{_num(report['global_metrics']['hit8_auc'])} | {_num(report['global_metrics']['danger_auc'])} | "
        f"{_num(report['global_metrics']['quality_top_quality'])} |",
        "",
        "## Best Candidate Sides",
        "",
        "| Symbol/Side/Window | Status | Baseline Hit8 | Quality Top Hit8 | Quality Top Quality | Hit8 AUC | Danger AUC |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["best_candidates"]:
        lines.append(
            f"| {row['symbol']} {row['side']} {row['lookback_days']}d | {row['research_status']} | "
            f"{_pct(row['baseline_hit8_rate'])} | {_pct(row['quality_top_hit_rate'])} | "
            f"{_num(row['quality_top_quality'])} | {_num(row['hit8_auc'])} | {_num(row['danger_auc'])} |"
        )
    lines.extend([
        "",
        "## Weak Candidate Sides",
        "",
        "| Symbol/Side/Window | Status | Baseline Hit8 | Quality Top Hit8 | Quality Top Quality | Hit8 AUC | Danger AUC |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in report["weak_candidates"]:
        lines.append(
            f"| {row['symbol']} {row['side']} {row['lookback_days']}d | {row['research_status']} | "
            f"{_pct(row['baseline_hit8_rate'])} | {_pct(row['quality_top_hit_rate'])} | "
            f"{_num(row['quality_top_quality'])} | {_num(row['hit8_auc'])} | {_num(row['danger_auc'])} |"
        )
    lines.extend([
        "",
        "## V1 Versus V2",
        "",
        "- V1 comparison uses `long_net_return_12` / `short_net_return_12` correlations on the held-out test segment.",
        "- V2 comparisons use only held-out probability/prediction outputs against operable targets.",
        "- Results remain research evidence and are not suitable for live promotion without walk-forward validation.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_symbol_manifests(results: list[dict[str, Any]], model_dir: Path, stamp: str) -> None:
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        by_symbol.setdefault(str(result["symbol"]), []).append(result)
    for symbol, payloads in by_symbol.items():
        directory = model_dir / symbol / stamp
        directory.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": "aegis_turbo_v2_research_manifest_v1",
            "created_at": utc_now().isoformat(),
            "symbol": symbol,
            "promotion_status": "RESEARCH_ONLY",
            "not_live_active": True,
            "runs": payloads,
        }
        (directory / "research_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    symbols = parse_symbols(args.symbols, args.only_symbol, args.limit_symbols)
    sides = parse_sides(args.side)
    lookbacks = list(dict.fromkeys(args.lookback_days or list(DEFAULT_TURBO_CONFIG.lookback_days)))
    stamp = utc_stamp()
    model_dir = Path(args.model_dir)
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    context_markets: dict[str, Any] = {}
    if args.feature_set in {"operable_v3", "combined_v3"}:
        context_markets = {
            value: load_signal_market(DEFAULT_TURBO_CONFIG.config_path, symbol_override=value)
            for value in ("BTCUSDT", "ETHUSDT")
        }
    for symbol in symbols:
        try:
            market = load_signal_market(DEFAULT_TURBO_CONFIG.config_path, symbol_override=symbol)
            for lookback_days in lookbacks:
                dataset = build_recent_dataset(symbol, int(lookback_days), save=False, market=market)["dataset"]
                dataset = apply_feature_set(dataset, market, args.feature_set, context_markets=context_markets)
                for side in sides:
                    result = train_side_models(
                        dataset,
                        symbol=symbol,
                        side=side,
                        lookback_days=int(lookback_days),
                        horizon=int(args.horizon),
                        run_dir=model_dir / symbol / stamp,
                        save_models=not args.no_save_models,
                        fast=bool(args.fast),
                    )
                    results.append(result)
        except Exception as exc:
            errors.append({"symbol": symbol, "error": repr(exc)})
    if not args.no_save_models:
        write_symbol_manifests(results, model_dir, stamp)
    rows = [summary_row(result) for result in results]
    report: dict[str, Any] = {
        "schema_version": "aegis_turbo_v2_research_training_report_v1",
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "created_at": utc_now().isoformat(),
        "run_stamp": stamp,
        "mode": "RESEARCH_ONLY",
        "symbols": symbols,
        "sides": [side.upper() for side in sides],
        "lookback_days": lookbacks,
        "horizon_candles": int(args.horizon),
        "feature_set": args.feature_set,
        "fast": bool(args.fast),
        "save_models": not args.no_save_models,
        "model_dir": str(model_dir),
        "active_manifest_touched": False,
        "live_inference_changed": False,
        "results": results,
        "errors": errors,
        "research_status_counts": dict(Counter(row.get("research_status") for row in rows)),
        "global_metrics": global_metrics(rows),
        "best_candidates": ranked_results(rows, True),
        "weak_candidates": ranked_results(rows, False),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"aegis_turbo_v2_research_train_{stamp}.md"
    json_path = out_dir / f"aegis_turbo_v2_research_train_{stamp}.json"
    summary_path = out_dir / f"aegis_turbo_v2_research_summary_{stamp}.csv"
    buckets_path = out_dir / f"aegis_turbo_v2_research_buckets_{stamp}.csv"
    symbol_side_path = out_dir / f"aegis_turbo_v2_research_symbol_side_{stamp}.csv"
    report.update({
        "md_path": str(md_path),
        "json_path": str(json_path),
        "summary_csv_path": str(summary_path),
        "buckets_csv_path": str(buckets_path),
        "symbol_side_csv_path": str(symbol_side_path),
    })
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(md_path, report, rows)
    write_csv(summary_path, rows, SUMMARY_COLUMNS)
    write_csv(symbol_side_path, rows, SUMMARY_COLUMNS)
    buckets = bucket_rows(results)
    bucket_columns = sorted({key for row in buckets for key in row}) if buckets else ["symbol", "side", "model_family"]
    write_csv(buckets_path, buckets, bucket_columns)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Aegis Turbo V2 operable models under a research-only output path.")
    parser.add_argument("--symbols", help="Comma-separated symbols; defaults to all Turbo symbols.")
    parser.add_argument("--only-symbol", help="Train only one symbol; overrides --symbols.")
    parser.add_argument("--limit-symbols", type=int)
    parser.add_argument("--lookback-days", type=int, action="append")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--side", choices=("LONG", "SHORT", "BOTH"), default="BOTH")
    parser.add_argument("--feature-set", choices=FEATURE_SETS, default="base")
    parser.add_argument("--out-dir", default="/home/jasan/Develop")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--no-save-models", action="store_true")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({
        "md": report["md_path"],
        "json": report["json_path"],
        "summary_csv": report["summary_csv_path"],
        "buckets_csv": report["buckets_csv_path"],
        "symbol_side_csv": report["symbol_side_csv_path"],
        "model_dir": report["model_dir"],
        "research_status_counts": report["research_status_counts"],
        "errors": report["errors"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
