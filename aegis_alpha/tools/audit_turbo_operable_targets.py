#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aegis_alpha.signals.common import load_signal_market
from aegis_alpha.turbo.config import DEFAULT_TURBO_CONFIG
from aegis_alpha.turbo.recent_dataset import (
    OPERABLE_TARGET_NAMES,
    OPERABLE_TARGET_SCHEMA_VERSION,
    TRADE_QUALITY_FORMULA,
    build_recent_dataset,
)
from aegis_alpha.turbo.snapshot_utils import normalize_turbo_symbol


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

SUMMARY_COLUMNS = (
    "symbol",
    "side",
    "lookback_days",
    "horizon_candles",
    "sample_count",
    "hit5_before_minus5_rate",
    "hit8_before_minus5_rate",
    "hit10_before_minus8_rate",
    "hit15_before_minus10_rate",
    "avg_mfe",
    "avg_mae",
    "p50_mae",
    "p75_mae",
    "p90_mae",
    "p95_mae",
    "avg_mfe_mae_ratio",
    "avg_trade_quality",
    "p25_trade_quality",
    "p50_trade_quality",
    "p75_trade_quality",
    "mae_danger_rate",
    "mae_severe_rate",
    "ambiguous_same_candle_rate",
    "corr_v1_net_return_trade_quality",
    "corr_v1_net_return_hit8",
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_stamp() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%SZ")


def parse_symbols(raw: str | None) -> list[str]:
    if not raw:
        return list(EXPECTED_SYMBOLS)
    return list(dict.fromkeys(normalize_turbo_symbol(part) for part in raw.split(",") if part.strip()))


def metric_mean(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else None


def metric_quantile(values: np.ndarray, q: float) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, q)) if len(values) else None


def safe_corr(left: np.ndarray, right: np.ndarray) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 3 or float(np.std(x[valid])) <= 1e-12 or float(np.std(y[valid])) <= 1e-12:
        return None
    result = float(np.corrcoef(x[valid], y[valid])[0, 1])
    return result if math.isfinite(result) else None


def summarize_arrays(
    symbol: str,
    side: str,
    lookback_days: int | str,
    horizon: int,
    datasets: list[dict[str, Any]],
) -> dict[str, Any]:
    def concatenate(key: str) -> np.ndarray:
        values = [np.asarray(dataset[key]) for dataset in datasets if key in dataset]
        return np.concatenate(values) if values else np.asarray([], dtype=np.float32)

    prefix = f"{side}_"
    suffix = f"_{horizon}"
    hit5 = concatenate(f"{prefix}hit5_before_minus5{suffix}")
    hit8 = concatenate(f"{prefix}hit8_before_minus5{suffix}")
    hit10 = concatenate(f"{prefix}hit10_before_minus8{suffix}")
    hit15 = concatenate(f"{prefix}hit15_before_minus10{suffix}")
    mfe = concatenate(f"{prefix}mfe{suffix}")
    mae = concatenate(f"{prefix}mae{suffix}")
    ratio = concatenate(f"{prefix}mfe_mae_ratio{suffix}")
    quality = concatenate(f"{prefix}trade_quality{suffix}")
    danger = concatenate(f"{prefix}mae_danger{suffix}")
    severe = concatenate(f"{prefix}mae_severe{suffix}")
    ambiguous = concatenate(f"{prefix}ambiguous_hit_stop{suffix}")
    v1_return = concatenate(f"{side}_net_return_12") if horizon == 12 else np.asarray([], dtype=np.float32)
    return {
        "symbol": symbol,
        "side": side.upper(),
        "lookback_days": lookback_days,
        "horizon_candles": horizon,
        "sample_count": int(len(quality)),
        "hit5_before_minus5_rate": metric_mean(hit5),
        "hit8_before_minus5_rate": metric_mean(hit8),
        "hit10_before_minus8_rate": metric_mean(hit10),
        "hit15_before_minus10_rate": metric_mean(hit15),
        "avg_mfe": metric_mean(mfe),
        "avg_mae": metric_mean(mae),
        "p50_mae": metric_quantile(mae, 0.50),
        "p75_mae": metric_quantile(mae, 0.75),
        "p90_mae": metric_quantile(mae, 0.90),
        "p95_mae": metric_quantile(mae, 0.95),
        "avg_mfe_mae_ratio": metric_mean(ratio),
        "avg_trade_quality": metric_mean(quality),
        "p25_trade_quality": metric_quantile(quality, 0.25),
        "p50_trade_quality": metric_quantile(quality, 0.50),
        "p75_trade_quality": metric_quantile(quality, 0.75),
        "mae_danger_rate": metric_mean(danger),
        "mae_severe_rate": metric_mean(severe),
        "ambiguous_same_candle_rate": metric_mean(ambiguous),
        "corr_v1_net_return_trade_quality": safe_corr(v1_return, quality) if horizon == 12 else None,
        "corr_v1_net_return_hit8": safe_corr(v1_return, hit8) if horizon == 12 else None,
    }


def distribution_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    metrics = (
        "hit8_before_minus5_rate",
        "avg_mfe",
        "avg_mae",
        "p90_mae",
        "avg_trade_quality",
        "mae_danger_rate",
        "mae_severe_rate",
        "ambiguous_same_candle_rate",
        "corr_v1_net_return_trade_quality",
        "corr_v1_net_return_hit8",
    )
    for row in rows:
        for metric in metrics:
            result.append({
                "symbol": row["symbol"],
                "side": row["side"],
                "lookback_days": row["lookback_days"],
                "horizon_candles": row["horizon_candles"],
                "metric": metric,
                "value": row.get(metric),
                "sample_count": row["sample_count"],
            })
    return result


def finite_key(row: dict[str, Any], key: str, default: float) -> float:
    value = row.get(key)
    return float(value) if value is not None and math.isfinite(float(value)) else default


def rank_sides(rows: list[dict[str, Any]], best: bool) -> list[dict[str, Any]]:
    focused = [
        row for row in rows
        if row["horizon_candles"] == 12 and row["lookback_days"] == 30 and row["symbol"] != "GLOBAL"
    ]
    return sorted(
        focused,
        key=lambda row: (finite_key(row, "avg_trade_quality", -999.0), finite_key(row, "hit8_before_minus5_rate", -999.0)),
        reverse=best,
    )[:5]


def build_report(symbols: list[str], lookbacks: list[int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_dataset: dict[tuple[str, int], dict[str, Any]] = {}
    errors: list[dict[str, str]] = []
    for symbol in symbols:
        try:
            market = load_signal_market(DEFAULT_TURBO_CONFIG.config_path, symbol_override=symbol)
            for lookback in lookbacks:
                built = build_recent_dataset(symbol, lookback, save=False, market=market)
                per_dataset[(symbol, lookback)] = built["dataset"]
        except Exception as exc:
            errors.append({"symbol": symbol, "error": repr(exc)})

    rows: list[dict[str, Any]] = []
    for (symbol, lookback), dataset in per_dataset.items():
        for side in ("long", "short"):
            for horizon in (12, 24):
                rows.append(summarize_arrays(symbol, side, lookback, horizon, [dataset]))
    for side in ("long", "short"):
        for horizon in (12, 24):
            combined = list(per_dataset.values())
            if combined:
                rows.append(summarize_arrays("GLOBAL", side, "ALL", horizon, combined))

    best = rank_sides(rows, best=True)
    worst = rank_sides(rows, best=False)
    global_12 = [row for row in rows if row["symbol"] == "GLOBAL" and row["horizon_candles"] == 12]
    report = {
        "schema_version": "aegis_turbo_operable_targets_audit_v2",
        "created_at": utc_now().isoformat(),
        "symbols": symbols,
        "lookback_days": lookbacks,
        "operable_targets_schema_version": OPERABLE_TARGET_SCHEMA_VERSION,
        "operable_target_names": list(OPERABLE_TARGET_NAMES),
        "trade_quality_formula": TRADE_QUALITY_FORMULA,
        "ambiguous_same_candle_policy": "stop_first_hit_before_stop_false",
        "dataset_errors": errors,
        "summary_rows": rows,
        "global_horizon_12": global_12,
        "best_symbol_sides_30d_h12": best,
        "worst_symbol_sides_30d_h12": worst,
        "interpretation": [
            "Correlation with V1 target is reported only for the comparable 12-candle horizon.",
            "Weak correlation between net return and hit-before-stop/trade-quality supports target misalignment.",
            "Rows labeled GLOBAL pool overlapping 7/14/30-day windows and are descriptive, not independent samples.",
        ],
        "future_v2_model_candidates": [
            {"model": "HistGradientBoostingClassifier", "target": "hit8_before_minus5_12", "purpose": "operable_win_probability"},
            {"model": "HistGradientBoostingRegressor", "target": "trade_quality_12", "purpose": "bounded_trade_quality"},
            {"model": "HistGradientBoostingClassifier", "target": "mae_danger_12", "purpose": "dangerous_mae_probability"},
        ],
        "promotion_recommendation": [
            "Research-only; do not promote models based on dataset distributions.",
            "Future validation must require positive net expectancy, improved hit8 baseline, controlled p90 MAE, calibrated score buckets, and walk-forward stability.",
        ],
    }
    return report, rows


def render_value(value: Any, percentage: bool = False) -> str:
    if value is None:
        return "null"
    if percentage:
        return f"{float(value) * 100:.2f}%"
    return f"{float(value):.6f}"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    def side_table(rows: list[dict[str, Any]]) -> list[str]:
        return [
            f"| {row['symbol']} {row['side']} | {render_value(row['hit8_before_minus5_rate'], True)} | "
            f"{render_value(row['mae_danger_rate'], True)} | {render_value(row['avg_trade_quality'])} | "
            f"{render_value(row['corr_v1_net_return_trade_quality'])} |"
            for row in rows
        ]

    lines = [
        f"# Aegis Turbo Operable Targets Audit {report['created_at']}",
        "",
        "## Executive Summary",
        "",
        f"- Operable schema: `{report['operable_targets_schema_version']}`",
        "- Dataset calculation is read-only: no model training and no snapshot persistence.",
        "- `hit-before-stop` uses future high/low; same-candle target and stop is conservatively counted as stop first.",
        "- Correlations quantify whether V1 close-to-close return resembles operable path outcomes.",
        "",
        "## Global 12-Candle Distribution",
        "",
        "| Side | Hit8 Before -5 | MAE Danger | Avg Quality | Corr V1 vs Quality | Corr V1 vs Hit8 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["global_horizon_12"]:
        lines.append(
            f"| {row['side']} | {render_value(row['hit8_before_minus5_rate'], True)} | "
            f"{render_value(row['mae_danger_rate'], True)} | {render_value(row['avg_trade_quality'])} | "
            f"{render_value(row['corr_v1_net_return_trade_quality'])} | "
            f"{render_value(row['corr_v1_net_return_hit8'])} |"
        )
    lines.extend([
        "",
        "## Best Symbol/Sides (30d, 12 Candles, By Trade Quality)",
        "",
        "| Symbol/Side | Hit8 Before -5 | MAE Danger | Avg Quality | Corr V1 vs Quality |",
        "|---|---:|---:|---:|---:|",
        *side_table(report["best_symbol_sides_30d_h12"]),
        "",
        "## Worst Symbol/Sides (30d, 12 Candles, By Trade Quality)",
        "",
        "| Symbol/Side | Hit8 Before -5 | MAE Danger | Avg Quality | Corr V1 vs Quality |",
        "|---|---:|---:|---:|---:|",
        *side_table(report["worst_symbol_sides_30d_h12"]),
        "",
        "## Trade Quality Formula",
        "",
        f"- `{report['trade_quality_formula']}`",
        "",
        "## Candidate Fase C Models",
        "",
    ])
    for candidate in report["future_v2_model_candidates"]:
        lines.append(f"- `{candidate['model']}` on `{candidate['target']}`: {candidate['purpose']}.")
    lines.extend(["", "## Promotion Constraints", ""])
    for item in report["promotion_recommendation"]:
        lines.append(f"- {item}")
    if report["dataset_errors"]:
        lines.extend(["", "## Dataset Errors", "", f"- `{report['dataset_errors']}`"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...] | list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit read-only distributions of path-aware Turbo operable targets.")
    parser.add_argument("--symbols", help="Comma-separated symbols; default is all Turbo symbols.")
    parser.add_argument("--lookback-days", type=int, action="append")
    parser.add_argument("--out-dir", default="/home/jasan/Develop")
    args = parser.parse_args()

    symbols = parse_symbols(args.symbols)
    lookbacks = list(dict.fromkeys(args.lookback_days or list(DEFAULT_TURBO_CONFIG.lookback_days)))
    report, rows = build_report(symbols, lookbacks)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    md_path = out_dir / f"aegis_turbo_operable_targets_audit_{stamp}.md"
    json_path = out_dir / f"aegis_turbo_operable_targets_audit_{stamp}.json"
    summary_path = out_dir / f"aegis_turbo_operable_targets_summary_{stamp}.csv"
    distributions_path = out_dir / f"aegis_turbo_operable_targets_distributions_{stamp}.csv"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(md_path, report)
    write_csv(summary_path, rows, SUMMARY_COLUMNS)
    distributions = distribution_rows(rows)
    write_csv(
        distributions_path,
        distributions,
        ("symbol", "side", "lookback_days", "horizon_candles", "metric", "value", "sample_count"),
    )
    print(json.dumps({
        "md": str(md_path),
        "json": str(json_path),
        "summary_csv": str(summary_path),
        "distributions_csv": str(distributions_path),
        "summary_rows": len(rows),
        "dataset_errors": report["dataset_errors"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
