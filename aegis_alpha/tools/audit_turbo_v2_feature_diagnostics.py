#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.signals.common import load_signal_market  # noqa: E402
from aegis_alpha.turbo.config import DEFAULT_TURBO_CONFIG  # noqa: E402
from aegis_alpha.turbo.feature_importance_operable_v2 import (  # noqa: E402
    feature_target_statistics,
    highly_correlated_features,
    permutation_importance_by_fold,
)
from aegis_alpha.turbo.operable_feature_builder_v2 import apply_feature_set  # noqa: E402
from aegis_alpha.turbo.recent_dataset import build_recent_dataset  # noqa: E402
from aegis_alpha.turbo.snapshot_utils import normalize_turbo_symbol  # noqa: E402


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_symbols(raw: str) -> list[str]:
    return list(dict.fromkeys(normalize_turbo_symbol(item) for item in raw.split(",") if item.strip()))


def parse_sides(raw: str) -> list[str]:
    return ["short", "long"] if raw.upper() == "BOTH" else [raw.lower()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = sorted({key for row in rows for key in row}) if rows else ["symbol", "side"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _num(value: Any) -> str:
    return "null" if value is None else f"{float(value):.4f}"


def _top(rows: list[dict[str, Any]], key: str, count: int = 10) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: abs(float(row.get(key) or 0.0)), reverse=True)[:count]


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        f"# Aegis Turbo V2 Feature Diagnostics {report['created_at']}",
        "",
        "## Scope",
        "",
        f"- Symbols: `{', '.join(report['symbols'])}`",
        f"- Sides: `{', '.join(report['sides'])}`",
        f"- Feature set: `{report['feature_set']}`",
        f"- Permutation importance: `{report['permutation']}`",
        "- Mode: `RESEARCH_ONLY`; no active model or inference path is changed.",
        "",
        "## Feature Quality",
        "",
        f"- Constant features: `{report['quality_summary']['constant_count']}`",
        f"- More than 90% zero: `{report['quality_summary']['mostly_zero_count']}`",
        f"- Highly correlated pairs (`|corr| >= 0.995`): `{report['quality_summary']['high_correlation_pair_count']}`",
        "",
        "## Strongest Target Correlations",
        "",
        "| Symbol/Side | Target | Feature | Correlation |",
        "|---|---|---|---:|",
    ]
    for row in report["top_correlations"]:
        lines.append(f"| {row['symbol']} {row['side']} | {row['target']} | {row['feature_name']} | {_num(row['correlation'])} |")
    lines.extend(["", "## Permutation Importance", ""])
    if report["permutation"]:
        lines.extend([
            "| Symbol/Side/Fold | Model | Feature | Importance |",
            "|---|---|---|---:|",
        ])
        for row in report["top_permutation_importance"]:
            lines.append(
                f"| {row['symbol']} {row['side']} F{row['fold']} | {row['model_family']} | "
                f"{row['feature_name']} | {_num(row['permutation_importance_mean'])} |"
            )
        lines.extend(["", "### Stability", ""])
        for row in report["importance_stability"]:
            lines.append(
                f"- `{row['symbol']} {row['side']} {row['model_family']}` top-10 adjacent overlap: "
                f"`{_num(row['adjacent_top10_jaccard_mean'])}`; stable features: `{row['stable_top_features']}`"
            )
    else:
        lines.append("- Not requested; execute with `--permutation` to evaluate fold-level importance.")
    lines.extend([
        "",
        "## Limitations",
        "",
        "- Base diagnostics analyse historical research datasets only.",
        "- `HistGradientBoosting` does not expose a supported built-in feature importance; permutation importance is used.",
        "- Research operable wick/body and volume features use documented proxies unless raw aligned open/volume are supplied.",
        "- BTC/ETH cross-symbol context remains pending; it is not injected in this phase.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    symbols = parse_symbols(args.symbols)
    sides = parse_sides(args.side)
    stamp = utc_stamp()
    quality_rows: list[dict[str, Any]] = []
    correlations: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    dataset_diagnostics: list[dict[str, Any]] = []
    for symbol in symbols:
        try:
            market = load_signal_market(DEFAULT_TURBO_CONFIG.config_path, symbol_override=symbol)
            base = build_recent_dataset(symbol, int(args.lookback_days), save=False, market=market)["dataset"]
            dataset = apply_feature_set(base, market, args.feature_set)
            dataset_diagnostics.append({
                "symbol": symbol,
                "feature_set": args.feature_set,
                "feature_count": int(np.asarray(dataset["X"]).shape[1]),
                **dataset.get("feature_diagnostics", {}),
            })
            for side in sides:
                stats = feature_target_statistics(dataset, side, int(args.horizon))
                for row in stats:
                    quality_rows.append({"symbol": symbol, "side": side.upper(), **row})
                    for key, target in (
                        ("corr_hit8", "hit8"),
                        ("corr_trade_quality", "trade_quality"),
                        ("corr_mae_danger", "mae_danger"),
                    ):
                        correlations.append({
                            "symbol": symbol,
                            "side": side.upper(),
                            "feature_name": row["feature_name"],
                            "feature_family": row["feature_family"],
                            "target": target,
                            "correlation": row.get(key),
                        })
                if args.permutation:
                    importance = permutation_importance_by_fold(
                        dataset,
                        symbol=symbol,
                        side=side,
                        lookback_days=int(args.lookback_days),
                        horizon=int(args.horizon),
                        fold_count=int(args.fold_count),
                        max_features=int(args.max_features),
                        sample_size=int(args.sample_size),
                        fast=bool(args.fast),
                    )
                    importance_rows.extend(importance["importance_rows"])
                    stability_rows.extend({"symbol": symbol, "side": side.upper(), **row} for row in importance["stability"])
                for pair in highly_correlated_features(dataset):
                    correlations.append({"symbol": symbol, "side": side.upper(), "target": "duplicate_pair", **pair})
        except Exception as exc:
            errors.append({"symbol": symbol, "error": repr(exc)})
    correlation_values = [row for row in correlations if row.get("correlation") is not None and row.get("target") != "duplicate_pair"]
    top_correlations: list[dict[str, Any]] = []
    for symbol in symbols:
        for side in [item.upper() for item in sides]:
            for target in ("hit8", "trade_quality", "mae_danger"):
                top_correlations.extend(_top([
                    row for row in correlation_values
                    if row["symbol"] == symbol and row["side"] == side and row["target"] == target
                ], "correlation", 3))
    constant = [row for row in quality_rows if float(row.get("constant_rate") or 0.0) > 0]
    mostly_zero = [row for row in quality_rows if float(row.get("zero_rate") or 0.0) >= 0.90]
    correlated_pairs = [row for row in correlations if row.get("target") == "duplicate_pair"]
    report: dict[str, Any] = {
        "schema_version": "aegis_turbo_v2_feature_diagnostics_v1",
        "created_at": utc_iso(),
        "mode": "RESEARCH_ONLY",
        "symbols": symbols,
        "sides": [side.upper() for side in sides],
        "lookback_days": int(args.lookback_days),
        "horizon_candles": int(args.horizon),
        "feature_set": args.feature_set,
        "permutation": bool(args.permutation),
        "dataset_diagnostics": dataset_diagnostics,
        "quality_summary": {
            "feature_rows": len(quality_rows),
            "constant_count": len(constant),
            "mostly_zero_count": len(mostly_zero),
            "high_correlation_pair_count": len(correlated_pairs),
        },
        "top_correlations": top_correlations,
        "top_permutation_importance": sorted(
            importance_rows,
            key=lambda row: float(row.get("permutation_importance_mean") or 0.0),
            reverse=True,
        )[:40],
        "importance_stability": stability_rows,
        "errors": errors,
        "limitations": [
            "HistGradientBoosting_has_no_supported_builtin_feature_importances",
            "operable_v2_uses_documented_open_and_volume_proxies_with_current_SignalMarket",
            "btc_eth_cross_symbol_context_not_added",
        ],
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "md": out_dir / f"aegis_turbo_v2_feature_diagnostics_{stamp}.md",
        "json": out_dir / f"aegis_turbo_v2_feature_diagnostics_{stamp}.json",
        "rankings_csv": out_dir / f"aegis_turbo_v2_feature_rankings_{stamp}.csv",
        "quality_csv": out_dir / f"aegis_turbo_v2_feature_quality_{stamp}.csv",
        "correlations_csv": out_dir / f"aegis_turbo_v2_feature_correlations_{stamp}.csv",
    }
    report["paths"] = {key: str(value) for key, value in paths.items()}
    paths["json"].write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(paths["md"], report)
    write_csv(paths["rankings_csv"], importance_rows or top_correlations)
    write_csv(paths["quality_csv"], quality_rows)
    write_csv(paths["correlations_csv"], correlations)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit feature quality and fold-level importance for Turbo V2 research models.")
    parser.add_argument("--symbols", default="ADAUSDT,AVAXUSDT")
    parser.add_argument("--side", choices=("LONG", "SHORT", "BOTH"), default="SHORT")
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--feature-set", choices=("base", "operable_v2", "combined"), default="base")
    parser.add_argument("--out-dir", default="/home/jasan/Develop")
    parser.add_argument("--permutation", action="store_true")
    parser.add_argument("--max-features", type=int, default=50)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--fold-count", type=int, default=4)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"paths": report["paths"], "quality_summary": report["quality_summary"], "errors": report["errors"]}, indent=2))


if __name__ == "__main__":
    main()
