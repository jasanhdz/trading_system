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

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.config import REPO_ROOT  # noqa: E402
from aegis_alpha.signals.common import load_signal_market  # noqa: E402
from aegis_alpha.turbo.config import DEFAULT_TURBO_CONFIG  # noqa: E402
from aegis_alpha.turbo.operable_feature_builder_v2 import apply_feature_set  # noqa: E402
from aegis_alpha.turbo.recent_dataset import build_recent_dataset  # noqa: E402
from aegis_alpha.turbo.walk_forward_operable_v2 import run_walk_forward  # noqa: E402


SYMBOL = "ADAUSDT"
SIDE = "SHORT"
PRIMARY_FEATURE_SET = "operable_v2"
FEATURE_SETS = ("operable_v2", "base", "combined")
DEFAULT_MODEL_DIR = REPO_ROOT / "aegis_alpha" / "models" / "research" / "turbo_v2_ada_matrix"
SUMMARY_COLUMNS = (
    "rank",
    "symbol",
    "side",
    "feature_set",
    "lookback_days",
    "horizon_candles",
    "fold_count",
    "valid_fold_count",
    "recommendation",
    "ada_research_status",
    "research_score",
    "stability_score",
    "decay_score",
    "baseline_hit8_mean",
    "baseline_quality_mean",
    "baseline_danger_mean",
    "baseline_p90_mae_mean",
    "v2_hit8_auc_mean",
    "v2_hit8_auc_min",
    "hit8_top_decile_lift_mean",
    "hit8_top_decile_lift_min",
    "latest_fold_hit8_lift",
    "v2_quality_corr_mean",
    "v2_quality_corr_min",
    "quality_top_decile_lift_mean",
    "quality_top_decile_lift_min",
    "latest_fold_quality_lift",
    "latest_fold_quality_p90_mae",
    "latest_fold_baseline_p90_mae",
    "v2_danger_auc_mean",
    "danger_filter_usefulness_mean",
    "latest_fold_baseline_danger_rate",
    "estimated_fee_bps",
    "estimated_slippage_bps",
    "cost_proxy_return",
    "top_decile_quality_mean",
    "net_quality_lift_after_cost_proxy",
    "top_decile_net_quality_after_cost_proxy",
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_stamp() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%SZ")


def parse_csv_strings(value: str) -> list[str]:
    return list(dict.fromkeys(item.strip().lower() for item in value.split(",") if item.strip()))


def parse_csv_ints(value: str) -> list[int]:
    return list(dict.fromkeys(int(item.strip()) for item in value.split(",") if item.strip()))


def finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def mean(values: list[Any]) -> float | None:
    valid = [finite(value, float("nan")) for value in values]
    valid = [value for value in valid if math.isfinite(value)]
    return float(np.mean(valid)) if valid else None


def validate_research_model_dir(model_dir: Path) -> None:
    if "active" in {part.lower() for part in model_dir.parts}:
        raise ValueError(f"matrix research directory must not contain active: {model_dir}")


def _family(fold: dict[str, Any], family: str, *keys: str) -> Any:
    current: Any = (fold.get("families") or {}).get(family, {})
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def enrich_summary(
    summary: dict[str, Any],
    folds: list[dict[str, Any]],
    *,
    fee_bps: float,
    slippage_bps: float,
) -> dict[str, Any]:
    valid_folds = [fold for fold in folds if fold.get("model_status") == "trained"]
    cost_proxy = (float(fee_bps) + float(slippage_bps)) / 10000.0
    row = dict(summary)
    row.update({
        "estimated_fee_bps": float(fee_bps),
        "estimated_slippage_bps": float(slippage_bps),
        "cost_proxy_return": cost_proxy,
        "baseline_p90_mae_mean": mean([fold.get("baseline_test", {}).get("p90_mae") for fold in valid_folds]),
        "top_decile_quality_mean": mean([
            _family(fold, "trade_quality_regressor", "top_decile", "avg_trade_quality") for fold in valid_folds
        ]),
    })
    row["net_quality_lift_after_cost_proxy"] = (
        finite(row.get("quality_top_decile_lift_mean")) - cost_proxy
        if row.get("quality_top_decile_lift_mean") is not None else None
    )
    row["top_decile_net_quality_after_cost_proxy"] = (
        finite(row.get("top_decile_quality_mean")) - cost_proxy
        if row.get("top_decile_quality_mean") is not None else None
    )
    row["latest_p90_mae_delta"] = (
        finite(row.get("latest_fold_quality_p90_mae")) - finite(row.get("latest_fold_baseline_p90_mae"))
        if row.get("latest_fold_quality_p90_mae") is not None and row.get("latest_fold_baseline_p90_mae") is not None
        else None
    )
    return row


def classify_ada_short_config(row: dict[str, Any]) -> str:
    p90_ok = (
        row.get("latest_fold_quality_p90_mae") is not None
        and row.get("latest_fold_baseline_p90_mae") is not None
        and finite(row["latest_fold_quality_p90_mae"]) <= finite(row["latest_fold_baseline_p90_mae"]) * 1.15
    )
    strong = (
        row.get("recommendation") == "WALK_FORWARD_PROMISING"
        and int(row.get("valid_fold_count") or 0) >= 4
        and finite(row.get("quality_top_decile_lift_mean")) > 0.0
        and finite(row.get("hit8_top_decile_lift_mean")) > 0.0
        and finite(row.get("latest_fold_quality_lift")) > 0.0
        and finite(row.get("v2_hit8_auc_mean")) > 0.55
        and finite(row.get("v2_quality_corr_mean")) >= 0.0
        and p90_ok
    )
    if strong:
        return "ADA_SHORT_STRONG_RESEARCH"
    bad = (
        row.get("recommendation") == "WALK_FORWARD_BAD"
        or int(row.get("valid_fold_count") or 0) < 3
        or (
            finite(row.get("quality_top_decile_lift_mean")) <= 0.0
            and finite(row.get("hit8_top_decile_lift_mean")) <= 0.0
        )
        or (
            finite(row.get("latest_fold_quality_lift")) < 0.0
            and finite(row.get("quality_top_decile_lift_mean")) <= 0.0
            and finite(row.get("hit8_top_decile_lift_mean")) <= 0.0
        )
    )
    return "ADA_SHORT_BAD_RESEARCH" if bad else "ADA_SHORT_MIXED_RESEARCH"


def research_score(row: dict[str, Any]) -> float:
    score = (
        finite(row.get("stability_score"))
        + finite(row.get("quality_top_decile_lift_mean")) * 3.0
        + finite(row.get("hit8_top_decile_lift_mean")) * 2.0
        + finite(row.get("v2_hit8_auc_mean"))
        + finite(row.get("v2_danger_auc_mean")) * 0.5
    )
    if finite(row.get("quality_top_decile_lift_min")) < 0.0:
        score -= 0.25
    if finite(row.get("hit8_top_decile_lift_min")) < 0.0:
        score -= 0.25
    if finite(row.get("latest_fold_quality_lift")) < 0.0:
        score -= 0.30
    if (
        row.get("latest_fold_quality_p90_mae") is not None
        and row.get("latest_fold_baseline_p90_mae") is not None
        and finite(row["latest_fold_quality_p90_mae"]) > finite(row["latest_fold_baseline_p90_mae"]) * 1.15
    ):
        score -= 0.30
    return float(score)


def rank_ada_short_configs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        candidate = dict(row)
        candidate["ada_research_status"] = classify_ada_short_config(candidate)
        candidate["research_score"] = research_score(candidate)
        enriched.append(candidate)
    ranked = sorted(enriched, key=lambda row: float(row["research_score"]), reverse=True)
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    return ranked


def select_reference_configuration(ranked: list[dict[str, Any]]) -> dict[str, Any]:
    for row in ranked:
        if row.get("ada_research_status") == "ADA_SHORT_STRONG_RESEARCH":
            return row
    if not ranked:
        raise ValueError("no evaluated configuration available for references")
    return ranked[0]


def fold_rows(result_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in result_rows:
        summary = result["summary"]
        for fold in result["folds"]:
            rows.append({
                "symbol": SYMBOL,
                "side": SIDE,
                "feature_set": summary.get("feature_set"),
                "lookback_days": summary.get("lookback_days"),
                "horizon_candles": summary.get("horizon_candles"),
                "fold": fold.get("fold"),
                "model_status": fold.get("model_status"),
                "baseline_hit8": fold.get("baseline_test", {}).get("hit8_rate"),
                "baseline_quality": fold.get("baseline_test", {}).get("avg_trade_quality"),
                "baseline_danger": fold.get("baseline_test", {}).get("mae_danger_rate"),
                "baseline_p90_mae": fold.get("baseline_test", {}).get("p90_mae"),
                "hit8_auc": _family(fold, "hit8_classifier", "test_metrics", "roc_auc"),
                "hit8_lift": _family(fold, "hit8_classifier", "top_decile", "hit8_lift_vs_baseline"),
                "quality_corr": _family(fold, "trade_quality_regressor", "test_metrics", "spearman"),
                "quality_lift": _family(fold, "trade_quality_regressor", "top_decile", "quality_lift_vs_baseline"),
                "quality_p90_mae": _family(fold, "trade_quality_regressor", "top_decile", "p90_mae"),
                "danger_auc": _family(fold, "mae_danger_classifier", "test_metrics", "roc_auc"),
                "danger_filter_usefulness": _family(fold, "mae_danger_classifier", "usefulness_as_filter"),
            })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...] | None = None) -> None:
    fields = list(columns or sorted({key for row in rows for key in row}) or ["symbol", "side"])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _num(value: Any) -> str:
    return "null" if value is None else f"{float(value):.4f}"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    rows = report["ranking"]
    best = rows[0] if rows else {}
    worst = rows[-1] if rows else {}
    lines = [
        f"# Aegis ADAUSDT SHORT V2 Matrix {report['created_at']}",
        "",
        "## Safety",
        "",
        "- Mode: `RESEARCH_ONLY`",
        "- Symbol/side: `ADAUSDT SHORT` only.",
        "- No models are saved; no `active/` models or `active_manifest.json` are modified.",
        "- No live inference, YAML or PM2 action is involved.",
        f"- Cost proxy: `{report['estimated_fee_bps']} bps fee + {report['estimated_slippage_bps']} bps slippage`.",
        "",
        "## Ranking",
        "",
        "| Rank | Feature Set | Window | Horizon | Status | Score | Hit8 AUC | Hit8 Lift | Quality Lift | Quality Corr | Danger AUC | Latest Quality Lift | P90 Delta |",
        "|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['rank']} | {row['feature_set']} | {row['lookback_days']}d | {row['horizon_candles']} | "
            f"{row['ada_research_status']} | {_num(row['research_score'])} | {_num(row.get('v2_hit8_auc_mean'))} | "
            f"{_num(row.get('hit8_top_decile_lift_mean'))} | {_num(row.get('quality_top_decile_lift_mean'))} | "
            f"{_num(row.get('v2_quality_corr_mean'))} | {_num(row.get('v2_danger_auc_mean'))} | "
            f"{_num(row.get('latest_fold_quality_lift'))} | {_num(row.get('latest_p90_mae_delta'))} |"
        )
    lines.extend([
        "",
        "## Best Configuration",
        "",
        (
            f"- `{best.get('feature_set')} {best.get('lookback_days')}d h{best.get('horizon_candles')}`: "
            f"`{best.get('ada_research_status')}` with research score `{_num(best.get('research_score'))}`."
            if best else "- No valid configuration."
        ),
        "",
        "## Worst Configuration",
        "",
        (
            f"- `{worst.get('feature_set')} {worst.get('lookback_days')}d h{worst.get('horizon_candles')}`: "
            f"`{worst.get('ada_research_status')}` with research score `{_num(worst.get('research_score'))}`."
            if worst else "- No valid configuration."
        ),
        "",
        "## Decision",
        "",
        f"- Strong research configurations found: `{report['strong_config_count']}`.",
        f"- Recommendation: `{report['next_recommendation']}`.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_configurations(feature_sets: list[str], lookbacks: list[int], horizons: list[int], reference_best_only: bool) -> list[tuple[str, int, int]]:
    if not reference_best_only:
        return [(feature_set, lookback, horizon) for feature_set in feature_sets for lookback in lookbacks for horizon in horizons]
    return [(PRIMARY_FEATURE_SET, lookback, horizon) for lookback in lookbacks for horizon in horizons]


def run(args: argparse.Namespace) -> dict[str, Any]:
    feature_sets = parse_csv_strings(args.feature_sets)
    invalid = [value for value in feature_sets if value not in FEATURE_SETS]
    if invalid:
        raise ValueError(f"unsupported feature sets: {invalid}")
    lookbacks = parse_csv_ints(args.lookback_days)
    horizons = parse_csv_ints(args.horizons)
    model_dir = Path(args.model_dir)
    validate_research_model_dir(model_dir)
    market = load_signal_market(DEFAULT_TURBO_CONFIG.config_path, symbol_override=SYMBOL)
    dataset_cache: dict[tuple[int, str], dict[str, Any]] = {}
    result_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    def evaluate(feature_set: str, lookback: int, horizon: int) -> None:
        try:
            key = (lookback, feature_set)
            if key not in dataset_cache:
                base = build_recent_dataset(SYMBOL, lookback, save=False, market=market)["dataset"]
                dataset_cache[key] = apply_feature_set(base, market, feature_set)
            result = run_walk_forward(
                dataset_cache[key],
                symbol=SYMBOL,
                side=SIDE.lower(),
                lookback_days=lookback,
                horizon=horizon,
                fold_count=int(args.fold_count),
                train_ratio=float(args.train_ratio),
                validation_ratio=float(args.validation_ratio),
                test_ratio=float(args.test_ratio),
                expanding_window=True,
                min_train_samples=int(args.min_train_samples),
                min_test_samples=int(args.min_test_samples),
                run_dir=model_dir / SYMBOL / f"{feature_set}_{lookback}d_h{horizon}",
                save_models=False,
                fast=bool(args.fast),
            )
            summary = enrich_summary(
                result["summary"],
                result["folds"],
                fee_bps=float(args.fee_bps),
                slippage_bps=float(args.slippage_bps),
            )
            result_rows.append({"summary": summary, "folds": result["folds"]})
        except Exception as exc:
            errors.append({"feature_set": feature_set, "lookback_days": str(lookback), "horizon": str(horizon), "error": repr(exc)})

    for configuration in build_configurations(feature_sets, lookbacks, horizons, bool(args.reference_best_only)):
        evaluate(*configuration)
    if args.reference_best_only and PRIMARY_FEATURE_SET in feature_sets and result_rows:
        primary_ranking = rank_ada_short_configs([result["summary"] for result in result_rows])
        best = select_reference_configuration(primary_ranking)
        for feature_set in feature_sets:
            if feature_set != PRIMARY_FEATURE_SET:
                evaluate(feature_set, int(best["lookback_days"]), int(best["horizon_candles"]))

    ranking = rank_ada_short_configs([result["summary"] for result in result_rows])
    strong_count = sum(row["ada_research_status"] == "ADA_SHORT_STRONG_RESEARCH" for row in ranking)
    next_recommendation = (
        "consider_shadow_metadata_ada_short_only_default_off"
        if strong_count > 0
        else "do_not_add_shadow_runtime_redesign_features_or_model"
    )
    token = utc_stamp()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "md": out_dir / f"aegis_ada_short_v2_matrix_{token}.md",
        "json": out_dir / f"aegis_ada_short_v2_matrix_{token}.json",
        "summary_csv": out_dir / f"aegis_ada_short_v2_matrix_summary_{token}.csv",
        "folds_csv": out_dir / f"aegis_ada_short_v2_matrix_folds_{token}.csv",
        "ranking_csv": out_dir / f"aegis_ada_short_v2_matrix_ranking_{token}.csv",
    }
    report: dict[str, Any] = {
        "schema_version": "aegis_ada_short_operable_v2_matrix_v1",
        "created_at": utc_now().isoformat(),
        "mode": "RESEARCH_ONLY",
        "symbol": SYMBOL,
        "side": SIDE,
        "feature_sets_requested": feature_sets,
        "lookback_days_requested": lookbacks,
        "horizons_requested": horizons,
        "fold_count": int(args.fold_count),
        "reference_best_only": bool(args.reference_best_only),
        "fast": bool(args.fast),
        "estimated_fee_bps": float(args.fee_bps),
        "estimated_slippage_bps": float(args.slippage_bps),
        "save_models": False,
        "active_manifest_touched": False,
        "live_inference_changed": False,
        "ranking": ranking,
        "results": result_rows,
        "errors": errors,
        "strong_config_count": strong_count,
        "next_recommendation": next_recommendation,
        "paths": {key: str(value) for key, value in paths.items()},
    }
    paths["json"].write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(paths["md"], report)
    write_csv(paths["summary_csv"], ranking, SUMMARY_COLUMNS)
    write_csv(paths["ranking_csv"], ranking, SUMMARY_COLUMNS)
    write_csv(paths["folds_csv"], fold_rows(result_rows))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Research-only ADAUSDT SHORT operable V2 matrix evaluator.")
    parser.add_argument("--feature-sets", default="operable_v2,base,combined")
    parser.add_argument("--lookback-days", default="7,14,30")
    parser.add_argument("--horizons", default="12,24")
    parser.add_argument("--fold-count", type=int, default=4)
    parser.add_argument("--train-ratio", type=float, default=0.50)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--min-train-samples", type=int, default=1000)
    parser.add_argument("--min-test-samples", type=int, default=300)
    parser.add_argument("--fee-bps", type=float, default=8.0)
    parser.add_argument("--slippage-bps", type=float, default=3.0)
    parser.add_argument("--out-dir", default="/home/jasan/Develop")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--reference-best-only", action="store_true")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({
        "paths": report["paths"],
        "evaluated_configurations": len(report["ranking"]),
        "strong_config_count": report["strong_config_count"],
        "next_recommendation": report["next_recommendation"],
        "errors": report["errors"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
