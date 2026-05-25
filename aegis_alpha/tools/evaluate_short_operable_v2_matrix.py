#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.config import REPO_ROOT  # noqa: E402
from aegis_alpha.signals.common import load_signal_market  # noqa: E402
from aegis_alpha.tools.evaluate_ada_short_operable_v2_matrix import (  # noqa: E402
    _family,
    enrich_summary,
    finite,
    parse_csv_ints,
    parse_csv_strings,
    utc_now,
    utc_stamp,
    validate_research_model_dir,
)
from aegis_alpha.turbo.config import DEFAULT_TURBO_CONFIG  # noqa: E402
from aegis_alpha.turbo.operable_feature_builder_v3 import FEATURE_SETS, apply_feature_set  # noqa: E402
from aegis_alpha.turbo.recent_dataset import build_recent_dataset  # noqa: E402
from aegis_alpha.turbo.snapshot_utils import normalize_turbo_symbol  # noqa: E402
from aegis_alpha.turbo.walk_forward_operable_v2 import run_walk_forward  # noqa: E402


SIDE = "SHORT"
PRIMARY_FEATURE_SET = "operable_v2"
DEFAULT_SYMBOLS = (
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
DEFAULT_MODEL_DIR = REPO_ROOT / "aegis_alpha" / "models" / "research" / "turbo_v2_short_global_matrix"
STATUS_PRIORITY = {
    "SHORT_STRONG_RESEARCH": 0,
    "SHORT_MIXED_RESEARCH": 1,
    "SHORT_BAD_RESEARCH": 2,
    "INSUFFICIENT_DATA": 3,
}
SUMMARY_COLUMNS = (
    "rank",
    "symbol",
    "side",
    "feature_set",
    "feature_count",
    "base_feature_count",
    "new_feature_count",
    "operable_v2_feature_count",
    "operable_v3_feature_count",
    "feature_schema_hash",
    "lookback_days",
    "horizon_candles",
    "fold_count",
    "valid_fold_count",
    "recommendation",
    "short_research_status",
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
    "latest_p90_mae_delta",
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


def parse_symbols(raw: str | None) -> list[str]:
    requested = [part.strip() for part in (raw or ",".join(DEFAULT_SYMBOLS)).split(",") if part.strip()]
    return list(dict.fromkeys(normalize_turbo_symbol(symbol) for symbol in requested))


def classify_short_config(row: dict[str, Any]) -> str:
    valid_folds = int(row.get("valid_fold_count") or 0)
    if row.get("config_error") or valid_folds < 3:
        return "INSUFFICIENT_DATA"
    latest_p90 = row.get("latest_fold_quality_p90_mae")
    baseline_p90 = row.get("latest_fold_baseline_p90_mae")
    p90_ratio = (
        finite(latest_p90) / max(finite(baseline_p90), 1e-12)
        if latest_p90 is not None and baseline_p90 is not None
        else float("inf")
    )
    strong = (
        row.get("recommendation") == "WALK_FORWARD_PROMISING"
        and valid_folds >= 4
        and row.get("feature_set") in {PRIMARY_FEATURE_SET, "operable_v3", "combined_v3"}
        and finite(row.get("quality_top_decile_lift_mean")) > 0.0
        and finite(row.get("hit8_top_decile_lift_mean")) > 0.0
        and finite(row.get("latest_fold_quality_lift")) > 0.0
        and finite(row.get("v2_hit8_auc_mean")) > 0.55
        and finite(row.get("v2_quality_corr_mean")) >= 0.0
        and p90_ratio <= 1.15
        and finite(row.get("net_quality_lift_after_cost_proxy")) > 0.0
    )
    if strong:
        return "SHORT_STRONG_RESEARCH"
    bad = (
        row.get("recommendation") == "WALK_FORWARD_BAD"
        or (
            finite(row.get("quality_top_decile_lift_mean")) < 0.0
            and finite(row.get("hit8_top_decile_lift_mean")) < 0.0
        )
        or (
            finite(row.get("latest_fold_quality_lift")) < 0.0
            and finite(row.get("quality_top_decile_lift_mean")) <= 0.0
            and finite(row.get("hit8_top_decile_lift_mean")) <= 0.0
        )
        or p90_ratio > 1.30
    )
    return "SHORT_BAD_RESEARCH" if bad else "SHORT_MIXED_RESEARCH"


def research_score(row: dict[str, Any]) -> float:
    score = (
        finite(row.get("stability_score"))
        + finite(row.get("quality_top_decile_lift_mean")) * 3.0
        + finite(row.get("hit8_top_decile_lift_mean")) * 2.0
        + finite(row.get("v2_hit8_auc_mean"))
        + finite(row.get("v2_danger_auc_mean")) * 0.5
        + finite(row.get("net_quality_lift_after_cost_proxy")) * 2.0
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


def rank_short_configs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        candidate = dict(row)
        candidate["short_research_status"] = classify_short_config(candidate)
        candidate["research_score"] = research_score(candidate)
        enriched.append(candidate)
    ranked = sorted(
        enriched,
        key=lambda row: (
            STATUS_PRIORITY[row["short_research_status"]],
            -finite(row.get("research_score")),
            -finite(row.get("latest_fold_quality_lift")),
            finite(row.get("latest_p90_mae_delta"), float("inf")),
        ),
    )
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    return ranked


def best_configuration_by_symbol(ranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for row in ranked:
        symbol = str(row.get("symbol"))
        if symbol not in seen:
            seen.add(symbol)
            rows.append(row)
    return rows


def fold_rows(result_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in result_rows:
        summary = result["summary"]
        for fold in result["folds"]:
            rows.append({
                "symbol": summary.get("symbol"),
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
    rows = report["visible_ranking"]
    sections = {
        status: [row for row in report["ranking"] if row["short_research_status"] == status]
        for status in STATUS_PRIORITY
    }
    lines = [
        f"# Aegis Turbo V2 Global SHORT Matrix {report['created_at']}",
        "",
        "## Safety",
        "",
        "- Mode: `RESEARCH_ONLY`",
        "- Side: `SHORT` only.",
        "- No models are saved; no `active/` models or `active_manifest.json` are modified.",
        "- No live inference, YAML or PM2 action is involved.",
        f"- Cost proxy: `{report['estimated_fee_bps']} bps fee + {report['estimated_slippage_bps']} bps slippage`.",
        "",
        "## Ranking",
        "",
        "| Rank | Symbol | Set | Window | H | Status | Score | Hit8 AUC | Hit8 Lift | Quality Lift | Quality Corr | Danger AUC | Latest Quality | P90 Delta |",
        "|---:|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['rank']} | {row['symbol']} | {row['feature_set']} | {row['lookback_days']}d | "
            f"{row['horizon_candles']} | {row['short_research_status']} | {_num(row.get('research_score'))} | "
            f"{_num(row.get('v2_hit8_auc_mean'))} | {_num(row.get('hit8_top_decile_lift_mean'))} | "
            f"{_num(row.get('quality_top_decile_lift_mean'))} | {_num(row.get('v2_quality_corr_mean'))} | "
            f"{_num(row.get('v2_danger_auc_mean'))} | {_num(row.get('latest_fold_quality_lift'))} | "
            f"{_num(row.get('latest_p90_mae_delta'))} |"
        )
    for status, title in (
        ("SHORT_STRONG_RESEARCH", "Strong Candidates"),
        ("SHORT_MIXED_RESEARCH", "Mixed Candidates"),
        ("SHORT_BAD_RESEARCH", "Bad Candidates"),
        ("INSUFFICIENT_DATA", "Insufficient Data"),
    ):
        lines.extend(["", f"## {title}", ""])
        candidates = sections[status]
        if not candidates:
            lines.append("- None.")
        else:
            for row in candidates:
                lines.append(
                    f"- `{row['symbol']} {row['feature_set']} {row['lookback_days']}d h{row['horizon_candles']}`: "
                    f"score `{_num(row['research_score'])}`, quality lift `{_num(row.get('quality_top_decile_lift_mean'))}`, "
                    f"hit8 lift `{_num(row.get('hit8_top_decile_lift_mean'))}`."
                )
    lines.extend(["", "## Best Configuration By Symbol", ""])
    for row in report["best_by_symbol"]:
        lines.append(
            f"- `{row['symbol']}`: `{row['feature_set']} {row['lookback_days']}d h{row['horizon_candles']}` "
            f"`{row['short_research_status']}`."
        )
    lines.extend([
        "",
        "## Decision",
        "",
        f"- Strong candidates found: `{report['strong_candidate_count']}`.",
        f"- Recommendation: `{report['next_recommendation']}`.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_configurations(
    symbols: list[str],
    feature_sets: list[str],
    lookbacks: list[int],
    horizons: list[int],
    max_configs: int | None,
) -> list[tuple[str, str, int, int]]:
    configurations = list(dict.fromkeys(
        (symbol, feature_set, lookback, horizon)
        for symbol in symbols
        for feature_set in feature_sets
        for lookback in lookbacks
        for horizon in horizons
    ))
    return configurations[:max_configs] if max_configs else configurations


def run(args: argparse.Namespace) -> dict[str, Any]:
    symbols = parse_symbols(args.symbols)
    feature_sets = parse_csv_strings(args.feature_sets)
    invalid = [value for value in feature_sets if value not in FEATURE_SETS]
    if invalid:
        raise ValueError(f"unsupported feature sets: {invalid}")
    lookbacks = parse_csv_ints(args.lookback_days)
    horizons = parse_csv_ints(args.horizons)
    model_dir = Path(args.model_dir)
    validate_research_model_dir(model_dir)
    result_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    market_cache: dict[str, Any] = {}
    dataset_cache: dict[tuple[str, int, str], dict[str, Any]] = {}
    context_markets: dict[str, Any] = {}
    evaluated: set[tuple[str, str, int, int]] = set()

    def evaluate(configuration: tuple[str, str, int, int]) -> None:
        symbol, feature_set, lookback, horizon = configuration
        if configuration in evaluated:
            return
        evaluated.add(configuration)
        try:
            if symbol not in market_cache:
                market_cache[symbol] = load_signal_market(DEFAULT_TURBO_CONFIG.config_path, symbol_override=symbol)
            key = (symbol, lookback, feature_set)
            if key not in dataset_cache:
                base = build_recent_dataset(symbol, lookback, save=False, market=market_cache[symbol])["dataset"]
                if feature_set in {"operable_v3", "combined_v3"} and not context_markets:
                    context_markets.update({
                        value: load_signal_market(DEFAULT_TURBO_CONFIG.config_path, symbol_override=value)
                        for value in ("BTCUSDT", "ETHUSDT")
                    })
                dataset_cache[key] = apply_feature_set(
                    base,
                    market_cache[symbol],
                    feature_set,
                    context_markets=context_markets,
                )
            result = run_walk_forward(
                dataset_cache[key],
                symbol=symbol,
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
                run_dir=model_dir / symbol / f"{feature_set}_{lookback}d_h{horizon}",
                save_models=False,
                fast=bool(args.fast),
            )
            result_rows.append({
                "summary": enrich_summary(
                    result["summary"],
                    result["folds"],
                    fee_bps=float(args.fee_bps),
                    slippage_bps=float(args.slippage_bps),
                ),
                "folds": result["folds"],
            })
        except Exception as exc:
            errors.append({
                "symbol": symbol,
                "feature_set": feature_set,
                "lookback_days": lookback,
                "horizon_candles": horizon,
                "error": repr(exc),
            })

    principal_configs = build_configurations(symbols, feature_sets, lookbacks, horizons, args.max_configs)
    for configuration in principal_configs:
        evaluate(configuration)

    if args.include_reference_feature_sets:
        current_ranking = rank_short_configs([result["summary"] for result in result_rows])
        reference_symbols = {
            row["symbol"]
            for row in current_ranking
            if row["short_research_status"] in {"SHORT_STRONG_RESEARCH", "SHORT_MIXED_RESEARCH"}
        }
        for symbol in reference_symbols:
            for feature_set in ("base", "combined"):
                for lookback in lookbacks:
                    for horizon in horizons:
                        evaluate((symbol, feature_set, lookback, horizon))

    ranking = rank_short_configs([result["summary"] for result in result_rows])
    strong_rows = [row for row in ranking if row["short_research_status"] == "SHORT_STRONG_RESEARCH"]
    status_counts = dict(Counter(row["short_research_status"] for row in ranking))
    next_recommendation = (
        "generate_shadow_artifacts_for_strong_candidates_metadata_only_default_off"
        if strong_rows else "do_not_generate_shadow_artifacts_continue_feature_model_research"
    )
    token = utc_stamp()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "md": out_dir / f"aegis_short_v2_global_matrix_{token}.md",
        "json": out_dir / f"aegis_short_v2_global_matrix_{token}.json",
        "summary_csv": out_dir / f"aegis_short_v2_global_matrix_summary_{token}.csv",
        "folds_csv": out_dir / f"aegis_short_v2_global_matrix_folds_{token}.csv",
        "ranking_csv": out_dir / f"aegis_short_v2_global_matrix_ranking_{token}.csv",
        "strong_candidates_csv": out_dir / f"aegis_short_v2_global_matrix_strong_candidates_{token}.csv",
    }
    visible_ranking = strong_rows if args.strong_only_report else ranking
    report: dict[str, Any] = {
        "schema_version": "aegis_short_operable_v2_global_matrix_v1",
        "created_at": utc_now().isoformat(),
        "mode": "RESEARCH_ONLY",
        "side": SIDE,
        "symbols_requested": symbols,
        "feature_sets_requested": feature_sets,
        "lookback_days_requested": lookbacks,
        "horizons_requested": horizons,
        "fold_count": int(args.fold_count),
        "fast": bool(args.fast),
        "estimated_fee_bps": float(args.fee_bps),
        "estimated_slippage_bps": float(args.slippage_bps),
        "include_reference_feature_sets": bool(args.include_reference_feature_sets),
        "strong_only_report": bool(args.strong_only_report),
        "skip_existing": bool(args.skip_existing),
        "save_models": False,
        "active_manifest_touched": False,
        "live_inference_changed": False,
        "ranking": ranking,
        "visible_ranking": visible_ranking,
        "best_by_symbol": best_configuration_by_symbol(ranking),
        "results": result_rows,
        "errors": errors,
        "status_counts": status_counts,
        "strong_candidate_count": len(strong_rows),
        "next_recommendation": next_recommendation,
        "paths": {key: str(value) for key, value in paths.items()},
    }
    paths["json"].write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(paths["md"], report)
    write_csv(paths["summary_csv"], ranking, SUMMARY_COLUMNS)
    write_csv(paths["ranking_csv"], visible_ranking, SUMMARY_COLUMNS)
    write_csv(paths["folds_csv"], fold_rows(result_rows))
    write_csv(paths["strong_candidates_csv"], strong_rows, SUMMARY_COLUMNS)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Research-only global SHORT operable V2 matrix evaluator.")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--feature-sets", default=PRIMARY_FEATURE_SET)
    parser.add_argument("--lookback-days", default="30")
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
    parser.add_argument("--include-reference-feature-sets", action="store_true")
    parser.add_argument("--strong-only-report", action="store_true")
    parser.add_argument("--max-configs", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({
        "paths": report["paths"],
        "evaluated_configurations": len(report["ranking"]),
        "status_counts": report["status_counts"],
        "strong_candidate_count": report["strong_candidate_count"],
        "next_recommendation": report["next_recommendation"],
        "errors": report["errors"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
