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
    FEATURE_SETS,
    enrich_summary,
    finite,
    parse_csv_ints,
    parse_csv_strings,
    utc_now,
    utc_stamp,
    validate_research_model_dir,
)
from aegis_alpha.tools.evaluate_short_operable_v2_matrix import (  # noqa: E402
    DEFAULT_SYMBOLS,
    SIDE,
    research_score,
)
from aegis_alpha.turbo.config import DEFAULT_TURBO_CONFIG  # noqa: E402
from aegis_alpha.turbo.operable_feature_builder_v2 import apply_feature_set  # noqa: E402
from aegis_alpha.turbo.recent_dataset import build_recent_dataset  # noqa: E402
from aegis_alpha.turbo.snapshot_utils import normalize_turbo_symbol  # noqa: E402
from aegis_alpha.turbo.walk_forward_operable_v2 import run_walk_forward  # noqa: E402


PRESETS = {
    "all": DEFAULT_SYMBOLS,
    "mixed": (
        "BTCUSDT",
        "ETHUSDT",
        "XRPUSDT",
        "BNBUSDT",
        "AVAXUSDT",
        "SUIUSDT",
        "LINKUSDT",
        "LTCUSDT",
    ),
    "controls": ("DOGEUSDT", "ADAUSDT", "SOLUSDT"),
}
DEFAULT_MODEL_DIR = REPO_ROOT / "aegis_alpha" / "models" / "research" / "turbo_v2_short_config_optimization"
STATUS_PRIORITY = {
    "SHORT_STRONG_RESEARCH": 0,
    "SHORT_MIXED_RESEARCH": 1,
    "SHORT_BAD_RESEARCH": 2,
    "INSUFFICIENT_DATA": 3,
}
BEST_STATUS = {
    "SHORT_STRONG_RESEARCH": "STRONG_BEST",
    "SHORT_MIXED_RESEARCH": "MIXED_BEST",
    "SHORT_BAD_RESEARCH": "BAD_BEST",
    "INSUFFICIENT_DATA": "NO_VALID_CONFIG",
}
FEATURE_PREFERENCE = {"operable_v2": 0, "combined": 1, "base": 2}
LOOKBACK_PREFERENCE = {30: 0, 14: 1, 7: 2}
HORIZON_PREFERENCE = {12: 0, 24: 1}
OUTPUT_COLUMNS = (
    "rank",
    "symbol",
    "side",
    "best_status",
    "best_reason",
    "default_status",
    "promoted_from_default",
    "feature_set",
    "lookback_days",
    "horizon_candles",
    "fold_count",
    "valid_fold_count",
    "recommendation",
    "short_research_status",
    "research_score",
    "selection_score",
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
    "net_quality_lift_after_cost_proxy",
    "top_decile_net_quality_after_cost_proxy",
)


def parse_symbols(raw: str | None, preset: str) -> list[str]:
    source = raw if raw else ",".join(PRESETS[preset])
    values = [part.strip() for part in source.split(",") if part.strip()]
    return list(dict.fromkeys(normalize_turbo_symbol(value) for value in values))


def classify_optimization_config(row: dict[str, Any]) -> str:
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


def selection_score(row: dict[str, Any]) -> float:
    score = research_score(row)
    if finite(row.get("latest_fold_quality_lift")) < 0.0:
        score -= 0.20
    if finite(row.get("quality_top_decile_lift_min")) < 0.0:
        score -= 0.10
    if finite(row.get("hit8_top_decile_lift_min")) < 0.0:
        score -= 0.10
    if finite(row.get("v2_quality_corr_mean")) < 0.0:
        score -= 0.10
    if (
        row.get("latest_fold_quality_p90_mae") is not None
        and row.get("latest_fold_baseline_p90_mae") is not None
        and finite(row["latest_fold_quality_p90_mae"]) > finite(row["latest_fold_baseline_p90_mae"]) * 1.15
    ):
        score -= 0.20
    return float(score)


def rank_optimization_configs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for row in rows:
        candidate = dict(row)
        candidate["short_research_status"] = classify_optimization_config(candidate)
        candidate["research_score"] = research_score(candidate)
        candidate["selection_score"] = selection_score(candidate)
        ranked.append(candidate)
    ranked.sort(
        key=lambda row: (
            STATUS_PRIORITY[row["short_research_status"]],
            -finite(row.get("selection_score")),
            FEATURE_PREFERENCE.get(str(row.get("feature_set")), 99),
            LOOKBACK_PREFERENCE.get(int(row.get("lookback_days") or 0), 99),
            HORIZON_PREFERENCE.get(int(row.get("horizon_candles") or 0), 99),
        )
    )
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    return ranked


def select_best_short_config_for_symbol(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "short_research_status": "INSUFFICIENT_DATA",
            "best_status": "NO_VALID_CONFIG",
            "best_reason": "no_valid_configuration",
        }
    ranked = rank_optimization_configs(rows)
    leader = ranked[0]
    near_ties = [
        row for row in ranked
        if row["short_research_status"] == leader["short_research_status"]
        and finite(row["selection_score"]) >= finite(leader["selection_score"]) - 0.03
    ]
    best = min(
        near_ties,
        key=lambda row: (
            FEATURE_PREFERENCE.get(str(row.get("feature_set")), 99),
            LOOKBACK_PREFERENCE.get(int(row.get("lookback_days") or 0), 99),
            HORIZON_PREFERENCE.get(int(row.get("horizon_candles") or 0), 99),
            -finite(row.get("selection_score")),
        ),
    )
    selected = dict(best)
    selected["best_status"] = BEST_STATUS[selected["short_research_status"]]
    selected["best_reason"] = (
        f"priority={selected['short_research_status']};"
        f"selection_score={finite(selected.get('selection_score')):.4f};"
        "tie_preference=operable_v2_then_30d_then_h12"
    )
    return selected


def select_best_by_symbol(rows: list[dict[str, Any]], symbols: list[str]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for symbol in symbols:
        symbol_rows = [row for row in rows if row.get("symbol") == symbol]
        best = select_best_short_config_for_symbol(symbol_rows)
        best["symbol"] = symbol
        best["side"] = SIDE
        selected.append(best)
    return selected


def default_configuration_for_symbol(rows: list[dict[str, Any]], symbol: str) -> dict[str, Any] | None:
    defaults = [
        row for row in rows
        if row.get("symbol") == symbol
        and row.get("feature_set") == "operable_v2"
        and int(row.get("lookback_days") or 0) == 30
        and int(row.get("horizon_candles") or 0) in {12, 24}
    ]
    return select_best_short_config_for_symbol(defaults) if defaults else None


def annotate_default_comparison(best_rows: list[dict[str, Any]], all_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for row in best_rows:
        candidate = dict(row)
        default = default_configuration_for_symbol(all_rows, str(row["symbol"]))
        candidate["default_status"] = default.get("short_research_status") if default else "NOT_EVALUATED"
        candidate["promoted_from_default"] = (
            candidate.get("best_status") == "STRONG_BEST"
            and candidate["default_status"] in {"SHORT_MIXED_RESEARCH", "SHORT_BAD_RESEARCH"}
        )
        annotated.append(candidate)
    return annotated


def build_configurations(
    symbols: list[str],
    feature_sets: list[str],
    lookbacks: list[int],
    horizons: list[int],
    max_configs: int | None,
) -> list[tuple[str, str, int, int]]:
    rows = list(dict.fromkeys(
        (symbol, feature_set, lookback, horizon)
        for symbol in symbols
        for feature_set in feature_sets
        for lookback in lookbacks
        for horizon in horizons
    ))
    return rows[:max_configs] if max_configs else rows


def write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...] | None = None) -> None:
    fields = list(columns or sorted({key for row in rows for key in row}) or ["symbol", "side"])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _num(value: Any) -> str:
    return "null" if value is None else f"{float(value):.4f}"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    best = report["best_by_symbol"]
    lines = [
        f"# Aegis Turbo V2 SHORT Config Optimization {report['created_at']}",
        "",
        "## Safety",
        "",
        "- Mode: `RESEARCH_ONLY`.",
        "- No model artifacts are saved; no `active/` models or active manifests are modified.",
        "- No live inference, YAML, thresholds or PM2 action is involved.",
        f"- Cost proxy: `{report['estimated_fee_bps']} bps fee + {report['estimated_slippage_bps']} bps slippage`.",
        "",
        "## Best By Symbol",
        "",
        "| Symbol | Status | Set | Window | H | Score | Hit8 AUC | Hit8 Lift | Quality Lift | Quality Corr | Danger AUC | Latest Quality | P90 Delta |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best:
        lines.append(
            f"| {row['symbol']} | {row['best_status']} | {row.get('feature_set', '-')} | "
            f"{row.get('lookback_days', '-')} | {row.get('horizon_candles', '-')} | "
            f"{_num(row.get('selection_score'))} | {_num(row.get('v2_hit8_auc_mean'))} | "
            f"{_num(row.get('hit8_top_decile_lift_mean'))} | {_num(row.get('quality_top_decile_lift_mean'))} | "
            f"{_num(row.get('v2_quality_corr_mean'))} | {_num(row.get('v2_danger_auc_mean'))} | "
            f"{_num(row.get('latest_fold_quality_lift'))} | {_num(row.get('latest_p90_mae_delta'))} |"
        )
    for key, title in (
        ("strong_best", "Strong Best"),
        ("still_mixed", "Mixed Best"),
        ("bad_best", "Bad Best"),
    ):
        lines.extend(["", f"## {title}", ""])
        candidates = report[key]
        if not candidates:
            lines.append("- None.")
        for row in candidates:
            lines.append(
                f"- `{row['symbol']}`: `{row.get('feature_set')} {row.get('lookback_days')}d h{row.get('horizon_candles')}` "
                f"score `{_num(row.get('selection_score'))}`; default `{row.get('default_status')}`."
            )
    lines.extend([
        "",
        "## Top 20 Configurations",
        "",
        "| Rank | Symbol | Status | Set | Window | H | Score | Quality Lift | Hit8 Lift |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in report["ranking"][:20]:
        lines.append(
            f"| {row['rank']} | {row['symbol']} | {row['short_research_status']} | {row['feature_set']} | "
            f"{row['lookback_days']} | {row['horizon_candles']} | {_num(row['selection_score'])} | "
            f"{_num(row.get('quality_top_decile_lift_mean'))} | {_num(row.get('hit8_top_decile_lift_mean'))} |"
        )
    lines.extend([
        "",
        "## Winning Configuration Counts",
        "",
        f"- Feature set: `{report['winning_feature_set_counts']}`.",
        f"- Lookback: `{report['winning_lookback_counts']}`.",
        f"- Horizon: `{report['winning_horizon_counts']}`.",
        f"- Promoted from default MIXED/BAD to STRONG: `{len(report['promoted_to_strong'])}`.",
        "",
        "## Decision",
        "",
        f"- Recommendation: `{report['next_recommendation']}`.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    symbols = parse_symbols(args.symbols, args.preset)
    feature_sets = parse_csv_strings(args.feature_sets)
    invalid = [feature_set for feature_set in feature_sets if feature_set not in FEATURE_SETS]
    if invalid:
        raise ValueError(f"unsupported feature sets: {invalid}")
    lookbacks = parse_csv_ints(args.lookback_days)
    horizons = parse_csv_ints(args.horizons)
    model_dir = Path(args.model_dir)
    validate_research_model_dir(model_dir)
    configurations = build_configurations(symbols, feature_sets, lookbacks, horizons, args.max_configs)
    market_cache: dict[str, Any] = {}
    dataset_cache: dict[tuple[str, int, str], dict[str, Any]] = {}
    result_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    seeded_rows: list[dict[str, Any]] = []
    seeded_symbols: list[str] = []
    if args.seed_report:
        seed_payload = json.loads(Path(args.seed_report).read_text(encoding="utf-8"))
        seeded_rows = [dict(row) for row in seed_payload.get("ranking", [])]
        seeded_symbols = list(dict.fromkeys(str(row.get("symbol")) for row in seeded_rows if row.get("symbol")))
        errors.extend(seed_payload.get("errors", []))

    for symbol, feature_set, lookback, horizon in configurations:
        try:
            if symbol not in market_cache:
                market_cache[symbol] = load_signal_market(DEFAULT_TURBO_CONFIG.config_path, symbol_override=symbol)
            dataset_key = (symbol, lookback, feature_set)
            if dataset_key not in dataset_cache:
                base = build_recent_dataset(symbol, lookback, save=False, market=market_cache[symbol])["dataset"]
                dataset_cache[dataset_key] = apply_feature_set(base, market_cache[symbol], feature_set)
            result = run_walk_forward(
                dataset_cache[dataset_key],
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

    all_summaries = seeded_rows + [result["summary"] for result in result_rows]
    ranking = rank_optimization_configs(all_summaries)
    all_symbols = list(dict.fromkeys(seeded_symbols + symbols))
    best_rows = annotate_default_comparison(select_best_by_symbol(all_summaries, all_symbols), all_summaries)
    strong_best = [row for row in best_rows if row["best_status"] == "STRONG_BEST"]
    still_mixed = [row for row in best_rows if row["best_status"] == "MIXED_BEST"]
    bad_best = [row for row in best_rows if row["best_status"] in {"BAD_BEST", "NO_VALID_CONFIG"}]
    promoted_to_strong = [row for row in best_rows if row.get("promoted_from_default")]
    token = utc_stamp()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "md": out_dir / f"aegis_short_v2_config_optimization_{token}.md",
        "json": out_dir / f"aegis_short_v2_config_optimization_{token}.json",
        "all_configs_csv": out_dir / f"aegis_short_v2_config_optimization_all_configs_{token}.csv",
        "best_by_symbol_csv": out_dir / f"aegis_short_v2_config_optimization_best_by_symbol_{token}.csv",
        "promoted_to_strong_csv": out_dir / f"aegis_short_v2_config_optimization_promoted_to_strong_{token}.csv",
        "still_mixed_csv": out_dir / f"aegis_short_v2_config_optimization_still_mixed_{token}.csv",
        "bad_csv": out_dir / f"aegis_short_v2_config_optimization_bad_{token}.csv",
    }
    report: dict[str, Any] = {
        "schema_version": "aegis_short_operable_v2_config_optimization_v1",
        "created_at": utc_now().isoformat(),
        "mode": "RESEARCH_ONLY",
        "side": SIDE,
        "preset": args.preset,
        "symbols_requested": symbols,
        "seed_report": args.seed_report,
        "seeded_configuration_count": len(seeded_rows),
        "feature_sets_requested": feature_sets,
        "lookback_days_requested": lookbacks,
        "horizons_requested": horizons,
        "fold_count": int(args.fold_count),
        "fast": bool(args.fast),
        "skip_existing": bool(args.skip_existing),
        "estimated_fee_bps": float(args.fee_bps),
        "estimated_slippage_bps": float(args.slippage_bps),
        "save_models": False,
        "shadow_models_generated": False,
        "active_manifest_touched": False,
        "live_inference_changed": False,
        "configuration_count": len(ranking),
        "ranking": ranking,
        "best_by_symbol": best_rows,
        "strong_best": strong_best,
        "promoted_to_strong": promoted_to_strong,
        "still_mixed": still_mixed,
        "bad_best": bad_best,
        "errors": errors,
        "best_status_counts": dict(Counter(row["best_status"] for row in best_rows)),
        "winning_feature_set_counts": dict(Counter(row.get("feature_set") for row in best_rows if row.get("feature_set"))),
        "winning_lookback_counts": dict(Counter(row.get("lookback_days") for row in best_rows if row.get("lookback_days"))),
        "winning_horizon_counts": dict(Counter(row.get("horizon_candles") for row in best_rows if row.get("horizon_candles"))),
        "next_recommendation": (
            "generate_shadow_artifacts_for_strong_best_only_metadata_only_default_off"
            if strong_best else "do_not_generate_shadow_artifacts_continue_research"
        ),
        "paths": {key: str(value) for key, value in paths.items()},
    }
    paths["json"].write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(paths["md"], report)
    write_csv(paths["all_configs_csv"], ranking, OUTPUT_COLUMNS)
    write_csv(paths["best_by_symbol_csv"], best_rows, OUTPUT_COLUMNS)
    write_csv(paths["promoted_to_strong_csv"], promoted_to_strong, OUTPUT_COLUMNS)
    write_csv(paths["still_mixed_csv"], still_mixed, OUTPUT_COLUMNS)
    write_csv(paths["bad_csv"], bad_best, OUTPUT_COLUMNS)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Research-only optimizer for Aegis Turbo V2 SHORT configurations.")
    parser.add_argument("--preset", choices=tuple(PRESETS), default="mixed")
    parser.add_argument("--symbols")
    parser.add_argument("--feature-sets", default="base,operable_v2,combined")
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
    parser.add_argument("--max-configs", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--seed-report")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({
        "paths": report["paths"],
        "configuration_count": report["configuration_count"],
        "best_status_counts": report["best_status_counts"],
        "strong_best": [row["symbol"] for row in report["strong_best"]],
        "promoted_to_strong": [row["symbol"] for row in report["promoted_to_strong"]],
        "errors": report["errors"],
        "next_recommendation": report["next_recommendation"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
