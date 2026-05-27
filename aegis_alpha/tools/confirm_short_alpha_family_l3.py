#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.signals.common import load_signal_market  # noqa: E402
from aegis_alpha.tools.confirm_short_v3_lockbox import build_last_block_fold  # noqa: E402
from aegis_alpha.tools.train_short_alpha_family_l2_research import (  # noqa: E402
    DECISION_MODES,
    FEATURE_MODES,
    _fit_fold_predictions,
    _mode_row,
    compute_alpha_target_arrays,
    select_alpha_family_features,
)
from aegis_alpha.turbo.config import DEFAULT_TURBO_CONFIG  # noqa: E402
from aegis_alpha.turbo.operable_feature_builder_v3 import FEATURE_SETS, apply_feature_set  # noqa: E402
from aegis_alpha.turbo.recent_dataset import build_recent_dataset  # noqa: E402
from aegis_alpha.turbo.snapshot_utils import normalize_turbo_symbol  # noqa: E402
from aegis_alpha.turbo.walk_forward_operable_v2 import temporal_folds  # noqa: E402


MODE = "RESEARCH_ONLY"
SIDE = "SHORT"
SCHEMA_VERSION = "aegis_short_alpha_family_l3_confirmation_v1"
LOCKBOX_MODES = ("last-block", "rolling-forward", "recent-only")
DEFAULT_FROZEN_ALPHA_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "symbol": "BNBUSDT",
        "side": SIDE,
        "alpha_family": "momentum_burst_lower_target",
        "feature_set": "combined_v3",
        "feature_mode": "selected_family",
        "lookback_days": 30,
        "target_name": "hit5_before_minus3",
        "horizon_candles": 12,
        "decision_mode": "hit_primary",
        "source": "phase_l2_alpha_promising",
    },
    {
        "symbol": "SOLUSDT",
        "side": SIDE,
        "alpha_family": "momentum_burst_lower_target",
        "feature_set": "combined_v3",
        "feature_mode": "selected_family",
        "lookback_days": 30,
        "target_name": "hit6_before_minus4",
        "horizon_candles": 24,
        "decision_mode": "hit_primary",
        "source": "phase_l2_alpha_promising",
    },
    {
        "symbol": "XRPUSDT",
        "side": SIDE,
        "alpha_family": "momentum_burst_lower_target",
        "feature_set": "combined_v3",
        "feature_mode": "selected_family",
        "lookback_days": 30,
        "target_name": "hit5_before_minus3",
        "horizon_candles": 24,
        "decision_mode": "hit_primary",
        "source": "phase_l2_alpha_promising",
    },
)
CSV_COLUMNS = (
    "symbol",
    "side",
    "alpha_family",
    "feature_set",
    "feature_mode",
    "lookback_days",
    "target_name",
    "horizon_candles",
    "decision_mode",
    "source",
    "lockbox_mode",
    "l3_status",
    "l3_reason",
    "recommended_next_step",
    "model_status",
    "train_samples",
    "validation_samples",
    "test_samples",
    "valid_fold_count",
    "feature_count",
    "missing_family_features",
    "baseline_target_hit_rate",
    "baseline_trade_quality",
    "baseline_mae_danger",
    "baseline_p90_mae",
    "selected_fraction",
    "selected_count",
    "selected_target_hit_rate",
    "selected_target_lift",
    "selected_quality_mean",
    "selected_quality_lift",
    "selected_net_quality_lift",
    "selected_mae_danger_rate",
    "selected_mae_danger_delta",
    "selected_p90_mae",
    "selected_p90_mae_delta",
    "selected_avg_mfe",
    "selected_avg_mae",
    "p90_mae_ratio",
    "mae_danger_ratio",
    "target_auc",
    "target_average_precision",
    "quality_corr",
    "danger_auc",
    "danger_filter_usefulness",
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_stamp() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%SZ")


def finite(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def default_frozen_alpha_configs() -> list[dict[str, Any]]:
    return deepcopy(list(DEFAULT_FROZEN_ALPHA_CONFIGS))


def _normalized_config(raw: dict[str, Any]) -> dict[str, Any]:
    side = str(raw.get("side", SIDE)).upper()
    if side != SIDE:
        raise ValueError(f"L3 alpha lockbox is restricted to SHORT configs: {side}")
    feature_set = str(raw.get("feature_set", "")).lower()
    feature_mode = str(raw.get("feature_mode", "")).lower()
    decision_mode = str(raw.get("decision_mode", ""))
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unsupported frozen feature_set: {feature_set}")
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"unsupported frozen feature_mode: {feature_mode}")
    if decision_mode not in DECISION_MODES:
        raise ValueError(f"unsupported frozen decision_mode: {decision_mode}")
    return {
        "symbol": normalize_turbo_symbol(str(raw["symbol"])),
        "side": side,
        "alpha_family": str(raw["alpha_family"]),
        "feature_set": feature_set,
        "feature_mode": feature_mode,
        "lookback_days": int(raw["lookback_days"]),
        "target_name": str(raw["target_name"]),
        "horizon_candles": int(raw.get("horizon_candles", raw.get("horizon"))),
        "decision_mode": decision_mode,
        "source": str(raw.get("source", "phase_l2_alpha_promising")),
    }


def load_frozen_alpha_configs(configs_json: str | None, symbols_raw: str | None) -> list[dict[str, Any]]:
    if configs_json:
        payload = json.loads(Path(configs_json).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("configs", payload.get("promising", payload.get("best_by_symbol", [])))
        configs = [_normalized_config(dict(item)) for item in payload]
    else:
        configs = default_frozen_alpha_configs()
    requested: set[str] | None = None
    if symbols_raw:
        requested = {
            normalize_turbo_symbol(value.strip())
            for value in symbols_raw.split(",")
            if value.strip()
        }
    selected = [dict(config) for config in configs if requested is None or config["symbol"] in requested]
    seen: set[str] = set()
    for config in selected:
        if config["symbol"] in seen:
            raise ValueError(f"multiple frozen L3 configs for symbol: {config['symbol']}")
        seen.add(config["symbol"])
    return selected


def _required_selected_count(test_samples: int) -> int:
    return min(30, max(1, math.ceil(float(test_samples) * 0.05)))


def _risk_ratios(row: dict[str, Any]) -> tuple[float, float]:
    baseline_p90 = finite(row.get("baseline_p90_mae"), 0.0) or 0.0
    selected_p90 = finite(row.get("selected_p90_mae"), float("inf"))
    baseline_danger = finite(row.get("baseline_mae_danger"), 0.0) or 0.0
    selected_danger = finite(row.get("selected_mae_danger_rate"), float("inf"))
    p90_ratio = (
        float(selected_p90) / max(baseline_p90, 1e-12)
        if selected_p90 is not None else float("inf")
    )
    danger_ratio = (
        float(selected_danger) / max(baseline_danger, 1e-12)
        if selected_danger is not None else float("inf")
    )
    return p90_ratio, danger_ratio


def classify_l3_alpha_candidate(
    row: dict[str, Any],
    *,
    min_test_samples: int = 300,
    strict: bool = False,
) -> str:
    if (
        row.get("config_error")
        or row.get("model_status") != "trained"
        or int(row.get("test_samples") or 0) < min_test_samples
        or int(row.get("feature_count") or 0) <= 0
    ):
        return "L3_ALPHA_INSUFFICIENT_DATA"
    test_samples = int(row.get("test_samples") or 0)
    selected_count = int(row.get("selected_count") or 0)
    if selected_count <= 0:
        return "L3_ALPHA_FAILED"
    low_count = selected_count < _required_selected_count(test_samples)
    target_lift = finite(row.get("selected_target_lift"), -1.0)
    quality_lift = finite(row.get("selected_quality_lift"), -1.0)
    net_quality = finite(row.get("selected_net_quality_lift"), -1.0)
    target_auc = finite(row.get("target_auc"), 0.0)
    quality_corr = finite(row.get("quality_corr"), -1.0)
    danger_help = finite(row.get("danger_filter_usefulness"), 0.0)
    p90_ratio, danger_ratio = _risk_ratios(row)
    symbol = str(row.get("symbol", ""))
    symbol_gate_failed = (
        symbol == "BNBUSDT" and float(net_quality) <= 0.02
    ) or (
        symbol == "SOLUSDT" and float(target_lift) <= 0.03
    ) or (
        symbol == "XRPUSDT" and float(quality_lift) <= 0.03
    )
    failed = (
        float(target_lift) <= 0.0
        or float(quality_lift) <= 0.0
        or float(net_quality) <= 0.0
        or p90_ratio > 1.30
        or danger_ratio > 1.40
        or (float(target_auc) < 0.45 and float(quality_corr) < 0.0)
        or symbol_gate_failed
    )
    if failed:
        return "L3_ALPHA_FAILED"
    confirmed = (
        not low_count
        and p90_ratio <= (1.10 if strict else 1.15)
        and danger_ratio <= (1.10 if strict else 1.20)
        and (float(target_auc) > 0.53 or float(quality_corr) > 0.03)
        and float(danger_help) >= (-0.10 if strict else -0.20)
    )
    return "L3_ALPHA_CONFIRMED" if confirmed else "L3_ALPHA_WEAK"


def l3_reason(row: dict[str, Any]) -> str:
    status = row["l3_status"]
    if status == "L3_ALPHA_CONFIRMED":
        return "frozen_alpha_config_improves_target_quality_after_cost_with_controlled_risk"
    if status == "L3_ALPHA_INSUFFICIENT_DATA":
        return "insufficient_lockbox_samples_model_diversity_or_family_features"
    if row.get("symbol") == "BNBUSDT" and (finite(row.get("selected_net_quality_lift"), -1.0) or -1.0) <= 0.02:
        return "bnb_minimum_net_quality_lift_not_met"
    if row.get("symbol") == "SOLUSDT" and (finite(row.get("selected_target_lift"), -1.0) or -1.0) <= 0.03:
        return "sol_minimum_target_lift_not_met"
    if row.get("symbol") == "XRPUSDT" and (finite(row.get("selected_quality_lift"), -1.0) or -1.0) <= 0.03:
        return "xrp_minimum_quality_lift_not_met"
    if status == "L3_ALPHA_FAILED":
        return "frozen_alpha_edge_or_risk_control_not_confirmed"
    return "frozen_alpha_partially_improves_but_confirmation_gate_not_met"


def recommended_next_step(status: str) -> str:
    if status == "L3_ALPHA_CONFIRMED":
        return "eligible_for_metadata_only_shadow_artifact_after_review"
    if status == "L3_ALPHA_WEAK":
        return "continue_research_no_shadow_artifact"
    if status == "L3_ALPHA_FAILED":
        return "disable_alpha_hypothesis_for_now"
    return "collect_more_data_before_decision"


def _folds_for_mode(sample_count: int, args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.lockbox_mode == "rolling-forward":
        return temporal_folds(
            sample_count,
            fold_count=int(args.fold_count),
            train_ratio=0.50,
            validation_ratio=0.15,
            test_ratio=float(args.lockbox_test_ratio),
            expanding_window=True,
            min_train_samples=int(args.min_train_samples),
            min_test_samples=int(args.min_test_samples),
        )
    fold = build_last_block_fold(
        sample_count,
        test_ratio=float(args.lockbox_test_ratio),
        min_train_samples=int(args.min_train_samples),
        min_test_samples=int(args.min_test_samples),
        recent_only=args.lockbox_mode == "recent-only",
    )
    return [fold] if fold is not None else []


def _finalize_row(row: dict[str, Any], config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    row.update(config)
    row["lockbox_mode"] = args.lockbox_mode
    row["valid_fold_count"] = int(row.get("valid_fold_count", 1 if row.get("model_status") == "trained" else 0))
    p90_ratio, danger_ratio = _risk_ratios(row)
    row["p90_mae_ratio"] = p90_ratio
    row["mae_danger_ratio"] = danger_ratio
    row["l3_status"] = classify_l3_alpha_candidate(
        row,
        min_test_samples=int(args.min_test_samples),
        strict=bool(args.strict),
    )
    row["l3_reason"] = l3_reason(row)
    row["recommended_next_step"] = recommended_next_step(row["l3_status"])
    return row


def evaluate_frozen_config(
    config: dict[str, Any],
    dataset: dict[str, Any],
    targets: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    folds = _folds_for_mode(len(np.asarray(dataset["X"])), args)
    fold_rows: list[dict[str, Any]] = []
    base_config = {
        **config,
        "feature_count": int(np.asarray(dataset["X"]).shape[1]),
        "missing_family_features": ",".join(dataset.get("missing_family_features", [])),
    }
    for fold in folds:
        trained = _fit_fold_predictions(
            dataset,
            targets,
            fold,
            symbol=config["symbol"],
            lookback_days=int(config["lookback_days"]),
            target_name=config["target_name"],
            max_iter=60 if args.fast else 120,
        )
        row = _mode_row(
            trained,
            targets,
            config=base_config,
            decision_mode=config["decision_mode"],
            cost_proxy=(float(args.fee_bps) + float(args.slippage_bps)) / 10000.0,
        )
        fold_rows.append(_finalize_row(row, config, args))
    if not fold_rows:
        empty = {
            **base_config,
            "lockbox_mode": args.lockbox_mode,
            "model_status": "insufficient_split_samples",
            "test_samples": 0,
            "valid_fold_count": 0,
        }
        return _finalize_row(empty, config, args), []
    latest = dict(fold_rows[-1])
    latest["valid_fold_count"] = sum(1 for row in fold_rows if row.get("model_status") == "trained")
    latest = _finalize_row(latest, config, args)
    return latest, fold_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _num(value: Any) -> str:
    number = finite(value)
    return "null" if number is None else f"{number:.4f}"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        f"# Aegis SHORT Alpha-Family L3 Confirmation {report['created_at']}",
        "",
        "## Safety",
        "",
        "- Mode: `RESEARCH_ONLY`.",
        "- Alpha family, target, horizon, feature mode and decision mode are frozen from Phase L2.",
        "- No candidate alternatives are ranked or selected in this evaluation.",
        "- No model artifacts, `active/` models, active manifests or live inference are changed.",
        "- Methodological note: true independent confirmation requires observations not used during L2 selection.",
        "",
        "## Results",
        "",
        "| Symbol | Alpha Family | Target | Horizon | Decision Mode | Status | Target Lift | Quality Lift | Net Quality Lift | P90 Delta | Selected Fraction | Reason |",
        "|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["results"]:
        lines.append(
            f"| {row['symbol']} | {row['alpha_family']} | {row['target_name']} | {row['horizon_candles']} | "
            f"{row['decision_mode']} | {row['l3_status']} | {_num(row.get('selected_target_lift'))} | "
            f"{_num(row.get('selected_quality_lift'))} | {_num(row.get('selected_net_quality_lift'))} | "
            f"{_num(row.get('selected_p90_mae_delta'))} | {_num(row.get('selected_fraction'))} | "
            f"{row['l3_reason']} |"
        )
    for key, title in (
        ("confirmed", "Confirmed"),
        ("weak", "Weak"),
        ("failed", "Failed"),
        ("insufficient_data", "Insufficient Data"),
    ):
        lines.extend(["", f"## {title}", ""])
        rows = report[key]
        if not rows:
            lines.append("- None.")
        for row in rows:
            lines.append(f"- `{row['symbol']}`: `{row['l3_status']}` via `{row['decision_mode']}`.")
    lines.extend(["", "## Recommendation", "", f"- `{report['next_recommendation']}`.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    configs = load_frozen_alpha_configs(args.configs_json, args.symbols)
    context_markets: dict[str, Any] = {}
    if not args.disable_cross_context and any(config["feature_set"] == "combined_v3" for config in configs):
        context_markets = {
            symbol: load_signal_market(DEFAULT_TURBO_CONFIG.config_path, symbol_override=symbol)
            for symbol in ("BTCUSDT", "ETHUSDT")
        }
    results: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    cost_proxy = (float(args.fee_bps) + float(args.slippage_bps)) / 10000.0
    for config in configs:
        try:
            market = load_signal_market(DEFAULT_TURBO_CONFIG.config_path, symbol_override=config["symbol"])
            base = build_recent_dataset(config["symbol"], config["lookback_days"], save=False, market=market)["dataset"]
            combined = apply_feature_set(base, market, config["feature_set"], context_markets=context_markets)
            dataset = select_alpha_family_features(combined, config["alpha_family"], config["feature_mode"])
            targets = compute_alpha_target_arrays(
                market,
                np.asarray(base["step"], dtype=np.int64),
                side=SIDE,
                target_name=config["target_name"],
                horizon=config["horizon_candles"],
                cost_proxy=cost_proxy,
            )
            result, local_folds = evaluate_frozen_config(config, dataset, targets, args)
            results.append(result)
            fold_rows.extend(local_folds)
        except Exception as exc:
            errors.append({**config, "error": repr(exc)})
            row = {
                **config,
                "lockbox_mode": args.lockbox_mode,
                "model_status": "evaluation_error",
                "feature_count": 0,
                "test_samples": 0,
                "valid_fold_count": 0,
                "config_error": repr(exc),
            }
            results.append(_finalize_row(row, config, args))
    confirmed = [row for row in results if row["l3_status"] == "L3_ALPHA_CONFIRMED"]
    weak = [row for row in results if row["l3_status"] == "L3_ALPHA_WEAK"]
    failed = [row for row in results if row["l3_status"] == "L3_ALPHA_FAILED"]
    insufficient = [row for row in results if row["l3_status"] == "L3_ALPHA_INSUFFICIENT_DATA"]
    token = utc_stamp()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "md": out_dir / f"aegis_short_alpha_l3_confirmation_{token}.md",
        "json": out_dir / f"aegis_short_alpha_l3_confirmation_{token}.json",
        "summary_csv": out_dir / f"aegis_short_alpha_l3_summary_{token}.csv",
        "confirmed_csv": out_dir / f"aegis_short_alpha_l3_confirmed_{token}.csv",
        "weak_csv": out_dir / f"aegis_short_alpha_l3_weak_{token}.csv",
        "failed_csv": out_dir / f"aegis_short_alpha_l3_failed_{token}.csv",
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now().isoformat(),
        "mode": MODE,
        "side": SIDE,
        "lockbox_mode": args.lockbox_mode,
        "lockbox_test_ratio": float(args.lockbox_test_ratio),
        "strict": bool(args.strict),
        "frozen_alpha_configs": configs,
        "evaluated_config_count": len(configs),
        "selection_policy": "one_frozen_l2_alpha_configuration_per_symbol_no_alternative_search",
        "independence_note": "strict_independence_requires_observations_not_used_during_phase_l2_selection",
        "disable_cross_context": bool(args.disable_cross_context),
        "cost_proxy": {"fee_bps": float(args.fee_bps), "slippage_bps": float(args.slippage_bps)},
        "results": results,
        "fold_results": fold_rows,
        "confirmed": confirmed,
        "weak": weak,
        "failed": failed,
        "insufficient_data": insufficient,
        "status_counts": dict(Counter(row["l3_status"] for row in results)),
        "errors": errors,
        "models_trained_in_memory_only": True,
        "model_artifacts_written": False,
        "shadow_models_generated": False,
        "active_manifest_touched": False,
        "live_inference_changed": False,
        "next_recommendation": (
            "add_l3_confirmed_to_metadata_only_shadow_candidate_list_after_review"
            if confirmed else "no_l3_shadow_candidates_continue_research"
        ),
        "paths": {name: str(path) for name, path in paths.items()},
    }
    paths["json"].write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(paths["md"], report)
    write_csv(paths["summary_csv"], results)
    write_csv(paths["confirmed_csv"], confirmed)
    write_csv(paths["weak_csv"], weak)
    write_csv(paths["failed_csv"], failed)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Research-only frozen confirmation for SHORT alpha-family L2 candidates.")
    parser.add_argument("--configs-json")
    parser.add_argument("--symbols")
    parser.add_argument("--out-dir", default="/home/jasan/Develop")
    parser.add_argument("--lockbox-mode", choices=LOCKBOX_MODES, default="last-block")
    parser.add_argument("--lockbox-test-ratio", type=float, default=0.20)
    parser.add_argument("--fold-count", type=int, default=4)
    parser.add_argument("--min-train-samples", type=int, default=1000)
    parser.add_argument("--min-test-samples", type=int, default=300)
    parser.add_argument("--fee-bps", type=float, default=8.0)
    parser.add_argument("--slippage-bps", type=float, default=3.0)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--disable-cross-context", action="store_true")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({
        "paths": report["paths"],
        "results": [
            {
                "symbol": row["symbol"],
                "target": row["target_name"],
                "horizon": row["horizon_candles"],
                "mode": row["decision_mode"],
                "status": row["l3_status"],
            }
            for row in report["results"]
        ],
        "errors": report["errors"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
