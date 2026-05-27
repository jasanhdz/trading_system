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

from aegis_alpha.signals.common import load_signal_market  # noqa: E402
from aegis_alpha.tools.confirm_short_v3_lockbox import build_last_block_fold  # noqa: E402
from aegis_alpha.tools.evaluate_ada_short_operable_v2_matrix import finite  # noqa: E402
from aegis_alpha.tools.repair_short_v3_failure_modes import (  # noqa: E402
    _baseline,
    _danger_filter_usefulness,
    evaluate_repair_mode,
)
from aegis_alpha.turbo.config import DEFAULT_TURBO_CONFIG  # noqa: E402
from aegis_alpha.turbo.operable_feature_builder_v3 import FEATURE_SETS, apply_feature_set  # noqa: E402
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
from aegis_alpha.turbo.walk_forward_operable_v2 import temporal_folds  # noqa: E402


MODE = "RESEARCH_ONLY"
SIDE = "SHORT"
LOCKBOX_MODES = ("last-block", "rolling-forward", "recent-only")
SUPPORTED_REPAIR_MODES = (
    "hit8_primary",
    "quality_primary",
    "danger_filtered",
    "quality_primary_danger_filtered",
    "hit8_primary_danger_filtered",
    "top_bucket_only",
    "top_bucket_only_danger_filtered",
)
DEFAULT_FROZEN_REPAIRED_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "symbol": "LINKUSDT",
        "side": SIDE,
        "feature_set": "operable_v3",
        "lookback_days": 14,
        "horizon_candles": 12,
        "repair_mode": "top_bucket_only",
        "source": "phase_j0_repair",
    },
    {
        "symbol": "ADAUSDT",
        "side": SIDE,
        "feature_set": "operable_v2",
        "lookback_days": 30,
        "horizon_candles": 24,
        "repair_mode": "hit8_primary",
        "source": "phase_j0_repair",
    },
    {
        "symbol": "SOLUSDT",
        "side": SIDE,
        "feature_set": "operable_v2",
        "lookback_days": 14,
        "horizon_candles": 12,
        "repair_mode": "top_bucket_only",
        "source": "phase_j0_repair",
    },
    {
        "symbol": "BTCUSDT",
        "side": SIDE,
        "feature_set": "combined_v3",
        "lookback_days": 30,
        "horizon_candles": 12,
        "repair_mode": "top_bucket_only",
        "source": "phase_j0_repair",
        "warning": "fragile_hit8_lift",
    },
    {
        "symbol": "DOGEUSDT",
        "side": SIDE,
        "feature_set": "operable_v2",
        "lookback_days": 30,
        "horizon_candles": 12,
        "repair_mode": "hit8_primary",
        "source": "phase_j0_repair",
        "warning": "previous_lockbox_failed",
    },
)
CSV_COLUMNS = (
    "symbol",
    "side",
    "feature_set",
    "lookback_days",
    "horizon_candles",
    "repair_mode",
    "source",
    "warning",
    "lockbox_mode",
    "repair_lockbox_status",
    "repair_lockbox_reason",
    "recommended_next_step",
    "train_samples",
    "validation_samples",
    "test_samples",
    "baseline_hit8",
    "baseline_quality",
    "baseline_mae_danger",
    "baseline_p90_mae",
    "selected_fraction",
    "selected_count",
    "selected_hit8_rate",
    "selected_hit8_lift",
    "selected_quality_mean",
    "selected_quality_lift",
    "selected_net_quality_lift_after_cost",
    "selected_mae_danger_rate",
    "selected_mae_danger_delta",
    "selected_p90_mae",
    "selected_p90_mae_delta",
    "p90_mae_ratio",
    "selected_avg_mfe",
    "selected_avg_mae",
    "hit8_auc",
    "quality_corr",
    "danger_auc",
    "danger_filter_usefulness",
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_stamp() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%SZ")


def default_frozen_repaired_configs() -> list[dict[str, Any]]:
    return [dict(config) for config in DEFAULT_FROZEN_REPAIRED_CONFIGS]


def _normalized_config(raw: dict[str, Any]) -> dict[str, Any]:
    feature_set = str(raw.get("feature_set", "")).lower()
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unsupported frozen feature_set: {feature_set}")
    side = str(raw.get("side", SIDE)).upper()
    if side != SIDE:
        raise ValueError(f"repair lockbox is restricted to SHORT configs: {side}")
    repair_mode = str(raw.get("repair_mode", ""))
    if repair_mode not in SUPPORTED_REPAIR_MODES:
        raise ValueError(f"unsupported frozen repair_mode: {repair_mode}")
    horizon = raw.get("horizon_candles", raw.get("horizon"))
    normalized = {
        "symbol": normalize_turbo_symbol(str(raw["symbol"])),
        "side": side,
        "feature_set": feature_set,
        "lookback_days": int(raw["lookback_days"]),
        "horizon_candles": int(horizon),
        "repair_mode": repair_mode,
        "source": str(raw.get("source", "phase_j0_repair")),
    }
    if raw.get("warning"):
        normalized["warning"] = str(raw["warning"])
    return normalized


def load_frozen_repaired_configs(configs_json: str | None, symbols_raw: str | None) -> list[dict[str, Any]]:
    if configs_json:
        payload = json.loads(Path(configs_json).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("configs", payload.get("repaired_confirmed", payload.get("best_by_symbol", [])))
        configs = [_normalized_config(dict(item)) for item in payload]
    else:
        configs = default_frozen_repaired_configs()
    requested = None
    if symbols_raw:
        requested = {
            normalize_turbo_symbol(item.strip())
            for item in symbols_raw.split(",")
            if item.strip()
        }
    selected = [dict(config) for config in configs if requested is None or config["symbol"] in requested]
    seen: set[str] = set()
    for config in selected:
        if config["symbol"] in seen:
            raise ValueError(f"multiple frozen repaired configs for symbol: {config['symbol']}")
        seen.add(config["symbol"])
    return selected


def _required_selected_count(test_samples: int) -> int:
    return min(30, max(1, math.ceil(test_samples * 0.05)))


def classify_repair_lockbox_candidate(
    row: dict[str, Any],
    *,
    min_test_samples: int = 300,
    strict: bool = False,
) -> str:
    if (
        row.get("config_error")
        or row.get("model_status") != "trained"
        or int(row.get("test_samples") or 0) < min_test_samples
    ):
        return "REPAIR_LOCKBOX_INSUFFICIENT_DATA"
    test_samples = int(row.get("test_samples") or 0)
    selected_count = int(row.get("selected_count") or 0)
    if selected_count <= 0:
        return "REPAIR_LOCKBOX_FAILED"
    count_low = selected_count < _required_selected_count(test_samples)
    hit8_lift = finite(row.get("selected_hit8_lift"))
    quality_lift = finite(row.get("selected_quality_lift"))
    net_quality = finite(row.get("selected_net_quality_lift_after_cost"))
    p90_ratio = (
        finite(row.get("selected_p90_mae")) / max(finite(row.get("baseline_p90_mae")), 1e-12)
        if row.get("selected_p90_mae") is not None and row.get("baseline_p90_mae") is not None
        else float("inf")
    )
    danger_ratio = (
        finite(row.get("selected_mae_danger_rate")) / max(finite(row.get("baseline_mae_danger")), 1e-12)
        if row.get("selected_mae_danger_rate") is not None and row.get("baseline_mae_danger") is not None
        else float("inf")
    )
    symbol = row.get("symbol")
    general_hit_floor = 0.01 if strict else 0.0
    general_quality_floor = 0.02 if strict else 0.0
    strict_symbol_failure = (
        symbol == "DOGEUSDT" and hit8_lift <= 0.03
    ) or (
        symbol == "BTCUSDT" and (hit8_lift <= 0.01 or quality_lift <= 0.02)
    )
    failed = (
        hit8_lift <= general_hit_floor
        or quality_lift <= general_quality_floor
        or net_quality <= 0.0
        or p90_ratio > 1.30
        or danger_ratio > 1.40
        or strict_symbol_failure
    )
    if failed:
        return "REPAIR_LOCKBOX_FAILED"
    confirmed = (
        not count_low
        and p90_ratio <= (1.10 if strict else 1.15)
        and danger_ratio <= (1.10 if strict else 1.20)
    )
    return "REPAIR_LOCKBOX_CONFIRMED" if confirmed else "REPAIR_LOCKBOX_WEAK"


def repair_lockbox_reason(row: dict[str, Any]) -> str:
    status = row["repair_lockbox_status"]
    if status == "REPAIR_LOCKBOX_CONFIRMED":
        return "frozen_repair_improves_hit8_quality_after_cost_with_controlled_risk"
    if status == "REPAIR_LOCKBOX_INSUFFICIENT_DATA":
        return "insufficient_lockbox_samples_or_model_diversity"
    if row.get("symbol") == "DOGEUSDT" and finite(row.get("selected_hit8_lift")) <= 0.03:
        return "doge_strict_hit8_confirmation_not_met"
    if row.get("symbol") == "BTCUSDT" and (
        finite(row.get("selected_hit8_lift")) <= 0.01
        or finite(row.get("selected_quality_lift")) <= 0.02
    ):
        return "btc_fragile_candidate_strict_confirmation_not_met"
    if status == "REPAIR_LOCKBOX_FAILED":
        return "frozen_repair_edge_or_risk_control_not_confirmed"
    return "frozen_repair_partially_improves_but_not_confirmed"


def recommended_next_step(status: str) -> str:
    if status == "REPAIR_LOCKBOX_CONFIRMED":
        return "eligible_for_metadata_only_shadow_artifact_after_review"
    if status == "REPAIR_LOCKBOX_WEAK":
        return "continue_research_no_shadow_artifact"
    if status == "REPAIR_LOCKBOX_FAILED":
        return "disable_repair_candidate_for_now"
    return "collect_more_data_before_decision"


def _fit_on_fold(dataset: dict[str, Any], config: dict[str, Any], fold: dict[str, Any], *, fast: bool) -> dict[str, Any]:
    x = np.asarray(dataset["X"], dtype=np.float32)
    arrays = _side_arrays(dataset, SIDE.lower(), int(config["horizon_candles"]))
    arrays["mfe"] = np.asarray(dataset[f"short_mfe_{config['horizon_candles']}"], dtype=np.float32)
    split = {name: np.asarray(fold[name], dtype=np.int64) for name in ("train", "validation", "test")}
    if len(np.unique(arrays["hit8"][split["train"]])) < 2 or len(np.unique(arrays["danger"][split["train"]])) < 2:
        return {"model_status": "insufficient_class_diversity", "split": split}
    max_iter = 60 if fast else 140
    predictions: dict[str, np.ndarray] = {}
    for name, target, constructor in (
        ("hit8", arrays["hit8"], _classifier),
        ("quality", arrays["quality"], _regressor),
        ("danger", arrays["danger"], _classifier),
    ):
        seed = _model_seed(
            config["symbol"],
            SIDE.lower(),
            int(config["lookback_days"]),
            f"repair_lockbox:{config['repair_mode']}:{config['feature_set']}:{config['horizon_candles']}:{name}",
        )
        model = constructor(max_iter, seed)
        model.fit(x[split["train"]], target[split["train"]])
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


def _finalize_row(row: dict[str, Any], config: dict[str, Any], trained: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    split = trained.get("split") or {}
    row.update({
        **config,
        "lockbox_mode": args.lockbox_mode,
        "model_status": trained.get("model_status"),
        "train_samples": int(len(split.get("train", []))),
        "validation_samples": int(len(split.get("validation", []))),
        "test_samples": int(len(split.get("test", []))),
        "selected_mae_danger_delta": (
            finite(row.get("selected_mae_danger_rate")) - finite(row.get("baseline_mae_danger"))
            if row.get("selected_mae_danger_rate") is not None and row.get("baseline_mae_danger") is not None
            else None
        ),
        "p90_mae_ratio": (
            finite(row.get("selected_p90_mae")) / max(finite(row.get("baseline_p90_mae")), 1e-12)
            if row.get("selected_p90_mae") is not None and row.get("baseline_p90_mae") is not None
            else None
        ),
    })
    row["repair_lockbox_status"] = classify_repair_lockbox_candidate(
        row,
        min_test_samples=int(args.min_test_samples),
        strict=bool(args.strict),
    )
    row["repair_lockbox_reason"] = repair_lockbox_reason(row)
    row["recommended_next_step"] = recommended_next_step(row["repair_lockbox_status"])
    return row


def _evaluate_frozen_config(config: dict[str, Any], dataset: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    folds = _folds_for_mode(len(np.asarray(dataset["X"])), args)
    rows: list[dict[str, Any]] = []
    for fold in folds:
        trained = _fit_on_fold(dataset, config, fold, fast=bool(args.fast))
        evaluated = evaluate_repair_mode(
            trained,
            config,
            repair_mode=config["repair_mode"],
            original_lockbox_status="REPAIRED_CONFIRMED_J0",
            failure_mode="frozen_repair_confirmation",
            fee_bps=float(args.fee_bps),
            slippage_bps=float(args.slippage_bps),
        )
        rows.append(_finalize_row(evaluated, config, trained, args))
    if not rows:
        row = {
            **config,
            "lockbox_mode": args.lockbox_mode,
            "model_status": "insufficient_data",
            "test_samples": 0,
        }
        row["repair_lockbox_status"] = "REPAIR_LOCKBOX_INSUFFICIENT_DATA"
        row["repair_lockbox_reason"] = repair_lockbox_reason(row)
        row["recommended_next_step"] = recommended_next_step(row["repair_lockbox_status"])
        return row, []
    return rows[-1], rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _num(value: Any) -> str:
    return "null" if value is None else f"{float(value):.4f}"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        f"# Aegis Repaired SHORT V3 Lockbox {report['created_at']}",
        "",
        "## Safety",
        "",
        "- Mode: `RESEARCH_ONLY`.",
        "- Repaired configurations and repair modes are frozen from Phase J.0; no alternatives are selected.",
        "- Frozen-mode evaluation is confirmation only; strict independence requires observations not used during Phase J.0 selection.",
        "- No model artifacts are saved; no `active/` models or active manifests are modified.",
        "- No live inference, YAML, thresholds or PM2 action is involved.",
        "",
        "## Results",
        "",
        "| Symbol | Repair Mode | Config | Status | Hit8 Lift | Quality Lift | Net Quality Lift | P90 Delta | Selected Fraction | Reason |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["results"]:
        lines.append(
            f"| {row['symbol']} | {row['repair_mode']} | {row['feature_set']} {row['lookback_days']}d h{row['horizon_candles']} | "
            f"{row['repair_lockbox_status']} | {_num(row.get('selected_hit8_lift'))} | "
            f"{_num(row.get('selected_quality_lift'))} | {_num(row.get('selected_net_quality_lift_after_cost'))} | "
            f"{_num(row.get('selected_p90_mae_delta'))} | {_num(row.get('selected_fraction'))} | "
            f"{row['repair_lockbox_reason']} |"
        )
    for key, title in (
        ("confirmed", "Confirmed Repairs"),
        ("weak", "Weak Repairs"),
        ("failed", "Failed Repairs"),
        ("insufficient_data", "Insufficient Data"),
    ):
        lines.extend(["", f"## {title}", ""])
        rows = report[key]
        if not rows:
            lines.append("- None.")
        for row in rows:
            lines.append(f"- `{row['symbol']}`: `{row.get('repair_mode', '-')}` -> `{row['repair_lockbox_status']}`.")
    lines.extend(["", "## Decision", "", f"- Recommendation: `{report['next_recommendation']}`.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    configs = load_frozen_repaired_configs(args.configs_json, args.symbols)
    context_markets: dict[str, Any] = {}
    needs_context = any(config["feature_set"] in {"operable_v3", "combined_v3"} for config in configs)
    if needs_context and not args.disable_cross_context:
        context_markets = {
            symbol: load_signal_market(DEFAULT_TURBO_CONFIG.config_path, symbol_override=symbol)
            for symbol in ("BTCUSDT", "ETHUSDT")
        }
    results: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for config in configs:
        try:
            market = load_signal_market(DEFAULT_TURBO_CONFIG.config_path, symbol_override=config["symbol"])
            base = build_recent_dataset(config["symbol"], config["lookback_days"], save=False, market=market)["dataset"]
            dataset = apply_feature_set(base, market, config["feature_set"], context_markets=context_markets)
            result, folds = _evaluate_frozen_config(config, dataset, args)
            results.append(result)
            fold_rows.extend(folds)
        except Exception as exc:
            error = {**config, "error": repr(exc)}
            errors.append(error)
            results.append({
                **config,
                "lockbox_mode": args.lockbox_mode,
                "model_status": "evaluation_error",
                "test_samples": 0,
                "config_error": repr(exc),
                "repair_lockbox_status": "REPAIR_LOCKBOX_INSUFFICIENT_DATA",
                "repair_lockbox_reason": "configuration_evaluation_error",
                "recommended_next_step": "collect_more_data_before_decision",
            })
    confirmed = [row for row in results if row["repair_lockbox_status"] == "REPAIR_LOCKBOX_CONFIRMED"]
    weak = [row for row in results if row["repair_lockbox_status"] == "REPAIR_LOCKBOX_WEAK"]
    failed = [row for row in results if row["repair_lockbox_status"] == "REPAIR_LOCKBOX_FAILED"]
    insufficient = [row for row in results if row["repair_lockbox_status"] == "REPAIR_LOCKBOX_INSUFFICIENT_DATA"]
    token = utc_stamp()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "md": out_dir / f"aegis_repaired_short_v3_lockbox_{token}.md",
        "json": out_dir / f"aegis_repaired_short_v3_lockbox_{token}.json",
        "summary_csv": out_dir / f"aegis_repaired_short_v3_lockbox_summary_{token}.csv",
        "confirmed_csv": out_dir / f"aegis_repaired_short_v3_lockbox_confirmed_{token}.csv",
        "weak_csv": out_dir / f"aegis_repaired_short_v3_lockbox_weak_{token}.csv",
        "failed_csv": out_dir / f"aegis_repaired_short_v3_lockbox_failed_{token}.csv",
    }
    report = {
        "schema_version": "aegis_repaired_short_v3_lockbox_v1",
        "created_at": utc_now().isoformat(),
        "mode": MODE,
        "side": SIDE,
        "lockbox_mode": args.lockbox_mode,
        "lockbox_test_ratio": float(args.lockbox_test_ratio),
        "strict": bool(args.strict),
        "frozen_repaired_configs": configs,
        "evaluated_config_count": len(configs),
        "selection_policy": "one_frozen_repair_mode_per_symbol_no_alternative_search",
        "lockbox_independence_note": "strict_independence_requires_observations_not_used_during_phase_j0_selection",
        "disable_cross_context": bool(args.disable_cross_context),
        "results": results,
        "fold_results": fold_rows,
        "confirmed": confirmed,
        "weak": weak,
        "failed": failed,
        "insufficient_data": insufficient,
        "status_counts": dict(Counter(row["repair_lockbox_status"] for row in results)),
        "errors": errors,
        "save_models": False,
        "shadow_models_generated": False,
        "active_manifest_touched": False,
        "live_inference_changed": False,
        "next_recommendation": (
            "add_repair_lockbox_confirmed_to_metadata_only_shadow_candidate_list_after_review"
            if confirmed else "no_repaired_shadow_candidates_continue_research"
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
    parser = argparse.ArgumentParser(description="Research-only lockbox confirmation for frozen repaired SHORT V3 candidates.")
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
        "evaluated_config_count": report["evaluated_config_count"],
        "status_counts": report["status_counts"],
        "confirmed": [row["symbol"] for row in report["confirmed"]],
        "weak": [row["symbol"] for row in report["weak"]],
        "failed": [row["symbol"] for row in report["failed"]],
        "insufficient_data": [row["symbol"] for row in report["insufficient_data"]],
        "errors": report["errors"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
