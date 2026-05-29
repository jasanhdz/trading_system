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
from aegis_alpha.tools.train_short_alpha_family_l2_research import (  # noqa: E402
    _fit_fold_predictions,
    _percentile_ranks,
    _top_fraction_mask,
    compute_alpha_target_arrays,
    select_alpha_family_features,
    selection_mask as l2_selection_mask,
)
from aegis_alpha.turbo.config import DEFAULT_TURBO_CONFIG  # noqa: E402
from aegis_alpha.turbo.operable_feature_builder_v3 import apply_feature_set  # noqa: E402
from aegis_alpha.turbo.recent_dataset import build_recent_dataset  # noqa: E402
from aegis_alpha.turbo.snapshot_utils import normalize_turbo_symbol  # noqa: E402
from aegis_alpha.turbo.walk_forward_operable_v2 import temporal_folds  # noqa: E402


MODE = "RESEARCH_ONLY"
SIDE = "SHORT"
SCHEMA_VERSION = "aegis_sol_link_final_repair_m_v1"
BASE_ENTRY_MODES = (
    "hit_primary",
    "quality_primary",
    "quality_primary_danger_filtered",
    "top_bucket_consensus",
    "top_bucket_consensus_danger_filtered",
)
LINK_MODES = (
    "hit_primary",
    "quality_primary",
    "quality_primary_danger_filtered",
    "top_bucket_only",
    "trend_confirmed_quality",
    "avoid_only_candidate",
)
CSV_COLUMNS = (
    "symbol", "side", "alpha_family", "feature_set", "feature_mode", "lookback_days",
    "target_name", "horizon_candles", "decision_mode", "lockbox_mode", "m_status",
    "m_reason", "recommended_next_step", "model_status", "train_samples",
    "validation_samples", "test_samples", "valid_fold_count", "feature_count",
    "missing_family_features", "missing_trend_confirmation_features",
    "baseline_target_hit_rate", "baseline_trade_quality", "baseline_mae_danger",
    "baseline_p90_mae", "selected_fraction", "selected_count",
    "selected_target_hit_rate", "selected_target_lift", "selected_quality_mean",
    "selected_quality_lift", "selected_net_quality_lift", "selected_mae_danger_rate",
    "selected_mae_danger_delta", "selected_p90_mae", "selected_p90_mae_delta",
    "selected_avg_mfe", "selected_avg_mae", "p90_mae_ratio", "mae_danger_ratio",
    "target_auc", "target_average_precision", "quality_corr", "danger_auc",
    "danger_filter_usefulness", "avoid_selected_count", "avoid_selected_fraction",
    "avoid_quality_delta_vs_baseline", "avoid_mae_danger_delta",
    "avoid_p90_mae_delta", "avoid_usefulness_score",
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


def _mean(values: np.ndarray) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if len(array) else None


def _quantile(values: np.ndarray, q: float) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.quantile(array, q)) if len(array) else None


def default_m_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for target_name in ("hit5_before_minus3", "hit6_before_minus4"):
        for horizon in (12, 24):
            configs.append({
                "symbol": "SOLUSDT",
                "side": SIDE,
                "alpha_family": "momentum_burst_lower_target",
                "feature_set": "combined_v3",
                "feature_mode": "selected_family",
                "lookback_days": 30,
                "target_name": target_name,
                "horizon_candles": horizon,
                "decision_modes": list(BASE_ENTRY_MODES),
            })
    for target_name in ("hit3_before_minus2", "hit5_before_minus3"):
        for horizon in (12, 24):
            configs.append({
                "symbol": "LINKUSDT",
                "side": SIDE,
                "alpha_family": "slow_trend_short",
                "feature_set": "combined_v3",
                "feature_mode": "selected_family",
                "lookback_days": 30,
                "target_name": target_name,
                "horizon_candles": horizon,
                "decision_modes": list(LINK_MODES),
            })
    return configs


def parse_symbols(raw: str | None) -> set[str]:
    if not raw:
        return {"SOLUSDT", "LINKUSDT"}
    requested = {normalize_turbo_symbol(value.strip()) for value in raw.split(",") if value.strip()}
    unsupported = requested - {"SOLUSDT", "LINKUSDT"}
    if unsupported:
        raise ValueError(f"Fase M only supports SOLUSDT/LINKUSDT: {sorted(unsupported)}")
    return requested


def configs_for_symbols(symbols: set[str], feature_mode: str) -> list[dict[str, Any]]:
    if feature_mode not in {"selected_family", "combined_v3_all"}:
        raise ValueError(f"unsupported feature_mode: {feature_mode}")
    configs = []
    for config in default_m_configs():
        if config["symbol"] in symbols:
            updated = dict(config)
            updated["feature_mode"] = feature_mode
            configs.append(updated)
    return configs


def _required_selected_count(test_samples: int) -> int:
    return min(30, max(1, math.ceil(float(test_samples) * 0.05)))


def _risk_ratios(row: dict[str, Any]) -> tuple[float, float]:
    baseline_p90 = finite(row.get("baseline_p90_mae"), 0.0) or 0.0
    selected_p90 = finite(row.get("selected_p90_mae"), float("inf"))
    baseline_danger = finite(row.get("baseline_mae_danger"), 0.0) or 0.0
    selected_danger = finite(row.get("selected_mae_danger_rate"), float("inf"))
    p90_ratio = float(selected_p90) / max(baseline_p90, 1e-12) if selected_p90 is not None else float("inf")
    danger_ratio = float(selected_danger) / max(baseline_danger, 1e-12) if selected_danger is not None else float("inf")
    return p90_ratio, danger_ratio


def classify_m_final_candidate(row: dict[str, Any]) -> str:
    if (
        row.get("model_status") != "trained"
        or int(row.get("test_samples") or 0) <= 0
        or int(row.get("feature_count") or 0) <= 0
    ):
        return "M_FINAL_INSUFFICIENT_DATA"
    if row.get("decision_mode") == "avoid_only_candidate":
        if (
            (finite(row.get("avoid_quality_delta_vs_baseline"), 0.0) or 0.0) < 0.0
            and (finite(row.get("avoid_mae_danger_delta"), 0.0) or 0.0) > 0.0
            and (finite(row.get("avoid_p90_mae_delta"), 0.0) or 0.0) > 0.0
        ):
            return "M_FINAL_AVOID_ONLY"
        return "M_FINAL_FAILED"
    selected_count = int(row.get("selected_count") or 0)
    if selected_count <= 0:
        return "M_FINAL_FAILED"
    if selected_count < _required_selected_count(int(row.get("test_samples") or 0)):
        return "M_FINAL_WEAK"
    target_lift = finite(row.get("selected_target_lift"), -1.0) or -1.0
    quality_lift = finite(row.get("selected_quality_lift"), -1.0) or -1.0
    net_lift = finite(row.get("selected_net_quality_lift"), -1.0) or -1.0
    target_auc = finite(row.get("target_auc"), 0.0) or 0.0
    quality_corr = finite(row.get("quality_corr"), 0.0) or 0.0
    p90_ratio, danger_ratio = _risk_ratios(row)
    if target_lift <= 0.0 or quality_lift <= 0.0 or net_lift <= 0.0:
        return "M_FINAL_FAILED"
    if p90_ratio > 1.30 or danger_ratio > 1.40:
        return "M_FINAL_FAILED"
    symbol = str(row.get("symbol"))
    symbol_gate = (
        symbol == "SOLUSDT" and target_lift > 0.04 and net_lift > 0.03
    ) or (
        symbol == "LINKUSDT" and quality_lift > 0.05 and net_lift > 0.04 and p90_ratio <= 1.0
    )
    confirmed = (
        symbol_gate
        and p90_ratio <= 1.15
        and danger_ratio <= 1.20
        and (target_auc > 0.53 or quality_corr > 0.03)
    )
    return "M_FINAL_CONFIRMED" if confirmed else "M_FINAL_WEAK"


def m_reason(row: dict[str, Any]) -> str:
    status = row["m_status"]
    if status == "M_FINAL_CONFIRMED":
        return "final_repair_entry_edge_confirmed_with_controlled_risk"
    if status == "M_FINAL_AVOID_ONLY":
        return "bad_zone_filter_candidate_detects_lower_quality_higher_risk_bucket"
    if status == "M_FINAL_INSUFFICIENT_DATA":
        return "insufficient_samples_features_or_model_diversity"
    if status == "M_FINAL_FAILED":
        return "final_repair_edge_or_risk_gate_failed"
    return "partial_edge_but_not_enough_for_final_repair_confirmation"


def recommended_next_step(status: str) -> str:
    if status == "M_FINAL_CONFIRMED":
        return "eligible_for_metadata_only_shadow_candidate_as_final_repair_after_review"
    if status == "M_FINAL_AVOID_ONLY":
        return "research_avoid_or_reducer_shadow_not_entry_artifact"
    if status == "M_FINAL_WEAK":
        return "continue_research_no_shadow_entry_artifact"
    if status == "M_FINAL_FAILED":
        return "disable_final_repair_hypothesis_for_now"
    return "collect_more_data_before_decision"


def _feature_column(dataset: dict[str, Any], name: str, test: np.ndarray) -> np.ndarray | None:
    names = np.asarray(dataset.get("feature_names", [])).astype(str)
    matches = np.flatnonzero(names == name)
    if len(matches) == 0:
        return None
    return np.asarray(dataset["X"], dtype=np.float32)[test, int(matches[0])]


def trend_confirmed_quality_mask(dataset: dict[str, Any], test: np.ndarray, quality_pred: np.ndarray) -> tuple[np.ndarray, list[str]]:
    required = ("local_trend_down_score", "btc_eth_long_contradiction", "short_room_to_fall_12", "short_room_to_fall_24")
    values = {name: _feature_column(dataset, name, test) for name in required}
    missing = [name for name, column in values.items() if column is None]
    chop = _feature_column(dataset, "local_chop_score", test)
    if missing:
        return _top_fraction_mask(quality_pred), missing
    trend = values["local_trend_down_score"]
    contradiction = values["btc_eth_long_contradiction"]
    room = np.maximum(values["short_room_to_fall_12"], values["short_room_to_fall_24"])
    eligible = (
        trend >= np.quantile(trend, 0.50)
    ) & (
        contradiction <= np.quantile(contradiction, 0.60)
    ) & (
        room >= np.quantile(room, 0.50)
    )
    if chop is not None:
        eligible &= chop <= np.quantile(chop, 0.75)
    mask = _top_fraction_mask(quality_pred, eligible=eligible)
    if not np.any(mask):
        return _top_fraction_mask(quality_pred), ["trend_filter_no_eligible_rows"]
    return mask, []


def decision_mask(
    mode: str,
    trained: dict[str, Any],
    dataset: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    hit = np.asarray(trained["hit_prob"], dtype=np.float64)
    quality = np.asarray(trained["quality_pred"], dtype=np.float64)
    danger = np.asarray(trained["danger_prob"], dtype=np.float64)
    if mode in BASE_ENTRY_MODES:
        return l2_selection_mask(mode, hit, quality, danger), []
    if mode == "top_bucket_only":
        return _top_fraction_mask(quality), []
    if mode == "trend_confirmed_quality":
        return trend_confirmed_quality_mask(dataset, np.asarray(trained["test"], dtype=np.int64), quality)
    if mode == "avoid_only_candidate":
        avoid_score = _percentile_ranks(danger) + _percentile_ranks(-quality) + _percentile_ranks(-hit)
        return _top_fraction_mask(avoid_score), []
    raise ValueError(f"unsupported final repair decision mode: {mode}")


def _selection_stats(
    trained: dict[str, Any],
    targets: dict[str, np.ndarray],
    mask: np.ndarray,
    cost_proxy: float,
) -> dict[str, Any]:
    test = np.asarray(trained["test"], dtype=np.int64)
    hit = targets["hit"][test]
    quality = targets["quality"][test]
    danger = targets["danger"][test]
    mae = targets["mae"][test]
    mfe = targets["mfe"][test]
    selected_count = int(mask.sum())
    baseline_hit = _mean(hit)
    baseline_quality = _mean(quality)
    baseline_danger = _mean(danger)
    baseline_p90 = _quantile(mae, 0.90)
    selected_hit = _mean(hit[mask]) if selected_count else None
    selected_quality = _mean(quality[mask]) if selected_count else None
    selected_danger = _mean(danger[mask]) if selected_count else None
    selected_p90 = _quantile(mae[mask], 0.90) if selected_count else None
    return {
        "baseline_target_hit_rate": baseline_hit,
        "baseline_trade_quality": baseline_quality,
        "baseline_mae_danger": baseline_danger,
        "baseline_p90_mae": baseline_p90,
        "selected_fraction": selected_count / max(1, len(test)),
        "selected_count": selected_count,
        "selected_target_hit_rate": selected_hit,
        "selected_target_lift": None if selected_hit is None or baseline_hit is None else selected_hit - baseline_hit,
        "selected_quality_mean": selected_quality,
        "selected_quality_lift": None if selected_quality is None or baseline_quality is None else selected_quality - baseline_quality,
        "selected_net_quality_lift": None if selected_quality is None or baseline_quality is None else selected_quality - baseline_quality - cost_proxy,
        "selected_mae_danger_rate": selected_danger,
        "selected_mae_danger_delta": None if selected_danger is None or baseline_danger is None else selected_danger - baseline_danger,
        "selected_p90_mae": selected_p90,
        "selected_p90_mae_delta": None if selected_p90 is None or baseline_p90 is None else selected_p90 - baseline_p90,
        "selected_avg_mfe": _mean(mfe[mask]) if selected_count else None,
        "selected_avg_mae": _mean(mae[mask]) if selected_count else None,
    }


def evaluate_mode(
    trained: dict[str, Any],
    targets: dict[str, np.ndarray],
    dataset: dict[str, Any],
    config: dict[str, Any],
    mode: str,
    cost_proxy: float,
) -> dict[str, Any]:
    base = {
        **config,
        "decision_mode": mode,
        "model_status": trained.get("model_status"),
        "train_samples": trained.get("train_samples"),
        "validation_samples": trained.get("validation_samples"),
        "test_samples": trained.get("test_samples"),
        "target_auc": trained.get("target_auc"),
        "target_average_precision": trained.get("target_average_precision"),
        "quality_corr": trained.get("quality_corr"),
        "danger_auc": trained.get("danger_auc"),
        "danger_filter_usefulness": trained.get("danger_filter_usefulness"),
        "feature_count": int(np.asarray(dataset["X"]).shape[1]),
        "missing_family_features": ",".join(dataset.get("missing_family_features", [])),
    }
    if trained.get("model_status") != "trained":
        base["m_status"] = "M_FINAL_INSUFFICIENT_DATA"
        base["m_reason"] = m_reason(base)
        base["recommended_next_step"] = recommended_next_step(base["m_status"])
        return base
    mask, missing_trend = decision_mask(mode, trained, dataset)
    row = {**base, **_selection_stats(trained, targets, mask, cost_proxy)}
    row["missing_trend_confirmation_features"] = ",".join(missing_trend)
    if mode == "avoid_only_candidate":
        row["avoid_selected_count"] = row["selected_count"]
        row["avoid_selected_fraction"] = row["selected_fraction"]
        row["avoid_quality_delta_vs_baseline"] = row["selected_quality_lift"]
        row["avoid_mae_danger_delta"] = row["selected_mae_danger_delta"]
        row["avoid_p90_mae_delta"] = row["selected_p90_mae_delta"]
        row["avoid_usefulness_score"] = (
            -float(row["avoid_quality_delta_vs_baseline"] or 0.0)
            + float(row["avoid_mae_danger_delta"] or 0.0)
            + float(row["avoid_p90_mae_delta"] or 0.0)
        )
    p90_ratio, danger_ratio = _risk_ratios(row)
    row["p90_mae_ratio"] = p90_ratio
    row["mae_danger_ratio"] = danger_ratio
    row["m_status"] = classify_m_final_candidate(row)
    row["m_reason"] = m_reason(row)
    row["recommended_next_step"] = recommended_next_step(row["m_status"])
    return row


def _folds_for_mode(sample_count: int, args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.lockbox_mode == "rolling-forward":
        return temporal_folds(
            sample_count,
            fold_count=4,
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


def evaluate_config(
    config: dict[str, Any],
    dataset: dict[str, Any],
    targets: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    splits = _folds_for_mode(len(np.asarray(dataset["X"])), args)
    if not splits:
        for mode in config["decision_modes"]:
            row = {
                **config,
                "decision_mode": mode,
                "model_status": "insufficient_split_samples",
                "test_samples": 0,
                "feature_count": int(np.asarray(dataset["X"]).shape[1]),
                "m_status": "M_FINAL_INSUFFICIENT_DATA",
            }
            row["m_reason"] = m_reason(row)
            row["recommended_next_step"] = recommended_next_step(row["m_status"])
            rows.append(row)
        return rows, []
    for fold_index, split in enumerate(splits, start=1):
        trained = _fit_fold_predictions(
            dataset,
            targets,
            split,
            symbol=config["symbol"],
            lookback_days=int(config["lookback_days"]),
            target_name=config["target_name"],
            max_iter=60 if args.fast else 120,
        )
        for mode in config["decision_modes"]:
            row = evaluate_mode(
                trained,
                targets,
                dataset,
                config,
                mode,
                (float(args.fee_bps) + float(args.slippage_bps)) / 10000.0,
            )
            row["fold"] = fold_index
            rows.append(row)
            folds.append(row)
    return rows, folds


def select_best_m_by_symbol(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    priority = {
        "M_FINAL_CONFIRMED": 5,
        "M_FINAL_AVOID_ONLY": 4,
        "M_FINAL_WEAK": 3,
        "M_FINAL_FAILED": 2,
        "M_FINAL_INSUFFICIENT_DATA": 1,
    }
    target_preference = {
        "LINKUSDT": {"hit3_before_minus2": 2, "hit5_before_minus3": 1},
        "SOLUSDT": {"hit5_before_minus3": 2, "hit6_before_minus4": 1},
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["symbol"]), []).append(row)
    best: list[dict[str, Any]] = []
    for symbol, local in grouped.items():
        selected = max(
            local,
            key=lambda row: (
                priority.get(str(row.get("m_status")), 0),
                finite(row.get("selected_net_quality_lift"), -999.0) if row.get("m_status") != "M_FINAL_AVOID_ONLY" else finite(row.get("avoid_usefulness_score"), -999.0),
                finite(row.get("selected_target_lift"), -999.0),
                -finite(row.get("selected_p90_mae_delta"), 999.0),
                finite(row.get("target_auc"), -999.0),
                target_preference.get(symbol, {}).get(str(row.get("target_name")), 0),
            ),
        )
        best.append(dict(selected))
    return sorted(best, key=lambda row: str(row["symbol"]))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _num(value: Any) -> str:
    number = finite(value)
    return "null" if number is None else f"{number:.4f}"


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        f"# Aegis SOL/LINK Final SHORT Repair M {report['created_at']}",
        "",
        "## Safety",
        "",
        "- Mode: `RESEARCH_ONLY`.",
        "- No shadow artifacts, active models, active manifests or live inference are changed.",
        "- Only SOLUSDT/LINKUSDT final repair configurations are evaluated.",
        "",
        "## Best By Symbol",
        "",
        "| Symbol | Family | Target | Horizon | Mode | Status | Target Lift | Quality Lift | Net Lift | P90 Delta | Selected Fraction | Reason |",
        "|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["best_by_symbol"]:
        lines.append(
            f"| {row['symbol']} | {row['alpha_family']} | {row['target_name']} | {row['horizon_candles']} | "
            f"{row['decision_mode']} | {row['m_status']} | {_num(row.get('selected_target_lift'))} | "
            f"{_num(row.get('selected_quality_lift'))} | {_num(row.get('selected_net_quality_lift'))} | "
            f"{_num(row.get('selected_p90_mae_delta'))} | {_num(row.get('selected_fraction'))} | {row['m_reason']} |"
        )
    for key, title in (
        ("confirmed", "Confirmed Entry Candidates"),
        ("avoid_only", "Avoid-Only Candidates"),
        ("weak", "Weak Candidates"),
        ("failed", "Failed Candidates"),
    ):
        lines.extend(["", f"## {title}", ""])
        values = report[key]
        if not values:
            lines.append("- None.")
        for row in values:
            lines.append(f"- `{row['symbol']}`: `{row['m_status']}` via `{row['decision_mode']}`.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    symbols = parse_symbols(args.symbols)
    configs = configs_for_symbols(symbols, args.feature_mode)
    if not args.include_avoid_only:
        for config in configs:
            config["decision_modes"] = [mode for mode in config["decision_modes"] if mode != "avoid_only_candidate"]
    context_markets: dict[str, Any] = {}
    if not args.disable_cross_context:
        context_markets = {
            symbol: load_signal_market(DEFAULT_TURBO_CONFIG.config_path, symbol_override=symbol)
            for symbol in ("BTCUSDT", "ETHUSDT")
        }
    all_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    by_symbol_data: dict[tuple[str, int], tuple[Any, dict[str, Any]]] = {}
    for config in configs:
        try:
            cache_key = (config["symbol"], int(config["lookback_days"]))
            if cache_key not in by_symbol_data:
                market = load_signal_market(DEFAULT_TURBO_CONFIG.config_path, symbol_override=config["symbol"])
                base = build_recent_dataset(config["symbol"], config["lookback_days"], save=False, market=market)["dataset"]
                combined = apply_feature_set(base, market, config["feature_set"], context_markets=context_markets)
                by_symbol_data[cache_key] = (market, combined)
            market, combined = by_symbol_data[cache_key]
            dataset = select_alpha_family_features(combined, config["alpha_family"], config["feature_mode"])
            targets = compute_alpha_target_arrays(
                market,
                np.asarray(combined["step"], dtype=np.int64),
                side=SIDE,
                target_name=config["target_name"],
                horizon=config["horizon_candles"],
                cost_proxy=(float(args.fee_bps) + float(args.slippage_bps)) / 10000.0,
            )
            rows, folds = evaluate_config(config, dataset, targets, args)
            all_rows.extend(rows)
            fold_rows.extend(folds)
        except Exception as exc:
            errors.append({**config, "error": repr(exc)})
            for mode in config["decision_modes"]:
                all_rows.append({
                    **config,
                    "decision_mode": mode,
                    "model_status": "evaluation_error",
                    "test_samples": 0,
                    "feature_count": 0,
                    "config_error": repr(exc),
                    "m_status": "M_FINAL_INSUFFICIENT_DATA",
                    "m_reason": "configuration_evaluation_error",
                    "recommended_next_step": "collect_more_data_before_decision",
                })
    best = select_best_m_by_symbol(all_rows)
    confirmed = [row for row in best if row["m_status"] == "M_FINAL_CONFIRMED"]
    avoid_only = [row for row in best if row["m_status"] == "M_FINAL_AVOID_ONLY"]
    weak = [row for row in best if row["m_status"] == "M_FINAL_WEAK"]
    failed = [row for row in best if row["m_status"] in {"M_FINAL_FAILED", "M_FINAL_INSUFFICIENT_DATA"}]
    stamp = utc_stamp()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "md": out_dir / f"aegis_sol_link_final_repair_m_{stamp}.md",
        "json": out_dir / f"aegis_sol_link_final_repair_m_{stamp}.json",
        "all_configs_csv": out_dir / f"aegis_sol_link_final_repair_m_all_configs_{stamp}.csv",
        "best_by_symbol_csv": out_dir / f"aegis_sol_link_final_repair_m_best_by_symbol_{stamp}.csv",
        "confirmed_csv": out_dir / f"aegis_sol_link_final_repair_m_confirmed_{stamp}.csv",
        "weak_csv": out_dir / f"aegis_sol_link_final_repair_m_weak_{stamp}.csv",
        "failed_csv": out_dir / f"aegis_sol_link_final_repair_m_failed_{stamp}.csv",
        "avoid_only_csv": out_dir / f"aegis_sol_link_final_repair_m_avoid_only_{stamp}.csv",
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now().isoformat(),
        "mode": MODE,
        "symbols": sorted(symbols),
        "lockbox_mode": args.lockbox_mode,
        "feature_mode": args.feature_mode,
        "configs": configs,
        "all_config_rows": all_rows,
        "fold_rows": fold_rows,
        "best_by_symbol": best,
        "confirmed": confirmed,
        "avoid_only": avoid_only,
        "weak": weak,
        "failed": failed,
        "status_counts": dict(Counter(row["m_status"] for row in best)),
        "errors": errors,
        "models_trained_in_memory_only": True,
        "model_artifacts_written": False,
        "shadow_models_generated": False,
        "active_manifest_touched": False,
        "live_inference_changed": False,
        "paths": {key: str(path) for key, path in paths.items()},
    }
    paths["json"].write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(paths["md"], report)
    _write_csv(paths["all_configs_csv"], all_rows)
    _write_csv(paths["best_by_symbol_csv"], best)
    _write_csv(paths["confirmed_csv"], confirmed)
    _write_csv(paths["weak_csv"], weak)
    _write_csv(paths["failed_csv"], failed)
    _write_csv(paths["avoid_only_csv"], avoid_only)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Research-only final repair for SOLUSDT/LINKUSDT SHORT.")
    parser.add_argument("--symbols", default="SOLUSDT,LINKUSDT")
    parser.add_argument("--out-dir", default="/home/jasan/Develop")
    parser.add_argument("--lockbox-mode", choices=("last-block", "rolling-forward", "recent-only"), default="last-block")
    parser.add_argument("--lockbox-test-ratio", type=float, default=0.20)
    parser.add_argument("--min-train-samples", type=int, default=1000)
    parser.add_argument("--min-test-samples", type=int, default=300)
    parser.add_argument("--feature-mode", choices=("selected_family", "combined_v3_all"), default="selected_family")
    parser.add_argument("--fee-bps", type=float, default=8.0)
    parser.add_argument("--slippage-bps", type=float, default=3.0)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--disable-cross-context", action="store_true")
    parser.add_argument("--include-avoid-only", action="store_true")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({
        "paths": report["paths"],
        "best_by_symbol": [
            {
                "symbol": row["symbol"],
                "target": row["target_name"],
                "horizon": row["horizon_candles"],
                "mode": row["decision_mode"],
                "status": row["m_status"],
            }
            for row in report["best_by_symbol"]
        ],
        "errors": report["errors"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
