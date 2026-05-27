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
from aegis_alpha.turbo.config import DEFAULT_TURBO_CONFIG  # noqa: E402
from aegis_alpha.turbo.operable_feature_builder_v3 import apply_feature_set  # noqa: E402
from aegis_alpha.turbo.recent_dataset import (  # noqa: E402
    build_recent_dataset,
    compute_path_outcome,
    compute_trade_quality,
)
from aegis_alpha.turbo.snapshot_utils import normalize_turbo_symbol  # noqa: E402


MODE = "RESEARCH_ONLY"
DEFAULT_FAILED_SYMBOLS = ("LINKUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT")
CONFIRMED_CONTROLS = ("AVAXUSDT", "SUIUSDT", "LTCUSDT", "ETHUSDT")
REPAIRED_CONTROLS = ("ADAUSDT", "DOGEUSDT", "BTCUSDT")
TURBO_SYMBOLS = (
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT",
    "ADAUSDT", "AVAXUSDT", "LINKUSDT", "SUIUSDT", "LTCUSDT",
)
ALPHA_FAMILIES = (
    "breakdown_continuation_short",
    "fake_breakdown_reversal",
    "mean_reversion_after_extension",
    "momentum_burst",
    "slow_trend_short",
    "avoid_only",
)
BUCKET_LABELS = ("LOW", "MEDIUM", "HIGH", "EXTREME")
ALT_HIT_RULES: dict[str, tuple[float, float]] = {
    "hit3_before_minus2": (0.003, 0.002),
    "hit5_before_minus3": (0.005, 0.003),
    "hit6_before_minus4": (0.006, 0.004),
    "hit8_before_minus5": (0.008, 0.005),
    "hit10_before_minus6": (0.010, 0.006),
    "hit10_before_minus8": (0.010, 0.008),
}
SUMMARY_COLUMNS = (
    "symbol", "side", "role", "sample_count", "primary_failure_reason", "best_alpha_family",
    "alternative_target_recommendation", "avoid_only_potential", "confidence",
    "validation_status", "recommended_next_research_phase", "best_horizon", "best_bucket_net_quality",
    "best_bucket_hit5", "best_bucket_hit8", "cross_symbol_context_available",
)
BUCKET_COLUMNS = (
    "symbol", "side", "role", "horizon_candles", "alpha_family", "bucket", "count",
    "hit3", "hit5", "hit6", "hit8", "hit10", "avg_trade_quality", "avg_mfe",
    "avg_mae", "p90_mae", "mfe_mae_ratio", "danger_rate",
    "net_quality_proxy_after_cost", "reversal_hit5", "reversal_hit8",
    "avoid_short_after_sweep_quality", "avoided_loss_if_filter_active",
    "avoided_mae_if_filter_active", "false_block_rate", "saved_pnl_proxy",
    "suggested_alpha_family",
)
TARGET_COLUMNS = (
    "symbol", "side", "role", "horizon_candles", "sample_count", "hit3", "hit5",
    "hit6", "hit8", "hit10_minus6", "hit10_minus8", "avg_trade_quality",
    "avg_mfe", "avg_mae", "p90_mae", "danger_rate", "time_to_green_avg",
    "time_to_minus3_avg", "time_to_minus5_avg", "reversal_hit5_after_sweep",
    "reversal_hit8_after_sweep",
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_stamp() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%SZ")


def finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def metric_mean(values: np.ndarray) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if len(array) else None


def metric_quantile(values: np.ndarray, quantile: float) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.quantile(array, quantile)) if len(array) else None


def parse_symbols(raw: str | None, *, include_confirmed: bool, include_repaired: bool, include_longs: bool) -> list[str]:
    values = list(DEFAULT_FAILED_SYMBOLS if not raw else tuple(item.strip() for item in raw.split(",") if item.strip()))
    if include_confirmed:
        values.extend(CONFIRMED_CONTROLS)
    if include_repaired:
        values.extend(REPAIRED_CONTROLS)
    if include_longs and not raw:
        values.extend(TURBO_SYMBOLS)
    return list(dict.fromkeys(normalize_turbo_symbol(value) for value in values))


def parse_sides(side: str, include_longs: bool) -> list[str]:
    requested = ["SHORT", "LONG"] if side.upper() == "BOTH" else [side.upper()]
    if include_longs and "LONG" not in requested:
        requested.append("LONG")
    if any(item not in {"LONG", "SHORT"} for item in requested):
        raise ValueError(f"unsupported side scope: {side}")
    return requested


def role_for_symbol(symbol: str) -> str:
    if symbol in CONFIRMED_CONTROLS:
        return "confirmed_control"
    if symbol in REPAIRED_CONTROLS:
        return "repaired_control"
    return "failed_candidate"


def bucketize_score(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if not len(values):
        return np.asarray([], dtype="U8")
    finite_values = values[np.isfinite(values)]
    if not len(finite_values) or float(np.max(finite_values) - np.min(finite_values)) <= 1e-12:
        return np.full(len(values), "LOW", dtype="U8")
    q25, q50, q75 = np.quantile(finite_values, (0.25, 0.50, 0.75))
    buckets = np.full(len(values), "LOW", dtype="U8")
    buckets[values > q25] = "MEDIUM"
    buckets[values > q50] = "HIGH"
    buckets[values > q75] = "EXTREME"
    return buckets


def _feature_map(feature_dataset: dict[str, Any]) -> dict[str, np.ndarray]:
    names = [str(name) for name in np.asarray(feature_dataset["feature_names"]).tolist()]
    x = np.asarray(feature_dataset["X"], dtype=np.float64)
    return {name: x[:, idx] for idx, name in enumerate(names)}


def _get(features: dict[str, np.ndarray], name: str, count: int) -> np.ndarray:
    return np.asarray(features.get(name, np.zeros(count)), dtype=np.float64)


def _clip_unit(value: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(value, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def compute_family_scores(feature_dataset: dict[str, Any], side: str) -> dict[str, np.ndarray]:
    features = _feature_map(feature_dataset)
    count = len(np.asarray(feature_dataset["X"]))
    short = side.upper() == "SHORT"
    if short:
        breakdown = _clip_unit(
            _get(features, "short_breakdown_strength_12", count) / 0.005 * 0.25
            + _get(features, "short_breakdown_strength_24", count) / 0.008 * 0.15
            + _get(features, "short_breakdown_followthrough_3", count) * 0.20
            + _get(features, "short_breakdown_volume_confirmed", count) * 0.20
            + _get(features, "short_breakdown_body_confirmed", count) * 0.20
        )
        fake = _clip_unit(
            _get(features, "short_failed_breakdown_risk_12", count) * 0.25
            + _get(features, "short_lower_wick_sweep_risk", count) * 0.20
            + _get(features, "short_close_back_inside_range", count) * 0.20
            + _get(features, "short_reversal_after_low_sweep", count) * 0.20
            + _get(features, "short_absorption_risk", count) * 0.15
        )
        extension = _clip_unit(
            _get(features, "short_extension_below_ema21", count) / 0.010 * 0.20
            + _get(features, "short_extension_below_ema200", count) / 0.020 * 0.15
            + _get(features, "local_exhaustion_score", count) * 0.25
            + _get(features, "short_volume_climax_risk", count) * 0.20
            + _get(features, "short_volatility_exhaustion_risk", count) * 0.20
        )
        momentum = _clip_unit(
            _get(features, "local_momentum_down_score", count) * 0.25
            + _get(features, "local_breakdown_score", count) * 0.25
            + _get(features, "volume_ratio_12", count) / 2.0 * 0.15
            + _get(features, "trend_efficiency_12", count) * 0.20
            + _get(features, "short_breakdown_body_confirmed", count) * 0.15
        )
        slow_trend = _clip_unit(
            np.maximum(0.0, -_get(features, "ema21_slope_12", count)) / 0.005 * 0.25
            + np.maximum(0.0, -_get(features, "ema200_slope_64", count)) / 0.005 * 0.20
            + _get(features, "ema_stack_bearish", count) * 0.20
            + _get(features, "local_trend_down_score", count) * 0.20
            + (1.0 - _get(features, "atr_percentile_64", count)) * 0.15
        )
    else:
        breakdown = _clip_unit(
            _get(features, "breakout_up_strength_12", count) / 0.005 * 0.35
            + (_get(features, "return_30m", count) > 0.0).astype(float) * 0.20
            + (_get(features, "return_60m", count) > 0.0).astype(float) * 0.20
            + _get(features, "volume_ratio_12", count) / 2.0 * 0.25
        )
        fake = _clip_unit(
            _get(features, "failed_breakout_up_risk", count) * 0.40
            + _get(features, "sweep_high_reversal", count) * 0.35
            + _get(features, "upper_wick_ratio", count) * 0.25
        )
        extension = _clip_unit(
            np.maximum(0.0, _get(features, "distance_ema21", count)) / 0.010 * 0.30
            + np.maximum(0.0, _get(features, "distance_ema200", count)) / 0.020 * 0.20
            + _get(features, "atr_percentile_64", count) * 0.25
            + _get(features, "upper_wick_ratio", count) * 0.25
        )
        momentum = _clip_unit(
            np.maximum(0.0, _get(features, "momentum_acceleration_12", count)) / 0.005 * 0.25
            + _get(features, "volume_ratio_12", count) / 2.0 * 0.20
            + _get(features, "trend_efficiency_12", count) * 0.25
            + (_get(features, "return_30m", count) > 0.0).astype(float) * 0.30
        )
        slow_trend = _clip_unit(
            np.maximum(0.0, _get(features, "ema21_slope_12", count)) / 0.005 * 0.30
            + np.maximum(0.0, _get(features, "ema200_slope_64", count)) / 0.005 * 0.25
            + _get(features, "ema_stack_bullish", count) * 0.25
            + (1.0 - _get(features, "atr_percentile_64", count)) * 0.20
        )
    return {
        "breakdown_continuation_short": breakdown,
        "fake_breakdown_reversal": fake,
        "mean_reversion_after_extension": extension,
        "momentum_burst": momentum,
        "slow_trend_short": slow_trend,
        "avoid_only": np.maximum(fake, extension),
    }


def _first_close_move(entry: float, future_close: np.ndarray, side: str, threshold: float) -> int:
    if side.upper() == "SHORT":
        hits = entry / np.maximum(future_close, 1e-12) - 1.0 >= threshold
    else:
        hits = future_close / max(entry, 1e-12) - 1.0 >= threshold
    positions = np.flatnonzero(hits)
    return int(positions[0] + 1) if len(positions) else -1


def compute_alternative_outcomes(
    market: Any,
    steps: np.ndarray,
    side: str,
    horizon: int,
    fee_round_trip: float,
) -> dict[str, np.ndarray]:
    rows: dict[str, list[float]] = {
        name: [] for name in (
            *ALT_HIT_RULES.keys(), "time_to_green", "time_to_minus3", "time_to_minus5",
            "max_mfe", "max_mae", "mfe_mae_ratio", "trade_quality", "danger",
            "reversal_hit5_after_sweep", "reversal_hit8_after_sweep",
        )
    }
    normalized_side = side.lower()
    reversal_side = "long" if normalized_side == "short" else "short"
    for raw_step in np.asarray(steps, dtype=np.int64):
        step = int(raw_step)
        entry = float(market.close[step])
        high = np.asarray(market.high[step + 1 : step + horizon + 1], dtype=np.float32)
        low = np.asarray(market.low[step + 1 : step + horizon + 1], dtype=np.float32)
        future_close = np.asarray(market.close[step + 1 : step + horizon + 1], dtype=np.float32)
        outcomes = {
            name: compute_path_outcome(entry, high, low, normalized_side, target, stop)
            for name, (target, stop) in ALT_HIT_RULES.items()
        }
        reversal5 = compute_path_outcome(entry, high, low, reversal_side, 0.005, 0.003)
        reversal8 = compute_path_outcome(entry, high, low, reversal_side, 0.008, 0.005)
        canonical = outcomes["hit8_before_minus5"]
        for name, outcome in outcomes.items():
            rows[name].append(float(bool(outcome["hit_before_stop"])))
        mfe, mae = float(canonical["mfe"]), float(canonical["mae"])
        rows["time_to_green"].append(float(_first_close_move(entry, future_close, side, 0.0)))
        rows["time_to_minus3"].append(float(outcomes["hit5_before_minus3"]["time_to_stop"]))
        rows["time_to_minus5"].append(float(canonical["time_to_stop"]))
        rows["max_mfe"].append(mfe)
        rows["max_mae"].append(mae)
        rows["mfe_mae_ratio"].append(mfe / max(mae, 1e-6))
        rows["trade_quality"].append(compute_trade_quality(
            bool(outcomes["hit5_before_minus3"]["hit_before_stop"]),
            bool(canonical["hit_before_stop"]),
            mfe,
            mae,
            fee_round_trip,
        ))
        rows["danger"].append(float(mae >= 0.005))
        rows["reversal_hit5_after_sweep"].append(float(bool(reversal5["hit_before_stop"])))
        rows["reversal_hit8_after_sweep"].append(float(bool(reversal8["hit_before_stop"])))
    return {name: np.asarray(values, dtype=np.float32) for name, values in rows.items()}


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float | None:
    return metric_mean(np.asarray(values)[mask]) if np.any(mask) else None


def summarize_targets(symbol: str, side: str, role: str, horizon: int, outcomes: dict[str, np.ndarray]) -> dict[str, Any]:
    count = len(outcomes["trade_quality"])
    return {
        "symbol": symbol,
        "side": side,
        "role": role,
        "horizon_candles": horizon,
        "sample_count": count,
        "hit3": metric_mean(outcomes["hit3_before_minus2"]),
        "hit5": metric_mean(outcomes["hit5_before_minus3"]),
        "hit6": metric_mean(outcomes["hit6_before_minus4"]),
        "hit8": metric_mean(outcomes["hit8_before_minus5"]),
        "hit10_minus6": metric_mean(outcomes["hit10_before_minus6"]),
        "hit10_minus8": metric_mean(outcomes["hit10_before_minus8"]),
        "avg_trade_quality": metric_mean(outcomes["trade_quality"]),
        "avg_mfe": metric_mean(outcomes["max_mfe"]),
        "avg_mae": metric_mean(outcomes["max_mae"]),
        "p90_mae": metric_quantile(outcomes["max_mae"], 0.90),
        "danger_rate": metric_mean(outcomes["danger"]),
        "time_to_green_avg": metric_mean(outcomes["time_to_green"][outcomes["time_to_green"] >= 0]),
        "time_to_minus3_avg": metric_mean(outcomes["time_to_minus3"][outcomes["time_to_minus3"] >= 0]),
        "time_to_minus5_avg": metric_mean(outcomes["time_to_minus5"][outcomes["time_to_minus5"] >= 0]),
        "reversal_hit5_after_sweep": metric_mean(outcomes["reversal_hit5_after_sweep"]),
        "reversal_hit8_after_sweep": metric_mean(outcomes["reversal_hit8_after_sweep"]),
    }


def family_bucket_rows(
    symbol: str,
    side: str,
    role: str,
    horizon: int,
    scores: dict[str, np.ndarray],
    outcomes: dict[str, np.ndarray],
    cost_proxy: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, score in scores.items():
        buckets = bucketize_score(score)
        for bucket in BUCKET_LABELS:
            mask = buckets == bucket
            if not np.any(mask):
                continue
            quality = _masked_mean(outcomes["trade_quality"], mask)
            mfe = _masked_mean(outcomes["max_mfe"], mask)
            mae = _masked_mean(outcomes["max_mae"], mask)
            reversal5 = _masked_mean(outcomes["reversal_hit5_after_sweep"], mask)
            reversal8 = _masked_mean(outcomes["reversal_hit8_after_sweep"], mask)
            avoided_loss = _masked_mean(np.maximum(0.0, -outcomes["trade_quality"]), mask)
            false_block = _masked_mean((outcomes["trade_quality"] > 0).astype(np.float32), mask)
            row = {
                "symbol": symbol,
                "side": side,
                "role": role,
                "horizon_candles": horizon,
                "alpha_family": family,
                "bucket": bucket,
                "count": int(np.sum(mask)),
                "hit3": _masked_mean(outcomes["hit3_before_minus2"], mask),
                "hit5": _masked_mean(outcomes["hit5_before_minus3"], mask),
                "hit6": _masked_mean(outcomes["hit6_before_minus4"], mask),
                "hit8": _masked_mean(outcomes["hit8_before_minus5"], mask),
                "hit10": _masked_mean(outcomes["hit10_before_minus6"], mask),
                "avg_trade_quality": quality,
                "avg_mfe": mfe,
                "avg_mae": mae,
                "p90_mae": metric_quantile(outcomes["max_mae"][mask], 0.90),
                "mfe_mae_ratio": (finite(mfe) / max(finite(mae), 1e-6)) if mfe is not None and mae is not None else None,
                "danger_rate": _masked_mean(outcomes["danger"], mask),
                "net_quality_proxy_after_cost": finite(quality) - cost_proxy if quality is not None else None,
                "reversal_hit5": reversal5,
                "reversal_hit8": reversal8,
                "avoid_short_after_sweep_quality": (
                    finite(reversal5) - finite(row_quality)
                    if (row_quality := quality) is not None and reversal5 is not None else None
                ),
                "avoided_loss_if_filter_active": avoided_loss if family == "avoid_only" else None,
                "avoided_mae_if_filter_active": mae if family == "avoid_only" else None,
                "false_block_rate": false_block if family == "avoid_only" else None,
                "saved_pnl_proxy": (finite(avoided_loss) - cost_proxy) if family == "avoid_only" and avoided_loss is not None else None,
                "suggested_alpha_family": family,
            }
            rows.append(row)
    return rows


def _high_rows(rows: list[dict[str, Any]], family: str) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if row.get("alpha_family") == family and row.get("bucket") in {"HIGH", "EXTREME"}
    ]


def _best_row(rows: list[dict[str, Any]], family: str, key: str = "net_quality_proxy_after_cost") -> dict[str, Any] | None:
    relevant = _high_rows(rows, family)
    return max(relevant, key=lambda row: finite(row.get(key), -999.0), default=None)


def diagnose_symbol_alpha_profile(symbol: str, side: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    breakdown = _best_row(rows, "breakdown_continuation_short")
    fake = _best_row(rows, "fake_breakdown_reversal", "reversal_hit5")
    extension = _best_row(rows, "mean_reversion_after_extension")
    momentum = _best_row(rows, "momentum_burst")
    slow = _best_row(rows, "slow_trend_short")
    avoid = _best_row(rows, "avoid_only", "saved_pnl_proxy")
    candidates = [row for row in (breakdown, momentum, slow) if row is not None]
    best_entry = max(candidates, key=lambda row: finite(row.get("net_quality_proxy_after_cost"), -999.0), default=None)
    family = "avoid_only_or_no_trade"
    target = "none"
    reason = "no_positive_net_quality_bucket"
    next_phase = "do_not_train_entry_model_consider_avoid_filter"
    confidence = "LOW"
    if fake and finite(fake.get("reversal_hit5")) > finite(fake.get("hit5")) + 0.03 and finite(fake.get("avg_trade_quality")) <= 0.0:
        family = "fake_breakdown_reversal"
        target = "reversal_hit5_after_sweep"
        reason = "fake_breakdown_bucket_favors_opposite_side_over_continuation"
        next_phase = "train_avoid_short_after_sweep_or_reversal_research_model"
        confidence = "MODERATE"
    elif extension and finite(extension.get("danger_rate")) >= 0.55 and finite(extension.get("avg_trade_quality")) <= 0.0:
        family = "mean_reversion_after_extension"
        target = "avoid_short_after_extension"
        reason = "extension_bucket_has_high_mae_and_nonpositive_quality"
        next_phase = "train_avoid_filter_for_overextended_entries"
        confidence = "MODERATE"
    elif best_entry and finite(best_entry.get("net_quality_proxy_after_cost"), -999.0) > 0.0:
        if (
            best_entry["alpha_family"] == "momentum_burst"
            and finite(best_entry.get("hit5")) > finite(best_entry.get("hit8")) + 0.04
        ):
            family = "momentum_burst_lower_target"
            target = "hit5_before_minus3_or_hit6_before_minus4"
            reason = "momentum_bucket_has_short_horizon_edge_but_hit8_decay"
        elif best_entry["alpha_family"] == "slow_trend_short" and finite(best_entry.get("hit5")) > finite(best_entry.get("hit8")) + 0.04:
            family = "slow_trend_short"
            target = "hit3_before_minus2_or_hit5_before_minus3"
            reason = "slow_trend_bucket_favors_lower_targets"
        else:
            family = best_entry["alpha_family"]
            target = "hit8_before_minus5"
            reason = "continuation_bucket_retains_positive_net_quality"
        next_phase = f"train_{family}_candidate_model"
        confidence = "MODERATE"
    elif avoid and finite(avoid.get("saved_pnl_proxy"), -999.0) > 0.0:
        reason = "avoid_filter_has_positive_saved_loss_proxy_without_entry_edge"
        confidence = "MODERATE"
    best_reference = best_entry or fake or extension or avoid or {}
    return {
        "symbol": symbol,
        "side": side,
        "primary_failure_reason": reason,
        "best_alpha_family": family,
        "alternative_target_recommendation": target,
        "avoid_only_potential": family in {"avoid_only_or_no_trade", "mean_reversion_after_extension", "fake_breakdown_reversal"},
        "confidence": confidence,
        "validation_status": "REQUIRES_LOCKBOX_VALIDATION",
        "recommended_next_research_phase": next_phase,
        "best_horizon": best_reference.get("horizon_candles"),
        "best_bucket_net_quality": best_reference.get("net_quality_proxy_after_cost"),
        "best_bucket_hit5": best_reference.get("hit5"),
        "best_bucket_hit8": best_reference.get("hit8"),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _render(value: Any) -> str:
    return "null" if value is None else f"{float(value):.4f}"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        f"# Aegis Failed Alpha Symbol Profile {report['created_at']}",
        "",
        "## Safety",
        "",
        "- Mode: `RESEARCH_ONLY`.",
        "- This tool computes diagnostic features and path outcomes only; it does not train or persist models.",
        "- Family selection is in-sample discovery; all candidates require independent lockbox validation.",
        "- No `active/` or shadow model artifacts, active manifests, live inference, YAML, thresholds or PM2 changes.",
        "",
        "## Symbol Diagnosis",
        "",
        "| Symbol | Side | Role | Failure Reason | Best Alpha Family | Target | Avoid Only | Confidence | Next Phase |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in report["profiles"]:
        lines.append(
            f"| {row['symbol']} | {row['side']} | {row['role']} | {row['primary_failure_reason']} | "
            f"{row['best_alpha_family']} | {row['alternative_target_recommendation']} | "
            f"{row['avoid_only_potential']} | {row['confidence']} | {row['recommended_next_research_phase']} |"
        )
    lines.extend(["", "## Failed Candidate Detail", ""])
    for row in report["profiles"]:
        if row["role"] != "failed_candidate":
            continue
        lines.append(
            f"- `{row['symbol']} {row['side']}`: `{row['best_alpha_family']}`, target "
            f"`{row['alternative_target_recommendation']}`, net bucket quality `{_render(row.get('best_bucket_net_quality'))}`."
        )
    lines.extend(["", "## Control Comparison", ""])
    if report["control_comparison"]:
        for item in report["control_comparison"]:
            lines.append(f"- {item}")
    else:
        lines.append("- Controls were not requested.")
    lines.extend(["", "## Recommendations", ""])
    for item in report["recommendations"]:
        lines.append(f"- {item}")
    if report["errors"]:
        lines.extend(["", "## Errors", "", f"- `{report['errors']}`"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    symbols = parse_symbols(
        args.symbols,
        include_confirmed=bool(args.include_confirmed_controls),
        include_repaired=bool(args.include_repaired_controls),
        include_longs=bool(args.include_longs),
    )
    sides = parse_sides(args.side, bool(args.include_longs))
    horizons = list(dict.fromkeys(int(value.strip()) for value in args.horizons.split(",") if value.strip()))
    if any(value <= 0 or value > 24 for value in horizons):
        raise ValueError("horizons must be positive and at most 24 candles")
    context_markets: dict[str, Any] = {}
    context_warning: str | None = None
    try:
        context_markets = {
            symbol: load_signal_market(DEFAULT_TURBO_CONFIG.config_path, symbol_override=symbol)
            for symbol in ("BTCUSDT", "ETHUSDT")
        }
    except Exception as exc:
        context_warning = f"cross_symbol_context_unavailable:{exc!r}"
    cost_proxy = (float(args.fee_bps) + float(args.slippage_bps)) / 10000.0
    profiles: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for symbol in symbols:
        try:
            market = load_signal_market(DEFAULT_TURBO_CONFIG.config_path, symbol_override=symbol)
            built = build_recent_dataset(symbol, int(args.lookback_days), save=False, market=market)
            base = built["dataset"]
            feature_dataset = apply_feature_set(base, market, "combined_v3", context_markets=context_markets)
            for side in sides:
                local_buckets: list[dict[str, Any]] = []
                scores = compute_family_scores(feature_dataset, side)
                for horizon in horizons:
                    outcomes = compute_alternative_outcomes(
                        market,
                        np.asarray(base["step"], dtype=np.int64),
                        side,
                        horizon,
                        float(market.cfg.risk.total_fee * 2.0),
                    )
                    target_rows.append(summarize_targets(symbol, side, role_for_symbol(symbol), horizon, outcomes))
                    local_buckets.extend(family_bucket_rows(
                        symbol, side, role_for_symbol(symbol), horizon, scores, outcomes, cost_proxy
                    ))
                profile = diagnose_symbol_alpha_profile(symbol, side, local_buckets)
                diagnostics = feature_dataset.get("feature_diagnostics", {}).get("v3", {})
                profile.update({
                    "role": role_for_symbol(symbol),
                    "sample_count": int(len(base["step"])),
                    "cross_symbol_context_available": diagnostics.get("cross_symbol_context_available", False),
                })
                profiles.append(profile)
                bucket_rows.extend(local_buckets)
        except Exception as exc:
            errors.append({"symbol": symbol, "error": repr(exc)})
    control_rows = [row for row in profiles if row["role"] != "failed_candidate"]
    failed_rows = [row for row in profiles if row["role"] == "failed_candidate"]
    control_comparison: list[str] = []
    if control_rows:
        control_families = Counter(row["best_alpha_family"] for row in control_rows)
        failed_families = Counter(row["best_alpha_family"] for row in failed_rows)
        control_comparison = [
            f"Control best-family distribution: {dict(control_families)}.",
            f"Failed-candidate best-family distribution: {dict(failed_families)}.",
            "Differences are descriptive diagnostics, not independent validation or promotion evidence.",
        ]
    recommendations = [
        "Create L2 models only for symbols with a clear alternative family and validate them with a new temporal lockbox.",
        "Symbols classified as avoid-only should receive filter research, not an entry model.",
        "LONG diagnostics, when requested, must remain a separate pipeline from SHORT research.",
    ]
    stamp = utc_stamp()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "md": out_dir / f"aegis_failed_alpha_profile_{stamp}.md",
        "json": out_dir / f"aegis_failed_alpha_profile_{stamp}.json",
        "summary_csv": out_dir / f"aegis_failed_alpha_profile_summary_{stamp}.csv",
        "buckets_csv": out_dir / f"aegis_failed_alpha_profile_buckets_{stamp}.csv",
        "targets_csv": out_dir / f"aegis_failed_alpha_profile_targets_{stamp}.csv",
        "recommendations_csv": out_dir / f"aegis_failed_alpha_profile_recommendations_{stamp}.csv",
    }
    report = {
        "schema_version": "aegis_failed_alpha_profile_l1_v1",
        "created_at": utc_now().isoformat(),
        "mode": MODE,
        "discovery_only_no_out_of_sample_confirmation": True,
        "symbols": symbols,
        "sides": sides,
        "lookback_days": int(args.lookback_days),
        "horizons": horizons,
        "cost_proxy": {"fee_bps": float(args.fee_bps), "slippage_bps": float(args.slippage_bps)},
        "alpha_families": list(ALPHA_FAMILIES),
        "alternative_hit_rules": ALT_HIT_RULES,
        "cross_symbol_context_warning": context_warning,
        "profiles": profiles,
        "bucket_rows": bucket_rows,
        "target_rows": target_rows,
        "control_comparison": control_comparison,
        "recommendations": recommendations,
        "errors": errors,
        "models_trained": False,
        "model_artifacts_written": False,
        "shadow_models_generated": False,
        "active_manifest_touched": False,
        "live_inference_changed": False,
        "paths": {name: str(path) for name, path in paths.items()},
    }
    paths["json"].write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(paths["md"], report)
    _write_csv(paths["summary_csv"], profiles, SUMMARY_COLUMNS)
    _write_csv(paths["buckets_csv"], bucket_rows, BUCKET_COLUMNS)
    _write_csv(paths["targets_csv"], target_rows, TARGET_COLUMNS)
    recommendation_rows = [
        {
            "symbol": row["symbol"],
            "side": row["side"],
            "best_alpha_family": row["best_alpha_family"],
            "recommended_target": row["alternative_target_recommendation"],
            "avoid_only_potential": row["avoid_only_potential"],
            "confidence": row["confidence"],
            "validation_status": row["validation_status"],
            "next_research_phase": row["recommended_next_research_phase"],
        }
        for row in profiles
    ]
    _write_csv(
        paths["recommendations_csv"],
        recommendation_rows,
        ("symbol", "side", "best_alpha_family", "recommended_target", "avoid_only_potential", "confidence", "validation_status", "next_research_phase"),
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Research-only alternative alpha profiling for failed Turbo SHORT symbols.")
    parser.add_argument("--symbols")
    parser.add_argument("--side", choices=("SHORT", "LONG", "BOTH"), default="SHORT")
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument("--horizons", default="6,12,24")
    parser.add_argument("--out-dir", default="/home/jasan/Develop")
    parser.add_argument("--fee-bps", type=float, default=8.0)
    parser.add_argument("--slippage-bps", type=float, default=3.0)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--include-longs", action="store_true")
    parser.add_argument("--include-confirmed-controls", action="store_true")
    parser.add_argument("--include-repaired-controls", action="store_true")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({
        "paths": report["paths"],
        "profiles": [
            {
                "symbol": row["symbol"],
                "side": row["side"],
                "family": row["best_alpha_family"],
                "target": row["alternative_target_recommendation"],
                "confidence": row["confidence"],
            }
            for row in report["profiles"]
        ],
        "errors": report["errors"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
