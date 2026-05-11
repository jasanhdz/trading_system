from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from aegis_alpha.entry_quality.model_loader import load_entry_quality_models, normalize_symbol
from aegis_alpha.turbo.config import DEFAULT_TURBO_CONFIG
from aegis_alpha.turbo.snapshot_utils import (
    load_turbo_snapshot_status,
    turbo_snapshot_path,
)


FEATURE_CACHE_TTL_SECONDS = 30.0
_FEATURE_CACHE: dict[tuple[str, str], tuple[float, "EntryQualityFeatureVector"]] = {}


@dataclass(frozen=True)
class EntryQualityFeatureVector:
    x: np.ndarray | None
    feature_columns: list[str]
    missing_features: list[str]
    feature_status: str
    reason: str
    feature_timestamp: str | None


def _score_from_turbo(turbo_context: dict[str, Any] | None) -> dict[str, Any]:
    turbo = (turbo_context or {}).get("turbo") if isinstance(turbo_context, dict) else None
    if not isinstance(turbo, dict):
        turbo = turbo_context if isinstance(turbo_context, dict) else {}
    raw = turbo.get("raw") if isinstance(turbo.get("raw"), dict) else {}
    recent = raw.get("recent_scores") if isinstance(raw.get("recent_scores"), dict) else turbo.get("recent_scores", {})
    votes = raw.get("votes") if isinstance(raw.get("votes"), dict) else turbo.get("votes", {})
    action = str(raw.get("action") or turbo.get("action") or "HOLD").upper()
    return {
        "long_score_7d": recent.get("long_7d"),
        "long_score_14d": recent.get("long_14d"),
        "long_score_30d": recent.get("long_30d"),
        "short_score_7d": recent.get("short_7d"),
        "short_score_14d": recent.get("short_14d"),
        "short_score_30d": recent.get("short_30d"),
        "votes_long": votes.get("long", 0),
        "votes_short": votes.get("short", 0),
        "votes_neutral": votes.get("neutral", 0),
        "turbo_score": raw.get("turbo_score", turbo.get("turbo_score")),
        "turbo_action_long": 1.0 if action == "LONG" else 0.0,
        "turbo_action_short": 1.0 if action == "SHORT" else 0.0,
        "turbo_action_hold": 1.0 if action not in {"LONG", "SHORT"} else 0.0,
        "side_long": 1.0 if action == "LONG" else 0.0,
        "side_short": 1.0 if action == "SHORT" else 0.0,
    }


def _latest_snapshot(symbol: str) -> tuple[dict[str, Any], dict[str, float]]:
    best_status: dict[str, Any] | None = None
    best_values: dict[str, float] = {}
    for lookback in DEFAULT_TURBO_CONFIG.lookback_days:
        path = turbo_snapshot_path(int(lookback), symbol)
        status = load_turbo_snapshot_status(path, include_sample_count=False)
        if not status.get("exists"):
            continue
        if best_status is not None:
            current_key = (status.get("feature_timestamp") or "", status.get("snapshot_mtime") or "")
            best_key = (best_status.get("feature_timestamp") or "", best_status.get("snapshot_mtime") or "")
            if current_key <= best_key:
                continue
        try:
            with np.load(path, allow_pickle=True) as data:
                x = data["live_X"][-1] if "live_X" in data and len(data["live_X"]) else data["X"][-1]
                names = [str(item) for item in data["feature_names"].tolist()]
                best_values = {name: float(value) for name, value in zip(names, x)}
                best_status = status
        except Exception as exc:
            status["error"] = f"snapshot_feature_read_error:{exc!r}"
            best_status = status
    if best_status is None:
        best_status = load_turbo_snapshot_status(turbo_snapshot_path(int(DEFAULT_TURBO_CONFIG.lookback_days[0]), symbol), include_sample_count=False)
    return best_status, best_values


def _safe_number(value: Any) -> float:
    try:
        out = float(np.asarray(value).item())
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _put_if_available(out: dict[str, float], key: str, value: Any, missing: set[str]) -> None:
    number = _safe_number(value)
    if np.isfinite(number):
        out[key] = number
        missing.discard(key)


def build_entry_quality_features(symbol: str, turbo_context: dict[str, Any] | None = None) -> EntryQualityFeatureVector:
    normalized = normalize_symbol(symbol)
    cache = load_entry_quality_models()
    feature_columns = list(cache.feature_columns)
    status, snapshot = _latest_snapshot(normalized)
    feature_timestamp = status.get("feature_timestamp") or status.get("last_ts")
    cache_key = (normalized, str(feature_timestamp))
    cached = _FEATURE_CACHE.get(cache_key)
    now = time.time()
    if cached is not None and now - cached[0] <= FEATURE_CACHE_TTL_SECONDS:
        base = cached[1]
        # Turbo scores can change with hot-loaded models even when feature timestamp is stable.
        return _merge_context_features(base, turbo_context, cache.symbol_encoding, normalized)

    if not status.get("exists"):
        return EntryQualityFeatureVector(None, feature_columns, feature_columns, "insufficient", "entry_quality_turbo_snapshot_missing", feature_timestamp)
    if not status.get("is_fresh", False):
        return EntryQualityFeatureVector(None, feature_columns, feature_columns, "insufficient", "turbo_snapshot_stale_for_entry_quality", feature_timestamp)

    missing = set(feature_columns)
    out: dict[str, float] = {}
    # Approximate current Phase 1 columns from the already-built Turbo edge snapshot.
    mapping = {
        "ret_1": "last_log_ret",
        "ret_2": "mean_6_log_ret",
        "ret_3": "mean_6_log_ret",
        "ret_6": "mean_6_log_ret",
        "ret_12": "mean_12_log_ret",
        "momentum_3": "mean_6_log_ret",
        "momentum_6": "mean_6_log_ret",
        "momentum_12": "mean_12_log_ret",
        "candle_body_pct": "last_trend_efficiency",
        "price_to_ema_9": "last_ema_9_norm",
        "price_to_ema_21": "last_ema_21_norm",
        "ema_9_slope": "last_ema_1h_slope",
        "ema_21_slope": "last_ema_4h_slope",
        "atr_pct": "last_vol_regime",
        "realized_vol_12": "std_12_log_ret",
        "realized_vol_36": "std_64_log_ret",
        "high_low_range_pct": "last_high_norm",
        "volume_zscore_36": "last_vol_z",
        "volume_ratio_12": "last_vol_norm",
        "mtf_15m_ret_1": "mean_6_log_ret",
        "mtf_15m_ret_2": "mean_12_log_ret",
        "mtf_15m_ema_9_slope": "last_ema_1h_slope",
        "mtf_15m_price_to_ema_21": "last_ema_21_norm",
        "mtf_15m_atr_pct": "last_vol_regime",
        "mtf_1h_ret_1": "mean_12_log_ret",
        "mtf_1h_ret_2": "mean_64_log_ret",
        "mtf_1h_ema_9_slope": "last_ema_1h_slope",
        "mtf_1h_price_to_ema_21": "last_ema_21_norm",
        "mtf_1h_atr_pct": "last_vol_regime",
    }
    for target, source in mapping.items():
        _put_if_available(out, target, snapshot.get(source), missing)
    ema_slope = out.get("mtf_1h_ema_9_slope", out.get("ema_9_slope", 0.0))
    out["mtf_15m_trend_direction"] = 1.0 if ema_slope > 0 else (-1.0 if ema_slope < 0 else 0.0)
    out["mtf_1h_trend_direction"] = out["mtf_15m_trend_direction"]
    out["long_mtf_agreement"] = 1.0 if out.get("momentum_3", 0.0) > 0 and out["mtf_15m_trend_direction"] >= 0 else 0.0
    out["short_mtf_agreement"] = 1.0 if out.get("momentum_3", 0.0) < 0 and out["mtf_15m_trend_direction"] <= 0 else 0.0
    out["mtf_conflict"] = 1.0 if (out.get("momentum_6", 0.0) * out["mtf_15m_trend_direction"]) < 0 else 0.0
    for key in ("mtf_15m_trend_direction", "mtf_1h_trend_direction", "long_mtf_agreement", "short_mtf_agreement", "mtf_conflict"):
        missing.discard(key)
    symbol_code = cache.symbol_encoding.get(normalized)
    if symbol_code is not None:
        out["symbol_code"] = float(symbol_code)
        missing.discard("symbol_code")
    for symbol_key in cache.symbol_encoding:
        key = f"symbol_is_{symbol_key}"
        out[key] = 1.0 if symbol_key == normalized else 0.0
        missing.discard(key)
    out["candidate_reason_code"] = 0.0
    missing.discard("candidate_reason_code")

    vector = _to_vector(out, feature_columns, missing)
    base = EntryQualityFeatureVector(
        vector,
        feature_columns,
        sorted(missing),
        "partial" if missing else "ok",
        "entry_quality_features_partial" if missing else "entry_quality_features_ok",
        str(feature_timestamp) if feature_timestamp else None,
    )
    _FEATURE_CACHE[cache_key] = (now, base)
    return _merge_context_features(base, turbo_context, cache.symbol_encoding, normalized)


def _merge_context_features(
    base: EntryQualityFeatureVector,
    turbo_context: dict[str, Any] | None,
    symbol_encoding: dict[str, int],
    symbol: str,
) -> EntryQualityFeatureVector:
    if base.x is None:
        return base
    values = {col: float(base.x[0, idx]) for idx, col in enumerate(base.feature_columns)}
    missing = set(base.missing_features)
    score_values = _score_from_turbo(turbo_context)
    for key, value in score_values.items():
        if key in values:
            _put_if_available(values, key, value, missing)
    long_scores = [values.get(f"long_score_{days}d", np.nan) for days in (7, 14, 30)]
    short_scores = [values.get(f"short_score_{days}d", np.nan) for days in (7, 14, 30)]
    long_finite = [item for item in long_scores if np.isfinite(item)]
    short_finite = [item for item in short_scores if np.isfinite(item)]
    if long_finite and short_finite:
        _put_if_available(values, "score_gap", max(long_finite) - max(short_finite), missing)
    if "turbo_score" in values and not np.isfinite(values["turbo_score"]):
        side_score = max(long_finite + short_finite) if long_finite or short_finite else np.nan
        _put_if_available(values, "turbo_score", side_score, missing)
    if symbol in symbol_encoding:
        _put_if_available(values, "symbol_code", symbol_encoding[symbol], missing)
    vector = _to_vector(values, base.feature_columns, missing)
    missing_list = sorted(missing)
    missing_ratio = len(missing_list) / max(len(base.feature_columns), 1)
    status = "ok" if not missing_list else ("partial" if missing_ratio < 0.45 else "insufficient")
    reason = "entry_quality_features_ok" if status == "ok" else ("entry_quality_features_partial" if status == "partial" else "insufficient_entry_quality_features")
    return EntryQualityFeatureVector(vector, base.feature_columns, missing_list, status, reason, base.feature_timestamp)


def _to_vector(values: dict[str, float], feature_columns: list[str], missing: set[str]) -> np.ndarray:
    row = []
    for col in feature_columns:
        value = values.get(col, np.nan)
        if np.isfinite(value):
            missing.discard(col)
        row.append(value)
    return np.asarray([row], dtype=np.float32)


def entry_quality_feature_cache_status() -> dict[str, Any]:
    return {
        "cache_size": len(_FEATURE_CACHE),
        "ttl_seconds": FEATURE_CACHE_TTL_SECONDS,
    }
