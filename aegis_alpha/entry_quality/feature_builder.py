from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from aegis_alpha.entry_quality.model_loader import load_entry_quality_models, normalize_symbol
from aegis_alpha.entry_quality.runtime_feature_cache import get_runtime_market_features, runtime_feature_cache_status
from aegis_alpha.turbo.config import DEFAULT_TURBO_CONFIG
from aegis_alpha.turbo.snapshot_utils import (
    load_turbo_snapshot_status,
    turbo_snapshot_path,
)


FEATURE_CACHE_TTL_SECONDS = 30.0
_FEATURE_CACHE: dict[tuple[str, str], tuple[float, "EntryQualityFeatureVector"]] = {}
_LAST_FINAL_STATUS_BY_SYMBOL: dict[str, str] = {}
_LAST_FINAL_PARITY_BY_SYMBOL: dict[str, float] = {}
_LAST_FINAL_MISSING_BY_SYMBOL: dict[str, list[str]] = {}


@dataclass(frozen=True)
class EntryQualityFeatureVector:
    x: np.ndarray | None
    feature_columns: list[str]
    missing_features: list[str]
    feature_status: str
    reason: str
    feature_timestamp: str | None
    feature_parity_pct: float = 0.0
    approximated_features: list[str] | None = None
    critical_missing_groups: list[str] | None = None
    feature_build_latency_ms: float = 0.0


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


FEATURE_GROUPS: dict[str, list[str]] = {
    "turbo_scores_votes": [
        "long_score_7d",
        "long_score_14d",
        "long_score_30d",
        "short_score_7d",
        "short_score_14d",
        "short_score_30d",
        "votes_long",
        "votes_short",
        "votes_neutral",
        "turbo_score",
        "score_gap",
    ],
    "5m_returns_momentum": ["ret_1", "ret_2", "ret_3", "ret_6", "ret_12", "momentum_3", "momentum_6", "momentum_12"],
    "5m_candle_anatomy": ["candle_body_pct", "upper_wick_pct", "lower_wick_pct", "high_low_range_pct", "green_candle_count_3", "red_candle_count_3"],
    "5m_ema": ["ema_9", "ema_21", "ema_50", "price_to_ema_9", "price_to_ema_21", "ema_9_slope", "ema_21_slope"],
    "5m_atr_volatility": ["atr_14", "atr_pct", "realized_vol_12", "realized_vol_36", "atr_percentile_7d"],
    "volume_quote": ["volume_zscore_36", "volume_ratio_12", "quote_volume"],
    "15m_mtf": ["mtf_15m_ret_1", "mtf_15m_ret_2", "mtf_15m_ema_9_slope", "mtf_15m_price_to_ema_21", "mtf_15m_atr_pct", "mtf_15m_trend_direction"],
    "1h_mtf": ["mtf_1h_ret_1", "mtf_1h_ret_2", "mtf_1h_ema_9_slope", "mtf_1h_price_to_ema_21", "mtf_1h_atr_pct", "mtf_1h_trend_direction"],
    "symbol_side_encoding": ["symbol_code", "side_long", "side_short", "turbo_action_long", "turbo_action_short", "turbo_action_hold"],
}

CRITICAL_GROUPS = {"turbo_scores_votes", "5m_returns_momentum", "symbol_side_encoding"}


def _group_coverage(values: dict[str, float], columns: list[str]) -> dict[str, float]:
    present = set(columns)
    coverage: dict[str, float] = {}
    for group, group_cols in FEATURE_GROUPS.items():
        expected = [col for col in group_cols if col in present]
        if not expected:
            continue
        available = sum(1 for col in expected if np.isfinite(values.get(col, np.nan)))
        coverage[group] = available / len(expected)
    return coverage


def _grade_features(values: dict[str, float], feature_columns: list[str], missing: set[str]) -> tuple[str, str, float, list[str]]:
    total = max(len(feature_columns), 1)
    missing_count = len([col for col in feature_columns if col in missing])
    parity_pct = round(100.0 * (total - missing_count) / total, 2)
    coverage = _group_coverage(values, feature_columns)
    critical_missing = [group for group in CRITICAL_GROUPS if coverage.get(group, 0.0) < 0.60]
    if parity_pct >= 95.0 and not critical_missing:
        return "ok", "entry_quality_features_ok", parity_pct, critical_missing
    if parity_pct >= 70.0 and not critical_missing:
        return "partial", "entry_quality_features_partial", parity_pct, critical_missing
    return "insufficient", "insufficient_entry_quality_features", parity_pct, critical_missing


def build_entry_quality_features(symbol: str, turbo_context: dict[str, Any] | None = None) -> EntryQualityFeatureVector:
    start = time.perf_counter()
    normalized = normalize_symbol(symbol)
    cache = load_entry_quality_models()
    feature_columns = list(cache.feature_columns)
    status, snapshot = _latest_snapshot(normalized)
    market = get_runtime_market_features(normalized)
    feature_timestamp = market.feature_timestamp or status.get("feature_timestamp") or status.get("last_ts")
    cache_key = (normalized, str(feature_timestamp))
    cached = _FEATURE_CACHE.get(cache_key)
    now = time.time()
    if cached is not None and now - cached[0] <= FEATURE_CACHE_TTL_SECONDS:
        base = cached[1]
        # Turbo scores can change with hot-loaded models even when feature timestamp is stable.
        return _merge_context_features(base, turbo_context, cache.symbol_encoding, normalized)

    if not status.get("exists") and not market.values:
        latency_ms = (time.perf_counter() - start) * 1000
        return EntryQualityFeatureVector(None, feature_columns, feature_columns, "insufficient", "entry_quality_runtime_features_missing", feature_timestamp, 0.0, [], list(CRITICAL_GROUPS), latency_ms)

    missing = set(feature_columns)
    out: dict[str, float] = {}
    # Fallback approximations from the already-built Turbo edge snapshot.
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
    for key, value in market.values.items():
        if key in feature_columns:
            _put_if_available(out, key, value, missing)
    if "long_mtf_agreement" not in out or "short_mtf_agreement" not in out or "mtf_conflict" not in out:
        ema_slope = out.get("mtf_1h_ema_9_slope", out.get("ema_9_slope", 0.0))
        out["mtf_15m_trend_direction"] = out.get("mtf_15m_trend_direction", 1.0 if ema_slope > 0 else (-1.0 if ema_slope < 0 else 0.0))
        out["mtf_1h_trend_direction"] = out.get("mtf_1h_trend_direction", out["mtf_15m_trend_direction"])
        out["long_mtf_agreement"] = 1.0 if out.get("momentum_3", 0.0) > 0 and out["mtf_15m_trend_direction"] > 0 and out["mtf_1h_trend_direction"] >= 0 else 0.0
        out["short_mtf_agreement"] = 1.0 if out.get("momentum_3", 0.0) < 0 and out["mtf_15m_trend_direction"] < 0 and out["mtf_1h_trend_direction"] <= 0 else 0.0
        five_min_side = np.sign(out.get("momentum_6", 0.0))
        mtf_side = np.sign(out["mtf_15m_trend_direction"] + out["mtf_1h_trend_direction"])
        out["mtf_conflict"] = 1.0 if five_min_side != 0 and mtf_side != 0 and five_min_side != mtf_side else 0.0
    for key in ("mtf_15m_trend_direction", "mtf_1h_trend_direction", "long_mtf_agreement", "short_mtf_agreement", "mtf_conflict"):
        if np.isfinite(out.get(key, np.nan)):
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
    feature_status, reason, parity_pct, critical_missing = _grade_features(out, feature_columns, missing)
    base = EntryQualityFeatureVector(
        vector,
        feature_columns,
        sorted(missing),
        feature_status,
        reason,
        str(feature_timestamp) if feature_timestamp else None,
        parity_pct,
        list(market.approximated_features),
        critical_missing,
        (time.perf_counter() - start) * 1000,
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
    status, reason, parity_pct, critical_missing = _grade_features(values, base.feature_columns, missing)
    if "turbo_scores_votes" in critical_missing:
        reason = "missing_turbo_context_features"
    _LAST_FINAL_STATUS_BY_SYMBOL[symbol] = status
    _LAST_FINAL_PARITY_BY_SYMBOL[symbol] = parity_pct
    _LAST_FINAL_MISSING_BY_SYMBOL[symbol] = missing_list
    return EntryQualityFeatureVector(
        vector,
        base.feature_columns,
        missing_list,
        status,
        reason,
        base.feature_timestamp,
        parity_pct,
        list(base.approximated_features or []),
        critical_missing,
        base.feature_build_latency_ms,
    )


def _to_vector(values: dict[str, float], feature_columns: list[str], missing: set[str]) -> np.ndarray:
    row = []
    for col in feature_columns:
        value = values.get(col, np.nan)
        if np.isfinite(value):
            missing.discard(col)
        row.append(value)
    return np.asarray([row], dtype=np.float32)


def entry_quality_feature_cache_status() -> dict[str, Any]:
    market_status = runtime_feature_cache_status()
    missing_counter: dict[str, int] = {}
    last_build_latency = 0.0
    for missing_list in _LAST_FINAL_MISSING_BY_SYMBOL.values():
        for missing in missing_list:
            missing_counter[missing] = missing_counter.get(missing, 0) + 1
    for (_symbol, _timestamp), (_cached_at, vector) in _FEATURE_CACHE.items():
        last_build_latency = max(last_build_latency, vector.feature_build_latency_ms)
    missing_top = sorted(missing_counter.items(), key=lambda item: item[1], reverse=True)[:20]
    return {
        "feature_vector_cache_size": len(_FEATURE_CACHE),
        "ttl_seconds": FEATURE_CACHE_TTL_SECONDS,
        "cache_ttl_seconds": FEATURE_CACHE_TTL_SECONDS,
        "last_feature_status_by_symbol": dict(_LAST_FINAL_STATUS_BY_SYMBOL),
        "last_feature_parity_pct_by_symbol": dict(_LAST_FINAL_PARITY_BY_SYMBOL),
        "missing_features_top": [{"feature": feature, "count": count} for feature, count in missing_top],
        "last_feature_build_latency_ms": round(float(last_build_latency), 3),
        **market_status,
    }


def clear_entry_quality_feature_cache() -> None:
    _FEATURE_CACHE.clear()
    _LAST_FINAL_STATUS_BY_SYMBOL.clear()
    _LAST_FINAL_PARITY_BY_SYMBOL.clear()
    _LAST_FINAL_MISSING_BY_SYMBOL.clear()
