from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from aegis_alpha.config import REPO_ROOT
from aegis_alpha.entry_quality.runtime_feature_cache import get_runtime_market_features


LATEST_NEWS_PATH = REPO_ROOT / "aegis_alpha/data/processed/event_risk/latest_event_sentiment_risk.json"
EVENT_RISK_YAML_PATH = REPO_ROOT / "binance-futures-bot-ts/regime_config.live.yaml"
NEWS_CACHE_TTL_SECONDS = 60.0
YAML_CACHE_TTL_SECONDS = 60.0
MODE_TO_SCORE = {
    "UNKNOWN": -1.0,
    "NORMAL": 0.0,
    "CAUTION": 1.0,
    "RISK_OFF": 2.0,
    "MANUAL_ONLY": 3.0,
}
ACTION_TO_SCORE = {
    "UNKNOWN": -1.0,
    "HOLD": 0.0,
    "LONG": 1.0,
    "SHORT": -1.0,
}
SIDE_TO_SCORE = {
    "UNKNOWN": 0.0,
    "HOLD": 0.0,
    "LONG": 1.0,
    "SHORT": -1.0,
}
RECOMMENDATION_TO_SCORE = {
    "UNKNOWN": -1.0,
    "ALLOW_SHADOW": 1.0,
    "BLOCK_SHADOW": -1.0,
    "INSUFFICIENT_DATA": -0.5,
    "MODEL_ERROR": -0.5,
}

_NEWS_CACHE: dict[str, Any] = {
    "loaded_at": 0.0,
    "payload": None,
    "error": None,
}
_YAML_CACHE: dict[str, Any] = {
    "loaded_at": 0.0,
    "payload": None,
    "error": None,
}

FEATURE_GROUPS: dict[str, list[str]] = {
    "turbo": [
        "eq_long_score_7d",
        "eq_long_score_14d",
        "eq_long_score_30d",
        "eq_short_score_7d",
        "eq_short_score_14d",
        "eq_short_score_30d",
        "eq_votes_long",
        "eq_votes_short",
        "eq_votes_neutral",
        "eq_turbo_score",
        "eq_score_gap",
        "turbo_action_score",
    ],
    "entry_quality": [
        "entry_quality_score",
        "tail_risk_score",
        "entry_quality_recommendation_score",
        "rule_based_shadow_block",
    ],
    "market_mtf": [
        "eq_entry_price",
        "eq_ret_1",
        "eq_ret_2",
        "eq_ret_3",
        "eq_ret_6",
        "eq_ret_12",
        "eq_momentum_3",
        "eq_momentum_6",
        "eq_momentum_12",
        "eq_candle_body_pct",
        "eq_upper_wick_pct",
        "eq_lower_wick_pct",
        "eq_green_candle_count_3",
        "eq_red_candle_count_3",
        "eq_ema_9",
        "eq_ema_21",
        "eq_ema_50",
        "eq_price_to_ema_9",
        "eq_price_to_ema_21",
        "eq_ema_9_slope",
        "eq_ema_21_slope",
        "eq_atr_14",
        "eq_atr_pct",
        "eq_realized_vol_12",
        "eq_realized_vol_36",
        "eq_high_low_range_pct",
        "eq_atr_percentile_7d",
        "eq_volume_zscore_36",
        "eq_volume_ratio_12",
        "eq_quote_volume",
        "eq_mtf_15m_ret_1",
        "eq_mtf_15m_ret_2",
        "eq_mtf_15m_ema_9_slope",
        "eq_mtf_15m_price_to_ema_21",
        "eq_mtf_15m_atr_pct",
        "eq_mtf_15m_trend_direction",
        "eq_mtf_1h_ret_1",
        "eq_mtf_1h_ret_2",
        "eq_mtf_1h_ema_9_slope",
        "eq_mtf_1h_price_to_ema_21",
        "eq_mtf_1h_atr_pct",
        "eq_mtf_1h_trend_direction",
        "eq_long_mtf_agreement",
        "eq_short_mtf_agreement",
        "eq_mtf_conflict",
    ],
    "event_risk": [
        "event_risk_mode_score",
        "event_risk_auto_mode_score",
        "event_risk_auto_confidence",
        "event_risk_auto_reason_count",
        "event_risk_mode_normal",
        "event_risk_mode_caution",
        "event_risk_mode_risk_off",
        "event_risk_mode_manual_only",
        "event_risk_mode_unknown",
        "event_risk_auto_mode_normal",
        "event_risk_auto_mode_caution",
        "event_risk_auto_mode_risk_off",
        "event_risk_auto_mode_manual_only",
        "event_risk_auto_mode_unknown",
    ],
    "news_sentiment": [
        "news_sentiment_mode_score",
        "news_sentiment_risk_score",
        "news_sentiment_confidence",
        "fear_greed_value",
        "news_sentiment_mode_normal",
        "news_sentiment_mode_caution",
        "news_sentiment_mode_risk_off",
        "news_sentiment_mode_manual_only",
        "news_sentiment_mode_unknown",
    ],
    "btc_eth_context": [
        "btc_action_score",
        "btc_score",
        "btc_tail_risk_score",
        "eth_action_score",
        "eth_score",
        "eth_tail_risk_score",
        "btc_eth_confirm_direction",
    ],
    "portfolio": [
        "open_positions_count",
        "same_direction_positions",
        "wallet_balance",
        "available_balance",
        "margin_used_ratio",
        "daily_pnl_pct",
        "consecutive_losses",
        "recent_closed_pnl",
        "last_trade_outcome_same_symbol",
        "portfolio_context_available",
    ],
    "symbol_side": ["side_score", "source_type_score"],
    "time": ["hour_sin", "hour_cos", "day_sin", "day_cos", "session_score"],
    "availability": ["context_available", "event_risk_auto_available", "news_sentiment_available"],
}
CRITICAL_GROUPS = {"turbo", "symbol_side", "entry_quality", "market_mtf"}


def normalize_symbol(symbol: str) -> str:
    return str(symbol or "").strip().upper().replace("/", "").replace("-", "")


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(np.asarray(value).item())
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _score_gap(recent_scores: dict[str, Any]) -> float:
    long_scores = [
        safe_float(recent_scores.get("long_7d")),
        safe_float(recent_scores.get("long_14d")),
        safe_float(recent_scores.get("long_30d")),
    ]
    short_scores = [
        safe_float(recent_scores.get("short_7d")),
        safe_float(recent_scores.get("short_14d")),
        safe_float(recent_scores.get("short_30d")),
    ]
    long_finite = [item for item in long_scores if math.isfinite(item)]
    short_finite = [item for item in short_scores if math.isfinite(item)]
    if not long_finite or not short_finite:
        return math.nan
    return float(max(long_finite) - max(short_finite))


def _mode_score(mode: Any) -> float:
    return MODE_TO_SCORE.get(str(mode or "UNKNOWN").upper(), -1.0)


def _action_score(action: Any) -> float:
    return ACTION_TO_SCORE.get(str(action or "UNKNOWN").upper(), -1.0)


def _side_score(side: Any) -> float:
    return SIDE_TO_SCORE.get(str(side or "UNKNOWN").upper(), 0.0)


def _recommendation_score(value: Any) -> float:
    return RECOMMENDATION_TO_SCORE.get(str(value or "UNKNOWN").upper(), -1.0)


def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        try:
            return datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        except ValueError:
            return None


def _time_features(ts: datetime | None) -> tuple[float, float, float, float, float]:
    if ts is None:
        ts = datetime.now(timezone.utc)
    hour = ts.hour + ts.minute / 60.0
    dow = float(ts.weekday())
    if 0 <= ts.hour < 8:
        session = 0.0
    elif 8 <= ts.hour < 13:
        session = 1.0
    elif 13 <= ts.hour < 21:
        session = 2.0
    else:
        session = 3.0
    return (
        math.sin(2.0 * math.pi * hour / 24.0),
        math.cos(2.0 * math.pi * hour / 24.0),
        math.sin(2.0 * math.pi * dow / 7.0),
        math.cos(2.0 * math.pi * dow / 7.0),
        session,
    )


def load_latest_news_sentiment() -> dict[str, Any] | None:
    now = time.time()
    if now - float(_NEWS_CACHE.get("loaded_at") or 0.0) < NEWS_CACHE_TTL_SECONDS:
        return _NEWS_CACHE.get("payload")
    try:
        if not LATEST_NEWS_PATH.exists():
            _NEWS_CACHE.update({"loaded_at": now, "payload": None, "error": "missing"})
            return None
        payload = json.loads(LATEST_NEWS_PATH.read_text(encoding="utf-8"))
        _NEWS_CACHE.update({"loaded_at": now, "payload": payload, "error": None})
        return payload
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        _NEWS_CACHE.update({"loaded_at": now, "payload": None, "error": repr(exc)})
        return None


def _load_event_risk_overlay_config() -> dict[str, Any]:
    now = time.time()
    if now - float(_YAML_CACHE.get("loaded_at") or 0.0) < YAML_CACHE_TTL_SECONDS:
        return dict(_YAML_CACHE.get("payload") or {})
    payload: dict[str, Any] = {}
    try:
        if not EVENT_RISK_YAML_PATH.exists():
            _YAML_CACHE.update({"loaded_at": now, "payload": payload, "error": "missing"})
            return payload
        try:
            import yaml  # type: ignore

            raw = yaml.safe_load(EVENT_RISK_YAML_PATH.read_text(encoding="utf-8")) or {}
            event_risk = (((raw.get("aegis") or {}).get("event_risk")) or {})
            payload = event_risk if isinstance(event_risk, dict) else {}
        except Exception:
            payload = _parse_event_risk_yaml_fallback(EVENT_RISK_YAML_PATH.read_text(encoding="utf-8"))
        _YAML_CACHE.update({"loaded_at": now, "payload": payload, "error": None})
        return dict(payload)
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        _YAML_CACHE.update({"loaded_at": now, "payload": {}, "error": repr(exc)})
        return {}


def _parse_event_risk_yaml_fallback(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    in_event_risk = False
    base_indent: int | None = None
    out: dict[str, Any] = {}
    for raw_line in lines:
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))
        if stripped == "event_risk:":
            in_event_risk = True
            base_indent = indent
            continue
        if not in_event_risk or base_indent is None:
            continue
        if indent <= base_indent:
            break
        if indent == base_indent + 2 and ":" in stripped:
            key, value = stripped.split(":", 1)
            value = value.strip()
            if value == "":
                continue
            out[key.strip()] = _parse_scalar(value)
    return out


def _parse_scalar(value: str) -> Any:
    normalized = value.strip().strip("'\"")
    lower = normalized.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"null", "none", "~"}:
        return None
    number = safe_float(normalized)
    return number if math.isfinite(number) else normalized


def _fear_greed_value(news: dict[str, Any] | None) -> float:
    if not isinstance(news, dict):
        return math.nan
    for event in news.get("top_events") or []:
        title = str((event or {}).get("title") or "")
        if "Fear & Greed Index" not in title:
            continue
        for part in reversed(title.split()):
            value = safe_float(part)
            if math.isfinite(value):
                return value
    return math.nan


def _set_mode_onehots(features: dict[str, float], prefix: str, mode: str) -> None:
    normalized = str(mode or "UNKNOWN").upper()
    for value in ("NORMAL", "CAUTION", "RISK_OFF", "MANUAL_ONLY", "UNKNOWN"):
        features[f"{prefix}_{value.lower()}"] = 1.0 if normalized == value else 0.0


def _put_if_finite(features: dict[str, float], key: str, value: Any) -> bool:
    number = safe_float(value)
    if math.isfinite(number):
        features[key] = number
        return True
    return False


def _merge_market_features(features: dict[str, float], symbol: str) -> tuple[list[str], list[str]]:
    approximated: list[str] = []
    warnings: list[str] = []
    market = get_runtime_market_features(symbol)
    if market.approximated_features:
        approximated.extend([f"eq_{item}" for item in market.approximated_features])
    if market.warnings:
        warnings.extend(market.warnings)
    values = market.values or {}
    mapping = {
        "eq_entry_price": "close",
        "eq_ret_1": "ret_1",
        "eq_ret_2": "ret_2",
        "eq_ret_3": "ret_3",
        "eq_ret_6": "ret_6",
        "eq_ret_12": "ret_12",
        "eq_momentum_3": "momentum_3",
        "eq_momentum_6": "momentum_6",
        "eq_momentum_12": "momentum_12",
        "eq_candle_body_pct": "candle_body_pct",
        "eq_upper_wick_pct": "upper_wick_pct",
        "eq_lower_wick_pct": "lower_wick_pct",
        "eq_green_candle_count_3": "green_candle_count_3",
        "eq_red_candle_count_3": "red_candle_count_3",
        "eq_ema_9": "ema_9",
        "eq_ema_21": "ema_21",
        "eq_ema_50": "ema_50",
        "eq_price_to_ema_9": "price_to_ema_9",
        "eq_price_to_ema_21": "price_to_ema_21",
        "eq_ema_9_slope": "ema_9_slope",
        "eq_ema_21_slope": "ema_21_slope",
        "eq_atr_14": "atr_14",
        "eq_atr_pct": "atr_pct",
        "eq_realized_vol_12": "realized_vol_12",
        "eq_realized_vol_36": "realized_vol_36",
        "eq_high_low_range_pct": "high_low_range_pct",
        "eq_atr_percentile_7d": "atr_percentile_7d",
        "eq_volume_zscore_36": "volume_zscore_36",
        "eq_volume_ratio_12": "volume_ratio_12",
        "eq_quote_volume": "quote_volume",
        "eq_mtf_15m_ret_1": "mtf_15m_ret_1",
        "eq_mtf_15m_ret_2": "mtf_15m_ret_2",
        "eq_mtf_15m_ema_9_slope": "mtf_15m_ema_9_slope",
        "eq_mtf_15m_price_to_ema_21": "mtf_15m_price_to_ema_21",
        "eq_mtf_15m_atr_pct": "mtf_15m_atr_pct",
        "eq_mtf_15m_trend_direction": "mtf_15m_trend_direction",
        "eq_mtf_1h_ret_1": "mtf_1h_ret_1",
        "eq_mtf_1h_ret_2": "mtf_1h_ret_2",
        "eq_mtf_1h_ema_9_slope": "mtf_1h_ema_9_slope",
        "eq_mtf_1h_price_to_ema_21": "mtf_1h_price_to_ema_21",
        "eq_mtf_1h_atr_pct": "mtf_1h_atr_pct",
        "eq_mtf_1h_trend_direction": "mtf_1h_trend_direction",
        "eq_long_mtf_agreement": "long_mtf_agreement",
        "eq_short_mtf_agreement": "short_mtf_agreement",
        "eq_mtf_conflict": "mtf_conflict",
    }
    for target, source in mapping.items():
        _put_if_finite(features, target, values.get(source))
    return sorted(set(approximated)), warnings


def _portfolio_neutral_features(side: str) -> tuple[dict[str, float], list[str]]:
    _ = side
    return {
        "open_positions_count": 0.0,
        "same_direction_positions": 0.0,
        "wallet_balance": 0.0,
        "available_balance": 0.0,
        "margin_used_ratio": 0.0,
        "daily_pnl_pct": 0.0,
        "consecutive_losses": 0.0,
        "recent_closed_pnl": 0.0,
        "last_trade_outcome_same_symbol": 0.0,
        "portfolio_context_available": 0.0,
    }, [
        "portfolio_context_neutral",
        "open_positions_count_neutral",
        "same_direction_positions_neutral",
        "wallet_balance_neutral",
        "available_balance_neutral",
        "margin_used_ratio_neutral",
        "daily_pnl_pct_neutral",
        "consecutive_losses_neutral",
        "recent_closed_pnl_neutral",
        "last_trade_outcome_same_symbol_neutral",
    ]


def _group_coverage(features: dict[str, float], feature_columns: list[str]) -> dict[str, float]:
    expected_set = set(feature_columns)
    coverage: dict[str, float] = {}
    for group, group_columns in FEATURE_GROUPS.items():
        expected = [item for item in group_columns if item in expected_set]
        if not expected:
            continue
        available = sum(1 for item in expected if math.isfinite(safe_float(features.get(item))))
        coverage[group] = round(100.0 * available / max(len(expected), 1), 3)
    return coverage


def _missing_by_group(missing: list[str]) -> dict[str, list[str]]:
    missing_set = set(missing)
    grouped: dict[str, list[str]] = {}
    matched: set[str] = set()
    for group, group_columns in FEATURE_GROUPS.items():
        items = sorted(missing_set.intersection(group_columns))
        if items:
            grouped[group] = items
            matched.update(items)
    other = sorted(missing_set - matched)
    if other:
        grouped["other"] = other
    return grouped


def build_decision_brain_features(
    *,
    symbol: str,
    side: str | None,
    turbo_context: dict[str, Any] | None,
    entry_quality_model: dict[str, Any] | None,
    event_risk_auto: dict[str, Any] | None,
    news_sentiment: dict[str, Any] | None,
    feature_columns: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    start = time.perf_counter()
    normalized_symbol = normalize_symbol(symbol)
    turbo = turbo_context if isinstance(turbo_context, dict) else {}
    raw = turbo.get("raw") if isinstance(turbo.get("raw"), dict) else {}
    gated = turbo.get("gated") if isinstance(turbo.get("gated"), dict) else {}
    votes = raw.get("votes") or turbo.get("votes") or {}
    recent_scores = raw.get("recent_scores") or turbo.get("recent_scores") or {}
    freshness = turbo.get("freshness") or raw.get("freshness") or {}
    eq = entry_quality_model if isinstance(entry_quality_model, dict) else turbo.get("entry_quality_model")
    eq = eq if isinstance(eq, dict) else {}
    event_auto = event_risk_auto if isinstance(event_risk_auto, dict) else turbo.get("event_risk_auto")
    event_auto = event_auto if isinstance(event_auto, dict) else {}
    news = news_sentiment if isinstance(news_sentiment, dict) else load_latest_news_sentiment()
    news = news if isinstance(news, dict) else {}
    overlay = _load_event_risk_overlay_config()
    approximated_features: list[str] = []
    feature_warnings: list[str] = []

    turbo_action = raw.get("action") or gated.get("action") or turbo.get("action") or side or "UNKNOWN"
    resolved_side = str(side or turbo_action or "UNKNOWN").upper()
    if resolved_side == "HOLD":
        resolved_side = str(raw.get("action") or turbo.get("action") or "UNKNOWN").upper()
    event_auto_mode = str(event_auto.get("suggested_mode") or "UNKNOWN").upper()
    event_overlay_mode = str(
        turbo.get("event_risk_mode")
        or turbo.get("event_risk_overlay_mode")
        or overlay.get("mode")
        or "UNKNOWN"
    ).upper()
    news_mode = str(news.get("suggested_mode") or "UNKNOWN").upper()
    btc = event_auto.get("btc_context") if isinstance(event_auto.get("btc_context"), dict) else {}
    eth = event_auto.get("eth_context") if isinstance(event_auto.get("eth_context"), dict) else {}
    btc_action = btc.get("turbo_action") or btc.get("gated_action") or "UNKNOWN"
    eth_action = eth.get("turbo_action") or eth.get("gated_action") or "UNKNOWN"
    ts = _parse_dt(freshness.get("feature_timestamp") or turbo.get("timestamp"))
    hour_sin, hour_cos, day_sin, day_cos, session = _time_features(ts)

    features: dict[str, float] = {
        "eq_leverage": safe_float(turbo.get("leverage_suggestion") or raw.get("leverage_suggestion")),
        "eq_long_score_7d": safe_float(recent_scores.get("long_7d")),
        "eq_long_score_14d": safe_float(recent_scores.get("long_14d")),
        "eq_long_score_30d": safe_float(recent_scores.get("long_30d")),
        "eq_short_score_7d": safe_float(recent_scores.get("short_7d")),
        "eq_short_score_14d": safe_float(recent_scores.get("short_14d")),
        "eq_short_score_30d": safe_float(recent_scores.get("short_30d")),
        "eq_votes_long": safe_float(votes.get("long")),
        "eq_votes_short": safe_float(votes.get("short")),
        "eq_votes_neutral": safe_float(votes.get("neutral")),
        "eq_turbo_score": safe_float(raw.get("turbo_score", turbo.get("turbo_score"))),
        "eq_score_gap": _score_gap(recent_scores),
        "source_type_score": 1.0,
        "side_score": _side_score(resolved_side),
        "turbo_action_score": _action_score(turbo_action),
        "entry_quality_score": safe_float(eq.get("entry_quality_score")),
        "tail_risk_score": safe_float(eq.get("tail_risk_score")),
        "entry_quality_recommendation_score": _recommendation_score(eq.get("recommendation")),
        "rule_based_shadow_block": 1.0 if str(eq.get("recommendation") or "").upper() == "BLOCK_SHADOW" else 0.0,
        "event_risk_mode_score": _mode_score(event_overlay_mode),
        "event_risk_auto_mode_score": _mode_score(event_auto_mode),
        "event_risk_auto_confidence": safe_float(event_auto.get("confidence")),
        "event_risk_auto_reason_count": float(len(event_auto.get("reasons") or [])),
        "news_sentiment_mode_score": _mode_score(news_mode),
        "news_sentiment_risk_score": safe_float(news.get("risk_score")),
        "news_sentiment_confidence": safe_float(news.get("confidence")),
        "fear_greed_value": _fear_greed_value(news),
        "btc_action_score": _action_score(btc_action),
        "btc_score": safe_float(btc.get("turbo_score"), 0.0),
        "btc_tail_risk_score": safe_float(btc.get("tail_risk_score"), 0.0),
        "eth_action_score": _action_score(eth_action),
        "eth_score": safe_float(eth.get("turbo_score"), 0.0),
        "eth_tail_risk_score": safe_float(eth.get("tail_risk_score"), 0.0),
        "btc_eth_confirm_direction": 1.0 if (
            (resolved_side == "LONG" and str(btc_action).upper() == "LONG" and str(eth_action).upper() == "LONG")
            or (resolved_side == "SHORT" and str(btc_action).upper() == "SHORT" and str(eth_action).upper() == "SHORT")
        ) else 0.0,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "day_sin": day_sin,
        "day_cos": day_cos,
        "session_score": session,
        "context_available": 1.0 if turbo else 0.0,
        "event_risk_auto_available": 1.0 if event_auto else 0.0,
        "news_sentiment_available": 1.0 if news else 0.0,
        "portfolio_context_available": 0.0,
    }
    market_approximated, market_warnings = _merge_market_features(features, normalized_symbol)
    approximated_features.extend(market_approximated)
    feature_warnings.extend(market_warnings)
    portfolio_features, portfolio_approximated = _portfolio_neutral_features(resolved_side)
    features.update(portfolio_features)
    approximated_features.extend(portfolio_approximated)
    _set_mode_onehots(features, "event_risk_mode", event_overlay_mode)
    _set_mode_onehots(features, "event_risk_auto_mode", event_auto_mode)
    _set_mode_onehots(features, "news_sentiment_mode", news_mode)

    vector: list[float] = []
    missing: list[str] = []
    for column in feature_columns:
        value = features.get(column, math.nan)
        if not math.isfinite(safe_float(value)):
            missing.append(column)
        vector.append(safe_float(value))

    present_count = len(feature_columns) - len(missing)
    parity_pct = round(100.0 * present_count / max(len(feature_columns), 1), 3)
    group_coverage = _group_coverage(features, feature_columns)
    critical_missing: list[str] = []
    if not turbo:
        critical_missing.append("turbo_context")
    if normalized_symbol == "" or resolved_side in {"", "UNKNOWN"}:
        critical_missing.append("symbol_side")
    if not eq:
        critical_missing.append("entry_quality_model")
    if group_coverage.get("market_mtf", 0.0) < 60.0:
        critical_missing.append("basic_market_context")
    for group in sorted(CRITICAL_GROUPS):
        if group_coverage.get(group, 0.0) < 60.0 and group not in {"market_mtf"}:
            critical_missing.append(group)
    critical_missing = sorted(set(critical_missing))
    if parity_pct >= 95.0 and not critical_missing:
        status = "ok"
    elif parity_pct >= 75.0 and not critical_missing:
        status = "partial"
    else:
        status = "insufficient"
    available_feature_groups = sorted(group for group, coverage in group_coverage.items() if coverage >= 60.0)
    missing_by_group = _missing_by_group(missing)
    metadata = {
        "symbol": normalized_symbol,
        "side": resolved_side,
        "feature_status": status,
        "feature_parity_pct": parity_pct,
        "missing_features_count": len(missing),
        "missing_features": missing[:80],
        "missing_features_by_group": missing_by_group,
        "critical_missing_groups": critical_missing,
        "available_feature_groups": available_feature_groups,
        "feature_group_coverage_pct": group_coverage,
        "approximated_features": sorted(set(approximated_features))[:120],
        "feature_warnings": feature_warnings[:20],
        "feature_build_latency_ms": round((time.perf_counter() - start) * 1000, 3),
    }
    return np.asarray(vector, dtype=np.float32).reshape(1, -1), metadata
