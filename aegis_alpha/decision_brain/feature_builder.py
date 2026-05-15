from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


LATEST_NEWS_PATH = Path("aegis_alpha/data/processed/event_risk/latest_event_sentiment_risk.json")
NEWS_CACHE_TTL_SECONDS = 60.0
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

    turbo_action = raw.get("action") or gated.get("action") or turbo.get("action") or side or "UNKNOWN"
    resolved_side = str(side or turbo_action or "UNKNOWN").upper()
    if resolved_side == "HOLD":
        resolved_side = str(raw.get("action") or turbo.get("action") or "UNKNOWN").upper()
    event_auto_mode = str(event_auto.get("suggested_mode") or "UNKNOWN").upper()
    event_overlay_mode = str(turbo.get("event_risk_mode") or "UNKNOWN").upper()
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
        "btc_score": safe_float(btc.get("turbo_score")),
        "btc_tail_risk_score": safe_float(btc.get("tail_risk_score")),
        "eth_action_score": _action_score(eth_action),
        "eth_score": safe_float(eth.get("turbo_score")),
        "eth_tail_risk_score": safe_float(eth.get("tail_risk_score")),
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
    critical_missing: list[str] = []
    if not turbo:
        critical_missing.append("turbo_context")
    if normalized_symbol == "" or resolved_side in {"", "UNKNOWN", "HOLD"}:
        critical_missing.append("symbol_side")
    if not eq:
        critical_missing.append("entry_quality_model")
    if not any(math.isfinite(features.get(name, math.nan)) for name in ("eq_ret_1", "eq_atr_pct", "eq_price_to_ema_21", "eq_atr_percentile_7d")):
        critical_missing.append("basic_market_context")
    if parity_pct >= 95.0 and not critical_missing:
        status = "ok"
    elif parity_pct >= 70.0 and not critical_missing:
        status = "partial"
    else:
        status = "insufficient"
    metadata = {
        "symbol": normalized_symbol,
        "side": resolved_side,
        "feature_status": status,
        "feature_parity_pct": parity_pct,
        "missing_features_count": len(missing),
        "missing_features": missing[:80],
        "critical_missing_groups": critical_missing,
        "feature_build_latency_ms": round((time.perf_counter() - start) * 1000, 3),
    }
    return np.asarray(vector, dtype=np.float32).reshape(1, -1), metadata
