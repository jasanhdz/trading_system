from __future__ import annotations

import logging
import os
import json
import time
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from aegis_alpha.signals.common import load_signal_market
from aegis_alpha.turbo.config import DEFAULT_TURBO_CONFIG, get_runtime_turbo_config
from aegis_alpha.turbo.snapshot_utils import (
    TURBO_MAX_FEATURE_AGE_SECONDS,
    load_turbo_snapshot_status,
    normalize_turbo_symbol,
    turbo_symbol_model_dir,
    turbo_snapshot_path,
)
from aegis_alpha.turbo.turbo_risk import build_turbo_risk_status
from aegis_alpha.turbo.turbo_schema import build_turbo_signal


LOGGER = logging.getLogger(__name__)
TURBO_ALLOW_MARKET_REBUILD = False
_MODEL_CACHE: dict[str, tuple[int | None, Any | None]] = {}
_STALE_LOG_LAST_EMITTED: dict[str, float] = {}
LAST_TURBO_RUNTIME: dict[str, Any] = {
    "reason": None,
    "turbo_score": None,
    "freshness": None,
    "snapshot_path": None,
}
LAST_TURBO_RUNTIME_BY_SYMBOL: dict[str, dict[str, Any]] = {}


def _model_path(side: str, lookback_days: int, symbol: str) -> Path:
    symbol_dir = turbo_symbol_model_dir(symbol)
    manifest_path = symbol_dir / "active_manifest.json"
    manifest_key = f"{side}_{lookback_days}d"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            model_paths = manifest.get("model_paths") if isinstance(manifest, dict) else None
            path_text = model_paths.get(manifest_key) if isinstance(model_paths, dict) else None
            if path_text:
                candidate = Path(path_text)
                if not candidate.is_absolute():
                    candidate = Path.cwd() / candidate
                if candidate.exists():
                    return candidate
        except Exception as exc:
            LOGGER.warning("aegis_turbo_active_manifest_ignored path=%s error=%r", manifest_path, exc)

    active_path = symbol_dir / "active" / f"turbo_{side}_edge_{lookback_days}d_v010.joblib"
    if active_path.exists():
        return active_path

    legacy_path = DEFAULT_TURBO_CONFIG.model_dir / f"turbo_{side}_edge_{lookback_days}d_v010.joblib"
    if normalize_turbo_symbol(symbol) == normalize_turbo_symbol(DEFAULT_TURBO_CONFIG.symbol) and legacy_path.exists():
        return legacy_path
    return symbol_dir / f"turbo_{side}_edge_{lookback_days}d_v010.joblib"


def _load_estimator(path_text: str) -> Any | None:
    path = Path(path_text)
    if not path.exists():
        _MODEL_CACHE[path_text] = (None, None)
        return None
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        mtime_ns = None
    cached = _MODEL_CACHE.get(path_text)
    if cached is not None and cached[0] == mtime_ns:
        return cached[1]
    bundle = joblib.load(path)
    if isinstance(bundle, dict):
        estimator = bundle.get("estimator")
    else:
        estimator = bundle
    _MODEL_CACHE[path_text] = (mtime_ns, estimator)
    LOGGER.warning("aegis_turbo_model_loaded path=%s loaded=%s", path.name, estimator is not None)
    return estimator


def model_cache_keys() -> list[str]:
    return sorted(_MODEL_CACHE.keys())


def runtime_symbols() -> list[str]:
    symbols = set(LAST_TURBO_RUNTIME_BY_SYMBOL.keys())
    symbols.add(normalize_turbo_symbol(DEFAULT_TURBO_CONFIG.symbol))
    return sorted(symbols)


def runtime_status_by_symbol() -> dict[str, dict[str, Any]]:
    return {symbol: dict(status) for symbol, status in LAST_TURBO_RUNTIME_BY_SYMBOL.items()}


def _update_runtime(symbol: str, payload: dict[str, Any]) -> None:
    normalized = normalize_turbo_symbol(symbol)
    LAST_TURBO_RUNTIME.update(payload)
    LAST_TURBO_RUNTIME_BY_SYMBOL[normalized] = dict(payload)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(np.asarray(value).item())
    except Exception:
        return None


def _cache_bucket(ttl_seconds: int = 60) -> int:
    return int(time.time() // ttl_seconds)


def _utc_iso(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat()
    except Exception:
        return None


def _stale_reason(rate_limited_key: str, freshness: dict[str, Any]) -> None:
    last = _STALE_LOG_LAST_EMITTED.get(rate_limited_key, 0.0)
    now = time.time()
    if now - last < 600:
        return
    _STALE_LOG_LAST_EMITTED[rate_limited_key] = now
    LOGGER.warning(
        "turbo_snapshot_stale_blocked path=%s feature_timestamp=%s feature_age_seconds=%s max_feature_age_seconds=%s",
        freshness.get("path"),
        freshness.get("feature_timestamp"),
        freshness.get("feature_age_seconds"),
        freshness.get("max_feature_age_seconds"),
    )


def _snapshot_candidates(symbol: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for lookback_days in DEFAULT_TURBO_CONFIG.lookback_days:
        path = turbo_snapshot_path(int(lookback_days), symbol)
        status = load_turbo_snapshot_status(path, include_sample_count=False)
        status["lookback_days"] = int(lookback_days)
        candidates.append(status)
    return candidates


def _select_turbo_snapshot(symbol: str) -> dict[str, Any]:
    candidates = [candidate for candidate in _snapshot_candidates(symbol) if candidate.get("exists")]
    if not candidates:
        return {
            "path": None,
            "freshness": {
                "path": None,
                "exists": False,
                "snapshot_mtime": None,
                "snapshot_age_seconds": None,
                "feature_timestamp": None,
                "feature_age_seconds": None,
                "max_feature_age_seconds": TURBO_MAX_FEATURE_AGE_SECONDS,
                "is_fresh": False,
                "sample_count": 0,
                "last_ts": None,
                "error": "missing_turbo_snapshot",
            },
        }
    candidates.sort(
        key=lambda item: (
            item.get("feature_timestamp") is not None,
            item.get("feature_timestamp") or "",
            item.get("snapshot_mtime") or "",
            -int(item.get("lookback_days") or 0),
        ),
        reverse=True,
    )
    selected = candidates[0]
    selected["path"] = selected.get("path") or str(turbo_snapshot_path(int(selected.get("lookback_days") or 7), symbol))
    return {"path": selected["path"], "freshness": selected}


@lru_cache(maxsize=4)
def _load_market_cached(bucket: int):
    return load_signal_market(DEFAULT_TURBO_CONFIG.config_path)


@lru_cache(maxsize=8)
def _safe_candidate_cached(symbol: str, bucket: int) -> dict[str, Any]:
    from aegis_alpha.inference.shadow_candidate import evaluate_shadow_candidate

    return evaluate_shadow_candidate(symbol)


def _safe_context(symbol: str, market_payload: dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(market_payload, dict) and "shadow" in market_payload:
        shadow = market_payload.get("shadow") or {}
        return {
            "regime": str(shadow.get("regime", "unknown")),
            "tail_risk_score": _safe_float(shadow.get("tail_risk_score")),
            "edge_h12": _safe_float(shadow.get("edge_score_h12")),
            "safe_action": str(shadow.get("action", "unknown")),
            "safe_reason": str(shadow.get("reason", "unknown")),
        }
    try:
        evaluated = _safe_candidate_cached(symbol, _cache_bucket())
        shadow = evaluated.get("shadow", {})
        return {
            "regime": str(shadow.get("regime", "unknown")),
            "tail_risk_score": _safe_float(shadow.get("tail_risk_score")),
            "edge_h12": _safe_float(shadow.get("edge_score_h12")),
            "safe_action": str(shadow.get("action", "unknown")),
            "safe_reason": str(shadow.get("reason", "unknown")),
        }
    except Exception as exc:
        LOGGER.exception("safe context unavailable for turbo")
        return {
            "regime": "unknown",
            "tail_risk_score": None,
            "edge_h12": None,
            "safe_action": "unknown",
            "safe_reason": f"safe_context_error: {exc!r}",
        }


def _current_feature(symbol: str) -> tuple[np.ndarray | None, str]:
    selected = _select_turbo_snapshot(symbol)
    path = selected.get("path")
    if path:
        data = np.load(path, allow_pickle=True)
        live_x = data.get("live_X")
        if live_x is not None and len(live_x) > 0:
            x = np.asarray(live_x, dtype=np.float32)
        else:
            x = np.asarray(data["X"], dtype=np.float32)
        feature_timestamp = data.get("feature_timestamp")
        if feature_timestamp is not None:
            if hasattr(feature_timestamp, "item"):
                feature_timestamp = feature_timestamp.item()
            return x[-1:].astype(np.float32), str(feature_timestamp)
        timestamps = data["timestamp"].astype(str)
        if len(x) > 0:
            return x[-1:].astype(np.float32), str(timestamps[-1])
    if not TURBO_ALLOW_MARKET_REBUILD:
        return None, ""
    market = load_signal_market(DEFAULT_TURBO_CONFIG.config_path, symbol_override=symbol)
    if len(market.signal_features) == 0:
        return None, ""
    # Aegis is currently single-symbol; use the configured market if the
    # requested symbol differs, but preserve the requested symbol in response.
    return market.signal_features[-1:].astype(np.float32), str(market.timestamps[market.steps[-1]])


def _score_models(x: np.ndarray, symbol: str) -> tuple[dict[str, float | None], dict[str, int]]:
    scores: dict[str, float | None] = {}
    votes = {"long": 0, "short": 0, "neutral": 0}
    for lookback_days in DEFAULT_TURBO_CONFIG.lookback_days:
        long_est = _load_estimator(str(_model_path("long", int(lookback_days), symbol)))
        short_est = _load_estimator(str(_model_path("short", int(lookback_days), symbol)))
        long_score = float(long_est.predict(x)[0]) if long_est is not None else None
        short_score = float(short_est.predict(x)[0]) if short_est is not None else None
        scores[f"long_{lookback_days}d"] = long_score
        scores[f"short_{lookback_days}d"] = short_score
        if long_score is None or short_score is None:
            votes["neutral"] += 1
        elif max(long_score, short_score) <= 0.0:
            votes["neutral"] += 1
        elif long_score >= short_score:
            votes["long"] += 1
        else:
            votes["short"] += 1
    return scores, votes


def _turbo_score(direction: str, scores: dict[str, float | None], votes: dict[str, int], safe_context: dict[str, Any]) -> float:
    agreement = votes.get(direction, 0) / max(len(DEFAULT_TURBO_CONFIG.lookback_days), 1)
    side_values = [
        float(value)
        for key, value in scores.items()
        if key.startswith(direction) and value is not None and np.isfinite(float(value))
    ]
    magnitude = max(side_values) if side_values else 0.0
    magnitude_score = float(np.clip(magnitude / 0.003, 0.0, 1.0))
    score = 0.45 * agreement + 0.45 * magnitude_score
    edge_h12 = safe_context.get("edge_h12")
    if edge_h12 is not None and direction == "long" and float(edge_h12) > 0.0:
        score += 0.05
    tail = safe_context.get("tail_risk_score")
    if tail is not None:
        score -= float(np.clip((float(tail) - 0.35) / 0.35, 0.0, 1.0)) * 0.20
    if safe_context.get("safe_reason") == "regime_block":
        score -= 0.15
    return float(np.clip(score, 0.0, 1.0))


def _raw_turbo_score(direction: str, scores: dict[str, float | None], votes: dict[str, int]) -> float:
    agreement = votes.get(direction, 0) / max(len(DEFAULT_TURBO_CONFIG.lookback_days), 1)
    side_values = [
        float(value)
        for key, value in scores.items()
        if key.startswith(direction) and value is not None and np.isfinite(float(value))
    ]
    magnitude = max(side_values) if side_values else 0.0
    magnitude_score = float(np.clip(magnitude / 0.003, 0.0, 1.0))
    return float(np.clip(0.85 * agreement + 0.15 * magnitude_score, 0.0, 1.0))


def _sizing(score: float) -> tuple[str, float, float]:
    cfg = get_runtime_turbo_config()
    if score >= cfg.thresholds.min_turbo_score_premium:
        return "premium", cfg.leverage.premium, cfg.position_fraction.premium
    if score >= cfg.thresholds.min_turbo_score_shadow:
        return "normal", cfg.leverage.normal, cfg.position_fraction.normal
    if cfg.thresholds.min_turbo_score_conservative > 0 and score >= cfg.thresholds.min_turbo_score_conservative:
        return "conservative", cfg.leverage.conservative, cfg.position_fraction.conservative
    return "blocked", 0.0, 0.0


def _raw_decision(scores: dict[str, float | None], votes: dict[str, int]) -> dict[str, Any]:
    cfg = get_runtime_turbo_config()
    direction = "long" if votes["long"] >= votes["short"] else "short"
    agreement_count = max(votes["long"], votes["short"])
    score = _raw_turbo_score(direction, scores, votes) if agreement_count else 0.0
    confidence, leverage, fraction = _sizing(score)
    base = {
        "action": "HOLD",
        "would_execute": False,
        "reason": "insufficient_recent_model_agreement",
        "turbo_score": score,
        "confidence": "blocked",
        "leverage_suggestion": 0.0,
        "position_fraction": 0.0,
        "votes": votes,
        "recent_scores": scores,
    }
    if agreement_count < cfg.thresholds.min_agreement_count:
        return base
    if direction == "short" and not cfg.thresholds.experimental_short:
        return {**base, "reason": "short_disabled_in_turbo_v010"}
    if confidence == "blocked":
        return {**base, "reason": "turbo_score_below_shadow_threshold"}
    return {
        "action": direction.upper(),
        "would_execute": True,
        "reason": f"raw_recent_{direction}_agreement_{agreement_count}_of_3",
        "turbo_score": score,
        "confidence": confidence,
        "leverage_suggestion": leverage,
        "position_fraction": fraction,
        "votes": votes,
        "recent_scores": scores,
    }


def _gated_decision(raw: dict[str, Any], risk_guard: dict[str, Any], safe_context: dict[str, Any]) -> dict[str, Any]:
    cfg = get_runtime_turbo_config()
    action = str(raw.get("action", "HOLD")).upper()
    if action not in {"LONG", "SHORT"} or not bool(raw.get("would_execute", False)):
        return {
            "action": action,
            "would_execute": False,
            "reason": str(raw.get("reason", "raw_hold")),
            "blocked_by": None,
        }
    if risk_guard.get("blocked"):
        return {
            "action": "HOLD",
            "would_execute": False,
            "reason": f"risk_guard_{risk_guard.get('reason')}",
            "blocked_by": "risk_guard",
        }
    if cfg.thresholds.block_if_safe_regime_toxic and safe_context.get("safe_reason") == "regime_block":
        return {
            "action": "HOLD",
            "would_execute": False,
            "reason": "safe_regime_block",
            "blocked_by": "safe_regime",
        }
    tail = safe_context.get("tail_risk_score")
    if tail is not None and float(tail) >= 0.50 and action == "LONG":
        return {
            "action": "HOLD",
            "would_execute": False,
            "reason": "safe_tail_risk_block",
            "blocked_by": "safe_tail_risk",
        }
    return {
        "action": action,
        "would_execute": True,
        "reason": str(raw.get("reason", "raw_passed_gates")),
        "blocked_by": None,
    }


def evaluate_turbo_shadow(symbol: str, market_payload: dict | None = None) -> dict[str, Any]:
    symbol = normalize_turbo_symbol(symbol)
    risk_guard_warning = None
    try:
        risk_guard = build_turbo_risk_status()
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.exception("turbo risk guard unavailable")
        risk_guard_warning = repr(exc)
        risk_guard = {
            "blocked": True,
            "reason": "risk_guard_unavailable",
            "warning": risk_guard_warning,
        }
    safe_context = _safe_context(symbol, market_payload)
    selected_snapshot = _select_turbo_snapshot(symbol)
    freshness = selected_snapshot["freshness"]
    freshness["feature_timestamp"] = _utc_iso(freshness.get("feature_timestamp"))
    if freshness.get("snapshot_mtime"):
        freshness["snapshot_mtime"] = _utc_iso(freshness.get("snapshot_mtime"))
    if not selected_snapshot.get("path") or not freshness.get("exists"):
        reason = "missing_turbo_artifacts_for_symbol"
        raw = {
            "action": "HOLD",
            "would_execute": False,
            "reason": reason,
            "turbo_score": 0.0,
            "confidence": "blocked",
            "leverage_suggestion": 0.0,
            "position_fraction": 0.0,
            "votes": {"long": 0, "short": 0, "neutral": 0},
            "recent_scores": {
                "long_7d": None,
                "short_7d": None,
                "long_14d": None,
                "short_14d": None,
                "long_30d": None,
                "short_30d": None,
            },
        }
        gated = {
            "action": "HOLD",
            "would_execute": False,
            "reason": reason,
            "blocked_by": "missing_artifacts",
        }
        signal = build_turbo_signal(
            symbol=symbol,
            action="HOLD",
            would_execute=False,
            reason=reason,
            turbo_score=0.0,
            confidence="blocked",
            leverage_suggestion=0.0,
            position_fraction=0.0,
            recent_scores=raw["recent_scores"],
            safe_context=safe_context,
            risk_guard=risk_guard,
            freshness=freshness,
            raw=raw,
            gated=gated,
            enabled=False,
            timestamp=None,
        )
        _update_runtime(symbol, {
            "reason": reason,
            "turbo_score": 0.0,
            "freshness": freshness,
            "snapshot_path": selected_snapshot.get("path"),
        })
        return signal
    if not freshness.get("feature_timestamp") or freshness.get("feature_age_seconds") is None:
        _stale_reason(symbol, freshness)
        reason = "missing_or_invalid_turbo_feature_timestamp"
        raw = {
            "action": "HOLD",
            "would_execute": False,
            "reason": reason,
            "turbo_score": 0.0,
            "confidence": "blocked",
            "leverage_suggestion": 0.0,
            "position_fraction": 0.0,
            "votes": {"long": 0, "short": 0, "neutral": 0},
            "recent_scores": {
                "long_7d": None,
                "short_7d": None,
                "long_14d": None,
                "short_14d": None,
                "long_30d": None,
                "short_30d": None,
            },
        }
        gated = {
            "action": "HOLD",
            "would_execute": False,
            "reason": reason,
            "blocked_by": "stale_snapshot",
        }
        signal = build_turbo_signal(
            symbol=symbol,
            action="HOLD",
            would_execute=False,
            reason=reason,
            turbo_score=0.0,
            confidence="blocked",
            leverage_suggestion=0.0,
            position_fraction=0.0,
            recent_scores=raw["recent_scores"],
            safe_context=safe_context,
            risk_guard=risk_guard,
            freshness=freshness,
            raw=raw,
            gated=gated,
            enabled=True,
            timestamp=None,
        )
        _update_runtime(symbol, {
            "reason": reason,
            "turbo_score": 0.0,
            "freshness": freshness,
            "snapshot_path": selected_snapshot.get("path"),
        })
        if risk_guard_warning:
            signal["risk_guard_warning"] = "history_parse_error_skipped"
            signal["risk_guard_error"] = risk_guard_warning
        return signal
    if float(freshness["feature_age_seconds"]) > TURBO_MAX_FEATURE_AGE_SECONDS:
        _stale_reason(symbol, freshness)
        reason = "stale_turbo_snapshot"
        raw = {
            "action": "HOLD",
            "would_execute": False,
            "reason": reason,
            "turbo_score": 0.0,
            "confidence": "blocked",
            "leverage_suggestion": 0.0,
            "position_fraction": 0.0,
            "votes": {"long": 0, "short": 0, "neutral": 0},
            "recent_scores": {
                "long_7d": None,
                "short_7d": None,
                "long_14d": None,
                "short_14d": None,
                "long_30d": None,
                "short_30d": None,
            },
        }
        gated = {
            "action": "HOLD",
            "would_execute": False,
            "reason": reason,
            "blocked_by": "stale_snapshot",
        }
        signal = build_turbo_signal(
            symbol=symbol,
            action="HOLD",
            would_execute=False,
            reason=reason,
            turbo_score=0.0,
            confidence="blocked",
            leverage_suggestion=0.0,
            position_fraction=0.0,
            recent_scores=raw["recent_scores"],
            safe_context=safe_context,
            risk_guard=risk_guard,
            freshness=freshness,
            raw=raw,
            gated=gated,
            enabled=True,
            timestamp=freshness.get("feature_timestamp"),
        )
        _update_runtime(symbol, {
            "reason": reason,
            "turbo_score": 0.0,
            "freshness": freshness,
            "snapshot_path": selected_snapshot.get("path"),
        })
        if risk_guard_warning:
            signal["risk_guard_warning"] = "history_parse_error_skipped"
            signal["risk_guard_error"] = risk_guard_warning
        return signal
    base_scores = {
        "long_7d": None,
        "short_7d": None,
        "long_14d": None,
        "short_14d": None,
        "long_30d": None,
        "short_30d": None,
    }
    try:
        x, timestamp = _current_feature(symbol)
        if x is None:
            raw = {
                "action": "HOLD",
                "would_execute": False,
                "reason": "insufficient_market_features",
                "turbo_score": 0.0,
                "confidence": "blocked",
                "leverage_suggestion": 0.0,
                "position_fraction": 0.0,
                "votes": {"long": 0, "short": 0, "neutral": 0},
                "recent_scores": base_scores,
            }
            gated = _gated_decision(raw, risk_guard, safe_context)
            signal = build_turbo_signal(
                symbol=symbol,
                reason=gated["reason"],
                recent_scores=base_scores,
                safe_context=safe_context,
                risk_guard=risk_guard,
                freshness=freshness,
                raw=raw,
                gated=gated,
            )
            _update_runtime(symbol, {
                "reason": gated["reason"],
                "turbo_score": 0.0,
                "freshness": freshness,
                "snapshot_path": selected_snapshot.get("path"),
            })
            return signal
        scores, votes = _score_models(x, symbol)
        if all(value is None for value in scores.values()):
            raw = {
                "action": "HOLD",
                "would_execute": False,
                "reason": "no_recent_turbo_models",
                "turbo_score": 0.0,
                "confidence": "blocked",
                "leverage_suggestion": 0.0,
                "position_fraction": 0.0,
                "votes": votes,
                "recent_scores": scores,
            }
            gated = _gated_decision(raw, risk_guard, safe_context)
            signal = build_turbo_signal(
                symbol=symbol,
                timestamp=timestamp,
                reason=gated["reason"],
                votes=votes,
                recent_scores=scores,
                safe_context=safe_context,
                risk_guard=risk_guard,
                freshness=freshness,
                raw=raw,
                gated=gated,
            )
            _update_runtime(symbol, {
                "reason": gated["reason"],
                "turbo_score": 0.0,
                "freshness": freshness,
                "snapshot_path": selected_snapshot.get("path"),
            })
            return signal
        raw = _raw_decision(scores, votes)
        gated = _gated_decision(raw, risk_guard, safe_context)
        final_blocked = gated["action"] == "HOLD"
        signal = build_turbo_signal(
            symbol=symbol,
            timestamp=timestamp,
            action=str(gated["action"]),
            would_execute=bool(gated["would_execute"]),
            reason=str(gated["reason"]),
            turbo_score=float(raw["turbo_score"]),
            confidence="blocked" if final_blocked else str(raw["confidence"]),
            leverage_suggestion=0.0 if final_blocked else float(raw["leverage_suggestion"]),
            position_fraction=0.0 if final_blocked else float(raw["position_fraction"]),
            votes=votes,
            recent_scores=scores,
            safe_context=safe_context,
            risk_guard=risk_guard,
            freshness=freshness,
            raw=raw,
            gated=gated,
        )
        _update_runtime(symbol, {
            "reason": str(gated["reason"]),
            "turbo_score": float(raw["turbo_score"]),
            "freshness": freshness,
            "snapshot_path": selected_snapshot.get("path"),
        })
        if risk_guard_warning:
            signal["risk_guard_warning"] = "history_parse_error_skipped"
            signal["risk_guard_error"] = risk_guard_warning
        return signal
    except Exception as exc:  # pragma: no cover - endpoint safety
        LOGGER.exception("turbo shadow evaluation failed")
        raw = {
            "action": "HOLD",
            "would_execute": False,
            "reason": "turbo_error",
            "turbo_score": 0.0,
            "confidence": "blocked",
            "leverage_suggestion": 0.0,
            "position_fraction": 0.0,
            "votes": {"long": 0, "short": 0, "neutral": 0},
            "recent_scores": base_scores,
        }
        gated = _gated_decision(raw, risk_guard, safe_context)
        return build_turbo_signal(
            symbol=symbol,
            enabled=False,
            reason="turbo_error",
            recent_scores=base_scores,
            safe_context={**safe_context, "error": repr(exc)},
            risk_guard=risk_guard,
            freshness=freshness,
            raw=raw,
            gated=gated,
        )
