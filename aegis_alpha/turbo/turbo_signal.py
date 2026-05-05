from __future__ import annotations

import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from aegis_alpha.signals.common import load_signal_market
from aegis_alpha.turbo.config import DEFAULT_TURBO_CONFIG
from aegis_alpha.turbo.turbo_risk import build_turbo_risk_status
from aegis_alpha.turbo.turbo_schema import build_turbo_signal


LOGGER = logging.getLogger(__name__)


def _model_path(side: str, lookback_days: int) -> Path:
    return DEFAULT_TURBO_CONFIG.model_dir / f"turbo_{side}_edge_{lookback_days}d_v010.joblib"


@lru_cache(maxsize=12)
def _load_estimator(path_text: str) -> Any | None:
    path = Path(path_text)
    if not path.exists():
        return None
    bundle = joblib.load(path)
    if isinstance(bundle, dict):
        return bundle.get("estimator")
    return bundle


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(np.asarray(value).item())
    except Exception:
        return None


def _cache_bucket(ttl_seconds: int = 60) -> int:
    return int(time.time() // ttl_seconds)


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
    for lookback_days in reversed(DEFAULT_TURBO_CONFIG.lookback_days):
        dataset_path = DEFAULT_TURBO_CONFIG.data_dir / f"turbo_recent_{lookback_days}d.npz"
        if dataset_path.exists():
            data = np.load(dataset_path, allow_pickle=True)
            x = np.asarray(data["X"], dtype=np.float32)
            timestamps = data["timestamp"].astype(str)
            if len(x) > 0:
                return x[-1:].astype(np.float32), str(timestamps[-1])
    market = _load_market_cached(_cache_bucket())
    if len(market.signal_features) == 0:
        return None, ""
    # Aegis is currently single-symbol; use the configured market if the
    # requested symbol differs, but preserve the requested symbol in response.
    return market.signal_features[-1:].astype(np.float32), str(market.timestamps[market.steps[-1]])


def _score_models(x: np.ndarray) -> tuple[dict[str, float | None], dict[str, int]]:
    scores: dict[str, float | None] = {}
    votes = {"long": 0, "short": 0, "neutral": 0}
    for lookback_days in DEFAULT_TURBO_CONFIG.lookback_days:
        long_est = _load_estimator(str(_model_path("long", int(lookback_days))))
        short_est = _load_estimator(str(_model_path("short", int(lookback_days))))
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
    cfg = DEFAULT_TURBO_CONFIG
    if score >= cfg.thresholds.min_turbo_score_premium:
        return "premium", cfg.leverage.premium, cfg.position_fraction.premium
    if score >= cfg.thresholds.min_turbo_score_shadow:
        return "normal", cfg.leverage.normal, cfg.position_fraction.normal
    return "blocked", 0.0, 0.0


def _raw_decision(scores: dict[str, float | None], votes: dict[str, int]) -> dict[str, Any]:
    cfg = DEFAULT_TURBO_CONFIG
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
    if DEFAULT_TURBO_CONFIG.thresholds.block_if_safe_regime_toxic and safe_context.get("safe_reason") == "regime_block":
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
    cfg = DEFAULT_TURBO_CONFIG
    risk_guard = build_turbo_risk_status()
    safe_context = _safe_context(symbol, market_payload)
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
            return build_turbo_signal(
                symbol=symbol,
                reason=gated["reason"],
                recent_scores=base_scores,
                safe_context=safe_context,
                risk_guard=risk_guard,
                raw=raw,
                gated=gated,
            )
        scores, votes = _score_models(x)
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
            return build_turbo_signal(
                symbol=symbol,
                timestamp=timestamp,
                reason=gated["reason"],
                votes=votes,
                recent_scores=scores,
                safe_context=safe_context,
                risk_guard=risk_guard,
                raw=raw,
                gated=gated,
            )
        raw = _raw_decision(scores, votes)
        gated = _gated_decision(raw, risk_guard, safe_context)
        final_blocked = gated["action"] == "HOLD"
        return build_turbo_signal(
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
            raw=raw,
            gated=gated,
        )
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
            raw=raw,
            gated=gated,
        )
