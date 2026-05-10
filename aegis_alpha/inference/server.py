#!/usr/bin/env python3
from __future__ import annotations

import os
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
from fastapi import FastAPI

from aegis_alpha.config import load_config
from aegis_alpha.features.regime_detector import Regime, detect_regime
from aegis_alpha.inference.gates import gate_action
from aegis_alpha.inference.model_loader import AegisModelLoader
from aegis_alpha.inference import shadow_candidate
from aegis_alpha.inference.shadow_candidate import evaluate_shadow_candidate
from aegis_alpha.inference.shadow_logger import safe_log_shadow_signal
from aegis_alpha.inference.schemas import (
    AegisPredictResponse,
    ExitSignalRequest,
    ExitSignalResponse,
    LegacyMlV2Response,
    PredictRequest,
    Probabilities,
    RegimePayload,
    RiskContext,
    SignalQuality,
)
from aegis_alpha.turbo.config import DEFAULT_TURBO_CONFIG, runtime_turbo_config_status
from aegis_alpha.turbo.turbo_logger import safe_log_turbo_shadow
from aegis_alpha.turbo.snapshot_utils import load_turbo_snapshot_status, normalize_turbo_symbol, turbo_snapshot_path
from aegis_alpha.turbo.turbo_schema import build_turbo_signal
from aegis_alpha.turbo import turbo_signal
from aegis_alpha.turbo.turbo_signal import (
    LAST_TURBO_RUNTIME,
    evaluate_turbo_shadow,
    model_cache_keys,
    model_set_cache_status,
    runtime_status_by_symbol,
    runtime_symbols,
)

logging.basicConfig(
    level=getattr(logging, os.environ.get("AEGIS_LOG_LEVEL", "WARNING").upper(), logging.WARNING),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("aegis_alpha.inference.server")
cfg = load_config(os.environ.get("AEGIS_CONFIG"))
app = FastAPI(title="Aegis Alpha Inference", version=cfg.model.version)
loader = AegisModelLoader(cfg)
START_TIME = time.time()
LAST_PREDICT: dict[str, Any] = {
    "total_ms": None,
    "safe_shadow_ms": None,
    "turbo_ms": None,
    "feature_ms": None,
    "model_load_ms": 0.0,
    "logger_ms": None,
    "fallback_used": False,
    "error": None,
}


def _neutral_features() -> np.ndarray:
    return np.zeros((cfg.model.window_size, cfg.model.n_features), dtype=np.float32)


def _account_flat() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0, 0.0, cfg.risk.min_flat_steps / 288.0], dtype=np.float32)


def _build_response(symbol: str) -> AegisPredictResponse:
    start = time.perf_counter()
    features = _neutral_features()
    regime: Regime = detect_regime(features)
    pred = loader.predict(features, _account_flat())
    top_prob = max(pred.probs.values())
    long_short_gap = abs(pred.probs.get("long", 0.0) - pred.probs.get("short", 0.0))
    gated_action, reason, has_conviction = gate_action(
        pred.raw_action,
        top_prob,
        long_short_gap,
        regime.type,
        cfg.gates,
    )
    return AegisPredictResponse(
        model_name=cfg.model.name,
        model_version=cfg.model.version,
        symbol=symbol,
        raw_action=pred.raw_action,
        gated_action=gated_action,
        probs=Probabilities(**pred.probs),
        top_prob=top_prob,
        long_short_gap=long_short_gap,
        signal_quality=SignalQuality(has_conviction=has_conviction, reason=reason if pred.model_loaded else pred.reason),
        risk_context=RiskContext(
            leverage=cfg.risk.leverage,
            position_fraction=cfg.risk.position_fraction,
            recommended_risk=cfg.risk.position_fraction if has_conviction else 0.0,
        ),
        regime=RegimePayload(type=regime.type, confidence=regime.confidence),
        features={"cvd_z": 0.0, "cvd_slope": 0.0, "weakness": 0.0, "volatility_z": 0.0},
        metadata={"model_loaded": pred.model_loaded, "model_reason": pred.reason, "feature_ms": round((time.perf_counter() - start) * 1000, 3)},
    )


def _model_dump(payload):
    return payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _runtime_turbo_snapshot_status_for_symbol(symbol: str) -> dict[str, Any]:
    statuses = {
        f"{lookback_days}d": load_turbo_snapshot_status(turbo_snapshot_path(int(lookback_days), symbol), include_sample_count=True)
        for lookback_days in DEFAULT_TURBO_CONFIG.lookback_days
    }
    selected_key = None
    selected_status = None
    for key, status in statuses.items():
        if not status.get("exists"):
            continue
        if selected_status is None:
            selected_key, selected_status = key, status
            continue
        current_ts = status.get("feature_timestamp") or ""
        selected_ts = selected_status.get("feature_timestamp") or ""
        current_mtime = status.get("snapshot_mtime") or ""
        selected_mtime = selected_status.get("snapshot_mtime") or ""
        if (current_ts, current_mtime) > (selected_ts, selected_mtime):
            selected_key, selected_status = key, status
    if selected_status is None:
        selected_status = next(iter(statuses.values())) if statuses else {
            "path": None,
            "exists": False,
            "snapshot_mtime": None,
            "snapshot_age_seconds": None,
            "feature_timestamp": None,
            "feature_age_seconds": None,
            "max_feature_age_seconds": None,
            "is_fresh": False,
            "sample_count": 0,
            "last_ts": None,
        }
    return {
        "symbol": normalize_turbo_symbol(symbol),
        "selected": selected_key,
        "last_feature_timestamp": selected_status.get("feature_timestamp") or selected_status.get("last_ts"),
        "feature_age_seconds": selected_status.get("feature_age_seconds"),
        "is_fresh": selected_status.get("is_fresh", False),
        "max_feature_age_seconds": selected_status.get("max_feature_age_seconds"),
        "snapshot_paths": {key: value.get("path") for key, value in statuses.items()},
        "snapshot_statuses": statuses,
        "last_turbo_reason": LAST_TURBO_RUNTIME.get("reason"),
        "last_turbo_score": LAST_TURBO_RUNTIME.get("turbo_score"),
    }


def _runtime_turbo_snapshot_status() -> dict[str, Any]:
    symbols = set(runtime_symbols())
    symbols.add(normalize_turbo_symbol(cfg.symbol))
    runtime_by_symbol = runtime_status_by_symbol()
    statuses = {symbol: _runtime_turbo_snapshot_status_for_symbol(symbol) for symbol in sorted(symbols)}
    for symbol, runtime in runtime_by_symbol.items():
        if symbol in statuses:
            statuses[symbol]["last_turbo_reason"] = runtime.get("reason")
            statuses[symbol]["last_turbo_score"] = runtime.get("turbo_score")
    return statuses


def _defensive_aegis_block(symbol: str, reason: str, error: str | None = None) -> dict:
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
        "blocked_by": "defensive_fallback",
    }
    warning = {"warning": error} if error else {}
    return {
        "candidate": "defensive_fallback",
        "candidate_status": "NOT_LOADED",
        "live_enabled": False,
        "prod": {
            "allowed": False,
            "action": "HOLD",
            "execute": False,
            "reason": "defensive_fallback",
        },
        "shadow": {
            "enabled": False,
            "mode": "SHADOW_ONLY",
            "action": "HOLD",
            "would_execute": False,
            "execute": False,
            "reason": reason,
            **warning,
        },
        "turbo": build_turbo_signal(
            symbol=symbol,
            enabled=False,
            action="HOLD",
            would_execute=False,
            reason=reason,
            raw=raw,
            gated=gated,
            safe_context={"regime": "unknown", "safe_action": "HOLD", "safe_reason": reason},
            risk_guard={"blocked": True, "reason": reason},
        ),
    }


def _defensive_legacy_response(symbol: str, reason: str = "predict_fallback_error", error: str | None = None) -> LegacyMlV2Response:
    return LegacyMlV2Response(
        symbol=symbol,
        long_prob=0.0,
        short_prob=0.0,
        neutral_prob=1.0,
        close_prob=0.0,
        consensus_level=1.0,
        meta_verdict="AEGIS_DEFENSIVE_HOLD",
        smart_leverage=0.0,
        features={"cvd_z": 0.0, "cvd_slope": 0.0, "weakness": 0.0, "volatility_z": 0.0},
        aegis=_defensive_aegis_block(symbol, reason, error),
    )


def _aegis_shadow_block(symbol: str, request_source: str = "ml-v2-predict") -> dict:
    shadow_eval = evaluate_shadow_candidate(symbol)
    candidate_status = str(shadow_eval.get("candidate_status", "UNKNOWN"))
    candidate_live_enabled = bool(shadow_eval.get("live_enabled", False))
    prod_allowed = bool(candidate_status == "LIVE_APPROVED" and candidate_live_enabled)
    prod = {
        "allowed": False,
        "action": "HOLD",
        "execute": False,
        "reason": "candidate_not_live" if not prod_allowed else "live_execution_disabled_by_aegis_v060",
    }
    shadow = dict(shadow_eval.get("shadow", {}))
    shadow["execute"] = False
    shadow["mode"] = "SHADOW_ONLY"
    shadow["enabled"] = True
    log_start = time.perf_counter()
    log_path, log_warning = safe_log_shadow_signal(
        shadow,
        price=shadow_eval.get("price"),
        model_paths=shadow_eval.get("model_paths", {}),
        model_versions=shadow_eval.get("model_versions", {}),
        request_source=request_source,
    )
    shadow["logger_ms"] = round((time.perf_counter() - log_start) * 1000, 3)
    if log_warning:
        shadow["logging_warning"] = log_warning
    elif log_path is not None:
        shadow["log_path"] = str(log_path)
    return {
        "candidate": shadow_eval.get("candidate", "no_candidate_loaded"),
        "candidate_status": candidate_status,
        "live_enabled": False,
        "prod": prod,
        "shadow": shadow,
    }


def _turbo_shadow_block(symbol: str, aegis_block: dict | None = None) -> dict:
    try:
        turbo = evaluate_turbo_shadow(symbol, market_payload=aegis_block)
    except Exception as exc:  # pragma: no cover - endpoint safety
        turbo = build_turbo_signal(symbol=symbol, enabled=False, reason="turbo_error")
        turbo["error"] = repr(exc)
    turbo["execute"] = False
    turbo["live_enabled"] = False
    turbo["production_allowed"] = False
    log_start = time.perf_counter()
    log_path, log_warning = safe_log_turbo_shadow(turbo)
    turbo["logger_ms"] = round((time.perf_counter() - log_start) * 1000, 3)
    if log_warning:
        turbo["logging_warning"] = log_warning
    elif log_path is not None:
        turbo["log_path"] = str(log_path)
    return turbo


@app.get("/health")
def health():
    return {
        "service": "aegis_alpha",
        "status": "healthy",
        "model_name": cfg.model.name,
        "model_version": cfg.model.version,
        "ts": _now_iso(),
    }


@app.get("/debug/runtime")
def debug_runtime():
    sklearn_version = None
    try:
        import sklearn

        sklearn_version = sklearn.__version__
    except Exception:
        sklearn_version = "unavailable"
    return {
        "service": "aegis_alpha",
        "pid": os.getpid(),
        "uptime": round(time.time() - START_TIME, 3),
        "loaded_candidate": shadow_candidate.runtime_debug(),
        "model_cache_keys": {
            "ppo_loaded": loader.loaded,
            "ppo_path": str(loader.model_path),
            "turbo": model_cache_keys(),
            "turbo_sets": model_set_cache_status(),
        },
        "last_predict_latency": {k: v for k, v in LAST_PREDICT.items() if k.endswith("_ms") or k == "fallback_used"},
        "last_predict_error": LAST_PREDICT.get("error"),
        "turbo_snapshot_status": _runtime_turbo_snapshot_status(),
        "turbo_config_status": runtime_turbo_config_status(),
        "shadow_available": True,
        "turbo_available": True,
        "python_version": sys.version,
        "platform": platform.platform(),
        "sklearn_version": sklearn_version,
    }


@app.post("/predict", response_model=AegisPredictResponse)
def predict(req: PredictRequest):
    return _build_response(req.symbol)


@app.post("/ml-v2/predict", response_model=LegacyMlV2Response)
def ml_v2_predict(req: PredictRequest):
    total_start = time.perf_counter()
    timings: dict[str, float | None] = {
        "feature_ms": None,
        "safe_shadow_ms": None,
        "turbo_ms": None,
        "model_load_ms": 0.0,
        "logger_ms": None,
    }
    fallback_used = False
    error = None
    try:
        feature_start = time.perf_counter()
        aegis_response = _build_response(req.symbol)
        timings["feature_ms"] = round((time.perf_counter() - feature_start) * 1000, 3)

        shadow_start = time.perf_counter()
        try:
            aegis_block = _aegis_shadow_block(req.symbol, request_source="ml-v2-predict")
        except Exception as exc:  # pragma: no cover - endpoint safety
            fallback_used = True
            LOGGER.exception("aegis_shadow_block_failed")
            aegis_block = _defensive_aegis_block(req.symbol, "shadow_fallback_error", repr(exc))
        timings["safe_shadow_ms"] = round((time.perf_counter() - shadow_start) * 1000, 3)

        turbo_start = time.perf_counter()
        try:
            aegis_block["turbo"] = _turbo_shadow_block(req.symbol, aegis_block)
        except Exception as exc:  # pragma: no cover - endpoint safety
            fallback_used = True
            LOGGER.exception("aegis_turbo_block_failed")
            aegis_block["turbo"] = _defensive_aegis_block(req.symbol, "turbo_fallback_error", repr(exc))["turbo"]
        timings["turbo_ms"] = round((time.perf_counter() - turbo_start) * 1000, 3)

        aegis_block["model"] = _model_dump(aegis_response)
        total_ms = round((time.perf_counter() - total_start) * 1000, 3)
        timings["logger_ms"] = round(float((aegis_block.get("shadow") or {}).get("logger_ms", 0.0)) + float((aegis_block.get("turbo") or {}).get("logger_ms", 0.0)), 3)
        LAST_PREDICT.update({**timings, "total_ms": total_ms, "fallback_used": fallback_used, "error": error})
        LOGGER.warning(
            "aegis_predict_latency symbol=%s total_ms=%.3f feature_ms=%s safe_shadow_ms=%s turbo_ms=%s logger_ms=%s fallback_used=%s",
            req.symbol,
            total_ms,
            timings["feature_ms"],
            timings["safe_shadow_ms"],
            timings["turbo_ms"],
            timings["logger_ms"],
            fallback_used,
        )
        return LegacyMlV2Response(
            symbol=req.symbol,
            long_prob=aegis_response.probs.long,
            short_prob=aegis_response.probs.short,
            neutral_prob=aegis_response.probs.idle,
            close_prob=aegis_response.probs.close,
            consensus_level=aegis_response.top_prob,
            meta_verdict=f"AEGIS_ALPHA_{aegis_response.gated_action}",
            smart_leverage=0.0,
            features=aegis_response.features,
            aegis=aegis_block,
        )
    except Exception as exc:  # pragma: no cover - endpoint safety
        error = repr(exc)
        total_ms = round((time.perf_counter() - total_start) * 1000, 3)
        LAST_PREDICT.update({**timings, "total_ms": total_ms, "fallback_used": True, "error": error})
        LOGGER.exception("aegis_predict_fallback symbol=%s total_ms=%.3f", req.symbol, total_ms)
        return _defensive_legacy_response(req.symbol, "predict_fallback_error", error)


@app.post("/ml-v2/shadow_signal")
def ml_v2_shadow_signal(req: PredictRequest):
    return _aegis_shadow_block(req.symbol, request_source="ml-v2-shadow-signal")


@app.post("/ml-v2/turbo_shadow")
def ml_v2_turbo_shadow(req: PredictRequest):
    return _turbo_shadow_block(req.symbol)


@app.post("/ml-v2/exit_signal", response_model=ExitSignalResponse)
def ml_v2_exit_signal(req: ExitSignalRequest):
    if req.current_pnl <= -0.08 or req.duration_minutes >= 360:
        return ExitSignalResponse(action="CLOSE", confidence=0.70, reason="defensive_exit_rule")
    return ExitSignalResponse(action="HOLD", confidence=0.50, reason="no_exit_conviction")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("AEGIS_HOST", "0.0.0.0"),
        port=int(os.environ.get("AEGIS_PORT", "8001")),
        access_log=False,
        log_level=os.environ.get("AEGIS_LOG_LEVEL", "warning"),
    )
