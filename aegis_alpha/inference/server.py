#!/usr/bin/env python3
from __future__ import annotations

import os

import numpy as np
from fastapi import FastAPI

from aegis_alpha.config import load_config
from aegis_alpha.features.regime_detector import Regime, detect_regime
from aegis_alpha.inference.gates import gate_action
from aegis_alpha.inference.model_loader import AegisModelLoader
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

cfg = load_config(os.environ.get("AEGIS_CONFIG"))
app = FastAPI(title="Aegis Alpha Inference", version=cfg.model.version)
loader = AegisModelLoader(cfg)


def _neutral_features() -> np.ndarray:
    return np.zeros((cfg.model.window_size, cfg.model.n_features), dtype=np.float32)


def _account_flat() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0, 0.0, cfg.risk.min_flat_steps / 288.0], dtype=np.float32)


def _build_response(symbol: str) -> AegisPredictResponse:
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
        metadata={"model_loaded": pred.model_loaded, "model_reason": pred.reason},
    )


def _model_dump(payload):
    return payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()


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
    log_path, log_warning = safe_log_shadow_signal(
        shadow,
        price=shadow_eval.get("price"),
        model_paths=shadow_eval.get("model_paths", {}),
        model_versions=shadow_eval.get("model_versions", {}),
        request_source=request_source,
    )
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


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "aegis_alpha",
        "model_loaded": loader.loaded,
        "model_name": cfg.model.name,
        "model_version": cfg.model.version,
    }


@app.post("/predict", response_model=AegisPredictResponse)
def predict(req: PredictRequest):
    return _build_response(req.symbol)


@app.post("/ml-v2/predict", response_model=LegacyMlV2Response)
def ml_v2_predict(req: PredictRequest):
    aegis_response = _build_response(req.symbol)
    aegis_block = _aegis_shadow_block(req.symbol, request_source="ml-v2-predict")
    aegis_block["model"] = _model_dump(aegis_response)
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


@app.post("/ml-v2/shadow_signal")
def ml_v2_shadow_signal(req: PredictRequest):
    return _aegis_shadow_block(req.symbol, request_source="ml-v2-shadow-signal")


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
