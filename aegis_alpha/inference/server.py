#!/usr/bin/env python3
from __future__ import annotations

import os

import numpy as np
from fastapi import FastAPI

from aegis_alpha.config import load_config
from aegis_alpha.features.regime_detector import Regime, detect_regime
from aegis_alpha.inference.gates import gate_action
from aegis_alpha.inference.model_loader import AegisModelLoader
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
    return np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


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
    aegis = _build_response(req.symbol)
    return LegacyMlV2Response(
        symbol=req.symbol,
        long_prob=aegis.probs.long,
        short_prob=aegis.probs.short,
        neutral_prob=aegis.probs.idle,
        close_prob=aegis.probs.close,
        consensus_level=aegis.top_prob,
        meta_verdict=f"AEGIS_ALPHA_{aegis.gated_action}",
        smart_leverage=cfg.risk.leverage,
        features=aegis.features,
        aegis=aegis,
    )


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
