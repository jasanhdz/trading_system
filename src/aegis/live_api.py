"""Local HTTP boundary for the owner-authorized current Aegis decision brain.

Endpoints:
  GET  /health              — health check
  POST /ml-v2/predict       — Aegis direction prediction (includes E4 block)
  POST /ml-v2/e4_tail_risk  — E4 tail risk score lookup (precomputed)
  POST /ml-v2/exit_signal   — exit signal (not implemented)
  GET  /diagnostics/runtime — runtime diagnostics
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from aegis.config import CANONICAL_SYMBOLS
from aegis.context_inference import predict_from_snapshot
from aegis.market_context import MarketContextError, market_snapshot_from_context
from aegis.live_decision import (
    CurrentBrainDecisionService,
    CurrentBrainEngine,
    CurrentBrainError,
    PublicKlineSnapshotProvider,
    trace_id,
)
from aegis.hybrid_live_experiment import build_hybrid_live_experiment_selector
from aegis.research.dual_side_shadow import build_composite_research_observer
from aegis.v17_execution_challenger import load_v17_challenger_config

from aegis.risk_guard.domain import FROZEN_TAIL_RISK_THRESHOLD, RiskGuardConfig
from aegis.risk_guard.precompute import (
    E4PrecomputeService,
    SOURCE_FEED_LAG_TOLERANCE_S,
)
from aegis.risk_guard.observability import E4EvidenceRecorder


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    symbol: str = Field(min_length=1, max_length=20)
    market_context: dict[str, Any] | None = None


class E4TailRiskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    symbol: str = Field(min_length=1, max_length=20)
    side: str = Field(pattern=r"^(LONG|SHORT)$")
    decision_at: str = Field(description="ISO 8601 timestamp, 5m aligned")


# Defect #8: Pinned manifest hashes (frozen, never computed dynamically)
PINNED_MODELS_JOBLIB_SHA256 = "d9b02b8045a365541eec8ae02c4b67d99972af1be5fa327f48cac6c95c60dd9f"
PINNED_FEATURE_SCHEMA_SHA256 = "9f86bf95bd78508698a5a1eac9147becaae48565aca6f3fcc1a8e0597d5ba1f2"


def _build_e4_config() -> RiskGuardConfig:
    root = Path(__file__).resolve().parents[2]
    models_path = root / "sandbox/aegis_strategy_router/experiments/aegis_e4_robust_training/artifacts/run_01/development_models.joblib"
    schema_path = root / "sandbox/aegis_strategy_router/experiments/aegis_e4_robust_training/artifacts/dataset_v1/feature_schema.json"

    return RiskGuardConfig(
        enabled=True,
        mode="observe_only",
        tail_risk_threshold=FROZEN_TAIL_RISK_THRESHOLD,
        fail_closed=True,
        models_joblib_path=str(models_path),
        models_joblib_sha256=PINNED_MODELS_JOBLIB_SHA256,
        feature_schema_path=str(schema_path),
        feature_schema_sha256=PINNED_FEATURE_SCHEMA_SHA256,
    )


def _build_e4_unavailable_response(
    symbol: str, side: str, decision_at: str, reason: str
) -> dict[str, Any]:
    return {
        "available": False,
        "symbol": symbol,
        "side": side,
        "decision_at": decision_at,
        "score": None,
        "threshold": FROZEN_TAIL_RISK_THRESHOLD,
        "decision": "BLOCK",
        "reason": reason,
        "model_version": "E4_V1_FROZEN",
        "feature_snapshot_hash": "",
        "feature_available_at": None,
        "source_feed_lag_ms": None,
        "computed_at": None,
        "cache_age_ms": None,
    }


def create_app(
    service: CurrentBrainDecisionService | None = None,
    e4_service: E4PrecomputeService | None = None,
    evidence_recorder: E4EvidenceRecorder | None = None,
    now_fn: Callable[[], datetime] | None = None,
) -> FastAPI:
    runtime = service or build_service()
    e4 = e4_service
    evidence = evidence_recorder or E4EvidenceRecorder()
    current_time = now_fn or (lambda: datetime.now(timezone.utc))

    if e4 is None:
        try:
            e4_config = _build_e4_config()
            e4 = E4PrecomputeService(e4_config, evidence_recorder=evidence)
            e4.initialize()
            e4.start_background()
        except Exception:
            import logging
            logging.getLogger(__name__).exception("E4 precompute service failed to initialize")
            e4 = None

    app = FastAPI(title="Aegis Current Brain API", version="2")

    @app.get("/health")
    async def health() -> dict:
        payload = dict(runtime.health())
        if e4 is not None:
            payload["e4"] = e4.health()
        if payload.get("ready") is not True:
            raise HTTPException(status_code=503, detail=payload)
        return payload

    @app.post("/ml-v2/predict")
    async def predict(request: PredictRequest) -> dict:
        symbol = request.symbol.strip().upper()
        if symbol not in CANONICAL_SYMBOLS:
            raise HTTPException(
                status_code=422, detail="AEGIS_CURRENT_BRAIN_SYMBOL_UNAUTHORIZED"
            )
        try:
            request_trace_id = trace_id()
            if request.market_context is not None:
                snapshot = market_snapshot_from_context(
                    request.market_context,
                    expected_symbol=symbol,
                )
                result = dict(
                    predict_from_snapshot(runtime, snapshot, symbol, request_trace_id)
                )
                result.setdefault("metadata", {})["market_context"] = {
                    "version": request.market_context.get("version"),
                    "source": "TYPESCRIPT_SHARED_WEBSOCKET",
                    "closed_at": snapshot.closed_at.isoformat().replace("+00:00", "Z"),
                    "rest_snapshot_provider_used": False,
                }
            else:
                # Backward-compatible recovery path. Normal bot inference supplies
                # market_context; symbol-only requests retain the previous REST
                # provider for bootstrap/recovery and external diagnostics.
                result = dict(runtime.predict(symbol, request_trace_id))
                result.setdefault("metadata", {})["market_context"] = {
                    "source": "PYTHON_PUBLIC_REST_RECOVERY",
                    "rest_snapshot_provider_used": True,
                }
        except MarketContextError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except CurrentBrainError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail="AEGIS_CURRENT_BRAIN_INFERENCE_FAILED"
            ) from exc

        e4_block = _build_e4_unavailable_response(
            symbol, "LONG", "", "E4_NOT_AVAILABLE"
        )

        if e4 is not None:
            side = _extract_side_from_aegis(result)
            if side:
                decision_at = _cycle_start(current_time())
                e4_block = _lookup_e4_response(
                    e4, symbol, side, decision_at, current_time()
                )

        result.setdefault("aegis", {})
        result["aegis"]["e4_tail_risk"] = e4_block
        return result

    @app.post("/ml-v2/e4_tail_risk")
    async def e4_tail_risk(request: E4TailRiskRequest) -> dict:
        symbol = request.symbol.strip().upper()
        side = request.side.strip().upper()

        if symbol not in CANONICAL_SYMBOLS:
            raise HTTPException(
                status_code=422, detail="E4_SYMBOL_NOT_IN_FROZEN_UNIVERSE"
            )

        if e4 is None:
            return _build_e4_unavailable_response(symbol, side, "", "E4_SERVICE_UNAVAILABLE")

        try:
            requested_dt = datetime.fromisoformat(request.decision_at.replace("Z", "+00:00"))
        except ValueError:
            return _build_e4_unavailable_response(
                symbol, side, request.decision_at, "E4_INVALID_DECISION_AT"
            )
        if requested_dt.tzinfo is None:
            return _build_e4_unavailable_response(
                symbol, side, request.decision_at, "E4_INVALID_DECISION_AT_TIMEZONE"
            )

        return _lookup_e4_response(e4, symbol, side, requested_dt, current_time())

    @app.post("/ml-v2/exit_signal")
    async def exit_signal(_: PredictRequest) -> dict:
        raise HTTPException(
            status_code=501, detail="AEGIS_CURRENT_BRAIN_EXIT_SIGNAL_NOT_PRESENT"
        )

    @app.get("/diagnostics/runtime")
    async def diagnostics() -> dict:
        payload = dict(runtime.health())
        if e4 is not None:
            payload["e4"] = e4.health()
        return payload

    return app


def _cycle_start(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.replace(minute=(value.minute // 5) * 5, second=0, microsecond=0)


def _lookup_e4_response(
    service: E4PrecomputeService,
    symbol: str,
    side: str,
    requested_dt: datetime,
    now: datetime,
) -> dict[str, Any]:
    if requested_dt.tzinfo is None:
        return _build_e4_unavailable_response(
            symbol, side, str(requested_dt), "E4_INVALID_DECISION_AT_TIMEZONE"
        )
    requested_dt = requested_dt.astimezone(timezone.utc)
    expected_cycle = _cycle_start(now)
    if requested_dt != expected_cycle:
        return _build_e4_unavailable_response(
            symbol,
            side,
            requested_dt.isoformat(),
            f"E4_DECISION_AT_MISMATCH: requested={requested_dt.isoformat()}, "
            f"expected={expected_cycle.isoformat()}",
        )

    cycle = service.last_cycle
    if cycle is None or cycle.decision_at != expected_cycle:
        return _build_e4_unavailable_response(
            symbol, side, requested_dt.isoformat(), "E4_EXPECTED_CYCLE_UNAVAILABLE"
        )
    attempt = service.last_attempt
    if attempt and attempt.decision_at == expected_cycle and attempt.error is not None:
        return _build_e4_unavailable_response(
            symbol,
            side,
            requested_dt.isoformat(),
            f"E4_CYCLE_ERROR: {attempt.error}",
        )

    score = service.lookup(symbol, side, requested_dt)
    if score is None:
        return _build_e4_unavailable_response(
            symbol, side, requested_dt.isoformat(), "E4_SCORE_NOT_CACHED"
        )
    lags = list((score.source_feed_lag_ms or {}).values())
    if any(
        not isinstance(lag, (int, float))
        or lag != lag
        or lag > SOURCE_FEED_LAG_TOLERANCE_S * 1000
        for lag in lags
    ):
        return _build_e4_unavailable_response(
            symbol, side, requested_dt.isoformat(), "E4_SOURCE_FEED_LAG"
        )

    cache_age_ms = max(0.0, (now - score.computed_at).total_seconds() * 1000)
    return {
        "available": True,
        "symbol": symbol,
        "side": side,
        "decision_at": requested_dt.isoformat(),
        "score": score.score,
        "threshold": score.threshold,
        "decision": score.risk_decision,
        "reason": score.reason,
        "model_version": score.model_version,
        "feature_snapshot_hash": score.feature_snapshot_hash,
        "feature_available_at": (
            score.feature_available_at.isoformat() if score.feature_available_at else None
        ),
        "source_feed_lag_ms": score.source_feed_lag_ms,
        "computed_at": score.computed_at.isoformat(),
        "cache_age_ms": cache_age_ms,
        "snapshot_id": score.snapshot_id,
    }


def _extract_side_from_aegis(result: dict) -> str | None:
    """Extract side from Aegis prediction result."""
    aegis = result.get("aegis", {})
    turbo = aegis.get("turbo", {})
    action = turbo.get("action", "")
    if action in ("LONG", "SHORT"):
        return action

    decision_brain = aegis.get("decision_brain", {})
    side = decision_brain.get("side", "")
    if side in ("LONG", "SHORT"):
        return side

    return None


def build_service() -> CurrentBrainDecisionService:
    root = Path(__file__).resolve().parents[2]
    observer = build_composite_research_observer(
        root / "config/entry_quality_v2.yaml",
        root / "config/entry_quality_v3_dual_shadow.yaml",
        root / "config/committee_v2_shadow.yaml",
        root / "config/committee_v21_shadow.yaml",
        root / "config/short_probability_shadow.yaml",
        root / "config/entry_intelligence_shadow.yaml",
        root / "config/hybrid_directional_shadow.yaml",
        repo_root=root,
    )
    service = CurrentBrainDecisionService(
        CurrentBrainEngine(),
        PublicKlineSnapshotProvider(),
        research_observer=observer,
        hybrid_live_selector=build_hybrid_live_experiment_selector(
            root / "config/hybrid_directional_live_experiment.yaml",
            repo_root=root,
        ),
        v17_challenger_config=load_v17_challenger_config(
            root / "config/v17_execution_challenger.yaml"
        ),
    )
    service.initialize()
    return service


def main() -> int:
    parser = argparse.ArgumentParser(description="Aegis current-brain local HTTP API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    if args.host != "127.0.0.1" or args.port != 8001:
        raise SystemExit("AEGIS_CURRENT_BRAIN_BINDING_PROHIBITED")
    uvicorn.run(
        "aegis.live_api:create_app",
        host=args.host,
        port=args.port,
        workers=1,
        factory=True,
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
