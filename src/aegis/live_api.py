"""Local HTTP boundary for the owner-authorized current Aegis decision brain."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from aegis.config import CANONICAL_SYMBOLS
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


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    symbol: str = Field(min_length=1, max_length=20)


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


def create_app(service: CurrentBrainDecisionService | None = None) -> FastAPI:
    runtime = service or build_service()
    app = FastAPI(title="Aegis Current Brain API", version="1")

    @app.get("/health")
    async def health() -> dict:
        payload = dict(runtime.health())
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
            return dict(runtime.predict(symbol, trace_id()))
        except CurrentBrainError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail="AEGIS_CURRENT_BRAIN_INFERENCE_FAILED"
            ) from exc

    @app.post("/ml-v2/exit_signal")
    async def exit_signal(_: PredictRequest) -> dict:
        raise HTTPException(
            status_code=501, detail="AEGIS_CURRENT_BRAIN_EXIT_SIGNAL_NOT_PRESENT"
        )

    @app.get("/diagnostics/runtime")
    async def diagnostics() -> dict:
        return dict(runtime.health())

    return app


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
