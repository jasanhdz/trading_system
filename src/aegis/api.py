"""Minimal versioned HTTP API for the scientific brain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .domain import BrainManifest, DecisionOutcome, DecisionRequest, DecisionResponse, decision_outcome_from_dict, decision_request_from_dict
from .runtime import BrainRuntime
from .utils import to_primitive


@dataclass
class BrainApi:
    runtime: BrainRuntime

    def health(self) -> Mapping[str, str]:
        return {"status": "alive"}

    def ready(self) -> Mapping[str, str | bool]:
        return {"status": "ready" if self.runtime.ready else "not_ready", "ready": self.runtime.ready}

    def manifest(self) -> BrainManifest:
        return self.runtime.manifest()

    def evaluate(self, request: DecisionRequest) -> DecisionResponse:
        return self.runtime.evaluate(request)

    def submit_outcome(self, outcome: DecisionOutcome) -> None:
        self.runtime.evidence.record_outcome(outcome)


def create_app(api: BrainApi) -> Any:
    """Bind the transport-neutral service without exposing operational routes."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    app = FastAPI(title="Aegis Scientific Brain", version=api.runtime.config.contract_version)

    @app.exception_handler(Exception)
    async def structured_error(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=422 if isinstance(exc, ValueError) else 503,
                            content={"error": {"code": type(exc).__name__, "message": str(exc)}, "executable": False})

    @app.get("/health")
    def health() -> Mapping[str, str]: return api.health()

    @app.get("/ready")
    def ready() -> Mapping[str, str | bool]: return api.ready()

    @app.get("/manifest")
    def manifest() -> Any: return to_primitive(api.manifest())

    @app.post("/v1/decisions/evaluate")
    def evaluate(payload: dict[str, Any]) -> Any:
        return to_primitive(api.evaluate(decision_request_from_dict(payload)))

    @app.post("/v1/evidence/outcome", status_code=204)
    def outcome(payload: dict[str, Any]) -> None:
        api.submit_outcome(decision_outcome_from_dict(payload))

    return app
