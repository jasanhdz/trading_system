"""FastAPI service that returns ML model probabilities for supplied candles."""
from __future__ import annotations

import logging
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

ADVANCED_MODELS_ROOT = (REPO_ROOT / "models" / "advanced").resolve()
LEGACY_MODELS_ROOT = (REPO_ROOT / "models" / "trained").resolve()

if ADVANCED_MODELS_ROOT.exists():
    os.environ.setdefault("ML_MODELS_ROOT", str(ADVANCED_MODELS_ROOT))
elif "ML_MODELS_ROOT" not in os.environ and LEGACY_MODELS_ROOT.exists():
    os.environ["ML_MODELS_ROOT"] = str(LEGACY_MODELS_ROOT)

from binance_futures_bot_py.src.core.types import Candle
from binance_futures_bot_py.src.infra.config import Config
from binance_futures_bot_py.src.strategies.ml_probability import (
    MLProbabilityStrategy,
    _candles_to_frame,
)


class ServiceLogger:
    """Minimal logger compatible with the bot's Logger protocol."""

    def __init__(self, name: str = "ml_probability_service") -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
        )
        self._logger = logging.getLogger(name)

    def _fmt(self, message: str, context: Optional[Dict] = None) -> str:
        return f"{message} | {context}" if context else message

    def debug(self, message: str, context: Optional[Dict] = None, **_: object) -> None:
        self._logger.debug(self._fmt(message, context))

    def info(self, message: str, context: Optional[Dict] = None, **_: object) -> None:
        self._logger.info(self._fmt(message, context))

    def warning(self, message: str, context: Optional[Dict] = None, **_: object) -> None:
        self._logger.warning(self._fmt(message, context))

    def error(self, message: str, context: Optional[Dict] = None, **_: object) -> None:
        self._logger.error(self._fmt(message, context))

    def critical(self, message: str, context: Optional[Dict] = None, **_: object) -> None:
        self._logger.critical(self._fmt(message, context))

    def warn(self, message: str, context: Optional[Dict] = None, **_: object) -> None:
        self.warning(message, context)

    def bind(self, **_: object) -> "ServiceLogger":
        return self


def normalize_symbol(raw: str) -> str:
    return raw.replace("/", "").upper()


def payloads_to_sorted_candles(payloads: Iterable[CandlePayload]) -> List[Candle]:
    return sorted((payload.to_candle() for payload in payloads), key=lambda c: c.close_time)


class CandlePayload(BaseModel):
    open_time: int = Field(..., description="Open timestamp in ms")
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int = Field(..., description="Close timestamp in ms")

    @validator("close_time")
    def validate_order(cls, v: int, values: Dict[str, int]) -> int:
        if "open_time" in values and v < values["open_time"]:
            raise ValueError("close_time must be greater than or equal to open_time")
        return v

    def to_candle(self) -> Candle:
        return Candle(
            open_time=int(self.open_time),
            open=float(self.open),
            high=float(self.high),
            low=float(self.low),
            close=float(self.close),
            volume=float(self.volume),
            close_time=int(self.close_time),
        )


class ProbabilityRequest(BaseModel):
    symbol: str = Field(..., description="Symbol identifier, e.g. XRPUSDT or XRP/USDT")
    candles: List[CandlePayload] = Field(..., description="Ordered OHLCV candles (oldest -> newest)")
    timeframe: Optional[str] = Field(None, description="Timeframe label (defaults to config)")
    force_refresh: bool = Field(False, description="Ignore cached results for identical candles")
    extra_candles: Dict[str, List[CandlePayload]] = Field(
        default_factory=dict,
        description="Optional additional candle sets keyed by timeframe (e.g. {'15m': [...]})",
    )

    @validator("candles")
    def ensure_non_empty(cls, v: List[CandlePayload]) -> List[CandlePayload]:
        if not v:
            raise ValueError("candles payload must contain at least one entry")
        return v

    @validator("extra_candles")
    def ensure_extra_non_empty(cls, v: Dict[str, List[CandlePayload]]) -> Dict[str, List[CandlePayload]]:
        for timeframe, candles in v.items():
            if not candles:
                raise ValueError(f"extra_candles[{timeframe}] must not be empty")
        return v


class TimeframeProbability(BaseModel):
    long_prob: float
    short_prob: float


class ProbabilityResponse(BaseModel):
    symbol: str
    primary_timeframe: str
    long_prob: float
    short_prob: float
    probabilities: Dict[str, TimeframeProbability]


@dataclass
class CacheEntry:
    signature: Tuple[Tuple[str, int, int], ...]
    payload: ProbabilityResponse


CONFIG = Config()
STRATEGY = MLProbabilityStrategy(
    history_bars=max(512, getattr(CONFIG, "ML_HISTORY_BARS", 512))
)
LOGGER = ServiceLogger()
CACHE: Dict[Tuple[str, str], CacheEntry] = {}

router = APIRouter(prefix="/ml", tags=["ml"])


def compute_single_timeframe_probabilities(
    strategy: MLProbabilityStrategy,
    config: Config,
    logger: ServiceLogger,
    symbol: str,
    timeframe: str,
    candles: List[Candle],
) -> Tuple[float, float]:
    try:
        predictor = strategy._ensure_predictor(symbol, timeframe, config, logger)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error(
            "ml_model_load_fail",
            {"err": str(exc), "symbol": symbol, "timeframe": timeframe},
        )
        raise HTTPException(
            status_code=503,
            detail=f"ML model not available for symbol '{symbol}' timeframe '{timeframe}'",
        ) from exc

    limit = max(strategy.history_bars, getattr(config, "ML_HISTORY_BARS", strategy.history_bars))
    min_required = min(limit, max(getattr(config, "ML_MULTI_TF_MIN_BARS", limit), 64))

    if len(candles) < min_required:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "insufficient_candles",
                "timeframe": timeframe,
                "required": min_required,
                "received": len(candles),
            },
        )

    ordered = sorted(candles, key=lambda c: c.close_time)

    try:
        frame = _candles_to_frame(ordered)
        probs = predictor.predict(frame)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error(
            "ml_prediction_fail",
            {"err": str(exc), "symbol": symbol, "timeframe": timeframe},
        )
        raise HTTPException(status_code=500, detail=f"ML prediction failed for {timeframe}") from exc

    return float(probs["long"]), float(probs["short"])


def compute_multi_timeframe_probabilities(
    strategy: MLProbabilityStrategy,
    config: Config,
    logger: ServiceLogger,
    symbol: str,
    candles_by_timeframe: Dict[str, List[Candle]],
) -> Dict[str, Tuple[float, float]]:
    results: Dict[str, Tuple[float, float]] = OrderedDict()
    for timeframe, candles in candles_by_timeframe.items():
        results[timeframe] = compute_single_timeframe_probabilities(
            strategy=strategy,
            config=config,
            logger=logger,
            symbol=normalize_symbol(symbol),
            timeframe=timeframe,
            candles=candles,
        )
    return results


@router.post("/probabilities", response_model=ProbabilityResponse)
async def probability_endpoint(request: ProbabilityRequest) -> ProbabilityResponse:
    symbol = normalize_symbol(request.symbol)

    try:
        symbol_cfg = CONFIG.get_symbol_config(symbol)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown symbol '{request.symbol}': {exc}",
        ) from exc

    primary_timeframe = (request.timeframe or symbol_cfg.timeframe).strip()
    if not primary_timeframe:
        raise HTTPException(status_code=400, detail="Unable to resolve primary timeframe")

    candles_by_timeframe: OrderedDict[str, List[Candle]] = OrderedDict()
    candles_by_timeframe[primary_timeframe] = payloads_to_sorted_candles(request.candles)

    resolved_timeframes = STRATEGY._resolve_timeframes(symbol_cfg, CONFIG)
    if primary_timeframe not in resolved_timeframes:
        resolved_timeframes = [primary_timeframe] + resolved_timeframes

    for timeframe in resolved_timeframes:
        if timeframe == primary_timeframe:
            continue
        extra_payloads = request.extra_candles.get(timeframe)
        if extra_payloads:
            candles_by_timeframe[timeframe] = payloads_to_sorted_candles(extra_payloads)

    for timeframe, payloads in request.extra_candles.items():
        if timeframe not in candles_by_timeframe:
            candles_by_timeframe[timeframe] = payloads_to_sorted_candles(payloads)

    missing_timeframes = [tf for tf in resolved_timeframes if tf not in candles_by_timeframe]
    if missing_timeframes:
        LOGGER.debug(
            "ml_prob_service_missing_tf",
            {"symbol": symbol_cfg.symbol, "timeframes": missing_timeframes},
        )

    signature = tuple(
        (tf, candles[-1].close_time, len(candles)) for tf, candles in sorted(candles_by_timeframe.items())
    )

    cache_key = (symbol_cfg.symbol, primary_timeframe)
    if not request.force_refresh:
        cached = CACHE.get(cache_key)
        if cached and cached.signature == signature:
            return cached.payload

    probabilities = compute_multi_timeframe_probabilities(
        strategy=STRATEGY,
        config=CONFIG,
        logger=LOGGER,
        symbol=symbol_cfg.symbol,
        candles_by_timeframe=candles_by_timeframe,
    )

    timeframe_payload = {
        tf: TimeframeProbability(long_prob=long_prob, short_prob=short_prob)
        for tf, (long_prob, short_prob) in probabilities.items()
    }

    if primary_timeframe not in timeframe_payload:
        raise HTTPException(
            status_code=422,
            detail=f"No probabilities produced for primary timeframe '{primary_timeframe}'",
        )

    primary_probs = timeframe_payload[primary_timeframe]

    response = ProbabilityResponse(
        symbol=symbol_cfg.symbol,
        primary_timeframe=primary_timeframe,
        long_prob=primary_probs.long_prob,
        short_prob=primary_probs.short_prob,
        probabilities=timeframe_payload,
    )

    CACHE[cache_key] = CacheEntry(signature=signature, payload=response)
    return response


app = FastAPI(title="ML Probability Service", version="0.2.0")
app.include_router(router)


@app.get("/health", tags=["meta"])
async def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "services.ml_probability_service:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
