"""Current-brain evaluation and lossless TypeScript compatibility mapping."""

from __future__ import annotations

import hashlib
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol
from urllib.parse import urlparse

import requests

from aegis.config import CANONICAL_SYMBOLS, load_brain_config
from aegis.domain import (
    Candle,
    FeedQuality,
    MarketSnapshot,
    PortfolioContext,
    SymbolSeries,
    TradeSide,
)
from aegis.features import DeterministicFeaturePipeline, MarketSnapshotValidator
from aegis.prospective.model_qualification import (
    CANDIDATE_IDENTITY,
    FEATURE_SCHEMA,
    SOURCE_BUNDLE_SHA256,
    load_qualified_candidate,
    qualify_inference,
)
from aegis.training.experiment import evaluate_authoritative_feature_batch
from aegis.utils import Sha256HashProvider, sha256_file, to_primitive


MODEL_BUNDLE_SHA256 = "23b22403b70f7d6c385d1214e6543197f4ca4e57269af19b1013987891ed550a"
MODEL_ARTIFACT_SHA256 = SOURCE_BUNDLE_SHA256
CONFIGURATION_SHA256 = "f944b0210b31928a519dc63459be3f1d53de811517dc1bbe9753596314579ec1"
FEATURE_COUNT = 83
SERVICE_VERSION = "aegis-current-brain-http-v1"
CANONICAL_DECISION_CONTRACT = "aegis-current-brain-live-decision-v1"
CANONICAL_LIVE_AUTHORITY = "OWNER_AUTHORIZED_CURRENT_PYTHON_COMMITTEE_LIVE_INTEGRATION"
PUBLIC_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
INTERVAL_MS = 300_000


class CurrentBrainError(RuntimeError):
    pass


class SnapshotProvider(Protocol):
    def snapshot(self) -> MarketSnapshot: ...


class ResearchBatchObserver(Protocol):
    @property
    def mode(self) -> Any: ...

    def observe_batch(self, batch: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def health(self) -> Mapping[str, Any]: ...


def _utc(milliseconds: int) -> datetime:
    return datetime.fromtimestamp(milliseconds / 1000, tz=timezone.utc)


class PublicKlineSnapshotProvider:
    """Fetch only the public USD-M kline surface used by prospective Shadow."""

    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        now: Callable[[], datetime] | None = None,
        timeout_seconds: float = 10.0,
        attempts: int = 3,
    ) -> None:
        parsed = urlparse(PUBLIC_KLINES_URL)
        if parsed.scheme != "https" or parsed.netloc != "fapi.binance.com" or parsed.path != "/fapi/v1/klines":
            raise CurrentBrainError("AEGIS_CURRENT_BRAIN_PUBLIC_ENDPOINT_INVALID")
        self._session = session or requests.Session()
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._timeout_seconds = timeout_seconds
        self._attempts = attempts

    def _candles(self, symbol: str, limit: int = 96) -> tuple[Candle, ...]:
        if symbol not in CANONICAL_SYMBOLS:
            raise CurrentBrainError("AEGIS_CURRENT_BRAIN_SYMBOL_UNAUTHORIZED")
        last_error: Exception | None = None
        for attempt in range(self._attempts):
            try:
                response = self._session.get(
                    PUBLIC_KLINES_URL,
                    params={"symbol": symbol, "interval": "5m", "limit": str(limit)},
                    timeout=self._timeout_seconds,
                    allow_redirects=False,
                )
                if response.is_redirect or response.is_permanent_redirect:
                    raise CurrentBrainError("AEGIS_CURRENT_BRAIN_PUBLIC_REDIRECT_PROHIBITED")
                response.raise_for_status()
                rows = response.json()
                if not isinstance(rows, list):
                    raise CurrentBrainError("AEGIS_CURRENT_BRAIN_PUBLIC_RESPONSE_INVALID")
                now_ms = int(self._now().timestamp() * 1000)
                candles: list[Candle] = []
                for row in rows:
                    if not isinstance(row, list) or len(row) < 6:
                        raise CurrentBrainError("AEGIS_CURRENT_BRAIN_PUBLIC_RESPONSE_INVALID")
                    open_ms = int(row[0])
                    close_ms = open_ms + INTERVAL_MS
                    if close_ms > now_ms:
                        continue
                    values = tuple(float(row[index]) for index in range(1, 6))
                    if not all(math.isfinite(value) and value >= 0 for value in values):
                        raise CurrentBrainError("AEGIS_CURRENT_BRAIN_PUBLIC_RESPONSE_INVALID")
                    candles.append(Candle(
                        _utc(open_ms), _utc(close_ms), values[0], values[1], values[2], values[3], values[4],
                        True, "BINANCE_USDM_PUBLIC_KLINES", str(open_ms),
                    ))
                if len(candles) < 60:
                    raise CurrentBrainError("AEGIS_CURRENT_BRAIN_INCOMPLETE_SERIES")
                return tuple(candles)
            except (requests.RequestException, ValueError, CurrentBrainError) as exc:
                last_error = exc
                if attempt + 1 < self._attempts:
                    time.sleep(0.25 * (attempt + 1))
        raise CurrentBrainError("AEGIS_CURRENT_BRAIN_PUBLIC_CONNECTIVITY_FAILED") from last_error

    def snapshot(self) -> MarketSnapshot:
        with ThreadPoolExecutor(max_workers=len(CANONICAL_SYMBOLS)) as pool:
            futures = {symbol: pool.submit(self._candles, symbol) for symbol in CANONICAL_SYMBOLS}
            by_symbol = {symbol: future.result() for symbol, future in futures.items()}
        closed_at = min(candles[-1].close_time for candles in by_symbol.values())
        series = []
        now = self._now()
        for symbol in CANONICAL_SYMBOLS:
            candles = tuple(item for item in by_symbol[symbol] if item.close_time <= closed_at)[-96:]
            if len(candles) < 60 or candles[-1].close_time != closed_at:
                raise CurrentBrainError("AEGIS_CURRENT_BRAIN_UNALIGNED_SERIES")
            series.append(SymbolSeries(
                symbol,
                candles,
                closed_at,
                FeedQuality(source_lag_ms=max(0, int((now - closed_at).total_seconds() * 1000))),
            ))
        return MarketSnapshot(
            closed_at,
            "5m",
            load_brain_config(Path(__file__).resolve().parents[2] / "config").universe.symbol_set_hash,
            tuple(series),
            PortfolioContext(available_slots=1, operational_time=closed_at),
        )


@dataclass(frozen=True)
class CurrentBrainPaths:
    repo_root: Path
    config_dir: Path
    candidate_path: Path

    @classmethod
    def defaults(cls) -> "CurrentBrainPaths":
        root = Path(__file__).resolve().parents[2]
        return cls(
            root,
            root / "config",
            root / "config/bundles/aegis-prospective-shadow-candidate-v1.json",
        )


class CurrentBrainEngine:
    """Load and evaluate the exact qualified artifact without Shadow replay."""

    def __init__(self, paths: CurrentBrainPaths | None = None) -> None:
        self.paths = paths or CurrentBrainPaths.defaults()
        self._hashing = Sha256HashProvider()
        self._candidate = None
        self._config = None
        self._features = None
        self._validator = None
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def initialize(self) -> None:
        if self._ready:
            return
        if sha256_file(self.paths.candidate_path) != MODEL_BUNDLE_SHA256:
            raise CurrentBrainError("AEGIS_CURRENT_BRAIN_MODEL_BUNDLE_HASH_MISMATCH")
        candidate = load_qualified_candidate(self.paths.candidate_path)
        if candidate.model_identity != CANDIDATE_IDENTITY or candidate.model_artifact_hash != MODEL_ARTIFACT_SHA256:
            raise CurrentBrainError("AEGIS_CURRENT_BRAIN_MODEL_IDENTITY_MISMATCH")
        config = load_brain_config(self.paths.config_dir)
        if config.config_hash != CONFIGURATION_SHA256:
            raise CurrentBrainError("AEGIS_CURRENT_BRAIN_CONFIGURATION_HASH_MISMATCH")
        features = DeterministicFeaturePipeline(
            candidate.source.normalizer,
            schema_version=candidate.source.feature_schema_version,
        )
        if features.schema_version != FEATURE_SCHEMA or len(features.feature_names) != FEATURE_COUNT:
            raise CurrentBrainError("AEGIS_CURRENT_BRAIN_FEATURE_CONTRACT_MISMATCH")
        self_check = qualify_inference(candidate)
        if (
            self_check.get("finite_output") != "PASS"
            or self_check.get("non_degeneracy") != "PASS"
            or self_check.get("batch_single_agreement") != "PASS"
            or self_check.get("repeated_inference") != "BYTE_IDENTICAL"
        ):
            raise CurrentBrainError("AEGIS_CURRENT_BRAIN_SELF_CHECK_FAILED")
        self._candidate = candidate
        self._config = config
        self._features = features
        self._validator = MarketSnapshotValidator(config.universe)
        self._ready = True

    def evaluate(self, snapshot: MarketSnapshot) -> Mapping[str, Any]:
        if (
            not self._ready
            or self._candidate is None
            or self._config is None
            or self._features is None
            or self._validator is None
        ):
            raise CurrentBrainError("AEGIS_CURRENT_BRAIN_NOT_READY")
        now = datetime.now(timezone.utc)
        self._validator.validate(snapshot, now)
        features = self._features.transform(snapshot)
        cycle_material = {
            "closed_at": snapshot.closed_at,
            "feature_values": tuple(row.raw_values for row in features.rows),
        }
        cycle_hash = self._hashing.digest_value(cycle_material)
        pipeline = evaluate_authoritative_feature_batch(
            self._candidate.source,
            features,
            timestamp=snapshot.closed_at,
            config={"protocol": {"friction_fraction": 0.001}},
            request_id=f"live-http-request-{cycle_hash[:24]}",
            decision_cycle_id=f"live-http-cycle-{cycle_hash[:24]}",
            portfolio=snapshot.portfolio,
        )
        selected = {item.candidate_hash for item in pipeline.selection.selected}
        predictions = {item.symbol: pipeline.predictions.for_symbol(item.symbol) for item in features.rows}
        layers = {item.symbol: item for item in pipeline.layers.results}
        candidates = {item.symbol: item for item in pipeline.candidates.candidates}
        market_series = {item.symbol: item for item in snapshot.series}
        results: dict[str, Any] = {}
        for row in features.rows:
            symbol_predictions = predictions[row.symbol]
            layer = layers[row.symbol]
            candidate = candidates[row.symbol]
            market_bar = market_series[row.symbol].candles[-1]
            vector_hash = self._hashing.digest_value({"names": features.feature_names, "values": row.raw_values})
            results[row.symbol] = {
                "symbol": row.symbol,
                "market_timestamp": snapshot.closed_at.isoformat().replace("+00:00", "Z"),
                "feature_schema": features.schema_version,
                "feature_schema_hash": features.feature_hash,
                "feature_vector_hash": vector_hash,
                "feature_count": len(features.feature_names),
                "feature_quality": to_primitive(row.quality),
                "research_features": dict(zip(features.feature_names, row.raw_values)),
                "research_normalized_features": dict(
                    zip(features.feature_names, row.normalized_values)
                ),
                "market_bar": {
                    "open": market_bar.open,
                    "high": market_bar.high,
                    "low": market_bar.low,
                    "close": market_bar.close,
                },
                "predictions": [to_primitive(value) for value in symbol_predictions],
                "layer": to_primitive(layer),
                "candidate": to_primitive(candidate),
                "selected": candidate.candidate_hash in selected,
                "selection_status": pipeline.selection.status.value,
            }
        return {
            "schema_id": "aegis-current-brain-canonical-batch-v1",
            "decision_cycle_id": f"live-http-cycle-{cycle_hash[:24]}",
            "market_timestamp": snapshot.closed_at.isoformat().replace("+00:00", "Z"),
            "model_identifier": CANDIDATE_IDENTITY,
            "model_sha256": MODEL_ARTIFACT_SHA256,
            "bundle_sha256": MODEL_BUNDLE_SHA256,
            "configuration_sha256": CONFIGURATION_SHA256,
            "feature_schema": FEATURE_SCHEMA,
            "feature_count": FEATURE_COUNT,
            "results": results,
        }


def compatibility_response(batch: Mapping[str, Any], symbol: str, trace_id: str) -> Mapping[str, Any]:
    normalized = symbol.strip().upper()
    if normalized not in CANONICAL_SYMBOLS:
        raise CurrentBrainError("AEGIS_CURRENT_BRAIN_SYMBOL_UNAUTHORIZED")
    result = batch["results"][normalized]
    predictions = result["predictions"]
    if not predictions:
        raise CurrentBrainError("AEGIS_CURRENT_BRAIN_MODEL_OUTPUT_MISSING")
    count = len(predictions)
    long_probability = math.fsum(float(item["long_probability"]) for item in predictions) / count
    short_probability = math.fsum(float(item["short_probability"]) for item in predictions) / count
    neutral_probability = math.fsum(float(item["neutral_probability"]) for item in predictions) / count
    if not all(math.isfinite(value) for value in (long_probability, short_probability, neutral_probability)):
        raise CurrentBrainError("AEGIS_CURRENT_BRAIN_MODEL_OUTPUT_NONFINITE")
    votes = {
        "long": sum(item["side"] == TradeSide.LONG.value for item in predictions),
        "short": sum(item["side"] == TradeSide.SHORT.value for item in predictions),
        "neutral": sum(item["side"] == TradeSide.NO_TRADE.value for item in predictions),
    }
    candidate = result["candidate"]
    layer = result["layer"]
    v2_by_symbol = batch.get("_entry_quality_v2", {})
    v2 = v2_by_symbol.get(normalized) if isinstance(v2_by_symbol, Mapping) else None
    v2_live = isinstance(v2, Mapping) and v2.get("mode") == "LIVE"
    selected = bool(v2["selected"]) if v2_live else bool(result["selected"])
    side_value = candidate["side"]
    side = str(getattr(side_value, "value", side_value))
    if v2_live:
        side = "SHORT"
    action = side if selected and side in {"LONG", "SHORT"} else "HOLD"
    reasons = [str(getattr(value, "value", value)) for value in candidate["reason_codes"]]
    if v2_live:
        reason = (
            "ENTRY_QUALITY_V2_SELECTED"
            if selected
            else "ENTRY_QUALITY_V2_NOT_SELECTED"
        )
    else:
        reason = "CURRENT_BRAIN_SELECTED" if selected else ",".join(reasons) or "CURRENT_BRAIN_NO_TRADE"
    eq_allowed = "EQM_QUALITY_LOW" not in reasons
    raw = {
        "action": action,
        "would_execute": selected,
        "reason": reason,
        "turbo_score": float(candidate["raw_score"]),
        "votes": votes,
    }
    return {
        "symbol": normalized,
        "long_prob": long_probability,
        "short_prob": short_probability,
        "neutral_prob": neutral_probability,
        "meta_verdict": "CURRENT_BRAIN_SELECTED" if selected else "CURRENT_BRAIN_NO_TRADE",
        "features": {
            "fallback": False,
            "schema": result["feature_schema"],
            "count": result["feature_count"],
            "schema_hash": result["feature_schema_hash"],
            "vector_hash": result["feature_vector_hash"],
            "market_timestamp": result["market_timestamp"],
            "quality": result["feature_quality"],
        },
        "aegis": {
            "candidate": batch["model_identifier"],
            "candidate_status": "OWNER_AUTHORIZED_CURRENT_PYTHON_COMMITTEE_LIVE_INTEGRATION",
            "live_enabled": True,
            "prod": {"allowed": selected, "action": action, "execute": selected, "reason": reason},
            "turbo": {
                "enabled": True,
                "execute": selected,
                "live_enabled": True,
                "production_allowed": True,
                "action": action,
                "would_execute": selected,
                "reason": reason,
                "turbo_score": float(candidate["raw_score"]),
                "raw": raw,
                "gated": {"action": action, "would_execute": selected, "reason": reason, "blocked_by": None},
            },
            "entry_quality_model": {
                "mode": "CURRENT_BRAIN_LIVE",
                "execute": False,
                "production_allowed": False,
                "status": "LOADED",
                "symbol": normalized,
                "model_version": batch["model_identifier"],
                "model_scope": "global",
                "entry_quality_score": float(layer["eqm_score"]),
                "tail_risk_score": float(layer["rv2_tail_risk"]),
                "recommendation": "ALLOW_SHADOW" if eq_allowed else "BLOCK_SHADOW",
                "reason": "CURRENT_BRAIN_EQM",
                "feature_status": "ok",
                "feature_parity_pct": 100.0,
                "missing_features_count": 0,
                "missing_features": [],
            },
            "decision_brain": {
                "contract_version": CANONICAL_DECISION_CONTRACT,
                "authority": CANONICAL_LIVE_AUTHORITY,
                "mode": "CURRENT_BRAIN_LIVE",
                "execute": selected,
                "selected": selected,
                "production_allowed": True,
                "status": "LOADED",
                "model_version": batch["model_identifier"],
                "model_sha256": batch["model_sha256"],
                "bundle_sha256": batch["bundle_sha256"],
                "configuration_sha256": batch["configuration_sha256"],
                "feature_schema": result["feature_schema"],
                "feature_count": result["feature_count"],
                "fallback": False,
                "symbol": normalized,
                "side": side if side in {"LONG", "SHORT"} else "HOLD",
                "decision": "ENTER_NOW" if selected else "DO_NOT_ENTER",
                "recommendation": "ENTER_NOW" if selected else "DO_NOT_ENTER",
                "reason": reason,
                "feature_status": "ok",
                "feature_parity_pct": 100.0,
                "missing_features_count": 0,
                "missing_features": [],
            },
            "entry_quality_v2": (
                dict(v2)
                if isinstance(v2, Mapping)
                else {
                    "schema_id": "aegis-entry-quality-v2-http-shadow-v1",
                    "mode": "UNAVAILABLE",
                    "status": "NOT_CONFIGURED",
                    "selected": False,
                    "exchange_authority": False,
                }
            ),
        },
        "metadata": {
            "fallback": False,
            "trace_id": trace_id,
            "service_version": SERVICE_VERSION,
            "decision_cycle_id": batch["decision_cycle_id"],
            "model_sha256": batch["model_sha256"],
            "bundle_sha256": batch["bundle_sha256"],
            "configuration_sha256": batch["configuration_sha256"],
            "canonical_raw_score": float(candidate["raw_score"]),
            "canonical_calibrated_score": float(candidate["calibrated_score"]),
            "directional_estimator_count": count,
            "leverage_recommendation": "NOT_PRESENT",
            "position_fraction": "NOT_PRESENT",
        },
    }


class CurrentBrainDecisionService:
    def __init__(
        self,
        engine: CurrentBrainEngine,
        provider: SnapshotProvider,
        *,
        cache_seconds: float = 30.0,
        research_observer: ResearchBatchObserver | None = None,
    ) -> None:
        self.engine = engine
        self.provider = provider
        self.cache_seconds = cache_seconds
        self.research_observer = research_observer
        self.started_at = datetime.now(timezone.utc)
        self._cache: tuple[float, Mapping[str, Any]] | None = None
        self._lock = threading.Lock()
        self.request_count = 0
        self.error_count = 0
        self.research_error_count = 0
        self.last_inference_at: datetime | None = None

    def initialize(self) -> None:
        self.engine.initialize()

    def _batch(self) -> Mapping[str, Any]:
        now = time.monotonic()
        with self._lock:
            if self._cache is not None and now - self._cache[0] <= self.cache_seconds:
                return self._cache[1]
            batch = self.engine.evaluate(self.provider.snapshot())
            if self.research_observer is not None:
                try:
                    overlay = self.research_observer.observe_batch(batch)
                except Exception as exc:
                    self.research_error_count += 1
                    mode = str(getattr(self.research_observer.mode, "value", self.research_observer.mode))
                    if mode == "LIVE":
                        raise CurrentBrainError(
                            "AEGIS_ENTRY_QUALITY_V2_LIVE_EVALUATION_FAILED"
                        ) from exc
                    overlay = {}
                batch = {**batch, "_entry_quality_v2": overlay}
            self._cache = (now, batch)
            self.last_inference_at = datetime.now(timezone.utc)
            return batch

    def predict(self, symbol: str, trace_id: str) -> Mapping[str, Any]:
        self.request_count += 1
        try:
            return compatibility_response(self._batch(), symbol, trace_id)
        except Exception:
            self.error_count += 1
            raise

    def health(self) -> Mapping[str, Any]:
        health = {
            "status": "ready" if self.engine.ready else "not_ready",
            "ready": self.engine.ready,
            "model_identifier": CANDIDATE_IDENTITY,
            "model_sha256": MODEL_ARTIFACT_SHA256,
            "bundle_sha256": MODEL_BUNDLE_SHA256,
            "configuration_sha256": CONFIGURATION_SHA256,
            "feature_schema": FEATURE_SCHEMA,
            "feature_count": FEATURE_COUNT,
            "service_version": SERVICE_VERSION,
            "startup_timestamp": self.started_at.isoformat().replace("+00:00", "Z"),
            "latest_inference_timestamp": (
                self.last_inference_at.isoformat().replace("+00:00", "Z")
                if self.last_inference_at
                else None
            ),
            "requests": self.request_count,
            "errors": self.error_count,
            "research_errors": self.research_error_count,
        }
        if self.research_observer is not None:
            health["entry_quality_v2"] = dict(self.research_observer.health())
        return health


def trace_id() -> str:
    material = f"{time.time_ns()}:{threading.get_ident()}".encode("ascii")
    return hashlib.sha256(material).hexdigest()[:24]
