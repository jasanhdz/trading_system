"""Hybrid directional committee observer with no execution authority."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from ..config import CANONICAL_SYMBOLS
from ..features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from ..training.hybrid_directional import (
    DirectionalSide,
    HybridDirectionalArtifact,
    load_hybrid_directional_artifact,
)
from ..utils import Sha256HashProvider, sha256_file
from .shadow_runtime import EntryQualityV2Mode, _AppendOnlyJournal, _mapping

CONFIG_SCHEMA = "aegis-hybrid-directional-shadow-runtime-v1"
SIGNAL_SCHEMA = "aegis-hybrid-directional-shadow-signal-v1"
OUTCOME_SCHEMA = "aegis-hybrid-directional-shadow-outcome-v1"


class HybridDirectionalShadowError(RuntimeError):
    pass


@dataclass(frozen=True)
class HybridDirectionalShadowConfig:
    config_path: Path
    config_sha256: str
    artifact_path: Path
    artifact_sha256: str
    signal_journal: Path
    outcome_journal: Path
    horizon_bars: int
    round_trip_cost_fraction: float


def load_hybrid_directional_shadow_config(
    path: Path, *, repo_root: Path
) -> HybridDirectionalShadowConfig:
    resolved = path.resolve()
    root = repo_root.resolve()
    try:
        payload = _mapping(yaml.safe_load(resolved.read_text()), "hybrid_shadow")
        artifact = _mapping(payload["artifact"], "artifact")
        evidence = _mapping(payload["evidence"], "evidence")
        promotion = _mapping(payload["promotion"], "promotion")
        artifact_path = (root / str(artifact["path"])).resolve()
        readiness_path = (root / str(artifact["readiness_path"])).resolve()
        journal_root = (resolved.parent / str(evidence["journal_root"])).resolve()
    except (OSError, KeyError, TypeError, yaml.YAMLError) as exc:
        raise HybridDirectionalShadowError(
            "AEGIS_HYBRID_SHADOW_CONFIG_INVALID"
        ) from exc
    data_root = (root / "data").resolve()
    if (
        payload.get("schema_version") != CONFIG_SCHEMA
        or payload.get("enabled") is not True
        or payload.get("mode") != "SHADOW"
        or payload.get("runtime_authority") != "OBSERVATIONAL_ONLY"
        or payload.get("feature_schema") != FEATURE_SCHEMA_VERSION
        or int(payload.get("feature_count", 0)) != len(FEATURE_NAMES)
        or promotion.get("automatic_live_activation") is not False
        or promotion.get("current_live_eligible") is not False
        or promotion.get("owner_authorization_required") is not True
        or not artifact_path.is_file()
        or sha256_file(artifact_path) != str(artifact["sha256"])
        or not readiness_path.is_file()
        or sha256_file(readiness_path) != str(artifact["readiness_sha256"])
        or artifact.get("offline_validation_state") != "FAILED"
        or (journal_root != data_root and data_root not in journal_root.parents)
    ):
        raise HybridDirectionalShadowError("AEGIS_HYBRID_SHADOW_AUTHORITY_INVALID")
    signal_journal = journal_root / str(evidence["signal_journal"])
    outcome_journal = journal_root / str(evidence["outcome_journal"])
    if signal_journal.parent != journal_root or outcome_journal.parent != journal_root:
        raise HybridDirectionalShadowError(
            "AEGIS_HYBRID_SHADOW_JOURNAL_PATH_PROHIBITED"
        )
    horizon = int(evidence["horizon_bars"])
    cost = float(evidence["round_trip_cost_fraction"])
    if horizon <= 0 or not math.isfinite(cost) or not 0.0 <= cost < 1.0:
        raise HybridDirectionalShadowError("AEGIS_HYBRID_SHADOW_LIMIT_INVALID")
    return HybridDirectionalShadowConfig(
        config_path=resolved,
        config_sha256=sha256_file(resolved),
        artifact_path=artifact_path,
        artifact_sha256=str(artifact["sha256"]),
        signal_journal=signal_journal,
        outcome_journal=outcome_journal,
        horizon_bars=horizon,
        round_trip_cost_fraction=cost,
    )


class HybridDirectionalShadowRuntime:
    def __init__(
        self,
        config: HybridDirectionalShadowConfig,
        artifact: HybridDirectionalArtifact,
    ) -> None:
        self.config = config
        self.artifact = artifact
        self._hashing = Sha256HashProvider()
        self._signals = _AppendOnlyJournal(config.signal_journal, "event_id")
        self._outcomes = _AppendOnlyJournal(config.outcome_journal, "event_id")
        self._processed_timestamps = {
            str(row["market_timestamp"]) for row in self._signals.rows
        }
        self._matured = {str(row["source_event_id"]) for row in self._outcomes.rows}
        self._lock = threading.Lock()
        self.last_observation_at: datetime | None = None
        self.observation_errors = 0

    @property
    def mode(self) -> EntryQualityV2Mode:
        return EntryQualityV2Mode.SHADOW

    def observe_batch(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        timestamp = str(batch["market_timestamp"])
        with self._lock:
            if timestamp not in self._processed_timestamps:
                try:
                    for row in self._build_rows(batch):
                        self._signals.append(row)
                    self._processed_timestamps.add(timestamp)
                    self._mature_outcomes()
                    self.last_observation_at = datetime.now(timezone.utc)
                except Exception:
                    self.observation_errors += 1
                    raise
            return self._overlay(timestamp)

    def _build_rows(self, batch: Mapping[str, Any]) -> list[dict[str, Any]]:
        results = _mapping(batch["results"], "results")
        if set(results) != set(CANONICAL_SYMBOLS):
            raise HybridDirectionalShadowError("AEGIS_HYBRID_SHADOW_SYMBOLS_INVALID")
        rows = []
        for symbol in CANONICAL_SYMBOLS:
            result = _mapping(results[symbol], symbol)
            features = _mapping(result["research_features"], "research_features")
            vector = tuple(float(features[name]) for name in FEATURE_NAMES)
            if not all(math.isfinite(value) for value in vector):
                raise HybridDirectionalShadowError(
                    "AEGIS_HYBRID_SHADOW_FEATURE_NONFINITE"
                )
            predictions = {
                side.value: self._prediction_mapping(
                    self.artifact.predict(side, vector)
                )
                for side in DirectionalSide
            }
            event_id = self._hashing.digest_value(
                {
                    "schema": SIGNAL_SCHEMA,
                    "timestamp": result["market_timestamp"],
                    "symbol": symbol,
                    "feature_vector_hash": result["feature_vector_hash"],
                    "artifact": self.config.artifact_sha256,
                }
            )
            bar = _mapping(result["market_bar"], "market_bar")
            rows.append(
                {
                    "schema_id": SIGNAL_SCHEMA,
                    "event_id": event_id,
                    "decision_cycle_id": batch["decision_cycle_id"],
                    "market_timestamp": result["market_timestamp"],
                    "symbol": symbol,
                    "timeframe": "5m",
                    "feature_schema": result["feature_schema"],
                    "feature_vector_hash": result["feature_vector_hash"],
                    "artifact_sha256": self.config.artifact_sha256,
                    "market_bar": {
                        name: float(bar[name])
                        for name in ("open", "high", "low", "close")
                    },
                    "predictions": predictions,
                    "control_selected": bool(result["selected"]),
                    "control_side": str(result["candidate"]["side"]),
                    "selection_effect": "NONE",
                    "exchange_authority": False,
                    "exchange_mutations": 0,
                }
            )
        return rows

    @staticmethod
    def _prediction_mapping(value) -> Mapping[str, Any]:
        return {
            "side": value.side.value,
            "opportunity_probability": value.opportunity_probability,
            "danger_probability": value.danger_probability,
            "mae_q50": value.mae_q50,
            "mae_q90": value.mae_q90,
            "mfe_q50": value.mfe_q50,
            "net_return_mean": value.net_return_mean,
            "shadow_rank_score": value.shadow_rank_score,
            "selection_effect": "NONE",
            "exchange_authority": False,
        }

    def _mature_outcomes(self) -> None:
        for symbol in CANONICAL_SYMBOLS:
            symbol_rows = [row for row in self._signals.rows.recent if row["symbol"] == symbol]
            if len(symbol_rows) <= self.config.horizon_bars:
                continue
            source = symbol_rows[-self.config.horizon_bars - 1]
            source_id = str(source["event_id"])
            if source_id in self._matured:
                continue
            future = symbol_rows[-self.config.horizon_bars :]
            entry = float(future[0]["market_bar"]["open"])
            directional = {}
            for side in DirectionalSide:
                if side is DirectionalSide.LONG:
                    favorable = [
                        max(0.0, (float(row["market_bar"]["high"]) - entry) / entry)
                        for row in future
                    ]
                    adverse = [
                        max(0.0, (entry - float(row["market_bar"]["low"])) / entry)
                        for row in future
                    ]
                    terminal = float(future[-1]["market_bar"]["close"]) / entry - 1.0
                else:
                    favorable = [
                        max(0.0, (entry - float(row["market_bar"]["low"])) / entry)
                        for row in future
                    ]
                    adverse = [
                        max(0.0, (float(row["market_bar"]["high"]) - entry) / entry)
                        for row in future
                    ]
                    terminal = 1.0 - float(future[-1]["market_bar"]["close"]) / entry
                directional[side.value] = {
                    "mfe_fraction": max(favorable),
                    "mae_fraction": max(adverse),
                    "net_return_after_costs": terminal
                    - self.config.round_trip_cost_fraction,
                }
            outcome_id = self._hashing.digest_value(
                {"schema": OUTCOME_SCHEMA, "source_event_id": source_id}
            )
            self._outcomes.append(
                {
                    "schema_id": OUTCOME_SCHEMA,
                    "event_id": outcome_id,
                    "source_event_id": source_id,
                    "market_timestamp": source["market_timestamp"],
                    "matured_at": future[-1]["market_timestamp"],
                    "symbol": symbol,
                    "horizon_bars": self.config.horizon_bars,
                    "directional_outcomes": directional,
                    "selection_effect": "NONE",
                    "exchange_authority": False,
                    "exchange_mutations": 0,
                }
            )
            self._matured.add(source_id)

    def _overlay(self, timestamp: str) -> Mapping[str, Any]:
        latest = {
            str(row["symbol"]): row
            for row in self._signals.rows.recent
            if str(row["market_timestamp"]) == timestamp
        }
        return {
            symbol: {
                "schema_id": "aegis-hybrid-directional-http-shadow-v1",
                "mode": "SHADOW",
                "status": "OFFLINE_VALIDATION_FAILED_OBSERVATION_ONLY",
                "predictions": dict(latest[symbol]["predictions"]),
                "selection_effect": "NONE",
                "exchange_authority": False,
                "exchange_mutations": 0,
            }
            for symbol in CANONICAL_SYMBOLS
            if symbol in latest
        }

    def health(self) -> Mapping[str, Any]:
        return {
            "status": "ACTIVE",
            "mode": "SHADOW",
            "offline_validation": "FAILED",
            "signal_records": len(self._signals.rows),
            "outcome_records": len(self._outcomes.rows),
            "observation_errors": self.observation_errors,
            "selection_effect": "NONE",
            "exchange_authority": False,
            "exchange_mutations": 0,
            "last_observation_at": (
                self.last_observation_at.isoformat().replace("+00:00", "Z")
                if self.last_observation_at
                else None
            ),
        }


def build_hybrid_directional_shadow_observer(
    config_path: Path, *, repo_root: Path
) -> HybridDirectionalShadowRuntime:
    config = load_hybrid_directional_shadow_config(config_path, repo_root=repo_root)
    return HybridDirectionalShadowRuntime(
        config, load_hybrid_directional_artifact(config.artifact_path)
    )
