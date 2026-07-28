"""Shadow-only evidence for distinct SHORT probability semantics."""

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
from ..training.short_opportunity import (
    ShortOpportunityArtifact,
    load_short_opportunity_artifact,
)
from ..utils import Sha256HashProvider, sha256_file
from .shadow_runtime import (
    EntryQualityV2Mode,
    _AppendOnlyJournal,
    _mapping,
)


class ShortProbabilityShadowError(RuntimeError):
    pass


@dataclass(frozen=True)
class ShortProbabilityShadowConfig:
    config_path: Path
    config_sha256: str
    profitability_artifact_path: Path
    profitability_artifact_sha256: str
    clean_entry_artifact_path: Path
    clean_entry_artifact_sha256: str
    signal_journal: Path
    outcome_journal: Path
    horizon_bars: int
    round_trip_cost_fraction: float
    maximum_clean_mae_fraction: float


def load_short_probability_shadow_config(
    path: Path,
    *,
    repo_root: Path,
) -> ShortProbabilityShadowConfig:
    resolved = path.resolve()
    try:
        payload = _mapping(
            yaml.safe_load(resolved.read_text(encoding="utf-8")),
            "short_probability_shadow",
        )
        if (
            payload.get("schema_version")
            != "aegis-short-probability-semantics-shadow-v1"
            or payload.get("mode") != "SHADOW"
            or payload.get("runtime_authority") != "OBSERVATIONAL_ONLY"
        ):
            raise ShortProbabilityShadowError(
                "AEGIS_SHORT_PROBABILITY_SHADOW_CONFIG_INVALID"
            )
        artifacts = _mapping(payload["artifacts"], "artifacts")
        profitability = _mapping(artifacts["profitability"], "profitability")
        clean_entry = _mapping(artifacts["clean_entry"], "clean_entry")
        evidence = _mapping(payload["evidence"], "evidence")
        profitability_path = (repo_root / str(profitability["path"])).resolve()
        clean_entry_path = (repo_root / str(clean_entry["path"])).resolve()
        journal_root = (resolved.parent / str(evidence["journal_root"])).resolve()
        data_root = (repo_root / "data").resolve()
        if journal_root != data_root and data_root not in journal_root.parents:
            raise ShortProbabilityShadowError(
                "AEGIS_SHORT_PROBABILITY_JOURNAL_ROOT_PROHIBITED"
            )
        signal_journal = journal_root / str(evidence["signal_journal"])
        outcome_journal = journal_root / str(evidence["outcome_journal"])
        if (
            signal_journal.parent != journal_root
            or outcome_journal.parent != journal_root
        ):
            raise ShortProbabilityShadowError(
                "AEGIS_SHORT_PROBABILITY_JOURNAL_PATH_PROHIBITED"
            )
        for artifact_path, expected_hash in (
            (profitability_path, str(profitability["sha256"])),
            (clean_entry_path, str(clean_entry["sha256"])),
        ):
            if (
                not artifact_path.is_file()
                or sha256_file(artifact_path) != expected_hash
            ):
                raise ShortProbabilityShadowError(
                    "AEGIS_SHORT_PROBABILITY_ARTIFACT_AUTHORITY_MISMATCH"
                )
        horizon = int(evidence["horizon_bars"])
        cost = float(evidence["round_trip_cost_fraction"])
        maximum_mae = float(evidence["maximum_clean_mae_fraction"])
        if (
            horizon <= 0
            or not math.isfinite(cost)
            or not 0.0 <= cost < 1.0
            or not math.isfinite(maximum_mae)
            or not 0.0 < maximum_mae < 1.0
        ):
            raise ShortProbabilityShadowError(
                "AEGIS_SHORT_PROBABILITY_OUTCOME_CONTRACT_INVALID"
            )
        return ShortProbabilityShadowConfig(
            config_path=resolved,
            config_sha256=sha256_file(resolved),
            profitability_artifact_path=profitability_path,
            profitability_artifact_sha256=str(profitability["sha256"]),
            clean_entry_artifact_path=clean_entry_path,
            clean_entry_artifact_sha256=str(clean_entry["sha256"]),
            signal_journal=signal_journal,
            outcome_journal=outcome_journal,
            horizon_bars=horizon,
            round_trip_cost_fraction=cost,
            maximum_clean_mae_fraction=maximum_mae,
        )
    except (KeyError, OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        if isinstance(exc, ShortProbabilityShadowError):
            raise
        raise ShortProbabilityShadowError(
            "AEGIS_SHORT_PROBABILITY_SHADOW_CONFIG_INVALID"
        ) from exc


class ShortProbabilityShadowRuntime:
    def __init__(
        self,
        config: ShortProbabilityShadowConfig,
        profitability_model: ShortOpportunityArtifact,
        clean_entry_model: ShortOpportunityArtifact,
    ) -> None:
        if (
            profitability_model.probability_semantics
            != "TERMINAL_NET_POSITIVE_H12_AFTER_COSTS"
            or clean_entry_model.probability_semantics != "CLEAN_ENTRY_LOW_MAE_H12"
        ):
            raise ShortProbabilityShadowError(
                "AEGIS_SHORT_PROBABILITY_SEMANTICS_MISMATCH"
            )
        self.config = config
        self._profitability_model = profitability_model
        self._clean_entry_model = clean_entry_model
        self._hashing = Sha256HashProvider()
        self._signals = _AppendOnlyJournal(config.signal_journal, "event_id")
        self._outcomes = _AppendOnlyJournal(config.outcome_journal, "event_id")
        self._processed_cycles = {
            str(row["decision_cycle_id"]) for row in self._signals.rows
        }
        self._lock = threading.Lock()
        self.last_observation_at: datetime | None = None
        self.observation_errors = 0

    @property
    def mode(self) -> EntryQualityV2Mode:
        return EntryQualityV2Mode.SHADOW

    def observe_batch(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        cycle = str(batch["decision_cycle_id"])
        with self._lock:
            if cycle not in self._processed_cycles:
                try:
                    for row in self._build_rows(batch):
                        self._signals.append(row)
                    self._processed_cycles.add(cycle)
                    self._mature_outcomes()
                    self.last_observation_at = datetime.now(timezone.utc)
                except Exception:
                    self.observation_errors += 1
                    return {}
            return self._overlay(cycle)

    def _build_rows(self, batch: Mapping[str, Any]) -> list[dict[str, Any]]:
        cycle = str(batch["decision_cycle_id"])
        results = _mapping(batch["results"], "results")
        if (
            set(results) != set(CANONICAL_SYMBOLS)
            or batch.get("feature_schema") != FEATURE_SCHEMA_VERSION
            or int(batch.get("feature_count", 0)) != len(FEATURE_NAMES)
        ):
            raise ShortProbabilityShadowError(
                "AEGIS_SHORT_PROBABILITY_CANONICAL_BATCH_INVALID"
            )
        rows: list[dict[str, Any]] = []
        for symbol in CANONICAL_SYMBOLS:
            result = _mapping(results[symbol], symbol)
            features = _mapping(result["research_features"], "research_features")
            predictions = tuple(
                _mapping(item, "prediction") for item in result["predictions"]
            )
            if not predictions:
                raise ShortProbabilityShadowError(
                    "AEGIS_SHORT_PROBABILITY_PREDICTIONS_MISSING"
                )
            short_side_authority = math.fsum(
                float(item["short_probability"]) for item in predictions
            ) / len(predictions)
            vector = [float(features[name]) for name in FEATURE_NAMES]
            profitability = self._profitability_model.probability(symbol, vector)
            clean_entry = self._clean_entry_model.probability(symbol, vector)
            if not all(
                math.isfinite(value)
                for value in (short_side_authority, profitability, clean_entry)
            ):
                raise ShortProbabilityShadowError("AEGIS_SHORT_PROBABILITY_NONFINITE")
            bar = _mapping(result["market_bar"], "market_bar")
            event_id = self._hashing.digest_value(
                {
                    "schema": "aegis-short-probability-shadow-signal-v1",
                    "cycle": cycle,
                    "symbol": symbol,
                    "config": self.config.config_sha256,
                }
            )
            rows.append(
                {
                    "schema_id": "aegis-short-probability-shadow-signal-v1",
                    "event_id": event_id,
                    "decision_cycle_id": cycle,
                    "market_timestamp": result["market_timestamp"],
                    "symbol": symbol,
                    "timeframe": "5m",
                    "feature_schema": result["feature_schema"],
                    "feature_vector_hash": result["feature_vector_hash"],
                    "market_bar": {
                        name: float(bar[name])
                        for name in ("open", "high", "low", "close")
                    },
                    "probabilities": {
                        "short_side_authority": short_side_authority,
                        "short_side_authority_semantics": (
                            "SIDE_AUTHORITY_NOT_PROFITABILITY_CONFIDENCE"
                        ),
                        "terminal_net_positive_h12_after_costs": profitability,
                        "terminal_net_positive_semantics": (
                            "P_NET_SHORT_RETURN_GT_ZERO_AT_H12_AFTER_COSTS"
                        ),
                        "clean_entry_low_mae_h12": clean_entry,
                        "clean_entry_semantics": (
                            "P_CLEAN_SHORT_PATH_WITH_BOUNDED_MAE_H12"
                        ),
                    },
                    "control": {
                        "selected": bool(result["selected"]),
                        "side": str(result["candidate"]["side"]),
                    },
                    "legacy_expected_return_semantics": "UNDER_SEPARATE_AUDIT",
                    "selection_effect": "NONE",
                    "exchange_authority": False,
                    "exchange_mutations": 0,
                }
            )
        return rows

    def _mature_outcomes(self) -> None:
        existing = self._outcomes.payloads
        by_symbol: dict[str, list[dict[str, Any]]] = {
            symbol: [] for symbol in CANONICAL_SYMBOLS
        }
        for row in self._signals.rows:
            by_symbol[str(row["symbol"])].append(row)
        for rows in by_symbol.values():
            rows.sort(key=lambda row: str(row["market_timestamp"]))
            for index, signal in enumerate(rows):
                event_id = str(signal["event_id"])
                if event_id in existing:
                    continue
                future = rows[index + 1 : index + 1 + self.config.horizon_bars]
                if len(future) < self.config.horizon_bars:
                    continue
                entry = float(signal["market_bar"]["close"])
                if entry <= 0.0:
                    raise ShortProbabilityShadowError(
                        "AEGIS_SHORT_PROBABILITY_PRICE_INVALID"
                    )
                exit_price = float(future[-1]["market_bar"]["close"])
                lows = [float(row["market_bar"]["low"]) for row in future]
                highs = [float(row["market_bar"]["high"]) for row in future]
                gross = (entry - exit_price) / entry
                net = gross - self.config.round_trip_cost_fraction
                mfe = max(0.0, (entry - min(lows)) / entry)
                mae = max(0.0, (max(highs) - entry) / entry)
                self._outcomes.append(
                    {
                        "schema_id": "aegis-short-probability-shadow-outcome-v1",
                        "event_id": event_id,
                        "symbol": signal["symbol"],
                        "signal_timestamp": signal["market_timestamp"],
                        "maturity_timestamp": future[-1]["market_timestamp"],
                        "horizon_bars": self.config.horizon_bars,
                        "gross_short_return_fraction": gross,
                        "net_short_return_fraction": net,
                        "mfe_fraction": mfe,
                        "mae_fraction": mae,
                        "terminal_net_positive": net > 0.0,
                        "clean_entry_low_mae": (
                            net > 0.0 and mae <= self.config.maximum_clean_mae_fraction
                        ),
                        "round_trip_cost_fraction": (
                            self.config.round_trip_cost_fraction
                        ),
                        "exchange_mutations": 0,
                    }
                )

    def _overlay(self, cycle: str) -> Mapping[str, Any]:
        rows = [
            row for row in self._signals.rows if str(row["decision_cycle_id"]) == cycle
        ]
        return {
            str(row["symbol"]): {
                "schema_id": "aegis-short-probability-http-shadow-v1",
                "mode": "SHADOW",
                "status": "ACTIVE",
                "probabilities": dict(row["probabilities"]),
                "selection_effect": "NONE",
                "exchange_authority": False,
            }
            for row in rows
        }

    def health(self) -> Mapping[str, Any]:
        return {
            "status": "ACTIVE",
            "mode": "SHADOW",
            "config_sha256": self.config.config_sha256,
            "signal_records": len(self._signals.rows),
            "matured_outcomes": len(self._outcomes.rows),
            "observation_errors": self.observation_errors,
            "last_observation_at": (
                self.last_observation_at.isoformat().replace("+00:00", "Z")
                if self.last_observation_at
                else None
            ),
            "profitability_semantics": (
                self._profitability_model.probability_semantics
            ),
            "clean_entry_semantics": self._clean_entry_model.probability_semantics,
            "selection_effect": "NONE",
            "exchange_authority": False,
            "exchange_mutations": 0,
        }


class UnavailableShortProbabilityShadowObserver:
    def __init__(self, reason: str) -> None:
        self.reason = reason

    @property
    def mode(self) -> EntryQualityV2Mode:
        return EntryQualityV2Mode.SHADOW

    def observe_batch(self, _: Mapping[str, Any]) -> Mapping[str, Any]:
        return {}

    def health(self) -> Mapping[str, Any]:
        return {
            "status": "UNAVAILABLE",
            "mode": "SHADOW",
            "reason": self.reason,
            "selection_effect": "NONE",
            "exchange_authority": False,
            "exchange_mutations": 0,
        }


def build_short_probability_shadow_observer(
    config_path: Path,
    *,
    repo_root: Path,
) -> ShortProbabilityShadowRuntime | UnavailableShortProbabilityShadowObserver:
    try:
        config = load_short_probability_shadow_config(
            config_path,
            repo_root=repo_root,
        )
        return ShortProbabilityShadowRuntime(
            config,
            load_short_opportunity_artifact(config.profitability_artifact_path),
            load_short_opportunity_artifact(config.clean_entry_artifact_path),
        )
    except Exception as exc:
        return UnavailableShortProbabilityShadowObserver(type(exc).__name__)
