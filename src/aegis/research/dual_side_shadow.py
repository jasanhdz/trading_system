"""Bilateral entry-quality evidence collector with no exchange authority."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import yaml

from ..config import CANONICAL_SYMBOLS
from ..training.long_opportunity import (
    LongOpportunityArtifact,
    load_long_opportunity_artifact,
)
from ..utils import Sha256HashProvider, sha256_file, to_primitive
from .committee_v2_shadow import (
    CommitteeV2ShadowRuntime,
    UnavailableCommitteeV2ShadowObserver,
    build_committee_v2_shadow_observer,
)
from .regime_v2 import (
    DirectionRegime,
    FactorizedRegimeAnalyzer,
    RegimeV2Observation,
    StructureRegime,
)
from .shadow_runtime import (
    BatchObserver,
    EntryQualityV2Mode,
    _AppendOnlyJournal,
    _mapping,
    _timestamp,
    load_entry_quality_v2_config,
)


class DualSideShadowError(RuntimeError):
    pass


class DualSideMode(str, Enum):
    SHADOW = "SHADOW"
    LIVE = "LIVE"


@dataclass(frozen=True)
class DualSideShadowConfig:
    mode: DualSideMode
    config_path: Path
    config_sha256: str
    artifact_path: Path
    artifact_sha256: str
    readiness_path: Path
    readiness_sha256: str
    signal_journal: Path
    outcome_journal: Path
    horizon_bars: int
    round_trip_cost_fraction: float
    maximum_candidates_per_cycle: int


def load_dual_side_shadow_config(
    path: Path,
    *,
    repo_root: Path,
) -> DualSideShadowConfig:
    resolved = path.resolve()
    try:
        payload = _mapping(yaml.safe_load(resolved.read_text()), "dual_side")
        if (
            payload.get("schema_version")
            != "aegis-entry-quality-v3-dual-shadow-runtime-v1"
            or payload.get("side") != "LONG"
        ):
            raise DualSideShadowError("AEGIS_DUAL_SIDE_CONFIG_INVALID")
        mode = DualSideMode(str(payload["mode"]))
        artifact = _mapping(payload["artifact"], "artifact")
        evidence = _mapping(payload["evidence"], "evidence")
        selection = _mapping(payload["selection"], "selection")
        promotion = _mapping(payload["promotion"], "promotion")
        artifact_path = (repo_root / str(artifact["path"])).resolve()
        readiness_path = (repo_root / str(artifact["readiness_path"])).resolve()
        journal_root = (resolved.parent / str(evidence["journal_root"])).resolve()
        data_root = (repo_root / "data").resolve()
        if journal_root != data_root and data_root not in journal_root.parents:
            raise DualSideShadowError("AEGIS_DUAL_SIDE_JOURNAL_ROOT_PROHIBITED")
        if (
            not artifact_path.is_file()
            or sha256_file(artifact_path) != str(artifact["sha256"])
            or not readiness_path.is_file()
            or sha256_file(readiness_path) != str(artifact["readiness_sha256"])
        ):
            raise DualSideShadowError("AEGIS_DUAL_SIDE_ARTIFACT_AUTHORITY_MISMATCH")
        if mode is DualSideMode.LIVE:
            if (
                artifact.get("offline_validation_state") != "PASSED"
                or not bool(promotion.get("current_live_eligible"))
                or not bool(promotion.get("require_shadow_evidence_pass"))
            ):
                raise DualSideShadowError(
                    "AEGIS_DUAL_SIDE_LIVE_PROMOTION_PROHIBITED"
                )
        signal_journal = journal_root / str(evidence["signal_journal"])
        outcome_journal = journal_root / str(evidence["outcome_journal"])
        if (
            signal_journal.parent != journal_root
            or outcome_journal.parent != journal_root
        ):
            raise DualSideShadowError("AEGIS_DUAL_SIDE_JOURNAL_PATH_PROHIBITED")
        horizon = int(evidence["horizon_bars"])
        cost = float(evidence["round_trip_cost_fraction"])
        maximum = int(selection["maximum_candidates_per_cycle"])
        if (
            horizon <= 0
            or maximum <= 0
            or not math.isfinite(cost)
            or not 0.0 <= cost < 1.0
        ):
            raise DualSideShadowError("AEGIS_DUAL_SIDE_RUNTIME_LIMIT_INVALID")
        return DualSideShadowConfig(
            mode=mode,
            config_path=resolved,
            config_sha256=sha256_file(resolved),
            artifact_path=artifact_path,
            artifact_sha256=str(artifact["sha256"]),
            readiness_path=readiness_path,
            readiness_sha256=str(artifact["readiness_sha256"]),
            signal_journal=signal_journal,
            outcome_journal=outcome_journal,
            horizon_bars=horizon,
            round_trip_cost_fraction=cost,
            maximum_candidates_per_cycle=maximum,
        )
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        if isinstance(exc, DualSideShadowError):
            raise
        raise DualSideShadowError("AEGIS_DUAL_SIDE_CONFIG_INVALID") from exc


class DualSideEntryQualityShadowRuntime:
    def __init__(
        self,
        config: DualSideShadowConfig,
        artifact: LongOpportunityArtifact,
        regime_analyzer: FactorizedRegimeAnalyzer,
    ) -> None:
        self.config = config
        self.artifact = artifact
        self._regime = regime_analyzer
        self._hashing = Sha256HashProvider()
        self._signals = _AppendOnlyJournal(config.signal_journal, "event_id")
        self._outcomes = _AppendOnlyJournal(config.outcome_journal, "event_id")
        self._processed_cycles = {
            str(row["decision_cycle_id"]) for row in self._signals.rows
        }
        self._processed_timestamps = {
            str(row["market_timestamp"]) for row in self._signals.rows
        }
        self._lock = threading.Lock()
        self.last_observation_at: datetime | None = None
        self.observation_errors = 0

    @property
    def mode(self) -> EntryQualityV2Mode:
        return EntryQualityV2Mode.SHADOW

    def observe_batch(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        cycle = str(batch["decision_cycle_id"])
        timestamp = str(batch["market_timestamp"])
        with self._lock:
            if cycle in self._processed_cycles or timestamp in self._processed_timestamps:
                return self._overlay(cycle=cycle, timestamp=timestamp)
            try:
                rows = self._build_rows(batch)
                ranking = sorted(
                    (
                        row
                        for row in rows
                        if float(row["long_shadow"]["score"]) > 0.0
                    ),
                    key=lambda row: (
                        -float(row["long_shadow"]["score"]),
                        str(row["symbol"]),
                    ),
                )
                selected = {
                    str(row["symbol"])
                    for row in ranking[
                        : self.config.maximum_candidates_per_cycle
                    ]
                }
                regime_ranking = [
                    row
                    for row in ranking
                    if bool(row["long_shadow"]["bull_trend_context"])
                ]
                regime_selected = {
                    str(row["symbol"])
                    for row in regime_ranking[
                        : self.config.maximum_candidates_per_cycle
                    ]
                }
                for row in rows:
                    shadow = row["long_shadow"]
                    shadow["model_only_selected"] = row["symbol"] in selected
                    shadow["regime_confirmed_selected"] = (
                        row["symbol"] in regime_selected
                    )
                    shadow["paper_action"] = (
                        "LONG"
                        if shadow["regime_confirmed_selected"]
                        else "NO_TRADE"
                    )
                    self._signals.append(row)
                self._processed_cycles.add(cycle)
                self._processed_timestamps.add(timestamp)
                self._mature_outcomes()
                self.last_observation_at = datetime.now(timezone.utc)
                return self._overlay(cycle=cycle, timestamp=timestamp)
            except Exception:
                self.observation_errors += 1
                raise

    def _build_rows(self, batch: Mapping[str, Any]) -> list[dict[str, Any]]:
        results = _mapping(batch["results"], "results")
        if set(results) != set(CANONICAL_SYMBOLS):
            raise DualSideShadowError("AEGIS_DUAL_SIDE_SYMBOL_POPULATION_INVALID")
        rows = []
        for symbol in CANONICAL_SYMBOLS:
            result = _mapping(results[symbol], symbol)
            features = _mapping(result["research_features"], "features")
            predictions = [
                _mapping(value, "prediction") for value in result["predictions"]
            ]
            if not predictions:
                raise DualSideShadowError("AEGIS_DUAL_SIDE_PREDICTION_MISSING")
            long_probability = math.fsum(
                float(item["long_probability"]) for item in predictions
            ) / len(predictions)
            short_probability = math.fsum(
                float(item["short_probability"]) for item in predictions
            ) / len(predictions)
            expected_long_return = math.fsum(
                float(item["expected_return"]) for item in predictions
            ) / len(predictions)
            opportunity = self.artifact.probability(
                symbol,
                [float(features[name]) for name in self.artifact.feature_names],
            )
            regime = self._regime.observe(
                RegimeV2Observation(
                    symbol=symbol,
                    timestamp=_timestamp(str(result["market_timestamp"])),
                    market_direction_6=float(features["market_direction_6"]),
                    range_mean_24=float(features["range_mean_24"]),
                    range_expansion=float(features["range_expansion"]),
                    chop_12=float(features["chop_12"]),
                    trend_strength_12=float(features["trend_strength_12"]),
                )
            )
            # The promoted base bundle is SHORT-only. Shadow ranks the genuine
            # LONG artifact directly instead of multiplying it by a SHORT-model
            # expected-return head that is not authoritative for LONG.
            score = opportunity
            event_id = self._hashing.digest_value(
                {
                    "schema": "aegis-entry-quality-v3-long-shadow-event-v1",
                    "cycle": batch["decision_cycle_id"],
                    "symbol": symbol,
                    "config": self.config.config_sha256,
                }
            )
            bar = _mapping(result["market_bar"], "market_bar")
            layer = _mapping(result["layer"], "layer")
            rows.append(
                {
                    "schema_id": "aegis-entry-quality-v3-long-shadow-event-v1",
                    "event_id": event_id,
                    "decision_cycle_id": batch["decision_cycle_id"],
                    "market_timestamp": result["market_timestamp"],
                    "symbol": symbol,
                    "timeframe": "5m",
                    "feature_schema": result["feature_schema"],
                    "feature_vector_hash": result["feature_vector_hash"],
                    "market_bar": {
                        name: float(bar[name])
                        for name in ("open", "high", "low", "close")
                    },
                    "control": {
                        "selected": bool(result["selected"]),
                        "side": str(result["candidate"]["side"]),
                    },
                    "long_shadow": {
                        "mode": "SHADOW",
                        "status": "OFFLINE_VALIDATION_FAILED_OBSERVATION_ONLY",
                        "artifact_sha256": self.config.artifact_sha256,
                        "base_long_probability": long_probability,
                        "base_short_probability": short_probability,
                        "base_expected_return": expected_long_return,
                        "base_directional_authority": "SHORT_ONLY",
                        "experimental_long_probability": opportunity,
                        "opportunity_probability": opportunity,
                        "score": score,
                        "score_authority": "OBSERVATIONAL_RANK_ONLY",
                        "qmae_observation": (
                            float(layer["qmae_q90"])
                            if layer.get("qmae_q90") is not None
                            else None
                        ),
                        "qmae_directional_authority": (
                            "NOT_ESTABLISHED_FOR_LONG"
                        ),
                        "regime": to_primitive(regime),
                        "bull_trend_context": (
                            regime.evidence_ready
                            and regime.direction is DirectionRegime.BULLISH
                            and regime.structure is StructureRegime.TREND
                        ),
                        "model_only_selected": False,
                        "regime_confirmed_selected": False,
                        "paper_action": "NO_TRADE",
                        "exchange_authority": False,
                    },
                }
            )
        return rows

    def _mature_outcomes(self) -> None:
        by_symbol = {symbol: [] for symbol in CANONICAL_SYMBOLS}
        for row in self._signals.rows:
            by_symbol[str(row["symbol"])].append(row)
        for rows in by_symbol.values():
            rows.sort(key=lambda row: str(row["market_timestamp"]))
            for index, signal in enumerate(rows):
                event_id = str(signal["event_id"])
                if event_id in self._outcomes.payloads:
                    continue
                future = rows[index + 1 : index + 1 + self.config.horizon_bars]
                if len(future) < self.config.horizon_bars:
                    continue
                entry = float(signal["market_bar"]["close"])
                if entry <= 0.0:
                    raise DualSideShadowError("AEGIS_DUAL_SIDE_PRICE_INVALID")
                exit_price = float(future[-1]["market_bar"]["close"])
                highs = [float(row["market_bar"]["high"]) for row in future]
                lows = [float(row["market_bar"]["low"]) for row in future]
                gross = (exit_price - entry) / entry
                self._outcomes.append(
                    {
                        "schema_id": (
                            "aegis-entry-quality-v3-long-paper-outcome-v1"
                        ),
                        "event_id": event_id,
                        "symbol": signal["symbol"],
                        "side": "LONG",
                        "signal_timestamp": signal["market_timestamp"],
                        "maturity_timestamp": future[-1]["market_timestamp"],
                        "gross_return_fraction": gross,
                        "net_return_fraction": (
                            gross - self.config.round_trip_cost_fraction
                        ),
                        "mfe_fraction": max(
                            0.0, (max(highs) - entry) / entry
                        ),
                        "mae_fraction": max(
                            0.0, (entry - min(lows)) / entry
                        ),
                        "model_only_selected": bool(
                            signal["long_shadow"]["model_only_selected"]
                        ),
                        "regime_confirmed_selected": bool(
                            signal["long_shadow"]["regime_confirmed_selected"]
                        ),
                        "exchange_mutations": 0,
                    }
                )

    def _overlay(self, *, cycle: str, timestamp: str) -> Mapping[str, Any]:
        rows = [
            row
            for row in self._signals.rows
            if str(row["decision_cycle_id"]) == cycle
            or str(row["market_timestamp"]) == timestamp
        ]
        latest = {str(row["symbol"]): row for row in rows}
        return {
            symbol: {
                "schema_id": "aegis-entry-quality-v3-long-http-shadow-v1",
                "mode": "SHADOW",
                "status": row["long_shadow"]["status"],
                "paper_action": row["long_shadow"]["paper_action"],
                "model_only_selected": bool(
                    row["long_shadow"]["model_only_selected"]
                ),
                "regime_confirmed_selected": bool(
                    row["long_shadow"]["regime_confirmed_selected"]
                ),
                "score": float(row["long_shadow"]["score"]),
                "exchange_authority": False,
            }
            for symbol, row in latest.items()
        }

    def health(self) -> Mapping[str, Any]:
        return {
            "status": "ACTIVE",
            "mode": "SHADOW",
            "direction": "LONG",
            "offline_validation": "FAILED",
            "signal_records": len(self._signals.rows),
            "paper_outcomes": len(self._outcomes.rows),
            "observation_errors": self.observation_errors,
            "exchange_authority": False,
            "exchange_mutations": 0,
            "last_observation_at": (
                self.last_observation_at.isoformat().replace("+00:00", "Z")
                if self.last_observation_at
                else None
            ),
        }


class CompositeResearchObserver:
    def __init__(
        self,
        primary: BatchObserver,
        dual: DualSideEntryQualityShadowRuntime,
        committee: (
            CommitteeV2ShadowRuntime
            | UnavailableCommitteeV2ShadowObserver
            | None
        ) = None,
    ):
        self.primary = primary
        self.dual = dual
        self.committee = committee

    @property
    def mode(self) -> EntryQualityV2Mode:
        return self.primary.mode

    def observe_batch(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        primary = self.primary.observe_batch(batch)
        dual = self.dual.observe_batch(batch)
        committee = (
            self.committee.observe_batch(
                batch,
                primary_overlay=primary,
                dual_overlay=dual,
            )
            if self.committee is not None
            else {}
        )
        return {
            symbol: {
                **dict(primary.get(symbol, {})),
                "dual_side_shadow": dict(dual.get(symbol, {})),
                "committee_v2_shadow": dict(committee.get(symbol, {})),
            }
            for symbol in CANONICAL_SYMBOLS
        }

    def health(self) -> Mapping[str, Any]:
        health = {
            **dict(self.primary.health()),
            "dual_side_shadow": dict(self.dual.health()),
        }
        if self.committee is not None:
            health["committee_v2_shadow"] = dict(self.committee.health())
        return health


def build_composite_research_observer(
    primary_config_path: Path,
    dual_config_path: Path,
    committee_config_path: Path | None = None,
    *,
    repo_root: Path,
) -> BatchObserver:
    from .shadow_runtime import build_entry_quality_v2_observer

    primary = build_entry_quality_v2_observer(
        primary_config_path, repo_root=repo_root
    )
    dual_config = load_dual_side_shadow_config(
        dual_config_path, repo_root=repo_root
    )
    artifact = load_long_opportunity_artifact(dual_config.artifact_path)
    primary_config = load_entry_quality_v2_config(
        primary_config_path, repo_root=repo_root
    )
    dual = DualSideEntryQualityShadowRuntime(
        dual_config,
        artifact,
        FactorizedRegimeAnalyzer(primary_config.regime_settings),
    )
    committee = (
        build_committee_v2_shadow_observer(
            committee_config_path,
            repo_root=repo_root,
        )
        if committee_config_path is not None
        else None
    )
    return CompositeResearchObserver(primary, dual, committee)
