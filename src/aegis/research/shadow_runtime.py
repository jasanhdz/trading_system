"""Append-only entry-quality V2 Shadow and paper evidence runtime.

This module has no exchange or order surface. In SHADOW mode it observes the
same canonical batch served to TypeScript while leaving the operational
selection untouched. LIVE mode is rejected unless an exact model artifact and
promotion record are configured and hash-verified.
"""

from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Protocol

import yaml

from ..config import CANONICAL_SYMBOLS
from ..utils import Sha256HashProvider, canonical_json, sha256_file, to_primitive
from .entry_quality import EntryQualityInputs, MaeAwareScoreContract, score_entry_quality
from .regime_v2 import (
    FactorizedRegimeAnalyzer,
    RegimeV2Observation,
    RegimeV2Settings,
)

if TYPE_CHECKING:
    from ..training.short_opportunity import ShortOpportunityArtifact


class EntryQualityV2Error(RuntimeError):
    pass


class EntryQualityV2Mode(str, Enum):
    SHADOW = "SHADOW"
    LIVE = "LIVE"


@dataclass(frozen=True)
class EntryQualityV2RuntimeConfig:
    schema_version: str
    mode: EntryQualityV2Mode
    config_path: Path
    config_sha256: str
    journal_root: Path
    signal_journal: Path
    outcome_journal: Path
    horizon_bars: int
    round_trip_cost_fraction: float
    opportunity_source: str
    opportunity_artifact_path: Path | None
    opportunity_artifact_sha256: str | None
    maximum_candidates_per_cycle: int
    minimum_score: float
    require_current_layer_eligible: bool
    score_contract: MaeAwareScoreContract
    regime_settings: RegimeV2Settings
    promotion_record_path: Path | None
    promotion_record_sha256: str | None


class BatchObserver(Protocol):
    @property
    def mode(self) -> EntryQualityV2Mode: ...

    def observe_batch(self, batch: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def health(self) -> Mapping[str, Any]: ...


def _mapping(value: Any, identity: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EntryQualityV2Error(f"{identity} must be a mapping")
    return value


def _optional_path(root: Path, value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return (root / str(value)).resolve()


def _bounded_fraction(value: Any, identity: str, *, upper_inclusive: bool = True) -> float:
    result = float(value)
    upper_ok = result <= 1.0 if upper_inclusive else result < 1.0
    if not math.isfinite(result) or result < 0.0 or not upper_ok:
        raise EntryQualityV2Error(f"{identity} is invalid")
    return result


def load_entry_quality_v2_config(
    path: Path,
    *,
    repo_root: Path | None = None,
) -> EntryQualityV2RuntimeConfig:
    resolved = path.resolve()
    root = (repo_root or resolved.parent.parent).resolve()
    try:
        payload = _mapping(yaml.safe_load(resolved.read_text(encoding="utf-8")), "entry_quality_v2")
    except (OSError, yaml.YAMLError) as exc:
        raise EntryQualityV2Error("AEGIS_ENTRY_QUALITY_V2_CONFIG_INVALID") from exc
    if payload.get("schema_version") != "aegis-entry-quality-v2-runtime-v1":
        raise EntryQualityV2Error("AEGIS_ENTRY_QUALITY_V2_CONFIG_SCHEMA_INVALID")
    try:
        mode = EntryQualityV2Mode(str(payload["mode"]))
    except (KeyError, ValueError) as exc:
        raise EntryQualityV2Error("AEGIS_ENTRY_QUALITY_V2_MODE_INVALID") from exc

    evidence = _mapping(payload.get("evidence"), "evidence")
    opportunity = _mapping(payload.get("opportunity"), "opportunity")
    selection = _mapping(payload.get("selection"), "selection")
    score = _mapping(payload.get("score"), "score")
    regime = _mapping(payload.get("regime"), "regime")
    promotion = _mapping(payload.get("live_promotion"), "live_promotion")

    journal_root = (resolved.parent / str(evidence["journal_root"])).resolve()
    data_root = (root / "data").resolve()
    if journal_root != data_root and data_root not in journal_root.parents:
        raise EntryQualityV2Error("AEGIS_ENTRY_QUALITY_V2_JOURNAL_ROOT_PROHIBITED")
    signal_journal = journal_root / str(evidence["signal_journal"])
    outcome_journal = journal_root / str(evidence["outcome_journal"])
    if signal_journal.parent != journal_root or outcome_journal.parent != journal_root:
        raise EntryQualityV2Error("AEGIS_ENTRY_QUALITY_V2_JOURNAL_PATH_PROHIBITED")

    horizon_bars = int(evidence["horizon_bars"])
    maximum_candidates = int(selection["maximum_candidates_per_cycle"])
    minimum_score = float(selection["minimum_score"])
    if horizon_bars <= 0 or maximum_candidates <= 0 or not math.isfinite(minimum_score):
        raise EntryQualityV2Error("AEGIS_ENTRY_QUALITY_V2_RUNTIME_LIMIT_INVALID")

    score_contract = MaeAwareScoreContract(
        schema_version=str(score["schema_version"]),
        qmae_penalty=float(score["qmae_penalty"]),
        tail_risk_penalty=float(score["tail_risk_penalty"]),
        maximum_qmae_fraction=float(score["maximum_qmae_fraction"]),
        maximum_tail_probability=float(score["maximum_tail_probability"]),
        require_bearish_trend_context=bool(score["require_bearish_trend_context"]),
    )
    regime_settings = RegimeV2Settings(
        schema_version=str(regime["schema_version"]),
        history_window=int(regime["history_window"]),
        minimum_history=int(regime["minimum_history"]),
        low_volatility_quantile=float(regime["low_volatility_quantile"]),
        high_volatility_quantile=float(regime["high_volatility_quantile"]),
        trend_enter_fraction=float(regime["trend_enter_fraction"]),
        trend_exit_fraction=float(regime["trend_exit_fraction"]),
        trend_strength_enter=float(regime["trend_strength_enter"]),
        trend_strength_exit=float(regime["trend_strength_exit"]),
        chop_enter_fraction=float(regime["chop_enter_fraction"]),
        chop_exit_fraction=float(regime["chop_exit_fraction"]),
        high_expansion_ratio=float(regime["high_expansion_ratio"]),
        low_expansion_ratio=float(regime["low_expansion_ratio"]),
        minimum_state_bars=int(regime["minimum_state_bars"]),
    )
    artifact_path = _optional_path(root, opportunity.get("artifact_path"))
    promotion_path = _optional_path(root, promotion.get("promotion_record_path"))
    config = EntryQualityV2RuntimeConfig(
        schema_version=str(payload["schema_version"]),
        mode=mode,
        config_path=resolved,
        config_sha256=sha256_file(resolved),
        journal_root=journal_root,
        signal_journal=signal_journal,
        outcome_journal=outcome_journal,
        horizon_bars=horizon_bars,
        round_trip_cost_fraction=_bounded_fraction(
            evidence["round_trip_cost_fraction"], "round_trip_cost_fraction"
        ),
        opportunity_source=str(opportunity["source"]),
        opportunity_artifact_path=artifact_path,
        opportunity_artifact_sha256=(
            str(opportunity["artifact_sha256"]) if opportunity.get("artifact_sha256") else None
        ),
        maximum_candidates_per_cycle=maximum_candidates,
        minimum_score=minimum_score,
        require_current_layer_eligible=bool(selection["require_current_layer_eligible"]),
        score_contract=score_contract,
        regime_settings=regime_settings,
        promotion_record_path=promotion_path,
        promotion_record_sha256=(
            str(promotion["promotion_record_sha256"])
            if promotion.get("promotion_record_sha256")
            else None
        ),
    )
    _validate_live_promotion(config, promotion)
    return config


def _validate_live_promotion(
    config: EntryQualityV2RuntimeConfig,
    promotion: Mapping[str, Any],
) -> None:
    artifact_values = (
        config.opportunity_artifact_path,
        config.opportunity_artifact_sha256,
    )
    if any(value is None for value in artifact_values) and any(
        value is not None for value in artifact_values
    ):
        raise EntryQualityV2Error(
            "AEGIS_ENTRY_QUALITY_V2_ARTIFACT_AUTHORITY_INCOMPLETE"
        )
    if config.opportunity_artifact_path is not None:
        assert config.opportunity_artifact_sha256 is not None
        if (
            not config.opportunity_artifact_path.is_file()
            or sha256_file(config.opportunity_artifact_path)
            != config.opportunity_artifact_sha256
        ):
            raise EntryQualityV2Error(
                "AEGIS_ENTRY_QUALITY_V2_ARTIFACT_AUTHORITY_MISMATCH"
            )
    if config.mode is EntryQualityV2Mode.SHADOW:
        return
    if not bool(promotion.get("required", True)):
        raise EntryQualityV2Error("AEGIS_ENTRY_QUALITY_V2_LIVE_PROMOTION_REQUIRED")
    required_source = str(promotion.get("required_opportunity_source", ""))
    if config.opportunity_source != required_source:
        raise EntryQualityV2Error("AEGIS_ENTRY_QUALITY_V2_LIVE_MODEL_NOT_PROMOTED")
    required = (
        config.opportunity_artifact_path,
        config.opportunity_artifact_sha256,
        config.promotion_record_path,
        config.promotion_record_sha256,
    )
    if any(value is None for value in required):
        raise EntryQualityV2Error("AEGIS_ENTRY_QUALITY_V2_LIVE_AUTHORITY_INCOMPLETE")
    assert config.opportunity_artifact_path is not None
    assert config.opportunity_artifact_sha256 is not None
    assert config.promotion_record_path is not None
    assert config.promotion_record_sha256 is not None
    if (
        sha256_file(config.opportunity_artifact_path) != config.opportunity_artifact_sha256
        or sha256_file(config.promotion_record_path) != config.promotion_record_sha256
    ):
        raise EntryQualityV2Error("AEGIS_ENTRY_QUALITY_V2_LIVE_AUTHORITY_MISMATCH")
    try:
        record = _mapping(
            json.loads(config.promotion_record_path.read_text(encoding="utf-8")),
            "live_promotion_record",
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise EntryQualityV2Error(
            "AEGIS_ENTRY_QUALITY_V2_LIVE_PROMOTION_INVALID"
        ) from exc
    if (
        record.get("schema_id")
        != "aegis-entry-quality-v2-live-promotion-v1"
        or record.get("state") != "OWNER_APPROVED_FOR_LIVE_SWITCH"
        or record.get("artifact_sha256")
        != config.opportunity_artifact_sha256
        or record.get("opportunity_source") != required_source
        or record.get("automatic_activation") is not False
    ):
        raise EntryQualityV2Error(
            "AEGIS_ENTRY_QUALITY_V2_LIVE_PROMOTION_INVALID"
        )


class _AppendOnlyJournal:
    def __init__(self, path: Path, identity_field: str) -> None:
        self.path = path
        self.identity_field = identity_field
        self.rows: list[dict[str, Any]] = []
        self.payloads: dict[str, str] = {}
        if path.exists():
            self._recover()

    def _recover(self) -> None:
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line:
                    raise EntryQualityV2Error("AEGIS_ENTRY_QUALITY_V2_JOURNAL_INVALID")
                row = json.loads(line)
                identity = str(row[self.identity_field])
                payload = canonical_json(row)
                existing = self.payloads.get(identity)
                if existing is not None and existing != payload:
                    raise EntryQualityV2Error("AEGIS_ENTRY_QUALITY_V2_JOURNAL_CONFLICT")
                if existing is None:
                    self.rows.append(row)
                    self.payloads[identity] = payload
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise EntryQualityV2Error("AEGIS_ENTRY_QUALITY_V2_JOURNAL_INVALID") from exc

    def append(self, row: Mapping[str, Any]) -> bool:
        identity = str(row[self.identity_field])
        payload = canonical_json(row)
        existing = self.payloads.get(identity)
        if existing == payload:
            return False
        if existing is not None:
            raise EntryQualityV2Error("AEGIS_ENTRY_QUALITY_V2_JOURNAL_CONFLICT")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(payload + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(self.path, 0o600)
        normalized = json.loads(payload)
        self.rows.append(normalized)
        self.payloads[identity] = payload
        return True


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise EntryQualityV2Error("AEGIS_ENTRY_QUALITY_V2_TIMESTAMP_INVALID")
    return parsed.astimezone(timezone.utc)


class EntryQualityV2ShadowRuntime:
    """Observe canonical batches and persist counterfactual evidence."""

    def __init__(self, config: EntryQualityV2RuntimeConfig) -> None:
        self.config = config
        self._hashing = Sha256HashProvider()
        self._regime = FactorizedRegimeAnalyzer(config.regime_settings)
        self._opportunity_model: ShortOpportunityArtifact | None = None
        if config.opportunity_artifact_path is not None:
            from ..training.short_opportunity import load_short_opportunity_artifact

            self._opportunity_model = load_short_opportunity_artifact(
                config.opportunity_artifact_path
            )
        self._signals = _AppendOnlyJournal(config.signal_journal, "event_id")
        self._outcomes = _AppendOnlyJournal(config.outcome_journal, "event_id")
        self._lock = threading.Lock()
        self._processed_cycles = {
            str(row["decision_cycle_id"]) for row in self._signals.rows
        }
        self._processed_market_timestamps = {
            str(row["market_timestamp"]) for row in self._signals.rows
        }
        self._restore_regime()
        self.last_observation_at: datetime | None = None
        self.observation_errors = 0

    @property
    def mode(self) -> EntryQualityV2Mode:
        return self.config.mode

    def _restore_regime(self) -> None:
        observations: list[RegimeV2Observation] = []
        for row in self._signals.rows:
            value = _mapping(row["regime_input"], "regime_input")
            observations.append(
                RegimeV2Observation(
                    symbol=str(row["symbol"]),
                    timestamp=_timestamp(str(row["market_timestamp"])),
                    market_direction_6=float(value["market_direction_6"]),
                    range_mean_24=float(value["range_mean_24"]),
                    range_expansion=float(value["range_expansion"]),
                    chop_12=float(value["chop_12"]),
                    trend_strength_12=float(value["trend_strength_12"]),
                )
            )
        for observation in sorted(observations, key=lambda item: (item.timestamp, item.symbol)):
            self._regime.observe(observation)

    @staticmethod
    def _mean_prediction(predictions: list[Mapping[str, Any]], field: str) -> float:
        values = [float(item[field]) for item in predictions]
        if not values or not all(math.isfinite(value) for value in values):
            raise EntryQualityV2Error("AEGIS_ENTRY_QUALITY_V2_PREDICTION_INVALID")
        return math.fsum(values) / len(values)

    def observe_batch(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        cycle = str(batch["decision_cycle_id"])
        market_timestamp = str(batch["market_timestamp"])
        with self._lock:
            if cycle in self._processed_cycles:
                return self._overlay_for_cycle(cycle)
            if market_timestamp in self._processed_market_timestamps:
                return self._overlay_for_market_timestamp(market_timestamp)
            try:
                rows = self._build_rows(batch)
                ranking = sorted(
                    (
                        row for row in rows
                        if row["v2"]["eligible"]
                        and row["v2"]["score"] >= self.config.minimum_score
                    ),
                    key=lambda row: (-float(row["v2"]["score"]), str(row["symbol"])),
                )
                selected = {
                    str(row["symbol"])
                    for row in ranking[: self.config.maximum_candidates_per_cycle]
                }
                for row in rows:
                    row["v2"]["selected"] = row["symbol"] in selected
                    row["v2"]["paper_action"] = (
                        "SHORT" if row["symbol"] in selected else "NO_TRADE"
                    )
                    self._signals.append(row)
                self._processed_cycles.add(cycle)
                self._processed_market_timestamps.add(market_timestamp)
                self._mature_outcomes()
                self.last_observation_at = datetime.now(timezone.utc)
                return self._overlay_for_cycle(cycle)
            except Exception:
                self.observation_errors += 1
                raise

    def _build_rows(self, batch: Mapping[str, Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        results = _mapping(batch["results"], "results")
        if set(results) != set(CANONICAL_SYMBOLS):
            raise EntryQualityV2Error("AEGIS_ENTRY_QUALITY_V2_SYMBOL_POPULATION_INVALID")
        for symbol in CANONICAL_SYMBOLS:
            result = _mapping(results[symbol], symbol)
            features = _mapping(result["research_features"], "research_features")
            normalized_features = _mapping(
                result["research_normalized_features"],
                "research_normalized_features",
            )
            predictions = [
                _mapping(value, "prediction") for value in result["predictions"]
            ]
            layer = _mapping(result["layer"], "layer")
            regime_input = {
                name: float(features[name])
                for name in (
                    "market_direction_6",
                    "range_mean_24",
                    "range_expansion",
                    "chop_12",
                    "trend_strength_12",
                )
            }
            regime = self._regime.observe(
                RegimeV2Observation(
                    symbol=symbol,
                    timestamp=_timestamp(str(result["market_timestamp"])),
                    **regime_input,
                )
            )
            expected_short_return = -self._mean_prediction(predictions, "expected_return")
            opportunity_probability = (
                self._opportunity_model.probability(
                    symbol,
                    [
                        float(features[name])
                        for name in self._opportunity_model.feature_names
                    ],
                )
                if self._opportunity_model is not None
                else self._mean_prediction(predictions, "quality_probability")
            )
            qmae = layer.get("qmae_q90")
            if qmae is None:
                qmae = 0.0
            score = score_entry_quality(
                EntryQualityInputs(
                    symbol=symbol,
                    expected_short_return=expected_short_return,
                    opportunity_probability=opportunity_probability,
                    qmae_q90=float(qmae),
                    tail_risk_probability=float(layer["rv2_tail_risk"]),
                    qmae_valid=layer.get("qmae_q90") is not None,
                    calibration_valid=all(
                        bool(item["calibration_valid"]) for item in predictions
                    ),
                    regime=regime,
                ),
                self.config.score_contract,
            )
            current_layer_eligible = bool(layer["eligible"])
            eligible = score.eligible and (
                current_layer_eligible
                or not self.config.require_current_layer_eligible
            )
            event_id = self._hashing.digest_value(
                {
                    "schema": "aegis-entry-quality-v2-shadow-event-v1",
                    "cycle": batch["decision_cycle_id"],
                    "symbol": symbol,
                    "config": self.config.config_sha256,
                }
            )
            market_bar = _mapping(result["market_bar"], "market_bar")
            rows.append(
                {
                    "schema_id": "aegis-entry-quality-v2-shadow-event-v1",
                    "event_id": event_id,
                    "decision_cycle_id": batch["decision_cycle_id"],
                    "market_timestamp": result["market_timestamp"],
                    "symbol": symbol,
                    "timeframe": "5m",
                    "mode": self.config.mode.value,
                    "config_sha256": self.config.config_sha256,
                    "feature_schema": result["feature_schema"],
                    "feature_vector_hash": result["feature_vector_hash"],
                    "feature_values": {
                        name: float(features[name]) for name in sorted(features)
                    },
                    "normalized_feature_values": {
                        name: float(normalized_features[name])
                        for name in sorted(normalized_features)
                    },
                    "regime_input": regime_input,
                    "market_bar": {
                        name: float(market_bar[name])
                        for name in ("open", "high", "low", "close")
                    },
                    "control": {
                        "selected": bool(result["selected"]),
                        "side": str(result["candidate"]["side"]),
                        "raw_score": float(result["candidate"]["raw_score"]),
                        "calibrated_score": float(
                            result["candidate"]["calibrated_score"]
                        ),
                        "reason_codes": [
                            str(value) for value in result["candidate"]["reason_codes"]
                        ],
                    },
                    "v2": {
                        "status": (
                            "PROMOTED_MODEL_ACTIVE"
                            if (
                                self._opportunity_model is not None
                                and self.config.mode is EntryQualityV2Mode.LIVE
                            )
                            else "VALIDATED_MODEL_SHADOW_ACTIVE"
                            if self._opportunity_model is not None
                            else "OBSERVATIONAL_PROXY_NOT_PROMOTED"
                        ),
                        "opportunity_source": self.config.opportunity_source,
                        "opportunity_probability": opportunity_probability,
                        "expected_short_return": expected_short_return,
                        "qmae_q90": float(qmae),
                        "tail_risk_probability": float(layer["rv2_tail_risk"]),
                        "current_layer_eligible": current_layer_eligible,
                        "eligible": eligible,
                        "score": score.score,
                        "score_components": {
                            "expected_clean_return": score.expected_clean_return,
                            "qmae_penalty": score.qmae_penalty,
                            "tail_risk_penalty": score.tail_risk_penalty,
                        },
                        "reason_codes": list(score.reason_codes),
                        "regime": to_primitive(regime),
                        "selected": False,
                        "paper_action": "NO_TRADE",
                        "exchange_authority": (
                            self.config.mode is EntryQualityV2Mode.LIVE
                        ),
                    },
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
                future = rows[index + 1:index + 1 + self.config.horizon_bars]
                if len(future) < self.config.horizon_bars:
                    continue
                entry = float(signal["market_bar"]["close"])
                exit_price = float(future[-1]["market_bar"]["close"])
                lows = [float(row["market_bar"]["low"]) for row in future]
                highs = [float(row["market_bar"]["high"]) for row in future]
                if entry <= 0.0:
                    raise EntryQualityV2Error(
                        "AEGIS_ENTRY_QUALITY_V2_MARKET_PRICE_INVALID"
                    )
                gross = (entry - exit_price) / entry
                mfe = max(0.0, (entry - min(lows)) / entry)
                mae = max(0.0, (max(highs) - entry) / entry)
                self._outcomes.append(
                    {
                        "schema_id": "aegis-entry-quality-v2-paper-outcome-v1",
                        "event_id": event_id,
                        "symbol": signal["symbol"],
                        "side": "SHORT",
                        "signal_timestamp": signal["market_timestamp"],
                        "maturity_timestamp": future[-1]["market_timestamp"],
                        "horizon_bars": self.config.horizon_bars,
                        "gross_return_fraction": gross,
                        "net_return_fraction": (
                            gross - self.config.round_trip_cost_fraction
                        ),
                        "mfe_fraction": mfe,
                        "mae_fraction": mae,
                        "round_trip_cost_fraction": (
                            self.config.round_trip_cost_fraction
                        ),
                        "paper_selected": bool(signal["v2"]["selected"]),
                        "execution": (
                            "PAPER_SELECTED"
                            if bool(signal["v2"]["selected"])
                            else "COUNTERFACTUAL_OBSERVATION"
                        ),
                        "exchange_mutations": 0,
                    }
                )

    def _overlay_for_cycle(self, cycle: str) -> Mapping[str, Any]:
        rows = [
            row for row in self._signals.rows
            if str(row["decision_cycle_id"]) == cycle
        ]
        return {
            str(row["symbol"]): {
                "schema_id": "aegis-entry-quality-v2-http-shadow-v1",
                "mode": self.config.mode.value,
                "config_sha256": self.config.config_sha256,
                "status": row["v2"]["status"],
                "selected": bool(row["v2"]["selected"]),
                "paper_action": row["v2"]["paper_action"],
                "score": float(row["v2"]["score"]),
                "eligible": bool(row["v2"]["eligible"]),
                "opportunity_source": row["v2"]["opportunity_source"],
                "regime": row["v2"]["regime"],
                "exchange_authority": self.config.mode is EntryQualityV2Mode.LIVE,
            }
            for row in rows
        }

    def _overlay_for_market_timestamp(
        self,
        market_timestamp: str,
    ) -> Mapping[str, Any]:
        rows = [
            row
            for row in self._signals.rows
            if str(row["market_timestamp"]) == market_timestamp
        ]
        latest_by_symbol = {
            str(row["symbol"]): row for row in rows
        }
        return {
            symbol: {
                "schema_id": "aegis-entry-quality-v2-http-shadow-v1",
                "mode": self.config.mode.value,
                "config_sha256": row["config_sha256"],
                "status": row["v2"]["status"],
                "selected": bool(row["v2"]["selected"]),
                "paper_action": row["v2"]["paper_action"],
                "score": float(row["v2"]["score"]),
                "eligible": bool(row["v2"]["eligible"]),
                "opportunity_source": row["v2"]["opportunity_source"],
                "regime": row["v2"]["regime"],
                "exchange_authority": (
                    self.config.mode is EntryQualityV2Mode.LIVE
                ),
            }
            for symbol, row in latest_by_symbol.items()
        }

    def health(self) -> Mapping[str, Any]:
        return {
            "status": "ACTIVE",
            "mode": self.config.mode.value,
            "config_sha256": self.config.config_sha256,
            "signal_records": len(self._signals.rows),
            "paper_outcomes": len(self._outcomes.rows),
            "observation_errors": self.observation_errors,
            "latest_observation_timestamp": (
                _iso(self.last_observation_at) if self.last_observation_at else None
            ),
            "exchange_authority": False,
            "opportunity_source": self.config.opportunity_source,
            "opportunity_artifact_loaded": self._opportunity_model is not None,
            "live_promotion_ready": (
                self.config.mode is EntryQualityV2Mode.LIVE
            ),
        }


class UnavailableEntryQualityV2Observer:
    """Health-visible fallback used only when SHADOW initialization fails."""

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
            "exchange_authority": False,
        }


def build_entry_quality_v2_observer(
    config_path: Path,
    *,
    repo_root: Path | None = None,
) -> BatchObserver:
    try:
        config = load_entry_quality_v2_config(config_path, repo_root=repo_root)
        return EntryQualityV2ShadowRuntime(config)
    except Exception as exc:
        try:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            requested_mode = str(payload.get("mode", "")) if isinstance(payload, Mapping) else ""
        except Exception:
            requested_mode = ""
        if requested_mode == EntryQualityV2Mode.LIVE.value:
            raise
        return UnavailableEntryQualityV2Observer(type(exc).__name__)
