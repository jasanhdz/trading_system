"""Preregistered calibrated-risk Committee V2.1 Shadow observer."""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from ..config import CANONICAL_SYMBOLS
from ..features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from ..utils import Sha256HashProvider, sha256_file
from .shadow_runtime import EntryQualityV2Mode, _AppendOnlyJournal, _mapping

PREREGISTRATION_SCHEMA = "aegis-specialized-committee-v21-preregistration-v1"
RUNTIME_SCHEMA = "aegis-specialized-committee-v21-shadow-runtime-v1"
ARTIFACT_SCHEMA = "aegis-specialized-committee-v21-calibrated-risk-v1"
SIGNAL_SCHEMA = "aegis-specialized-committee-v21-shadow-signal-v1"
OUTCOME_SCHEMA = "aegis-specialized-committee-v21-shadow-outcome-v1"
HTTP_SCHEMA = "aegis-specialized-committee-v21-http-shadow-v1"


class CommitteeV21ShadowError(RuntimeError):
    pass


@dataclass(frozen=True)
class CommitteeV21Contract:
    path: Path
    sha256: str
    experiment_id: str
    numeric_terms: tuple[str, ...]
    categorical_terms: Mapping[str, tuple[str, ...]]
    interaction_terms: tuple[tuple[str, str], ...]
    context_interactions: Mapping[str, tuple[str, ...]]
    probability_minimum: float
    probability_maximum: float
    horizon_bars: int
    round_trip_cost_fraction: float


@dataclass(frozen=True)
class CommitteeV21Artifact:
    path: Path
    sha256: str
    model_id: str
    contract_sha256: str
    numeric_means: Mapping[str, float]
    numeric_scales: Mapping[str, float]
    coefficients: Mapping[str, float]
    intercept: float
    calibration_slope: float
    calibration_intercept: float
    calibrated_risk_threshold: float
    training_end_utc: str
    calibration_end_utc: str


@dataclass(frozen=True)
class CommitteeV21ShadowConfig:
    path: Path
    sha256: str
    experiment_id: str
    contract: CommitteeV21Contract
    artifact: CommitteeV21Artifact
    signal_journal: Path
    outcome_journal: Path
    maximum_paper_entries_per_cycle: int
    evidence_start_utc: datetime


def _resolve(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _finite(value: Any, identity: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise CommitteeV21ShadowError(
            f"AEGIS_COMMITTEE_V21_VALUE_INVALID:{identity}"
        ) from exc
    if not math.isfinite(parsed):
        raise CommitteeV21ShadowError(f"AEGIS_COMMITTEE_V21_NONFINITE:{identity}")
    return parsed


def _timestamp(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CommitteeV21ShadowError("AEGIS_COMMITTEE_V21_TIMESTAMP_INVALID")
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _category_text(value: Any) -> str:
    raw = getattr(value, "value", value)
    text = str(raw)
    return text.rpartition(".")[2] if "." in text else text


def _tuple_mapping(
    value: Mapping[str, Any],
    identity: str,
) -> Mapping[str, tuple[str, ...]]:
    result = {}
    for name, entries in value.items():
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            raise CommitteeV21ShadowError(
                f"AEGIS_COMMITTEE_V21_CONTRACT_INVALID:{identity}:{name}"
            )
        result[str(name)] = tuple(str(entry) for entry in entries)
    return result


def load_committee_v21_contract(path: Path) -> CommitteeV21Contract:
    resolved = path.resolve()
    try:
        payload = _mapping(
            yaml.safe_load(resolved.read_text(encoding="utf-8")),
            "committee_v21_preregistration",
        )
        authority = _mapping(payload["authority"], "authority")
        outcome = _mapping(payload["outcome"], "outcome")
        model = _mapping(payload["model"], "model")
        clip = _mapping(model["probability_clip"], "probability_clip")
        categorical = _tuple_mapping(
            _mapping(payload["categorical_terms"], "categorical_terms"),
            "categorical_terms",
        )
        context = _tuple_mapping(
            _mapping(payload["context_interactions"], "context_interactions"),
            "context_interactions",
        )
        numeric = tuple(str(value) for value in payload["numeric_terms"])
        interactions = tuple(
            (str(pair[0]), str(pair[1])) for pair in payload["interaction_terms"]
        )
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise CommitteeV21ShadowError(
            "AEGIS_COMMITTEE_V21_PREREGISTRATION_INVALID"
        ) from exc
    if (
        payload.get("schema_version") != PREREGISTRATION_SCHEMA
        or payload.get("mode") != "SHADOW"
        or payload.get("runtime_authority") != "OBSERVATIONAL_ONLY"
        or authority.get("feature_schema") != FEATURE_SCHEMA_VERSION
        or int(authority.get("feature_count", 0)) != len(FEATURE_NAMES)
        or authority.get("exchange_authority") is not False
        or authority.get("network_access") is not False
        or authority.get("runtime_training") is not False
        or authority.get("automatic_promotion") is not False
        or authority.get("owner_review_required") is not True
        or model.get("family")
        != "L2_LOGISTIC_WITH_EXPLICIT_INTERACTIONS_AND_PLATT_CALIBRATION"
    ):
        raise CommitteeV21ShadowError(
            "AEGIS_COMMITTEE_V21_PREREGISTRATION_AUTHORITY_INVALID"
        )
    if (
        not numeric
        or len(numeric) != len(set(numeric))
        or set(categorical)
        != {
            "symbol",
            "direction_regime",
            "volatility_regime",
            "structure_regime",
        }
        or any(
            left not in numeric or right not in numeric for left, right in interactions
        )
        or any(term not in numeric for terms in context.values() for term in terms)
    ):
        raise CommitteeV21ShadowError("AEGIS_COMMITTEE_V21_FEATURE_CONTRACT_INVALID")
    minimum = _finite(clip["minimum"], "probability_minimum")
    maximum = _finite(clip["maximum"], "probability_maximum")
    horizon = int(outcome["horizon_bars"])
    cost = _finite(
        outcome["round_trip_cost_fraction"],
        "round_trip_cost_fraction",
    )
    if not 0.0 < minimum < maximum < 1.0 or horizon <= 0 or not 0.0 <= cost < 0.01:
        raise CommitteeV21ShadowError("AEGIS_COMMITTEE_V21_OUTCOME_CONTRACT_INVALID")
    return CommitteeV21Contract(
        path=resolved,
        sha256=sha256_file(resolved),
        experiment_id=str(payload["experiment_id"]),
        numeric_terms=numeric,
        categorical_terms=categorical,
        interaction_terms=interactions,
        context_interactions=context,
        probability_minimum=minimum,
        probability_maximum=maximum,
        horizon_bars=horizon,
        round_trip_cost_fraction=cost,
    )


def basis_term_names(contract: CommitteeV21Contract) -> tuple[str, ...]:
    names = [f"numeric:{name}" for name in contract.numeric_terms]
    for category, values in contract.categorical_terms.items():
        names.extend(f"category:{category}={value}" for value in values)
    names.extend(
        f"interaction:{left}*{right}" for left, right in contract.interaction_terms
    )
    for category, terms in contract.context_interactions.items():
        for value in contract.categorical_terms[category]:
            names.extend(f"context:{category}={value}*{term}" for term in terms)
    return tuple(names)


def load_committee_v21_artifact(
    path: Path,
    *,
    contract: CommitteeV21Contract,
) -> CommitteeV21Artifact:
    resolved = path.resolve()
    try:
        payload = _mapping(
            json.loads(resolved.read_text(encoding="utf-8")),
            "committee_v21_artifact",
        )
        standardization = _mapping(
            payload["standardization"],
            "standardization",
        )
        means = {
            str(name): _finite(value, f"mean:{name}")
            for name, value in _mapping(
                standardization["means"],
                "means",
            ).items()
        }
        scales = {
            str(name): _finite(value, f"scale:{name}")
            for name, value in _mapping(
                standardization["scales"],
                "scales",
            ).items()
        }
        logistic = _mapping(payload["logistic"], "logistic")
        coefficients = {
            str(name): _finite(value, f"coefficient:{name}")
            for name, value in _mapping(
                logistic["coefficients"],
                "coefficients",
            ).items()
        }
        calibration = _mapping(payload["calibration"], "calibration")
        threshold = _mapping(payload["threshold"], "threshold")
        provenance = _mapping(payload["provenance"], "provenance")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CommitteeV21ShadowError("AEGIS_COMMITTEE_V21_ARTIFACT_INVALID") from exc
    expected_terms = set(basis_term_names(contract))
    if (
        payload.get("schema_id") != ARTIFACT_SCHEMA
        or payload.get("contract_sha256") != contract.sha256
        or payload.get("feature_schema") != FEATURE_SCHEMA_VERSION
        or int(payload.get("feature_count", 0)) != len(FEATURE_NAMES)
        or set(means) != set(contract.numeric_terms)
        or set(scales) != set(contract.numeric_terms)
        or any(value <= 0.0 for value in scales.values())
        or set(coefficients) != expected_terms
        or calibration.get("method") != "PLATT_LOGISTIC"
        or threshold.get("source") != "CALIBRATION_RISK_DISTRIBUTION_Q70"
        or payload.get("exchange_authority") is not False
        or payload.get("automatic_promotion") is not False
    ):
        raise CommitteeV21ShadowError("AEGIS_COMMITTEE_V21_ARTIFACT_CONTRACT_MISMATCH")
    risk_threshold = _finite(
        threshold["calibrated_risk_probability"],
        "calibrated_risk_threshold",
    )
    if not 0.0 < risk_threshold < 1.0:
        raise CommitteeV21ShadowError("AEGIS_COMMITTEE_V21_THRESHOLD_INVALID")
    return CommitteeV21Artifact(
        path=resolved,
        sha256=sha256_file(resolved),
        model_id=str(payload["model_id"]),
        contract_sha256=str(payload["contract_sha256"]),
        numeric_means=means,
        numeric_scales=scales,
        coefficients=coefficients,
        intercept=_finite(logistic["intercept"], "intercept"),
        calibration_slope=_finite(calibration["slope"], "calibration_slope"),
        calibration_intercept=_finite(
            calibration["intercept"],
            "calibration_intercept",
        ),
        calibrated_risk_threshold=risk_threshold,
        training_end_utc=str(provenance["training_end_utc"]),
        calibration_end_utc=str(provenance["calibration_end_utc"]),
    )


def load_committee_v21_shadow_config(
    path: Path,
    *,
    repo_root: Path,
) -> CommitteeV21ShadowConfig:
    root = repo_root.resolve()
    resolved = path.resolve()
    try:
        payload = _mapping(
            yaml.safe_load(resolved.read_text(encoding="utf-8")),
            "committee_v21_runtime",
        )
        authority = _mapping(payload["authority"], "authority")
        artifact_ref = _mapping(payload["artifact"], "artifact")
        evidence = _mapping(payload["evidence"], "evidence")
        selector = _mapping(payload["counterfactual"], "counterfactual")
        contract_path = _resolve(root, authority["preregistration_path"])
        contract = load_committee_v21_contract(contract_path)
        artifact_path = _resolve(root, artifact_ref["path"])
        artifact = load_committee_v21_artifact(
            artifact_path,
            contract=contract,
        )
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        if isinstance(exc, CommitteeV21ShadowError):
            raise
        raise CommitteeV21ShadowError(
            "AEGIS_COMMITTEE_V21_RUNTIME_CONFIG_INVALID"
        ) from exc
    if (
        payload.get("schema_version") != RUNTIME_SCHEMA
        or payload.get("enabled") is not True
        or payload.get("mode") != "SHADOW"
        or payload.get("runtime_authority") != "OBSERVATIONAL_ONLY"
        or payload.get("experiment_id") != contract.experiment_id
        or authority.get("preregistration_sha256") != contract.sha256
        or artifact_ref.get("sha256") != artifact.sha256
        or authority.get("exchange_authority") is not False
        or authority.get("runtime_training") is not False
        or authority.get("automatic_promotion") is not False
        or selector.get("control_authority") != "CURRENT_CANONICAL_SELECTION"
        or selector.get("fabricated_votes_prohibited") is not True
    ):
        raise CommitteeV21ShadowError("AEGIS_COMMITTEE_V21_RUNTIME_AUTHORITY_INVALID")
    journal_root = _resolve(resolved.parent, evidence["journal_root"])
    data_root = (root / "data").resolve()
    if journal_root != data_root and data_root not in journal_root.parents:
        raise CommitteeV21ShadowError("AEGIS_COMMITTEE_V21_JOURNAL_ROOT_PROHIBITED")
    signals = journal_root / str(evidence["signal_journal"])
    outcomes = journal_root / str(evidence["outcome_journal"])
    if signals.parent != journal_root or outcomes.parent != journal_root:
        raise CommitteeV21ShadowError("AEGIS_COMMITTEE_V21_JOURNAL_PATH_PROHIBITED")
    maximum = int(selector["maximum_paper_entries_per_cycle"])
    if maximum != 1:
        raise CommitteeV21ShadowError("AEGIS_COMMITTEE_V21_SELECTION_LIMIT_INVALID")
    evidence_start = _timestamp(evidence["evidence_start_utc"])
    if evidence_start <= _timestamp(artifact.calibration_end_utc):
        raise CommitteeV21ShadowError("AEGIS_COMMITTEE_V21_EVIDENCE_WINDOW_INVALID")
    return CommitteeV21ShadowConfig(
        path=resolved,
        sha256=sha256_file(resolved),
        experiment_id=contract.experiment_id,
        contract=contract,
        artifact=artifact,
        signal_journal=signals,
        outcome_journal=outcomes,
        maximum_paper_entries_per_cycle=maximum,
        evidence_start_utc=evidence_start,
    )


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        decay = math.exp(-value)
        return 1.0 / (1.0 + decay)
    growth = math.exp(value)
    return growth / (1.0 + growth)


def standardized_numeric_values(
    contract: CommitteeV21Contract,
    artifact: CommitteeV21Artifact,
    observation: Mapping[str, Any],
) -> Mapping[str, float]:
    return standardized_numeric_values_from_stats(
        contract,
        artifact.numeric_means,
        artifact.numeric_scales,
        observation,
    )


def standardized_numeric_values_from_stats(
    contract: CommitteeV21Contract,
    means: Mapping[str, float],
    scales: Mapping[str, float],
    observation: Mapping[str, Any],
) -> Mapping[str, float]:
    if set(means) != set(contract.numeric_terms) or set(scales) != set(
        contract.numeric_terms
    ):
        raise CommitteeV21ShadowError(
            "AEGIS_COMMITTEE_V21_STANDARDIZATION_CONTRACT_MISMATCH"
        )
    return {
        name: (_finite(observation[name], name) - _finite(means[name], f"mean:{name}"))
        / _finite(scales[name], f"scale:{name}")
        for name in contract.numeric_terms
    }


def committee_v21_basis(
    contract: CommitteeV21Contract,
    artifact: CommitteeV21Artifact,
    observation: Mapping[str, Any],
) -> Mapping[str, float]:
    basis = committee_v21_basis_from_stats(
        contract,
        artifact.numeric_means,
        artifact.numeric_scales,
        observation,
    )
    if set(basis) != set(artifact.coefficients):
        raise CommitteeV21ShadowError("AEGIS_COMMITTEE_V21_BASIS_CONTRACT_MISMATCH")
    return basis


def committee_v21_basis_from_stats(
    contract: CommitteeV21Contract,
    means: Mapping[str, float],
    scales: Mapping[str, float],
    observation: Mapping[str, Any],
) -> Mapping[str, float]:
    numeric = standardized_numeric_values_from_stats(
        contract,
        means,
        scales,
        observation,
    )
    basis: dict[str, float] = {
        f"numeric:{name}": value for name, value in numeric.items()
    }
    for category, allowed in contract.categorical_terms.items():
        observed = str(observation[category])
        if observed not in allowed:
            raise CommitteeV21ShadowError(
                f"AEGIS_COMMITTEE_V21_CATEGORY_INVALID:{category}:{observed}"
            )
        for value in allowed:
            basis[f"category:{category}={value}"] = float(observed == value)
    for left, right in contract.interaction_terms:
        basis[f"interaction:{left}*{right}"] = numeric[left] * numeric[right]
    for category, terms in contract.context_interactions.items():
        observed = str(observation[category])
        for value in contract.categorical_terms[category]:
            active = float(observed == value)
            for term in terms:
                basis[f"context:{category}={value}*{term}"] = active * numeric[term]
    return basis


def committee_v21_calibrated_risk(
    contract: CommitteeV21Contract,
    artifact: CommitteeV21Artifact,
    observation: Mapping[str, Any],
) -> Mapping[str, float | str]:
    basis = committee_v21_basis(contract, artifact, observation)
    logit = artifact.intercept + math.fsum(
        artifact.coefficients[name] * value for name, value in basis.items()
    )
    raw_probability = min(
        contract.probability_maximum,
        max(contract.probability_minimum, _sigmoid(logit)),
    )
    raw_logit = math.log(raw_probability / (1.0 - raw_probability))
    calibrated = _sigmoid(
        artifact.calibration_slope * raw_logit + artifact.calibration_intercept
    )
    calibrated = min(
        contract.probability_maximum,
        max(contract.probability_minimum, calibrated),
    )
    threshold = artifact.calibrated_risk_threshold
    return {
        "raw_risk_probability": raw_probability,
        "calibrated_risk_probability": calibrated,
        "calibrated_risk_threshold": threshold,
        "risk_band": "HIGH" if calibrated > threshold else "LOW_OR_MEDIUM",
    }


def committee_v21_counterfactual(
    *,
    control_selected: bool,
    control_side: str,
    calibrated_risk_probability: float,
    calibrated_risk_threshold: float,
) -> Mapping[str, Any]:
    selected_short = control_selected and control_side.endswith("SHORT")
    if not selected_short:
        action = "DO_NOT_ENTER"
        reason = "CONTROL_NOT_SELECTED"
    elif calibrated_risk_probability <= calibrated_risk_threshold:
        action = "ENTER_NOW"
        reason = "CONTROL_SELECTED_CALIBRATED_RISK_RETAINED"
    else:
        action = "WAIT_CONFIRMATION"
        reason = "CONTROL_SELECTED_CALIBRATED_RISK_HIGH"
    control_action = "ENTER_NOW" if selected_short else "DO_NOT_ENTER"
    return {
        "paper_action": action,
        "reason": reason,
        "control_action": control_action,
        "would_change_control": action != control_action,
    }


def committee_v21_observation(
    result: Mapping[str, Any],
    *,
    symbol: str,
    primary_overlay: Mapping[str, Any],
) -> Mapping[str, Any]:
    features = _mapping(result["research_features"], "research_features")
    candidate = _mapping(result["candidate"], "candidate")
    layer = _mapping(result["layer"], "layer")
    regime = _mapping(primary_overlay.get("regime", {}), "regime")
    if layer.get("qmae_q90") is None:
        raise CommitteeV21ShadowError("AEGIS_COMMITTEE_V21_QMAE_MISSING")
    values = {
        name: _finite(features[name], name)
        for name in (
            "failed_breakdown_proxy",
            "fake_breakdown_risk_proxy",
            "rebound_risk_proxy",
            "squeeze_risk_proxy_causal",
            "immediate_reversal_risk_proxy",
            "overextended_down_risk_proxy",
            "low_room_to_fall_risk_proxy",
            "high_wick_reclaim_risk_proxy",
            "squeeze_plus_reclaim_risk_proxy",
            "extension_down_proxy",
            "exhaustion_down_proxy",
            "room_to_fall_proxy_24",
            "lower_wick_fraction",
            "close_position_in_range",
            "ret_3",
            "ret_6",
            "ret_12",
            "atr_12",
            "volume_zscore_24",
            "range_expansion",
            "chop_12",
            "trend_strength_12",
            "market_direction_6",
            "consecutive_red_count",
            "trend_compression",
            "high_vol_regime_proxy",
            "low_vol_regime_proxy",
        )
    }
    values.update(
        {
            "control_calibrated_score": _finite(
                candidate["calibrated_score"],
                "control_calibrated_score",
            ),
            "qmae_q90": _finite(layer["qmae_q90"], "qmae_q90"),
            "tail_risk_probability": _finite(
                layer["rv2_tail_risk"],
                "tail_risk_probability",
            ),
            "trrm_compatibility": _finite(
                layer["trrm_compatibility"],
                "trrm_compatibility",
            ),
            "symbol": symbol,
            "direction_regime": _category_text(regime.get("direction", "UNKNOWN")),
            "volatility_regime": _category_text(regime.get("volatility", "UNKNOWN")),
            "structure_regime": _category_text(regime.get("structure", "UNKNOWN")),
        }
    )
    return values


class CommitteeV21ShadowRuntime:
    def __init__(self, config: CommitteeV21ShadowConfig) -> None:
        self.config = config
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

    def observe_batch(
        self,
        batch: Mapping[str, Any],
        *,
        primary_overlay: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        cycle = str(batch["decision_cycle_id"])
        timestamp = str(batch["market_timestamp"])
        market_time = _timestamp(timestamp)
        if market_time < self.config.evidence_start_utc:
            return {}
        with self._lock:
            if (
                cycle in self._processed_cycles
                or timestamp in self._processed_timestamps
            ):
                return self._overlay(cycle=cycle, timestamp=timestamp)
            try:
                rows = self._build_rows(
                    batch,
                    primary_overlay=primary_overlay or {},
                )
                if (
                    sum(
                        row["counterfactual"]["paper_action"] == "ENTER_NOW"
                        for row in rows
                    )
                    > self.config.maximum_paper_entries_per_cycle
                ):
                    raise CommitteeV21ShadowError(
                        "AEGIS_COMMITTEE_V21_SELECTION_LIMIT_EXCEEDED"
                    )
                for row in rows:
                    self._signals.append(row)
                self._processed_cycles.add(cycle)
                self._processed_timestamps.add(timestamp)
                self._mature_outcomes()
                self.last_observation_at = datetime.now(timezone.utc)
                return self._overlay(cycle=cycle, timestamp=timestamp)
            except Exception:
                self.observation_errors += 1
                return {}

    def _build_rows(
        self,
        batch: Mapping[str, Any],
        *,
        primary_overlay: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        results = _mapping(batch["results"], "results")
        if (
            set(results) != set(CANONICAL_SYMBOLS)
            or batch.get("feature_schema") != FEATURE_SCHEMA_VERSION
            or int(batch.get("feature_count", 0)) != len(FEATURE_NAMES)
        ):
            raise CommitteeV21ShadowError("AEGIS_COMMITTEE_V21_CANONICAL_BATCH_INVALID")
        rows = []
        for symbol in CANONICAL_SYMBOLS:
            result = _mapping(results[symbol], symbol)
            primary = _mapping(primary_overlay.get(symbol, {}), "primary_overlay")
            observation = committee_v21_observation(
                result,
                symbol=symbol,
                primary_overlay=primary,
            )
            risk = committee_v21_calibrated_risk(
                self.config.contract,
                self.config.artifact,
                observation,
            )
            candidate = _mapping(result["candidate"], "candidate")
            control_selected = bool(result["selected"])
            control_side = str(candidate["side"])
            counterfactual = committee_v21_counterfactual(
                control_selected=control_selected,
                control_side=control_side,
                calibrated_risk_probability=float(risk["calibrated_risk_probability"]),
                calibrated_risk_threshold=float(risk["calibrated_risk_threshold"]),
            )
            event_id = self._hashing.digest_value(
                {
                    "schema": SIGNAL_SCHEMA,
                    "experiment": self.config.experiment_id,
                    "cycle": batch["decision_cycle_id"],
                    "symbol": symbol,
                    "config": self.config.sha256,
                    "artifact": self.config.artifact.sha256,
                }
            )
            bar = _mapping(result["market_bar"], "market_bar")
            rows.append(
                {
                    "schema_id": SIGNAL_SCHEMA,
                    "event_id": event_id,
                    "experiment_id": self.config.experiment_id,
                    "runtime_config_sha256": self.config.sha256,
                    "preregistration_sha256": self.config.contract.sha256,
                    "artifact_sha256": self.config.artifact.sha256,
                    "decision_cycle_id": batch["decision_cycle_id"],
                    "market_timestamp": result["market_timestamp"],
                    "symbol": symbol,
                    "timeframe": "5m",
                    "feature_schema": result["feature_schema"],
                    "feature_vector_hash": result["feature_vector_hash"],
                    "market_bar": {
                        name: _finite(bar[name], name)
                        for name in ("open", "high", "low", "close")
                    },
                    "control": {
                        "selected": control_selected,
                        "side": control_side,
                    },
                    "context": {
                        name: observation[name]
                        for name in (
                            "direction_regime",
                            "volatility_regime",
                            "structure_regime",
                        )
                    },
                    "risk": dict(risk),
                    "counterfactual": {
                        **dict(counterfactual),
                        "mode": "COUNTERFACTUAL_ONLY",
                        "wait_interpretation": "ABSTENTION_ONLY",
                        "exchange_authority": False,
                    },
                    "exchange_authority": False,
                    "exchange_mutations": 0,
                }
            )
        return rows

    def _mature_outcomes(self) -> None:
        by_symbol = {symbol: [] for symbol in CANONICAL_SYMBOLS}
        for row in self._signals.rows.recent:
            by_symbol[str(row["symbol"])].append(row)
        for rows in by_symbol.values():
            rows.sort(key=lambda row: str(row["market_timestamp"]))
            for index, signal in enumerate(rows):
                event_id = str(signal["event_id"])
                if event_id in self._outcomes.payloads:
                    continue
                future = rows[index + 1 : index + 1 + self.config.contract.horizon_bars]
                if len(future) < self.config.contract.horizon_bars:
                    continue
                entry = _finite(signal["market_bar"]["close"], "entry")
                exit_price = _finite(
                    future[-1]["market_bar"]["close"],
                    "exit_price",
                )
                highs = [_finite(row["market_bar"]["high"], "high") for row in future]
                lows = [_finite(row["market_bar"]["low"], "low") for row in future]
                gross = (entry - exit_price) / entry
                self._outcomes.append(
                    {
                        "schema_id": OUTCOME_SCHEMA,
                        "event_id": event_id,
                        "experiment_id": self.config.experiment_id,
                        "symbol": signal["symbol"],
                        "side": "SHORT",
                        "signal_timestamp": signal["market_timestamp"],
                        "maturity_timestamp": future[-1]["market_timestamp"],
                        "gross_return_fraction": gross,
                        "net_return_fraction": (
                            gross - self.config.contract.round_trip_cost_fraction
                        ),
                        "mae_fraction": max(
                            0.0,
                            (max(highs) - entry) / entry,
                        ),
                        "mfe_fraction": max(
                            0.0,
                            (entry - min(lows)) / entry,
                        ),
                        "control_selected": bool(signal["control"]["selected"]),
                        "committee_paper_action": signal["counterfactual"][
                            "paper_action"
                        ],
                        "calibrated_risk_probability": signal["risk"][
                            "calibrated_risk_probability"
                        ],
                        "exchange_mutations": 0,
                    }
                )

    def _overlay(self, *, cycle: str, timestamp: str) -> Mapping[str, Any]:
        latest = {
            str(row["symbol"]): row
            for row in self._signals.rows.recent
            if str(row["decision_cycle_id"]) == cycle
            or str(row["market_timestamp"]) == timestamp
        }
        return {
            symbol: {
                "schema_id": HTTP_SCHEMA,
                "experiment_id": self.config.experiment_id,
                "mode": "SHADOW",
                "runtime_authority": "OBSERVATIONAL_ONLY",
                "artifact_sha256": self.config.artifact.sha256,
                "control_selected": bool(row["control"]["selected"]),
                "calibrated_risk_probability": row["risk"][
                    "calibrated_risk_probability"
                ],
                "calibrated_risk_threshold": row["risk"]["calibrated_risk_threshold"],
                "risk_band": row["risk"]["risk_band"],
                "paper_action": row["counterfactual"]["paper_action"],
                "paper_reason": row["counterfactual"]["reason"],
                "would_change_control": bool(
                    row["counterfactual"]["would_change_control"]
                ),
                "exchange_authority": False,
                "exchange_mutations": 0,
            }
            for symbol, row in latest.items()
        }

    def health(self) -> Mapping[str, Any]:
        return {
            "status": "ACTIVE",
            "experiment_id": self.config.experiment_id,
            "mode": "SHADOW",
            "runtime_authority": "OBSERVATIONAL_ONLY",
            "runtime_config_sha256": self.config.sha256,
            "preregistration_sha256": self.config.contract.sha256,
            "artifact_sha256": self.config.artifact.sha256,
            "model_id": self.config.artifact.model_id,
            "calibrated_risk_threshold": (
                self.config.artifact.calibrated_risk_threshold
            ),
            "evidence_start_utc": _iso(self.config.evidence_start_utc),
            "signal_records": len(self._signals.rows),
            "paper_outcomes": len(self._outcomes.rows),
            "observation_errors": self.observation_errors,
            "last_observation_at": (
                _iso(self.last_observation_at) if self.last_observation_at else None
            ),
            "exchange_authority": False,
            "exchange_mutations": 0,
            "runtime_training": False,
            "automatic_promotion": False,
        }


class UnavailableCommitteeV21ShadowObserver:
    def __init__(self, reason: str) -> None:
        self.reason = reason

    @property
    def mode(self) -> EntryQualityV2Mode:
        return EntryQualityV2Mode.SHADOW

    def observe_batch(
        self,
        _: Mapping[str, Any],
        *,
        primary_overlay: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        return {}

    def health(self) -> Mapping[str, Any]:
        return {
            "status": "UNAVAILABLE",
            "mode": "SHADOW",
            "reason": self.reason,
            "exchange_authority": False,
            "exchange_mutations": 0,
        }


def build_committee_v21_shadow_observer(
    config_path: Path,
    *,
    repo_root: Path,
) -> CommitteeV21ShadowRuntime | UnavailableCommitteeV21ShadowObserver:
    try:
        return CommitteeV21ShadowRuntime(
            load_committee_v21_shadow_config(
                config_path,
                repo_root=repo_root,
            )
        )
    except Exception as exc:
        return UnavailableCommitteeV21ShadowObserver(type(exc).__name__)
