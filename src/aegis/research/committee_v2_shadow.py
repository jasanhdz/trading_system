"""Specialized Committee V2 evidence observer with no operational authority."""

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
from ..utils import Sha256HashProvider, sha256_file
from .shadow_runtime import (
    EntryQualityV2Mode,
    _AppendOnlyJournal,
    _mapping,
)

COMMITTEE_SCHEMA = "aegis-specialized-committee-v2-shadow-runtime-v1"
COMMITTEE_SIGNAL_SCHEMA = "aegis-specialized-committee-v2-shadow-signal-v1"
COMMITTEE_OUTCOME_SCHEMA = "aegis-specialized-committee-v2-shadow-outcome-v1"
COMMITTEE_HTTP_SCHEMA = "aegis-specialized-committee-v2-http-shadow-v1"
EXPECTED_MEMBERS = (
    "short_opportunity",
    "short_reversal_risk",
    "entry_timing",
    "qmae",
    "tail_risk",
    "regime",
    "long_opportunity",
)
REVERSAL_FLAG_FEATURES = (
    "failed_breakdown_proxy",
    "fake_breakdown_risk_proxy",
    "rebound_risk_proxy",
    "squeeze_risk_proxy_causal",
    "immediate_reversal_risk_proxy",
    "overextended_down_risk_proxy",
    "low_room_to_fall_risk_proxy",
    "high_wick_reclaim_risk_proxy",
    "squeeze_plus_reclaim_risk_proxy",
)


class CommitteeV2ShadowError(RuntimeError):
    pass


@dataclass(frozen=True)
class CommitteeV2ShadowConfig:
    committee_id: str
    config_path: Path
    config_sha256: str
    signal_journal: Path
    outcome_journal: Path
    horizon_bars: int
    round_trip_cost_fraction: float
    maximum_paper_entries_per_cycle: int
    member_contracts: Mapping[str, Mapping[str, Any]]


def load_committee_v2_shadow_config(
    path: Path,
    *,
    repo_root: Path,
) -> CommitteeV2ShadowConfig:
    resolved = path.resolve()
    root = repo_root.resolve()
    try:
        payload = _mapping(
            yaml.safe_load(resolved.read_text(encoding="utf-8")),
            "committee_v2",
        )
        evidence = _mapping(payload["evidence"], "evidence")
        members = _mapping(payload["members"], "members")
        selector = _mapping(payload["meta_selector"], "meta_selector")
        promotion = _mapping(payload["promotion"], "promotion")
    except (OSError, KeyError, TypeError, yaml.YAMLError) as exc:
        raise CommitteeV2ShadowError("AEGIS_COMMITTEE_V2_CONFIG_INVALID") from exc

    if (
        payload.get("schema_version") != COMMITTEE_SCHEMA
        or payload.get("enabled") is not True
        or payload.get("mode") != "SHADOW"
        or payload.get("runtime_authority") != "OBSERVATIONAL_ONLY"
        or payload.get("feature_schema") != FEATURE_SCHEMA_VERSION
        or int(payload.get("feature_count", 0)) != len(FEATURE_NAMES)
    ):
        raise CommitteeV2ShadowError("AEGIS_COMMITTEE_V2_SHADOW_AUTHORITY_INVALID")
    if set(members) != set(EXPECTED_MEMBERS):
        raise CommitteeV2ShadowError("AEGIS_COMMITTEE_V2_MEMBER_POPULATION_INVALID")
    directional = [
        name
        for name, contract in members.items()
        if bool(_mapping(contract, name).get("directional_vote_eligible"))
    ]
    if set(directional) != {"short_opportunity"} or len(directional) != 1:
        raise CommitteeV2ShadowError("AEGIS_COMMITTEE_V2_DIRECTIONAL_EVIDENCE_INVALID")
    if (
        selector.get("mode") != "COUNTERFACTUAL_ONLY"
        or selector.get("control_authority") != "CURRENT_CANONICAL_SELECTION"
        or selector.get("fabricated_votes_prohibited") is not True
        or selector.get("majority_vote_prohibited") is not True
        or promotion.get("automatic_training") is not False
        or promotion.get("automatic_promotion") is not False
        or promotion.get("live_authority") is not False
        or promotion.get("owner_authorization_required") is not True
    ):
        raise CommitteeV2ShadowError("AEGIS_COMMITTEE_V2_PROMOTION_AUTHORITY_INVALID")

    journal_root = (resolved.parent / str(evidence["journal_root"])).resolve()
    data_root = (root / "data").resolve()
    if journal_root != data_root and data_root not in journal_root.parents:
        raise CommitteeV2ShadowError("AEGIS_COMMITTEE_V2_JOURNAL_ROOT_PROHIBITED")
    signal_journal = journal_root / str(evidence["signal_journal"])
    outcome_journal = journal_root / str(evidence["outcome_journal"])
    if signal_journal.parent != journal_root or outcome_journal.parent != journal_root:
        raise CommitteeV2ShadowError("AEGIS_COMMITTEE_V2_JOURNAL_PATH_PROHIBITED")

    horizon = int(evidence["horizon_bars"])
    cost = float(evidence["round_trip_cost_fraction"])
    maximum = int(selector["maximum_paper_entries_per_cycle"])
    if horizon <= 0 or maximum != 1 or not math.isfinite(cost) or not 0.0 <= cost < 1.0:
        raise CommitteeV2ShadowError("AEGIS_COMMITTEE_V2_RUNTIME_LIMIT_INVALID")
    return CommitteeV2ShadowConfig(
        committee_id=str(payload["committee_id"]),
        config_path=resolved,
        config_sha256=sha256_file(resolved),
        signal_journal=signal_journal,
        outcome_journal=outcome_journal,
        horizon_bars=horizon,
        round_trip_cost_fraction=cost,
        maximum_paper_entries_per_cycle=maximum,
        member_contracts={
            name: dict(_mapping(members[name], name)) for name in EXPECTED_MEMBERS
        },
    )


def _finite(value: Any, identity: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise CommitteeV2ShadowError(f"AEGIS_COMMITTEE_V2_NONFINITE:{identity}")
    return result


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def committee_v2_counterfactual(
    features: Mapping[str, Any],
    *,
    control_selected: bool,
    control_side: str,
) -> Mapping[str, Any]:
    risk_flags = {
        name: _finite(features[name], name) > 0.0
        for name in REVERSAL_FLAG_FEATURES
    }
    observed_risk_count = sum(risk_flags.values())
    control_enter = control_selected and control_side.endswith("SHORT")
    paper_action = (
        "ENTER_NOW"
        if control_enter and observed_risk_count == 0
        else "WAIT_CONFIRMATION"
        if control_enter
        else "DO_NOT_ENTER"
    )
    reason = (
        "CONTROL_SELECTED_NO_OBSERVED_REVERSAL_FLAG"
        if paper_action == "ENTER_NOW"
        else "CONTROL_SELECTED_REVERSAL_RISK_OBSERVED"
        if paper_action == "WAIT_CONFIRMATION"
        else "CONTROL_NOT_SELECTED"
    )
    control_action = "ENTER_NOW" if control_enter else "DO_NOT_ENTER"
    return {
        "risk_flags": risk_flags,
        "observed_risk_count": observed_risk_count,
        "paper_action": paper_action,
        "reason": reason,
        "control_action": control_action,
        "would_change_control": paper_action != control_action,
    }


class CommitteeV2ShadowRuntime:
    """Record specialist evidence while preserving the canonical control."""

    def __init__(self, config: CommitteeV2ShadowConfig) -> None:
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
        dual_overlay: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        cycle = str(batch["decision_cycle_id"])
        timestamp = str(batch["market_timestamp"])
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
                    dual_overlay=dual_overlay or {},
                )
                if (
                    sum(
                        row["meta_selector"]["paper_action"] == "ENTER_NOW"
                        for row in rows
                    )
                    > self.config.maximum_paper_entries_per_cycle
                ):
                    raise CommitteeV2ShadowError(
                        "AEGIS_COMMITTEE_V2_SELECTION_LIMIT_EXCEEDED"
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
                raise

    def _build_rows(
        self,
        batch: Mapping[str, Any],
        *,
        primary_overlay: Mapping[str, Any],
        dual_overlay: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        results = _mapping(batch["results"], "results")
        if (
            set(results) != set(CANONICAL_SYMBOLS)
            or batch.get("feature_schema") != FEATURE_SCHEMA_VERSION
            or int(batch.get("feature_count", 0)) != len(FEATURE_NAMES)
        ):
            raise CommitteeV2ShadowError("AEGIS_COMMITTEE_V2_CANONICAL_BATCH_INVALID")

        rows: list[dict[str, Any]] = []
        for symbol in CANONICAL_SYMBOLS:
            result = _mapping(results[symbol], symbol)
            features = _mapping(result["research_features"], "features")
            predictions = tuple(
                _mapping(value, "prediction") for value in result["predictions"]
            )
            if not predictions:
                raise CommitteeV2ShadowError("AEGIS_COMMITTEE_V2_PREDICTION_MISSING")
            candidate = _mapping(result["candidate"], "candidate")
            layer = _mapping(result["layer"], "layer")
            primary = _mapping(
                primary_overlay.get(symbol, {}),
                "primary_overlay",
            )
            dual = _mapping(dual_overlay.get(symbol, {}), "dual_overlay")

            control_selected = bool(result["selected"])
            control_side = str(candidate["side"])
            counterfactual = committee_v2_counterfactual(
                features,
                control_selected=control_selected,
                control_side=control_side,
            )
            risk_flags = counterfactual["risk_flags"]
            observed_risk_count = int(
                counterfactual["observed_risk_count"]
            )
            timing_action = str(counterfactual["paper_action"])
            timing_reason = str(counterfactual["reason"])
            long_status = str(
                dual.get(
                    "status",
                    "OFFLINE_VALIDATION_FAILED_OBSERVATION_ONLY",
                )
            )
            event_id = self._hashing.digest_value(
                {
                    "schema": COMMITTEE_SIGNAL_SCHEMA,
                    "committee": self.config.committee_id,
                    "cycle": batch["decision_cycle_id"],
                    "symbol": symbol,
                    "config": self.config.config_sha256,
                }
            )
            bar = _mapping(result["market_bar"], "market_bar")
            rows.append(
                {
                    "schema_id": COMMITTEE_SIGNAL_SCHEMA,
                    "event_id": event_id,
                    "committee_id": self.config.committee_id,
                    "config_sha256": self.config.config_sha256,
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
                        "raw_score": _finite(candidate["raw_score"], "raw_score"),
                        "calibrated_score": _finite(
                            candidate["calibrated_score"],
                            "calibrated_score",
                        ),
                    },
                    "committee_semantics": {
                        "directional_estimator_count": len(predictions),
                        "eligible_directional_member_count": 1,
                        "directional_consensus": (
                            "NOT_APPLICABLE_SINGLE_ELIGIBLE_DIRECTIONAL_MEMBER"
                        ),
                        "short_probability_semantics": (
                            "SIDE_AUTHORITY_NOT_PROFITABILITY_CONFIDENCE"
                        ),
                        "candidate_confidence_semantics": (
                            "NOT_APPLICABLE_SINGLE_ESTIMATOR"
                        ),
                        "fabricated_votes": 0,
                    },
                    "members": {
                        "short_opportunity": {
                            "status": "ACTIVE_CONTROL_CHAMPION",
                            "directional_vote_eligible": True,
                            "side": control_side,
                            "control_selected": control_selected,
                            "shadow_selected": bool(primary.get("selected", False)),
                            "score": primary.get("score"),
                            "source": self.config.member_contracts["short_opportunity"][
                                "source"
                            ],
                        },
                        "short_reversal_risk": {
                            "status": "OBSERVATIONAL_FEATURE_PROXY",
                            "directional_vote_eligible": False,
                            "observed_flag_count": observed_risk_count,
                            "flags": risk_flags,
                            "extension_down_proxy": _finite(
                                features["extension_down_proxy"],
                                "extension_down_proxy",
                            ),
                            "exhaustion_down_proxy": _finite(
                                features["exhaustion_down_proxy"],
                                "exhaustion_down_proxy",
                            ),
                        },
                        "entry_timing": {
                            "status": "UNVALIDATED_COUNTERFACTUAL_OBSERVER",
                            "directional_vote_eligible": False,
                            "paper_action": timing_action,
                            "reason": timing_reason,
                        },
                        "qmae": {
                            "status": "ACTIVE_RISK_OBSERVER",
                            "directional_vote_eligible": False,
                            "q90": (
                                _finite(layer["qmae_q90"], "qmae_q90")
                                if layer.get("qmae_q90") is not None
                                else None
                            ),
                            "canonical_eligible": bool(layer["eligible"]),
                        },
                        "tail_risk": {
                            "status": "ACTIVE_RISK_OBSERVER",
                            "directional_vote_eligible": False,
                            "probability": _finite(
                                layer["rv2_tail_risk"],
                                "rv2_tail_risk",
                            ),
                            "compatibility": _finite(
                                layer["trrm_compatibility"],
                                "trrm_compatibility",
                            ),
                        },
                        "regime": {
                            "status": "FACTORIZED_SHADOW_CONTEXT",
                            "directional_vote_eligible": False,
                            "observation": dict(primary.get("regime", {})),
                        },
                        "long_opportunity": {
                            "status": long_status,
                            "directional_vote_eligible": False,
                            "model_only_selected": bool(
                                dual.get("model_only_selected", False)
                            ),
                            "regime_confirmed_selected": bool(
                                dual.get("regime_confirmed_selected", False)
                            ),
                            "score": dual.get("score"),
                            "live_eligible": False,
                        },
                    },
                    "meta_selector": {
                        "mode": "COUNTERFACTUAL_ONLY",
                        "paper_action": timing_action,
                        "reason": timing_reason,
                        "control_action": counterfactual[
                            "control_action"
                        ],
                        "would_change_control": bool(
                            counterfactual["would_change_control"]
                        ),
                        "exchange_authority": False,
                    },
                    "exchange_authority": False,
                    "exchange_mutations": 0,
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
                entry = _finite(signal["market_bar"]["close"], "entry")
                if entry <= 0.0:
                    raise CommitteeV2ShadowError("AEGIS_COMMITTEE_V2_PRICE_INVALID")
                exit_price = _finite(
                    future[-1]["market_bar"]["close"],
                    "exit_price",
                )
                highs = [_finite(row["market_bar"]["high"], "high") for row in future]
                lows = [_finite(row["market_bar"]["low"], "low") for row in future]
                gross = (entry - exit_price) / entry
                self._outcomes.append(
                    {
                        "schema_id": COMMITTEE_OUTCOME_SCHEMA,
                        "event_id": event_id,
                        "committee_id": self.config.committee_id,
                        "symbol": signal["symbol"],
                        "side": "SHORT",
                        "signal_timestamp": signal["market_timestamp"],
                        "maturity_timestamp": future[-1]["market_timestamp"],
                        "gross_return_fraction": gross,
                        "net_return_fraction": (
                            gross - self.config.round_trip_cost_fraction
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
                        "committee_paper_action": signal["meta_selector"][
                            "paper_action"
                        ],
                        "exchange_mutations": 0,
                    }
                )

    def _overlay(
        self,
        *,
        cycle: str,
        timestamp: str,
    ) -> Mapping[str, Any]:
        rows = [
            row
            for row in self._signals.rows
            if str(row["decision_cycle_id"]) == cycle
            or str(row["market_timestamp"]) == timestamp
        ]
        latest = {str(row["symbol"]): row for row in rows}
        return {
            symbol: {
                "schema_id": COMMITTEE_HTTP_SCHEMA,
                "committee_id": self.config.committee_id,
                "config_sha256": self.config.config_sha256,
                "mode": "SHADOW",
                "runtime_authority": "OBSERVATIONAL_ONLY",
                "directional_consensus": row["committee_semantics"][
                    "directional_consensus"
                ],
                "eligible_directional_member_count": 1,
                "short_probability_semantics": row["committee_semantics"][
                    "short_probability_semantics"
                ],
                "candidate_confidence_semantics": row["committee_semantics"][
                    "candidate_confidence_semantics"
                ],
                "control_selected": bool(row["control"]["selected"]),
                "paper_action": row["meta_selector"]["paper_action"],
                "paper_reason": row["meta_selector"]["reason"],
                "would_change_control": bool(
                    row["meta_selector"]["would_change_control"]
                ),
                "member_status": {
                    name: member["status"] for name, member in row["members"].items()
                },
                "exchange_authority": False,
                "exchange_mutations": 0,
            }
            for symbol, row in latest.items()
        }

    def health(self) -> Mapping[str, Any]:
        return {
            "status": "ACTIVE",
            "committee_id": self.config.committee_id,
            "mode": "SHADOW",
            "runtime_authority": "OBSERVATIONAL_ONLY",
            "config_sha256": self.config.config_sha256,
            "directional_consensus": (
                "NOT_APPLICABLE_SINGLE_ELIGIBLE_DIRECTIONAL_MEMBER"
            ),
            "eligible_directional_member_count": 1,
            "member_count": len(EXPECTED_MEMBERS),
            "signal_records": len(self._signals.rows),
            "paper_outcomes": len(self._outcomes.rows),
            "observation_errors": self.observation_errors,
            "last_observation_at": (
                _iso(self.last_observation_at) if self.last_observation_at else None
            ),
            "exchange_authority": False,
            "exchange_mutations": 0,
            "automatic_promotion": False,
        }


class UnavailableCommitteeV2ShadowObserver:
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
        dual_overlay: Mapping[str, Any] | None = None,
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


def build_committee_v2_shadow_observer(
    config_path: Path,
    *,
    repo_root: Path,
) -> CommitteeV2ShadowRuntime | UnavailableCommitteeV2ShadowObserver:
    try:
        return CommitteeV2ShadowRuntime(
            load_committee_v2_shadow_config(
                config_path,
                repo_root=repo_root,
            )
        )
    except Exception as exc:
        return UnavailableCommitteeV2ShadowObserver(type(exc).__name__)
