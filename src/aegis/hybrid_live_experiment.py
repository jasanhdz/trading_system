"""Owner-authorized LONG/SHORT hybrid selection for a bounded Live experiment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import yaml

from .config import CANONICAL_SYMBOLS
from .directional_confirmation import (
    ConfirmationState,
    DirectionalConfirmationPolicy,
    assess_directional_confirmation,
    directional_confirmation_features,
    directional_relative_quality,
)
from .features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from .research.shadow_runtime import _AppendOnlyJournal, _mapping
from .utils import Sha256HashProvider, sha256_file

CONFIG_SCHEMA = "aegis-hybrid-directional-live-experiment-v2"
CONTRACT_VERSION = "aegis-hybrid-directional-live-decision-v2"
AUTHORITY = "OWNER_AUTHORIZED_HYBRID_DIRECTIONAL_MULTI_SYMBOL_5M_QUALITY_SELECTION_V2"
MODEL_IDENTIFIER = "aegis-hybrid-directional-committee-v1"
MODEL_SHA256 = "f52dcaa12fe94b6cc9023c25cf95ea2d6fc16296c9b65c2c93d00e13e66ba0e8"
CONFIGURATION_SHA256 = (
    "26507443adf07dfc5a90d48a1c5f472f989a26cfe929740bd9e2009c39aaa3a9"
)
DECISION_SCHEMA = "aegis-hybrid-directional-live-evidence-v2"


class HybridLiveExperimentError(RuntimeError):
    pass


@dataclass(frozen=True)
class HybridLiveExperimentConfig:
    path: Path
    artifact_path: Path
    readiness_path: Path
    decision_journal: Path
    maximum_selected_per_cycle: int
    maximum_selected_per_symbol: int
    confirmation_policy: DirectionalConfirmationPolicy


def load_hybrid_live_experiment_config(
    path: Path, *, repo_root: Path
) -> HybridLiveExperimentConfig:
    resolved = path.resolve()
    root = repo_root.resolve()
    try:
        payload = _mapping(yaml.safe_load(resolved.read_text()), "hybrid_live")
        artifact = _mapping(payload["artifact"], "artifact")
        selection = _mapping(payload["selection"], "selection")
        quality = _mapping(payload["quality"], "quality")
        confirmation = _mapping(payload["confirmation"], "confirmation")
        evidence = _mapping(payload["evidence"], "evidence")
        execution = _mapping(payload["execution"], "execution")
        validation = _mapping(payload["validation_evidence"], "validation")
        artifact_path = (root / str(artifact["path"])).resolve()
        readiness_path = (root / str(artifact["readiness_path"])).resolve()
        journal_root = (resolved.parent / str(evidence["journal_root"])).resolve()
    except (OSError, KeyError, TypeError, yaml.YAMLError) as exc:
        raise HybridLiveExperimentError("AEGIS_HYBRID_LIVE_CONFIG_INVALID") from exc

    data_root = (root / "data").resolve()
    expected_symbols = list(CANONICAL_SYMBOLS)
    if (
        sha256_file(resolved) != CONFIGURATION_SHA256
        or payload.get("schema_version") != CONFIG_SCHEMA
        or payload.get("enabled") is not True
        or payload.get("mode") != "LIVE"
        or payload.get("runtime_authority") != "OWNER_AUTHORIZED_REAL_MONEY_EXPERIMENT"
        or payload.get("owner_authorization") != AUTHORITY
        or payload.get("owner_acknowledges_offline_validation_failed") is not True
        or payload.get("feature_schema") != FEATURE_SCHEMA_VERSION
        or int(payload.get("feature_count", 0)) != len(FEATURE_NAMES)
        or list(payload.get("symbols", ())) != expected_symbols
        or artifact.get("sha256") != MODEL_SHA256
        or artifact.get("offline_validation_state") != "FAILED"
        or artifact.get("artifact_runtime_authority") != "SHADOW_ONLY"
        or artifact.get("artifact_contents_modified") is not False
        or not artifact_path.is_file()
        or sha256_file(artifact_path) != MODEL_SHA256
        or not readiness_path.is_file()
        or sha256_file(readiness_path) != str(artifact["readiness_sha256"])
        or selection.get("cadence") != "EVERY_CLOSED_5M_BAR"
        or selection.get("anchor_rule") != "EVERY_NEW_COORDINATED_CLOSED_BAR"
        or selection.get("candidate_population") != "ALL_11_SYMBOLS_X_LONG_SHORT"
        or selection.get("cross_side_arbitration") != "QUALITY_CONFIRMED_PER_SYMBOL"
        or int(selection.get("maximum_selected_per_cycle", 0)) != len(CANONICAL_SYMBOLS)
        or int(selection.get("maximum_selected_per_symbol", 0)) != 1
        or selection.get("minimum_score") != "REPLACED_BY_QUALITY_POLICY"
        or selection.get("idempotency_identity") != "DECISION_CYCLE_X_SYMBOL_X_SIDE"
        or int(selection.get("fabricated_votes", -1)) != 0
        or evidence.get("record_all_candidates") is not True
        or evidence.get("record_non_selected_candidates") is not True
        or evidence.get("record_confirmation_features") is not True
        or execution.get("typescript_guards_unchanged") is not True
        or execution.get("typescript_sizing_unchanged") is not True
        or execution.get("typescript_leverage_unchanged") is not True
        or execution.get("typescript_brackets_unchanged") is not True
        or int(execution.get("python_exchange_mutations", -1)) != 0
        or quality.get("semantics") != "CROSS_SECTIONAL_CALIBRATED_WITH_NET_TOLERANCE"
        or execution.get("selection_authority")
        != "HYBRID_DIRECTIONAL_RELATIVE_QUALITY_AND_CONFIRMATION"
        or validation.get("replay_conclusion")
        != "PRICE_PROTECTION_DOES_NOT_ESTABLISH_POSITIVE_EXPECTANCY"
        or validation.get("robust_positive_folds_and_directions") != "0_of_8"
        or (journal_root != data_root and data_root not in journal_root.parents)
    ):
        raise HybridLiveExperimentError("AEGIS_HYBRID_LIVE_AUTHORITY_INVALID")
    journal = journal_root / str(evidence["decision_journal"])
    if journal.parent != journal_root:
        raise HybridLiveExperimentError("AEGIS_HYBRID_LIVE_JOURNAL_PROHIBITED")
    try:
        policy = DirectionalConfirmationPolicy(
            round_trip_cost_fraction=0.001,
            minimum_opportunity_probability_long=float(
                quality["minimum_opportunity_probability_long"]
            ),
            minimum_opportunity_probability_short=float(
                quality["minimum_opportunity_probability_short"]
            ),
            maximum_danger_probability=float(quality["maximum_danger_probability"]),
            minimum_net_return_fraction=float(quality["minimum_net_return_fraction"]),
            minimum_opportunity_percentile=float(
                quality["minimum_opportunity_percentile"]
            ),
            minimum_danger_quality_percentile=float(
                quality["minimum_danger_quality_percentile"]
            ),
            minimum_net_return_percentile=float(
                quality["minimum_net_return_percentile"]
            ),
            minimum_path_efficiency_percentile=float(
                quality["minimum_path_efficiency_percentile"]
            ),
            minimum_confirmation_components=int(
                confirmation["minimum_components_passed"]
            ),
            minimum_close_location=float(confirmation["minimum_close_location"]),
            minimum_volume_zscore=float(confirmation["minimum_volume_zscore"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HybridLiveExperimentError("AEGIS_HYBRID_LIVE_POLICY_INVALID") from exc
    return HybridLiveExperimentConfig(
        path=resolved,
        artifact_path=artifact_path,
        readiness_path=readiness_path,
        decision_journal=journal,
        maximum_selected_per_cycle=len(CANONICAL_SYMBOLS),
        maximum_selected_per_symbol=1,
        confirmation_policy=policy,
    )


class HybridLiveExperimentSelector:
    """Rank the real hybrid outputs without changing any predicted value."""

    def __init__(self, config: HybridLiveExperimentConfig) -> None:
        self.config = config
        self._hashing = Sha256HashProvider()
        self._journal = _AppendOnlyJournal(config.decision_journal, "event_id")
        self._processed_cycles = {
            str(row["decision_cycle_id"]) for row in self._journal.rows
        }

    @staticmethod
    def _timestamp(value: object) -> datetime:
        if not isinstance(value, str):
            raise HybridLiveExperimentError("AEGIS_HYBRID_LIVE_TIMESTAMP_INVALID")
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise HybridLiveExperimentError("AEGIS_HYBRID_LIVE_TIMESTAMP_INVALID")
        return parsed

    @staticmethod
    def _prediction(
        overlay: Mapping[str, Any], symbol: str, side: str
    ) -> Mapping[str, float]:
        try:
            hybrid = _mapping(overlay[symbol]["hybrid_directional_shadow"], "hybrid")
            if (
                hybrid.get("mode") != "SHADOW"
                or hybrid.get("status") != "OFFLINE_VALIDATION_FAILED_OBSERVATION_ONLY"
            ):
                raise HybridLiveExperimentError("AEGIS_HYBRID_LIVE_INPUT_INVALID")
            raw = _mapping(hybrid["predictions"][side], "prediction")
            result = {
                name: float(raw[name])
                for name in (
                    "opportunity_probability",
                    "danger_probability",
                    "mae_q50",
                    "mae_q90",
                    "mfe_q50",
                    "net_return_mean",
                    "shadow_rank_score",
                )
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise HybridLiveExperimentError("AEGIS_HYBRID_LIVE_INPUT_INVALID") from exc
        if not all(math.isfinite(value) for value in result.values()):
            raise HybridLiveExperimentError("AEGIS_HYBRID_LIVE_INPUT_NONFINITE")
        return result

    def apply(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        overlay = _mapping(batch.get("_entry_quality_v2", {}), "research_overlay")
        if set(overlay) != set(CANONICAL_SYMBOLS):
            raise HybridLiveExperimentError("AEGIS_HYBRID_LIVE_SYMBOLS_INVALID")
        timestamp = self._timestamp(batch.get("market_timestamp"))
        if timestamp.minute % 5 != 0 or timestamp.second != 0:
            raise HybridLiveExperimentError("AEGIS_HYBRID_LIVE_TIMESTAMP_INVALID")
        results = _mapping(batch.get("results", {}), "results")
        if set(results) != set(CANONICAL_SYMBOLS):
            raise HybridLiveExperimentError("AEGIS_HYBRID_LIVE_SYMBOLS_INVALID")
        candidates: list[dict[str, Any]] = []
        for symbol in CANONICAL_SYMBOLS:
            research_features = _mapping(
                results[symbol].get("research_features", {}), "research_features"
            )
            for side in ("LONG", "SHORT"):
                prediction = self._prediction(overlay, symbol, side)
                confirmation_features = directional_confirmation_features(
                    research_features, side
                )
                candidates.append(
                    {
                        "symbol": symbol,
                        "side": side,
                        **prediction,
                        "confirmation_features": confirmation_features,
                    }
                )
        relative_quality = directional_relative_quality(candidates)
        for candidate, quality_values in zip(candidates, relative_quality):
            candidate["confirmation"] = assess_directional_confirmation(
                candidate,
                candidate["confirmation_features"],
                quality_values,
                self.config.confirmation_policy,
            )
        ranking = sorted(
            candidates,
            key=lambda item: (
                -float(item["shadow_rank_score"]),
                -float(item["net_return_mean"]),
                float(item["danger_probability"]),
                float(item["mae_q90"]),
                str(item["symbol"]),
                str(item["side"]),
            ),
        )
        selected_by_symbol: dict[str, dict[str, Any]] = {}
        for candidate in ranking:
            if (
                candidate["confirmation"]["state"] == ConfirmationState.CONFIRMED.value
                and candidate["symbol"] not in selected_by_symbol
            ):
                selected_by_symbol[str(candidate["symbol"])] = candidate
        selected_candidates = list(selected_by_symbol.values())[
            : self.config.maximum_selected_per_cycle
        ]
        cycle_id = str(batch["decision_cycle_id"])
        rows = []
        for rank, candidate in enumerate(ranking, start=1):
            is_selected = candidate in selected_candidates
            row = {
                "schema_id": DECISION_SCHEMA,
                "decision_cycle_id": cycle_id,
                "market_timestamp": batch["market_timestamp"],
                "closed_bar_evaluation": True,
                "rank": rank,
                **candidate,
                "selected": is_selected,
                "contract_version": CONTRACT_VERSION,
                "authority": AUTHORITY,
                "model_identifier": MODEL_IDENTIFIER,
                "model_sha256": MODEL_SHA256,
                "configuration_sha256": CONFIGURATION_SHA256,
                "fabricated_votes": 0,
                "python_exchange_mutations": 0,
            }
            row["event_id"] = self._hashing.digest_value(row)
            rows.append(row)
        if cycle_id not in self._processed_cycles:
            for row in rows:
                self._journal.append(row)
            self._processed_cycles.add(cycle_id)
        by_symbol: dict[str, Any] = {}
        for symbol in CANONICAL_SYMBOLS:
            symbol_rows = [row for row in rows if row["symbol"] == symbol]
            selected_row = next((row for row in symbol_rows if row["selected"]), None)
            by_symbol[symbol] = {
                "schema_id": "aegis-hybrid-directional-live-http-v2",
                "mode": "LIVE",
                "closed_bar_evaluation": True,
                "selected": selected_row is not None,
                "selected_side": selected_row["side"] if selected_row else None,
                "selected_prediction": dict(selected_row) if selected_row else None,
                "predictions": {
                    row["side"]: {
                        key: row[key]
                        for key in (
                            "opportunity_probability",
                            "danger_probability",
                            "mae_q50",
                            "mae_q90",
                            "mfe_q50",
                            "net_return_mean",
                            "shadow_rank_score",
                        )
                    }
                    for row in symbol_rows
                },
                "ranks": {row["side"]: row["rank"] for row in symbol_rows},
                "confirmation": {
                    row["side"]: dict(row["confirmation"]) for row in symbol_rows
                },
                "fabricated_votes": 0,
                "exchange_authority": True,
                "python_exchange_mutations": 0,
            }
        return {
            "schema_id": "aegis-hybrid-directional-live-batch-v2",
            "mode": "LIVE",
            "closed_bar_evaluation": True,
            "selected_symbol": (
                selected_candidates[0]["symbol"]
                if len(selected_candidates) == 1
                else None
            ),
            "selected_side": (
                selected_candidates[0]["side"]
                if len(selected_candidates) == 1
                else None
            ),
            "selected_symbols": [item["symbol"] for item in selected_candidates],
            "selected_sides": {
                item["symbol"]: item["side"] for item in selected_candidates
            },
            "selected_count": len(selected_candidates),
            "candidate_count": len(rows),
            "by_symbol": by_symbol,
        }

    def health(self) -> Mapping[str, Any]:
        return {
            "status": "ACTIVE",
            "mode": "LIVE",
            "contract_version": CONTRACT_VERSION,
            "authority": AUTHORITY,
            "model_identifier": MODEL_IDENTIFIER,
            "model_sha256": MODEL_SHA256,
            "configuration_sha256": CONFIGURATION_SHA256,
            "minimum_opportunity_probability_long": (
                self.config.confirmation_policy.minimum_opportunity_probability_long
            ),
            "minimum_opportunity_probability_short": (
                self.config.confirmation_policy.minimum_opportunity_probability_short
            ),
            "decision_records": len(self._journal.rows),
            "fabricated_votes": 0,
            "python_exchange_mutations": 0,
        }


def build_hybrid_live_experiment_selector(
    path: Path, *, repo_root: Path
) -> HybridLiveExperimentSelector:
    return HybridLiveExperimentSelector(
        load_hybrid_live_experiment_config(path, repo_root=repo_root)
    )
