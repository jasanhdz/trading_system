"""Causal entry-timing, regime, and ranking evidence with no trading authority."""

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
from ..features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from ..utils import Sha256HashProvider, sha256_file
from .committee_v2_shadow import REVERSAL_FLAG_FEATURES
from .regime_v2 import (
    DirectionRegime,
    FactorizedRegimeAnalyzer,
    RegimeV2Observation,
    RegimeV2Settings,
    StructureRegime,
)
from .shadow_runtime import (
    EntryQualityV2Mode,
    _AppendOnlyJournal,
    _mapping,
    _timestamp,
    load_entry_quality_v2_config,
)

SCHEMA = "aegis-entry-intelligence-shadow-runtime-v1"
SIGNAL_SCHEMA = "aegis-entry-intelligence-shadow-signal-v1"
OUTCOME_SCHEMA = "aegis-entry-intelligence-shadow-outcome-v1"
HTTP_SCHEMA = "aegis-entry-intelligence-http-shadow-v1"


class EntryIntelligenceShadowError(RuntimeError):
    pass


class TimingState(str, Enum):
    CANDIDATE_SEEN = "CANDIDATE_SEEN"
    WAITING_FOR_RETEST = "WAITING_FOR_RETEST"
    TIMING_CONFIRMED = "TIMING_CONFIRMED"
    INVALIDATED = "INVALIDATED"
    EXPIRED = "EXPIRED"
    NO_SETUP = "NO_SETUP"


@dataclass(frozen=True)
class EntryIntelligenceShadowConfig:
    observer_id: str
    config_path: Path
    config_sha256: str
    signal_journal: Path
    outcome_journal: Path
    horizon_bars: int
    round_trip_cost_fraction: float
    maximum_wait_bars: int
    require_global_bearish: bool
    require_local_bearish: bool
    require_trend_structure: bool
    require_short_ema_stack: bool
    require_bearish_confirmation_candle: bool
    maximum_counterfactual_candidates_per_cycle: int
    regime_settings: RegimeV2Settings


@dataclass
class _PendingSetup:
    setup_id: str
    origin_event_id: str
    age_bars: int


def load_entry_intelligence_shadow_config(
    path: Path,
    *,
    repo_root: Path,
    regime_config_path: Path,
) -> EntryIntelligenceShadowConfig:
    resolved = path.resolve()
    root = repo_root.resolve()
    try:
        payload = _mapping(
            yaml.safe_load(resolved.read_text(encoding="utf-8")), "entry_intelligence"
        )
        evidence = _mapping(payload["evidence"], "evidence")
        timing = _mapping(payload["timing"], "timing")
        ranking = _mapping(payload["ranking"], "ranking")
        promotion = _mapping(payload["promotion"], "promotion")
    except (OSError, KeyError, TypeError, yaml.YAMLError) as exc:
        raise EntryIntelligenceShadowError(
            "AEGIS_ENTRY_INTELLIGENCE_CONFIG_INVALID"
        ) from exc
    if (
        payload.get("schema_version") != SCHEMA
        or payload.get("enabled") is not True
        or payload.get("mode") != "SHADOW"
        or payload.get("runtime_authority") != "OBSERVATIONAL_ONLY"
        or payload.get("feature_schema") != FEATURE_SCHEMA_VERSION
        or int(payload.get("feature_count", 0)) != len(FEATURE_NAMES)
        or promotion.get("automatic_training") is not False
        or promotion.get("automatic_promotion") is not False
        or promotion.get("live_authority") is not False
        or promotion.get("owner_authorization_required") is not True
    ):
        raise EntryIntelligenceShadowError("AEGIS_ENTRY_INTELLIGENCE_AUTHORITY_INVALID")
    if (
        ranking.get("preserve_canonical_score") is not True
        or ranking.get("timing_state_precedes_score") is not True
    ):
        raise EntryIntelligenceShadowError("AEGIS_ENTRY_INTELLIGENCE_RANKING_INVALID")
    journal_root = (resolved.parent / str(evidence["journal_root"])).resolve()
    data_root = (root / "data").resolve()
    if journal_root != data_root and data_root not in journal_root.parents:
        raise EntryIntelligenceShadowError(
            "AEGIS_ENTRY_INTELLIGENCE_JOURNAL_ROOT_PROHIBITED"
        )
    signal_journal = journal_root / str(evidence["signal_journal"])
    outcome_journal = journal_root / str(evidence["outcome_journal"])
    if signal_journal.parent != journal_root or outcome_journal.parent != journal_root:
        raise EntryIntelligenceShadowError(
            "AEGIS_ENTRY_INTELLIGENCE_JOURNAL_PATH_PROHIBITED"
        )
    horizon = int(evidence["horizon_bars"])
    cost = float(evidence["round_trip_cost_fraction"])
    wait = int(timing["maximum_wait_bars"])
    maximum = int(ranking["maximum_counterfactual_candidates_per_cycle"])
    if (
        horizon <= 0
        or wait <= 0
        or maximum != 1
        or not math.isfinite(cost)
        or not 0.0 <= cost < 1.0
    ):
        raise EntryIntelligenceShadowError("AEGIS_ENTRY_INTELLIGENCE_LIMIT_INVALID")
    regime_settings = load_entry_quality_v2_config(
        regime_config_path,
        repo_root=root,
    ).regime_settings
    return EntryIntelligenceShadowConfig(
        observer_id=str(payload["observer_id"]),
        config_path=resolved,
        config_sha256=sha256_file(resolved),
        signal_journal=signal_journal,
        outcome_journal=outcome_journal,
        horizon_bars=horizon,
        round_trip_cost_fraction=cost,
        maximum_wait_bars=wait,
        require_global_bearish=bool(timing["require_global_bearish"]),
        require_local_bearish=bool(timing["require_local_bearish"]),
        require_trend_structure=bool(timing["require_trend_structure"]),
        require_short_ema_stack=bool(timing["require_short_ema_stack"]),
        require_bearish_confirmation_candle=bool(
            timing["require_bearish_confirmation_candle"]
        ),
        maximum_counterfactual_candidates_per_cycle=maximum,
        regime_settings=regime_settings,
    )


class EntryIntelligenceShadowRuntime:
    """Observe closed-candle alternatives without changing canonical selection."""

    def __init__(self, config: EntryIntelligenceShadowConfig) -> None:
        self.config = config
        self._hashing = Sha256HashProvider()
        self._global_regime = FactorizedRegimeAnalyzer(config.regime_settings)
        self._local_regime = FactorizedRegimeAnalyzer(config.regime_settings)
        self._signals = _AppendOnlyJournal(config.signal_journal, "event_id")
        self._outcomes = _AppendOnlyJournal(config.outcome_journal, "event_id")
        self._processed_timestamps = {
            str(row["market_timestamp"]) for row in self._signals.rows
        }
        self._pending: dict[str, _PendingSetup] = {}
        self._lock = threading.Lock()
        self.last_observation_at: datetime | None = None
        self.observation_errors = 0
        self._restore_state()

    @property
    def mode(self) -> EntryQualityV2Mode:
        return EntryQualityV2Mode.SHADOW

    def _restore_state(self) -> None:
        for row in sorted(
            self._signals.rows,
            key=lambda item: (str(item["market_timestamp"]), str(item["symbol"])),
        ):
            features = _mapping(row["regime_input"], "regime_input")
            self._observe_regime(
                str(row["symbol"]),
                _timestamp(str(row["market_timestamp"])),
                features,
            )
            timing = _mapping(row["entry_timing_shadow"], "entry_timing_shadow")
            symbol = str(row["symbol"])
            if timing["state"] == TimingState.WAITING_FOR_RETEST.value:
                self._pending[symbol] = _PendingSetup(
                    str(timing["setup_id"]),
                    str(timing["origin_event_id"]),
                    int(timing["age_bars"]),
                )
            else:
                self._pending.pop(symbol, None)

    def observe_batch(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        timestamp = str(batch["market_timestamp"])
        with self._lock:
            if timestamp in self._processed_timestamps:
                return self._overlay(timestamp)
            try:
                rows = self._build_rows(batch)
                self._apply_timing_ranks(rows)
                for row in rows:
                    self._signals.append(row)
                self._processed_timestamps.add(timestamp)
                self._mature_outcomes()
                self.last_observation_at = datetime.now(timezone.utc)
                return self._overlay(timestamp)
            except Exception:
                self.observation_errors += 1
                raise

    def _observe_regime(
        self,
        symbol: str,
        timestamp: datetime,
        features: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        shared = {
            "timestamp": timestamp,
            "range_mean_24": float(features["range_mean_24"]),
            "range_expansion": float(features["range_expansion"]),
            "chop_12": float(features["chop_12"]),
            "trend_strength_12": float(features["trend_strength_12"]),
        }
        global_result = self._global_regime.observe(
            RegimeV2Observation(
                symbol=f"GLOBAL::{symbol}",
                market_direction_6=float(features["market_direction_6"]),
                **shared,
            )
        )
        local_result = self._local_regime.observe(
            RegimeV2Observation(
                symbol=symbol,
                market_direction_6=float(features["ret_6"]),
                **shared,
            )
        )
        if global_result.direction is local_result.direction:
            alignment = f"ALIGNED_{global_result.direction.value}"
        elif DirectionRegime.UNKNOWN in {
            global_result.direction,
            local_result.direction,
        }:
            alignment = "UNKNOWN"
        else:
            alignment = "DIVERGENT"
        extension = (
            "EXTENDED_DOWN"
            if float(features["overextended_down_risk_proxy"]) > 0.0
            else (
                "EXTENDED_UP"
                if float(features["ret_12"]) > float(features["atr_12"]) * 3.0
                else "NORMAL"
            )
        )
        return {
            "schema_id": "aegis-factorized-regime-v3-shadow-v1",
            "mode": "SHADOW",
            "global_direction": global_result.direction.value,
            "symbol_direction": local_result.direction.value,
            "volatility": local_result.volatility.value,
            "structure": local_result.structure.value,
            "alignment": alignment,
            "extension": extension,
            "liquidity": "NOT_PRESENT_NO_CAUSAL_FEATURE",
            "evidence_ready": global_result.evidence_ready
            and local_result.evidence_ready,
            "global_stability_bars": global_result.direction_stability_bars,
            "symbol_stability_bars": local_result.direction_stability_bars,
            "volatility_stability_bars": local_result.volatility_stability_bars,
            "structure_stability_bars": local_result.structure_stability_bars,
            "selection_effect": "NONE",
            "exchange_authority": False,
        }

    def _build_rows(self, batch: Mapping[str, Any]) -> list[dict[str, Any]]:
        results = _mapping(batch["results"], "results")
        if set(results) != set(CANONICAL_SYMBOLS):
            raise EntryIntelligenceShadowError(
                "AEGIS_ENTRY_INTELLIGENCE_SYMBOLS_INVALID"
            )
        raw_ranking = batch.get("ranking")
        if isinstance(raw_ranking, list):
            canonical_evaluation = {}
            for item in raw_ranking:
                ranked = _mapping(item, "ranking_item")
                canonical_evaluation[str(ranked["symbol"])] = {
                    "rank": int(ranked["rank"]),
                    "eligible": bool(ranked["eligible"]),
                    "reason_codes": [str(reason) for reason in ranked["reason_codes"]],
                }
        else:
            canonical_evaluation = {
                symbol: {
                    "rank": index + 1,
                    "eligible": bool(results[symbol]["candidate"]["eligible"]),
                    "reason_codes": [],
                }
                for index, symbol in enumerate(
                    sorted(
                        CANONICAL_SYMBOLS,
                        key=lambda name: (
                            -float(results[name]["candidate"]["calibrated_score"]),
                            name,
                        ),
                    )
                )
            }
        rows: list[dict[str, Any]] = []
        for symbol in CANONICAL_SYMBOLS:
            result = _mapping(results[symbol], symbol)
            features = _mapping(result["research_features"], "research_features")
            candidate = _mapping(result["candidate"], "candidate")
            timestamp = _timestamp(str(result["market_timestamp"]))
            event_id = self._hashing.digest_value(
                {
                    "schema": SIGNAL_SCHEMA,
                    "observer": self.config.observer_id,
                    "timestamp": str(result["market_timestamp"]),
                    "symbol": symbol,
                    "feature_vector_hash": result["feature_vector_hash"],
                }
            )
            regime = self._observe_regime(symbol, timestamp, features)
            timing = self._timing_observation(
                symbol=symbol,
                event_id=event_id,
                candidate=candidate,
                selected=bool(result["selected"]),
                features=features,
                regime=regime,
            )
            bar = _mapping(result["market_bar"], "market_bar")
            layer = _mapping(result["layer"], "layer")
            rows.append(
                {
                    "schema_id": SIGNAL_SCHEMA,
                    "event_id": event_id,
                    "observer_id": self.config.observer_id,
                    "config_sha256": self.config.config_sha256,
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
                    "regime_input": {
                        name: float(features[name])
                        for name in (
                            "market_direction_6",
                            "ret_6",
                            "ret_12",
                            "atr_12",
                            "range_mean_24",
                            "range_expansion",
                            "chop_12",
                            "trend_strength_12",
                            "overextended_down_risk_proxy",
                        )
                    },
                    "control": {
                        "canonical_rank": canonical_evaluation[symbol]["rank"],
                        "selected": bool(result["selected"]),
                        "side": str(candidate["side"]),
                        "eligible": canonical_evaluation[symbol]["eligible"],
                        "reason_codes": canonical_evaluation[symbol]["reason_codes"],
                        "score": float(candidate["calibrated_score"]),
                        "candidate_hash": str(candidate["candidate_hash"]),
                    },
                    "uncertainty": {
                        "value": None,
                        "semantics": "NOT_APPLICABLE_SINGLE_ESTIMATOR",
                        "legacy_disagreement_value": float(layer["model_disagreement"]),
                        "selection_effect": "NONE",
                    },
                    "regime_v3_shadow": regime,
                    "entry_timing_shadow": timing,
                    "counterfactuals": {
                        "CONTROL_IMMEDIATE": (
                            "ENTER_NOW" if result["selected"] else "DO_NOT_ENTER"
                        ),
                        "CONTEXT_FILTERED": (
                            "ENTER_NOW"
                            if result["selected"] and self._short_context(regime)
                            else "DO_NOT_ENTER"
                        ),
                        "WAIT_RETEST": timing["paper_action"],
                        "EXHAUSTION_AVOID": (
                            "DO_NOT_ENTER"
                            if regime["extension"] == "EXTENDED_DOWN"
                            else "ENTER_NOW" if result["selected"] else "DO_NOT_ENTER"
                        ),
                        "TIMING_RANKED": "PENDING_CYCLE_RANK",
                    },
                    "timing_rank": None,
                    "timing_rank_selected": False,
                    "exchange_authority": False,
                    "exchange_mutations": 0,
                }
            )
        return rows

    def _timing_observation(
        self,
        *,
        symbol: str,
        event_id: str,
        candidate: Mapping[str, Any],
        selected: bool,
        features: Mapping[str, Any],
        regime: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        flags = {name: float(features[name]) > 0.0 for name in REVERSAL_FLAG_FEATURES}
        severe = any(
            flags[name]
            for name in (
                "failed_breakdown_proxy",
                "rebound_risk_proxy",
                "high_wick_reclaim_risk_proxy",
                "squeeze_plus_reclaim_risk_proxy",
            )
        )
        pending = self._pending.get(symbol)
        confirmed = self._timing_confirmed(features, regime, flags)
        if pending is not None:
            age = pending.age_bars + 1
            if severe:
                state, action, reason = (
                    TimingState.INVALIDATED,
                    "DO_NOT_ENTER",
                    "RECLAIM_OR_REVERSAL_INVALIDATED",
                )
                self._pending.pop(symbol, None)
            elif confirmed:
                state, action, reason = (
                    TimingState.TIMING_CONFIRMED,
                    "ENTER_NOW",
                    "CAUSAL_RETEST_CONFIRMED",
                )
                self._pending.pop(symbol, None)
            elif age >= self.config.maximum_wait_bars:
                state, action, reason = (
                    TimingState.EXPIRED,
                    "DO_NOT_ENTER",
                    "RETEST_WINDOW_EXPIRED",
                )
                self._pending.pop(symbol, None)
            else:
                state, action, reason = (
                    TimingState.WAITING_FOR_RETEST,
                    "WAIT_CONFIRMATION",
                    "RETEST_NOT_YET_CONFIRMED",
                )
                self._pending[symbol] = _PendingSetup(
                    pending.setup_id, pending.origin_event_id, age
                )
            setup_id, origin_event_id = pending.setup_id, pending.origin_event_id
        elif selected and str(candidate["side"]).endswith("SHORT"):
            setup_id = self._hashing.digest_value(
                {
                    "schema": "aegis-entry-timing-setup-v1",
                    "event": event_id,
                    "candidate": candidate["candidate_hash"],
                }
            )
            origin_event_id = event_id
            if severe:
                state, action, reason, age = (
                    TimingState.INVALIDATED,
                    "DO_NOT_ENTER",
                    "INITIAL_RECLAIM_OR_REVERSAL_INVALIDATED",
                    0,
                )
            elif confirmed and not any(flags.values()):
                state, action, reason, age = (
                    TimingState.TIMING_CONFIRMED,
                    "ENTER_NOW",
                    "INITIAL_TIMING_CLEAN",
                    0,
                )
            else:
                state, action, reason, age = (
                    TimingState.WAITING_FOR_RETEST,
                    "WAIT_CONFIRMATION",
                    "INITIAL_TIMING_RISK_REQUIRES_RETEST",
                    0,
                )
                self._pending[symbol] = _PendingSetup(setup_id, origin_event_id, age)
        else:
            setup_id = origin_event_id = None
            state, action, reason, age = (
                TimingState.NO_SETUP,
                "DO_NOT_ENTER",
                "CANONICAL_CANDIDATE_NOT_SELECTED",
                0,
            )
        return {
            "schema_id": "aegis-entry-timing-shadow-v1",
            "mode": "SHADOW",
            "state": state.value,
            "paper_action": action,
            "reason": reason,
            "setup_id": setup_id,
            "origin_event_id": origin_event_id,
            "age_bars": age,
            "maximum_wait_bars": self.config.maximum_wait_bars,
            "reversal_flags": flags,
            "selection_effect": "NONE",
            "exchange_authority": False,
        }

    def _timing_confirmed(
        self,
        features: Mapping[str, Any],
        regime: Mapping[str, Any],
        flags: Mapping[str, bool],
    ) -> bool:
        checks = (
            not self.config.require_global_bearish
            or regime["global_direction"] == "BEARISH",
            not self.config.require_local_bearish
            or regime["symbol_direction"] == "BEARISH",
            not self.config.require_trend_structure or regime["structure"] == "TREND",
            not self.config.require_short_ema_stack
            or float(features["trend_stack_short"]) > 0.0,
            not self.config.require_bearish_confirmation_candle
            or float(features["close_to_open_return"]) < 0.0,
            not any(flags.values()),
        )
        return all(checks)

    @staticmethod
    def _short_context(regime: Mapping[str, Any]) -> bool:
        return (
            regime["evidence_ready"] is True
            and regime["global_direction"] == "BEARISH"
            and regime["symbol_direction"] == "BEARISH"
            and regime["structure"] == "TREND"
        )

    def _apply_timing_ranks(self, rows: list[dict[str, Any]]) -> None:
        priorities = {
            TimingState.TIMING_CONFIRMED.value: 0,
            TimingState.WAITING_FOR_RETEST.value: 1,
            TimingState.CANDIDATE_SEEN.value: 2,
            TimingState.NO_SETUP.value: 3,
            TimingState.EXPIRED.value: 4,
            TimingState.INVALIDATED.value: 5,
        }
        ordered = sorted(
            rows,
            key=lambda row: (
                priorities[str(row["entry_timing_shadow"]["state"])],
                -float(row["control"]["score"]),
                str(row["symbol"]),
            ),
        )
        eligible = [
            row
            for row in ordered
            if bool(row["control"]["eligible"])
            and row["entry_timing_shadow"]["state"]
            == TimingState.TIMING_CONFIRMED.value
        ]
        selected = {
            row["event_id"]
            for row in eligible[
                : self.config.maximum_counterfactual_candidates_per_cycle
            ]
        }
        for rank, row in enumerate(ordered, start=1):
            row["timing_rank"] = rank
            row["timing_rank_selected"] = row["event_id"] in selected
            row["counterfactuals"]["TIMING_RANKED"] = (
                "ENTER_NOW" if row["event_id"] in selected else "DO_NOT_ENTER"
            )

    def _mature_outcomes(self) -> None:
        by_symbol = {symbol: [] for symbol in CANONICAL_SYMBOLS}
        for row in self._signals.rows:
            by_symbol[str(row["symbol"])].append(row)
        for rows in by_symbol.values():
            rows.sort(key=lambda row: str(row["market_timestamp"]))
            for index, signal in enumerate(rows):
                event_id = str(signal["event_id"])
                if (
                    event_id in self._outcomes.payloads
                    or not signal["control"]["selected"]
                ):
                    continue
                future = rows[index + 1 :]
                immediate = future[: self.config.horizon_bars]
                if len(immediate) < self.config.horizon_bars:
                    continue
                timing = signal["entry_timing_shadow"]
                setup_id = timing.get("setup_id")
                delayed_index = (
                    index
                    if timing["state"] == TimingState.TIMING_CONFIRMED.value
                    else None
                )
                terminal = timing["state"] in {
                    TimingState.INVALIDATED.value,
                    TimingState.EXPIRED.value,
                }
                if setup_id and delayed_index is None and not terminal:
                    for offset, future_row in enumerate(
                        future[: self.config.maximum_wait_bars], start=1
                    ):
                        future_timing = future_row["entry_timing_shadow"]
                        if future_timing.get("setup_id") != setup_id:
                            continue
                        if future_timing["state"] == TimingState.TIMING_CONFIRMED.value:
                            delayed_index = index + offset
                            break
                        if future_timing["state"] in {
                            TimingState.INVALIDATED.value,
                            TimingState.EXPIRED.value,
                        }:
                            terminal = True
                            break
                if delayed_index is None and not terminal:
                    continue
                delayed_future = (
                    rows[
                        delayed_index + 1 : delayed_index + 1 + self.config.horizon_bars
                    ]
                    if delayed_index is not None
                    else []
                )
                if (
                    delayed_index is not None
                    and len(delayed_future) < self.config.horizon_bars
                ):
                    continue
                immediate_metrics = self._short_metrics(signal, immediate)
                delayed_metrics = (
                    self._short_metrics(rows[delayed_index], delayed_future)
                    if delayed_index is not None
                    else None
                )
                self._outcomes.append(
                    {
                        "schema_id": OUTCOME_SCHEMA,
                        "event_id": event_id,
                        "symbol": signal["symbol"],
                        "signal_timestamp": signal["market_timestamp"],
                        "setup_id": setup_id,
                        "control_immediate": immediate_metrics,
                        "timing_entry": delayed_metrics,
                        "timing_disposition": (
                            "ENTERED" if delayed_metrics else "AVOIDED"
                        ),
                        "control_winner_after_costs": (
                            immediate_metrics["net_return_fraction"] > 0.0
                        ),
                        "timing_winner_after_costs": (
                            delayed_metrics["net_return_fraction"] > 0.0
                            if delayed_metrics is not None
                            else None
                        ),
                        "avoided_loss": (
                            delayed_metrics is None
                            and immediate_metrics["net_return_fraction"] <= 0.0
                        ),
                        "missed_winner": (
                            delayed_metrics is None
                            and immediate_metrics["net_return_fraction"] > 0.0
                        ),
                        "mae_improvement_fraction": (
                            immediate_metrics["mae_fraction"]
                            - delayed_metrics["mae_fraction"]
                            if delayed_metrics is not None
                            else None
                        ),
                        "exchange_mutations": 0,
                    }
                )

    def _short_metrics(
        self,
        entry_row: Mapping[str, Any],
        future: list[Mapping[str, Any]],
    ) -> Mapping[str, float]:
        entry = float(entry_row["market_bar"]["close"])
        exit_price = float(future[-1]["market_bar"]["close"])
        highs = [float(row["market_bar"]["high"]) for row in future]
        lows = [float(row["market_bar"]["low"]) for row in future]
        gross = (entry - exit_price) / entry
        return {
            "gross_return_fraction": gross,
            "net_return_fraction": gross - self.config.round_trip_cost_fraction,
            "mfe_fraction": max(0.0, (entry - min(lows)) / entry),
            "mae_fraction": max(0.0, (max(highs) - entry) / entry),
            "time_underwater_bars": sum(
                float(row["market_bar"]["close"]) > entry for row in future
            ),
        }

    def _overlay(self, timestamp: str) -> Mapping[str, Any]:
        latest = {
            str(row["symbol"]): row
            for row in self._signals.rows
            if str(row["market_timestamp"]) == timestamp
        }
        ranking = [
            {
                "rank": int(row["timing_rank"]),
                "canonical_rank": int(row["control"]["canonical_rank"]),
                "symbol": symbol,
                "canonical_score": float(row["control"]["score"]),
                "canonical_eligible": bool(row["control"]["eligible"]),
                "canonical_selected": bool(row["control"]["selected"]),
                "timing_state": row["entry_timing_shadow"]["state"],
                "timing_rank_selected": bool(row["timing_rank_selected"]),
            }
            for symbol, row in sorted(
                latest.items(), key=lambda item: int(item[1]["timing_rank"])
            )
        ]
        return {
            symbol: {
                "schema_id": HTTP_SCHEMA,
                "decision_cycle_id": row["decision_cycle_id"],
                "market_timestamp": row["market_timestamp"],
                "mode": "SHADOW",
                "status": "OBSERVATIONAL_ONLY",
                "uncertainty": dict(row["uncertainty"]),
                "regime_v3_shadow": dict(row["regime_v3_shadow"]),
                "entry_timing_shadow": dict(row["entry_timing_shadow"]),
                "counterfactuals": dict(row["counterfactuals"]),
                "candidate_ranking_shadow": ranking,
                "current_symbol_timing_rank": int(row["timing_rank"]),
                "current_symbol_canonical_rank": int(row["control"]["canonical_rank"]),
                "timing_rank_selected": bool(row["timing_rank_selected"]),
                "selection_effect": "NONE",
                "exchange_authority": False,
                "exchange_mutations": 0,
            }
            for symbol, row in latest.items()
        }

    def health(self) -> Mapping[str, Any]:
        return {
            "status": "ACTIVE",
            "mode": "SHADOW",
            "observer_id": self.config.observer_id,
            "signal_records": len(self._signals.rows),
            "paper_outcomes": len(self._outcomes.rows),
            "pending_setups": len(self._pending),
            "observation_errors": self.observation_errors,
            "exchange_authority": False,
            "exchange_mutations": 0,
            "last_observation_at": (
                self.last_observation_at.isoformat().replace("+00:00", "Z")
                if self.last_observation_at
                else None
            ),
        }


def build_entry_intelligence_shadow_observer(
    path: Path,
    *,
    repo_root: Path,
    regime_config_path: Path,
) -> EntryIntelligenceShadowRuntime:
    return EntryIntelligenceShadowRuntime(
        load_entry_intelligence_shadow_config(
            path,
            repo_root=repo_root,
            regime_config_path=regime_config_path,
        )
    )
