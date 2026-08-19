"""Frozen rules-only historical falsification for independent candidates."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from aegis_strategy_router.adapters.shared_market_data import SharedNeutralMinuteCandleSource
from aegis_strategy_router.candidates.contracts import CandidateEvaluation, Strategy
from aegis_strategy_router.candidates.frozen_rules import (
    feature_value,
    latest_pivots,
)
from aegis_strategy_router.candidates.registry import CandidateGeneratorRegistry, CandidateReplayContext
from aegis_strategy_router.domain.serialization import canonical_json_bytes, utc_datetime
from aegis_strategy_router.domain.types import DataStatus, MarketSnapshot, PivotKind, Side, Timeframe
from aegis_strategy_router.replay.general_market_pipeline import (
    ANCHOR_MINUTES,
    is_candidate_population_event,
    select_independent_episodes,
)
from aegis_strategy_router.replay.precomputed_snapshot_builder import PrecomputedSnapshotBuilder


RESULT_SCHEMA = "aegis-strategy-router-retrospective-symbol-v1"


@dataclass(frozen=True, slots=True)
class SymbolRunResult:
    symbol: str
    snapshots: int
    candidate_evaluations: int
    population_candidates: int
    independent_episodes: int
    suppressed_candidates: int
    unknown: int
    ineligible: int
    rejected_anchors: int
    first_anchor: str | None
    last_anchor: str | None
    status_counts: dict[str, int]


def run_symbol(
    *,
    symbol: str,
    candle_root: Path,
    start_at: datetime,
    last_at: datetime,
    output_root: Path,
) -> SymbolRunResult:
    source = SharedNeutralMinuteCandleSource((candle_root,))
    frame, coverage = source.load(symbol)
    builder = PrecomputedSnapshotBuilder()
    generators = CandidateGeneratorRegistry()
    replay = CandidateReplayContext()
    start = pd.Timestamp(utc_datetime(start_at)).ceil(f"{ANCHOR_MINUTES}min")
    last = pd.Timestamp(utc_datetime(last_at)).floor(f"{ANCHOR_MINUTES}min")
    anchors = pd.date_range(start, last, freq=f"{ANCHOR_MINUTES}min", tz="UTC")
    status_counts: Counter[tuple[str, str, str, str]] = Counter()
    substate_counts: Counter[str] = Counter()
    population: list[CandidateEvaluation] = []
    candidate_snapshots: dict[str, MarketSnapshot] = {}
    baseline_rows: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    snapshot_count = 0

    open_ms = frame["open_time_ms"].to_numpy(dtype="int64", copy=False)
    for value in anchors:
        decision_at = value.to_pydatetime()
        latest_open = int(value.timestamp() * 1_000) - 60_000
        index = int(np.searchsorted(open_ms, latest_open, side="left"))
        if index >= len(frame) or int(open_ms[index]) != latest_open:
            rejected.append({"anchor": value.isoformat(), "reason": "REFERENCE_CANDLE_MISSING"})
            continue
        reference_price = float(frame.iloc[index].close)
        try:
            source_hash = builder.causal_source_hash(symbol, frame, decision_at)
            snapshot = builder.build(
                symbol=symbol,
                decision_at=decision_at,
                built_at=decision_at,
                reference_price=reference_price,
                one_minute=frame,
                proposed_side=None,
                signal_id=None,
                source_versions={
                    "general_market_pipeline": "aegis-strategy-router-general-market-v1",
                    "fresh_candle_source_hash": source_hash,
                },
            )
            incomplete = tuple(
                state.timeframe.value
                for state in snapshot.timeframes
                if state.status is not DataStatus.AVAILABLE
                or (state.structural is not None and state.structural.status is not DataStatus.AVAILABLE)
            )
            if incomplete:
                raise ValueError(f"INCOMPLETE_SNAPSHOT:{','.join(incomplete)}")
            snapshot_count += 1
            for side in (Side.LONG, Side.SHORT):
                baseline = reconstruct_outcome(frame, snapshot, side)
                baseline["persistence_aligned"] = (
                    (1.0 if side is Side.LONG else -1.0)
                    * feature_value(snapshot, Timeframe.M15, "return_3_bps") > 0
                )
                baseline_rows.append(baseline)
                for candidate in generators.generate_all_replay(snapshot, side, replay):
                    status_counts[(candidate.strategy.value, side.value, candidate.status.value, symbol)] += 1
                    if candidate.substate is not None:
                        substate_counts[candidate.substate.value] += 1
                    if is_candidate_population_event(candidate):
                        population.append(candidate)
                        candidate_snapshots[candidate.candidate_episode_id] = snapshot
        except ValueError as error:
            rejected.append({"anchor": value.isoformat(), "reason": str(error)})

    independent, suppressed = select_independent_episodes(population)
    candidates_by_id = {item.candidate_episode_id: item for item in population}
    outcome_rows = []
    for episode in independent:
        candidate = candidates_by_id[episode.source_candidate_episode_id]
        snapshot = candidate_snapshots[candidate.candidate_episode_id]
        outcome = reconstruct_outcome(frame, snapshot, candidate.side, candidate)
        outcome.update({
            "independent_episode_id": episode.independent_episode_id,
            "candidate_episode_id": candidate.candidate_episode_id,
            "logical_setup_id": episode.logical_setup_id,
            "strategy": candidate.strategy.value,
            "substate": candidate.substate.value if candidate.substate else None,
            "disposition": candidate.disposition.value if candidate.disposition else None,
        })
        outcome_rows.append(outcome)

    destination = output_root / symbol
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(baseline_rows).to_parquet(destination / "baseline_anchors.parquet", index=False)
    pd.DataFrame(outcome_rows).to_parquet(destination / "independent_outcomes.parquet", index=False)
    with (destination / "population_candidates.jsonl").open("wb") as handle:
        for candidate in population:
            handle.write(canonical_json_bytes(candidate.to_primitive()) + b"\n")
    (destination / "suppressed_candidates.json").write_text(
        json.dumps([{"candidate_episode_id": item, "reason": reason} for item, reason in suppressed],
                   indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = SymbolRunResult(
        symbol=symbol,
        snapshots=snapshot_count,
        candidate_evaluations=sum(status_counts.values()),
        population_candidates=len(population),
        independent_episodes=len(independent),
        suppressed_candidates=len(suppressed),
        unknown=sum(count for (strategy, side, status, sym), count in status_counts.items() if status == "UNKNOWN"),
        ineligible=sum(count for (strategy, side, status, sym), count in status_counts.items() if status == "INELIGIBLE"),
        rejected_anchors=len(rejected),
        first_anchor=anchors[0].isoformat() if len(anchors) else None,
        last_anchor=anchors[-1].isoformat() if len(anchors) else None,
        status_counts={"|".join(key): count for key, count in sorted(status_counts.items())},
    )
    audit = {
        "schema": RESULT_SCHEMA,
        **asdict(result),
        "substate_counts": dict(sorted(substate_counts.items())),
        "candle_coverage": asdict(coverage),
        "rejected": rejected,
        "aegis_signals_loaded": False,
        "rules_changed": False,
        "sealed_holdouts_loaded": False,
    }
    (destination / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def reconstruct_outcome(
    frame: pd.DataFrame,
    snapshot: MarketSnapshot,
    side: Side,
    candidate: CandidateEvaluation | None = None,
) -> dict[str, Any]:
    direction = 1.0 if side is Side.LONG else -1.0
    start_ms = int(snapshot.decision_at.timestamp() * 1_000)
    finish_ms = start_ms + 60 * 60_000
    future = frame.loc[
        frame["open_time_ms"].ge(start_ms) & frame["open_time_ms"].lt(finish_ms)
    ].sort_values("open_time_ms", kind="mergesort")
    if len(future) != 60 or int(future.iloc[0].open_time_ms) != start_ms:
        raise ValueError(f"OUTCOME_HORIZON_INCOMPLETE:{snapshot.symbol}:{snapshot.decision_at.isoformat()}")
    state15 = next(state for state in snapshot.timeframes if state.timeframe is Timeframe.M15)
    if state15.structural is None or state15.structural.atr14 is None:
        raise ValueError("REFERENCE_ATR_UNAVAILABLE")
    reference = snapshot.reference_price
    atr = float(state15.structural.atr14)
    barrier_abs = 0.5 * atr
    barrier_bps = barrier_abs / reference * 10_000.0
    favorable_price = reference + direction * barrier_abs
    adverse_price = reference - direction * barrier_abs
    high = future["high"].to_numpy(float)
    low = future["low"].to_numpy(float)
    close = future["close"].to_numpy(float)
    if side is Side.LONG:
        favorable_hits = high >= favorable_price
        adverse_hits = low <= adverse_price
        favorable_excursion = (high - reference) / reference * 10_000.0
        adverse_excursion = (reference - low) / reference * 10_000.0
    else:
        favorable_hits = low <= favorable_price
        adverse_hits = high >= adverse_price
        favorable_excursion = (reference - low) / reference * 10_000.0
        adverse_excursion = (high - reference) / reference * 10_000.0
    first_favorable = _first_true(favorable_hits)
    first_adverse = _first_true(adverse_hits)
    if first_adverse is not None and (first_favorable is None or first_adverse <= first_favorable):
        label = "ADVERSE_FIRST"
        gross = -barrier_bps
    elif first_favorable is not None:
        label = "FAVORABLE_FIRST"
        gross = barrier_bps
    else:
        label = "NEITHER"
        gross = direction * (close[-1] - reference) / reference * 10_000.0
    fixed_return = direction * (close[-1] - reference) / reference * 10_000.0
    mfe = max(0.0, float(np.max(favorable_excursion)))
    mae = max(0.0, float(np.max(adverse_excursion)))
    path = np.concatenate(([reference], close))
    path_total = float(np.abs(np.diff(path)).sum())
    efficiency = abs(float(path[-1] - path[0])) / path_total if path_total > 0 else 0.0
    latency_shortfall = direction * (float(future.iloc[0].open) - reference) / reference * 10_000.0
    metadata = dict(candidate.metadata) if candidate is not None else {}
    consumed = _consumed_move_bps(frame, metadata.get("setup_started_at"), snapshot, side)
    return {
        "snapshot_id": snapshot.snapshot_id,
        "symbol": snapshot.symbol,
        "side": side.value,
        "decision_at": snapshot.decision_at,
        "month": snapshot.decision_at.strftime("%Y-%m"),
        "hour_block": snapshot.decision_at.strftime("%Y-%m-%dT%H"),
        "reference_price": reference,
        "reference_atr15": atr,
        "barrier_bps": barrier_bps,
        "label": label,
        "favorable_first": label == "FAVORABLE_FIRST",
        "adverse_first": label == "ADVERSE_FIRST",
        "neither": label == "NEITHER",
        "fixed_return_bps": fixed_return,
        "gross_common_payoff_bps": gross,
        "net_common_payoff_bps": gross - 20.0,
        "latency_shortfall_bps": latency_shortfall,
        "latency_stressed_net_bps": gross - 20.0 - latency_shortfall,
        "mfe_bps": mfe,
        "mae_bps": mae,
        "mfe_gt_mae": mfe > mae,
        "mfe_mae_ratio": mfe / mae if mae > 0 else (math.inf if mfe > 0 else 0.0),
        "time_to_favorable_minutes": first_favorable + 1 if first_favorable is not None else math.nan,
        "time_to_adverse_minutes": first_adverse + 1 if first_adverse is not None else math.nan,
        "path_efficiency": efficiency,
        "structural_invalidation": _structural_invalidation(future, snapshot, side, candidate),
        "consumed_move_bps": consumed,
        "remaining_mfe_bps": mfe,
    }


def _first_true(values: np.ndarray) -> int | None:
    positions = np.flatnonzero(values)
    return int(positions[0]) if len(positions) else None


def _consumed_move_bps(
    frame: pd.DataFrame, started_at: object, snapshot: MarketSnapshot, side: Side
) -> float:
    if started_at is None:
        return 0.0
    timestamp = pd.Timestamp(started_at)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    latest_open = int(timestamp.timestamp() * 1_000) - 60_000
    rows = frame.loc[frame["open_time_ms"].eq(latest_open), "close"]
    if len(rows) != 1:
        return math.nan
    direction = 1.0 if side is Side.LONG else -1.0
    start_price = float(rows.iloc[0])
    return direction * (snapshot.reference_price - start_price) / start_price * 10_000.0


def _structural_invalidation(
    future: pd.DataFrame,
    snapshot: MarketSnapshot,
    side: Side,
    candidate: CandidateEvaluation | None,
) -> bool:
    if candidate is None:
        return False
    metadata = dict(candidate.metadata)
    direction = 1.0 if side is Side.LONG else -1.0
    if candidate.strategy is Strategy.BREAKOUT_RETEST and "level_price" in metadata:
        level = float(metadata["level_price"])
        closes = future.iloc[14::15].close.to_numpy(float)
        return bool(np.any(closes < level)) if side is Side.LONG else bool(np.any(closes > level))
    if candidate.strategy is Strategy.RANGE_MEAN_REVERSION and "support_price" in metadata:
        boundary = float(metadata["support_price"] if side is Side.LONG else metadata["resistance_price"])
        closes = future.iloc[14::15].close.to_numpy(float)
        return bool(np.any(closes < boundary)) if side is Side.LONG else bool(np.any(closes > boundary))
    timeframe = Timeframe.H1 if candidate.strategy is Strategy.PULLBACK_CONTINUATION else Timeframe.M15
    kind = PivotKind.LOW if side is Side.LONG else PivotKind.HIGH
    try:
        pivot = latest_pivots(snapshot, timeframe, kind, count=1)[-1].price
    except ValueError:
        return False
    step = 5 if timeframe is Timeframe.H1 else 15
    closes = future.iloc[step - 1 :: step].close.to_numpy(float)
    return bool(np.any(direction * (closes - pivot) < 0))
