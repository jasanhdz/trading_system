"""Causal dataset construction with sealed holdout labels."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aegis_strategy_router.adapters.shared_market_data import SharedNeutralMinuteCandleSource
from aegis_strategy_router.domain.serialization import canonical_json_bytes, content_hash, utc_datetime
from aegis_strategy_router.domain.types import DataStatus, Side, Timeframe
from aegis_strategy_router.replay.precomputed_snapshot_builder import PrecomputedSnapshotBuilder

from independent_entry_quality_v1.features import add_cross_market_features, extract_features, feature_hash


@dataclass(frozen=True, slots=True)
class SymbolDatasetAudit:
    symbol: str
    snapshots: int
    rows: int
    rejected: int
    holdout_rows: int
    labeled_rows: int


def split_for(timestamp: pd.Timestamp, splits: dict[str, list[str]]) -> str:
    for name, bounds in splits.items():
        start, end = pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1])
        if start <= timestamp < end:
            return name
    return "OUTSIDE"


def build_symbol_rows(
    *, symbol: str, candle_root: Path, config: dict[str, Any], output_root: Path
) -> SymbolDatasetAudit:
    source = SharedNeutralMinuteCandleSource((candle_root,))
    frame, coverage = source.load(symbol)
    builder = PrecomputedSnapshotBuilder()
    first = min(pd.Timestamp(bounds[0]) for bounds in config["splits"].values())
    last = max(pd.Timestamp(bounds[1]) for bounds in config["splits"].values()) - pd.Timedelta(hours=1)
    anchors = pd.date_range(first, last, freq="1h", tz="UTC")
    open_ms = frame.open_time_ms.to_numpy(dtype="int64", copy=False)
    rows, rejected = [], []
    groups: dict[str, str] = {}
    for timestamp in anchors:
        partition = split_for(timestamp, config["splits"])
        if partition == "OUTSIDE":
            continue
        latest_open = int(timestamp.timestamp() * 1_000) - 60_000
        position = int(np.searchsorted(open_ms, latest_open, side="left"))
        if position >= len(frame) or int(open_ms[position]) != latest_open:
            rejected.append({"decision_at": timestamp.isoformat(), "reason": "REFERENCE_CANDLE_MISSING"})
            continue
        try:
            reference = float(frame.iloc[position].close)
            source_hash = builder.causal_source_hash(symbol, frame, timestamp.to_pydatetime())
            snapshot = builder.build(
                symbol=symbol, decision_at=timestamp.to_pydatetime(), built_at=timestamp.to_pydatetime(),
                reference_price=reference, one_minute=frame,
                source_versions={"entry_quality_source_hash": source_hash},
            )
            incomplete = [
                state.timeframe.value for state in snapshot.timeframes
                if state.status is not DataStatus.AVAILABLE
                or (state.structural is not None and state.structural.status is not DataStatus.AVAILABLE)
            ]
            if incomplete:
                raise ValueError("INCOMPLETE_SNAPSHOT:" + ",".join(incomplete))
            group_id = content_hash({"experiment": "ENTRY_QUALITY_V1", "symbol": symbol, "decision_at": timestamp.to_pydatetime()})
            block_id = timestamp.strftime("%Y-%m-%dT%H")
            for side in (Side.LONG, Side.SHORT):
                features, local_groups, available_at = extract_features(snapshot, side)
                groups.update(local_groups)
                row = {
                    "row_id": content_hash({"group_id": group_id, "side": side}),
                    "market_state_group_id": group_id,
                    "temporal_block_id": block_id,
                    "symbol": symbol,
                    "decision_at": timestamp,
                    "side": side.value,
                    "snapshot_id": snapshot.snapshot_id,
                    "snapshot_schema_version": snapshot.schema_version,
                    "snapshot_schema_hash": snapshot.schema_hash,
                    "source_hash": source_hash,
                    "max_feature_available_at": available_at,
                    "split": partition,
                    "label_state": "SEALED" if partition == "FINAL_HOLDOUT" else ("EMBARGO" if partition.startswith("EMBARGO") else "LABELED"),
                }
                row.update({f"feature__{name}": value for name, value in features.items()})
                if row["label_state"] == "LABELED":
                    row.update(_targets(frame, snapshot, side, config))
                rows.append(row)
        except ValueError as error:
            rejected.append({"decision_at": timestamp.isoformat(), "reason": str(error)})
    destination = output_root / "by_symbol"
    destination.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_parquet(destination / f"{symbol}.parquet", index=False, compression="zstd")
    audit = SymbolDatasetAudit(
        symbol=symbol, snapshots=result.snapshot_id.nunique(), rows=len(result), rejected=len(rejected),
        holdout_rows=int(result.label_state.eq("SEALED").sum()), labeled_rows=int(result.label_state.eq("LABELED").sum()),
    )
    (destination / f"{symbol}.audit.json").write_text(json.dumps({
        **asdict(audit),
        "coverage": {"rows": coverage.rows, "first_open_ms": coverage.first_open_ms, "last_open_ms": coverage.last_open_ms},
        "rejections": rejected,
        "aegis_loaded": False, "phase2_candidates_loaded": False, "holdout_labels_built": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (destination / f"{symbol}.groups.json").write_text(json.dumps(groups, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def combine_symbol_rows(*, symbols: list[str], output_root: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    frames, groups = [], {}
    for symbol in symbols:
        frames.append(pd.read_parquet(output_root / "by_symbol" / f"{symbol}.parquet"))
        groups.update(json.loads((output_root / "by_symbol" / f"{symbol}.groups.json").read_text(encoding="utf-8")))
    frame = pd.concat(frames, ignore_index=True).sort_values(["decision_at", "symbol", "side"], kind="mergesort")
    frame, cross_groups = add_cross_market_features(frame)
    groups.update(cross_groups)
    feature_columns = sorted(column for column in frame if column.startswith("feature__"))
    missing_rows = frame[feature_columns].isna().any(axis=1)
    if missing_rows.any():
        # Cross-market context is a frozen input. Drop the whole UTC block so
        # missing BTC/ETH cannot be imputed or create asymmetric side/symbol rows.
        invalid_times = set(frame.loc[missing_rows, "decision_at"])
        frame = frame.loc[~frame.decision_at.isin(invalid_times)].copy()
    if frame[feature_columns].isna().any().any():
        raise ValueError("CROSS_MARKET_OR_FEATURE_JOIN_MISSING_AFTER_FAIL_CLOSED")
    frame["feature_values_hash"] = [feature_hash(row) for row in frame[feature_columns].to_dict("records")]
    development = frame.loc[frame.label_state.eq("LABELED")].copy()
    holdout = frame.loc[frame.label_state.eq("SEALED")].drop(
        columns=[column for column in frame if column.startswith("target__")], errors="ignore"
    )
    embargo = frame.loc[frame.label_state.eq("EMBARGO")].drop(
        columns=[column for column in frame if column.startswith("target__")], errors="ignore"
    )
    development.to_parquet(output_root / "development_labeled.parquet", index=False, compression="zstd")
    holdout.to_parquet(output_root / "final_holdout_features_sealed.parquet", index=False, compression="zstd")
    embargo.to_parquet(output_root / "embargo_features.parquet", index=False, compression="zstd")
    return development, groups


def _targets(frame: pd.DataFrame, snapshot: Any, side: Side, config: dict[str, Any]) -> dict[str, Any]:
    horizon = int(config["primary_target"]["horizon_minutes"])
    start_ms = int(snapshot.decision_at.timestamp() * 1_000)
    future = frame.loc[
        frame.open_time_ms.ge(start_ms) & frame.open_time_ms.lt(start_ms + horizon * 60_000)
    ].sort_values("open_time_ms", kind="mergesort")
    if len(future) != horizon:
        raise ValueError("OUTCOME_HORIZON_INCOMPLETE")
    state15 = next(state for state in snapshot.timeframes if state.timeframe is Timeframe.M15)
    atr = float(state15.structural.atr14)
    reference = snapshot.reference_price
    direction = 1.0 if side is Side.LONG else -1.0
    barrier_abs = float(config["primary_target"]["favorable_barrier_atr"]) * atr
    barrier_bps = barrier_abs / reference * 10_000.0
    high, low, close = (future[name].to_numpy(float) for name in ("high", "low", "close"))
    if side is Side.LONG:
        favorable = high >= reference + barrier_abs
        adverse = low <= reference - barrier_abs
        favorable_excursion = (high - reference) / reference * 10_000.0
        adverse_excursion = (reference - low) / reference * 10_000.0
    else:
        favorable = low <= reference - barrier_abs
        adverse = high >= reference + barrier_abs
        favorable_excursion = (reference - low) / reference * 10_000.0
        adverse_excursion = (high - reference) / reference * 10_000.0
    f_index, a_index = _first(favorable), _first(adverse)
    if a_index is not None and (f_index is None or a_index <= f_index):
        label, gross = "ADVERSE_FIRST", -barrier_bps
    elif f_index is not None:
        label, gross = "FAVORABLE_FIRST", barrier_bps
    else:
        label = "NEITHER"
        gross = direction * (close[-1] - reference) / reference * 10_000.0
    mfe = max(0.0, float(np.max(favorable_excursion)))
    mae = max(0.0, float(np.max(adverse_excursion)))
    absolute_excursion = max(float(np.max((high - reference) / reference * 10_000.0)), float(np.max((reference - low) / reference * 10_000.0)))
    opportunity_threshold = max(barrier_bps, float(config["primary_target"]["conservative_cost_bps"]))
    up_excursion = (high - reference) / reference * 10_000.0
    down_excursion = (reference - low) / reference * 10_000.0
    opportunity_hits = np.maximum(up_excursion, down_excursion) >= opportunity_threshold
    opportunity_index = _first(opportunity_hits)
    latency = direction * (float(future.iloc[0].open) - reference) / reference * 10_000.0
    return {
        "target__path_label": label,
        "target__favorable_first": int(label == "FAVORABLE_FIRST"),
        "target__adverse_first": int(label == "ADVERSE_FIRST"),
        "target__neither": int(label == "NEITHER"),
        "target__opportunity": int(absolute_excursion >= opportunity_threshold),
        "target__absolute_excursion_bps": absolute_excursion,
        "target__time_to_opportunity_minutes": opportunity_index + 1 if opportunity_index is not None else math.nan,
        "target__barrier_bps": barrier_bps,
        "target__mfe_bps": mfe,
        "target__mae_bps": mae,
        "target__mfe_minus_mae_bps": mfe - mae,
        "target__mfe_mae_ratio": mfe / mae if mae > 0 else (1_000.0 if mfe > 0 else 0.0),
        "target__fixed_return_bps": direction * (close[-1] - reference) / reference * 10_000.0,
        "target__time_to_favorable_minutes": f_index + 1 if f_index is not None else math.nan,
        "target__time_to_adverse_minutes": a_index + 1 if a_index is not None else math.nan,
        "target__gross_common_payoff_bps": gross,
        "target__net_common_payoff_bps": gross - float(config["primary_target"]["conservative_cost_bps"]),
        "target__latency_shortfall_bps": latency,
        "target__latency_stressed_net_bps": gross - float(config["primary_target"]["conservative_cost_bps"]) - latency,
        "target__net_positive": int(gross - float(config["primary_target"]["conservative_cost_bps"]) > 0.0),
    }


def _first(values: np.ndarray) -> int | None:
    positions = np.flatnonzero(values)
    return int(positions[0]) if len(positions) else None
