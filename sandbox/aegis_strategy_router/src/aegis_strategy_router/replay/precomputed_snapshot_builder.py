"""Precomputed implementation of the frozen causal snapshot contract.

This module changes replay cost, not semantics. Its acceptance test compares
canonical snapshot bytes against DeterministicSnapshotBuilder.
"""

from __future__ import annotations

import bisect
import hashlib
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

import numpy as np
import pandas as pd

from aegis.research.live_entry_multitimeframe import aggregate_klines, indicator_frame
from aegis_strategy_router.adapters.existing_features import (
    ExistingResearchFeatureAdapter,
    REQUIRED_WARMUP_BARS,
)
from aegis_strategy_router.domain.serialization import utc_datetime
from aegis_strategy_router.domain.types import (
    Candle,
    ConfirmedPivot,
    DataStatus,
    FeatureObservation,
    FeatureSet,
    LevelDistance,
    MarketSnapshot,
    PivotKind,
    Side,
    StructuralContext,
    Timeframe,
    TimeframeSnapshot,
)
from aegis_strategy_router.features.structural_levels import (
    CLUSTER_TOLERANCE_ATR,
    cluster_confirmed_pivots,
)
from aegis_strategy_router.replay.fresh_pipeline import CANDLE_COLUMNS
from aegis_strategy_router.replay.snapshot_builder import ALL_TIMEFRAMES
from aegis_strategy_router.schemas import (
    EXISTING_FEATURE_NAMES,
    FEATURE_OWNER,
    FeatureSchema,
    SNAPSHOT_SCHEMA_VERSION,
)


PRECOMPUTED_BUILDER_VERSION = "precomputed-causal-snapshot-builder-v1"


@dataclass(slots=True)
class _HashCursor:
    values: bytes
    row_count: int = 0
    hasher: object | None = None

    def digest_at(self, rows: int) -> str:
        if rows < 0:
            rows = 0
        if self.hasher is None or rows < self.row_count:
            return hashlib.sha256(self.values[: rows * 8]).hexdigest()
        self.hasher.update(self.values[self.row_count * 8 : rows * 8])
        self.row_count = rows
        return self.hasher.hexdigest()


@dataclass(frozen=True, slots=True)
class _PreparedTimeframe:
    timeframe: Timeframe
    candles: tuple[Candle, ...]
    close_ns: tuple[int, ...]
    indicators: pd.DataFrame
    atr14: tuple[float | None, ...]


@dataclass(slots=True)
class _PreparedSymbol:
    frame_identity: int
    frame: pd.DataFrame
    open_ms: np.ndarray
    hash_cursor: _HashCursor
    timeframes: dict[Timeframe, _PreparedTimeframe]
    structural_bases: dict[Timeframe, tuple[int, StructuralContext]]


class PrecomputedSnapshotBuilder:
    """Build the frozen snapshot using precomputed causal candle transforms."""

    def __init__(self) -> None:
        self.schema = FeatureSchema.existing_multitimeframe(
            timeframe.value for timeframe in ALL_TIMEFRAMES
        )
        self.feature_adapter = ExistingResearchFeatureAdapter(self.schema)
        self._prepared: dict[str, _PreparedSymbol] = {}

    def causal_source_hash(
        self, symbol: str, one_minute: pd.DataFrame, decision_at: datetime
    ) -> str:
        prepared = self._prepare(symbol, one_minute)
        boundary_ms = int(utc_datetime(decision_at).timestamp() * 1_000)
        rows = int(np.searchsorted(prepared.open_ms, boundary_ms - 60_000, side="right"))
        return prepared.hash_cursor.digest_at(rows)

    def build(
        self,
        *,
        symbol: str,
        decision_at: datetime,
        reference_price: float,
        one_minute: pd.DataFrame,
        proposed_side: Side | None = None,
        signal_id: str | None = None,
        built_at: datetime | None = None,
        source_versions: Mapping[str, str] | None = None,
    ) -> MarketSnapshot:
        boundary = utc_datetime(decision_at)
        prepared = self._prepare(symbol, one_minute)
        states = tuple(
            self._timeframe_state(prepared, timeframe, boundary, reference_price)
            for timeframe in ALL_TIMEFRAMES
        )
        for state in states:
            state.assert_causal(boundary)
        versions = {
            "feature_adapter": self.feature_adapter.source_version,
            "snapshot_builder": "deterministic-snapshot-builder-v1",
            "structural_adapter": "confirmed-pivots-complete-linkage-v1",
            **dict(source_versions or {}),
        }
        return MarketSnapshot.create(
            schema_version=SNAPSHOT_SCHEMA_VERSION,
            schema_hash=self.schema.hash,
            symbol=symbol,
            decision_at=boundary,
            built_at=utc_datetime(built_at or boundary),
            proposed_side=proposed_side,
            signal_id=signal_id,
            reference_price=reference_price,
            timeframes=states,
            source_versions=versions,
        )

    def _prepare(self, symbol: str, frame: pd.DataFrame) -> _PreparedSymbol:
        cached = self._prepared.get(symbol)
        if cached is not None and cached.frame_identity == id(frame):
            return cached
        boundary = pd.to_datetime(frame["open_time_ms"].max(), unit="ms", utc=True)
        boundary += pd.Timedelta(minutes=1)
        normalized = self.feature_adapter._closed_source(frame, boundary.to_pydatetime())
        timeframes = {}
        for timeframe in ALL_TIMEFRAMES:
            aggregated = aggregate_klines(normalized, timeframe.minutes)
            indicators = indicator_frame(normalized, timeframe.minutes).reset_index(drop=True)
            candles = tuple(_candle(row, timeframe) for row in aggregated.itertuples(index=False))
            close_ns = tuple(int(pd.Timestamp(candle.close_at).value) for candle in candles)
            timeframes[timeframe] = _PreparedTimeframe(
                timeframe=timeframe,
                candles=candles,
                close_ns=close_ns,
                indicators=indicators,
                atr14=_atr_path(candles),
            )
        hashes = pd.util.hash_pandas_object(
            normalized.loc[:, list(CANDLE_COLUMNS)], index=False
        ).values.tobytes()
        prepared = _PreparedSymbol(
            frame_identity=id(frame),
            frame=normalized,
            open_ms=normalized["open_time_ms"].to_numpy(dtype="int64", copy=True),
            hash_cursor=_HashCursor(values=hashes, hasher=hashlib.sha256()),
            timeframes=timeframes,
            structural_bases={},
        )
        self._prepared[symbol] = prepared
        return prepared

    def _timeframe_state(
        self,
        prepared: _PreparedSymbol,
        timeframe: Timeframe,
        boundary: datetime,
        reference_price: float,
    ) -> TimeframeSnapshot:
        table = prepared.timeframes[timeframe]
        count = bisect.bisect_right(table.close_ns, int(pd.Timestamp(boundary).value))
        feature_names = tuple(f"tf{timeframe.value}__{name}" for name in EXISTING_FEATURE_NAMES)
        if count == 0:
            observations = tuple(
                FeatureObservation(
                    name, None, None, None, FEATURE_OWNER, self.feature_adapter.source_version,
                    DataStatus.UNKNOWN, "NO_FULLY_CLOSED_BAR",
                )
                for name in feature_names
            )
            return TimeframeSnapshot(
                timeframe, DataStatus.UNKNOWN, 0, REQUIRED_WARMUP_BARS, None, None,
                FeatureSet(observations, self.schema.hash), None, "NO_FULLY_CLOSED_BAR",
            )
        candle = table.candles[count - 1]
        row = table.indicators.iloc[count - 1]
        prefix = f"tf{timeframe.minutes}m__"
        observations = []
        unavailable = 0
        for base_name, full_name in zip(EXISTING_FEATURE_NAMES, feature_names):
            raw = row.get(prefix + base_name)
            if raw is None or not math.isfinite(float(raw)):
                unavailable += 1
                observations.append(FeatureObservation(
                    full_name, None, candle.close_at, None, FEATURE_OWNER,
                    self.feature_adapter.source_version, DataStatus.UNKNOWN,
                    "FEATURE_WARMUP_INCOMPLETE",
                ))
            else:
                observations.append(FeatureObservation(
                    full_name, float(raw), candle.close_at, candle.close_at, FEATURE_OWNER,
                    self.feature_adapter.source_version, DataStatus.AVAILABLE,
                ))
        status = (
            DataStatus.AVAILABLE
            if count >= REQUIRED_WARMUP_BARS and unavailable == 0
            else DataStatus.UNKNOWN
        )
        structural = None
        if timeframe.structural_lookback is not None:
            structural = self._structural_context(
                prepared, table, count, boundary, reference_price
            )
        return TimeframeSnapshot(
            timeframe=timeframe,
            status=status,
            candle_count=count,
            required_warmup_bars=REQUIRED_WARMUP_BARS,
            latest_closed_at=candle.close_at,
            latest_candle=candle,
            features=FeatureSet(tuple(observations), self.schema.hash),
            structural=structural,
            reason=None if status is DataStatus.AVAILABLE else "FEATURE_WARMUP_INCOMPLETE",
        )

    def _structural_context(
        self,
        prepared: _PreparedSymbol,
        table: _PreparedTimeframe,
        count: int,
        boundary: datetime,
        reference_price: float,
    ) -> StructuralContext:
        cached = prepared.structural_bases.get(table.timeframe)
        base = cached[1] if cached is not None and cached[0] == count else None
        if base is None:
            atr = table.atr14[count - 1]
            pivots = _confirmed_pivots(table.candles, count, table.timeframe.structural_lookback or 0)
            if atr is None:
                base = StructuralContext(
                    status=DataStatus.UNKNOWN,
                    pivots=pivots,
                    reason="STRUCTURAL_ATR14_WARMUP_INCOMPLETE",
                )
            else:
                tolerance = CLUSTER_TOLERANCE_ATR * atr
                levels = cluster_confirmed_pivots(
                    pivots, timeframe=table.timeframe, tolerance=tolerance
                )
                base = StructuralContext(
                    status=DataStatus.AVAILABLE,
                    pivots=pivots,
                    levels=levels,
                    atr14=atr,
                    cluster_tolerance=tolerance,
                    reference_price=reference_price,
                )
            # Replays are chronological. Only the current closed-bar context is
            # reusable; retaining every historical context would not change a
            # snapshot but would make a multi-month replay consume unbounded RAM.
            prepared.structural_bases[table.timeframe] = (count, base)
        if base.status is not DataStatus.AVAILABLE or base.atr14 is None:
            return base
        below = sorted(
            (level for level in base.levels if level.price <= reference_price),
            key=lambda level: (-level.price, level.level_id),
        )
        above = sorted(
            (level for level in base.levels if level.price >= reference_price),
            key=lambda level: (level.price, level.level_id),
        )
        return StructuralContext(
            status=DataStatus.AVAILABLE,
            pivots=base.pivots,
            levels=base.levels,
            atr14=base.atr14,
            cluster_tolerance=base.cluster_tolerance,
            reference_price=reference_price,
            nearest_below=_distance(below[0], reference_price, base.atr14) if below else None,
            nearest_above=_distance(above[0], reference_price, base.atr14) if above else None,
        )


def _candle(row: object, timeframe: Timeframe) -> Candle:
    open_at = pd.Timestamp(getattr(row, "open_time")).to_pydatetime()
    close_at = pd.Timestamp(getattr(row, "close_time")).to_pydatetime()
    return Candle(
        open_at=open_at,
        close_at=close_at,
        open=float(getattr(row, "open")),
        high=float(getattr(row, "high")),
        low=float(getattr(row, "low")),
        close=float(getattr(row, "close")),
        volume=float(getattr(row, "volume")),
        taker_buy_volume=float(getattr(row, "taker_buy_volume")),
        available_at=close_at,
        source_id=f"public-1m:{int(pd.Timestamp(open_at).timestamp() * 1_000)}:{timeframe.value}",
        complete=bool(getattr(row, "bar_count") == timeframe.minutes),
    )


def _atr_path(candles: tuple[Candle, ...]) -> tuple[float | None, ...]:
    values: list[float | None] = []
    atr = None
    previous = None
    for index, candle in enumerate(candles):
        candidates = [candle.high - candle.low]
        if previous is not None:
            candidates.extend((abs(candle.high - previous), abs(candle.low - previous)))
        true_range = max(candidates)
        atr = true_range if atr is None else (13.0 / 14.0) * atr + (1.0 / 14.0) * true_range
        previous = candle.close
        values.append(float(atr) if index + 1 >= 14 and atr > 0 else None)
    return tuple(values)


def _confirmed_pivots(
    candles: tuple[Candle, ...], count: int, lookback: int
) -> tuple[ConfirmedPivot, ...]:
    lower = max(2, count - lookback)
    pivots = []
    for index in range(lower, count - 2):
        center = candles[index]
        neighbors = (
            candles[index - 2], candles[index - 1],
            candles[index + 1], candles[index + 2],
        )
        available_at = max(center.available_at, candles[index + 2].available_at)
        if all(center.high > value.high for value in neighbors):
            pivots.append(ConfirmedPivot(
                PivotKind.HIGH, center.high, index, center.close_at, available_at
            ))
        if all(center.low < value.low for value in neighbors):
            pivots.append(ConfirmedPivot(
                PivotKind.LOW, center.low, index, center.close_at, available_at
            ))
    return tuple(sorted(
        pivots,
        key=lambda value: (
            value.pivot_at, value.available_at, value.bar_index, value.price
        ),
    ))


def _distance(level: object, price: float, atr: float) -> LevelDistance:
    absolute = abs(level.price - price)
    return LevelDistance(
        level_id=level.level_id,
        price=level.price,
        distance_atr=absolute / atr,
        distance_bps=absolute / price * 10_000.0,
    )
