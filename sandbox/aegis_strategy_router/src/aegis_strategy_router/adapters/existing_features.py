"""Fail-closed adapter around the existing causal candle feature implementation."""

from __future__ import annotations

import hashlib
import inspect
import math
from datetime import datetime
from typing import Any

import pandas as pd

from aegis.research.live_entry_multitimeframe import aggregate_klines, indicator_frame
from aegis_strategy_router.domain.serialization import utc_datetime
from aegis_strategy_router.domain.types import (
    Candle,
    DataStatus,
    FeatureObservation,
    FeatureSet,
    Timeframe,
    TimeframeSnapshot,
)
from aegis_strategy_router.schemas import EXISTING_FEATURE_NAMES, FEATURE_OWNER, FeatureSchema


SOURCE_API_VERSION = "live-entry-multitimeframe-v1"
REQUIRED_WARMUP_BARS = 99
REQUIRED_COLUMNS = frozenset(
    {"open_time_ms", "open", "high", "low", "close", "volume", "taker_buy_volume"}
)
FORBIDDEN_COLUMN_TOKENS = (
    "future_", "mfe", "mae", "outcome", "target", "label", "realized_pnl",
    "actual_exit", "final_exit",
)


class SourceDataError(ValueError):
    """Source candles cannot satisfy the causal adapter contract."""


def source_code_hash() -> str:
    source = inspect.getsource(indicator_frame) + inspect.getsource(aggregate_klines)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


class ExistingResearchFeatureAdapter:
    """Converts one-minute public market history into immutable feature states."""

    def __init__(self, schema: FeatureSchema) -> None:
        self.schema = schema
        self.source_version = f"{SOURCE_API_VERSION}:{source_code_hash()}"

    def _closed_source(self, one_minute: pd.DataFrame, decision_at: datetime) -> pd.DataFrame:
        forbidden = sorted(
            str(column) for column in one_minute.columns
            if any(token in str(column).lower() for token in FORBIDDEN_COLUMN_TOKENS)
        )
        if forbidden:
            raise SourceDataError(f"forbidden outcome/future columns: {forbidden}")
        missing = REQUIRED_COLUMNS.difference(one_minute.columns)
        if missing:
            raise SourceDataError(f"missing one-minute columns: {sorted(missing)}")
        frame = one_minute.loc[:, sorted(REQUIRED_COLUMNS)].copy()
        numeric = ["open_time_ms", "open", "high", "low", "close", "volume", "taker_buy_volume"]
        for column in numeric:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[numeric].isna().any().any():
            raise SourceDataError("one-minute source contains non-numeric/null values")
        if frame["open_time_ms"].duplicated().any():
            raise SourceDataError("one-minute source contains duplicate open timestamps")
        frame = frame.sort_values("open_time_ms", kind="mergesort").reset_index(drop=True)
        gaps = frame["open_time_ms"].diff().dropna()
        if not gaps.eq(60_000).all():
            raise SourceDataError("one-minute source contains timestamp gaps")
        close_times = pd.to_datetime(frame["open_time_ms"], unit="ms", utc=True) + pd.Timedelta(minutes=1)
        boundary = pd.Timestamp(utc_datetime(decision_at))
        return frame.loc[close_times.le(boundary)].reset_index(drop=True)

    def aggregate_closed(
        self, one_minute: pd.DataFrame, timeframe: Timeframe, decision_at: datetime
    ) -> tuple[Candle, ...]:
        source = self._closed_source(one_minute, decision_at)
        if source.empty:
            return ()
        aggregated = aggregate_klines(source, timeframe.minutes)
        boundary = pd.Timestamp(utc_datetime(decision_at))
        aggregated = aggregated.loc[pd.to_datetime(aggregated["close_time"], utc=True).le(boundary)]
        candles = []
        for row in aggregated.itertuples(index=False):
            close_at = pd.Timestamp(row.close_time).to_pydatetime()
            open_at = pd.Timestamp(row.open_time).to_pydatetime()
            candles.append(Candle(
                open_at=open_at,
                close_at=close_at,
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
                taker_buy_volume=float(row.taker_buy_volume),
                available_at=close_at,
                source_id=f"public-1m:{int(pd.Timestamp(row.open_time).timestamp() * 1_000)}:{timeframe.value}",
                complete=bool(row.bar_count == timeframe.minutes),
            ))
        return tuple(candles)

    def build_timeframe(self, one_minute: pd.DataFrame, timeframe: Timeframe, decision_at: datetime) -> TimeframeSnapshot:
        boundary = utc_datetime(decision_at)
        feature_names = tuple(f"tf{timeframe.value}__{name}" for name in EXISTING_FEATURE_NAMES)
        try:
            source = self._closed_source(one_minute, boundary)
            candles = self.aggregate_closed(source, timeframe, boundary)
        except (SourceDataError, ValueError) as error:
            observations = tuple(
                FeatureObservation(name, None, None, None, FEATURE_OWNER, self.source_version,
                                   DataStatus.INVALID, f"SOURCE_INVALID:{error}")
                for name in feature_names
            )
            return TimeframeSnapshot(
                timeframe, DataStatus.INVALID, 0, REQUIRED_WARMUP_BARS, None, None,
                FeatureSet(observations, self.schema.hash), None, f"SOURCE_INVALID:{error}",
            )

        if not candles:
            observations = tuple(
                FeatureObservation(name, None, None, None, FEATURE_OWNER, self.source_version,
                                   DataStatus.UNKNOWN, "NO_FULLY_CLOSED_BAR")
                for name in feature_names
            )
            return TimeframeSnapshot(
                timeframe, DataStatus.UNKNOWN, 0, REQUIRED_WARMUP_BARS, None, None,
                FeatureSet(observations, self.schema.hash), None, "NO_FULLY_CLOSED_BAR",
            )

        indicators = indicator_frame(source, timeframe.minutes)
        indicators = indicators.loc[pd.to_datetime(indicators["close_time"], utc=True).le(pd.Timestamp(boundary))]
        last = indicators.iloc[-1]
        close_at = pd.Timestamp(last["close_time"]).to_pydatetime()
        prefix = f"tf{timeframe.minutes}m__"
        observations = []
        unavailable = 0
        for base_name, full_name in zip(EXISTING_FEATURE_NAMES, feature_names):
            raw = last.get(prefix + base_name)
            if raw is None or not math.isfinite(float(raw)):
                unavailable += 1
                observations.append(FeatureObservation(
                    full_name, None, close_at, None, FEATURE_OWNER, self.source_version,
                    DataStatus.UNKNOWN, "FEATURE_WARMUP_INCOMPLETE",
                ))
            else:
                observations.append(FeatureObservation(
                    full_name, float(raw), close_at, close_at, FEATURE_OWNER,
                    self.source_version, DataStatus.AVAILABLE,
                ))
        status = DataStatus.AVAILABLE if len(candles) >= REQUIRED_WARMUP_BARS and unavailable == 0 else DataStatus.UNKNOWN
        reason = None if status is DataStatus.AVAILABLE else "FEATURE_WARMUP_INCOMPLETE"
        return TimeframeSnapshot(
            timeframe=timeframe,
            status=status,
            candle_count=len(candles),
            required_warmup_bars=REQUIRED_WARMUP_BARS,
            latest_closed_at=candles[-1].close_at,
            latest_candle=candles[-1],
            features=FeatureSet(tuple(observations), self.schema.hash),
            structural=None,
            reason=reason,
        )
