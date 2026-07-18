"""Versioned path-aware SHORT labels over finalized future candles."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timedelta
from typing import Sequence

from ..domain import Candle


SHORT_LABEL_SCHEMA_VERSION = "aegis-labels-short-v4"


@dataclass(frozen=True)
class ShortLabelConfig:
    horizon_bars: int = 12
    historical_reference_leverage: float = 20.0
    fee_bps_per_side: float = 4.0
    slippage_bps_per_side: float = 1.0
    funding_bps_per_hour: float = 0.0
    clean_mfe_fraction: float = 0.08 / 20.0
    clean_mae_fraction: float = 0.055 / 20.0
    clean_mfe_mae_ratio: float = 1.25
    clean_time_fraction: float = 0.60
    initial_adverse_3_fraction: float = 0.045 / 20.0
    bad_mae_fraction: float = 0.06 / 20.0
    bad_low_mfe_fraction: float = 0.04 / 20.0
    bad_initial_adverse_3_fraction: float = 0.05 / 20.0
    tail_mae_fraction: float = 0.06 / 20.0
    entry_rule: str = "SIGNAL_CLOSE"

    def __post_init__(self) -> None:
        if self.entry_rule not in {"SIGNAL_CLOSE", "NEXT_BAR_OPEN"}:
            raise ValueError("unsupported SHORT label entry rule")

    @property
    def round_trip_cost_fraction(self) -> float:
        trading = 2.0 * (self.fee_bps_per_side + self.slippage_bps_per_side) / 10_000.0
        funding = self.funding_bps_per_hour / 10_000.0 * (self.horizon_bars * 5.0 / 60.0)
        return trading + funding


@dataclass(frozen=True)
class ShortPathLabel:
    schema_version: str
    valid: bool
    quarantine_reason: str | None
    entry_convention: str
    horizon_bars: int
    entry_price: float | None = None
    mfe_fraction: float | None = None
    mae_fraction: float | None = None
    net_quality_after_costs: float | None = None
    mfe_mae_ratio: float | None = None
    time_to_mfe: int | None = None
    time_to_mae: int | None = None
    mfe_before_mae: bool = False
    clean_entry: bool = False
    bad_entry: bool = False
    tail_event: bool = False
    hit_before_stop: bool = False
    stopped_before_hit: bool = False
    ambiguous_hit_stop: bool = False
    terminal_short_return: float | None = None
    round_trip_cost_fraction: float | None = None


def _quarantine(config: ShortLabelConfig, reason: str) -> ShortPathLabel:
    return ShortPathLabel(SHORT_LABEL_SCHEMA_VERSION, False, reason, config.entry_rule, config.horizon_bars)


def _hit_before_stop(
    entry: float, future: Sequence[Candle], *, target_fraction: float, stop_fraction: float,
) -> tuple[bool, bool, bool]:
    target = entry * (1.0 - target_fraction)
    stop = entry * (1.0 + stop_fraction)
    for candle in future:
        hit = candle.low <= target
        stopped = candle.high >= stop
        if hit and stopped:
            return False, True, True
        if stopped:
            return False, True, False
        if hit:
            return True, False, False
    return False, False, False


def build_short_path_label(
    signal: Candle, future: Sequence[Candle], config: ShortLabelConfig | None = None,
) -> ShortPathLabel:
    """Label one SHORT hypothesis using only closed bars t+1 through t+H."""
    settings = config or ShortLabelConfig()
    if len(future) != settings.horizon_bars:
        return _quarantine(settings, "INCOMPLETE_HORIZON")
    interval = timedelta(minutes=5)
    previous_close = signal.close_time
    for candle in future:
        if not candle.is_closed:
            return _quarantine(settings, "NON_FINAL_CANDLE")
        if candle.open_time != previous_close or candle.close_time - candle.open_time != interval:
            return _quarantine(settings, "FUTURE_GAP")
        previous_close = candle.close_time
    entry = future[0].open if settings.entry_rule == "NEXT_BAR_OPEN" else signal.close
    favorable = [max(0.0, (entry - candle.low) / entry) for candle in future]
    adverse = [max(0.0, (candle.high - entry) / entry) for candle in future]
    mfe = max(favorable); mae = max(adverse)
    mfe_index = favorable.index(mfe) + 1 if mfe > 0.0 else None
    mae_index = adverse.index(mae) + 1 if mae > 0.0 else None
    mfe_before_mae = mfe_index is not None and (mae_index is None or mfe_index < mae_index)
    ratio = mfe / mae if mae > 0.0 else math.inf if mfe > 0.0 else 0.0
    initial_adverse = max(adverse[: min(3, len(adverse))], default=0.0)
    net_quality = mfe - mae - settings.round_trip_cost_fraction
    hit, stopped, ambiguous = _hit_before_stop(
        entry, future, target_fraction=settings.clean_mfe_fraction,
        stop_fraction=settings.bad_mae_fraction,
    )
    clean = (
        not ambiguous and mfe_before_mae and mfe >= settings.clean_mfe_fraction
        and mae <= settings.clean_mae_fraction and ratio >= settings.clean_mfe_mae_ratio
        and mfe_index is not None and mfe_index <= settings.horizon_bars * settings.clean_time_fraction
        and net_quality > 0.0 and initial_adverse <= settings.initial_adverse_3_fraction
    )
    mae_before_mfe = mae_index is not None and (mfe_index is None or mae_index <= mfe_index)
    bad = (
        (mae_before_mfe and mae >= settings.bad_mae_fraction)
        or (mfe < settings.bad_low_mfe_fraction and mae >= 0.04 / settings.historical_reference_leverage)
        or initial_adverse >= settings.bad_initial_adverse_3_fraction
        or net_quality < 0.0
        or (ratio < 1.0 and mae >= 0.04 / settings.historical_reference_leverage)
        or ambiguous
    )
    return ShortPathLabel(
        schema_version=SHORT_LABEL_SCHEMA_VERSION, valid=True, quarantine_reason=None,
        entry_convention=settings.entry_rule, horizon_bars=settings.horizon_bars, entry_price=entry,
        mfe_fraction=mfe, mae_fraction=mae, net_quality_after_costs=net_quality,
        mfe_mae_ratio=ratio, time_to_mfe=mfe_index, time_to_mae=mae_index,
        mfe_before_mae=mfe_before_mae, clean_entry=clean, bad_entry=bad,
        tail_event=mae >= settings.tail_mae_fraction, hit_before_stop=hit,
        stopped_before_hit=stopped, ambiguous_hit_stop=ambiguous,
        terminal_short_return=(entry - future[-1].close) / entry,
        round_trip_cost_fraction=settings.round_trip_cost_fraction,
    )
