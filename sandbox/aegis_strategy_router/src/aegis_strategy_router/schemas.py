"""Versioned feature schema for the Phase 1 snapshot contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from aegis_strategy_router.domain.serialization import content_hash


SNAPSHOT_SCHEMA_VERSION = "aegis-strategy-router-snapshot-v2-latest-candle"
FEATURE_SCHEMA_VERSION = "aegis-strategy-router-features-v1"
FEATURE_OWNER = "aegis.research.live_entry_multitimeframe.indicator_frame"

EXISTING_FEATURE_NAMES = (
    "return_1_bps",
    "return_3_bps",
    "return_6_bps",
    "atr_pct_bps",
    "atr_percentile_96",
    "rsi6",
    "rsi12",
    "rsi24",
    "ema7_extension_atr",
    "ema25_extension_atr",
    "ema99_extension_atr",
    "ema7_slope_atr",
    "ema25_slope_atr",
    "trend_age",
    "prior_move_6_atr",
    "volume_ratio20",
    "volume_z50",
    "body_ratio",
    "clv",
    "taker_imbalance",
    "distance_recent_high_atr",
    "distance_recent_low_atr",
    "range_48_atr",
    "path_efficiency_6",
    "breakout_up",
    "breakout_down",
)


@dataclass(frozen=True, slots=True)
class FeatureDefinition:
    name: str
    dtype: str
    owner: str
    availability: str

    def to_primitive(self) -> dict[str, str]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "owner": self.owner,
            "availability": self.availability,
        }


@dataclass(frozen=True, slots=True)
class FeatureSchema:
    version: str
    definitions: tuple[FeatureDefinition, ...]

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.definitions, key=lambda item: item.name))
        if len({item.name for item in ordered}) != len(ordered):
            raise ValueError("feature schema names must be unique")
        object.__setattr__(self, "definitions", ordered)

    @classmethod
    def existing_multitimeframe(cls, timeframes: Iterable[str]) -> "FeatureSchema":
        definitions = []
        for timeframe in sorted(set(timeframes)):
            definitions.extend(
                FeatureDefinition(
                    name=f"tf{timeframe}__{name}",
                    dtype="float64",
                    owner=FEATURE_OWNER,
                    availability="fully_closed_candle.close_at",
                )
                for name in EXISTING_FEATURE_NAMES
            )
        return cls(FEATURE_SCHEMA_VERSION, tuple(definitions))

    @property
    def hash(self) -> str:
        return content_hash(self.to_primitive())

    def to_primitive(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "definitions": [item.to_primitive() for item in self.definitions],
        }
