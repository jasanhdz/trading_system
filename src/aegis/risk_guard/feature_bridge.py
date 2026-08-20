"""Feature Bridge — converts market data into E4's 146 features.

This module provides the missing link between Aegis signals and E4's
frozen model. It accepts either:
    1. Pre-computed feature rows (from development_labeled.parquet)
    2. Raw 1-minute candle DataFrames for ALL symbols (via E4 feature builder)

The bridge produces exactly the 146 features that E4's tail_risk model
expects, enabling real-time inference without pre-computed scores.

Architecture:
    candles_by_symbol = {BTCUSDT: df, ETHUSDT: df, ...}
            ↓
    build_neutral_symbol_panel() per symbol
            ↓
    CONCATENATE ALL symbols
            ↓
    add_cross_market()  ← requires multi-symbol panel
            ↓
    orient_sides()
            ↓
    146 feature__* columns
            ↓
    E4 Tail Risk Guard (frozen model)
            ↓
    Tail Risk Score → ALLOW / BLOCK
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Frozen E4 universe — exactly 11 symbols. If any is missing, fail closed.
FROZEN_E4_UNIVERSE: frozenset[str] = frozenset({
    "ADAUSDT", "AVAXUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT",
    "ETHUSDT", "LINKUSDT", "LTCUSDT", "SOLUSDT", "SUIUSDT", "XRPUSDT",
})

# Frozen E4 timeframes — not configurable.
FROZEN_E4_TIMEFROZEN: list[int] = [5, 15, 60, 240]

# Required candle columns (matches dataset.py load_candles).
REQUIRED_CANDLE_COLUMNS: frozenset[str] = frozenset({
    "open_time_ms", "open", "high", "low", "close", "volume", "taker_buy_volume",
})

# Anchor window: how many historical anchors before decision_at.
# diff(3) in cross__breadth_acceleration needs at least 4 anchors.
# diff(2) in cross__btc_impulse_acceleration needs at least 3.
# We use 5 anchors (20 min at 5-min cadence) for safety margin.
ANCHOR_COUNT = 5
ANCHOR_CADENCE_MINUTES = 5


@dataclass(frozen=True)
class FeatureRow:
    """A single row of 146 E4 features ready for model inference."""
    features: dict[str, float]
    symbol: str
    side: str
    timestamp: datetime
    feature_hash: str = ""
    max_available_at: datetime | None = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to a single-row DataFrame for model input."""
        return pd.DataFrame([self.features])


def validate_candles(candles: pd.DataFrame, symbol: str) -> None:
    """Validate 1-minute candle DataFrame before feature construction.

    Enforces the same invariants as the original E4 dataset builder:
        - Required columns present
        - No duplicate minutes
        - Exact 60-second gaps between candles

    Raises ValueError on any violation.
    """
    missing = REQUIRED_CANDLE_COLUMNS - set(candles.columns)
    if missing:
        raise ValueError(f"CANDLE_COLUMNS_MISSING:{symbol}:{sorted(missing)}")

    if candles.open_time_ms.duplicated().any():
        raise ValueError(f"CANDLE_DUPLICATE_MINUTE:{symbol}")

    gaps = np.diff(candles.open_time_ms.to_numpy(np.int64))
    if len(gaps) and not np.all(gaps == 60_000):
        raise ValueError(f"CANDLE_MINUTE_GAP:{symbol}")


def build_anchors(decision_at: datetime) -> pd.DatetimeIndex:
    """Build a window of historical anchors ending at decision_at.

    E4 features like cross__breadth_acceleration use diff(3) and
    cross__btc_impulse_acceleration use diff(2). With a single anchor
    these would always be 0. We provide ANCHOR_COUNT anchors spaced
    at ANCHOR_CADENCE_MINUTES intervals so the diff operations
    produce the same values as during training.
    """
    decision_ts = pd.Timestamp(decision_at)
    if decision_ts.tzinfo is None:
        decision_ts = decision_ts.tz_localize("UTC")
    else:
        decision_ts = decision_ts.tz_convert("UTC")
    start = decision_ts - timedelta(minutes=ANCHOR_COUNT * ANCHOR_CADENCE_MINUTES)
    anchors = pd.date_range(
        start, decision_ts, freq=f"{ANCHOR_CADENCE_MINUTES}min", inclusive="both", tz="UTC"
    )
    return anchors


class FeatureBridge:
    """Converts market data into E4's 146 features.

    Two modes:
        1. From pre-computed features (development_labeled.parquet)
        2. From raw 1-minute candles for ALL symbols (via E4 feature builder)

    Usage:
        bridge = FeatureBridge(feature_names=guard._tail_bundle["features"])

        # Mode 1: from pre-computed features
        row = bridge.from_feature_dict({"feature__base__tf5m__...": 0.5, ...})

        # Mode 2: from candle DataFrames for all symbols
        row = bridge.from_market_candles(
            candles_by_symbol={"BTCUSDT": btc_df, "ETHUSDT": eth_df, ...},
            target_symbol="BTCUSDT",
            side="SHORT",
            decision_at=datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc),
        )
    """

    def __init__(self, feature_names: list[str]) -> None:
        """Initialize with the 146 feature names.

        feature_names MUST be provided — typically from the guard's verified bundle:
            bridge = FeatureBridge(guard._tail_bundle["features"])
        """
        if not feature_names:
            raise ValueError("feature_names must be a non-empty list")
        self._feature_names = list(feature_names)

    def from_feature_dict(
        self,
        features: dict[str, Any],
        symbol: str,
        side: str,
        timestamp: datetime,
    ) -> FeatureRow:
        """Create a FeatureRow from a pre-computed feature dictionary.

        Validates:
            - All required features are present
            - All values are finite (no NaN, no inf)
            - symbol and side are non-empty
        """
        if not symbol:
            raise ValueError("symbol is required")
        if side not in ("LONG", "SHORT"):
            raise ValueError(f"side must be 'LONG' or 'SHORT', got '{side}'")

        missing = [f for f in self._feature_names if f not in features]
        if missing:
            raise ValueError(
                f"Missing {len(missing)} features: {missing[:5]}..."
            )

        non_finite = []
        feature_dict = {}
        for name in self._feature_names:
            val = features[name]
            try:
                fval = float(val)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Feature '{name}' cannot convert to float: {val}") from e
            if not np.isfinite(fval):
                non_finite.append(name)
            feature_dict[name] = fval

        if non_finite:
            raise ValueError(
                f"Non-finite values in {len(non_finite)} features: {non_finite[:5]}..."
            )

        raw = str(sorted(feature_dict.items())).encode()
        feature_hash = hashlib.sha256(raw).hexdigest()

        return FeatureRow(
            features=feature_dict,
            symbol=symbol,
            side=side,
            timestamp=timestamp,
            feature_hash=feature_hash,
        )

    def from_dataframe_row(
        self,
        row: pd.Series,
        symbol: str,
        side: str,
    ) -> FeatureRow:
        """Create a FeatureRow from a pandas Series (e.g., from parquet)."""
        features = {}
        for name in self._feature_names:
            if name in row.index:
                features[name] = float(row[name])
            else:
                raise ValueError(f"Feature '{name}' not found in row")

        ts = row.get("decision_at") or row.get("signal_timestamp")
        if ts is None:
            raise ValueError("No timestamp found in row")

        return self.from_feature_dict(features, symbol=symbol, side=side, timestamp=ts)

    def from_market_candles(
        self,
        candles_by_symbol: dict[str, pd.DataFrame],
        target_symbol: str,
        side: str,
        decision_at: datetime,
    ) -> FeatureRow:
        """Compute 146 E4 features from 1-minute candle data for ALL symbols.

        The E4 feature builder requires candles from exactly 11 tracked symbols
        (no more, no less) because add_cross_market() computes features from
        BTC/ETH cross-correlations and market-wide breadth/dispersion.

        Validates:
            - Exactly the 11 E4 symbols (fail closed if missing OR extra)
            - Each candle DataFrame passes validate_candles()
            - side is 'LONG' or 'SHORT'
            - target_symbol is in the universe
            - decision_at is aligned to 5-minute cadence

        Uses a window of historical anchors (not just decision_at) so that
        diff() operations in cross-market features produce the same values
        as during training.
        """
        from sandbox.aegis_strategy_router.experiments.aegis_e4_robust_training.src.aegis_e4.features import (
            build_neutral_symbol_panel,
            add_cross_market,
            orient_sides,
            assert_causal_availability,
        )

        if side not in ("LONG", "SHORT"):
            raise ValueError(f"side must be 'LONG' or 'SHORT', got '{side}'")
        if target_symbol not in FROZEN_E4_UNIVERSE:
            raise ValueError(
                f"target_symbol '{target_symbol}' not in E4 universe. "
                f"Must be one of: {sorted(FROZEN_E4_UNIVERSE)}"
            )

        provided = set(candles_by_symbol.keys())
        if provided != FROZEN_E4_UNIVERSE:
            missing = FROZEN_E4_UNIVERSE - provided
            extra = provided - FROZEN_E4_UNIVERSE
            parts = []
            if missing:
                parts.append(f"MISSING: {sorted(missing)}")
            if extra:
                parts.append(f"EXTRA: {sorted(extra)}")
            raise ValueError(
                f"UNIVERSE_MISMATCH: must be exactly {len(FROZEN_E4_UNIVERSE)} "
                f"E4 symbols. {'; '.join(parts)}"
            )

        decision_ts = pd.Timestamp(decision_at)
        if decision_ts.tzinfo is None:
            decision_ts = decision_ts.tz_localize("UTC")
        else:
            decision_ts = decision_ts.tz_convert("UTC")

        if decision_ts.minute % ANCHOR_CADENCE_MINUTES != 0:
            raise ValueError(
                f"decision_at must be aligned to {ANCHOR_CADENCE_MINUTES}-minute "
                f"cadence, got minute={decision_ts.minute}"
            )

        normalized_decision_at = decision_ts
        anchors = build_anchors(decision_at)

        panels = []
        all_families: dict[str, str] = {}
        for symbol, candles in candles_by_symbol.items():
            validate_candles(candles, symbol)
            panel, families = build_neutral_symbol_panel(
                candles, anchors, timeframes=FROZEN_E4_TIMEFROZEN
            )
            panel["symbol"] = symbol
            panels.append(panel)
            all_families.update(families)

        combined = pd.concat(panels, ignore_index=True)
        combined, cross_families = add_cross_market(combined)
        all_families.update(cross_families)

        oriented, oriented_families = orient_sides(combined, all_families)
        all_families.update(oriented_families)

        matched = oriented[
            (oriented["symbol"] == target_symbol)
            & (oriented["side"] == side)
            & (oriented["decision_at"] == normalized_decision_at)
        ]
        if matched.empty:
            raise ValueError(
                f"No features computed for {target_symbol}/{side} "
                f"at decision_at={normalized_decision_at}"
            )
        if len(matched) != 1:
            raise ValueError(
                f"Expected exactly 1 row for {target_symbol}/{side} "
                f"at decision_at={normalized_decision_at}, got {len(matched)}"
            )

        row = matched.iloc[0]

        avail_cols = [c for c in row.index if c.startswith("available_at__")]
        max_avail = None
        causal_errors = []
        for col in avail_cols:
            avail_val = row[col]
            if pd.isna(avail_val):
                continue
            avail_ts = pd.Timestamp(avail_val)
            if avail_ts.tzinfo is None:
                avail_ts = avail_ts.tz_localize("UTC")
            else:
                avail_ts = avail_ts.tz_convert("UTC")
            if avail_ts > normalized_decision_at:
                causal_errors.append(f"{col}={avail_ts} > decision_at={normalized_decision_at}")
            if max_avail is None or avail_ts > max_avail:
                max_avail = avail_ts

        if causal_errors:
            raise ValueError(
                f"NON_CAUSAL_DATA: {'; '.join(causal_errors[:3])}"
            )

        features = {
            name: float(row[name])
            for name in self._feature_names
            if name in row.index
        }

        feature_row = self.from_feature_dict(features, symbol=target_symbol, side=side, timestamp=decision_at)

        if max_avail is not None:
            max_avail_dt = max_avail.to_pydatetime()
            return FeatureRow(
                features=feature_row.features,
                symbol=feature_row.symbol,
                side=feature_row.side,
                timestamp=feature_row.timestamp,
                feature_hash=feature_row.feature_hash,
                max_available_at=max_avail_dt,
            )

        return feature_row

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    @property
    def feature_count(self) -> int:
        return len(self._feature_names)
