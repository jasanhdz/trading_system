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
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureRow:
    """A single row of 146 E4 features ready for model inference."""
    features: dict[str, float]
    symbol: str
    side: str
    timestamp: datetime
    feature_hash: str = ""

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to a single-row DataFrame for model input."""
        return pd.DataFrame([self.features])


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
            decision_at=datetime.now(timezone.utc),
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
        symbol: str = "",
        side: str = "",
        timestamp: datetime | None = None,
    ) -> FeatureRow:
        """Create a FeatureRow from a pre-computed feature dictionary.

        Validates:
            - All required features are present
            - All values are finite (no NaN, no inf)
        """
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
            timestamp=timestamp or datetime.now(timezone.utc),
            feature_hash=feature_hash,
        )

    def from_dataframe_row(
        self,
        row: pd.Series,
        symbol: str = "",
        side: str = "",
    ) -> FeatureRow:
        """Create a FeatureRow from a pandas Series (e.g., from parquet)."""
        features = {}
        for name in self._feature_names:
            if name in row.index:
                features[name] = float(row[name])
            else:
                raise ValueError(f"Feature '{name}' not found in row")

        return self.from_feature_dict(
            features,
            symbol=symbol or str(row.get("symbol", "")),
            side=side or str(row.get("side", "")),
            timestamp=row.get("decision_at") or row.get("signal_timestamp"),
        )

    def from_market_candles(
        self,
        candles_by_symbol: dict[str, pd.DataFrame],
        target_symbol: str,
        side: str,
        decision_at: datetime | None = None,
        timeframes: list[int] | None = None,
    ) -> FeatureRow:
        """Compute 146 E4 features from 1-minute candle data for ALL symbols.

        The E4 feature builder requires candles from ALL tracked symbols
        (not just the target) because add_cross_market() computes features
        from BTC/ETH cross-correlations and market-wide breadth/dispersion.

        Args:
            candles_by_symbol: {SYMBOL: one_minute_candle_DataFrame} for all symbols.
            target_symbol: The symbol to extract features for.
            side: "LONG" or "SHORT".
            decision_at: Timestamp of the decision (UTC).
            timeframes: Candle aggregation timeframes (default: [5, 15, 60, 240]).

        Returns:
            FeatureRow with the 146 features for the target symbol/side.

        Raises:
            ImportError: If E4 feature builder is not available.
            ValueError: If no features computed for the target.
        """
        from sandbox.aegis_strategy_router.experiments.aegis_e4_robust_training.src.aegis_e4.features import (
            build_neutral_symbol_panel,
            add_cross_market,
            orient_sides,
        )

        ts = decision_at or datetime.now(timezone.utc)
        tfs = timeframes or [5, 15, 60, 240]
        anchors = pd.DatetimeIndex([ts])

        panels = []
        all_families: dict[str, str] = {}
        for symbol, candles in candles_by_symbol.items():
            if candles is None or candles.empty:
                logger.warning("No candles for %s, skipping", symbol)
                continue
            panel, families = build_neutral_symbol_panel(candles, anchors, timeframes=tfs)
            panel["symbol"] = symbol
            panels.append(panel)
            all_families.update(families)

        if not panels:
            raise ValueError("No panels computed from any symbol")

        combined = pd.concat(panels, ignore_index=True)

        combined, cross_families = add_cross_market(combined)
        all_families.update(cross_families)

        oriented, oriented_families = orient_sides(combined, all_families)
        all_families.update(oriented_families)

        if side.upper() == "SHORT":
            matched = oriented[oriented["side"] == "SHORT"]
        else:
            matched = oriented[oriented["side"] == "LONG"]

        matched = matched[matched["symbol"] == target_symbol]
        if matched.empty:
            raise ValueError(
                f"No features computed for {target_symbol}/{side} "
                f"(available symbols: {oriented['symbol'].unique().tolist()})"
            )

        row = matched.iloc[0]
        features = {}
        for name in self._feature_names:
            if name in row.index:
                features[name] = float(row[name])

        return self.from_feature_dict(features, symbol=target_symbol, side=side, timestamp=ts)

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    @property
    def feature_count(self) -> int:
        return len(self._feature_names)
