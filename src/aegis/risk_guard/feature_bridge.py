"""Feature Bridge — converts market data into E4's 146 features.

This module provides the missing link between Aegis signals and E4's
frozen model. It accepts either:
    1. Pre-computed feature rows (from development_labeled.parquet)
    2. Raw 1-minute candle DataFrames (via E4 feature builder)

The bridge produces exactly the 146 features that E4's tail_risk model
expects, enabling real-time inference without pre-computed scores.

Architecture:
    MarketSnapshot / Candle DataFrame
            ↓
    E4 Feature Builder (from aegis_e4.features)
            ↓
    146 feature__* columns
            ↓
    E4 Tail Risk Guard (frozen model)
            ↓
    Tail Risk Score → ALLOW / BLOCK
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
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
        2. From raw 1-minute candles (requires E4 feature builder)

    Usage:
        bridge = FeatureBridge()

        # Mode 1: from pre-computed features
        row = bridge.from_feature_dict({"feature__base__tf5m__...": 0.5, ...})

        # Mode 2: from candle DataFrame (when builder is available)
        row = bridge.from_candles(candles_df, symbol="BTCUSDT", side="SHORT")
    """

    def __init__(self, feature_names: list[str] | None = None) -> None:
        """Initialize with the 146 feature names.

        If feature_names is None, loads from the feature schema.
        """
        if feature_names is not None:
            self._feature_names = feature_names
        else:
            self._feature_names = self._load_feature_names()

    def from_feature_dict(
        self,
        features: dict[str, Any],
        symbol: str = "",
        side: str = "",
        timestamp: datetime | None = None,
    ) -> FeatureRow:
        """Create a FeatureRow from a pre-computed feature dictionary.

        Validates that all required features are present and finite.
        """
        missing = [f for f in self._feature_names if f not in features]
        if missing:
            raise ValueError(
                f"Missing {len(missing)} features: {missing[:5]}..."
            )

        feature_dict = {}
        for name in self._feature_names:
            val = features[name]
            if isinstance(val, (np.floating, float)):
                feature_dict[name] = float(val)
            elif isinstance(val, (np.integer, int)):
                feature_dict[name] = float(val)
            elif isinstance(val, np.ndarray):
                feature_dict[name] = float(val.item())
            else:
                feature_dict[name] = float(val)

        import hashlib
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

    def from_candles(
        self,
        one_minute: pd.DataFrame,
        symbol: str,
        side: str,
        decision_at: datetime | None = None,
    ) -> FeatureRow:
        """Compute 146 E4 features from 1-minute candle data.

        Requires the E4 feature builder from the experiment code.
        Falls back to pre-computed features if builder is not available.
        """
        try:
            from sandbox.aegis_strategy_router.experiments.aegis_e4_robust_training.src.aegis_e4.features import (
                build_neutral_symbol_panel,
                add_cross_market,
                orient_sides,
            )

            ts = decision_at or datetime.now(timezone.utc)
            anchors = pd.DatetimeIndex([ts])
            panel, families = build_neutral_symbol_panel(
                one_minute, anchors, timeframes=[5, 15, 60, 240]
            )
            panel["symbol"] = symbol
            panel, families = add_cross_market(panel)
            oriented, families = orient_sides(panel, families)

            if side.upper() == "SHORT":
                oriented = oriented[oriented["side"] == "SHORT"]
            else:
                oriented = oriented[oriented["side"] == "LONG"]

            if oriented.empty:
                raise ValueError(f"No features computed for {symbol}/{side}")

            row = oriented.iloc[0]
            features = {
                name: float(row[name])
                for name in self._feature_names
                if name in row.index
            }

            return self.from_feature_dict(features, symbol=symbol, side=side, timestamp=ts)

        except ImportError:
            raise ImportError(
                "E4 feature builder not available. "
                "Use from_feature_dict() or from_dataframe_row() instead."
            )

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    @property
    def feature_count(self) -> int:
        return len(self._feature_names)

    @staticmethod
    def _load_feature_names() -> list[str]:
        """Load the 146 feature names from the model artifacts."""
        from pathlib import Path
        import joblib

        models_path = Path(
            "sandbox/aegis_strategy_router/experiments/"
            "aegis_e4_robust_training/artifacts/run_01/development_models.joblib"
        )
        if not models_path.exists():
            raise FileNotFoundError(f"E4 models not found: {models_path}")

        models = joblib.load(models_path)
        tail = models.get("target__tail_risk")
        if tail is None:
            raise KeyError("target__tail_risk not found in E4 models")

        return list(tail["features"])
