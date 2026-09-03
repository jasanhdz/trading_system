"""Deterministic replay engine for Risk Guard validation."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .domain import EntryDecision, RiskGuardConfig, Signal, Direction
from .direction_provider import DirectionProvider
from .entry_decision import EntryDecisionOrchestrator
from .e4_tail_risk_guard import E4TailRiskGuard
from .flags import RiskGuardFlags, RiskGuardMode
from .observability import RiskGuardMetrics, RiskGuardObserver

logger = logging.getLogger(__name__)


class _ReplayDirectionProvider(DirectionProvider):
    """Direction provider that extracts side from signal data for replay."""

    def __init__(self) -> None:
        self._last_signal: Signal | None = None

    def set_signal(self, signal: Signal) -> None:
        self._last_signal = signal

    def evaluate(self, symbol: str, context: dict[str, Any] | None = None) -> Signal:
        if self._last_signal is not None:
            return self._last_signal
        return Signal(
            signal_id=f"REPLAY-{symbol}",
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            side=Direction.SKIP,
            direction_source="REPLAY",
            direction_model_version="REPLAY_V1",
        )

    def name(self) -> str:
        return "REPLAY"

    def version(self) -> str:
        return "replay_v1"


class DeterministicReplay:
    """Replay historical signals through the risk guard architecture.

    Produces a counterfactual table:
    - What Aegis proposed
    - What E4 would have scored
    - What decision would have been made
    - What actually happened (if outcome data available)

    This enables offline validation without any production impact.
    """

    def __init__(self, config: RiskGuardConfig) -> None:
        self._config = config
        self._guard: E4TailRiskGuard | None = None
        self._orchestrator: EntryDecisionOrchestrator | None = None
        self._observer: RiskGuardObserver | None = None
        self._metrics: RiskGuardMetrics | None = None
        self._provider = _ReplayDirectionProvider()

    def initialize(self) -> None:
        """Initialize the replay engine with frozen artifacts."""
        self._guard = E4TailRiskGuard(self._config)
        self._guard.load()

        self._orchestrator = EntryDecisionOrchestrator(
            direction_provider=self._provider,
            risk_guard=self._guard,
            config=self._config,
        )
        self._observer = RiskGuardObserver()
        self._metrics = RiskGuardMetrics()

    def replay_signals(
        self,
        signals_df: pd.DataFrame,
        feature_rows: pd.DataFrame | None = None,
        pre_computed_scores: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Replay a DataFrame of historical signals through the risk guard.

        Args:
            signals_df: DataFrame with columns:
                signal_id, timestamp, symbol, side, direction_source, ...
            feature_rows: Optional DataFrame with pre-computed E4 features,
                indexed by (timestamp, symbol, side).
            pre_computed_scores: Optional Series of pre-computed tail risk scores.
                If provided, skips model inference and uses these scores directly.

        Returns:
            DataFrame with one row per signal, containing:
                signal_id, symbol, side, tail_risk_score, risk_decision,
                verdict, enforced, reason, feature_snapshot_hash
        """
        if self._orchestrator is None:
            raise RuntimeError("Replay engine not initialized. Call initialize() first.")

        results = []
        for i, (_, row) in enumerate(signals_df.iterrows()):
            signal = self._build_signal(row)
            self._provider.set_signal(signal)

            context = self._build_context(signal, row, feature_rows)

            if pre_computed_scores is not None and i < len(pre_computed_scores):
                context["_replay_pre_computed_score"] = float(pre_computed_scores.iloc[i])

            decision = self._orchestrator.evaluate(signal.symbol, context)

            self._observer.record(decision)
            self._metrics.record(decision)

            results.append(decimal_to_json_friendly(decision.to_dict()))

        return pd.DataFrame(results)

    def metrics_summary(self) -> dict[str, Any]:
        """Return metrics from the replay."""
        if self._metrics is None:
            return {}
        return self._metrics.summary()

    def _build_signal(self, row: pd.Series) -> Signal:
        """Build a Signal from a DataFrame row."""
        side_str = str(row.get("side", "SKIP")).upper()
        if side_str == "LONG":
            side = Direction.LONG
        elif side_str == "SHORT":
            side = Direction.SHORT
        else:
            side = Direction.SKIP

        ts = row.get("signal_timestamp") or row.get("timestamp")
        if isinstance(ts, str):
            from datetime import datetime, timezone
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))

        return Signal(
            signal_id=str(row.get("signal_id", "")),
            timestamp=ts,
            symbol=str(row.get("symbol", "")),
            side=side,
            direction_source=str(row.get("direction_source", "HISTORICAL")),
            direction_model_version=str(row.get("direction_model_version", "UNKNOWN")),
            turbo_score=float(row.get("turbo_score", 0.0)),
        )

    def _build_context(
        self,
        signal: Signal,
        row: pd.Series,
        feature_rows: pd.DataFrame | None,
    ) -> dict[str, Any]:
        """Build context dict for risk guard evaluation."""
        context: dict[str, Any] = {}

        if feature_rows is not None and len(feature_rows) > 0:
            key = (signal.timestamp, signal.symbol, signal.side.value)
            if key in feature_rows.index:
                feature_row = feature_rows.loc[key]
                if isinstance(feature_row, pd.DataFrame):
                    feature_row = feature_row.iloc[0]
                context["feature_row"] = feature_row

        feature_dict = {}
        for col in row.index:
            if col.startswith("feature__"):
                feature_dict[col] = row[col]
        if feature_dict:
            context["features"] = feature_dict

        return context


def decimal_to_json_friendly(obj: Any) -> Any:
    """Convert numpy/pandas types to JSON-friendly types."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: decimal_to_json_friendly(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decimal_to_json_friendly(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and (obj != obj):
        return None
    return obj
