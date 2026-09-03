"""Build byte-replayable causal snapshots without any trading capability."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Mapping

import pandas as pd

from aegis_strategy_router.adapters.existing_features import ExistingResearchFeatureAdapter
from aegis_strategy_router.domain.serialization import utc_datetime
from aegis_strategy_router.domain.types import DataStatus, MarketSnapshot, Side, StructuralContext, Timeframe
from aegis_strategy_router.features.structural_levels import StructuralLevelAdapter
from aegis_strategy_router.schemas import FeatureSchema, SNAPSHOT_SCHEMA_VERSION


ALL_TIMEFRAMES = (Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1)


@dataclass(frozen=True, slots=True)
class ReplayManifest:
    snapshot_id: str
    byte_length: int
    sha256: str

    @classmethod
    def from_snapshot(cls, snapshot: MarketSnapshot) -> "ReplayManifest":
        payload = snapshot.canonical_bytes()
        return cls(snapshot.snapshot_id, len(payload), hashlib.sha256(payload).hexdigest())


class DeterministicSnapshotBuilder:
    """Orchestrates only read-only feature and structure adapters."""

    def __init__(self) -> None:
        self.schema = FeatureSchema.existing_multitimeframe(timeframe.value for timeframe in ALL_TIMEFRAMES)
        self.feature_adapter = ExistingResearchFeatureAdapter(self.schema)
        self.structure_adapter = StructuralLevelAdapter()

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
        constructed_at = utc_datetime(built_at or boundary)
        timeframes = []
        for timeframe in ALL_TIMEFRAMES:
            state = self.feature_adapter.build_timeframe(one_minute, timeframe, boundary)
            if timeframe.structural_lookback is not None:
                if state.status is DataStatus.INVALID:
                    structural = StructuralContext(
                        status=DataStatus.INVALID, pivots=(), reason="SOURCE_INVALID"
                    )
                else:
                    candles = self.feature_adapter.aggregate_closed(one_minute, timeframe, boundary)
                    structural = self.structure_adapter.context(
                        candles, timeframe=timeframe, decision_at=boundary,
                        reference_price=reference_price,
                    )
                state = replace(state, structural=structural)
            state.assert_causal(boundary)
            timeframes.append(state)
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
            built_at=constructed_at,
            proposed_side=proposed_side,
            signal_id=signal_id,
            reference_price=reference_price,
            timeframes=timeframes,
            source_versions=versions,
        )
