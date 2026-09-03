import pandas as pd

from aegis_strategy_router.adapters.existing_features import ExistingResearchFeatureAdapter
from aegis_strategy_router.domain.types import DataStatus, Timeframe
from aegis_strategy_router.schemas import FeatureSchema
from conftest import make_one_minute


def _adapter() -> ExistingResearchFeatureAdapter:
    schema = FeatureSchema.existing_multitimeframe(timeframe.value for timeframe in Timeframe)
    return ExistingResearchFeatureAdapter(schema)


def test_4h_and_1d_warmup_are_real_not_nominal() -> None:
    source = make_one_minute(100 * 1_440)
    decision = pd.to_datetime(source.iloc[-1]["open_time_ms"], unit="ms", utc=True) + pd.Timedelta(minutes=1)
    adapter = _adapter()
    for timeframe in (Timeframe.H4, Timeframe.D1):
        state = adapter.build_timeframe(source, timeframe, decision.to_pydatetime())
        assert state.status is DataStatus.AVAILABLE
        assert state.candle_count >= state.required_warmup_bars
        assert all(feature.status is DataStatus.AVAILABLE for feature in state.features.observations)
        assert all(feature.available_at <= decision.to_pydatetime() for feature in state.features.observations)


def test_duplicate_source_timestamp_is_invalid_not_silently_deduplicated(one_minute: pd.DataFrame) -> None:
    duplicate = pd.concat([one_minute, one_minute.iloc[[0]]], ignore_index=True)
    decision = pd.to_datetime(one_minute.iloc[-1]["open_time_ms"], unit="ms", utc=True) + pd.Timedelta(minutes=1)
    state = _adapter().build_timeframe(duplicate, Timeframe.M15, decision.to_pydatetime())
    assert state.status is DataStatus.INVALID
    assert state.reason.startswith("SOURCE_INVALID")


def test_source_gap_is_invalid_not_silently_filled(one_minute: pd.DataFrame) -> None:
    gapped = one_minute.drop(index=100).reset_index(drop=True)
    decision = pd.to_datetime(one_minute.iloc[-1]["open_time_ms"], unit="ms", utc=True) + pd.Timedelta(minutes=1)
    state = _adapter().build_timeframe(gapped, Timeframe.M15, decision.to_pydatetime())
    assert state.status is DataStatus.INVALID
    assert "timestamp gaps" in state.reason
