import zipfile

import pytest

from aegis.domain import TradeSide
from aegis.research.market_event_fast_track_m1a import (
    CausalRegimeClassifier,
    DirectionAxis,
    FastTrackContractError,
    FlowBucket,
    MinuteBar,
    PatternThresholds,
    RegimeObservation,
    RegimeThresholds,
    VolatilityAxis,
    LiquidityAxis,
    detect_micro_patterns,
    extract_pattern_features,
    fit_pattern_thresholds_from_train,
    normalize_timestamp_ms,
    read_agg_trade_archive,
    read_kline_archive,
    resample_closed_bars,
)


def _bars(count=300, drift=0.0002, symbol="ADAUSDT"):
    start = 1_767_225_600_000
    rows = []
    price = 100.0
    for index in range(count):
        open_price = price
        close = open_price * (1.0 + drift)
        rows.append(
            MinuteBar(
                symbol, start + index * 60_000, open_price,
                max(open_price, close) * 1.0002, min(open_price, close) * 0.9998,
                close, 10.0, 1000.0 + index, 20, 600.0,
            )
        )
        price = close
    return tuple(rows)


def test_timestamp_normalization_handles_spot_microseconds() -> None:
    assert normalize_timestamp_ms(1_767_225_600_000) == 1_767_225_600_000
    assert normalize_timestamp_ms(1_767_225_600_000_000) == 1_767_225_600_000
    with pytest.raises(FastTrackContractError, match="TIMESTAMP"):
        normalize_timestamp_ms(123)


def test_archive_parsers_preserve_aggressor_semantics(tmp_path) -> None:
    kline = tmp_path / "kline.zip"
    with zipfile.ZipFile(kline, "w") as handle:
        handle.writestr(
            "rows.csv",
            "open_time,open,high,low,close,volume,close_time,quote_volume,trades,taker_buy_base,taker_buy_quote,ignore\n"
            "1767225600000000,100,101,99,100.5,10,1767225659999999,1000,20,6,600,0\n",
        )
    assert read_kline_archive(kline, "ADAUSDT")[0].open_time_ms == 1_767_225_600_000

    trades = tmp_path / "trades.zip"
    with zipfile.ZipFile(trades, "w") as handle:
        handle.writestr(
            "rows.csv",
            "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n"
            "1,100,2,1,1,1767225600100,false\n"
            "2,100,1,2,2,1767225600200,true\n",
        )
    bucket = read_agg_trade_archive(trades, "ADAUSDT")[0]
    assert bucket.aggressive_buy_quote == 200
    assert bucket.aggressive_sell_quote == 100
    assert bucket.imbalance == pytest.approx(1 / 3)


def test_resampling_requires_complete_closed_groups() -> None:
    bars = _bars(11)
    result = resample_closed_bars(bars, 5)
    assert len(result) == 2
    assert result[0].interval_minutes == 5
    assert result[0].open == bars[0].open
    assert result[0].close == bars[4].close
    incomplete = bars[:4] + bars[5:10]
    assert len(resample_closed_bars(incomplete, 5)) == 1


def test_regime_is_stateful_causal_and_pattern_engine_uses_context() -> None:
    minute = _bars(2000)
    hourly = resample_closed_bars(minute[:1800], 60)
    four_hourly = resample_closed_bars(minute[:1920], 240)
    classifier = CausalRegimeClassifier(
        RegimeThresholds(0.0001, 0.00005, 0.00001, 0.01, 0.02, 1, 10_000),
        minimum_state_bars=1,
    )
    regime = classifier.observe(hourly, four_hourly)
    assert regime.direction is DirectionAxis.BULL
    with pytest.raises(FastTrackContractError, match="NON_CHRONOLOGICAL"):
        classifier.observe(hourly, four_hourly)

    futures = list(_bars(300, drift=0.00005))
    # A compressed range followed by an explicit upside breakout with flow.
    prior_high = max(row.high for row in futures[-21:-1])
    last = futures[-1]
    futures[-1] = MinuteBar(
        last.symbol, last.open_time_ms, last.open, prior_high * 1.003,
        last.open * 0.9999, prior_high * 1.002, 100, 10_000, 200, 9000,
    )
    spot = tuple(
        MinuteBar(
            row.symbol, row.open_time_ms, row.open, row.high, row.low, row.close,
            row.base_volume, row.quote_volume, row.trade_count, row.taker_buy_quote,
        )
        for row in futures
    )
    flow = tuple(
        FlowBucket(row.symbol, row.open_time_ms, 900 if index == len(futures) - 1 else 600, 100 if index == len(futures) - 1 else 400, 20)
        for index, row in enumerate(futures)
    )
    context = RegimeObservation(
        futures[-1].close_time_ms, 0.01, 0.005, 1000,
        DirectionAxis.BULL, VolatilityAxis.EXPANDING, LiquidityAxis.NORMAL, "f" * 64,
    )
    thresholds = PatternThresholds(0.5, 2.0, 1.0, 0.001, 2.0, 0.01, 0.0001, 0.001, 0.0001, 0.01)
    candidates = detect_micro_patterns(
        symbol="ADAUSDT", futures=futures, spot=spot, flow=flow,
        regime=context, thresholds=thresholds, funding_rate=0.0001,
    )
    assert any(item.side is TradeSide.LONG for item in candidates)
    assert all(item.timestamp_ms == futures[-1].close_time_ms for item in candidates)


def test_pattern_engine_rejects_future_or_misaligned_sources() -> None:
    futures = _bars(300)
    spot = _bars(300)
    flow = tuple(FlowBucket(row.symbol, row.open_time_ms, 600, 400, 20) for row in futures)
    regime = RegimeObservation(
        futures[-1].close_time_ms, 0, 0, 1000,
        DirectionAxis.RANGE, VolatilityAxis.NORMAL, LiquidityAxis.NORMAL, "f" * 64,
    )
    thresholds = PatternThresholds(0.5, 2, 0.5, 0.001, 2, 0.01, 0.001, 0.001, 0.001, 0.01)
    shifted = (*spot[:-1], MinuteBar(
        spot[-1].symbol, spot[-1].open_time_ms + 60_000, spot[-1].open,
        spot[-1].high, spot[-1].low, spot[-1].close, spot[-1].base_volume,
        spot[-1].quote_volume, spot[-1].trade_count, spot[-1].taker_buy_quote,
    ))
    with pytest.raises(FastTrackContractError, match="CAUSALITY|CLOCK"):
        detect_micro_patterns(
            symbol="ADAUSDT", futures=futures, spot=shifted, flow=flow,
            regime=regime, thresholds=thresholds, funding_rate=None,
        )


def test_thresholds_are_fitted_only_from_explicit_train_snapshots() -> None:
    futures = _bars(300)
    spot = _bars(300)
    flow = tuple(FlowBucket(row.symbol, row.open_time_ms, 600, 400, 20) for row in futures)
    snapshot = extract_pattern_features(
        futures=futures, spot=spot, flow=flow, funding_rate=0.0
    )
    thresholds = fit_pattern_thresholds_from_train((snapshot,) * 1000)
    assert thresholds.minimum_flow_imbalance == pytest.approx(abs(snapshot["flow_3"]))
    assert thresholds.minimum_volume_ratio == pytest.approx(snapshot["volume_ratio"])
    with pytest.raises(FastTrackContractError, match="TRAIN_ROWS_INSUFFICIENT"):
        fit_pattern_thresholds_from_train((snapshot,) * 999)
