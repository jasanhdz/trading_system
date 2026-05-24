from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aegis_alpha.tools import audit_turbo_directional_metrics as audit


def _dt(minutes: int = 0) -> datetime:
    return datetime(2026, 5, 24, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=minutes)


def _candles(rows: list[tuple[int, float, float, float, float]]) -> list[audit.Candle]:
    return [
        audit.Candle(timestamp=_dt(minutes), open=open_, high=high, low=low, close=close, volume=1.0)
        for minutes, open_, high, low, close in rows
    ]


def test_long_outcome_hit5() -> None:
    signal = audit.Signal(symbol="TEST", timestamp=_dt(0), action="LONG", score=0.9)
    candles = _candles([(0, 100, 100, 100, 100), (5, 100, 105, 99.8, 104)])

    outcome = audit.compute_signal_outcome(signal, candles, 15, fee_bps=0, slippage_bps=0, leverage_proxy=20)

    assert outcome is not None
    assert round(outcome.mfe or 0, 4) == 0.05
    assert round(outcome.mae or 0, 4) == -0.002
    assert outcome.hit5BeforeMinus5 is True


def test_short_outcome_hit5() -> None:
    signal = audit.Signal(symbol="TEST", timestamp=_dt(0), action="SHORT", score=0.9)
    candles = _candles([(0, 100, 100, 100, 100), (5, 100, 100.2, 95, 96)])

    outcome = audit.compute_signal_outcome(signal, candles, 15, fee_bps=0, slippage_bps=0, leverage_proxy=20)

    assert outcome is not None
    assert round(outcome.mfe or 0, 4) == 0.05
    assert round(outcome.mae or 0, 4) == -0.002
    assert outcome.hit5BeforeMinus5 is True


def test_hit_before_stop_ordering() -> None:
    candles = _candles([(5, 100, 100.2, 99.4, 100), (10, 100, 101, 99.5, 100)])

    result, target_time, stop_time = audit.hit_before(
        side="LONG",
        entry_price=100,
        candles=candles,
        target_return=0.008,
        stop_return=0.005,
    )

    assert result is False
    assert target_time is None
    assert stop_time == 0


def test_score_bucket_assignment() -> None:
    buckets = [0.55, 0.60, 0.70, 0.80, 0.90]

    assert audit.assign_score_bucket(0.57, buckets) == "0.55-0.60"
    assert audit.assign_score_bucket(0.91, buckets) == "0.90-1.00"


def test_confusion_matrix_and_class_metrics() -> None:
    outcomes = [
        audit.SignalOutcome("TEST", _dt().isoformat(), "LONG", 0.9, 100, 60, 0.01, 0.01, 0.01, -0.001, 0.01, 0.001, True, True, True, 5, None, "LONG"),
        audit.SignalOutcome("TEST", _dt().isoformat(), "SHORT", 0.9, 100, 60, -0.01, -0.01, 0.01, -0.001, 0.01, 0.001, False, False, False, None, 5, "LONG"),
    ]

    matrix = audit.build_confusion(outcomes)
    metrics = audit.class_metrics(matrix)

    assert matrix["LONG"]["LONG"] == 1
    assert matrix["SHORT"]["LONG"] == 1
    assert metrics.precision_LONG == 1.0
    assert metrics.recall_LONG == 0.5


def test_directional_status_unknown_when_insufficient_samples() -> None:
    status, confidence, warnings, actions = audit.classify_symbol(
        symbol="TEST",
        long_metrics=audit.SideMetrics(count=1, netExpectancy60m=0.01),
        short_metrics=audit.SideMetrics(count=0),
        score_calibration="UNKNOWN",
        min_samples=20,
        leverage_proxy=20,
    )

    assert status == "UNKNOWN"
    assert confidence == 0.0
    assert "insufficient_directional_samples" in warnings
    assert "insufficient_directional_samples" in actions


def test_directional_status_green_yellow_red() -> None:
    green = audit.classify_symbol(
        symbol="GREEN",
        long_metrics=audit.SideMetrics(count=30, netExpectancy60m=0.002, p90Mae=0.002),
        short_metrics=audit.SideMetrics(count=0),
        score_calibration="IMPROVES",
        min_samples=20,
        leverage_proxy=20,
    )
    yellow = audit.classify_symbol(
        symbol="YELLOW",
        long_metrics=audit.SideMetrics(count=30, netExpectancy60m=0.002, p90Mae=0.02),
        short_metrics=audit.SideMetrics(count=0),
        score_calibration="NOT_CALIBRATED",
        min_samples=20,
        leverage_proxy=20,
    )
    red = audit.classify_symbol(
        symbol="RED",
        long_metrics=audit.SideMetrics(count=30, netExpectancy60m=-0.002, p90Mae=0.002),
        short_metrics=audit.SideMetrics(count=0, netExpectancy60m=None),
        score_calibration="IMPROVES",
        min_samples=20,
        leverage_proxy=20,
    )

    assert green[0] == "GREEN"
    assert yellow[0] == "YELLOW"
    assert red[0] == "RED"


def test_neutral_metrics() -> None:
    outcomes = [
        audit.SignalOutcome("TEST", _dt().isoformat(), "HOLD", 0.1, 100, 60, 0.001, 0.0, 0.001, -0.001, 0.0, None, None, None, None, None, None, "NEUTRAL"),
        audit.SignalOutcome("TEST", _dt().isoformat(), "HOLD", 0.1, 100, 60, 0.01, 0.0, 0.01, -0.001, 0.0, None, None, None, None, None, None, "LONG"),
    ]

    metrics = audit.neutral_metrics(outcomes, threshold_return=0.004)

    assert metrics.count == 2
    assert metrics.largeMoveRate == 0.5
    assert metrics.missedLongMoveRate == 0.5


def test_fees_slippage_reduce_expectancy() -> None:
    signal = audit.Signal(symbol="TEST", timestamp=_dt(0), action="LONG", score=0.9)
    candles = _candles([(0, 100, 100, 100, 100), (60, 100, 101, 100, 101)])

    gross = audit.compute_signal_outcome(signal, candles, 60, fee_bps=0, slippage_bps=0, leverage_proxy=20)
    net = audit.compute_signal_outcome(signal, candles, 60, fee_bps=8, slippage_bps=3, leverage_proxy=20)

    assert gross is not None and net is not None
    assert (net.netForwardReturn or 0) < (gross.netForwardReturn or 0)


def test_json_csv_serialization_minimal() -> None:
    report = {
        "generatedAt": _dt().isoformat(),
        "source": "test",
        "from": _dt().isoformat(),
        "to": _dt(60).isoformat(),
        "sampleCount": 0,
        "outcomeCount": 0,
        "classificationHorizonMinutes": 60,
        "symbolSummaries": [],
        "scoreBuckets": [],
        "confusionRows": [],
    }
    with tempfile.TemporaryDirectory() as tmp:
        paths = audit.write_reports(report, Path(tmp), "20260524T000000Z", write_md=True, write_json_flag=True, write_csv_flag=True)
        assert Path(paths["json"]).exists()
        assert json.loads(Path(paths["json"]).read_text(encoding="utf-8"))["source"] == "test"
        assert Path(paths["summary_csv"]).exists()


def run_manual_tests() -> None:
    tests = [
        test_long_outcome_hit5,
        test_short_outcome_hit5,
        test_hit_before_stop_ordering,
        test_score_bucket_assignment,
        test_confusion_matrix_and_class_metrics,
        test_directional_status_unknown_when_insufficient_samples,
        test_directional_status_green_yellow_red,
        test_neutral_metrics,
        test_fees_slippage_reduce_expectancy,
        test_json_csv_serialization_minimal,
    ]
    for test in tests:
        test()
    print("manual_turbo_directional_metrics_tests_passed")


if __name__ == "__main__":
    run_manual_tests()
