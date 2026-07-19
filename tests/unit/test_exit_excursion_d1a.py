from datetime import datetime, timedelta, timezone

import pytest

from aegis.data import CanonicalBar
from scripts.diagnostics.exit_excursion_d1a.experiment import (
    build_trajectories,
    cumulative_short_mfe,
    giveback_and_capture,
    short_mfe_mae,
    temporal_short_returns,
    temporal_exit_analysis,
    threshold_before_adverse,
)


def _bars(values: list[tuple[float, float, float]]) -> tuple[CanonicalBar, ...]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return tuple(
        CanonicalBar(start + timedelta(minutes=5 * index), close, high, low, close, 1.0)
        for index, (high, low, close) in enumerate(values)
    )


def test_short_mfe_mae_and_first_extrema_are_causal() -> None:
    path = _bars([(101, 99, 100), (103, 98, 99), (102, 95, 96)])
    mfe, mae, mfe_bar, mae_bar = short_mfe_mae(100.0, path)
    assert mfe == pytest.approx(0.05)
    assert mae == pytest.approx(0.03)
    assert mfe_bar == 3
    assert mae_bar == 2
    assert cumulative_short_mfe(100.0, path) == pytest.approx((0.01, 0.02, 0.05))
    assert temporal_short_returns(100.0, path) == pytest.approx((0.0, 0.01, 0.04))


def test_giveback_and_capture_handle_zero_mfe() -> None:
    assert giveback_and_capture(0.02, 0.005) == pytest.approx((0.015, 0.25))
    assert giveback_and_capture(0.0, -0.01) == (0.01, None)


def test_threshold_order_is_conservative_with_same_bar_ohlc() -> None:
    path = _bars([(100.3, 99.8, 100.0), (100.1, 99.5, 99.7)])
    assert threshold_before_adverse(100.0, path, favorable_bps=20, adverse_bps=20) == (False, True)
    assert threshold_before_adverse(100.0, path, favorable_bps=40, adverse_bps=20) == (False, False)
    assert threshold_before_adverse(100.0, path, favorable_bps=40, adverse_bps=40) == (True, False)


def test_h12_builder_reproduces_next_open_and_rejects_lookahead_boundary() -> None:
    from scripts.diagnostics.exit_excursion_d1a.experiment import FrozenEntry, DataLimitationError

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    prices = _bars([(101 - i, 99 - i, 100 - i) for i in range(14)])
    entry_price = prices[1].open
    path = prices[1:13]
    mfe, mae, _, _ = short_mfe_mae(entry_price, path)
    gross = (entry_price - path[-1].close) / entry_price
    entry = FrozenEntry(
        "trade", "ADAUSDT", 1, prices[0].timestamp, prices[1].timestamp, path[-1].timestamp,
        entry_price, path[-1].close, gross, mfe, mae, 0.0015, gross - 0.0015,
    )
    trajectories, excursions, quality = build_trajectories(
        (entry,), {"ADAUSDT": prices}, holding_bars=12, dev_boundary=path[-1].timestamp,
    )
    assert len(trajectories) == 12
    assert excursions[0]["gross_h12_return"] == pytest.approx(gross)
    assert quality["baseline_maximum_absolute_error"] == 0.0
    with pytest.raises(DataLimitationError, match="boundary"):
        build_trajectories(
            (entry,), {"ADAUSDT": prices}, holding_bars=12,
            dev_boundary=path[-1].timestamp - timedelta(seconds=1),
        )


def test_frozen_e3_entry_hash_and_count_remain_bound() -> None:
    import json
    from pathlib import Path
    from aegis.utils import Sha256HashProvider, sha256_file

    path = Path("reports/experiments/e3_validation_official/attempt_1/aegis-short-candidate-e3/runs/d742d9bc0ae867bb/econ_report.json")
    assert sha256_file(path) == "bff472758eacc211dff1b3e2209cbd96e8a845a68f45b9e31526ca2968e6e085"
    payload = json.loads(path.read_text())
    trades = [row for row in payload["report"]["trades"] if row["scenario_id"] == "B_BASE" and row["signal"]["strategy_id"] == "full_stack"]
    assert len(trades) == 1292
    assert Sha256HashProvider().digest_value(trades) == "7cf1bd1a9a4a78ac8557672acd95e0c2366ed8cfbe9029055edbb5b419c752c7"


def test_temporal_exit_metrics_are_deterministic_and_charge_duration_costs() -> None:
    from aegis.training.econ import COST_SCENARIOS

    rows = [
        {
            "signal_timestamp": f"2026-01-0{index + 1}T00:00:00Z", "fold": 1,
            "mtm_t5": value, "mtm_t10": value * 2,
        }
        for index, value in enumerate((0.01, -0.005, 0.002))
    ]
    kwargs = {
        "excursions": rows, "horizons": (1, 2), "scenarios": COST_SCENARIOS,
        "bootstrap_repetitions": 20, "seed": 7,
    }
    first = temporal_exit_analysis(**kwargs)
    second = temporal_exit_analysis(**kwargs)
    assert first == second
    t5 = first["horizons"]["T5"]["B_BASE"]["pooled"]
    t10 = first["horizons"]["T10"]["B_BASE"]["pooled"]
    assert t5["cost_per_trade"] == pytest.approx(COST_SCENARIOS[1].cost_fraction(1))
    assert t10["cost_per_trade"] == pytest.approx(COST_SCENARIOS[1].cost_fraction(2))
    assert t5["trades"] == 3
