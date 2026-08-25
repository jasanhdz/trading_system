from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aegis_range_v1.candidates import RangeCandidate
from aegis_range_v1.costs import BASELINE, adverse_fill
from aegis_range_v1.engine import RangeEngineV1
from aegis_range_v1.lifecycle import RangeLifecycleV1
from aegis_range_v1.models import PendingEntry, RegimeSnapshot
from aegis_range_v1.regime import RangeRegimeAdapter
from aegis_range_v1.regime_bridge import TypeScriptRegimeEvaluator
from aegis_range_v1.train_backtest import (
    CachedRangeRegimeAdapter,
    ObservedRangeEngineV1,
    ObservedRangeLifecycleV1,
    _deterministic_gzip_jsonl,
    _funding_slice,
    _scenario_returns,
    candidate_metrics,
    verify_authority,
)

from conftest import FakeRegimeEvaluator, make_5m
from test_engine_master import synthetic_history


def constant_snapshots(count: int) -> list[RegimeSnapshot | None]:
    snapshot = RegimeSnapshot("ACCUMULATION_RANGE", "LOW", 18.0, 0.4, 0.2, 1.0, "NONE", 0, "MIXED", 0.7, 2.0)
    return [None] * 159 + [snapshot] * (count - 159)


def test_train_authority_is_fail_closed_and_train_only(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    with pytest.raises(PermissionError, match="PARTITION_VIOLATION"):
        verify_authority(root, {})
    result = verify_authority(
        root,
        {"TRAIN_ACCESS": "true", "CALIBRATION_ACCESS": "false", "VALIDATION_ACCESS": "false", "HOLDOUT_ACCESS": "false"},
    )
    assert result["r1"] == {
        "manifest_sha256": "2a55d0a5511b178b8d8c8a5b0a7259ecffe800e59b715234c75ff2eea8639d5c",
        "verified_files": "42",
    }


def test_cached_adapter_exact_warmup_and_snapshot(origin):
    candles = [make_5m(index, origin) for index in range(161)]
    snapshots = constant_snapshots(len(candles))
    adapter = CachedRangeRegimeAdapter(candles, snapshots)
    with pytest.raises(ValueError, match="INSUFFICIENT_HISTORY"):
        adapter.snapshot("BTCUSDT", candles[:159])
    assert adapter.snapshot("BTCUSDT", candles[:160]) is snapshots[159]


def test_observed_lifecycle_captures_raw_entry_and_exit(origin, candidate):
    lifecycle = ObservedRangeLifecycleV1(candidate)
    pending = PendingEntry(
        "BTCUSDT", "LONG", origin, origin + timedelta(minutes=5), "episode", "range",
        origin - timedelta(hours=1), 90.0, 110.0, 100.0, 2.0, "ACCUMULATION_RANGE", 70.0, None,
    )
    lifecycle.schedule_entry(pending)
    position = lifecycle.consume_pending_entry(
        open_at=pending.entry_available_at, raw_open=94.0, same_split=True, episode_active=True
    )
    assert position is not None
    assert lifecycle.entry_events[0][1] == 94.0
    candle = make_5m(1, origin, open_=94.0, high=101.0, low=88.0, close=95.0)
    event = lifecycle.process_position_open_and_intrabar(candle, include_open_gaps=False)
    assert event is not None and event.reason == "STOP"
    captured_position, captured_event, exit_base = lifecycle.exit_events[0]
    assert captured_position.thesis_feature_hash == position.thesis_feature_hash
    assert captured_event == event
    assert exit_base == position.stop_at_entry


def test_observed_engine_preserves_decision_state(origin, candidate):
    candles = synthetic_history(origin, 180)
    direct = RangeEngineV1("BTCUSDT", candidate, RangeRegimeAdapter(FakeRegimeEvaluator()))
    observed = ObservedRangeEngineV1("BTCUSDT", candidate, CachedRangeRegimeAdapter(candles, constant_snapshots(len(candles))))
    keys = ("decision_at", "status", "episode_event", "signal", "entry_hash", "exit_reason", "range_id", "range_episode_id")
    for candle in candles:
        direct_output = direct.process(candle)
        observed_output = observed.process(candle)
        assert {key: direct_output.get(key) for key in keys} == {key: observed_output.get(key) for key in keys}
        assert direct.levels.structural_snapshot() == observed.levels.structural_snapshot()
        assert direct.episode == observed.episode
        direct.outputs.clear()


def test_funding_interval_excludes_entry_and_includes_exit(origin):
    events = [
        (origin, 0.001, 100.0),
        (origin + timedelta(hours=8), 0.002, 101.0),
        (origin + timedelta(hours=16), 0.003, 102.0),
    ]
    selected = _funding_slice(events, origin, origin + timedelta(hours=8))
    assert selected == ((0.002, 101.0),)
    long = _scenario_returns("LONG", 100.0, 105.0, selected)
    short = _scenario_returns("SHORT", 100.0, 95.0, selected)
    assert long.keys() == short.keys() == {"BASELINE", "STRESS_20", "STRESS_30"}
    assert long["BASELINE"]["funding_return"] < 0 < short["BASELINE"]["funding_return"]
    assert long["BASELINE"]["funding_return"] == -0.002 * 101.0 / long["BASELINE"]["entry_fill"]
    assert short["BASELINE"]["funding_return"] == 0.002 * 101.0 / short["BASELINE"]["entry_fill"]
    assert long["BASELINE"]["entry_fill"] == adverse_fill(100.0, "LONG", BASELINE.slippage_bps_per_side)


def test_stress_reprices_same_population_only():
    returns = _scenario_returns("LONG", 100.0, 105.0, ())
    assert set(returns) == {"BASELINE", "STRESS_20", "STRESS_30"}
    assert all(set(item) == {"entry_fill", "exit_fill", "gross_return", "fees", "funding_return", "net_return"} for item in returns.values())
    assert returns["BASELINE"]["net_return"] > returns["STRESS_20"]["net_return"] > returns["STRESS_30"]["net_return"]


def test_candidate_metrics_preserve_no_trade_episodes(candidate):
    episode = {
        "candidate_id": "C000", "candidate": candidate.as_dict(), "symbol": "BTCUSDT",
        "range_episode_id": "e0", "range_confirmed_at": "2024-01-03T00:00:00.000Z",
        "trade_count": 0, "false_range": False, "purged": False,
        "baseline_net_return": 0.0, "baseline_gross_return": 0.0,
        "stress_20_net_return": 0.0, "stress_20_gross_return": 0.0,
        "stress_30_net_return": 0.0, "stress_30_gross_return": 0.0,
    }
    operated = dict(episode, range_episode_id="e1", trade_count=1, baseline_net_return=0.01, baseline_gross_return=0.012, stress_20_net_return=0.009, stress_20_gross_return=0.011, stress_30_net_return=0.007, stress_30_gross_return=0.009)
    scenarios = {
        name: {"net_return": value, "gross_return": value + 0.002, "fees": 0.001, "funding_return": 0.0, "entry_fill": 100.0, "exit_fill": 101.0}
        for name, value in (("BASELINE", 0.01), ("STRESS_20", 0.009), ("STRESS_30", 0.007))
    }
    trade = {
        "candidate_id": "C000", "symbol": "BTCUSDT", "range_episode_id": "e1", "side": "LONG",
        "exit_at": "2024-01-03T01:00:00.000Z", "exit_reason": "TARGET", "thesis_feature_hash": "h",
        "purged": False, "scenarios": scenarios,
    }
    metrics = candidate_metrics("C000", [episode, operated], [trade])
    assert metrics["confirmed_episodes"] == 2
    assert metrics["operated_episodes"] == 1
    assert metrics["abstention_rate"] == 0.5
    assert metrics["scenarios"]["BASELINE"]["episode_net_expectancy"] == 0.01


def test_deterministic_gzip_jsonl(tmp_path):
    rows = [{"b": 2, "a": 1}, {"b": 4, "a": 3}]
    first = tmp_path / "first.jsonl.gz"
    second = tmp_path / "second.jsonl.gz"
    assert _deterministic_gzip_jsonl(first, rows) == _deterministic_gzip_jsonl(second, rows)
    assert first.read_bytes() == second.read_bytes()


def test_train_batch_bridge_matches_approved_single_window(origin):
    root = Path(__file__).resolve().parents[3]
    candles = tuple(make_5m(index, origin, open_=100 + index * 0.01, high=101 + index * 0.01, low=99 + index * 0.01, close=100.2 + index * 0.01) for index in range(160))
    expected = TypeScriptRegimeEvaluator(root).evaluate(symbol="BTCUSDT", candles=candles, timeframe="5m")
    payload = {
        "symbol": "BTCUSDT", "timeframe": "5m",
        "candles": [{"timestamp": int(item.open_time.timestamp() * 1000), "open": item.open, "high": item.high, "low": item.low, "close": item.close, "volume": item.volume} for item in candles],
    }
    process = subprocess.run(
        ["node", "-r", "ts-node/register", str(root / "sandbox/aegis_range_strategy_v1/scripts/regime_v2_train_batch_bridge.cjs")],
        cwd=root / "binance-futures-bot-ts", input=json.dumps(payload), capture_output=True, text=True, check=True,
    )
    actual = json.loads(process.stdout)
    assert actual == {
        "timestamp": int(candles[-1].open_time.timestamp() * 1000),
        "technicalRegime": expected["technicalRegime"],
        "transitionRisk": expected["transition"]["risk"],
        "adx": expected["indicators"]["adx"],
        "atrPercentile": expected["indicators"]["atrPercentile"],
        "bollingerWidthPercentile": expected["indicators"]["bollingerWidthPercentile"],
        "volumeRatio": expected["indicators"]["volumeRatio"],
        "rangeBreakout": expected["indicators"]["rangeBreakout"],
        "failedBreakoutCount": expected["indicators"]["failedBreakoutCount"],
        "structure": expected["indicators"]["structure"],
        "chopRisk": expected["scores"]["chopRisk"],
    }


def test_train_batch_bridge_does_not_truncate_pipe_output(origin):
    root = Path(__file__).resolve().parents[3]
    candles = [make_5m(index, origin, open_=100 + index * 0.001, high=101 + index * 0.001, low=99 + index * 0.001, close=100.2 + index * 0.001) for index in range(2200)]
    payload = {
        "symbol": "BTCUSDT", "timeframe": "5m",
        "candles": [{"timestamp": int(item.open_time.timestamp() * 1000), "open": item.open, "high": item.high, "low": item.low, "close": item.close, "volume": item.volume} for item in candles],
    }
    process = subprocess.run(
        ["node", "-r", "ts-node/register", str(root / "sandbox/aegis_range_strategy_v1/scripts/regime_v2_train_batch_bridge.cjs")],
        cwd=root / "binance-futures-bot-ts", input=json.dumps(payload), capture_output=True, text=True, check=True,
    )
    assert len(process.stdout.splitlines()) == len(candles) - 159
