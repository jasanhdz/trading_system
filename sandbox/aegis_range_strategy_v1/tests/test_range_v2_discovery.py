from __future__ import annotations

import gzip
import csv
import json
from datetime import datetime, timedelta, timezone

import pytest

from aegis_range_v1.models import Candle5m
from aegis_range_v1.readiness import SourceIntegrityError
from aegis_range_v1.range_v2_discovery import (
    assign_unique_weights,
    canonical_opportunity_id,
    causal_terciles,
    confirmation_entry,
    counterfactual_exit,
    deterministic_gzip_jsonl,
    failure_anatomy_summary,
    confirmation_q_summary,
    opportunity_path,
    prior_symbol_metrics,
    suitability_observation,
    stop_recovery,
    verify_authority,
    _load_rejection_context,
    FLAGS,
    REGIME_CACHE_MANIFEST_SHA256,
    RUN_A_HASHES,
    SYMBOLS,
)


ORIGIN = datetime(2024, 6, 1, tzinfo=timezone.utc)


def bar(index, open_=100, high=101, low=99, close=100, segment=0):
    opened = ORIGIN + timedelta(minutes=5 * index)
    return Candle5m("BTCUSDT", opened, opened + timedelta(minutes=5), open_, high, low, close, 1, segment)


def trade(**overrides):
    value = {
        "candidate_id": "C001", "symbol": "BTCUSDT", "side": "LONG",
        "decision_at": ORIGIN.isoformat(), "entry_at": ORIGIN.isoformat(), "exit_at": (ORIGIN + timedelta(minutes=5)).isoformat(),
        "entry_base": 100.0, "entry_fill": 100.02, "exit_base": 95.0, "exit_fill": 94.981,
        "exit_reason": "STOP", "support_at_entry": 96.0, "resistance_at_entry": 110.0, "midpoint_at_entry": 103.0,
        "ATR_entry": 2.0, "stop_at_entry": 95.0, "target_at_entry": 105.1,
        "gross_return": -0.05, "net_return": -0.051, "funding_return": 0.0,
    }
    value.update(overrides)
    return value


def episode(**overrides):
    value = {"range_confirmed_at": (ORIGIN - timedelta(hours=1)).isoformat(), "episode_end_at": None, "episode_end_reason": None, "false_range": False}
    value.update(overrides)
    return value


def test_canonical_id_ignores_outcomes_and_inverse_weights():
    first = trade(exit_reason="TARGET", net_return=99, stop_at_entry=94, target_at_entry=104)
    second = trade(exit_reason="STOP", net_return=-99, stop_at_entry=90, target_at_entry=108)
    assert canonical_opportunity_id(first) == canonical_opportunity_id(second)
    weighted = assign_unique_weights([first, second, trade(decision_at=(ORIGIN + timedelta(hours=1)).isoformat())])
    assert [row["unique_weight"] for row in weighted] == [0.5, 0.5, 1.0]
    assert sum(row["unique_weight"] for row in weighted) == 2


def test_conservative_stop_terminal_excludes_unknown_favorable_extreme():
    result = opportunity_path(trade(), [bar(0, high=102, low=98), bar(1, open_=97, high=120, low=94)])
    assert result["mfe_extremum_price"] == 102
    assert result["full_terminal_favorable_price"] == 120
    assert result["mae_extremum_price"] == 95
    assert result["mfe_extremum_frozen_range_position"] == pytest.approx(6 / 14)
    assert result["mae_extremum_signed_return_from_entry"] < 0
    assert result["mfe_extremum_signed_distance_to_midpoint"] < 0


def test_excursions_are_floored_at_entry_fill():
    row = trade(exit_at=ORIGIN.isoformat())
    result = opportunity_path(row, [bar(0, open_=100, high=100.01, low=95)])
    assert result["mfe"] == 0
    assert result["mfe_extremum_price"] == result["entry_fill"]


def test_single_terminal_open_midpoint_hit_is_not_lost():
    row = trade(exit_at=ORIGIN.isoformat(), exit_reason="MAX_HOLD", exit_base=103, exit_fill=102.9794)
    result = opportunity_path(row, [bar(0, open_=103, high=120, low=80)])
    assert result["midpoint_hit_while_model_open"] is True
    assert result["mfe_extremum_price"] == 103
    assert result["full_terminal_mfe_extremum_price"] == 120


def test_stop_recovery_starts_after_terminal_and_categories():
    row = {**trade(exit_at=ORIGIN.isoformat()), "canonical_opportunity_id": "x", "unique_weight": 1}
    bars = [bar(0, high=200, low=1)] + [bar(i, high=104 if i == 4 else 100, low=94) for i in range(1, 25)]
    result = stop_recovery(row, bars)
    assert result["horizon_15"]["complete_bars"] == 3
    assert result["horizon_15"]["midpoint_reached"] is False
    assert result["category"] == "STOP_THEN_MIDPOINT_RECOVERY"
    incomplete = stop_recovery(row, [bar(1, high=99, low=94)])
    assert incomplete["category"] == "STOP_AMBIGUOUS"


def test_stop_recovery_excursions_are_nonnegative():
    row = {**trade(exit_at=ORIGIN.isoformat()), "canonical_opportunity_id": "x", "unique_weight": 1}
    below_stop = stop_recovery(row, [bar(i, high=94, low=93) for i in range(1, 4)])
    above_stop = stop_recovery(row, [bar(i, high=97, low=96) for i in range(1, 4)])
    assert below_stop["horizon_15"]["favorable_excursion_from_stop"] == 0
    assert above_stop["horizon_15"]["adverse_continuation_from_stop"] == 0


def test_confirmation_a_and_b_timing_and_midpoint_cancellation():
    candles = [bar(-1, high=101, low=96, close=97), bar(0, high=101, low=98, close=99), bar(1, open_=100, high=101, low=99, close=100)]
    entered = confirmation_entry(trade(), episode(), candles, "NEXT_CLOSE_PROGRESS")
    assert entered["entry_at"] == bar(1).open_time.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    b_candles = [bar(-1, high=99, low=96, close=97), bar(0, high=99, low=98, close=98), bar(1, high=101, low=99, close=100), bar(2, open_=100), bar(3)]
    assert confirmation_entry(trade(), episode(), b_candles, "REJECTION_EXTREME_RECLAIM")["entry_at"] == bar(2).open_time.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    cancelled = [bar(-1, high=101, low=96, close=97), bar(0, high=104, low=98, close=99), bar(1)]
    assert confirmation_entry(trade(), episode(), cancelled, "NEXT_CLOSE_PROGRESS")["reason"] == "MIDPOINT_TOUCHED"


def test_confirmation_enforces_42bps_and_rr():
    candles = [bar(-1, high=101, low=96, close=97), bar(0, high=101, low=98, close=99), bar(1, open_=100)]
    assert confirmation_entry(trade(target_at_entry=100.4), episode(), candles, "NEXT_CLOSE_PROGRESS")["reason"] == "TARGET_DISTANCE_LT_42_BPS"
    assert confirmation_entry(trade(stop_at_entry=90, target_at_entry=105), episode(), candles, "NEXT_CLOSE_PROGRESS")["reason"] == "REWARD_RISK_LT_1"


def test_frozen_exit_is_adverse_first_breakout_and_max_hold():
    entry = {"status": "ENTERED", "method": "x", "entry_at": ORIGIN.isoformat(), "entry_base": 100}
    both = counterfactual_exit(trade(), entry, [bar(0, high=106, low=94)])
    assert both["exit_reason"] == "STOP"
    breakout_trade = trade(stop_at_entry=80, target_at_entry=120, support_at_entry=96, ATR_entry=2)
    breakout = counterfactual_exit(breakout_trade, entry, [bar(0, close=95), bar(1, close=95), bar(2, open_=94)])
    assert breakout["exit_reason"] == "TRADE_BREAKOUT" and breakout["exit_at"].endswith("00:10:00.000Z")
    flat = trade(stop_at_entry=80, target_at_entry=120, support_at_entry=80, resistance_at_entry=120)
    held = [bar(i) for i in range(145)]
    maximum = counterfactual_exit(flat, entry, held)
    assert maximum["exit_reason"] == "MAX_HOLD" and maximum["holding_bars"] == 144


def test_counterfactual_censors_a_data_gap():
    entry = {"status": "ENTERED", "method": "x", "entry_at": ORIGIN.isoformat(), "entry_base": 100}
    result = counterfactual_exit(trade(stop_at_entry=80, target_at_entry=120), entry, [bar(0), bar(2)])
    assert result["censored"] is True and result["purged"] is True


def observation(index, symbol="BTCUSDT", mature_offset=-1):
    decision = ORIGIN + timedelta(days=index)
    return {"symbol": symbol, "canonical_opportunity_id": f"{symbol}-{index}", "decision_at": decision.isoformat(), "maturity_at": (decision + timedelta(hours=12)).isoformat(), "midpoint_hit": True, "false_range": False, "breakout_after_boundary_touch": False, "bars_to_midpoint": 2, "structural_mae_after_boundary_touch": 1, "structural_mfe_after_boundary_touch": 2}


def test_prior_metrics_are_60d_prior_only_with_unique_sample_floor():
    rows = [observation(i) for i in range(31)]
    decision = ORIGIN + timedelta(days=31, hours=13)
    result = prior_symbol_metrics(rows + [{**observation(31), "maturity_at": (decision + timedelta(hours=1)).isoformat()}], decision, "BTCUSDT")
    assert result["status"] == "ELIGIBLE" and result["sample_count"] == 31
    assert result["MFE_to_MAE_ratio"] == 2
    insufficient = prior_symbol_metrics(rows[:29], decision, "BTCUSDT")
    assert insufficient["status"] == "INSUFFICIENT_HISTORY"
    assert causal_terciles([insufficient])["BTCUSDT"] == "INSUFFICIENT_HISTORY"
    assert causal_terciles([result, {**result, "symbol": "A", "boundary_to_midpoint_reversion_rate": 0.1}, {**result, "symbol": "Z", "boundary_to_midpoint_reversion_rate": 0.9}])["A"] == "LOW"


def test_suitability_failure_keeps_normalized_structural_extrema_and_breakout():
    candles = [bar(-1, high=101, low=99, close=100), bar(0, high=102, low=98), bar(1, high=101, low=99)]
    ended = episode(episode_end_at=(ORIGIN + timedelta(minutes=10)).isoformat(), episode_end_reason="CONFIRMED_BREAKOUT")
    result = suitability_observation({**trade(midpoint_at_entry=105), "canonical_opportunity_id": "x"}, ended, candles)
    assert result["midpoint_hit"] is False
    assert result["structural_mfe_after_boundary_touch"] == pytest.approx(0.02)
    assert result["structural_mae_after_boundary_touch"] == pytest.approx(0.02)
    assert result["breakout_after_boundary_touch"] is True
    unfavorable = suitability_observation({**trade(midpoint_at_entry=105), "canonical_opportunity_id": "x"}, episode(), [bar(-1, close=100), bar(0, high=99, low=98)])
    assert unfavorable["structural_mfe_after_boundary_touch"] == 0


def test_rejection_context_uses_fixed_atr_bins(tmp_path):
    root = tmp_path / "run_a" / "regime_cache"
    root.mkdir(parents=True)
    path = root / "BTCUSDT.csv.gz"
    fields = ["open_time", "atr_percentile", "technical_regime"]
    with gzip.open(path, "wt", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"open_time": (ORIGIN - timedelta(minutes=5)).isoformat(), "atr_percentile": 1 / 3, "technical_regime": "RANGE"})
    context = _load_rejection_context(tmp_path / "run_a", "BTCUSDT", {ORIGIN})[ORIGIN]
    assert context == {"atr_percentile_at_rejection": 1 / 3, "technical_regime_at_rejection": "RANGE", "atr_regime": "LOW"}


def test_failure_anatomy_and_q3_helpers_report_both_weights_and_counts():
    paths = []
    recoveries = []
    for index, amplitude in enumerate((0.01, 0.02, 0.03)):
        identifier = f"o{index}"
        paths.append({"canonical_opportunity_id": identifier, "row_weight": 1, "unique_weight": 1, "range_amplitude_pct": amplitude, "mfe": amplitude, "mae": amplitude / 2, "midpoint_hit_while_model_open": index > 0, "stop_first": index == 0, "target_first": index == 2})
        recovery = {"canonical_opportunity_id": identifier, "row_weight": 1, "unique_weight": 1, "range_amplitude_pct": amplitude, "episode_age_minutes": index * 800, "side": "LONG", "symbol": "BTCUSDT", "month": "2024-06", "atr_regime": "LOW", "category": "STOP_TRUE_FAILURE" if index == 0 else "STOP_AMBIGUOUS"}
        for minutes in (15, 30, 60, 120):
            recovery[f"horizon_{minutes}"] = {"entry_recovered": index > 0, "midpoint_reached": index == 2, "favorable_excursion_from_stop": amplitude, "adverse_continuation_from_stop": amplitude / 2, "mature": True}
        recoveries.append(recovery)
    summary = failure_anatomy_summary(paths, recoveries)
    assert summary["range_amplitude_pct_tercile_cutpoints"] == {"low_medium": pytest.approx(0.016666666666666666), "medium_high": pytest.approx(0.02333333333333333)}
    assert summary["views"]["unique_opportunity_weighted"]["v1_failure_anatomy"]["effective_weight"] == 3
    assert summary["views"]["candidate_weighted"]["stop_recovery"]["horizons"]["120"]["midpoint_recovery_rate"] == pytest.approx(1 / 3)

    originals = {("C1", "a"): {"exit_reason": "STOP"}, ("C2", "b"): {"exit_reason": "STOP"}, ("C3", "c"): {"exit_reason": "TARGET"}}
    rows = [
        {"candidate_id": "C1", "canonical_opportunity_id": "a", "row_weight": 1, "unique_weight": 1, "status": "NO_TRADE"},
        {"candidate_id": "C2", "canonical_opportunity_id": "b", "row_weight": 1, "unique_weight": 1, "status": "ENTERED", "exit_reason": "TARGET"},
        {"candidate_id": "C3", "canonical_opportunity_id": "c", "row_weight": 1, "unique_weight": 1, "status": "NO_TRADE"},
    ]
    q3 = confirmation_q_summary(rows, originals, "unique_weight")
    assert q3["original_stop_avoided_rate"] == 1
    assert q3["original_target_filtered"] == {"candidate_rows": 1, "effective_weight": 1.0, "rate": 1.0}


def test_deterministic_gzip(tmp_path):
    first, second = tmp_path / "a.gz", tmp_path / "b.gz"
    assert deterministic_gzip_jsonl(first, [{"z": 1, "a": 2}]) == deterministic_gzip_jsonl(second, [{"a": 2, "z": 1}])
    assert first.read_bytes() == second.read_bytes()
    with gzip.open(first, "rt", encoding="ascii") as handle:
        assert handle.read() == '{"a":2,"z":1}\n'


def test_authority_and_output_protection_fail_closed(tmp_path, monkeypatch):
    with pytest.raises(PermissionError, match="PARTITION"):
        verify_authority(tmp_path, tmp_path / "out", {"TRAIN_ACCESS": "true", "CALIBRATION_ACCESS": "true", "VALIDATION_ACCESS": "false", "HOLDOUT_ACCESS": "false"})
    repo = tmp_path / "repo"
    run_a = repo / "sandbox/aegis_range_strategy_v1/artifacts/r2_train/run_a"
    run_a.mkdir(parents=True)
    monkeypatch.setattr("aegis_range_v1.range_v2_discovery.SealedPartitionGuard.access_flags", lambda environment: {"TRAIN": True, "CALIBRATION": False, "VALIDATION": False, "HOLDOUT": False})
    with pytest.raises(PermissionError, match="OUTPUT_INSIDE_RUN_A"):
        verify_authority(repo, run_a / "diagnostic")


def test_authority_rejects_regime_cache_hash_drift(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    run_a = repo / "sandbox/aegis_range_strategy_v1/artifacts/r2_train/run_a"
    cache_root = run_a / "regime_cache"
    cache_root.mkdir(parents=True)
    for name in RUN_A_HASHES:
        (run_a / name).write_text("{}", encoding="ascii")
    (run_a / "run_manifest.json").write_text(json.dumps({"partition_flags": FLAGS, "artifacts": {"trades": {"rows": 22016}}}), encoding="ascii")
    regime = {"caches": {symbol: {"symbol": symbol, "sha256": "cache-ok"} for symbol in SYMBOLS}}
    (run_a / "regime_cache_manifest.json").write_text(json.dumps(regime), encoding="ascii")
    monkeypatch.setattr("aegis_range_v1.range_v2_discovery.SealedPartitionGuard.access_flags", lambda environment: FLAGS)

    def fake_hash(path):
        if path.name in RUN_A_HASHES:
            return RUN_A_HASHES[path.name]
        if path.name == "regime_cache_manifest.json":
            return REGIME_CACHE_MANIFEST_SHA256
        return "cache-bad" if path.name == "BTCUSDT.csv.gz" else "cache-ok"

    monkeypatch.setattr("aegis_range_v1.range_v2_discovery._sha256_file", fake_hash)
    with pytest.raises(SourceIntegrityError, match="REGIME_CACHE_DRIFT"):
        verify_authority(repo, tmp_path / "output")
