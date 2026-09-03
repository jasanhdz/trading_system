from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "sandbox/manual_visual_decision_replication_v1/src"))

from mvdr_v1.m1 import Bar, _aggregate, _side_features, apply_labels, audit_bars, build_manual_manifest, generate_universe, visual_frame


def bars(symbol_scale: float = 1.0) -> list[Bar]:
    start = datetime(2026, 8, 24, 20, tzinfo=timezone.utc)
    return [Bar(start + timedelta(minutes=i), symbol_scale + i * 0.0001, symbol_scale + i * 0.0001 + 0.001, symbol_scale + i * 0.0001 - 0.001, symbol_scale + i * 0.0001 + 0.0002, 100 + i, 50 + i / 2) for i in range(240)]


def test_trade_manifest_identity_is_frozen() -> None:
    manifest = build_manual_manifest(ROOT / "backtest_results/real_trade_analysis.json")
    assert len(manifest) == 9
    assert manifest[0]["entry_at_utc"] == "2026-08-25T00:05:37Z"
    assert manifest[-1]["side"] == "SHORT"


def test_aggregation_partial_bucket_uses_only_supplied_minutes() -> None:
    source = bars()[:7]
    result = _aggregate(source, 5)
    assert len(result) == 2
    assert result[-1].close == source[-1].close
    assert result[-1].volume == source[-2].volume + source[-1].volume


def test_visual_frame_never_uses_current_incomplete_minute() -> None:
    source = bars()
    decision = source[180].open_at + timedelta(seconds=30)
    frame = visual_frame(decision, source, bars(100.0))
    assert frame["latest_completed_1m_open_at"] == source[179].open_at.isoformat(timespec="seconds").replace("+00:00", "Z")


def test_market_audit_detects_gap() -> None:
    source = bars()[:3] + bars()[4:5]
    audit = audit_bars("SUIUSDT", source, source[0].open_at, source[0].open_at + timedelta(minutes=5))
    assert audit["gaps"] == ["2026-08-24T20:03:00Z"]


def test_side_transform_reverses_directional_displacement() -> None:
    features = {"momentum.sui_3m.n3.displacement": 0.1, "sr.swing.support_distance": 0.1, "sr.cluster.support_distance": 0.2, "sr.mtf_extrema.support_distance": 0.3, "sr.swing.resistance_distance": 0.4, "sr.cluster.resistance_distance": 0.5, "sr.mtf_extrema.resistance_distance": 0.6}
    assert _side_features(features, "LONG")["momentum.sui_3m.n3.displacement"] == 0.1
    assert _side_features(features, "SHORT")["momentum.sui_3m.n3.displacement"] == -0.1


def test_labeling_blackouts_manual_and_other_strategy_windows() -> None:
    frame = visual_frame(bars()[180].open_at, bars(), bars(100.0))
    universe = generate_universe([frame])
    entry = frame["decision_at_utc"]
    manifest = [{"entry_at_utc": entry, "side": "LONG", "manual_trade_id": "T1"}]
    labeled = apply_labels(universe, manifest, [datetime.fromisoformat(entry.replace("Z", "+00:00"))])
    assert labeled[0]["selected"]
    assert all(row["manual_entry_blackout"] and row["other_strategy_exclusion"] for row in labeled)
    assert not any(row["hard_negative_eligible"] for row in labeled)
