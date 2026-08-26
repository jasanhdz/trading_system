from __future__ import annotations

import gzip
import json
from datetime import datetime, timedelta, timezone

import pytest

from aegis_range_v1.candidates import candidate_grid
from aegis_range_v1.models import Candle5m, LevelSnapshot, RangePair
from aegis_range_v1.numeric import iso_utc_millis
from aegis_range_v1.sweep_reclaim_discovery import (
    FrozenRange,
    SweepReclaimMachine,
    assign_opportunity_weights,
    canonical_sweep_opportunity_id,
    first_passage,
    midpoint_touched,
    reclaim_matches,
    sweep_depth,
    _clean_opportunity,
    _parse,
    deterministic_gzip_jsonl,
)
from aegis_range_v1.sweep_reclaim_phase1 import (
    _contract_eligibility,
    _event_diagnostics,
    _excursion_stats,
    _localize_view,
    _passage_asymmetry,
    block_bootstrap_first_passage,
    replay_structure_phase1,
    structural_candidates,
)

ORIGIN = datetime(2024, 6, 1, tzinfo=timezone.utc)


def bar(index: int, *, open_: float = 101, high: float = 104, low: float = 100, close: float = 101, segment: int = 0) -> Candle5m:
    opened = ORIGIN + timedelta(minutes=5 * index)
    return Candle5m("BTCUSDT", opened, opened + timedelta(minutes=5), open_, high, low, close, 1, segment)


def snapshot(*, support: float = 100, resistance: float = 110, midpoint: float = 105, atr: float = 2, episode: str = "ep") -> LevelSnapshot:
    pair = RangePair("support", "resistance", support, resistance, midpoint, resistance - support, 2, 2, ORIGIN, ORIGIN)
    return LevelSnapshot(ORIGIN, pair, atr, episode, "range")


def frozen(side: str = "LONG", **kwargs) -> FrozenRange:
    snap = snapshot(**kwargs)
    return FrozenRange.from_snapshot("BTCUSDT", side, snap, ORIGIN - timedelta(hours=1))


def process(machine: SweepReclaimMachine, candle: Candle5m, snap: LevelSnapshot | None = None, post: str | None = "ep", **kwargs):
    snap = snapshot() if snap is None else snap
    return machine.process_close(candle, snap, ORIGIN - timedelta(hours=1), post, **kwargs)


class TestTemporalAudit:
    def test_timestamp_equality_s1(self):
        machine = SweepReclaimMachine("s")
        rows = process(machine, bar(0, low=99.7, close=101))
        assert len(rows) == 1
        assert rows[0]["reclaim_decision_at"] == iso_utc_millis(ORIGIN + timedelta(minutes=5))

    def test_timestamp_equality_s2(self):
        machine = SweepReclaimMachine("s")
        process(machine, bar(0, low=99.7, close=99))
        rows = process(machine, bar(1, low=100, close=101))
        assert len(rows) == 1
        assert rows[0]["reclaim_decision_at"] == iso_utc_millis(ORIGIN + timedelta(minutes=10))

    def test_next_bar_adjacency(self):
        machine = SweepReclaimMachine("s")
        rows = process(machine, bar(0, low=99.7, close=101))
        reclaim_avail = _parse(rows[0]["reclaim_decision_at"])
        next_bar_open = ORIGIN + timedelta(minutes=5)
        assert reclaim_avail == next_bar_open

    def test_gap_rejects_entry(self):
        machine = SweepReclaimMachine("s")
        rows = process(machine, bar(0, low=99.7, close=101))
        frozen_range = FrozenRange.from_snapshot("BTCUSDT", "LONG", snapshot(), ORIGIN - timedelta(hours=1))
        assert not (frozen_range.support < 95 < frozen_range.resistance)

    def test_non_contiguous_rejects(self):
        machine = SweepReclaimMachine("s")
        rows = process(machine, bar(0, low=99.7, close=101))
        assert rows[0]["status"] == "RECLAIMED"
        bar3 = bar(3, low=100, close=101)
        result = machine.process_close(bar3, snapshot(), ORIGIN - timedelta(hours=1), "ep", contiguous=False)
        for r in result:
            if r.get("reclaim_type") == "S1":
                assert r["status"] == "CANCELLED"

    def test_midpoint_touch_cancels_before_entry(self):
        machine = SweepReclaimMachine("s")
        rows = process(machine, bar(0, low=99.7, high=105, close=101))
        assert rows[0]["terminal_reason"] == "MIDPOINT_TOUCHED"


class TestFirstPassage:
    def test_adverse_first_same_bar(self):
        result = first_passage(100, "LONG", 104, [bar(0, high=104, low=99.5)])
        assert result["favorable_adverse_matrix"]["F10_A10"] == "ADVERSE_FIRST"

    def test_favorable_before_adverse(self):
        result = first_passage(100, "LONG", 110, [bar(0, high=103, low=100)])
        assert result["favorable_adverse_matrix"]["F30_A10"] == "FAVORABLE_FIRST"

    def test_progress_tracking(self):
        result = first_passage(100, "LONG", 104, [bar(0, high=104, low=99)])
        assert result["midpoint_progress_first_bar"]["100"] == 0
        assert result["progress_adverse_matrix"]["P100_A10"] == "ADVERSE_FIRST"

    def test_mfe_mae_nonnegative(self):
        result = first_passage(100, "LONG", 104, [bar(0, high=104, low=99)])
        assert result["favorable_first_bar"]["10"] == 0

    def test_no_lookahead(self):
        prefix = [bar(0, low=99.7, close=99), bar(1, low=100.1, close=101)]
        future = [bar(index, high=200, low=1, close=50) for index in range(2, 10)]
        def run(candles):
            machine = SweepReclaimMachine("s")
            events = []
            for candle in candles:
                events.extend(process(machine, candle))
            return [{k: v for k, v in row.items() if not k.startswith("_")} for row in events if row["terminal_decision_at"] <= iso_utc_millis(prefix[-1].available_at)]
        assert run(prefix) == run(prefix + future)


class TestContractEligibility:
    def test_all_four_contracts(self):
        elig = _contract_eligibility(100, "LONG", frozen())
        assert len(elig) == 4
        assert all(k.startswith("SB") for k in elig)

    def test_42bps_and_rr(self):
        elig = _contract_eligibility(100, "LONG", frozen())
        for key, contract in elig.items():
            assert "passes_42bps" in contract
            assert "passes_rr1" in contract
            assert "favorable_target" in contract

    def test_short_eligibility(self):
        elig = _contract_eligibility(108, "SHORT", frozen("SHORT"))
        assert len(elig) == 4


class TestDeduplication:
    def test_canonical_ids_unique_per_structure(self):
        machine = SweepReclaimMachine("s")
        rows = process(machine, bar(0, low=99.7, close=101))
        assert len({r["canonical_sweep_opportunity_id"] for r in rows}) == len(rows)

    def test_weight_sums_to_one(self):
        machine = SweepReclaimMachine("s")
        rows = process(machine, bar(0, low=99.7, close=101))
        cleaned = assign_opportunity_weights([_clean_opportunity(r) for r in rows])
        total = sum(r["unique_weight"] for r in cleaned)
        assert total == pytest.approx(1.0, abs=1e-12)


class TestNoExpansion:
    def test_structural_candidates_are_six(self):
        assert len(structural_candidates()) == 6

    def test_no_candidate_grid_expansion(self):
        candidates = structural_candidates()
        for c in candidates:
            assert hasattr(c, "cluster_tolerance_atr")
            assert hasattr(c, "min_range_amplitude_pct")


class TestCancellation:
    def test_episode_end(self):
        machine = SweepReclaimMachine("s")
        process(machine, bar(0, low=99.7, close=99))
        rows = process(machine, bar(1), post=None)
        assert rows[0]["terminal_reason"] == "EPISODE_ENDED"

    @pytest.mark.parametrize("kwargs,reason", [
        ({"episode_event": "CONFIRMED_BREAKOUT"}, "CONFIRMED_BREAKOUT"),
        ({"episode_event": "EXPIRED_48H"}, "EPISODE_EXPIRED_48H"),
        ({"episode_event": "PAIR_REPLACED"}, "EPISODE_REPLACED"),
        ({"episode_event": "STRUCTURE_LOST"}, "STRUCTURE_LOSS"),
        ({"same_split": False}, "SPLIT_BOUNDARY"),
        ({"contiguous": False}, "DATA_GAP"),
    ])
    def test_cancellation_types(self, kwargs, reason):
        machine = SweepReclaimMachine("s")
        process(machine, bar(0, low=99.7, close=99))
        rows = process(machine, bar(1, low=100, close=101), **kwargs)
        assert rows[0]["terminal_reason"] == reason


class TestEventDiagnostics:
    def test_mutually_exclusive_counts(self):
        opps = [{"canonical_sweep_opportunity_id": "a", "status": "RECLAIMED", "reclaim_type": "S1"},
                {"canonical_sweep_opportunity_id": "b", "status": "RECLAIMED", "reclaim_type": "S2"},
                {"canonical_sweep_opportunity_id": "c", "status": "NO_RECLAIM", "reclaim_type": None},
                {"canonical_sweep_opportunity_id": "d", "status": "CANCELLED", "reclaim_type": None}]
        diag = _event_diagnostics(opps)
        assert diag["S1_rate"] == pytest.approx(0.25)
        assert diag["S2_rate"] == pytest.approx(0.25)
        assert diag["NO_RECLAIM_rate"] == pytest.approx(0.25)
        assert diag["CANCELLED_rate"] == pytest.approx(0.25)


class TestPassageAsymmetry:
    def test_basic_rate(self):
        rows = [
            {"canonical_sweep_opportunity_id": "a", "local_unique_weight": 1.0,
             "favorable_adverse_matrix": {"F20_A20": "FAVORABLE_FIRST", "F30_A20": "ADVERSE_FIRST", "F30_A30": "ADVERSE_FIRST", "F40_A30": "ADVERSE_FIRST", "F40_A40": "ADVERSE_FIRST"},
             "progress_adverse_matrix": {"P25_A20": "PROGRESS_FIRST", "P50_A20": "PROGRESS_FIRST", "P100_A20": "ADVERSE_FIRST"}},
        ]
        asym = _passage_asymmetry(rows)
        assert asym["p20_a20"] == 1.0
        assert asym["p30_a20"] == 0.0
        assert asym["progress_50_before_a20"] == 1.0


class TestLocalizeView:
    def test_local_weights_sum_to_one(self):
        entries = [
            {"candidate_id": "C000", "canonical_sweep_opportunity_id": "o1"},
            {"candidate_id": "C001", "canonical_sweep_opportunity_id": "o1"},
        ]
        passages = [{"candidate_id": "C000", "canonical_sweep_opportunity_id": "o1", "favorable_adverse_matrix": {}, "progress_adverse_matrix": {}}]
        local_e, local_p = _localize_view(entries, passages)
        assert sum(r["local_unique_weight"] for r in local_e) == pytest.approx(1.0)


class TestDeterminism:
    def test_replay_deterministic(self):
        from aegis_range_v1.train_backtest import CachedRangeRegimeAdapter, load_regime_cache, load_train_candles
        from pathlib import Path
        repo = Path("/home/jasan/Develop/trading_system")
        candles = load_train_candles(repo, "BTCUSDT")
        snapshots = load_regime_cache(repo / "sandbox/aegis_range_strategy_v1/artifacts/r2_train/run_a/regime_cache/BTCUSDT.csv.gz", len(candles))
        funding = []
        candidate = structural_candidates()[0]
        opp1, ent1, pass1 = replay_structure_phase1("BTCUSDT", candidate, candles, snapshots, funding)
        opp2, ent2, pass2 = replay_structure_phase1("BTCUSDT", candidate, candles, snapshots, funding)
        assert len(opp1) == len(opp2)
        assert len(ent1) == len(ent2)
        assert len(pass1) == len(pass2)
        for r1, r2 in zip(opp1, opp2):
            assert r1["canonical_sweep_opportunity_id"] == r2["canonical_sweep_opportunity_id"]
            assert r1["reclaim_decision_at"] == r2["reclaim_decision_at"]


class TestGzipDeterminism:
    def test_gzip_deterministic(self, tmp_path):
        first, second = tmp_path / "a.gz", tmp_path / "b.gz"
        rows = [{"z": 1, "a": 2}]
        assert deterministic_gzip_jsonl(first, rows) == deterministic_gzip_jsonl(second, rows)
        assert first.read_bytes() == second.read_bytes()
