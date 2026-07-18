import json
from pathlib import Path

import pytest

import aegis.training.benchmark as benchmark
from aegis.freeze import BundleLifecycleState
from aegis.training.benchmark import BenchmarkIntegrityError, build_paired_benchmark, write_benchmark_report


def _historical(root: Path) -> Path:
    path = root / "forward" / "forward_decisions.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "ts": "2026-07-11 20:20:00", "symbol": "ADAUSDT", "tail_score": 0.2,
        "qmae_q90": 0.01, "eqm_score": 0.02, "hypothetical_action": "CANDIDATE_SHORT",
    }) + "\n", encoding="utf-8")
    return path


def _current(feature_timestamp: str = "2026-07-11T20:20:00Z") -> tuple[dict, ...]:
    return ({
        "timestamp": "2026-07-11T20:20:00Z", "feature_timestamp": feature_timestamp,
        "symbol": "ADAUSDT", "side": "SHORT", "tail_score": 0.1,
        "qmae_q90": 0.02, "eqm_score": 0.03, "hypothetical_action": "NO_TRADE",
    },)


def test_phase_f_pairs_exact_timestamp_symbol_without_writing_source(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "aegis_gen2"
    monkeypatch.setattr(benchmark, "GEN2_ROOT", root.resolve())
    source = _historical(root)
    before = source.read_bytes()
    report = build_paired_benchmark(
        current_rows=_current(), gen2_decisions_path=source,
        bundle_id="candidate-1", bundle_state=BundleLifecycleState.CANDIDATE,
    )
    assert report.matched_rows == 1 and report.new_no_trade_rate == 1.0
    assert report.source_hash_before == report.source_hash_after
    assert source.read_bytes() == before
    output = write_benchmark_report(report, tmp_path / "reports" / "report.json")
    assert output.is_file()
    with pytest.raises(BenchmarkIntegrityError, match="cannot be written"):
        write_benchmark_report(report, root / "report.json")


def test_phase_f_rejects_non_candidate_future_features_and_duplicates(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "aegis_gen2"
    monkeypatch.setattr(benchmark, "GEN2_ROOT", root.resolve())
    source = _historical(root)
    with pytest.raises(BenchmarkIntegrityError, match="CANDIDATE"):
        build_paired_benchmark(
            current_rows=_current(), gen2_decisions_path=source,
            bundle_id="experimental", bundle_state=BundleLifecycleState.EXPERIMENTAL,
        )
    with pytest.raises(BenchmarkIntegrityError, match="future"):
        build_paired_benchmark(
            current_rows=_current("2026-07-11T20:25:00Z"), gen2_decisions_path=source,
            bundle_id="candidate-1", bundle_state=BundleLifecycleState.CANDIDATE,
        )
    with pytest.raises(BenchmarkIntegrityError, match="duplicate"):
        build_paired_benchmark(
            current_rows=_current() * 2, gen2_decisions_path=source,
            bundle_id="candidate-1", bundle_state=BundleLifecycleState.CANDIDATE,
        )
