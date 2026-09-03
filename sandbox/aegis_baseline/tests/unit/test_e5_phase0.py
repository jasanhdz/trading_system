"""Synthetic-only contract tests for E5 Phase 0."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from aegis.e5_phase0.constants import (
    AUTHORITIES,
    CANONICAL_SYMBOLS,
    EXPECTED_LOCKBOX_STATE,
    HOLM_TEST_IDS,
    LABEL_ECONOMICS_REGISTRY,
    PHASE0_VERSION,
)
from aegis.e5_phase0.core import (
    OhlcBar,
    c1_seed,
    canonical_decimal,
    canonical_json_bytes,
    funding_event_in_interval,
    namespaced_seed,
    observation_id,
    parse_utc_ms,
    short_barrier_event,
    type7_quantile,
    wilder_atr,
)
from aegis.e5_phase0.errors import Phase0Error
from aegis.e5_phase0.funding import FundingRecord, funding_manifest, funding_pnl, normalize_records, parse_jsonl, serialize_jsonl, short_funding_return
from aegis.e5_phase0.matching import (
    C2Edge,
    C2Left,
    filter_c1_candidates,
    filter_c2_self_edges,
    randomized_augmenting_path_match,
    validate_c1_assignment,
    validate_c2_graph,
)
from aegis.e5_phase0.orchestrator import build_phase0_report, write_phase0_report
from aegis.e5_phase0.statistics import (
    AUTHORITY_CLASSIFICATION,
    finite_valid_ci90,
    fold_centered_residuals,
    holm_adjust,
    label_classification,
    pooled_positive_pnl_concentration,
    score_deciles,
    temporal_shift,
    validate_diagnostic_authority,
    validate_label_registry,
    validate_resampling_configuration,
)
from aegis.e5_phase0.synthetic import run_synthetic_checks
from aegis.e5_phase0.validation import (
    ARTIFACT_SCHEMAS,
    PHASE0_TEST_CATEGORIES,
    Checkpoint,
    ProhibitedDataGuard,
    RULE_IMPLEMENTATION_MATRIX,
    new_confirmation_ledger,
    resume_confirmation,
    start_confirmation,
    validate_checkpoint,
    validate_rule_matrix,
    verify_governance,
)


ROOT = Path(__file__).resolve().parents[2]
UTC = timezone.utc
HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64


@pytest.fixture(scope="module")
def synthetic_results():
    return {result.name: result for result in run_synthetic_checks()}


@pytest.mark.parametrize("category", PHASE0_TEST_CATEGORIES)
def test_all_38_specification_categories_pass(synthetic_results, category: str) -> None:
    assert synthetic_results[category].status == "PASS", synthetic_results[category].detail


def test_authority_files_and_commits_validate_without_reading_scientific_rows() -> None:
    result = verify_governance(require_clean=False)
    assert result.hashes == {authority.name: authority.sha256 for authority in AUTHORITIES}
    assert result.commits == {authority.name: authority.commit for authority in AUTHORITIES}


def test_rule_implementation_matrix_covers_every_specification_rule() -> None:
    validate_rule_matrix()
    assert tuple(sorted(RULE_IMPLEMENTATION_MATRIX)) == tuple(f"E5-R{number:03d}" for number in range(1, 54))
    assert all(item.module and item.test and item.artifact and item.failure_code for item in RULE_IMPLEMENTATION_MATRIX.values())


def test_canonical_identity_normalizes_equivalent_inputs() -> None:
    assert observation_id(1, 1000, "btcusdt", "trade") == observation_id("F1", 1000, " BTCUSDT ", "trade")
    with pytest.raises(Phase0Error, match="IDENTITY_INVALID"):
        observation_id(1, 1000, "BTC/USDT", "trade")


def test_time_and_funding_interval_are_utc_and_left_open_right_closed() -> None:
    entry = parse_utc_ms("2026-01-01T00:00:00Z")
    assert entry == parse_utc_ms(datetime(2026, 1, 1, tzinfo=UTC))
    assert not funding_event_in_interval(entry, entry + 2, entry)
    assert funding_event_in_interval(entry, entry + 2, entry + 1)
    assert funding_event_in_interval(entry, entry + 2, entry + 2)
    assert not funding_event_in_interval(entry, entry + 2, entry + 3)


@pytest.mark.parametrize(
    ("raw", "canonical"),
    (("0", "0.0"), ("00.0100", "0.01"), ("-0.0000", "0.0"), ("1E-4", "0.0001")),
)
def test_exact_decimal_canonicalization(raw: str, canonical: str) -> None:
    assert canonical_decimal(raw) == canonical


def test_type7_and_wilder_atr_are_exact_contracts() -> None:
    assert type7_quantile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    bars = tuple(OhlcBar(index * 300_000, 101.0, 99.0, 100.0) for index in range(16))
    atr = wilder_atr(bars)
    assert atr[13] is None and atr[14:] == (2.0, 2.0)
    with pytest.raises(Phase0Error, match="UNAUTHORIZED_SCIENTIFIC_CHOICE"):
        wilder_atr(bars, period=13)


def test_seed_vectors_match_execution_specification() -> None:
    assert c1_seed(0) == 0xE48971B81A0B46C0
    assert namespaced_seed("C2", "H12", "F1", "2025-04", 0)[1] == "b5879bbb4007d7f0287934c0dd7bff7d5512be5792a6c10c2f49d0c3b427f316"
    assert namespaced_seed("BOOTSTRAP", "H12", "F1", 0)[1] == "5711691ffdda359e4f597a10e6fff3b85f40472892ebc10bf6f5cbe21778a94f"


def test_c1_self_match_is_removed_before_selection_and_cannot_be_final() -> None:
    result = filter_c1_candidates("BTCUSDT", ["ETHUSDT", "BTCUSDT"])
    assert result.candidates == ("ETHUSDT",) and result.self_match_exclusions == 1
    with pytest.raises(Phase0Error, match="C1_NO_DISTINCT_SYMBOL_CONTROL"):
        filter_c1_candidates("BTCUSDT", ["BTCUSDT"])
    with pytest.raises(Phase0Error, match="C1_SELF_MATCH_DETECTED"):
        validate_c1_assignment("BTCUSDT", "BTCUSDT")


def test_c2_self_edge_filter_preserves_same_symbol_different_cycle() -> None:
    left = {"left": C2Left("left", "BTCUSDT", "cycle-a")}
    filtered = filter_c2_self_edges(left, (
        C2Edge("left", "BTCUSDT", "cycle-a"),
        C2Edge("left", "BTCUSDT", "cycle-b"),
        C2Edge("left", "ETHUSDT", "cycle-c"),
    ))
    assert filtered.self_edge_exclusions == 1
    assert {(edge.candidate_symbol, edge.candidate_cycle_id) for edge in filtered.edges} == {
        ("BTCUSDT", "cycle-b"), ("ETHUSDT", "cycle-c"),
    }
    with pytest.raises(Phase0Error, match="C2_SELF_EDGE_DETECTED"):
        validate_c2_graph(left, [C2Edge("left", "BTCUSDT", "cycle-a")])


def test_augmenting_path_is_order_independent_and_enforces_no_reuse() -> None:
    left = (C2Left("a", "BTCUSDT", "x"), C2Left("b", "ETHUSDT", "y"))
    edges = (C2Edge("a", "BTCUSDT", "r1"), C2Edge("a", "BTCUSDT", "r2"), C2Edge("b", "ETHUSDT", "r1"))
    kwargs = {"horizon": "H12", "fold": "F1", "month": "2025-04", "replicate_index": 0}
    first = randomized_augmenting_path_match(left, edges, **kwargs)
    second = randomized_augmenting_path_match(tuple(reversed(left)), tuple(reversed(edges)), **kwargs)
    assert first == second
    assert len({edge.right_id for edge in first.values()}) == 2


def test_funding_jsonl_duplicate_coverage_and_short_sign() -> None:
    raw = b"synthetic-binance-response"
    raw_hash = hashlib.sha256(raw).hexdigest()
    records = (
        FundingRecord.build("BTCUSDT", 1001, "0.0001", raw_hash),
        FundingRecord.build("BTCUSDT", 2000, "-0.0002", raw_hash),
    )
    payload = serialize_jsonl(records)
    assert payload == serialize_jsonl(tuple(reversed(records)))
    assert parse_jsonl(payload, {raw_hash}) == records
    total_return = short_funding_return(records, "BTCUSDT", 1000, 2000, True)
    assert total_return == Decimal("-0.0001")
    assert funding_pnl(total_return) == Decimal("-0.01000")
    with pytest.raises(Phase0Error, match="FUNDING_DUPLICATE_NATURAL_KEY"):
        normalize_records((records[0], records[0]), {raw_hash})
    coverage = {symbol: (0, 3000, True) for symbol in CANONICAL_SYMBOLS}
    manifest = funding_manifest(records, {"page-0000": raw}, coverage, 0, 3000)
    assert manifest["synthetic"] is True and manifest["scientific_use"] is False


def test_resampling_configuration_and_finite_valid_threshold() -> None:
    validate_resampling_configuration(10_000, 9_500)
    result = finite_valid_ci90([float(index) for index in range(9_500)] + [float("nan")] * 500)
    assert result.valid == 9_500
    with pytest.raises(Phase0Error, match="BOOTSTRAP_VALIDITY_FAILURE"):
        finite_valid_ci90([float(index) for index in range(9_499)] + [float("nan")] * 501)


def test_fold_centering_deciles_and_barrier_precedence() -> None:
    rebuilt = fold_centered_residuals({"F1": (0.1, 0.2)})["F1"]
    assert rebuilt == pytest.approx((0.1005, 0.2005))
    assert score_deciles([(1.0, "B", 1), (1.0, "A", 2), (0.0, "C", 3)]) == (10, 7, 4)
    assert short_barrier_event(100.0, 101.0, 99.0) == "ADVERSE_FIRST"


def test_temporal_permutation_is_nonzero_and_deterministic() -> None:
    first = temporal_shift("C_MONO_H12", "H12", "F3", 0, 5)
    second = temporal_shift("C_MONO_H12", "H12", "F3", 0, 5)
    assert first == second and 1 <= first < 5


def test_holm_registry_is_exact_and_ties_follow_registry_order() -> None:
    raw = {test_id: 0.001 for test_id in HOLM_TEST_IDS}
    decisions = holm_adjust(raw)
    assert tuple(item.test_id for item in decisions) == HOLM_TEST_IDS
    with pytest.raises(Phase0Error, match="HOLM_FAMILY_INCOMPLETE"):
        holm_adjust({**raw, "EXTRA": 0.1})


def test_label_concentration_and_ic_authorities_are_not_mutable() -> None:
    validate_label_registry(LABEL_ECONOMICS_REGISTRY)
    validate_diagnostic_authority(AUTHORITY_CLASSIFICATION)
    assert label_classification(False, True, 4, 1.0) == "LABEL_ECONOMICS_DISCONNECTED"
    assert pooled_positive_pnl_concentration({"BTCUSDT": [-1.0]}) == (0.0, "NO_POSITIVE_PNL")


def test_artifact_schema_registry_requires_synthetic_markers() -> None:
    from aegis.e5_phase0.synthetic import validate_all_artifact_schemas

    validate_all_artifact_schemas()
    assert len(ARTIFACT_SCHEMAS) >= 20
    schema_path = ROOT / "src/aegis/e5_phase0/schemas/e5_phase0_artifacts.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["properties"]["synthetic"]["const"] is True
    assert schema["properties"]["scientific_use"]["const"] is False


def test_prohibited_data_guard_rejects_paths_and_interfaces_without_opening() -> None:
    guard = ProhibitedDataGuard((Path("/deny/semi"),), (Path("/deny/lockbox"),))
    with pytest.raises(Phase0Error, match="SEMIBLIND_ACCESS_ATTEMPT"):
        guard.validate_path(Path("/deny/semi/rows.parquet"))
    with pytest.raises(Phase0Error, match="LOCKBOX_ACCESS_ATTEMPT"):
        guard.validate_interface("lockbox_query")
    with pytest.raises(Phase0Error, match="LOCKBOX_MUTATION_ATTEMPT"):
        guard.validate_interface("decrement_budget")
    assert {key: guard.report()[key] for key in EXPECTED_LOCKBOX_STATE} == EXPECTED_LOCKBOX_STATE


def test_confirmation_ledger_and_checkpoint_allow_only_same_run_resume() -> None:
    ledger = new_confirmation_ledger(HEX_A, HEX_B, HEX_C)
    dependencies = (HEX_A, HEX_B, HEX_C)
    started = start_confirmation(ledger, ledger.confirmation_run_id, dependencies)
    assert resume_confirmation(started, started.confirmation_run_id, dependencies) == started
    checkpoint = Checkpoint(started.confirmation_run_id, "SYNTHETIC", HEX_A, HEX_C, (HEX_B,))
    validate_checkpoint(checkpoint, started, HEX_A, HEX_C)
    with pytest.raises(Phase0Error, match="CONFIRMATION_DEPENDENCY_MISMATCH"):
        start_confirmation(started, started.confirmation_run_id, (HEX_B, HEX_B, HEX_C))


def test_end_to_end_phase0_report_is_deterministic_and_non_scientific(tmp_path: Path) -> None:
    first = build_phase0_report(require_clean=False)
    second = build_phase0_report(require_clean=False)
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert first["phase0_version"] == PHASE0_VERSION
    assert first["test_count"] == 38 and first["passed_count"] == 38
    assert first["final_status"] == "PASS"
    assert first["scientific_rows_inspected"] == 0
    assert first["scientific_datasets_created"] == 0
    report_path, hash_path = write_phase0_report(first, tmp_path / "e5_phase0_report.json")
    assert report_path.is_file() and hash_path.is_file()
    assert canonical_json_bytes(first) == report_path.read_bytes()


def test_end_to_end_phase0_uses_no_network(monkeypatch) -> None:
    import socket

    def reject_network(*args, **kwargs):
        raise AssertionError("Phase 0 attempted network access")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    report = build_phase0_report(require_clean=False)
    assert report["network_calls"] == 0


def test_fixture_is_explicitly_synthetic_and_non_scientific() -> None:
    fixture = json.loads((ROOT / "tests/fixtures/e5_phase0/synthetic_phase0_fixture.json").read_text(encoding="utf-8"))
    assert fixture["synthetic"] is True
    assert fixture["scientific_use"] is False
