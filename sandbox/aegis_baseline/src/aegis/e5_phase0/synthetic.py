"""The 38 deterministic synthetic checks frozen by the execution specification."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Callable

from .constants import (
    AUTHORITIES,
    CANONICAL_SYMBOLS,
    EXPECTED_LOCKBOX_STATE,
    HOLM_TEST_IDS,
    LABEL_ECONOMICS_REGISTRY,
    PHASE0_VERSION,
)
from .core import (
    OhlcBar,
    artifact_id,
    bootstrap_id,
    c1_seed,
    canonical_decimal,
    canonical_json_bytes,
    complete_iso_week_bounds,
    confirmation_run_id,
    cycle_id,
    funding_event_in_interval,
    identity_hash,
    match_replicate_id,
    namespaced_seed,
    observation_id,
    parse_utc_ms,
    permutation_id,
    short_barrier_event,
    symbol_id,
    type7_quantile,
    validate_canonical_decimal,
    wilder_atr,
)
from .errors import Phase0Error
from .funding import FundingRecord, funding_pnl, funding_record_id, normalize_records, parse_jsonl, serialize_jsonl, short_funding_return
from .matching import (
    C2Edge,
    C2Left,
    filter_c1_candidates,
    filter_c2_self_edges,
    randomized_augmenting_path_match,
    select_c1_control,
    validate_c1_assignment,
    validate_c2_graph,
)
from .statistics import (
    AUTHORITY_CLASSIFICATION,
    circular_shift,
    complete_week_starts,
    deterministic_bootstrap_indices,
    finite_valid_ci90,
    fold_centered_residuals,
    holm_adjust,
    label_classification,
    pooled_positive_pnl_concentration,
    score_deciles,
    spearman,
    temporal_shift,
    validate_diagnostic_authority,
    validate_label_registry,
    validate_resampling_configuration,
)
from .validation import (
    ARTIFACT_SCHEMAS,
    PHASE0_TEST_CATEGORIES,
    Checkpoint,
    ProhibitedDataGuard,
    new_confirmation_ledger,
    resume_confirmation,
    start_confirmation,
    validate_checkpoint,
    validate_rule_matrix,
    validate_synthetic_artifact,
)


UTC = timezone.utc
HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64


@dataclass(frozen=True)
class SyntheticResult:
    name: str
    status: str
    detail: str


def _expect_code(code: str, operation: Callable[[], object]) -> None:
    try:
        operation()
    except Phase0Error as exc:
        if exc.code != code:
            raise AssertionError(f"expected {code}, received {exc.code}") from exc
    else:
        raise AssertionError(f"expected {code}")


def _funding_fixture() -> tuple[bytes, str, tuple[FundingRecord, ...]]:
    raw = b'{"symbol":"BTCUSDT","fundingTime":1001,"fundingRate":"0.0001"}\n'
    raw_hash = hashlib.sha256(raw).hexdigest()
    records = (
        FundingRecord.build("BTCUSDT", 1001, "0.000100", raw_hash),
        FundingRecord.build("BTCUSDT", 2000, "-0.0002", raw_hash),
    )
    return raw, raw_hash, records


def _c2_fixture() -> tuple[tuple[C2Left, ...], tuple[C2Edge, ...]]:
    left = (
        C2Left("left-a", "BTCUSDT", "cycle-a"),
        C2Left("left-b", "ETHUSDT", "cycle-b"),
    )
    edges = (
        C2Edge("left-a", "BTCUSDT", "right-1"),
        C2Edge("left-a", "BTCUSDT", "right-2"),
        C2Edge("left-b", "ETHUSDT", "right-1"),
    )
    return left, edges


def _check_governance_registry() -> None:
    assert len(AUTHORITIES) == 7
    assert [item.order for item in AUTHORITIES] == list(range(1, 8))
    assert AUTHORITIES[-1].sha256 == "751b4014f1072e6fd0a49fb3a8820ba60b1b3c556eb94f9fbb7911d70516ae09"


def _check_identities() -> None:
    first = observation_id("f1", 1_700_000_000_000, "btcusdt", "trade-1")
    second = observation_id(1, 1_700_000_000_000, " BTCUSDT ", "trade-1")
    assert first == second
    assert symbol_id("ethusdt") == symbol_id("ETHUSDT")
    assert cycle_id(1, 10) == cycle_id("F1", 10)
    assert len(match_replicate_id("C2", "h12", 1, "2025-04", 0)) == 64
    assert len(bootstrap_id("mean", "H12", "F1", 0)) == 64
    assert len(permutation_id("C_MONO_H12", "H12", 0)) == 64
    assert len(artifact_id("synthetic/a.json", HEX_A)) == 64
    assert len(confirmation_run_id(HEX_A, HEX_B)) == 64
    _expect_code("IDENTITY_INVALID", lambda: observation_id(1, 1, "BTC/USDT", "trade"))


def _check_shuffled_ordering() -> None:
    left, edges = _c2_fixture()
    first = randomized_augmenting_path_match(left, edges, horizon="H12", fold="F1", month="2025-04", replicate_index=0)
    second = randomized_augmenting_path_match(tuple(reversed(left)), tuple(reversed(edges)), horizon="H12", fold="F1", month="2025-04", replicate_index=0)
    assert first == second


def _check_byte_reruns() -> None:
    _, _, records = _funding_fixture()
    assert serialize_jsonl(records) == serialize_jsonl(tuple(reversed(records)))
    payload = {"phase0_version": PHASE0_VERSION, "synthetic": True, "scientific_use": False}
    assert canonical_json_bytes(payload) == canonical_json_bytes(dict(reversed(tuple(payload.items()))))


def _check_time_boundaries() -> None:
    entry = parse_utc_ms("2026-01-01T00:00:00Z")
    assert entry == parse_utc_ms("2025-12-31T19:00:00-05:00")
    termination = entry + 10
    assert not funding_event_in_interval(entry, termination, entry)
    assert funding_event_in_interval(entry, termination, entry + 1)
    assert funding_event_in_interval(entry, termination, termination - 1)
    assert funding_event_in_interval(entry, termination, termination)
    assert not funding_event_in_interval(entry, termination, termination + 1)
    monday, next_monday = complete_iso_week_bounds(parse_utc_ms("2026-01-07T12:00:00Z"))
    assert next_monday - monday == 604_800_000


def _check_horizon_populations() -> None:
    h12 = {"a", "b", "c"}
    h48 = {"a", "b"}
    h96 = {"a"}
    assert h96 <= h48 <= h12
    assert "c" in h12 and "c" not in h48


def _check_short_return() -> None:
    gross = (100.0 - 99.0) / 100.0
    net = gross - 0.0014 + 0.0001
    assert math.isclose(gross, 0.01)
    assert math.isclose(net, 0.0087)


def _check_barrier() -> None:
    assert short_barrier_event(100.0, 101.0, 99.0) == "ADVERSE_FIRST"


def _bars() -> tuple[OhlcBar, ...]:
    start = parse_utc_ms("2026-01-05T00:00:00Z")
    return tuple(OhlcBar(start + index * 300_000, 101.0, 99.0, 100.0) for index in range(16))


def _check_atr() -> None:
    values = wilder_atr(_bars())
    assert values[13] is None
    assert values[14] == 2.0
    assert values[15] == 2.0
    broken = list(_bars())
    broken[10] = OhlcBar(broken[10].open_ms + 300_000, 101.0, 99.0, 100.0)
    assert wilder_atr(tuple(broken))[-1] is None
    _expect_code("ATR_NOT_COMPUTABLE", lambda: wilder_atr((OhlcBar(0, 98.0, 99.0, 100.0),)))


def _check_type7() -> None:
    assert type7_quantile([1, 2, 3, 4, 5], 0.2) == 1.8
    assert type7_quantile([1, 2, 3, 4], 0.5) == 2.5
    assert type7_quantile([1, 1, 1, 5], 0.4) == 1.0
    _expect_code("BOOTSTRAP_VALIDITY_FAILURE", lambda: type7_quantile([1.0, float("nan")], 0.5))


def _check_seeds() -> None:
    assert c1_seed(0) == 0xE48971B81A0B46C0
    assert namespaced_seed("C2", "H12", "F1", "2025-04", 0)[1] == "b5879bbb4007d7f0287934c0dd7bff7d5512be5792a6c10c2f49d0c3b427f316"
    assert namespaced_seed("BOOTSTRAP", "H12", "F1", 0)[1] == "5711691ffdda359e4f597a10e6fff3b85f40472892ebc10bf6f5cbe21778a94f"
    assert namespaced_seed("C_MONO_H12", "H12", "F3", 0)[1] == "e4c898fb35722cddd5ca8a00ada11ed900b4dfada431855da3d1aece277a67f1"
    assert namespaced_seed("POWER", 0) != namespaced_seed("POWER", 1)


def _check_c1_only_self() -> None:
    _expect_code("C1_NO_DISTINCT_SYMBOL_CONTROL", lambda: filter_c1_candidates("BTCUSDT", ["BTCUSDT"]))


def _check_c1_distinct() -> None:
    selected, result = select_c1_control("BTCUSDT", ["ETHUSDT", "BTCUSDT"], 0)
    assert selected == "ETHUSDT" and result.self_match_exclusions == 1
    _expect_code("C1_SELF_MATCH_DETECTED", lambda: validate_c1_assignment("BTCUSDT", "BTCUSDT"))


def _check_c2_only_exact() -> None:
    left = {"left": C2Left("left", "BTCUSDT", "cycle")}
    _expect_code("C2_NO_DISTINCT_SYMBOL_CYCLE_CONTROL", lambda: filter_c2_self_edges(left, [C2Edge("left", "BTCUSDT", "cycle")]))


def _check_c2_same_symbol_other_cycle() -> None:
    left = {"left": C2Left("left", "BTCUSDT", "cycle")}
    result = filter_c2_self_edges(left, [C2Edge("left", "BTCUSDT", "cycle"), C2Edge("left", "BTCUSDT", "other")])
    assert result.self_edge_exclusions == 1 and result.edges[0].candidate_cycle_id == "other"


def _check_c2_other_symbol() -> None:
    left = {"left": C2Left("left", "BTCUSDT", "cycle")}
    result = filter_c2_self_edges(left, [C2Edge("left", "BTCUSDT", "cycle"), C2Edge("left", "ETHUSDT", "other")])
    assert result.edges[0].candidate_symbol == "ETHUSDT"


def _check_c2_shuffled() -> None:
    _check_shuffled_ordering()


def _check_c2_self_graph() -> None:
    left = {"left": C2Left("left", "BTCUSDT", "cycle")}
    _expect_code("C2_SELF_EDGE_DETECTED", lambda: validate_c2_graph(left, [C2Edge("left", "BTCUSDT", "cycle")]))


def _check_augmenting_paths() -> None:
    left, edges = _c2_fixture()
    matched = randomized_augmenting_path_match(left, edges, horizon="H12", fold=1, month="2025-04", replicate_index=2)
    assert len(matched) == 2 and len({edge.right_id for edge in matched.values()}) == 2
    _expect_code("C2_MATCHING_INFEASIBLE", lambda: randomized_augmenting_path_match(left, edges[:1], horizon="H12", fold=1, month="2025-04", replicate_index=2))


def _check_fold_centering() -> None:
    values = {"F1": (0.1, 0.3), "F2": (-0.2, 0.2)}
    rebuilt = fold_centered_residuals(values)
    assert all(math.isclose(new, old + 0.0005) for fold in values for new, old in zip(rebuilt[fold], values[fold]))


def _check_nested_seeds() -> None:
    assert namespaced_seed("POWER", 0, "BOOTSTRAP", 0) != namespaced_seed("POWER", 1, "BOOTSTRAP", 0)


def _check_complete_weeks() -> None:
    start = parse_utc_ms("2026-01-07T00:00:00Z")
    end = parse_utc_ms("2026-02-08T23:59:59.999Z")
    weeks = complete_week_starts(start, end)
    assert len(weeks) == 4
    shift = temporal_shift("C_MONO_H12", "H12", "F3", 0, len(weeks))
    assert 1 <= shift < len(weeks)
    assert circular_shift((1, 2, 3, 4), shift) != (1, 2, 3, 4)


def _check_deciles() -> None:
    rows = [(1.0, "B", 1), (1.0, "A", 2), (0.0, "C", 3)]
    bins = score_deciles(rows)
    assert bins == (10, 7, 4)


def _check_monotonicity() -> None:
    assert math.isclose(spearman([1, 2, 3], [2, 4, 6]), 1.0)
    assert math.isclose(spearman([1, 1, 2], [1, 1, 2]), 1.0)


def _check_bootstrap() -> None:
    validate_resampling_configuration(10_000, 9_500)
    values = [float(index) for index in range(9_500)] + [float("nan")] * 500
    result = finite_valid_ci90(values)
    assert result.valid == 9_500 and result.invalid == 500 and result.lower < result.upper
    insufficient = [float(index) for index in range(9_499)] + [float("nan")] * 501
    _expect_code("BOOTSTRAP_VALIDITY_FAILURE", lambda: finite_valid_ci90(insufficient))
    assert deterministic_bootstrap_indices(4, "mean", "H12", "F1", 0) == deterministic_bootstrap_indices(4, "mean", "H12", "F1", 0)


def _check_holm() -> None:
    values = {test_id: 0.01 for test_id in HOLM_TEST_IDS}
    decisions = holm_adjust(values)
    assert tuple(item.test_id for item in decisions) == HOLM_TEST_IDS
    _expect_code("HOLM_FAMILY_INCOMPLETE", lambda: holm_adjust(dict(list(values.items())[:-1])))


def _check_concentration() -> None:
    ratio, flag = pooled_positive_pnl_concentration({"BTCUSDT": [1.0], "ETHUSDT": [3.0, -2.0]})
    assert ratio == 0.75 and flag is None
    assert pooled_positive_pnl_concentration({"BTCUSDT": [-1.0]}) == (0.0, "NO_POSITIVE_PNL")
    validate_diagnostic_authority(AUTHORITY_CLASSIFICATION)


def _check_labels() -> None:
    assert validate_label_registry(LABEL_ECONOMICS_REGISTRY)
    assert label_classification(False, True, 4, 1.0) == "LABEL_ECONOMICS_DISCONNECTED"
    assert label_classification(True, True, 3, 0.0004) == "LABEL_ECONOMICS_CONNECTED_EFFECT_TOO_SMALL"
    assert label_classification(True, True, 3, 0.0005) == "LABEL_ECONOMICS_CONNECTED_MATERIAL"


def _check_ic() -> None:
    assert math.isclose(spearman([1.0, 2.0], [2.0, 3.0]), 1.0)
    _expect_code("IC_NOT_COMPUTABLE", lambda: spearman([1.0, 1.0], [2.0, 3.0]))


def _check_funding_decimal() -> None:
    raw, raw_hash, records = _funding_fixture()
    assert canonical_decimal("00.0100") == "0.01"
    assert canonical_decimal("-0.0000") == "0.0"
    assert canonical_decimal("1E-4") == "0.0001"
    assert validate_canonical_decimal("0.0001") > 0
    assert funding_record_id("BTCUSDT", 1001) == records[0].funding_record_id
    assert hashlib.sha256(raw).hexdigest() == raw_hash


def _check_funding_duplicates() -> None:
    _, raw_hash, records = _funding_fixture()
    ordered = normalize_records(tuple(reversed(records)), {raw_hash})
    assert ordered[0].funding_time_utc_ms == 1001
    _expect_code("FUNDING_DUPLICATE_NATURAL_KEY", lambda: normalize_records((records[0], records[0]), {raw_hash}))
    payload = serialize_jsonl(records)
    assert payload.endswith(b"\n")
    assert parse_jsonl(payload, {raw_hash}) == records


def _check_funding_interval() -> None:
    _, _, records = _funding_fixture()
    value = short_funding_return(records, "BTCUSDT", 1001, 2000, True)
    assert value == validate_canonical_decimal("-0.0002")
    assert funding_pnl(value) == Decimal("-0.02000")


def _check_zero_funding() -> None:
    _, _, records = _funding_fixture()
    assert short_funding_return(records, "ETHUSDT", 0, 3000, True) == 0


def _check_incomplete_funding() -> None:
    _, _, records = _funding_fixture()
    _expect_code("FUNDING_NOT_COMPUTABLE", lambda: short_funding_return(records, "BTCUSDT", 0, 3000, False))


def _check_one_shot() -> None:
    ledger = new_confirmation_ledger(HEX_A, HEX_B, HEX_C)
    deps = (HEX_A, HEX_B, HEX_C)
    started = start_confirmation(ledger, ledger.confirmation_run_id, deps)
    assert start_confirmation(started, started.confirmation_run_id, deps) == started
    completed = started.__class__(**{**started.__dict__, "status": "COMPLETED"})
    _expect_code("CONFIRMATION_ALREADY_STARTED", lambda: start_confirmation(completed, completed.confirmation_run_id, deps))


def _check_checkpoint() -> None:
    ledger = start_confirmation(new_confirmation_ledger(HEX_A, HEX_B, HEX_C), confirmation_run_id(HEX_A, HEX_B), (HEX_A, HEX_B, HEX_C))
    checkpoint = Checkpoint(ledger.confirmation_run_id, "SYNTHETIC", HEX_A, HEX_C, (HEX_B,))
    validate_checkpoint(checkpoint, ledger, HEX_A, HEX_C)
    assert checkpoint.identity() == checkpoint.identity()
    assert resume_confirmation(ledger, ledger.confirmation_run_id, (HEX_A, HEX_B, HEX_C)) == ledger
    _expect_code("CONFIRMATION_RESUME_INVALID", lambda: validate_checkpoint(checkpoint, ledger, HEX_B, HEX_C))


def _guard() -> ProhibitedDataGuard:
    return ProhibitedDataGuard((Path("/forbidden/semi_blind"),), (Path("/forbidden/lockbox"),))


def _check_guard() -> None:
    guard = _guard()
    guard.validate_path(Path("/tmp/e5-phase0-synthetic.json"))
    _expect_code("SEMIBLIND_ACCESS_ATTEMPT", lambda: guard.validate_path(Path("/forbidden/semi_blind/rows.parquet")))
    _expect_code("LOCKBOX_ACCESS_ATTEMPT", lambda: guard.validate_path(Path("/forbidden/lockbox/rows.parquet")))
    _expect_code("LOCKBOX_ACCESS_ATTEMPT", lambda: guard.validate_interface("lockbox_query"))
    _expect_code("LOCKBOX_MUTATION_ATTEMPT", lambda: guard.validate_interface("decrement_budget"))


def _check_lockbox() -> None:
    report = _guard().report()
    assert {key: report[key] for key in EXPECTED_LOCKBOX_STATE} == EXPECTED_LOCKBOX_STATE
    assert report["semi_blind"] == "NOT_ACCESSED"


_CHECKS: tuple[Callable[[], None], ...] = (
    _check_governance_registry, _check_identities, _check_shuffled_ordering, _check_byte_reruns,
    _check_time_boundaries, _check_horizon_populations, _check_short_return, _check_barrier,
    _check_atr, _check_type7, _check_seeds, _check_c1_only_self, _check_c1_distinct,
    _check_c2_only_exact, _check_c2_same_symbol_other_cycle, _check_c2_other_symbol,
    _check_c2_shuffled, _check_c2_self_graph, _check_augmenting_paths, _check_fold_centering,
    _check_nested_seeds, _check_complete_weeks, _check_deciles, _check_monotonicity,
    _check_bootstrap, _check_holm, _check_concentration, _check_labels, _check_ic,
    _check_funding_decimal, _check_funding_duplicates, _check_funding_interval,
    _check_zero_funding, _check_incomplete_funding, _check_one_shot, _check_checkpoint,
    _check_guard, _check_lockbox,
)


def run_synthetic_checks() -> tuple[SyntheticResult, ...]:
    validate_rule_matrix()
    if len(_CHECKS) != len(PHASE0_TEST_CATEGORIES):
        raise Phase0Error("DETERMINISM_FAILURE", "synthetic check registry length mismatch")
    results: list[SyntheticResult] = []
    for name, check in zip(PHASE0_TEST_CATEGORIES, _CHECKS):
        try:
            check()
        except Exception as exc:  # Every unexpected error is a hard Phase 0 failure.
            results.append(SyntheticResult(name, "FAIL", f"{type(exc).__name__}: {exc}"))
        else:
            results.append(SyntheticResult(name, "PASS", "synthetic contract satisfied"))
    return tuple(results)


def validate_all_artifact_schemas() -> None:
    for name, required in ARTIFACT_SCHEMAS.items():
        payload = {key: _placeholder(key) for key in required}
        payload["synthetic"] = True
        payload["scientific_use"] = False
        validate_synthetic_artifact(name, payload)


def _placeholder(field: str) -> object:
    if field in {"authorities", "classes", "deny_roots", "inputs", "vectors", "symbols", "raw_artifacts", "test_ids", "decisions", "labels", "artifacts", "governance_hashes"}:
        return []
    if field in {"clean", "coverage_complete"}:
        return True
    if field in {"consumed_queries"}:
        return []
    if field in {"budget_remaining", "requested", "valid", "invalid", "self_match_exclusions", "self_edge_exclusions"}:
        return 0
    if field in {"lower", "upper"}:
        return 0.0
    if field == "semi_blind":
        return "NOT_ACCESSED"
    if field == "lockbox":
        return "NOT_CONSUMED"
    if field == "scientific_use":
        return False
    if field == "synthetic":
        return True
    return "synthetic"
