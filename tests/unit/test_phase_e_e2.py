import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from aegis.config import CANONICAL_SYMBOLS
from aegis.data import CanonicalBar
from aegis.training.dataset import build_e2_hourly_short_dataset, load_and_build_e2_hourly_dataset
from aegis.training.phase_e import ProductionScientificBackend
from aegis.training.preregistration import (
    E1_CANONICAL_HASH, E1_PHYSICAL_SHA256, PreregistrationError,
    SharedLockboxAuthority, load_and_validate_preregistration,
)
from aegis.training.run_state import (
    LockboxLeaseRecord, PhaseEErrorCode, PhaseETechnicalError, RunMode,
    SharedWindowLockboxLease,
)
from aegis.utils import Sha256HashProvider, sha256_file


ROOT = Path(__file__).resolve().parents[2]
E1 = ROOT / "config" / "experiments" / "aegis_short_candidate_e1.yaml"
E2 = ROOT / "config" / "experiments" / "aegis_short_candidate_e2.yaml"


def _copy_pair(tmp_path: Path) -> tuple[Path, Path]:
    e1 = tmp_path / E1.name; e2 = tmp_path / E2.name
    e1.write_bytes(E1.read_bytes()); e2.write_bytes(E2.read_bytes())
    return e1, e2


def _sampling() -> dict:
    return {
        "coordinated_symbols_required": 11, "history_bars": 288,
        "horizon_bars": 12, "stride_bars": 12,
        "expected_rows": {
            "first_anchor_utc": "2025-01-02T00:00:00Z",
            "last_dev_anchor_utc": "2025-01-02T01:00:00Z",
            "approximate_maximum_dev_rows": 22,
            "hard_stop_if_valid_rows_below_fraction": 0.90,
        },
    }


def _series() -> dict[str, tuple[CanonicalBar, ...]]:
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = {}
    for symbol_index, symbol in enumerate(CANONICAL_SYMBOLS):
        base = 10.0 + symbol_index
        rows[symbol] = tuple(
            CanonicalBar(
                start + timedelta(minutes=5 * index), base + index * 0.001,
                base + index * 0.001 + 0.01, base + index * 0.001 - 0.01,
                base + index * 0.001 + 0.002, 1000.0 + index,
            ) for index in range(314)
        )
    return rows


def test_e1_is_byte_identical_and_canonically_identical() -> None:
    assert sha256_file(E1) == E1_PHYSICAL_SHA256
    payload, audit = load_and_validate_preregistration(E1, audit_source=False)
    assert audit.content_hash == E1_CANONICAL_HASH
    assert payload["protocol"]["threshold_value"] is None


def test_e1_is_auditable_but_not_productively_executable() -> None:
    load_and_validate_preregistration(E1, audit_source=False)
    with pytest.raises(PreregistrationError, match="PROTOCOL_VERSION_NOT_EXECUTABLE"):
        load_and_validate_preregistration(E1, audit_source=False, require_executable=True)
    with pytest.raises(PhaseETechnicalError, match="PROTOCOL_VERSION_NOT_EXECUTABLE"):
        ProductionScientificBackend().build_dataset(yaml.safe_load(E1.read_text()), RunMode.VALIDATION_RUN)


def test_e2_supersedes_e1_and_is_stably_executable() -> None:
    payload, first = load_and_validate_preregistration(E2, audit_source=False, require_executable=True)
    _, second = load_and_validate_preregistration(E2, audit_source=False, require_executable=True)
    assert payload["supersedes"]["experiment_id"] == "aegis-short-candidate-e1"
    assert first.protocol_version == 2 and first.executable
    assert first.content_hash == second.content_hash


@pytest.mark.parametrize("block", ["source", "models", "econ", "promotion", "publication"])
def test_e2_rejects_any_inherited_scientific_change(tmp_path: Path, block: str) -> None:
    _, path = _copy_pair(tmp_path)
    payload = yaml.safe_load(path.read_text())
    payload[block]["unexpected_change"] = True
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(PreregistrationError, match="PREREGISTRATION_INHERITANCE_MISMATCH"):
        load_and_validate_preregistration(path, audit_source=False)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("sampling", "stride_bars"), 6),
        (("sampling", "closed_final_candles_only"), False),
        (("calibration", "measurement_block"), "CALIBRATION"),
        (("scoring", "fitting"), "ALLOWED"),
        (("threshold_derivation", "executed"), "POST_LOCKBOX"),
    ],
)
def test_e2_scientific_protocol_fields_fail_closed(tmp_path: Path, path: tuple[str, str], value: object) -> None:
    _, e2 = _copy_pair(tmp_path); payload = yaml.safe_load(e2.read_text())
    payload[path[0]][path[1]] = value; e2.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(PreregistrationError):
        load_and_validate_preregistration(e2, audit_source=False)


def test_fold_literal_one_second_drift_is_rejected(tmp_path: Path) -> None:
    _, e2 = _copy_pair(tmp_path); payload = yaml.safe_load(e2.read_text())
    payload["fold_protocol"]["folds"][0]["calibration_end"] = "2025-04-16T00:55:01Z"
    e2.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(PreregistrationError, match="literal dates"):
        load_and_validate_preregistration(e2, audit_source=False)


def test_fold_blocks_are_disjoint_embargoed_and_pre_reserve() -> None:
    payload, _ = load_and_validate_preregistration(E2, audit_source=False)
    reserve = datetime.fromisoformat(payload["refit"]["final_calibration_reserve_start"].replace("Z", "+00:00"))
    for fold in payload["fold_protocol"]["folds"]:
        cal_end = datetime.fromisoformat(fold["calibration_end"].replace("Z", "+00:00"))
        scoring_start = datetime.fromisoformat(fold["scoring_start"].replace("Z", "+00:00"))
        scoring_end = datetime.fromisoformat(fold["scoring_end"].replace("Z", "+00:00"))
        assert scoring_start - cal_end == timedelta(minutes=120)
        assert scoring_end < reserve


def test_hourly_sampling_uses_close_time_and_has_zero_h12_overlap() -> None:
    result = build_e2_hourly_short_dataset(_series(), _sampling(), dataset_id="fixture", source_finality_verified=True)
    timestamps = sorted({row.timestamp for row in result.dataset.rows})
    assert timestamps == [
        datetime(2025, 1, 2, 0, tzinfo=timezone.utc),
        datetime(2025, 1, 2, 1, tzinfo=timezone.utc),
    ]
    assert all(value.minute == value.second == value.microsecond == 0 for value in timestamps)
    assert timestamps[1] - timestamps[0] == timedelta(minutes=60)
    assert result.valid_cycle_count == 2 and result.dataset.row_count == 22
    assert all(count == 2 for count in result.rows_by_symbol.values())


def test_nonfinal_data_and_non_h12_stride_are_rejected() -> None:
    with pytest.raises(ValueError, match="final"):
        build_e2_hourly_short_dataset(_series(), _sampling(), dataset_id="x", source_finality_verified=False)
    sampling = _sampling(); sampling["stride_bars"] = 6
    with pytest.raises(ValueError, match="overlap"):
        build_e2_hourly_short_dataset(_series(), sampling, dataset_id="x", source_finality_verified=True)


def test_historical_gap_skips_entire_coordinated_cycle_and_hits_hard_stop() -> None:
    series = _series(); target = CANONICAL_SYMBOLS[0]
    series[target] = tuple(bar for bar in series[target] if bar.timestamp != datetime(2025, 1, 1, 23, 50, tzinfo=timezone.utc))
    with pytest.raises(ValueError, match="90%"):
        build_e2_hourly_short_dataset(series, _sampling(), dataset_id="x", source_finality_verified=True)


def test_future_gap_quarantines_label_cycle() -> None:
    series = _series(); target = CANONICAL_SYMBOLS[0]
    series[target] = tuple(bar for bar in series[target] if bar.timestamp != datetime(2025, 1, 2, 1, 20, tzinfo=timezone.utc))
    sampling = _sampling(); sampling["expected_rows"]["hard_stop_if_valid_rows_below_fraction"] = 0.0
    result = build_e2_hourly_short_dataset(series, sampling, dataset_id="x", source_finality_verified=True)
    assert result.quarantined_label_cycles >= 1


def test_prelockbox_loader_rejects_any_semiblind_boundary_request() -> None:
    payload = yaml.safe_load(E2.read_text())
    payload["sampling"]["expected_rows"]["last_dev_anchor_utc"] = "2026-04-27T00:00:00Z"
    with pytest.raises(ValueError, match="SEMI_BLIND_ACCESS_FORBIDDEN_PRE_LOCKBOX"):
        load_and_build_e2_hourly_dataset(object(), payload)  # type: ignore[arg-type]


def test_shared_authority_creation_does_not_consume_and_is_compatible(tmp_path: Path) -> None:
    authority = SharedLockboxAuthority(
        tmp_path / "authority.json", "window", "2026-04-27T00:00:00Z",
        "2026-07-11T09:20:00Z", 1,
    )
    hashes = {"physical": "a" * 64, "canonical": "b" * 64}
    created = authority.initialize(e1_hashes=hashes, e2_hashes=hashes)
    assert created["consumed_queries"] == [] and created["status"] == "NOT_CONSUMED"
    assert authority.audit_available() == created


def test_shared_authority_is_single_query_across_preregistrations(tmp_path: Path) -> None:
    authority = SharedLockboxAuthority(tmp_path / "authority.json", "window", "start", "end", 1)
    hashes = {"physical": "a" * 64, "canonical": "b" * 64}
    authority.initialize(e1_hashes=hashes, e2_hashes=hashes)
    lease = SharedWindowLockboxLease(tmp_path / "authority.lease.json", authority.path)
    record = LockboxLeaseRecord(
        "run-e1", "c" * 64, "b" * 64, "aegis-short-candidate-e1", "d" * 40,
        "e" * 64, datetime(2026, 7, 18, tzinfo=timezone.utc), 1, RunMode.FULL_RUN,
        "f" * 64, "a" * 64,
    )
    lease.acquire(record)
    with pytest.raises(PhaseETechnicalError) as captured:
        SharedWindowLockboxLease(tmp_path / "authority.lease.json", authority.path).acquire(
            LockboxLeaseRecord(**{**record.__dict__, "experiment_id": "aegis-short-candidate-e2", "run_id": "run-e2"})
        )
    assert captured.value.code is PhaseEErrorCode.LOCKBOX_ALREADY_CONSUMED
    assert len(json.loads(authority.path.read_text())["consumed_queries"]) == 1


def test_incompatible_shared_authority_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "authority.json"; path.write_text('{"status":"NOT_CONSUMED"}')
    authority = SharedLockboxAuthority(path, "window", "start", "end", 1)
    hashes = {"physical": "a", "canonical": "b"}
    with pytest.raises(PreregistrationError, match="INCOMPATIBLE"):
        authority.initialize(e1_hashes=hashes, e2_hashes=hashes)


def test_e2_block_hashes_are_recordable() -> None:
    payload, audit = load_and_validate_preregistration(E2, audit_source=False)
    hashes = {key: Sha256HashProvider().digest_value(payload[key]) for key in (
        "sampling", "fold_protocol", "calibration", "scoring", "refit",
        "threshold_derivation", "lockbox",
    )}
    assert audit.content_hash and len(set(hashes.values())) == len(hashes)
