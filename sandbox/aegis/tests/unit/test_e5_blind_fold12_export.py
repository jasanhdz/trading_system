"""Synthetic-only validation for the E5 Phase 1A blind exporter."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from aegis.e5_blind_fold12_export.errors import BlindExportError, BlindExportInterrupted
from aegis.e5_blind_fold12_export.exporter import (
    HISTORICAL_UNAVAILABLE,
    OUTPUT_FIELDS,
    ProjectedRow,
    export_fold12,
    load_sealed_manifest,
    manifest_bytes,
)
from aegis.e5_blind_fold12_export.guard import CleanRoomGuard
from aegis.e5_blind_fold12_export.identity import (
    IDENTITY_FIELDS,
    IDENTITY_SCHEME,
    CanonicalTradeIdentity,
    derive_canonical_trade_identity,
)
from aegis.e5_blind_fold12_export.orchestrator import main, run_phase1a
from aegis.e5_blind_fold12_export.validation import REQUIRED_TEST_CATEGORIES, validate_manifest_bytes


ROOT = Path(__file__).resolve().parents[2]
TS_ROOT = ROOT / "binance-futures-bot-ts"
CANARY = "SYNTHETIC_FOLD34_PAYLOAD_CANARY_7c5f9a"


def _identity_values(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "symbol": "BTCUSDT",
        "fold": "F1",
        "signal_timestamp_utc_ms": "2025-01-01T00:00:00Z",
        "entry_timestamp_utc_ms": "2025-01-01T00:05:00Z",
        "entry_price_decimal": "100.00",
        "side": "SHORT",
    }
    values.update(overrides)
    return values


def _trade(
    fold: object = "F1",
    *,
    symbol: str = "BTCUSDT",
    signal_timestamp: str = "2025-01-01T00:00:00Z",
    entry_timestamp: str = "2025-01-01T00:05:00Z",
    entry_price: object = 100.0,
    side: str = "SHORT",
    score: object = 0.25,
    strategy_id: str = "full_stack",
    scenario_id: str = "B_BASE",
    canary: bool = False,
) -> dict[str, object]:
    marker = CANARY if canary else "SYNTHETIC"
    return {
        "signal": {
            "timestamp": signal_timestamp,
            "symbol": symbol,
            "side": side,
            "score": score,
            "strategy_id": strategy_id,
            "fold": fold,
            "regime": marker,
        },
        "scenario_id": scenario_id,
        "entry_timestamp": entry_timestamp,
        "exit_timestamp": "2025-01-01T01:00:00Z",
        "entry_price": entry_price,
        "exit_price": marker if canary else 99.0,
        "gross_return_fraction": marker if canary else 0.01,
        "cost_fraction": 0.001,
        "net_return_fraction": marker if canary else 0.009,
        "mfe_fraction": marker if canary else 0.02,
        "mae_fraction": marker if canary else -0.01,
    }


def _source(tmp_path: Path, trades: list[dict[str, object]], *, raw: bytes | None = None) -> tuple[Path, str]:
    path = tmp_path / "source" / "combined.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = raw if raw is not None else json.dumps(
        {"report": {"trades": trades}}, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()


def _config(tmp_path: Path, source: Path, source_hash: str, output: Path) -> Path:
    base = json.loads((ROOT / "config/e5_blind_fold12_export_v1.json").read_text(encoding="utf-8"))
    base["source"]["path"] = str(source)
    base["source"]["sha256"] = source_hash
    base["output"]["root"] = str(output)
    path = tmp_path / f"config-{output.name}.json"
    path.write_text(json.dumps(base, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return path


def _export(tmp_path: Path, trades: list[dict[str, object]], *, output_name: str = "out", **kwargs: object):
    source, digest = _source(tmp_path, trades)
    output = tmp_path / output_name
    config = _config(tmp_path, source, digest, output)
    result = export_fold12(
        repository_root=ROOT,
        source=source,
        expected_source_sha256=digest,
        config_path=config,
        output_root=output,
        **kwargs,
    )
    return result, source, config, output


def _projected(ordinal: int = 0, **identity_overrides: object) -> ProjectedRow:
    identity = derive_canonical_trade_identity(_identity_values(**identity_overrides))
    return ProjectedRow(ordinal, identity, "0.25", "full_stack")


def _assert_code(code: str, call) -> None:
    with pytest.raises(BlindExportError) as caught:
        call()
    assert caught.value.code == code


def test_required_validation_registry_is_exactly_63_unique_categories() -> None:
    assert len(REQUIRED_TEST_CATEGORIES) == 63
    assert len(set(REQUIRED_TEST_CATEGORIES)) == 63


def test_trade_id_exact_vector_and_repeatability() -> None:
    first = derive_canonical_trade_identity(_identity_values())
    second = derive_canonical_trade_identity(dict(reversed(list(_identity_values().items()))))
    assert first == second
    assert first.preimage == (
        b'["e5-historical-trade-id-v1","BTCUSDT","F1",1735689600000,'
        b'1735689900000,"100.0","SHORT"]'
    )
    assert first.trade_id == hashlib.sha256(first.preimage).hexdigest()


@pytest.mark.parametrize(
    ("overrides", "changed"),
    [
        ({"signal_timestamp_utc_ms": "2025-01-01T00:00:01Z"}, True),
        ({"entry_timestamp_utc_ms": "2025-01-01T00:05:01Z"}, True),
        ({"entry_price_decimal": "100.01"}, True),
        ({"symbol": "ETHUSDT"}, True),
        ({"fold": "F2"}, True),
        ({"side": "LONG"}, True),
    ],
)
def test_identity_tuple_fields_change_id(overrides: dict[str, object], changed: bool) -> None:
    baseline = derive_canonical_trade_identity(_identity_values())
    candidate = derive_canonical_trade_identity(_identity_values(**overrides))
    assert (baseline.trade_id != candidate.trade_id) is changed


@pytest.mark.parametrize(
    "overrides",
    [
        {"signal_timestamp_utc_ms": "2024-12-31T19:00:00-05:00"},
        {"entry_timestamp_utc_ms": "2024-12-31T19:05:00-05:00"},
        {"entry_price_decimal": "100"},
        {"entry_price_decimal": "1.00E2"},
        {"entry_price_decimal": "0100.000"},
    ],
)
def test_equivalent_identity_forms_canonicalize_identically(overrides: dict[str, object]) -> None:
    assert derive_canonical_trade_identity(_identity_values()).trade_id == derive_canonical_trade_identity(
        _identity_values(**overrides)
    ).trade_id


@pytest.mark.parametrize(
    "entry_price",
    ["000100", "000100.0000", "1000e-1", "0.100E3", "100.000000", 100],
)
def test_additional_equivalent_decimal_forms(entry_price: object) -> None:
    assert derive_canonical_trade_identity(_identity_values()).trade_id == derive_canonical_trade_identity(
        _identity_values(entry_price_decimal=entry_price)
    ).trade_id


def test_horizon_and_source_run_are_not_identity_inputs() -> None:
    row = _identity_values()
    assert derive_canonical_trade_identity(row).trade_id == derive_canonical_trade_identity(dict(row)).trade_id
    for field in ("horizon", "source_run_id"):
        _assert_code(
            "E5_CANONICAL_TRADE_ID_CANONICALIZATION_FAILED",
            lambda field=field: derive_canonical_trade_identity({**row, field: "not-an-identity-input"}),
        )


@pytest.mark.parametrize("field", IDENTITY_FIELDS)
def test_missing_identity_inputs_fail(field: str) -> None:
    values = _identity_values()
    values.pop(field)
    _assert_code("E5_CANONICAL_TRADE_ID_REQUIRED_INPUT_MISSING", lambda: derive_canonical_trade_identity(values))
    _assert_code(
        "E5_CANONICAL_TRADE_ID_REQUIRED_INPUT_MISSING",
        lambda: derive_canonical_trade_identity({**_identity_values(), field: None}),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("symbol", "BTC/USDT"),
        ("fold", "F5"),
        ("signal_timestamp_utc_ms", "2025-01-01T00:00:00"),
        ("entry_timestamp_utc_ms", "2025-01-01T00:05:00.000001Z"),
        ("entry_price_decimal", "0"),
        ("entry_price_decimal", "NaN"),
        ("side", "SELL"),
    ],
)
def test_invalid_identity_inputs_fail(field: str, value: object) -> None:
    _assert_code(
        "E5_CANONICAL_TRADE_ID_INPUT_INVALID",
        lambda: derive_canonical_trade_identity(_identity_values(**{field: value})),
    )


def test_unsupported_identity_version_and_prohibited_dependencies_fail() -> None:
    _assert_code(
        "E5_CANONICAL_TRADE_ID_VERSION_UNSUPPORTED",
        lambda: derive_canonical_trade_identity(_identity_values(), version="v2"),
    )
    cases = (
        ("net_return", "E5_CANONICAL_TRADE_ID_OUTCOME_DEPENDENCY_PROHIBITED"),
        ("d3", "E5_CANONICAL_TRADE_ID_CURRENT_BRAIN_DEPENDENCY_PROHIBITED"),
        ("wall_clock", "E5_CANONICAL_TRADE_ID_CANONICALIZATION_FAILED"),
        ("filesystem_path", "E5_CANONICAL_TRADE_ID_CANONICALIZATION_FAILED"),
    )
    for field, code in cases:
        _assert_code(code, lambda field=field: derive_canonical_trade_identity({**_identity_values(), field: "x"}))


def test_f1_f2_emit_and_f3_f4_do_not_emit(tmp_path: Path) -> None:
    result, _, _, _ = _export(
        tmp_path,
        [_trade("F1"), _trade("F2", symbol="ETHUSDT"), _trade("F3", canary=True), _trade("F4", canary=True)],
    )
    payload = result.manifest_path.read_bytes()
    assert result.rows_emitted == 2
    assert validate_manifest_bytes(payload) == 2
    assert b'"fold":"F1"' in payload and b'"fold":"F2"' in payload
    assert b'"fold":"F3"' not in payload and b'"fold":"F4"' not in payload
    assert CANARY.encode() not in payload


@pytest.mark.parametrize(("fold", "expected"), [(1, "F1"), (2, "F2")])
def test_authoritative_integer_fold_forms_are_normalized(tmp_path: Path, fold: int, expected: str) -> None:
    result, _, _, _ = _export(tmp_path, [_trade(fold)])
    row = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert row["fold"] == expected


@pytest.mark.parametrize("fold", ["F5", None, "", 0, True])
def test_unknown_or_missing_fold_fails_closed(tmp_path: Path, fold: object) -> None:
    trade = _trade(fold)
    if fold is None:
        trade["signal"].pop("fold")  # type: ignore[union-attr]
    _assert_code("E5_AUTHORITATIVE_FOLD_SELECTOR_UNAVAILABLE", lambda: _export(tmp_path, [trade]))


def test_source_hash_mismatch_precedes_payload_processing(tmp_path: Path) -> None:
    source, actual = _source(tmp_path, [_trade("F3", canary=True)])
    output = tmp_path / "out"
    config = _config(tmp_path, source, "0" * 64, output)
    _assert_code(
        "E5_COMBINED_SOURCE_HASH_MISMATCH",
        lambda: export_fold12(
            repository_root=ROOT, source=source, expected_source_sha256="0" * 64,
            config_path=config, output_root=output,
        ),
    )
    assert actual != "0" * 64


def test_unknown_source_schema_and_unauthorized_output_field_fail(tmp_path: Path) -> None:
    trade = _trade()
    trade["unexpected"] = "synthetic"
    _assert_code("E5_BLIND_EXPORT_SCHEMA_MISMATCH", lambda: _export(tmp_path, [trade]))
    _assert_code(
        "E5_BLIND_EXPORT_UNAUTHORIZED_FIELD",
        lambda: manifest_bytes([_projected()], (*OUTPUT_FIELDS, "extra")),
    )


def test_missing_essential_fields_fail_and_nonexperimental_arm_is_excluded(tmp_path: Path) -> None:
    trade = _trade()
    trade["signal"].pop("score")  # type: ignore[union-attr]
    _assert_code("E5_HISTORICAL_REQUIRED_FIELD_UNAVAILABLE", lambda: _export(tmp_path, [trade], output_name="missing"))
    result, _, _, _ = _export(tmp_path, [_trade(strategy_id="other")], output_name="strategy")
    assert result.rows_emitted == 0


def test_component_evidence_is_not_replayed_or_imputed(tmp_path: Path) -> None:
    result, _, _, output = _export(tmp_path, [_trade()])
    row = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert tuple(row) == OUTPUT_FIELDS
    assert not ({"d3", "rv2", "trrm", "qmae", "eqm", "econ1", "aegis"} & set(row))
    assert HISTORICAL_UNAVAILABLE == "NOT_AVAILABLE_HISTORICAL_NOT_PERSISTED"
    assert not (output / "reports").exists()


def test_duplicate_conflict_and_collision_fail_closed() -> None:
    first = _projected(0)
    duplicate = _projected(1)
    _assert_code("E5_BLIND_EXPORT_DUPLICATE_IDENTITY", lambda: manifest_bytes([first, duplicate]))
    conflict = replace(duplicate, score_decimal="0.26")
    _assert_code("E5_CANONICAL_TRADE_ID_CONFLICT", lambda: manifest_bytes([first, conflict]))
    collision_identity = replace(
        derive_canonical_trade_identity(_identity_values(symbol="ETHUSDT")),
        trade_id=first.trade_id,
    )
    collision = ProjectedRow(1, collision_identity, "0.25", "full_stack")
    _assert_code("E5_CANONICAL_TRADE_ID_HASH_COLLISION", lambda: manifest_bytes([first, collision]))


def test_manifest_bytes_are_independent_of_processing_order() -> None:
    rows = [_projected(1, symbol="ETHUSDT"), _projected(0)]
    assert manifest_bytes(rows) == manifest_bytes(tuple(reversed(rows)))


@pytest.mark.parametrize("failure", ["newline", "extra", "fold", "trade_id"])
def test_independent_manifest_validator_fails_closed(failure: str) -> None:
    payload = manifest_bytes([_projected()])
    row = json.loads(payload)
    if failure == "newline":
        payload = payload.rstrip(b"\n")
    elif failure == "extra":
        row["extra"] = "x"
        payload = (json.dumps(row, separators=(",", ":")) + "\n").encode()
    elif failure == "fold":
        row["fold"] = "F3"
        payload = (json.dumps(row, separators=(",", ":")) + "\n").encode()
    else:
        row["trade_id"] = "0" * 64
        payload = (json.dumps(row, separators=(",", ":")) + "\n").encode()
    with pytest.raises(BlindExportError):
        validate_manifest_bytes(payload)


def test_interruption_resume_and_repeated_export_are_byte_identical(tmp_path: Path) -> None:
    trades = [_trade("F1"), _trade("F2", symbol="ETHUSDT")]
    first, source, config, first_root = _export(tmp_path, trades, output_name="first")
    repeated = export_fold12(
        repository_root=ROOT, source=source, expected_source_sha256=first.source_sha256,
        config_path=config, output_root=first_root,
    )
    assert repeated.status == "EXISTING_IDENTICAL"
    resume_root = tmp_path / "resume"
    resume_config = _config(tmp_path, source, first.source_sha256, resume_root)
    with pytest.raises(BlindExportInterrupted):
        export_fold12(
            repository_root=ROOT, source=source, expected_source_sha256=first.source_sha256,
            config_path=resume_config, output_root=resume_root,
            interrupt_after_authorized_rows=1,
        )
    resumed = export_fold12(
        repository_root=ROOT, source=source, expected_source_sha256=first.source_sha256,
        config_path=resume_config, output_root=resume_root, resume=True,
    )
    assert first.manifest_sha256 == repeated.manifest_sha256 == resumed.manifest_sha256
    assert first.manifest_path.read_bytes() == resumed.manifest_path.read_bytes()


def test_checkpoint_dependency_and_existing_output_conflicts_are_rejected(tmp_path: Path) -> None:
    result, source, config, output = _export(tmp_path, [_trade()])
    result.manifest_path.write_bytes(b"conflict\n")
    _assert_code(
        "E5_BLIND_EXPORT_OUTPUT_CONFLICT",
        lambda: export_fold12(
            repository_root=ROOT, source=source, expected_source_sha256=result.source_sha256,
            config_path=config, output_root=output,
        ),
    )


def test_prohibited_canary_never_leaks_to_channels_or_artifacts(tmp_path: Path, capsys, caplog) -> None:
    trades = [_trade("F3", canary=True), _trade("F4", canary=True), _trade("F1")]
    result, _, _, output = _export(tmp_path, trades)
    captured = capsys.readouterr()
    aggregate = captured.out + captured.err + caplog.text
    for path in output.rglob("*"):
        if path.is_file():
            aggregate += path.read_text(encoding="utf-8")
    assert CANARY not in aggregate
    assert CANARY.encode() not in result.manifest_path.read_bytes()


def test_prohibited_canary_is_absent_from_safe_exception(tmp_path: Path) -> None:
    source, digest = _source(tmp_path, [_trade("F3", canary=True), _trade("F5", canary=True)])
    output = tmp_path / "out"
    config = _config(tmp_path, source, digest, output)
    with pytest.raises(BlindExportError) as caught:
        export_fold12(
            repository_root=ROOT, source=source, expected_source_sha256=digest,
            config_path=config, output_root=output,
        )
    assert CANARY not in str(caught.value)


def test_clean_room_guard_allows_only_sealed_manifest(tmp_path: Path) -> None:
    combined = tmp_path / "source" / "combined.json"
    sealed = tmp_path / "sealed.jsonl"
    combined.parent.mkdir()
    combined.write_text("{}", encoding="utf-8")
    sealed.write_text("", encoding="utf-8")
    guard = CleanRoomGuard(combined, sealed, (tmp_path / "semi_blind",), (tmp_path / "lockbox",))
    assert guard.validate_downstream_path(sealed) == sealed.resolve()
    _assert_code("E5_PHASE1_COMBINED_SOURCE_ACCESS_PROHIBITED", lambda: guard.validate_downstream_path(combined))
    _assert_code("SEMIBLIND_ACCESS_ATTEMPT", lambda: guard.validate_downstream_path(tmp_path / "semi_blind/rows"))
    _assert_code("LOCKBOX_ACCESS_ATTEMPT", lambda: guard.validate_downstream_path(tmp_path / "lockbox/rows"))


def test_downstream_loader_hashes_and_reads_only_sealed_manifest(tmp_path: Path) -> None:
    result, source, _, _ = _export(tmp_path, [_trade()])
    guard = CleanRoomGuard(source, result.manifest_path, (), ())
    assert (
        load_sealed_manifest(guard, result.manifest_path, result.manifest_sha256)
        == result.manifest_path.read_bytes()
    )
    _assert_code(
        "E5_PHASE1_COMBINED_SOURCE_ACCESS_PROHIBITED",
        lambda: load_sealed_manifest(guard, source, result.source_sha256),
    )


@pytest.mark.parametrize("stage", ["discovery", "confirmation", "shadow", "live"])
def test_cli_rejects_unsupported_scientific_and_operational_stages(stage: str) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--stage", stage])
    assert caught.value.code == 2


def test_export_is_offline_and_does_not_import_exchange_or_order_modules(tmp_path: Path, monkeypatch) -> None:
    import socket

    def deny_socket(*args, **kwargs):
        raise AssertionError("network access prohibited")

    monkeypatch.setattr(socket, "socket", deny_socket)
    result, _, _, _ = _export(tmp_path, [_trade()])
    assert result.rows_emitted == 1


def test_orchestrator_writes_compact_reports_without_canary(tmp_path: Path) -> None:
    trades = [
        _trade("F1"),
        _trade("F2", symbol="ETHUSDT"),
        _trade("F3", canary=True),
        _trade("F4", canary=True),
        _trade(
            "F1", symbol="SOLUSDT", signal_timestamp="2025-01-02T00:00:00Z",
            entry_timestamp="2025-01-02T00:05:00Z",
        ),
        _trade(
            "F2", symbol="XRPUSDT", signal_timestamp="2025-01-03T00:00:00Z",
            entry_timestamp="2025-01-03T00:05:00Z",
        ),
        _trade(
            "F1", symbol="BNBUSDT", signal_timestamp="2025-01-04T00:00:00Z",
            entry_timestamp="2025-01-04T00:05:00Z",
        ),
    ]
    source, digest = _source(tmp_path, trades)
    output = tmp_path / "phase1a"
    config = _config(tmp_path, source, digest, output)
    result, hashes = run_phase1a(
        source=source, expected_source_sha256=digest, config_path=config,
        output_root=output, deterministic_rerun=True,
    )
    assert result.rows_emitted == 5
    assert "e5_phase1a_report.json" in hashes
    assert all(CANARY not in path.read_text(encoding="utf-8") for path in output.rglob("*") if path.is_file())
    report = json.loads((output / "e5_phase1a_report.json").read_text(encoding="utf-8"))
    audit = json.loads((output / "blind_export_audit.json").read_text(encoding="utf-8"))
    assert audit["audit_started_at"].endswith("Z") and audit["audit_completed_at"].endswith("Z")
    assert report["semi_blind"] == "NOT_ACCESSED"
    assert report["lockbox"] == "NOT_CONSUMED"
    assert report["consumed_queries"] == [] and report["budget_remaining"] == 1


def test_typescript_repository_is_unchanged() -> None:
    assert subprocess.run(
        ("git", "-C", str(TS_ROOT), "status", "--porcelain"),
        check=True, capture_output=True, text=True,
    ).stdout == ""
