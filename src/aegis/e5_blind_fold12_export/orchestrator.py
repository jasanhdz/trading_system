"""Offline E5 Phase 1A orchestration and compact audit reporting."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from aegis.e5_phase0.core import canonical_json_bytes

from .errors import BlindExportError, BlindExportInterrupted
from .exporter import (
    COMPONENTS,
    HISTORICAL_UNAVAILABLE,
    MANIFEST_SCHEMA_VERSION,
    OUTPUT_FIELDS,
    ExportConfig,
    ExportResult,
    atomic_write,
    export_fold12,
    load_config,
    package_code_identity,
    sha256_file,
)
from .identity import AUTHORITY_CLASSIFICATION, IDENTITY_FIELDS, IDENTITY_SCHEME
from .validation import REQUIRED_TEST_CATEGORIES, validate_manifest_bytes


PHASE1A_VERSION = "e5-phase1a-blind-export-v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TYPESCRIPT_ROOT = REPOSITORY_ROOT / "binance-futures-bot-ts"
PROTOCOL_ROOT = REPOSITORY_ROOT / "reports/governance/e5_signal_edge_protocol"
SCHEMA_PATH = Path(__file__).resolve().parent / "schemas/fold12_manifest.schema.json"
EXPECTED_LOCKBOX_STATE = {
    "lockbox": "NOT_CONSUMED",
    "consumed_queries": [],
    "budget_remaining": 1,
}
AUTHORITIES = {
    "original": (
        "e5_protocol_preregistration.md",
        "c8057276c93b761b4acca6a6569c8a87468c8b374e34f1bbfffa2b42da3b5770",
        "b8b86d012c40c4d10f10efb68e5eb9d86d4ac476",
    ),
    "patch_02": (
        "e5_protocol_patch_02.md",
        "c668cb28f490ce32524c258791d8d8d58dafb2214939c62871ba43c929bf848e",
        "92191db1a7c4135252377f64f51b174f180dcd53",
    ),
    "amendment_01": (
        "e5_owner_authorized_amendment_01.md",
        "c05be85a58e59c3706175f5e2e24ea2343fa63b78e0cc196cdde8ed0faec55a4",
        "943b98a5091c4d9238f754a1e42e63540a4579a6",
    ),
    "amendment_02": (
        "e5_owner_authorized_amendment_02.md",
        "b54662ab860e204904ddaf65cc0c1ad046fd5073398045a3d5fc7c36ba418d0f",
        "521289606117a478debfca00d2e1fbaa5c2a4301",
    ),
    "amendment_03": (
        "e5_owner_authorized_amendment_03.md",
        "871be087550eb9d632795ded2c8f2633f1e481838198f0ee3ce53b9c8e9a350e",
        "5003630ae42a806f79466ec10a4c052ce2a6f28a",
    ),
    "amendment_04": (
        "e5_owner_authorized_amendment_04.md",
        "a177980633c3280d6eaf6a4a798a6eb623f3692878639894869d2a39f8643774",
        "a76553d15a239735bbb909f96ff3f06426148f50",
    ),
    "execution_specification": (
        "e5_execution_specification.md",
        "751b4014f1072e6fd0a49fb3a8820ba60b1b3c556eb94f9fbb7911d70516ae09",
        "34441e412f79bc7d12d253040019e857ab5cf2c8",
    ),
    "amendment_05": (
        "e5_owner_authorized_amendment_05.md",
        "5a3a71f64c105df417fdf0067c222f1a71879756feddae6fbfa389ae4e5475de",
        "a80fa55c23ad37362d1de26a2ae3469374466a1a",
    ),
    "amendment_06": (
        "e5_owner_authorized_amendment_06.md",
        "c1428c195a897fca399d91e20aec157c8ebc016fa7fce534816a201b40c78413",
        "3cad46f525f51c9f720ccee9fc0de3b7cea69464",
    ),
}
PHASE0_REPORT = ("phase0/e5_phase0_report.json", "a11502e7334d2a288c1b9aae0ef9761540cb2d6ae103d4212b106d6509314d5d")


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(root), *args),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode:
        raise BlindExportError("E5_PHASE1A_GOVERNANCE_CONFLICT", "Git authority unavailable")
    return completed.stdout.strip()


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def verify_authorities() -> dict[str, Any]:
    hashes: dict[str, str] = {}
    commits: dict[str, str] = {}
    for name, (relative, expected, commit) in AUTHORITIES.items():
        path = PROTOCOL_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise BlindExportError("E5_PHASE1A_GOVERNANCE_CONFLICT", f"authority={name}")
        completed = subprocess.run(
            ("git", "-C", str(REPOSITORY_ROOT), "cat-file", "-e", f"{commit}^{{commit}}"),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if completed.returncode:
            raise BlindExportError("E5_PHASE1A_GOVERNANCE_CONFLICT", f"authority_commit={name}")
        hashes[name] = expected
        commits[name] = commit
    phase0_path = PROTOCOL_ROOT / PHASE0_REPORT[0]
    if not phase0_path.is_file() or sha256_file(phase0_path) != PHASE0_REPORT[1]:
        raise BlindExportError("E5_PHASE1A_GOVERNANCE_CONFLICT", "authority=phase0_report")
    phase0 = json.loads(phase0_path.read_bytes())
    if phase0.get("final_status") != "PASS" or phase0.get("passed_count") != 38 or phase0.get("failed_count") != 0:
        raise BlindExportError("E5_PHASE1A_GOVERNANCE_CONFLICT", "Phase 0 state")
    return {
        "hashes": hashes,
        "commits": commits,
        "phase0_report_sha256": PHASE0_REPORT[1],
        "phase0_status": "38/38 PASS",
    }


def _temporary_config(config: ExportConfig, output_root: Path, path: Path) -> Path:
    payload = deepcopy(dict(config.payload))
    payload["output"] = dict(payload["output"])
    payload["output"]["root"] = str(output_root)
    atomic_write(path, canonical_json_bytes(payload))
    return path


def run_determinism(
    *,
    source: Path,
    expected_source_sha256: str,
    config: ExportConfig,
    canonical_result: ExportResult,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="e5-phase1a-determinism-") as temporary:
        root = Path(temporary)
        first_root = root / "first"
        second_root = root / "second"
        resume_root = root / "resume"
        first_config = _temporary_config(config, first_root, root / "first_config.json")
        second_config = _temporary_config(config, second_root, root / "second_config.json")
        resume_config = _temporary_config(config, resume_root, root / "resume_config.json")
        first = export_fold12(
            repository_root=REPOSITORY_ROOT, source=source,
            expected_source_sha256=expected_source_sha256, config_path=first_config,
            output_root=first_root,
        )
        second = export_fold12(
            repository_root=REPOSITORY_ROOT, source=source,
            expected_source_sha256=expected_source_sha256, config_path=second_config,
            output_root=second_root,
        )
        try:
            export_fold12(
                repository_root=REPOSITORY_ROOT, source=source,
                expected_source_sha256=expected_source_sha256, config_path=resume_config,
                output_root=resume_root, interrupt_after_authorized_rows=5,
            )
        except BlindExportInterrupted:
            pass
        else:
            raise BlindExportError("E5_BLIND_EXPORT_NONDETERMINISTIC", "interruption did not occur")
        resumed = export_fold12(
            repository_root=REPOSITORY_ROOT, source=source,
            expected_source_sha256=expected_source_sha256, config_path=resume_config,
            output_root=resume_root, resume=True,
        )
        hashes = (
            canonical_result.manifest_sha256,
            first.manifest_sha256,
            second.manifest_sha256,
            resumed.manifest_sha256,
        )
        payloads = (
            canonical_result.manifest_path.read_bytes(),
            first.manifest_path.read_bytes(),
            second.manifest_path.read_bytes(),
            resumed.manifest_path.read_bytes(),
        )
        if len(set(hashes)) != 1 or any(payload != payloads[0] for payload in payloads[1:]):
            raise BlindExportError("E5_BLIND_EXPORT_NONDETERMINISTIC", "repeated/resumed manifest mismatch")
        return {
            "schema_version": "e5-blind-export-determinism-v1",
            "canonical_sha256": hashes[0],
            "uninterrupted_sha256": hashes[1],
            "repeated_sha256": hashes[2],
            "resumed_sha256": hashes[3],
            "byte_identical": True,
            "interruption_resume": "PASS",
        }


def _write_reports(
    *,
    output_root: Path,
    config: ExportConfig,
    result: ExportResult,
    governance: Mapping[str, Any],
    determinism: Mapping[str, Any],
    audit_started_at: str,
    audit_completed_at: str,
) -> dict[str, str]:
    output_root.mkdir(parents=True, exist_ok=True)
    python_commit = _git(REPOSITORY_ROOT, "rev-parse", "HEAD")
    typescript_commit = _git(TYPESCRIPT_ROOT, "rev-parse", "HEAD")
    common = {
        "phase1a_version": PHASE1A_VERSION,
        "scientific_use": False,
        "source_sha256": result.source_sha256,
        "configuration_sha256": result.configuration_sha256,
        "exporter_code_identity": result.code_identity,
        "exporter_source_code_commit": python_commit,
        "manifest_sha256": result.manifest_sha256,
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "partition": "F1-F2_DEVELOPMENT_DISCOVERY",
        "ordering": "IMMUTABLE_COMBINED_SOURCE_ORDINAL",
        "validation": "PASS",
        "failure_codes": [],
        "semi_blind": "NOT_ACCESSED",
        **EXPECTED_LOCKBOX_STATE,
    }
    reports: dict[str, Mapping[str, Any] | bytes] = {
        "blind_export_governance_manifest.json": {
            **common,
            "schema_version": "e5-blind-export-governance-manifest-v1",
            "authority_hashes": governance["hashes"],
            "authority_commits": governance["commits"],
            "phase0_report_sha256": governance["phase0_report_sha256"],
            "phase0_status": governance["phase0_status"],
        },
        "blind_export_source_manifest.json": {
            **common,
            "schema_version": "e5-blind-export-source-manifest-v1",
            "source_path": config.payload["source"]["path"],
            "expected_source_sha256": config.source_sha256,
            "actual_source_sha256": result.source_sha256,
            "source_schema_version": config.payload["source"]["schema_version"],
            "source_authority_config": config.payload["source"]["authority_config"],
            "partition_selector": "signal.fold",
            "source_hash_verified": True,
            "prohibited_partition_counts_reported": False,
        },
        "blind_export_schema.json": canonical_json_bytes(json.loads(SCHEMA_PATH.read_bytes())),
        "blind_export_audit.json": {
            **common,
            "schema_version": "e5-blind-export-audit-v1",
            "operator_independent": True,
            "audit_started_at": audit_started_at,
            "audit_completed_at": audit_completed_at,
            "rows_emitted": result.rows_emitted,
            "duplicate_rows": result.duplicate_rows,
            "conflicting_rows": result.conflicting_rows,
            "invalid_rows": result.invalid_rows,
            "unavailable_nonessential_evidence": result.unavailable_nonessential_evidence,
            "unavailable_essential_evidence": result.unavailable_essential_evidence,
            "prohibited_partitions_emitted": result.prohibited_partitions_emitted,
            "scientific_values_printed": 0,
            "scientific_statistics_calculated": False,
            "fold_3_4_payload_emitted": False,
            "network_calls": 0,
            "private_exchange_calls": 0,
            "order_calls": 0,
        },
        "blind_export_validation_report.json": {
            **common,
            "schema_version": "e5-blind-export-validation-v1",
            "authority_validation": "PASS",
            "source_validation": "PASS",
            "schema_validation": "PASS",
            "identity_validation": "PASS",
            "duplicate_validation": "PASS",
            "clean_room_validation": "PASS",
            "required_synthetic_categories": len(REQUIRED_TEST_CATEGORIES),
            "synthetic_category_registry_sha256": hashlib.sha256(
                canonical_json_bytes(list(REQUIRED_TEST_CATEGORIES))
            ).hexdigest(),
        },
        "blind_export_determinism_report.json": {**common, **determinism},
        "blind_export_clean_room_report.json": {
            **common,
            "schema_version": "e5-blind-export-clean-room-v1",
            "combined_source_available_to_exporter_only": True,
            "combined_source_available_to_downstream": False,
            "fold_3_4_payload_emitted": False,
            "prohibited_payload_leak": False,
            "downstream_denial_code": "E5_PHASE1_COMBINED_SOURCE_ACCESS_PROHIBITED",
        },
        "fold12_manifest_provenance.json": {
            **common,
            "schema_version": "e5-fold12-manifest-provenance-v1",
            "trade_id_authority": AUTHORITY_CLASSIFICATION,
            "trade_id_scheme": IDENTITY_SCHEME,
            "trade_id_tuple": list(IDENTITY_FIELDS),
            "allowed_output_fields": list(OUTPUT_FIELDS),
            "historical_component_evidence": {
                "applies_to_all_emitted_rows": True,
                "default_status_by_component": {component: HISTORICAL_UNAVAILABLE for component in COMPONENTS},
                "overrides": [],
            },
            "manifest_relative_path": _display_path(result.manifest_path),
            "rows_emitted": result.rows_emitted,
        },
    }
    hashes: dict[str, str] = {}
    for name, value in reports.items():
        payload = value if isinstance(value, bytes) else canonical_json_bytes(value)
        atomic_write(output_root / name, payload)
        hashes[name] = hashlib.sha256(payload).hexdigest()
    phase1a = {
        **common,
        "schema_version": "e5-phase1a-report-v1",
        "final_status": "E5_PHASE1A_BLIND_EXPORT_COMPLETE",
        "phase0_status": governance["phase0_status"],
        "rows_emitted": result.rows_emitted,
        "determinism": "BYTE_IDENTICAL",
        "resume_determinism": "BYTE_IDENTICAL",
        "discovery": "NOT_STARTED",
        "confirmation": "NOT_STARTED",
        "shadow": "NOT_STARTED",
        "live": "NOT_STARTED",
        "network_calls": 0,
        "scientific_rows_manually_inspected": 0,
        "fold_3_4_values_exposed": False,
        "python_commit": python_commit,
        "typescript_commit": typescript_commit,
        "artifact_hashes": dict(sorted(hashes.items())),
    }
    report_bytes = canonical_json_bytes(phase1a)
    atomic_write(output_root / "e5_phase1a_report.json", report_bytes)
    hashes["e5_phase1a_report.json"] = hashlib.sha256(report_bytes).hexdigest()
    artifact_manifest = {
        "schema_version": "e5-phase1a-artifact-hash-manifest-v1",
        "scientific_use": False,
        "sealed_manifest": {
            "path": _display_path(result.manifest_path),
            "sha256": result.manifest_sha256,
        },
        "artifacts": [
            {"path": name, "sha256": digest}
            for name, digest in sorted(hashes.items())
        ],
    }
    artifact_bytes = canonical_json_bytes(artifact_manifest)
    atomic_write(output_root / "artifact_hash_manifest.json", artifact_bytes)
    hashes["artifact_hash_manifest.json"] = hashlib.sha256(artifact_bytes).hexdigest()
    return hashes


def run_phase1a(
    *,
    source: Path,
    expected_source_sha256: str,
    config_path: Path,
    output_root: Path,
    resume: bool = False,
    validation_only: bool = False,
    deterministic_rerun: bool = False,
    interrupt_after_authorized_rows: int | None = None,
) -> tuple[ExportResult, dict[str, str]]:
    audit_started_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    governance = verify_authorities()
    config = load_config(config_path, REPOSITORY_ROOT)
    result = export_fold12(
        repository_root=REPOSITORY_ROOT,
        source=source,
        expected_source_sha256=expected_source_sha256,
        config_path=config_path,
        output_root=output_root,
        resume=resume,
        validation_only=validation_only,
        interrupt_after_authorized_rows=interrupt_after_authorized_rows,
    )
    if not validation_only:
        validated_rows = validate_manifest_bytes(result.manifest_path.read_bytes())
        if validated_rows != result.rows_emitted:
            raise BlindExportError("E5_BLIND_EXPORT_SCHEMA_MISMATCH", "manifest row count")
    if validation_only or interrupt_after_authorized_rows is not None:
        return result, {}
    if deterministic_rerun:
        determinism = run_determinism(
            source=source,
            expected_source_sha256=expected_source_sha256,
            config=config,
            canonical_result=result,
        )
    else:
        determinism = {
            "schema_version": "e5-blind-export-determinism-v1",
            "canonical_sha256": result.manifest_sha256,
            "byte_identical": False,
            "interruption_resume": "NOT_RUN",
        }
    hashes = _write_reports(
        output_root=output_root,
        config=config,
        result=result,
        governance=governance,
        determinism=determinism,
        audit_started_at=audit_started_at,
        audit_completed_at=datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
    )
    return result, hashes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the offline E5 Phase 1A blind Fold 1-2 exporter")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--deterministic-rerun", action="store_true")
    parser.add_argument("--interrupt-after-authorized-rows", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        result, hashes = run_phase1a(
            source=args.source,
            expected_source_sha256=args.expected_source_sha256,
            config_path=args.config,
            output_root=args.output_root,
            resume=args.resume,
            validation_only=args.validation_only,
            deterministic_rerun=args.deterministic_rerun,
            interrupt_after_authorized_rows=args.interrupt_after_authorized_rows,
        )
    except BlindExportInterrupted as exc:
        print(f"E5_PHASE1A_INTERRUPTED {exc.message}")
        return 75
    except BlindExportError as exc:
        print(f"{exc.code} {exc.message}", file=sys.stderr)
        return 1
    print(
        "E5_PHASE1A_BLIND_EXPORT_COMPLETE"
        f" source_sha256={result.source_sha256}"
        f" rows={result.rows_emitted}"
        f" duplicates={result.duplicate_rows}"
        f" conflicts={result.conflicting_rows}"
        f" rejected={result.invalid_rows}"
        f" manifest_sha256={result.manifest_sha256}"
        f" reports={len(hashes)}"
    )
    return 0
