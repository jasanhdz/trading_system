"""Offline E5 Phase 0 orchestrator; never loads scientific data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .constants import (
    AUTHORITIES,
    EXPECTED_LOCKBOX_STATE,
    PHASE0_REPORT_PATH,
    PHASE0_VERSION,
    PROTOCOL_ROOT,
    REPOSITORY_ROOT,
    TYPESCRIPT_ROOT,
)
from .core import canonical_json_bytes
from .errors import Phase0Error
from .synthetic import run_synthetic_checks, validate_all_artifact_schemas
from .validation import ARTIFACT_SCHEMAS, PHASE0_TEST_CATEGORIES, ProhibitedDataGuard, RULE_IMPLEMENTATION_MATRIX, verify_governance


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _implementation_hashes() -> dict[str, str]:
    package_root = Path(__file__).resolve().parent
    paths = sorted(
        (
            *(path for path in package_root.rglob("*") if path.is_file() and "__pycache__" not in path.parts),
            REPOSITORY_ROOT / "tests/unit/test_e5_phase0.py",
            REPOSITORY_ROOT / "tests/fixtures/e5_phase0/synthetic_phase0_fixture.json",
        ),
        key=lambda path: path.relative_to(REPOSITORY_ROOT).as_posix(),
    )
    return {path.relative_to(REPOSITORY_ROOT).as_posix(): _sha256_file(path) for path in paths}


def _guard() -> ProhibitedDataGuard:
    # Path identities only. Phase 0 never opens either denied root.
    return ProhibitedDataGuard(
        (REPOSITORY_ROOT / "data/semi_blind", REPOSITORY_ROOT / "reports/experiments/semi_blind"),
        (REPOSITORY_ROOT / "data/lockbox", REPOSITORY_ROOT / "reports/experiments/lockbox"),
    )


def build_phase0_report(*, require_clean: bool = True) -> dict[str, Any]:
    governance = verify_governance(require_clean=require_clean)
    validate_all_artifact_schemas()
    guard_report = _guard().report()
    results = run_synthetic_checks()
    failures = [result for result in results if result.status != "PASS"]
    implementation_hashes = _implementation_hashes()
    requirements_path = REPOSITORY_ROOT / "requirements.txt"
    dependency_hash = _sha256_file(requirements_path)
    test_suite_identity = hashlib.sha256(canonical_json_bytes({
        "categories": PHASE0_TEST_CATEGORIES,
        "implementation_hashes": implementation_hashes,
    })).hexdigest()
    matrix_hash = hashlib.sha256(canonical_json_bytes({key: asdict(value) for key, value in RULE_IMPLEMENTATION_MATRIX.items()})).hexdigest()
    report: dict[str, Any] = {
        "phase0_version": PHASE0_VERSION,
        "synthetic": True,
        "scientific_use": False,
        "execution_specification_sha256": AUTHORITIES[-1].sha256,
        "governance_hashes": governance.hashes,
        "governance_commits": governance.commits,
        "repository_commits": {
            "python": governance.python_commit,
            "typescript": governance.typescript_commit,
        },
        "source_tree_state": {
            "python": governance.python_tree,
            "typescript": governance.typescript_tree,
        },
        "runtime_versions": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "dependency_manifest_hash": dependency_hash,
        "implementation_matrix_sha256": matrix_hash,
        "test_suite_identity": test_suite_identity,
        "test_count": len(results),
        "passed_count": len(results) - len(failures),
        "failed_count": len(failures),
        "skipped_count": 0,
        "tests": [asdict(result) for result in results],
        "deterministic_rerun_result": "PENDING",
        "artifact_hashes": {
            **implementation_hashes,
            "requirements.txt": dependency_hash,
        },
        "artifact_schema_count": len(ARTIFACT_SCHEMAS),
        "prohibited_data_guard_result": guard_report,
        "semi_blind_state": "NOT_ACCESSED",
        **EXPECTED_LOCKBOX_STATE,
        "network_calls": 0,
        "scientific_rows_inspected": 0,
        "scientific_datasets_created": 0,
        "discovery_state": "NOT_STARTED",
        "confirmation_state": "NOT_STARTED",
        "final_status": "FAIL" if failures else "PASS",
    }
    first = canonical_json_bytes({**report, "deterministic_rerun_result": "BYTE_IDENTICAL"})
    second = canonical_json_bytes({**report, "deterministic_rerun_result": "BYTE_IDENTICAL"})
    if first != second:
        raise Phase0Error("DETERMINISM_FAILURE", "report reconstruction changed bytes")
    report["deterministic_rerun_result"] = "BYTE_IDENTICAL"
    return report


def write_phase0_report(report: dict[str, Any], report_path: Path = PHASE0_REPORT_PATH) -> tuple[Path, Path]:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_bytes = canonical_json_bytes(report)
    _atomic_write(report_path, report_bytes)
    hash_manifest_path = report_path.with_name("artifact_hash_manifest.json")
    report_identity = (
        report_path.relative_to(REPOSITORY_ROOT).as_posix()
        if report_path.is_relative_to(REPOSITORY_ROOT)
        else f"synthetic-output/{report_path.name}"
    )
    hash_manifest = {
        "schema_version": "e5-artifact-hash-manifest-v1",
        "synthetic": True,
        "scientific_use": False,
        "artifacts": [
            {
                "path": report_identity,
                "sha256": hashlib.sha256(report_bytes).hexdigest(),
            },
            *(
                {"path": path, "sha256": digest}
                for path, digest in sorted(report["artifact_hashes"].items())
            ),
        ],
    }
    _atomic_write(hash_manifest_path, canonical_json_bytes(hash_manifest))
    return report_path, hash_manifest_path


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run offline synthetic E5 Phase 0 validation")
    parser.add_argument("--report", type=Path, default=PHASE0_REPORT_PATH)
    parser.add_argument("--allow-development-tree", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        report = build_phase0_report(require_clean=not args.allow_development_tree)
        report_path, hash_path = write_phase0_report(report, args.report)
    except Phase0Error as exc:
        print(f"E5_PHASE_0_BLOCKED {exc}", file=sys.stderr)
        return 1
    if report["final_status"] != "PASS":
        print(f"E5_PHASE_0_BLOCKED report={report_path}", file=sys.stderr)
        return 1
    print(f"E5_PHASE_0_PASS report={report_path} hashes={hash_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
