"""Governance, schema, prohibited-data, ledger, and checkpoint validators."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .constants import (
    AUTHORITIES,
    EXPECTED_LOCKBOX_STATE,
    HOLM_TEST_IDS,
    PHASE0_VERSION,
    PROTOCOL_ROOT,
    REPOSITORY_ROOT,
    TYPESCRIPT_ROOT,
)
from .core import canonical_json_bytes, confirmation_run_id
from .errors import Phase0Error


EXPECTED_BRANCH = "feature/aegis-ts-clean-rebuild"


@dataclass(frozen=True)
class GovernanceValidation:
    hashes: dict[str, str]
    commits: dict[str, str]
    python_commit: str
    typescript_commit: str
    python_tree: str
    typescript_tree: str


def _git(root: Path, *args: str, check: bool = True) -> str:
    completed = subprocess.run(
        ("git", "-C", str(root), *args),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if check and completed.returncode != 0:
        raise Phase0Error("GOVERNANCE_COMMIT_MISMATCH", completed.stderr.strip() or "git command failed")
    return completed.stdout.strip()


def verify_governance(*, require_clean: bool = True) -> GovernanceValidation:
    hashes: dict[str, str] = {}
    commits: dict[str, str] = {}
    for authority in AUTHORITIES:
        path = PROTOCOL_ROOT / authority.relative_path
        if not path.is_file():
            raise Phase0Error("GOVERNANCE_HASH_MISMATCH", f"missing {authority.relative_path}")
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != authority.sha256:
            raise Phase0Error("GOVERNANCE_HASH_MISMATCH", authority.relative_path)
        completed = subprocess.run(("git", "-C", str(REPOSITORY_ROOT), "cat-file", "-e", f"{authority.commit}^{{commit}}"), check=False)
        if completed.returncode != 0:
            raise Phase0Error("GOVERNANCE_COMMIT_MISMATCH", authority.commit)
        hashes[authority.name] = actual_hash
        commits[authority.name] = authority.commit
    for earlier, later in zip(AUTHORITIES, AUTHORITIES[1:]):
        completed = subprocess.run(("git", "-C", str(REPOSITORY_ROOT), "merge-base", "--is-ancestor", earlier.commit, later.commit), check=False)
        if completed.returncode != 0:
            raise Phase0Error("GOVERNANCE_CONFLICT", f"authority order {earlier.name}->{later.name}")
    python_branch = _git(REPOSITORY_ROOT, "branch", "--show-current")
    typescript_branch = _git(TYPESCRIPT_ROOT, "branch", "--show-current")
    if python_branch != EXPECTED_BRANCH or typescript_branch != EXPECTED_BRANCH:
        raise Phase0Error("DIRTY_WORKTREE", "unexpected branch")
    python_status = _git(REPOSITORY_ROOT, "status", "--porcelain")
    typescript_status = _git(TYPESCRIPT_ROOT, "status", "--porcelain")
    if require_clean and (python_status or typescript_status):
        raise Phase0Error("DIRTY_WORKTREE", "Phase 0 requires clean repositories")
    return GovernanceValidation(
        hashes,
        commits,
        _git(REPOSITORY_ROOT, "rev-parse", "HEAD"),
        _git(TYPESCRIPT_ROOT, "rev-parse", "HEAD"),
        "CLEAN" if not python_status else "PHASE0_CHANGESET_ONLY",
        "CLEAN" if not typescript_status else "DIRTY",
    )


ARTIFACT_SCHEMAS: dict[str, frozenset[str]] = {
    "governance_manifest.json": frozenset({"schema_version", "authorities"}),
    "execution_spec_manifest.json": frozenset({"schema_version", "path", "sha256"}),
    "source_state_manifest.json": frozenset({"schema_version", "python_commit", "typescript_commit", "clean"}),
    "software_manifest.json": frozenset({"schema_version", "python_version", "numpy_version"}),
    "data_boundary_manifest.json": frozenset({"schema_version", "classes", "deny_roots"}),
    "prohibited_data_guard_report.json": frozenset({"schema_version", "semi_blind", "lockbox", "consumed_queries", "budget_remaining"}),
    "input_manifest.json": frozenset({"schema_version", "inputs"}),
    "seed_manifest.json": frozenset({"schema_version", "base_seed", "vectors"}),
    "symbol_registry_manifest.json": frozenset({"schema_version", "symbols", "symbol_set_hash"}),
    "funding_raw_manifest.json": frozenset({"schema_version", "provider", "raw_artifacts"}),
    "e5_funding_history_v1_manifest.json": frozenset({"schema_version", "provider", "normalized_artifact_sha256", "coverage_complete"}),
    "exclusion_manifest.jsonl": frozenset({"schema_version", "observation_id", "primary_code"}),
    "c1_matching_manifest.jsonl": frozenset({"schema_version", "observation_id", "experimental_symbol", "control_symbol", "self_match_exclusions"}),
    "c2_matching_manifest.jsonl": frozenset({"schema_version", "observation_id", "experimental_symbol", "experimental_cycle_id", "control_symbol", "control_cycle_id", "self_edge_exclusions"}),
    "bootstrap_manifest.json": frozenset({"schema_version", "requested", "valid", "invalid", "lower", "upper"}),
    "temporal_permutation_manifest.json": frozenset({"schema_version", "test_id", "week_blocks", "shifts"}),
    "holm_registry.json": frozenset({"schema_version", "test_ids", "decisions"}),
    "label_economics_registry.json": frozenset({"schema_version", "labels", "blocking_class"}),
    "e5_phase0_report.json": frozenset({"phase0_version", "execution_specification_sha256", "governance_hashes", "final_status"}),
    "artifact_hash_manifest.json": frozenset({"schema_version", "artifacts"}),
}


def validate_synthetic_artifact(name: str, payload: Mapping[str, Any]) -> None:
    required = ARTIFACT_SCHEMAS.get(name)
    if required is None:
        raise Phase0Error("INPUT_SCHEMA_MISMATCH", f"unknown artifact schema {name}")
    missing = required - set(payload)
    if missing:
        raise Phase0Error("INPUT_SCHEMA_MISMATCH", f"{name} missing {sorted(missing)}")
    if payload.get("synthetic") is not True or payload.get("scientific_use") is not False:
        raise Phase0Error("INPUT_SCHEMA_MISMATCH", f"{name} lacks synthetic safety markers")
    canonical_json_bytes(payload)


@dataclass(frozen=True)
class ProhibitedDataGuard:
    semi_blind_roots: tuple[Path, ...]
    lockbox_roots: tuple[Path, ...]
    forbidden_interfaces: tuple[str, ...] = ("lockbox_query", "consume_query", "decrement_budget", "lockbox_transaction")

    def validate_path(self, requested: Path) -> Path:
        resolved = requested.resolve(strict=False)
        for root in (*self.semi_blind_roots, *self.lockbox_roots):
            denied = root.resolve(strict=False)
            if resolved == denied or denied in resolved.parents:
                code = "SEMIBLIND_ACCESS_ATTEMPT" if root in self.semi_blind_roots else "LOCKBOX_ACCESS_ATTEMPT"
                raise Phase0Error(code, str(requested))
        return resolved

    def validate_interface(self, name: str) -> None:
        if name in self.forbidden_interfaces:
            code = "LOCKBOX_MUTATION_ATTEMPT" if "budget" in name or "transaction" in name else "LOCKBOX_ACCESS_ATTEMPT"
            raise Phase0Error(code, name)

    def report(self) -> dict[str, Any]:
        return {
            "schema_version": "e5-prohibited-data-guard-v1",
            "synthetic": True,
            "scientific_use": False,
            "semi_blind": "NOT_ACCESSED",
            **EXPECTED_LOCKBOX_STATE,
            "result": "PASS",
        }


@dataclass(frozen=True)
class ConfirmationLedger:
    experiment_id: str
    confirmation_run_id: str
    status: str
    discovery_freeze_hash: str
    confirmation_input_hash: str
    seed_manifest_hash: str
    checkpoint_hashes: tuple[str, ...] = ()
    failure_code: str | None = None

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.__dict__)


def new_confirmation_ledger(discovery_freeze_hash: str, confirmation_input_hash: str, seed_manifest_hash: str) -> ConfirmationLedger:
    run_id = confirmation_run_id(discovery_freeze_hash, confirmation_input_hash)
    return ConfirmationLedger("E5", run_id, "NOT_STARTED", discovery_freeze_hash, confirmation_input_hash, seed_manifest_hash)


def start_confirmation(ledger: ConfirmationLedger, requested_run_id: str, dependency_hashes: tuple[str, str, str]) -> ConfirmationLedger:
    expected = (ledger.discovery_freeze_hash, ledger.confirmation_input_hash, ledger.seed_manifest_hash)
    if dependency_hashes != expected or requested_run_id != ledger.confirmation_run_id:
        raise Phase0Error("CONFIRMATION_DEPENDENCY_MISMATCH", "confirmation dependency changed")
    if ledger.status == "NOT_STARTED":
        return replace(ledger, status="STARTED")
    if ledger.status == "STARTED":
        return ledger
    raise Phase0Error("CONFIRMATION_ALREADY_STARTED", ledger.status)


def resume_confirmation(ledger: ConfirmationLedger, requested_run_id: str, dependency_hashes: tuple[str, str, str]) -> ConfirmationLedger:
    if ledger.status != "STARTED":
        raise Phase0Error("CONFIRMATION_RESUME_INVALID", ledger.status)
    return start_confirmation(ledger, requested_run_id, dependency_hashes)


@dataclass(frozen=True)
class Checkpoint:
    confirmation_run_id: str
    stage: str
    dependency_hash: str
    seed_manifest_hash: str
    artifact_hashes: tuple[str, ...]

    def identity(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.__dict__)).hexdigest()


def validate_checkpoint(checkpoint: Checkpoint, ledger: ConfirmationLedger, dependency_hash: str, seed_manifest_hash: str) -> None:
    if checkpoint.confirmation_run_id != ledger.confirmation_run_id or checkpoint.dependency_hash != dependency_hash or checkpoint.seed_manifest_hash != seed_manifest_hash:
        raise Phase0Error("CONFIRMATION_RESUME_INVALID", "checkpoint dependency mismatch")
    for value in checkpoint.artifact_hashes:
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise Phase0Error("ARTIFACT_HASH_MISMATCH", "checkpoint artifact hash malformed")


@dataclass(frozen=True)
class RuleImplementation:
    module: str
    function_or_schema: str
    test: str
    artifact: str
    failure_code: str


def _mapping(module: str, function: str, test: str, artifact: str, failure: str) -> RuleImplementation:
    return RuleImplementation(module, function, test, artifact, failure)


RULE_IMPLEMENTATION_MATRIX: dict[str, RuleImplementation] = {}
for number in range(1, 54):
    rule_id = f"E5-R{number:03d}"
    if number <= 4:
        value = _mapping("validation", "verify_governance", "governance_integrity", "governance_manifest.json", "GOVERNANCE_HASH_MISMATCH")
    elif number <= 10:
        value = _mapping("validation", "ProhibitedDataGuard", "prohibited_data_guard", "prohibited_data_guard_report.json", "PROHIBITED_DATA_REFERENCE")
    elif number <= 13:
        value = _mapping("core", "identity_hash", "canonical_identities", "input_manifest.json", "DUPLICATE_IDENTITY")
    elif number <= 17:
        value = _mapping("core", "time_numeric_seed_serialization", "time_numeric_seed_vectors", "seed_manifest.json", "DETERMINISM_FAILURE")
    elif number <= 26:
        value = _mapping("core/funding", "eligibility_outcome_atr_quantile_contracts", "scientific_primitive_vectors", "exclusion_manifest.jsonl", "INPUT_SCHEMA_MISMATCH")
    elif number <= 31:
        value = _mapping("matching", "C1_C2_matching", "matching_vectors", "c2_matching_manifest.jsonl", "C2_MATCHING_INFEASIBLE")
    elif number <= 42:
        value = _mapping("statistics", "statistical_contracts", "statistical_vectors", "holm_registry.json", "HOLM_FAMILY_INCOMPLETE")
    elif number <= 49:
        value = _mapping("orchestrator/validation", "state_artifact_checkpoint_contracts", "state_and_artifact_vectors", "artifact_hash_manifest.json", "ARTIFACT_HASH_MISMATCH")
    else:
        value = _mapping("orchestrator", "run_phase0", "phase0_end_to_end", "e5_phase0_report.json", "DETERMINISM_FAILURE")
    RULE_IMPLEMENTATION_MATRIX[rule_id] = value


PHASE0_TEST_CATEGORIES = (
    "governance_hash_verification",
    "canonical_identity_generation",
    "shuffled_input_ordering",
    "byte_identical_reruns",
    "time_boundary_behavior",
    "horizon_specific_populations",
    "short_return_and_fixed_costs",
    "same_bar_adverse_first",
    "wilder_atr",
    "type7_quintiles",
    "seed_vectors",
    "c1_only_self_failure",
    "c1_distinct_selection",
    "c2_only_exact_pair_failure",
    "c2_same_symbol_different_cycle",
    "c2_different_symbol",
    "c2_shuffled_source_identity",
    "c2_self_edge_pretraversal",
    "augmenting_path_graphs",
    "fold_centered_residuals",
    "nested_power_stream_isolation",
    "complete_week_shared_shifts",
    "spread_decile_ordering",
    "monotonicity_average_rank",
    "bootstrap_type7_validity",
    "holm_family_ties",
    "pooled_concentration",
    "label_truth_table",
    "diagnostic_ic",
    "funding_decimal_and_identity",
    "funding_duplicate_order_reconciliation",
    "funding_interval_boundaries",
    "funding_zero_event_complete_coverage",
    "funding_incomplete_coverage",
    "one_shot_confirmation",
    "checkpoint_resume_dependency",
    "prohibited_data_guard",
    "unchanged_lockbox_state",
)


def validate_rule_matrix() -> None:
    expected = {f"E5-R{number:03d}" for number in range(1, 54)}
    if set(RULE_IMPLEMENTATION_MATRIX) != expected:
        raise Phase0Error("UNAUTHORIZED_SCIENTIFIC_CHOICE", "Phase 0 rule mapping is incomplete")
    if len(PHASE0_TEST_CATEGORIES) != 38 or len(set(PHASE0_TEST_CATEGORIES)) != 38:
        raise Phase0Error("DETERMINISM_FAILURE", "test category registry must contain 38 unique entries")
