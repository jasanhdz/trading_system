import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import aegis.training.phase_e as phase_e_module
from aegis.training.phase_e import (
    FULL_RUN_AUTHORIZATION, CalibrationResult, EconEvaluationResult,
    ModelCompetitionResult, PhaseEOrchestrator, PhaseEPreflight,
    PhaseEPreflightResult,
    PromotionCriterionCheck, QMAEValidationResult, SimulatedScientificBackend,
    evaluate_promotion_criteria,
)
from aegis.training.run_state import (
    EnvironmentFingerprint, PhaseEErrorCode, PhaseEState, PhaseETechnicalError,
    RunMode, RunStateStore,
)
from aegis.utils import Sha256HashProvider


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "experiments" / "aegis_short_candidate_e1.yaml"


class StubPreflight:
    def __init__(self, root: Path, *, consumed: bool = False) -> None:
        self.root = root
        self.payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        self.payload["lockbox"]["query_state_path"] = "fake-lockbox/budget.json"
        self.environment = EnvironmentFingerprint.current()
        self.result = PhaseEPreflightResult(
            experiment_id=self.payload["experiment_id"],
            preregistration_file_hash="a" * 64, preregistration_hash="b" * 64,
            competition_file_hash="c" * 64, d3_manifest_hash="d" * 64,
            d3_artifact_id="d3-fixture", git_commit="e" * 40,
            typescript_commit="f" * 40, branch="feature/aegis-ts-clean-rebuild",
            environment=self.environment, available_disk_bytes=10**9,
            lockbox_consumed=consumed, full_run_executed=False,
            threshold_pending=True, long_disabled=True, execution_enabled=False,
            checked_at=__import__("datetime").datetime(2026, 7, 18, tzinfo=__import__("datetime").timezone.utc),
        )

    def run(self, mode: RunMode):
        if mode is RunMode.FULL_RUN and self.result.lockbox_consumed:
            raise PhaseETechnicalError(PhaseEErrorCode.LOCKBOX_ALREADY_CONSUMED, "consumed")
        return self.payload, self.result


class ForbiddenBackend:
    def __getattr__(self, name: str):
        raise AssertionError(f"dry-run invoked scientific backend: {name}")


def orchestrator(tmp_path: Path, mode: RunMode, *, approving: bool = True, authorization: str | None = None):
    return PhaseEOrchestrator(
        preflight=StubPreflight(tmp_path),
        backend=SimulatedScientificBackend(approving=approving),
        reports_root=tmp_path / "reports", mode=mode,
        owner_authorization=authorization, smoke_approving=approving,
    )


def test_dry_run_stops_before_scientific_backend_and_is_idempotent(tmp_path: Path) -> None:
    runner = PhaseEOrchestrator(
        preflight=StubPreflight(tmp_path), backend=ForbiddenBackend(),
        reports_root=tmp_path / "reports", mode=RunMode.DRY_RUN,
    )
    first = runner.run()
    second = runner.run()
    assert first == second
    assert first.state is PhaseEState.RUN_SNAPSHOT_CREATED
    assert not (tmp_path / "fake-lockbox" / "budget.json").exists()
    manifest = json.loads((Path(first.run_dir) / "run_manifest.json").read_text())
    assert manifest["scientific_execution"] is False
    assert manifest["estimated_resources"]


def test_preflight_rejects_changed_preregistration_before_d3_access(tmp_path: Path) -> None:
    changed = tmp_path / "changed.yaml"
    changed.write_text(CONFIG.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    preflight = PhaseEPreflight(
        repository_root=ROOT, typescript_root=ROOT / "binance-futures-bot-ts",
        preregistration_path=changed,
        competition_path=ROOT / "config" / "scientific_competition_v1.yaml",
        strict_git=False,
    )
    with pytest.raises(PhaseETechnicalError) as captured:
        preflight.run(RunMode.DRY_RUN)
    assert captured.value.code is PhaseEErrorCode.PREREGISTRATION_MISMATCH


def test_preflight_classifies_d3_manifest_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenCanonicalSource:
        def __init__(self, *args, **kwargs):
            pass

        def audit(self, *, verify_content: bool):
            raise ValueError("manifest changed")

    monkeypatch.setattr(phase_e_module, "CanonicalSeriesSource", BrokenCanonicalSource)
    preflight = PhaseEPreflight(
        repository_root=ROOT, typescript_root=ROOT / "binance-futures-bot-ts",
        preregistration_path=CONFIG,
        competition_path=ROOT / "config" / "scientific_competition_v1.yaml",
        strict_git=False,
    )
    with pytest.raises(PhaseETechnicalError) as captured:
        preflight.run(RunMode.DRY_RUN)
    assert captured.value.code is PhaseEErrorCode.D3_HASH_MISMATCH


def test_smoke_candidate_uses_fake_lockbox_and_finishes_simulated(tmp_path: Path) -> None:
    result = orchestrator(tmp_path, RunMode.SMOKE_RUN, approving=True).run()
    assert result.state is PhaseEState.CANDIDATE_SIMULATED
    run_dir = Path(result.run_dir)
    assert (run_dir / "smoke_lockbox" / "lockbox_lease.json").is_file()
    assert (run_dir / "smoke_lockbox" / "lockbox_state.json").is_file()
    assert not (tmp_path / "fake-lockbox" / "budget.json").exists()
    assert json.loads((run_dir / "criteria_report.json").read_text())["all_mandatory_passed"] is True
    assert orchestrator(tmp_path, RunMode.SMOKE_RUN, approving=True).run() == result


def test_smoke_rejection_preserves_reports_without_candidate_artifacts(tmp_path: Path) -> None:
    result = orchestrator(tmp_path, RunMode.SMOKE_RUN, approving=False).run()
    assert result.state is PhaseEState.REJECTED_SIMULATED
    run_dir = Path(result.run_dir)
    assert (run_dir / "final_report.json").is_file()
    assert not (run_dir / "selection_policy.json").exists()
    assert not (run_dir / "system_freeze.json").exists()


def test_validation_run_stops_before_lockbox(tmp_path: Path) -> None:
    result = orchestrator(tmp_path, RunMode.VALIDATION_RUN).run()
    assert result.state is PhaseEState.QMAE_VALIDATED
    assert result.lockbox_consumed is False
    assert (Path(result.run_dir) / "experimental_bundle.json").is_file()
    assert not (tmp_path / "fake-lockbox" / "budget.json").exists()


@pytest.mark.parametrize("authorization", [None, "", "OWNER_AUTHORIZED", "owner_authorized_phase_e_full_run"])
def test_full_run_without_exact_authorization_fails_before_preflight(tmp_path: Path, authorization: str | None) -> None:
    with pytest.raises(PhaseETechnicalError) as captured:
        orchestrator(tmp_path, RunMode.FULL_RUN, authorization=authorization).run()
    assert captured.value.code is PhaseEErrorCode.PRECHECK_FAILED
    assert not (tmp_path / "fake-lockbox").exists()


def test_cli_full_run_without_authorization_has_no_outputs(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable, str(ROOT / "scripts" / "run_aegis_candidate_experiment.py"),
            "--mode", "full-run", "--reports-root", str(tmp_path / "reports"),
        ],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 2
    assert "FULL_RUN_OWNER_AUTHORIZATION_REQUIRED" in completed.stderr
    assert not (tmp_path / "reports").exists()


def test_fake_full_run_consumes_once_and_publishes_bound_candidate(tmp_path: Path) -> None:
    result = orchestrator(
        tmp_path, RunMode.FULL_RUN, authorization=FULL_RUN_AUTHORIZATION,
    ).run()
    assert result.state is PhaseEState.CANDIDATE
    run_dir = Path(result.run_dir)
    assert (tmp_path / "fake-lockbox" / "budget.json").is_file()
    assert (run_dir / "selection_policy.json").is_file()
    assert (run_dir / "system_freeze.json").is_file()
    assert list((run_dir / "registry").glob("*.json"))
    assert orchestrator(
        tmp_path, RunMode.FULL_RUN, authorization=FULL_RUN_AUTHORIZATION,
    ).run() == result


def test_consumed_fake_lockbox_blocks_full_run(tmp_path: Path) -> None:
    preflight = StubPreflight(tmp_path, consumed=True)
    runner = PhaseEOrchestrator(
        preflight=preflight, backend=SimulatedScientificBackend(approving=True),
        reports_root=tmp_path / "reports", mode=RunMode.FULL_RUN,
        owner_authorization=FULL_RUN_AUTHORIZATION,
    )
    with pytest.raises(PhaseETechnicalError) as captured:
        runner.run()
    assert captured.value.code is PhaseEErrorCode.LOCKBOX_ALREADY_CONSUMED


@pytest.mark.parametrize(
    ("change", "expected_state"),
    [
        ("model", False), ("calibration", False), ("qmae", False),
        ("econ", False), ("signals", False), ("profit_factor", False),
        ("expectancy", False), ("concentration", False), ("ece", False),
        ("none", True),
    ],
)
def test_promotion_criteria_fail_closed_by_component(change: str, expected_state: bool) -> None:
    prereg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    digest = Sha256HashProvider().digest_value
    competition = ModelCompetitionResult({}, {}, {}, change == "model", digest("competition"))
    maximum_ece = 0.20 if change == "ece" else 0.02
    calibration = CalibrationResult(
        change not in {"calibration", "ece"}, maximum_ece, {}, digest("calibration"),
    )
    coverage = 0.80 if change == "qmae" else 0.90
    qmae = QMAEValidationResult(
        change != "qmae", {str(i): coverage for i in range(1, 5)}, {}, {}, {}, digest("qmae"),
    )
    report = {"metrics": {"full_stack": {"B_BASE": {
        "trades": 99 if change == "signals" else 120,
        "profit_factor": 1.0 if change == "profit_factor" else 1.2,
        "expectancy": 0.0 if change == "expectancy" else 0.01,
        "maximum_symbol_share": 0.31 if change == "concentration" else 0.2,
    }}}}
    econ = EconEvaluationResult(
        report, (0.1, 0.2), 0.1, change != "econ",
        {str(i): 0.01 if change != "econ" else -0.01 for i in range(1, 5)},
        0.005, digest("econ"),
    )
    result = evaluate_promotion_criteria(
        prereg, competition=competition, calibration=calibration,
        qmae=qmae, econ=econ, evidence_path="fixture.json",
    )
    assert result.all_mandatory_passed is expected_state


def test_every_criterion_has_auditable_shape() -> None:
    item = PromotionCriterionCheck("minimum_test_signals", 100, 99, False, "econ.json", fold=4)
    assert item.side == "SHORT"
    assert item.severity == "MANDATORY"
    assert item.fold == 4


def test_promotion_result_contains_every_preregistered_mandatory_gate() -> None:
    prereg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    digest = Sha256HashProvider().digest_value
    result = evaluate_promotion_criteria(
        prereg,
        competition=ModelCompetitionResult({}, {}, {}, False, digest("competition")),
        calibration=CalibrationResult(True, 0.02, {}, digest("calibration")),
        qmae=QMAEValidationResult(True, {str(i): 0.90 for i in range(1, 5)}, {}, {}, {}, digest("qmae")),
        econ=EconEvaluationResult({"metrics": {"full_stack": {"B_BASE": {
            "trades": 120, "profit_factor": 1.2, "expectancy": 0.01,
            "maximum_symbol_share": 0.2,
        }}}}, (0.1,), 0.1, True, {str(i): 0.01 for i in range(1, 5)}, 0.005, digest("econ")),
        evidence_path="fixture.json",
    )
    names = {item.name for item in result.checks}
    assert {
        "minimum_test_signals", "minimum_positive_folds", "minimum_profit_factor_base",
        "minimum_net_expectancy_base", "minimum_worst_fold_expectancy",
        "maximum_symbol_concentration", "maximum_ece_each_fold", "models_beaten",
        "calibration_valid", "qmae_coverage_valid", "econ_positive_robust",
        "beat_best_directional_baseline_expectancy", "no_known_leakage", "side_separated",
    } <= names


@pytest.mark.parametrize("mode", list(RunMode))
def test_run_mode_is_stable_cli_contract(mode: RunMode) -> None:
    assert RunMode(mode.value) is mode


def test_state_history_and_evidence_are_recoverable(tmp_path: Path) -> None:
    result = orchestrator(tmp_path, RunMode.SMOKE_RUN, approving=True).run()
    run_dir = Path(result.run_dir)
    document = json.loads((run_dir / "state.json").read_text())
    states = [item["state"] for item in document["history"]]
    assert states[0] == "PRE_REGISTERED"
    assert states[-1] == "CANDIDATE_SIMULATED"
    lines = (run_dir / "evidence.jsonl").read_text().splitlines()
    assert lines
    assert len({json.loads(line)["event_id"] for line in lines}) == len(lines)


def test_crash_post_lockbox_is_terminal(tmp_path: Path) -> None:
    result = orchestrator(tmp_path, RunMode.SMOKE_RUN, approving=True).run()
    store = RunStateStore(
        Path(result.run_dir), run_id=result.run_id, experiment_id=result.experiment_id,
        preregistration_hash="b" * 64, git_commit="e" * 40,
        environment_hash=EnvironmentFingerprint.current().content_hash,
    )
    assert store.state is PhaseEState.CANDIDATE_SIMULATED
    with pytest.raises(PhaseETechnicalError):
        store.transition(PhaseEState.PREFLIGHT_VALIDATED, {})
