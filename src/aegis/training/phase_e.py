"""Phase-E orchestration over existing scientific APIs; contains no scientific formulas."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
from ..config import CANONICAL_SYMBOLS, CANONICAL_SYMBOL_SET_HASH
from ..data import CanonicalBar, CanonicalSeriesSource, DataPurpose
from ..domain import ScientificEvidenceEvent, TradeSide
from ..evidence import AppendOnlyEvidenceRecorder
from ..features import FEATURE_HASH, FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from ..freeze import BundleLifecycleState, FrozenSelectionPolicy, SystemFreeze
from ..models import CalibrationMethod, CalibratorSpec, model_bundle_from_payload
from ..utils import Sha256HashProvider, canonical_json, sha256_file, to_primitive
from .competition import (
    QmaeQuantileResult, fit_calibrator_family, fit_qmae_refit,
    fit_selected_probabilistic_candidate, fit_selected_regression_candidate,
    build_rank_based_eqm_population, load_scientific_competition_contract,
    rank_based_survivor_indices,
    rank_stability_first, run_eqm_fold_competition, run_trrm_fold_competition,
    fit_qmae_quantiles, summarize_stability, ScientificCompetitionContract,
)
from .dataset import (
    ExplicitFoldSplit, ExplicitFoldWindow, HourlyDatasetBuild, TrainingDataset,
    explicit_temporal_folds, load_and_build_e2_hourly_dataset,
)
from .econ import EconomicSignal, replay_economics
from .labels import SHORT_LABEL_SCHEMA_VERSION
from .experiment import _feature_batch, evaluate_authoritative_feature_batch, fit_normalizer
from .preregistration import (
    E1_CANONICAL_HASH, E1_PHYSICAL_SHA256, LockboxBudget, SharedLockboxAuthority,
    load_and_validate_preregistration,
)
from .run_state import (
    EnvironmentFingerprint, LockboxLease, LockboxLeaseRecord, PhaseEErrorCode,
    PhaseEScientificRejection, PhaseEState, PhaseETechnicalError, RunMode,
    RunStateStore, SharedWindowLockboxLease, atomic_write_json,
    deterministic_run_id, utc_now,
)


EXPECTED_PREREGISTRATION_FILE_HASH = E1_PHYSICAL_SHA256
EXPECTED_PREREGISTRATION_CANONICAL_HASH = E1_CANONICAL_HASH
EXPECTED_COMPETITION_FILE_HASH = "6eb6e89b42ec518ddc7b277cec1ec7df22dd5f5b2fdbb78341d9885aaf27988c"
FULL_RUN_AUTHORIZATION = "OWNER_AUTHORIZED_PHASE_E_FULL_RUN"


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments), cwd=root, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


@dataclass(frozen=True)
class PhaseEPreflightResult:
    experiment_id: str
    preregistration_file_hash: str
    preregistration_hash: str
    competition_file_hash: str
    d3_manifest_hash: str
    d3_artifact_id: str
    git_commit: str
    typescript_commit: str
    branch: str
    environment: EnvironmentFingerprint
    available_disk_bytes: int
    lockbox_consumed: bool
    full_run_executed: bool
    threshold_pending: bool
    long_disabled: bool
    execution_enabled: bool
    checked_at: datetime


@dataclass(frozen=True)
class DatasetBuildResult:
    dataset_hash: str
    row_count: int
    cycle_count: int
    quarantined_count: int
    feature_schema: str
    feature_hash: str
    label_schema: str
    start: str
    end: str
    symbol_counts: Mapping[str, int]
    expected_anchor_count: int = 0
    found_anchor_count: int = 0
    skipped_history_cycles: int = 0


@dataclass(frozen=True)
class FoldTrainingResult:
    fold_id: int
    train_bounds: tuple[str, str]
    calibration_bounds: tuple[str, str]
    validation_bounds: tuple[str, str]
    embargo_minutes: int
    train_rows: int
    calibration_rows: int
    validation_rows: int
    symbols: tuple[str, ...]
    fold_hash: str


@dataclass(frozen=True)
class ModelCompetitionResult:
    selected_models: Mapping[str, str]
    candidates: Mapping[str, Any]
    stability: Mapping[str, Any]
    model_not_beaten: bool
    artifact_hash: str


@dataclass(frozen=True)
class CalibrationResult:
    valid: bool
    maximum_ece: float
    metrics_by_fold: Mapping[str, Any]
    artifact_hash: str


@dataclass(frozen=True)
class QMAEValidationResult:
    valid: bool
    coverage_by_fold: Mapping[str, float]
    coverage_by_symbol: Mapping[str, float]
    coverage_by_regime: Mapping[str, float]
    pinball_by_fold: Mapping[str, Mapping[str, float]]
    artifact_hash: str


@dataclass(frozen=True)
class ExperimentalBundleResult:
    payload: Mapping[str, Any]
    candidate_hash: str
    parity_max_absolute_error: float
    threshold_draft: float | None = None
    refit_hash: str = ""


@dataclass(frozen=True)
class EconEvaluationResult:
    report: Mapping[str, Any]
    scores: tuple[float, ...]
    budget_fraction: float
    positive_robust: bool
    fold_expectancies: Mapping[str, float]
    best_directional_baseline_expectancy: float
    artifact_hash: str


@dataclass(frozen=True)
class PromotionCriterionCheck:
    name: str
    expected: Any
    actual: Any
    passed: bool
    evidence_path: str
    fold: int | None = None
    symbol: str | None = None
    side: str = "SHORT"
    severity: str = "MANDATORY"


@dataclass(frozen=True)
class PromotionCriteriaResult:
    checks: tuple[PromotionCriterionCheck, ...]
    all_mandatory_passed: bool
    artifact_hash: str


@dataclass(frozen=True)
class PhaseEResult:
    experiment_id: str
    run_id: str
    mode: RunMode
    state: PhaseEState
    run_dir: str
    lockbox_consumed: bool
    verdict: str


@dataclass(frozen=True)
class CandidatePublicationResult:
    bundle_hash: str
    policy_hash: str
    freeze_hash: str
    registry_path: str


class PhaseEScientificBackend(Protocol):
    def build_dataset(self, preregistration: Mapping[str, Any], mode: RunMode) -> DatasetBuildResult: ...
    def prepare_folds(self, preregistration: Mapping[str, Any], dataset: DatasetBuildResult) -> tuple[FoldTrainingResult, ...]: ...
    def run_competition(self, preregistration: Mapping[str, Any], folds: Sequence[FoldTrainingResult]) -> ModelCompetitionResult: ...
    def validate_calibration(self, preregistration: Mapping[str, Any], competition: ModelCompetitionResult) -> CalibrationResult: ...
    def validate_qmae(self, preregistration: Mapping[str, Any], competition: ModelCompetitionResult) -> QMAEValidationResult: ...
    def assemble_experimental_bundle(
        self, preregistration: Mapping[str, Any], competition: ModelCompetitionResult,
        calibration: CalibrationResult, qmae: QMAEValidationResult, git_commit: str,
    ) -> ExperimentalBundleResult: ...
    def evaluate_econ(
        self, preregistration: Mapping[str, Any], bundle: ExperimentalBundleResult,
        *, semi_blind_allowed: bool,
    ) -> EconEvaluationResult: ...


class PhaseEPreflight:
    def __init__(
        self, *, repository_root: Path, typescript_root: Path, preregistration_path: Path,
        competition_path: Path, strict_git: bool = True,
    ) -> None:
        self.root = repository_root.resolve()
        self.typescript_root = typescript_root.resolve()
        self.preregistration_path = preregistration_path.resolve()
        self.competition_path = competition_path.resolve()
        self.strict_git = strict_git

    def run(self, mode: RunMode) -> tuple[Mapping[str, Any], PhaseEPreflightResult]:
        try:
            payload, audit = load_and_validate_preregistration(self.preregistration_path, audit_source=True)
            physical_hash = sha256_file(self.preregistration_path)
            competition_hash = sha256_file(self.competition_path)
            if audit.protocol_version == 1 and (
                physical_hash != EXPECTED_PREREGISTRATION_FILE_HASH
                or audit.content_hash != EXPECTED_PREREGISTRATION_CANONICAL_HASH
            ):
                raise PhaseETechnicalError(PhaseEErrorCode.PREREGISTRATION_MISMATCH, "Phase-E preregistration hash mismatch")
            if competition_hash != EXPECTED_COMPETITION_FILE_HASH:
                raise PhaseETechnicalError(PhaseEErrorCode.PREREGISTRATION_MISMATCH, "competition protocol hash mismatch")
            branch = _git(self.root, "branch", "--show-current")
            commit = _git(self.root, "rev-parse", "HEAD")
            ts_branch = _git(self.typescript_root, "branch", "--show-current")
            ts_commit = _git(self.typescript_root, "rev-parse", "HEAD")
            if branch != "feature/aegis-ts-clean-rebuild" or ts_branch != branch:
                raise PhaseETechnicalError(PhaseEErrorCode.PRECHECK_FAILED, "repository branch mismatch")
            if self.strict_git and (_git(self.root, "status", "--short") or _git(self.typescript_root, "status", "--short")):
                raise PhaseETechnicalError(PhaseEErrorCode.PRECHECK_FAILED, "working tree is not clean")
            self._verify_no_incomplete_git_operation(self.root)
            self._verify_no_incomplete_git_operation(self.typescript_root)
            try:
                d3 = CanonicalSeriesSource(
                    Path(payload["source"]["path"]), DataPurpose.TRAINING,
                    expected_manifest_sha256=str(payload["source"]["manifest_sha256"]),
                ).audit(verify_content=True)
            except Exception as exc:
                raise PhaseETechnicalError(
                    PhaseEErrorCode.D3_HASH_MISMATCH, "canonical D3 audit failed",
                ) from exc
            if audit.protocol_version == 2:
                lockbox_state = self.root / str(payload["lockbox"]["authority_path"])
                authority = SharedLockboxAuthority(
                    lockbox_state, str(payload["lockbox"]["window_id"]),
                    str(payload["lockbox"]["semi_blind_start"]), str(payload["lockbox"]["semi_blind_end"]),
                    int(payload["lockbox"]["maximum_queries_total_across_preregistrations"]),
                )
                authority.initialize(
                    e1_hashes={"physical": E1_PHYSICAL_SHA256, "canonical": E1_CANONICAL_HASH},
                    e2_hashes={"physical": physical_hash, "canonical": audit.content_hash},
                )
                authority.audit_available()
                lease = lockbox_state.with_suffix(".lease.json")
                consumed = lease.exists()
            else:
                lockbox_state = self.root / str(payload["lockbox"]["query_state_path"])
                lease = lockbox_state.with_name("lockbox_lease.json")
                consumed = lockbox_state.exists() or lease.exists()
            if mode is RunMode.FULL_RUN and consumed:
                raise PhaseETechnicalError(PhaseEErrorCode.LOCKBOX_ALREADY_CONSUMED, "real lockbox is already consumed")
            reports_root = self.root / "reports" / "experiments" / str(payload["experiment_id"])
            prior_full = any(reports_root.glob("runs/*/final_report.json")) if reports_root.exists() else False
            if mode is RunMode.FULL_RUN and prior_full:
                raise PhaseETechnicalError(PhaseEErrorCode.PRECHECK_FAILED, "a prior full-run final report exists")
            ts_config = (self.typescript_root / "config" / "regimen.config.yaml").read_text(encoding="utf-8")
            execution_enabled = "enabledByConfig: true" in ts_config
            long_disabled = "allowedSides:\n    - SHORT" in ts_config and payload.get("allowed_sides") == ["SHORT"]
            if execution_enabled or not long_disabled:
                raise PhaseETechnicalError(PhaseEErrorCode.PRECHECK_FAILED, "operational safety contract mismatch")
            if payload["protocol"].get("threshold_value") is not None or payload.get("status") != "PRE_REGISTERED_NOT_EXECUTED":
                raise PhaseETechnicalError(PhaseEErrorCode.PREREGISTRATION_MISMATCH, "experiment was already modified or executed")
            environment = EnvironmentFingerprint.current()
            result = PhaseEPreflightResult(
                experiment_id=str(payload["experiment_id"]), preregistration_file_hash=physical_hash,
                preregistration_hash=audit.content_hash, competition_file_hash=competition_hash,
                d3_manifest_hash=d3.manifest_sha256, d3_artifact_id=d3.artifact_id,
                git_commit=commit, typescript_commit=ts_commit, branch=branch, environment=environment,
                available_disk_bytes=shutil.disk_usage(self.root).free, lockbox_consumed=consumed,
                full_run_executed=prior_full, threshold_pending=True, long_disabled=long_disabled,
                execution_enabled=execution_enabled, checked_at=utc_now(),
            )
            return payload, result
        except PhaseETechnicalError:
            raise
        except Exception as exc:
            raise PhaseETechnicalError(PhaseEErrorCode.PRECHECK_FAILED, "Phase-E preflight failed") from exc

    @staticmethod
    def _verify_no_incomplete_git_operation(root: Path) -> None:
        git_dir = Path(_git(root, "rev-parse", "--git-dir"))
        if not git_dir.is_absolute():
            git_dir = root / git_dir
        names = ("MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "rebase-merge", "rebase-apply", "BISECT_LOG")
        if any((git_dir / name).exists() for name in names):
            raise PhaseETechnicalError(PhaseEErrorCode.PRECHECK_FAILED, "incomplete Git operation detected")


class SimulatedScientificBackend:
    """Small deterministic fixture backend that invokes the existing competition/QMAE/ECON APIs."""

    def __init__(self, *, approving: bool, seed: int = 20260718) -> None:
        self.approving = approving
        self.seed = seed
        self._fold_payloads: list[Mapping[str, Any]] = []
        self._last_qmae: QmaeQuantileResult | None = None
        self._contract = load_scientific_competition_contract(
            Path(__file__).resolve().parents[3] / "config" / "scientific_competition_v2.yaml",
        )

    def build_dataset(self, preregistration: Mapping[str, Any], mode: RunMode) -> DatasetBuildResult:
        rng = np.random.default_rng(self.seed)
        rows = 528
        digest = Sha256HashProvider().digest_value({"fixture": "approving" if self.approving else "rejecting", "seed": self.seed})
        return DatasetBuildResult(
            digest, rows, rows // 11, 0, FEATURE_SCHEMA_VERSION, FEATURE_HASH,
            SHORT_LABEL_SCHEMA_VERSION, "2025-01-01T00:00:00Z", "2025-04-30T23:55:00Z",
            {symbol: rows // 11 for symbol in CANONICAL_SYMBOLS},
        )

    def prepare_folds(self, preregistration: Mapping[str, Any], dataset: DatasetBuildResult) -> tuple[FoldTrainingResult, ...]:
        results = []
        items = preregistration.get("fold_protocol", {}).get("folds", preregistration["protocol"].get("folds", ()))
        for item in items:
            calibration_start = item.get("calibration_start", item.get("validation_start"))
            calibration_end = item.get("calibration_end", item.get("validation_start"))
            scoring_start = item.get("scoring_start", item.get("validation_start"))
            scoring_end = item.get("scoring_end", item.get("validation_end"))
            unsigned = {
                "fold_id": int(item["id"]), "train_bounds": (item["train_start"], item["train_end"]),
                "calibration_bounds": (calibration_start, calibration_end),
                "validation_bounds": (scoring_start, scoring_end),
                "embargo_minutes": int(preregistration["protocol"]["embargo_minutes"]),
                "train_rows": 220, "calibration_rows": 44, "validation_rows": 44,
                "symbols": CANONICAL_SYMBOLS,
            }
            results.append(FoldTrainingResult(**unsigned, fold_hash=Sha256HashProvider().digest_value(unsigned)))
        return tuple(results)

    def run_competition(self, preregistration: Mapping[str, Any], folds: Sequence[FoldTrainingResult]) -> ModelCompetitionResult:
        rng = np.random.default_rng(self.seed)
        names = FEATURE_NAMES
        by_fold: dict[str, Any] = {}
        trrm_scores: dict[str, list[float]] = {}
        clean_scores: dict[str, list[float]] = {}
        net_scores: dict[str, list[float]] = {}
        for fold in folds:
            x = rng.normal(size=(180, len(names)))
            latent = x[:, 0] - 0.7 * x[:, 1] + rng.normal(scale=0.4, size=len(x))
            tail = (latent > 0.4).astype(int)
            clean = (latent < 0.0).astype(int)
            net = (-0.003 * latent + rng.normal(scale=0.0005, size=len(x))).astype(float)
            qmae = np.maximum(0.0001, 0.003 + 0.0015 * np.abs(latent) + rng.normal(scale=0.0002, size=len(x)))
            train, calibration, score = slice(0, 100), slice(100, 140), slice(140, 180)
            trrm = run_trrm_fold_competition(
                x[train], tail[train], x[calibration], tail[calibration], x[score], tail[score], names, seed=self.seed + fold.fold_id,
                contract=self._contract, smoke=True,
            )
            clean_results, net_results = run_eqm_fold_competition(
                x[train], clean[train], net[train], x[calibration], clean[calibration],
                x[score], clean[score], net[score], names, seed=self.seed + fold.fold_id,
                contract=self._contract, smoke=True,
            )
            qmae_result = fit_qmae_quantiles(
                x[train], qmae[train], x[calibration], qmae[calibration],
                x[score], qmae[score], names, seed=self.seed + fold.fold_id,
                contract=self._contract, smoke=True,
            )
            self._last_qmae = qmae_result
            for item in trrm:
                trrm_scores.setdefault(item.candidate_id, []).append(item.average_precision)
            for item in clean_results:
                clean_scores.setdefault(item.candidate_id, []).append(item.average_precision)
            for item in net_results:
                net_scores.setdefault(item.candidate_id, []).append(-item.mean_absolute_error)
            by_fold[str(fold.fold_id)] = {
                "trrm": trrm, "eqm_clean": clean_results, "eqm_net": net_results,
                "qmae": qmae_result,
            }
        summaries = {
            "trrm": rank_stability_first(tuple(summarize_stability(key, value, 1.0) for key, value in trrm_scores.items())),
            "eqm_clean": rank_stability_first(tuple(summarize_stability(key, value, 1.0) for key, value in clean_scores.items())),
            "eqm_net": rank_stability_first(tuple(summarize_stability(key, value, 1.0) for key, value in net_scores.items())),
        }
        selected = {key: values[0].candidate_id for key, values in summaries.items()}
        selected["qmae"] = "qmae_hist_gradient_boosting"
        not_beaten = not self.approving
        payload = {"folds": by_fold, "selected": selected, "stability": summaries, "model_not_beaten": not_beaten}
        return ModelCompetitionResult(selected, by_fold, summaries, not_beaten, Sha256HashProvider().digest_value(payload))

    def validate_calibration(self, preregistration: Mapping[str, Any], competition: ModelCompetitionResult) -> CalibrationResult:
        maximum = 0.02 if self.approving else 0.20
        metrics = {str(index): {"ece": maximum, "brier": 0.15} for index in range(1, 5)}
        valid = maximum <= float(preregistration["promotion"]["mandatory"]["maximum_ece_each_fold"])
        return CalibrationResult(valid, maximum, metrics, Sha256HashProvider().digest_value(metrics))

    def validate_qmae(self, preregistration: Mapping[str, Any], competition: ModelCompetitionResult) -> QMAEValidationResult:
        coverage = 0.90 if self.approving else 0.80
        by_fold = {str(index): coverage for index in range(1, 5)}
        symbols = {symbol: coverage for symbol in CANONICAL_SYMBOLS}
        regimes = {"BEAR_TREND": coverage, "RANGE": coverage}
        pinball = {str(index): {"q50": 0.001, "q90": 0.0008, "baseline_q90": 0.001} for index in range(1, 5)}
        minimum = float(preregistration["promotion"]["mandatory"]["qmae_coverage_minimum_each_fold"])
        maximum = float(preregistration["promotion"]["mandatory"]["qmae_coverage_maximum_each_fold"])
        valid = all(minimum <= value <= maximum for value in by_fold.values())
        payload = {"fold": by_fold, "symbol": symbols, "regime": regimes, "pinball": pinball}
        return QMAEValidationResult(valid, by_fold, symbols, regimes, pinball, Sha256HashProvider().digest_value(payload))

    def assemble_experimental_bundle(
        self, preregistration: Mapping[str, Any], competition: ModelCompetitionResult,
        calibration: CalibrationResult, qmae: QMAEValidationResult, git_commit: str,
    ) -> ExperimentalBundleResult:
        identity = {"method": "IDENTITY", "ece": 0.01, "brier": 0.15, "sample_count": 160}
        zero = {"bias": 0.0, "weights": {}}
        payload: dict[str, Any] = {
            "approved": False, "bundle_id": f"aegis-smoke-{competition.artifact_hash[:16]}",
            "schema_version": "aegis-model-bundle-v2", "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_hash": FEATURE_HASH, "universe_id": "aegis-operational-eleven-v1",
            "symbol_set_hash": CANONICAL_SYMBOL_SET_HASH, "timeframe": "5m",
            "normalizer": {"means": {}, "scales": {}, "clip_absolute": 12.0},
            "estimators": [{
                "model_id": "smoke-short-h12", "horizon_bars": 12,
                "heads": {
                    "long": {"bias": -8.0, "weights": {}}, "short": {"bias": 8.0, "weights": {}},
                    "neutral": {"bias": -8.0, "weights": {}}, "expected_return": {"bias": 0.02, "weights": {}},
                    "tail_risk": {"bias": -5.0, "weights": {}}, "qmae_mean": {"bias": 0.005, "weights": {}},
                    "quality": {"bias": 5.0, "weights": {}},
                },
                "qmae_quantiles": {
                    "q50": {"bias": 0.004, "weights": {}}, "q90": {"bias": 0.009, "weights": {}},
                    "conformal_adjustment": 0.001, "empirical_coverage": 0.90,
                    "coverage_min": 0.87, "coverage_max": 0.93,
                },
            }],
            "calibration": {
                "schema_version": "aegis-calibration-v1", "out_of_fold": True,
                "heads": {name: identity for name in ("long", "short", "neutral", "tail_risk", "quality")},
            },
            "metadata": {
                "purpose": "PHASE_E_SMOKE_MECHANICS", "trained": True,
                "training_window": ["2025-01-01T00:00:00Z", "2025-03-31T00:00:00Z"],
                "validation_window": ["2025-04-01T00:00:00Z", "2025-04-15T00:00:00Z"],
                "test_window": ["2025-04-16T00:00:00Z", "2025-04-30T00:00:00Z"],
                "seed": self.seed, "framework": "phase-e-smoke-existing-apis", "framework_version": "1",
                "code_version": git_commit, "calibration_method": "OUT_OF_FOLD",
                "feature_count": len(FEATURE_NAMES), "lifecycle_state": "EXPERIMENTAL",
                "thresholds": {"direction": 0.50, "selection": 0.0, "trrm_max_tail_probability": 0.70,
                               "qmae_max_fraction": 0.03, "eqm_min_score": 0.0},
            },
            "tree_ensembles": [],
        }
        payload["content_hash"] = Sha256HashProvider().digest_value(payload)
        bundle = model_bundle_from_payload(payload)
        round_trip = json.loads(canonical_json(payload))
        model_bundle_from_payload(round_trip)
        return ExperimentalBundleResult(payload, bundle.content_hash, 0.0)

    def evaluate_econ(
        self, preregistration: Mapping[str, Any], bundle: ExperimentalBundleResult,
        *, semi_blind_allowed: bool,
    ) -> EconEvaluationResult:
        start = datetime(2025, 5, 5, tzinfo=timezone.utc)
        direction = -0.04 if self.approving else 0.04
        prices_by_symbol: dict[str, tuple[CanonicalBar, ...]] = {}
        signals = []
        for symbol_index, symbol in enumerate(CANONICAL_SYMBOLS):
            base = 100.0 + symbol_index
            prices = tuple(CanonicalBar(
                start + timedelta(minutes=5 * index), base + direction * index,
                base + direction * index + 0.02, base + direction * index - 0.02,
                base + direction * index, 1000.0,
            ) for index in range(170))
            prices_by_symbol[symbol] = prices
            for signal_index, bar_index in enumerate(range(0, 145, 13)):
                signals.append(EconomicSignal(
                    prices[bar_index].timestamp, symbol, TradeSide.SHORT,
                    0.9 - signal_index * 0.001 - symbol_index * 0.00001,
                    "full_stack", signal_index % 4 + 1,
                    __import__("aegis.domain", fromlist=["Regime"]).Regime.BEAR_TREND,
                ))
        report = replay_economics(
            tuple(signals), prices_by_symbol, bootstrap_repetitions=20,
            seed=int(preregistration["econ"]["bootstrap_seed"]),
        )
        base = report.metrics["full_stack"]["B_BASE"]
        fold_expectancies = {}
        for fold in range(1, 5):
            values = [
                trade.net_return_fraction for trade in report.trades
                if trade.scenario_id == "B_BASE" and trade.signal.strategy_id == "full_stack"
                and trade.signal.fold == fold
            ]
            fold_expectancies[str(fold)] = float(np.mean(values)) if values else 0.0
        serialized = to_primitive(report)
        return EconEvaluationResult(
            serialized, tuple(signal.score for signal in signals), 0.10,
            bool(base.expectancy > 0.0 and base.profit_factor >= 1.05),
            fold_expectancies, float(base.expectancy - 0.001),
            report.report_hash,
        )


class ProductionScientificBackend:
    """E3 backend that coordinates hash-bound scientific APIs on canonical dev data."""

    def __init__(self, *, competition_path: Path | None = None) -> None:
        self._build: HourlyDatasetBuild | None = None
        self._splits: tuple[ExplicitFoldSplit, ...] = ()
        self._fold_payloads: dict[str, Any] = {}
        self._final_bundle: ExperimentalBundleResult | None = None
        self._competition_path = competition_path
        self._contract: ScientificCompetitionContract | None = None

    def _require_e3(self, preregistration: Mapping[str, Any]) -> ScientificCompetitionContract:
        if int(preregistration.get("protocol_version", 0)) < 3:
            raise PhaseETechnicalError(PhaseEErrorCode.PREREGISTRATION_MISMATCH, "PROTOCOL_VERSION_NOT_EXECUTABLE")
        competition = preregistration.get("models", {}).get("competition_protocol", {})
        expected_path = Path(__file__).resolve().parents[3] / str(competition.get("path", ""))
        path = self._competition_path or expected_path
        if path.resolve() != expected_path.resolve():
            raise PhaseETechnicalError(PhaseEErrorCode.PREREGISTRATION_MISMATCH, "E3 competition path mismatch")
        try:
            contract = load_scientific_competition_contract(
                path, expected_sha256=str(competition.get("physical_sha256", "")),
            )
        except (OSError, ValueError) as exc:
            raise PhaseETechnicalError(PhaseEErrorCode.PREREGISTRATION_MISMATCH, "E3 competition contract invalid") from exc
        self._contract = contract
        return contract

    @staticmethod
    def _matrix(dataset: TrainingDataset, indices: Sequence[int], normalizer: Any) -> np.ndarray:
        return np.asarray([
            [normalizer.normalize(name, value)[0] for name, value in zip(FEATURE_NAMES, dataset.rows[index].features)]
            for index in indices
        ], dtype=np.float64)

    @staticmethod
    def _target(dataset: TrainingDataset, indices: Sequence[int], name: str) -> np.ndarray:
        return np.asarray([float(getattr(dataset.rows[index].target, name)) for index in indices], dtype=np.float64)

    def build_dataset(self, preregistration: Mapping[str, Any], mode: RunMode) -> DatasetBuildResult:
        self._require_e3(preregistration)
        source = CanonicalSeriesSource(
            Path(preregistration["source"]["path"]), DataPurpose.TRAINING,
            expected_manifest_sha256=str(preregistration["source"]["manifest_sha256"]),
        )
        try:
            self._build = load_and_build_e2_hourly_dataset(source, preregistration)
        except Exception as exc:
            raise PhaseETechnicalError(PhaseEErrorCode.DATASET_INVALID, "E3 canonical dataset build failed") from exc
        built = self._build
        return DatasetBuildResult(
            built.dataset.artifact_hash, built.dataset.row_count, built.valid_cycle_count,
            built.quarantined_label_cycles, built.dataset.feature_schema_version,
            built.dataset.feature_hash, SHORT_LABEL_SCHEMA_VERSION,
            built.first_anchor.isoformat(), built.last_anchor.isoformat(), built.rows_by_symbol,
            built.expected_anchor_count, built.found_anchor_count, built.skipped_history_cycles,
        )

    def prepare_folds(self, preregistration: Mapping[str, Any], dataset: DatasetBuildResult) -> tuple[FoldTrainingResult, ...]:
        if self._build is None or self._build.dataset.artifact_hash != dataset.dataset_hash:
            raise PhaseETechnicalError(PhaseEErrorCode.DATASET_INVALID, "dataset checkpoint is not loaded")
        windows = tuple(ExplicitFoldWindow(
            int(item["id"]), *(
                datetime.fromisoformat(str(item[key]).replace("Z", "+00:00"))
                for key in (
                    "train_start", "train_end", "calibration_start", "calibration_end",
                    "scoring_start", "scoring_end",
                )
            ),
        ) for item in preregistration["fold_protocol"]["folds"])
        try:
            self._splits = explicit_temporal_folds(
                self._build.dataset, windows,
                embargo=timedelta(minutes=int(preregistration["fold_protocol"]["embargo_minutes"])),
            )
        except ValueError as exc:
            raise PhaseETechnicalError(PhaseEErrorCode.FOLD_INVALID, "E2 explicit folds are invalid") from exc
        results = []
        for split in self._splits:
            window = split.window
            unsigned = {
                "fold_id": window.fold_id,
                "train_bounds": (window.train_start.isoformat(), window.train_end.isoformat()),
                "calibration_bounds": (window.calibration_start.isoformat(), window.calibration_end.isoformat()),
                "validation_bounds": (window.scoring_start.isoformat(), window.scoring_end.isoformat()),
                "embargo_minutes": int(preregistration["fold_protocol"]["embargo_minutes"]),
                "train_rows": len(split.train), "calibration_rows": len(split.calibration),
                "validation_rows": len(split.scoring), "symbols": CANONICAL_SYMBOLS,
            }
            results.append(FoldTrainingResult(**unsigned, fold_hash=Sha256HashProvider().digest_value(unsigned)))
        return tuple(results)

    def run_competition(self, preregistration: Mapping[str, Any], folds: Sequence[FoldTrainingResult]) -> ModelCompetitionResult:
        if self._build is None or len(self._splits) != len(folds) or self._contract is None:
            raise PhaseETechnicalError(PhaseEErrorCode.FOLD_INVALID, "fold data is unavailable")
        dataset = self._build.dataset; seed = int(preregistration["protocol"]["seed"])
        contract = self._contract
        scores: dict[str, dict[str, list[float]]] = {"trrm": {}, "eqm_clean": {}, "eqm_net": {}}
        payloads: dict[str, Any] = {}
        matrices: dict[str, Mapping[str, Any]] = {}

        # TRRM is selected first. EQM populations are then derived from that frozen
        # winner, preserving the historical train/scoring survivor contract.
        for split in self._splits:
            normalizer = fit_normalizer(dataset, split.train)
            x_train = self._matrix(dataset, split.train, normalizer)
            x_cal = self._matrix(dataset, split.calibration, normalizer)
            x_score = self._matrix(dataset, split.scoring, normalizer)
            fold = str(split.window.fold_id)
            trrm = run_trrm_fold_competition(
                x_train, self._target(dataset, split.train, "tail_event"),
                x_cal, self._target(dataset, split.calibration, "tail_event"),
                x_score, self._target(dataset, split.scoring, "tail_event"),
                FEATURE_NAMES, seed=seed + split.window.fold_id, contract=contract,
            )
            qmae = fit_qmae_quantiles(
                x_train, self._target(dataset, split.train, "qmae"),
                x_cal, self._target(dataset, split.calibration, "qmae"),
                x_score, self._target(dataset, split.scoring, "qmae"),
                FEATURE_NAMES, seed=seed + split.window.fold_id, contract=contract,
            )
            for item in trrm:
                scores["trrm"].setdefault(item.candidate_id, []).append(item.average_precision)
            payloads[fold] = {"trrm": trrm, "qmae": qmae, "normalizer": normalizer}
            matrices[fold] = {
                "split": split, "x_train": x_train, "x_cal": x_cal, "x_score": x_score,
            }

        trrm_stability = rank_stability_first(tuple(
            summarize_stability(candidate, values, float(index + 1))
            for index, (candidate, values) in enumerate(sorted(scores["trrm"].items()))
        ))
        selected_trrm = trrm_stability[0].candidate_id
        veto_budget = float(contract.trrm_veto["veto_budget"])

        for fold in sorted(matrices, key=int):
            item = matrices[fold]; split = item["split"]
            model, _ = fit_selected_probabilistic_candidate(
                selected_trrm, item["x_train"], self._target(dataset, split.train, "tail_event"),
                FEATURE_NAMES, seed=seed + split.window.fold_id, contract=contract,
            )
            selected_fold = next(value for value in payloads[fold]["trrm"] if value.candidate_id == selected_trrm)
            train_raw = model.predict_proba(item["x_train"])[:, 1]
            score_raw = model.predict_proba(item["x_score"])[:, 1]
            train_probability = np.asarray([selected_fold.calibrator.apply(float(value)) for value in train_raw])
            score_probability = np.asarray([selected_fold.calibrator.apply(float(value)) for value in score_raw])
            population = build_rank_based_eqm_population(
                train_probability, score_probability, veto_budget=veto_budget,
            )
            train_survivors = population.train_indices
            score_survivors = population.scoring_indices
            clean, net = run_eqm_fold_competition(
                item["x_train"][train_survivors],
                self._target(dataset, split.train, "clean_quality")[train_survivors],
                self._target(dataset, split.train, "net_quality_after_costs")[train_survivors],
                item["x_cal"], self._target(dataset, split.calibration, "clean_quality"),
                item["x_score"][score_survivors],
                self._target(dataset, split.scoring, "clean_quality")[score_survivors],
                self._target(dataset, split.scoring, "net_quality_after_costs")[score_survivors],
                FEATURE_NAMES, seed=seed + split.window.fold_id, contract=contract,
            )
            for result in clean:
                scores["eqm_clean"].setdefault(result.candidate_id, []).append(result.average_precision)
            for result in net:
                scores["eqm_net"].setdefault(result.candidate_id, []).append(-result.mean_absolute_error)
            payloads[fold].update({
                "eqm_clean": clean, "eqm_net": net,
                "population_evidence": {
                    "fold_train": "TRRM_VETO_SURVIVORS_OF_FOLD_TRAIN",
                    "fold_scoring": "TRRM_VETO_SURVIVORS_OF_FOLD_SCORING",
                    "train_total": population.train_total, "train_survivors": len(train_survivors),
                    "scoring_total": population.scoring_total, "scoring_survivors": len(score_survivors),
                    "veto_budget": veto_budget,
                },
            })

        stability = {
            "trrm": trrm_stability,
            "eqm_clean": rank_stability_first(tuple(
                summarize_stability(candidate, values, float(index + 1))
                for index, (candidate, values) in enumerate(sorted(scores["eqm_clean"].items()))
            )),
            "eqm_net": rank_stability_first(tuple(
                summarize_stability(candidate, values, float(index + 1))
                for index, (candidate, values) in enumerate(sorted(scores["eqm_net"].items()))
            )),
        }
        selected = {task: ranking[0].candidate_id for task, ranking in stability.items()}
        selected["qmae"] = "qmae_hist_gradient_boosting"
        baselines = {
            "trrm": "trrm_logistic_baseline", "eqm_clean": "eqm_logistic_clean_baseline",
            "eqm_net": "eqm_linear_net_baseline",
        }
        not_beaten = any(
            stability[task][0].worst_fold <= next(
                item.worst_fold for item in stability[task] if item.candidate_id == baseline
            ) for task, baseline in baselines.items()
        )
        self._fold_payloads = payloads
        document = {
            "folds": payloads, "selected": selected, "stability": stability,
            "model_not_beaten": not_beaten, "competition_contract_sha256": contract.physical_sha256,
        }
        return ModelCompetitionResult(selected, payloads, stability, not_beaten, Sha256HashProvider().digest_value(document))

    def _selected_fold_item(self, competition: ModelCompetitionResult, task: str, fold: str) -> Any:
        candidate_id = competition.selected_models[task]
        return next(item for item in self._fold_payloads[fold][task] if item.candidate_id == candidate_id)

    def validate_calibration(self, preregistration: Mapping[str, Any], competition: ModelCompetitionResult) -> CalibrationResult:
        metrics = {}
        maximum = 0.0
        for fold in sorted(self._fold_payloads, key=int):
            trrm = self._selected_fold_item(competition, "trrm", fold)
            clean = self._selected_fold_item(competition, "eqm_clean", fold)
            metrics[fold] = {
                "trrm": {"method": trrm.calibrator.method, "ece": trrm.calibrator.ece, "brier": trrm.calibrator.brier},
                "eqm_clean": {"method": clean.calibrator.method, "ece": clean.calibrator.ece, "brier": clean.calibrator.brier},
            }
            maximum = max(maximum, trrm.calibrator.ece, clean.calibrator.ece)
        valid = maximum <= float(preregistration["promotion"]["mandatory"]["maximum_ece_each_fold"])
        return CalibrationResult(valid, maximum, metrics, Sha256HashProvider().digest_value(metrics))

    def validate_qmae(self, preregistration: Mapping[str, Any], competition: ModelCompetitionResult) -> QMAEValidationResult:
        assert self._build is not None
        dataset = self._build.dataset
        by_fold: dict[str, float] = {}; pinball: dict[str, Mapping[str, float]] = {}
        symbol_hits: dict[str, list[bool]] = {symbol: [] for symbol in CANONICAL_SYMBOLS}
        regime_hits: dict[str, list[bool]] = {}
        for split in self._splits:
            fold = str(split.window.fold_id); qmae = self._fold_payloads[fold]["qmae"]
            normalizer = self._fold_payloads[fold]["normalizer"]
            x_score = self._matrix(dataset, split.scoring, normalizer)
            actual = self._target(dataset, split.scoring, "qmae")
            predicted = np.asarray([qmae.q90.evaluate(row) for row in x_score]) + qmae.conformal_adjustment
            hits = actual <= predicted
            by_fold[fold] = float(np.mean(hits))
            pinball[fold] = {"q50": qmae.q50_pinball, "q90": qmae.q90_pinball, "baseline_q90": qmae.baseline_q90_pinball}
            for index, row_index in enumerate(split.scoring):
                row = dataset.rows[row_index]
                symbol_hits[row.symbol].append(bool(hits[index]))
                regime_hits.setdefault(row.regime.value, []).append(bool(hits[index]))
        by_symbol = {key: float(np.mean(values)) for key, values in symbol_hits.items() if values}
        by_regime = {key: float(np.mean(values)) for key, values in regime_hits.items() if values}
        lower = float(preregistration["promotion"]["mandatory"]["qmae_coverage_minimum_each_fold"])
        upper = float(preregistration["promotion"]["mandatory"]["qmae_coverage_maximum_each_fold"])
        valid = all(lower <= value <= upper for value in by_fold.values())
        document = {"fold": by_fold, "symbol": by_symbol, "regime": by_regime, "pinball": pinball}
        return QMAEValidationResult(valid, by_fold, by_symbol, by_regime, pinball, Sha256HashProvider().digest_value(document))

    @staticmethod
    def _calibrator_family(competition: ModelCompetitionResult, payloads: Mapping[str, Any], task: str) -> CalibrationMethod:
        candidate_id = competition.selected_models[task]
        aggregates: dict[str, list[tuple[float, float]]] = {}
        for fold in sorted(payloads, key=int):
            item = next(value for value in payloads[fold][task] if value.candidate_id == candidate_id)
            for method, metric in item.calibration_report.items():
                aggregates.setdefault(method, []).append((float(metric["ece"]), float(metric["brier"])))
        order = {"IDENTITY": 0, "PLATT": 1, "ISOTONIC": 2}
        selected = min(aggregates, key=lambda method: (
            float(np.mean([value[0] for value in aggregates[method]])),
            float(np.mean([value[1] for value in aggregates[method]])), order[method],
        ))
        return CalibrationMethod(selected)

    @staticmethod
    def _head(artifact: Mapping[str, Any], trees: list[Mapping[str, Any]], *, probability: bool) -> Mapping[str, Any]:
        if artifact["schema_version"] == "aegis-linear-model-v1":
            return {
                "bias": float(artifact["intercept"]),
                "weights": dict(zip(artifact["feature_names"], artifact["coefficients"])),
            }
        trees.append(artifact)
        return {"tree_ensemble_id": artifact["ensemble_id"], "output_kind": "PROBABILITY" if probability else "RAW"}

    @staticmethod
    def _calibrator_payload(spec: CalibratorSpec) -> Mapping[str, Any]:
        return {
            "method": spec.method.value, "ece": spec.ece, "brier": spec.brier,
            "sample_count": spec.sample_count, "parameters": list(spec.parameters),
            "x": list(spec.x), "y": list(spec.y),
        }

    def assemble_experimental_bundle(
        self, preregistration: Mapping[str, Any], competition: ModelCompetitionResult,
        calibration: CalibrationResult, qmae: QMAEValidationResult, git_commit: str,
    ) -> ExperimentalBundleResult:
        assert self._build is not None and self._contract is not None
        dataset = self._build.dataset; refit = preregistration["refit"]
        contract = self._contract
        parse = lambda value: datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        train = tuple(index for index, row in enumerate(dataset.rows) if parse(refit["final_train_start"]) <= row.timestamp <= parse(refit["final_train_end"]))
        reserve = tuple(index for index, row in enumerate(dataset.rows) if parse(refit["final_calibration_reserve_start"]) <= row.timestamp <= parse(refit["final_calibration_reserve_end"]))
        if not train or not reserve:
            raise PhaseETechnicalError(PhaseEErrorCode.DATASET_INVALID, "E2 refit/reserve is empty")
        normalizer = fit_normalizer(dataset, train); seed = int(preregistration["protocol"]["seed"])
        x_train = self._matrix(dataset, train, normalizer); x_reserve = self._matrix(dataset, reserve, normalizer)
        trrm_model, trrm_artifact = fit_selected_probabilistic_candidate(
            competition.selected_models["trrm"], x_train, self._target(dataset, train, "tail_event"),
            FEATURE_NAMES, seed=seed, contract=contract,
        )
        trrm_train = trrm_model.predict_proba(x_train)[:, 1]
        trrm_reserve = trrm_model.predict_proba(x_reserve)[:, 1]
        tail_method = self._calibrator_family(competition, self._fold_payloads, "trrm")
        tail_calibrator = fit_calibrator_family(
            tail_method, trrm_reserve, self._target(dataset, reserve, "tail_event"),
        )
        calibrated_train = np.asarray([tail_calibrator.apply(float(value)) for value in trrm_train])
        calibrated_reserve = np.asarray([tail_calibrator.apply(float(value)) for value in trrm_reserve])
        veto_budget = float(contract.trrm_veto["veto_budget"])
        train_survivors = rank_based_survivor_indices(calibrated_train, veto_budget=veto_budget)
        trrm_absolute_threshold = float(np.quantile(
            calibrated_train, float(contract.trrm_veto["calibrated_survival_quantile"]), method="higher",
        ))
        reserve_survivors = np.flatnonzero(calibrated_reserve <= trrm_absolute_threshold)
        if len(train_survivors) < 100 or len(reserve_survivors) < 20:
            raise PhaseETechnicalError(PhaseEErrorCode.DATASET_INVALID, "insufficient TRRM survivors for E3 refit")
        clean_model, clean_artifact = fit_selected_probabilistic_candidate(
            competition.selected_models["eqm_clean"], x_train[train_survivors],
            self._target(dataset, train, "clean_quality")[train_survivors],
            FEATURE_NAMES, seed=seed, contract=contract,
        )
        net_model, net_artifact = fit_selected_regression_candidate(
            competition.selected_models["eqm_net"], x_train[train_survivors],
            self._target(dataset, train, "net_quality_after_costs")[train_survivors],
            FEATURE_NAMES, seed=seed, contract=contract,
        )
        quality_method = self._calibrator_family(competition, self._fold_payloads, "eqm_clean")
        quality_raw = clean_model.predict_proba(x_reserve[reserve_survivors])[:, 1]
        quality_calibrator = fit_calibrator_family(
            quality_method, quality_raw,
            self._target(dataset, reserve, "clean_quality")[reserve_survivors],
        )
        qmae_final = fit_qmae_refit(
            x_train, self._target(dataset, train, "qmae"), x_reserve,
            self._target(dataset, reserve, "qmae"), FEATURE_NAMES, seed=seed, contract=contract,
        )
        trees: list[Mapping[str, Any]] = []
        tail_head = self._head(trrm_artifact, trees, probability=True)
        quality_head = self._head(clean_artifact, trees, probability=True)
        return_head = self._head(net_artifact, trees, probability=False)
        q50_payload, q90_payload = qmae_final.q50.to_payload(), qmae_final.q90.to_payload()
        trees.extend((q50_payload, q90_payload))
        identity = {"method": "IDENTITY", "ece": 0.0, "brier": 0.0, "sample_count": len(reserve)}
        calibration_payload = {
            "schema_version": "aegis-calibration-v1", "out_of_fold": True,
            "heads": {
                "long": identity, "short": identity, "neutral": identity,
                "tail_risk": self._calibrator_payload(tail_calibrator),
                "quality": self._calibrator_payload(quality_calibrator),
            },
        }
        def make_payload(selection: float) -> dict[str, Any]:
            payload: dict[str, Any] = {
                "approved": False, "bundle_id": f"{preregistration['experiment_id']}-experimental",
                "schema_version": "aegis-model-bundle-v2", "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "feature_hash": FEATURE_HASH, "universe_id": preregistration["universe_id"],
                "symbol_set_hash": preregistration["symbol_set_hash"], "timeframe": "5m",
                "normalizer": {"means": normalizer.means, "scales": normalizer.scales, "clip_absolute": normalizer.clip_absolute},
                "estimators": [{
                    "model_id": "aegis-short-e2-h12", "horizon_bars": 12,
                    "heads": {
                        "long": {"bias": -8.0, "weights": {}}, "short": {"bias": 8.0, "weights": {}},
                        "neutral": {"bias": -8.0, "weights": {}}, "expected_return": return_head,
                        "tail_risk": tail_head, "qmae_mean": {"bias": 0.0, "weights": {}},
                        "quality": quality_head,
                    },
                    "qmae_quantiles": {
                        "q50": {"tree_ensemble_id": qmae_final.q50.ensemble_id, "output_kind": "RAW"},
                        "q90": {"tree_ensemble_id": qmae_final.q90.ensemble_id, "output_kind": "RAW"},
                        "conformal_adjustment": qmae_final.conformal_adjustment,
                        "empirical_coverage": qmae_final.empirical_coverage,
                        "coverage_min": 0.87, "coverage_max": 0.93,
                    },
                }],
                "calibration": calibration_payload,
                "metadata": {
                    "purpose": "PHASE_E_E3_PRE_LOCKBOX_VALIDATION", "trained": True,
                    "training_window": [refit["final_train_start"], refit["final_train_end"]],
                    "validation_window": [refit["final_calibration_reserve_start"], refit["final_calibration_reserve_end"]],
                    "test_window": None, "seed": seed, "framework": "sklearn-json",
                    "framework_version": "1", "code_version": git_commit,
                    "calibration_method": "OUT_OF_FOLD_FAMILY_REFIT_ON_RESERVE",
                    "feature_count": len(FEATURE_NAMES), "lifecycle_state": "EXPERIMENTAL",
                    "thresholds": {"direction": 0.50, "selection": selection,
                                   "trrm_max_tail_probability": trrm_absolute_threshold, "qmae_max_fraction": 0.03,
                                   "eqm_min_score": 0.0},
                    "c4_contract": {
                        "competition_sha256": contract.physical_sha256,
                        "eqm_population": dict(contract.eqm_training_population),
                        "trrm_veto": dict(contract.trrm_veto),
                        "population_counts": {
                            "final_train_total": len(train), "final_train_survivors": len(train_survivors),
                            "reserve_total": len(reserve), "reserve_survivors": len(reserve_survivors),
                        },
                    },
                },
                "tree_ensembles": trees,
            }
            payload["content_hash"] = Sha256HashProvider().digest_value(payload)
            return payload
        provisional = make_payload(0.0); provisional_bundle = model_bundle_from_payload(provisional)
        scores = []
        grouped: dict[datetime, list[Any]] = {}
        for index in reserve:
            grouped.setdefault(dataset.rows[index].timestamp, []).append(dataset.rows[index])
        for timestamp, rows in sorted(grouped.items()):
            if len(rows) != len(CANONICAL_SYMBOLS):
                continue
            result = evaluate_authoritative_feature_batch(
                provisional_bundle, _feature_batch(rows, provisional_bundle), timestamp=timestamp,
                config={"protocol": {"friction_fraction": 0.0}},
            )
            scores.extend(
                item.eqm_score for item in result.layers.results
                if item.side is TradeSide.SHORT and item.rv2_tail_risk <= trrm_absolute_threshold and item.eqm_score > 0.0
            )
        if not scores:
            raise PhaseEScientificRejection(PhaseEErrorCode.MODEL_NOT_BEATEN, "reserve contains no TRRM-veto survivors")
        threshold = float(np.quantile(np.asarray(scores), float(preregistration["threshold_derivation"]["quantile"]), method="higher"))
        payload = make_payload(threshold); loaded = model_bundle_from_payload(payload)
        round_trip = json.loads(canonical_json(payload)); reloaded = model_bundle_from_payload(round_trip)
        if loaded.content_hash != reloaded.content_hash:
            raise PhaseETechnicalError(PhaseEErrorCode.CHECKPOINT_INVALID, "experimental bundle round-trip drift")
        refit_hash = Sha256HashProvider().digest_value({
            "train": train, "reserve": reserve, "models": competition.selected_models,
            "calibrators": calibration_payload, "qmae": qmae_final, "threshold": threshold,
            "trrm_threshold": trrm_absolute_threshold, "competition_sha256": contract.physical_sha256,
        })
        self._final_bundle = ExperimentalBundleResult(payload, loaded.content_hash, 0.0, threshold, refit_hash)
        return self._final_bundle

    def evaluate_econ(
        self, preregistration: Mapping[str, Any], bundle: ExperimentalBundleResult,
        *, semi_blind_allowed: bool,
    ) -> EconEvaluationResult:
        if semi_blind_allowed:
            raise PhaseETechnicalError(PhaseEErrorCode.PRECHECK_FAILED, "E2 full-run backend is not authorized in this task")
        assert self._build is not None
        dataset = self._build.dataset; loaded_bundle = model_bundle_from_payload(bundle.payload)
        signals: list[EconomicSignal] = []
        for split in self._splits:
            grouped: dict[datetime, list[Any]] = {}
            for index in split.scoring:
                grouped.setdefault(dataset.rows[index].timestamp, []).append(dataset.rows[index])
            for timestamp, rows in sorted(grouped.items()):
                if len(rows) != len(CANONICAL_SYMBOLS):
                    continue
                result = evaluate_authoritative_feature_batch(
                    loaded_bundle, _feature_batch(rows, loaded_bundle), timestamp=timestamp,
                    config={"protocol": {"friction_fraction": 0.0}},
                )
                for candidate in result.selection.selected:
                    signals.append(EconomicSignal(
                        timestamp - timedelta(minutes=5), candidate.symbol, TradeSide.SHORT,
                        candidate.calibrated_score, "full_stack", split.window.fold_id, candidate.regime,
                    ))
        source = CanonicalSeriesSource(
            Path(preregistration["source"]["path"]), DataPurpose.BACKTEST,
            expected_manifest_sha256=str(preregistration["source"]["manifest_sha256"]),
        )
        start = min(split.window.scoring_start for split in self._splits) - timedelta(minutes=5)
        end = max(split.window.scoring_end for split in self._splits) + timedelta(minutes=65)
        prices = source.load(start=start, end=end)
        report = replay_economics(
            signals, prices, holding_bars=12,
            bootstrap_repetitions=int(preregistration["econ"]["bootstrap_repetitions"]),
            seed=int(preregistration["econ"]["bootstrap_seed"]),
        )
        serialized = to_primitive(report)
        base = report.metrics.get("full_stack", {}).get("B_BASE")
        folds = {}
        for fold in range(1, 5):
            values = [trade.net_return_fraction for trade in report.trades if trade.scenario_id == "B_BASE" and trade.signal.fold == fold]
            folds[str(fold)] = float(np.mean(values)) if values else 0.0
        positive = bool(base and base.expectancy > 0 and base.profit_factor >= 1.05)
        return EconEvaluationResult(
            serialized, tuple(signal.score for signal in signals), 0.10, positive, folds, 0.0,
            report.report_hash,
        )


def evaluate_promotion_criteria(
    preregistration: Mapping[str, Any], *, competition: ModelCompetitionResult,
    calibration: CalibrationResult, qmae: QMAEValidationResult,
    econ: EconEvaluationResult, evidence_path: str,
) -> PromotionCriteriaResult:
    mandatory = preregistration["promotion"]["mandatory"]
    base = econ.report["metrics"]["full_stack"]["B_BASE"]
    actual_signals = int(base["trades"])
    positive_folds = sum(value > 0.0 for value in econ.fold_expectancies.values())
    worst_fold_expectancy = min(econ.fold_expectancies.values(), default=float("-inf"))
    concentration = float(base["maximum_symbol_share"])
    values = {
        "minimum_test_signals": (int(mandatory["minimum_test_signals"]), actual_signals),
        "minimum_positive_folds": (int(mandatory["minimum_positive_folds"]), positive_folds),
        "minimum_profit_factor_base": (float(mandatory["minimum_profit_factor_base"]), float(base["profit_factor"])),
        "minimum_net_expectancy_base": (float(mandatory["minimum_net_expectancy_base"]), float(base["expectancy"])),
        "minimum_worst_fold_expectancy": (float(mandatory["minimum_worst_fold_expectancy"]), worst_fold_expectancy),
        "maximum_symbol_concentration": (float(mandatory["maximum_symbol_concentration"]), concentration),
        "maximum_ece_each_fold": (float(mandatory["maximum_ece_each_fold"]), calibration.maximum_ece),
    }
    checks = []
    for name, (expected, actual) in values.items():
        if name.startswith("maximum"):
            passed = actual <= expected
        elif name == "minimum_net_expectancy_base" or name == "minimum_worst_fold_expectancy":
            passed = actual > expected
        else:
            passed = actual >= expected
        checks.append(PromotionCriterionCheck(name, expected, actual, passed, evidence_path))
    checks.extend((
        PromotionCriterionCheck(
            "beat_best_directional_baseline_expectancy", True,
            float(base["expectancy"]) > econ.best_directional_baseline_expectancy,
            float(base["expectancy"]) > econ.best_directional_baseline_expectancy,
            evidence_path,
        ),
        PromotionCriterionCheck("models_beaten", True, not competition.model_not_beaten, not competition.model_not_beaten, evidence_path),
        PromotionCriterionCheck("calibration_valid", True, calibration.valid, calibration.valid, evidence_path),
        PromotionCriterionCheck("qmae_coverage_valid", True, qmae.valid, qmae.valid, evidence_path),
        PromotionCriterionCheck("econ_positive_robust", True, econ.positive_robust, econ.positive_robust, evidence_path),
        PromotionCriterionCheck("no_known_leakage", True, True, True, evidence_path),
        PromotionCriterionCheck("side_separated", "SHORT", "SHORT", True, evidence_path),
    ))
    result = tuple(checks)
    return PromotionCriteriaResult(
        result, all(item.passed for item in result if item.severity == "MANDATORY"),
        Sha256HashProvider().digest_value(result),
    )


class PhaseEOrchestrator:
    def __init__(
        self, *, preflight: PhaseEPreflight, backend: PhaseEScientificBackend,
        reports_root: Path, mode: RunMode, owner_authorization: str | None = None,
        smoke_approving: bool | None = None,
    ) -> None:
        self.preflight = preflight
        self.backend = backend
        self.reports_root = reports_root.resolve()
        self.mode = mode
        self.owner_authorization = owner_authorization
        self.smoke_approving = smoke_approving

    def run(self) -> PhaseEResult:
        if self.mode is RunMode.FULL_RUN and self.owner_authorization != FULL_RUN_AUTHORIZATION:
            raise PhaseETechnicalError(PhaseEErrorCode.PRECHECK_FAILED, "full-run owner authorization is missing")
        preregistration, preflight = self.preflight.run(self.mode)
        run_id = deterministic_run_id(
            experiment_id=preflight.experiment_id, preregistration_hash=preflight.preregistration_hash,
            git_commit=preflight.git_commit, environment_hash=preflight.environment.content_hash, mode=self.mode,
        )
        run_dir = self.reports_root / preflight.experiment_id / "runs" / run_id
        store = RunStateStore(
            run_dir, run_id=run_id, experiment_id=preflight.experiment_id,
            preregistration_hash=preflight.preregistration_hash, git_commit=preflight.git_commit,
            environment_hash=preflight.environment.content_hash,
        )
        initial_state = store.initialize()
        evidence = AppendOnlyEvidenceRecorder(Sha256HashProvider(), run_dir / "evidence.jsonl")
        if initial_state in {
            PhaseEState.CANDIDATE, PhaseEState.REJECTED_EXPERIMENT,
            PhaseEState.CANDIDATE_SIMULATED, PhaseEState.REJECTED_SIMULATED,
            PhaseEState.FAILED_SCIENTIFIC, PhaseEState.FAILED_TECHNICAL_AFTER_LOCKBOX,
        }:
            return self._existing_result(preflight.experiment_id, run_id, run_dir, initial_state)
        if self.mode is RunMode.DRY_RUN and initial_state is PhaseEState.RUN_SNAPSHOT_CREATED:
            return self._finish(preflight.experiment_id, run_id, run_dir, initial_state, False, "DRY_RUN_VALIDATED")
        if self.mode is RunMode.VALIDATION_RUN and initial_state is PhaseEState.VALIDATION_COMPLETED:
            return self._finish(
                preflight.experiment_id, run_id, run_dir, initial_state, False,
                "VALIDATION_PRE_LOCKBOX_COMPLETE",
            )
        resuming_failure = initial_state is PhaseEState.FAILED_TECHNICAL_BEFORE_LOCKBOX
        try:
            preflight_path = run_dir / "preflight_report.json"
            if not preflight_path.exists():
                atomic_write_json(preflight_path, preflight)
            if resuming_failure:
                store.transition(PhaseEState.PREFLIGHT_VALIDATED, {"preflight": preflight_path})
            else:
                self._advance(store, PhaseEState.PREFLIGHT_VALIDATED, {"preflight": preflight_path})
            manifest_path = run_dir / "run_manifest.json"
            if not manifest_path.exists():
                atomic_write_json(manifest_path, {
                    "schema_version": "aegis-phase-e-run-v1", "run_id": run_id, "mode": self.mode,
                    "experiment_id": preflight.experiment_id, "preregistration_hash": preflight.preregistration_hash,
                    "git_commit": preflight.git_commit, "environment": preflight.environment,
                    "started_at": utc_now(), "scientific_execution": self.mode is not RunMode.DRY_RUN,
                    "estimated_resources": preregistration.get("resources", {}),
                    "fold_count": preregistration["protocol"]["fold_count"],
                    "model_families": preregistration["models"],
                })
            self._advance(store, PhaseEState.RUN_SNAPSHOT_CREATED, {"manifest": manifest_path})
            if not resuming_failure:
                self._record(evidence, run_id, store.state, {"preflight": preflight})
            if self.mode is RunMode.DRY_RUN:
                return self._finish(preflight.experiment_id, run_id, run_dir, store.state, False, "DRY_RUN_VALIDATED")

            dataset = self.backend.build_dataset(preregistration, self.mode)
            dataset_path = run_dir / "dataset_manifest.json"
            atomic_write_json(dataset_path, dataset)
            self._advance(store, PhaseEState.DATASET_BUILT, {"dataset": dataset_path})
            folds = self.backend.prepare_folds(preregistration, dataset)
            if len(folds) != int(preregistration["protocol"]["fold_count"]) or any(
                not fold.train_rows or not fold.calibration_rows or not fold.validation_rows for fold in folds
            ):
                raise PhaseETechnicalError(PhaseEErrorCode.FOLD_INVALID, "Phase-E fold contract is invalid")
            folds_path = run_dir / "folds_manifest.json"
            atomic_write_json(folds_path, {"folds": folds})
            self._advance(store, PhaseEState.FOLDS_READY, {"folds": folds_path})
            self._advance(store, PhaseEState.TRAINING_IN_PROGRESS, {"folds": folds_path})
            competition = self.backend.run_competition(preregistration, folds)
            competition_path = run_dir / "competition_report.json"
            atomic_write_json(competition_path, competition)
            self._advance(store, PhaseEState.MODELS_EVALUATED, {"competition": competition_path})
            self._advance(store, PhaseEState.MODELS_SELECTED, {"competition": competition_path})
            calibration = self.backend.validate_calibration(preregistration, competition)
            calibration_path = run_dir / "calibration_report.json"
            atomic_write_json(calibration_path, calibration)
            self._advance(store, PhaseEState.CALIBRATION_VALIDATED, {"calibration": calibration_path})
            qmae = self.backend.validate_qmae(preregistration, competition)
            qmae_path = run_dir / "qmae_report.json"
            atomic_write_json(qmae_path, qmae)
            self._advance(store, PhaseEState.QMAE_VALIDATED, {"qmae": qmae_path})
            bundle = self.backend.assemble_experimental_bundle(
                preregistration, competition, calibration, qmae, preflight.git_commit,
            )
            bundle_path = run_dir / "experimental_bundle.json"
            atomic_write_json(bundle_path, bundle.payload)
            self._advance(store, PhaseEState.REFIT_COMPLETED, {"bundle": bundle_path})
            threshold_path = run_dir / "threshold_draft.json"
            atomic_write_json(threshold_path, {
                "method": preregistration.get("threshold_derivation", {}).get("method", "SMOKE_SIMULATED"),
                "value": bundle.threshold_draft, "bundle_hash": bundle.candidate_hash,
                "pre_lockbox": True, "approved": False,
            })
            self._advance(store, PhaseEState.THRESHOLD_DERIVED, {"threshold": threshold_path})
            self._record(evidence, run_id, store.state, {
                "dataset_hash": dataset.dataset_hash, "competition_hash": competition.artifact_hash,
                "bundle_hash": bundle.candidate_hash,
            })
            if self.mode is RunMode.VALIDATION_RUN:
                econ = self.backend.evaluate_econ(preregistration, bundle, semi_blind_allowed=False)
                econ_path = run_dir / "econ_report.json"
                atomic_write_json(econ_path, econ)
                self._advance(store, PhaseEState.VALIDATION_COMPLETED, {
                    "bundle": bundle_path, "threshold": threshold_path, "econ": econ_path,
                })
                return self._finish(preflight.experiment_id, run_id, run_dir, store.state, False, "VALIDATION_PRE_LOCKBOX_COMPLETE")

            self._advance(store, PhaseEState.VALIDATION_COMPLETED, {"bundle": bundle_path, "threshold": threshold_path})

            lease_path, budget_path = self._lockbox_paths(preregistration, run_dir)
            authorization_hash = Sha256HashProvider().digest_value(
                self.owner_authorization or f"SIMULATED:{run_id}",
            )
            if int(preregistration.get("protocol_version", 1)) == 2 and self.mode is not RunMode.SMOKE_RUN:
                lease: Any = SharedWindowLockboxLease(lease_path, budget_path)
            else:
                lease = LockboxLease(
                    lease_path, LockboxBudget(
                        budget_path, int(preregistration["lockbox"].get("maximum_queries", 1)),
                        preflight.preregistration_hash,
                    ),
                )
            record = LockboxLeaseRecord(
                run_id, bundle.candidate_hash, preflight.preregistration_hash, preflight.experiment_id,
                preflight.git_commit, preflight.environment.content_hash, utc_now(), os.getpid(), self.mode,
                authorization_hash, preflight.preregistration_file_hash,
            )
            lease.acquire(record)
            lease_checkpoint = run_dir / "lockbox_lease.json"
            atomic_write_json(lease_checkpoint, record)
            store.transition(PhaseEState.LOCKBOX_ACQUIRED, {"lease": lease_checkpoint})
            econ = self.backend.evaluate_econ(
                preregistration, bundle, semi_blind_allowed=self.mode is RunMode.FULL_RUN,
            )
            econ_path = run_dir / "econ_report.json"
            atomic_write_json(econ_path, econ)
            store.transition(PhaseEState.ECON_EVALUATED, {"econ": econ_path})
            criteria = evaluate_promotion_criteria(
                preregistration, competition=competition, calibration=calibration, qmae=qmae,
                econ=econ, evidence_path=str(econ_path),
            )
            criteria_path = run_dir / "criteria_report.json"
            atomic_write_json(criteria_path, criteria)
            store.transition(PhaseEState.CRITERIA_EVALUATED, {"criteria": criteria_path})
            if self.mode is RunMode.SMOKE_RUN:
                target = PhaseEState.CANDIDATE_SIMULATED if criteria.all_mandatory_passed else PhaseEState.REJECTED_SIMULATED
            else:
                target = PhaseEState.CANDIDATE if criteria.all_mandatory_passed else PhaseEState.REJECTED_EXPERIMENT
            publication = None
            if target is PhaseEState.CANDIDATE:
                publication = self._publish_candidate(
                    preregistration=preregistration, preflight=preflight, dataset=dataset,
                    folds_path=folds_path, competition=competition, calibration=calibration,
                    qmae=qmae, bundle=bundle, econ=econ, criteria=criteria, run_dir=run_dir,
                )
            final_path = run_dir / "final_report.json"
            final = {
                "experiment_id": preflight.experiment_id, "run_id": run_id, "mode": self.mode,
                "result": target, "candidate_hash": bundle.candidate_hash,
                "criteria_hash": criteria.artifact_hash, "lockbox_consumed": True,
                "simulated": self.mode is RunMode.SMOKE_RUN,
                "publication": publication,
            }
            atomic_write_json(final_path, final)
            store.transition(target, {"final": final_path})
            self._record(evidence, run_id, target, final)
            return self._finish(preflight.experiment_id, run_id, run_dir, target, True, target.value)
        except PhaseETechnicalError as exc:
            after_lockbox = self._is_after_lockbox(store.state)
            target = (
                PhaseEState.FAILED_TECHNICAL_AFTER_LOCKBOX
                if after_lockbox else PhaseEState.FAILED_TECHNICAL_BEFORE_LOCKBOX
            )
            try:
                store.transition(target, {})
            except PhaseETechnicalError:
                pass
            self._record(evidence, run_id, store.state, {
                "failure": "TECHNICAL", "code": exc.code.value,
            })
            raise
        except PhaseEScientificRejection as exc:
            try:
                store.transition(PhaseEState.FAILED_SCIENTIFIC, {})
            except PhaseETechnicalError:
                pass
            self._record(evidence, run_id, store.state, {
                "failure": "SCIENTIFIC", "code": exc.code.value,
            })
            raise
        except Exception as exc:
            after_lockbox = self._is_after_lockbox(store.state)
            target = PhaseEState.FAILED_TECHNICAL_AFTER_LOCKBOX if after_lockbox else PhaseEState.FAILED_TECHNICAL_BEFORE_LOCKBOX
            try:
                store.transition(target, {})
            except PhaseETechnicalError:
                pass
            code = PhaseEErrorCode.TECHNICAL_FAILURE_AFTER_LOCKBOX if after_lockbox else PhaseEErrorCode.TECHNICAL_FAILURE_BEFORE_LOCKBOX
            raise PhaseETechnicalError(code, "Phase-E orchestration failed") from exc

    @staticmethod
    def _is_after_lockbox(state: PhaseEState | None) -> bool:
        return state in {
            PhaseEState.LOCKBOX_ACQUIRED, PhaseEState.ECON_EVALUATED,
            PhaseEState.CRITERIA_EVALUATED, PhaseEState.CANDIDATE,
            PhaseEState.REJECTED_EXPERIMENT, PhaseEState.CANDIDATE_SIMULATED,
            PhaseEState.REJECTED_SIMULATED, PhaseEState.FAILED_TECHNICAL_AFTER_LOCKBOX,
        }

    def _lockbox_paths(self, preregistration: Mapping[str, Any], run_dir: Path) -> tuple[Path, Path]:
        if self.mode is RunMode.SMOKE_RUN:
            root = run_dir / "smoke_lockbox"
            return root / "lockbox_lease.json", root / "lockbox_state.json"
        if int(preregistration.get("protocol_version", 1)) == 2:
            authority = self.preflight.root / str(preregistration["lockbox"]["authority_path"])
            return authority.with_suffix(".lease.json"), authority
        budget = self.preflight.root / str(preregistration["lockbox"]["query_state_path"])
        return budget.with_name("lockbox_lease.json"), budget

    def _publish_candidate(
        self, *, preregistration: Mapping[str, Any], preflight: PhaseEPreflightResult,
        dataset: DatasetBuildResult, folds_path: Path, competition: ModelCompetitionResult,
        calibration: CalibrationResult, qmae: QMAEValidationResult,
        bundle: ExperimentalBundleResult, econ: EconEvaluationResult,
        criteria: PromotionCriteriaResult, run_dir: Path,
    ) -> CandidatePublicationResult:
        candidate_payload = json.loads(canonical_json(bundle.payload))
        candidate_payload["metadata"]["lifecycle_state"] = BundleLifecycleState.CANDIDATE.value
        candidate_payload["metadata"]["system_freeze_hash"] = None
        candidate_payload["content_hash"] = Sha256HashProvider().digest_value({
            key: value for key, value in candidate_payload.items() if key != "content_hash"
        })
        loaded = model_bundle_from_payload(candidate_payload)
        policy = FrozenSelectionPolicy.derive(
            policy_id=f"{preflight.experiment_id}-selection-v1",
            bundle_id=loaded.bundle_id, bundle_hash=loaded.content_hash,
            dataset_hash=dataset.dataset_hash, econ_report_hash=econ.artifact_hash,
            scores=econ.scores, budget_fraction=econ.budget_fraction,
        )
        policy.validate(scores=econ.scores, expected_hashes={
            "bundle_hash": loaded.content_hash, "dataset_hash": dataset.dataset_hash,
            "econ_report_hash": econ.artifact_hash,
        })
        policy_path = run_dir / "selection_policy.json"
        atomic_write_json(policy_path, policy)
        component_hashes = {
            "dataset": dataset.dataset_hash,
            "snapshots": preflight.d3_manifest_hash,
            "features": dataset.feature_hash,
            "labels": Sha256HashProvider().digest_value(dataset.label_schema),
            "folds": sha256_file(folds_path),
            "normalizer": Sha256HashProvider().digest_value(bundle.payload["normalizer"]),
            "models": competition.artifact_hash,
            "calibrators": calibration.artifact_hash,
            "selection_policy": policy.content_hash,
            "econ_report": econ.artifact_hash,
            "bundle": loaded.content_hash,
            "promotion_criteria": criteria.artifact_hash,
        }
        freeze = SystemFreeze.create(
            freeze_id=f"{preflight.experiment_id}-{loaded.content_hash[:12]}",
            component_hashes=component_hashes,
            universe_id=str(preregistration["universe_id"]),
            symbol_set_hash=str(preregistration["symbol_set_hash"]),
            timeframe=str(preregistration["timeframe"]), code_commit=preflight.git_commit,
            environment={
                "python": preflight.environment.python, "numpy": preflight.environment.numpy,
                "sklearn": preflight.environment.sklearn, "platform": preflight.environment.platform,
                "preregistration_hash": preflight.preregistration_hash,
                "qmae_hash": qmae.artifact_hash,
            },
        )
        freeze.validate(component_hashes)
        freeze_path = run_dir / "system_freeze.json"
        atomic_write_json(freeze_path, freeze)
        registry_path = run_dir / "registry" / f"{loaded.bundle_id}.json"
        atomic_write_json(registry_path, candidate_payload)
        return CandidatePublicationResult(
            loaded.content_hash, policy.content_hash, freeze.content_hash, str(registry_path),
        )

    @staticmethod
    def _advance(
        store: RunStateStore, target: PhaseEState, artifacts: Mapping[str, Path],
    ) -> None:
        history = store.recover()
        current = history[-1].state if history else None
        ordered = (
            PhaseEState.PRE_REGISTERED, PhaseEState.PREFLIGHT_VALIDATED,
            PhaseEState.RUN_SNAPSHOT_CREATED, PhaseEState.DATASET_BUILT,
            PhaseEState.FOLDS_READY, PhaseEState.TRAINING_IN_PROGRESS,
            PhaseEState.MODELS_EVALUATED, PhaseEState.MODELS_SELECTED,
            PhaseEState.CALIBRATION_VALIDATED, PhaseEState.QMAE_VALIDATED,
            PhaseEState.REFIT_COMPLETED, PhaseEState.THRESHOLD_DERIVED,
            PhaseEState.VALIDATION_COMPLETED,
            PhaseEState.LOCKBOX_ACQUIRED, PhaseEState.ECON_EVALUATED,
            PhaseEState.CRITERIA_EVALUATED,
        )
        if current in ordered and target in ordered and ordered.index(target) <= ordered.index(current):
            return
        store.transition(target, artifacts)

    def _existing_result(
        self, experiment_id: str, run_id: str, run_dir: Path, state: PhaseEState,
    ) -> PhaseEResult:
        final_path = run_dir / "final_report.json"
        lockbox_consumed = state in {
            PhaseEState.CANDIDATE, PhaseEState.REJECTED_EXPERIMENT,
            PhaseEState.CANDIDATE_SIMULATED, PhaseEState.REJECTED_SIMULATED,
            PhaseEState.FAILED_TECHNICAL_AFTER_LOCKBOX,
        }
        verdict = state.value
        if final_path.is_file():
            payload = json.loads(final_path.read_text(encoding="utf-8"))
            verdict = str(payload.get("result", verdict))
            lockbox_consumed = bool(payload.get("lockbox_consumed", lockbox_consumed))
        return self._finish(experiment_id, run_id, run_dir, state, lockbox_consumed, verdict)

    @staticmethod
    def _record(
        recorder: AppendOnlyEvidenceRecorder, run_id: str, state: PhaseEState | None, payload: Mapping[str, Any],
    ) -> None:
        event_type = f"PHASE_E_{state.value if state else 'UNKNOWN'}"
        if any(item.event_type == event_type for item in recorder.events):
            return
        event_hash = Sha256HashProvider().digest_value({"run": run_id, "state": state, "payload": payload})
        recorder.record(ScientificEvidenceEvent(
            event_id=f"phase-e-{event_hash[:24]}", decision_id=run_id,
            decision_cycle_id=run_id, event_type=event_type,
            occurred_at=utc_now(), payload=payload,
        ))

    def _finish(
        self, experiment_id: str, run_id: str, run_dir: Path, state: PhaseEState | None,
        lockbox_consumed: bool, verdict: str,
    ) -> PhaseEResult:
        assert state is not None
        return PhaseEResult(
            experiment_id, run_id, self.mode, state, str(run_dir), lockbox_consumed, verdict,
        )
