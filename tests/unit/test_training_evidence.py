from datetime import timedelta
from pathlib import Path

import pytest

from aegis.domain import DecisionOutcome, FillOutcome, OutcomeExecutionStatus
from aegis.evidence import AppendOnlyEvidenceRecorder, EvidencePersistenceError, InMemoryEvidenceRecorder
from aegis.features import DeterministicFeaturePipeline
from aegis.training import CausalDatasetBuilder, DeterministicLinearTrainer, FileArtifactRegistry, OfflineModelEvaluator, TrainingTarget, walk_forward_splits
from aegis.utils import Sha256HashProvider


def test_training_reuses_features_walks_forward_and_publishes_immutably(snapshot_factory, tmp_path: Path) -> None:
    hashing = Sha256HashProvider()
    pipeline = DeterministicFeaturePipeline()
    base = snapshot_factory().closed_at
    snapshots = tuple(snapshot_factory(closed_at=base + timedelta(hours=index)) for index in range(12))
    targets = {}
    for snapshot in snapshots:
        for symbol_index, series in enumerate(snapshot.series):
            targets[(snapshot.closed_at, series.symbol)] = TrainingTarget(
                direction=1.0 if symbol_index % 2 else -1.0,
                expected_return=(symbol_index - 5) * 0.001, tail_event=float(symbol_index == 0),
                qmae=0.01 + symbol_index * 0.0001, clean_quality=float(symbol_index != 0),
            )
    dataset = CausalDatasetBuilder(pipeline, hashing, tuple(sorted(series.symbol for series in snapshots[0].series)), "5m").build("dataset-1", snapshots, targets)
    assert dataset.row_count == 132 and dataset.feature_hash == pipeline.feature_hash
    folds = walk_forward_splits(dataset, fold_count=3)
    assert len(folds) == 3
    for train, validation in folds:
        assert max(dataset.rows[index].timestamp for index in train) < min(dataset.rows[index].timestamp for index in validation)
    artifact = DeterministicLinearTrainer(hashing).train(dataset)
    report = OfflineModelEvaluator(hashing, maximum_direction_brier=2.0).evaluate(artifact, dataset)
    registry = FileArtifactRegistry(tmp_path / "registry", hashing)
    assert registry.publish(artifact, report) == artifact.artifact_id
    assert registry.load(artifact.artifact_id) == artifact
    with pytest.raises(FileExistsError):
        registry.publish(artifact, report)


def test_outcome_evidence_is_append_only_and_contains_no_policy_mutation(tmp_path: Path) -> None:
    hashing = Sha256HashProvider()
    recorder = AppendOnlyEvidenceRecorder(hashing, tmp_path / "events.jsonl")
    outcome = DecisionOutcome(
        decision_id="decision-1", decision_cycle_id="cycle-1", candidate_hash="a" * 64,
        accepted=True, executed=True, rejection_reason=None,
        fill=FillOutcome(OutcomeExecutionStatus.FILLED, 10.0, 0.5, None), closed_at=None,
        realized_pnl=None, close_reason=None, incidents=(), reconciled=True,
        occurred_at=__import__("datetime").datetime(2026, 7, 17, tzinfo=__import__("datetime").timezone.utc),
    )
    first = recorder.record_outcome(outcome)
    second = recorder.record_outcome(outcome)
    lines = (tmp_path / "events.jsonl").read_text().splitlines()
    assert len(lines) == 1 and second == first
    assert "policy" not in lines[0].lower() and "secret" not in lines[0].lower()
