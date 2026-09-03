from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from aegis.training.preregistration import LockboxBudget, PreregistrationError, load_and_validate_preregistration


CONFIG = Path(__file__).parents[2] / "config" / "experiments" / "aegis_short_candidate_e1.yaml"


def test_phase_e_is_frozen_pre_registered_and_not_executed() -> None:
    payload, audit = load_and_validate_preregistration(CONFIG, audit_source=False)
    assert audit.fold_count == 4 and audit.embargo_minutes == 120
    assert audit.threshold_pending is True and audit.full_run_executed is False
    assert payload["promotion"]["mandatory"]["minimum_test_signals"] == 100
    assert payload["publication"]["success_state"] == "CANDIDATE"
    assert payload["publication"]["automatic_promotion"] is False


def test_phase_e_rejects_threshold_and_temporal_leakage(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text())
    payload["protocol"]["threshold_value"] = 0.5
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(PreregistrationError, match="threshold"):
        load_and_validate_preregistration(path, audit_source=False)
    payload["protocol"]["threshold_value"] = None
    payload["protocol"]["folds"][0]["validation_start"] = payload["protocol"]["folds"][0]["train_end"]
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(PreregistrationError, match="embargo"):
        load_and_validate_preregistration(path, audit_source=False)


def test_lockbox_budget_is_persistent_atomic_and_single_query(tmp_path: Path) -> None:
    budget = LockboxBudget(tmp_path / "state.json", 1, "a" * 64)
    assert budget.consume(
        candidate_hash="b" * 64, purpose="single confirmation",
        occurred_at=datetime(2026, 7, 18, tzinfo=timezone.utc),
    ) == 1
    assert not (tmp_path / "state.json.tmp").exists()
    with pytest.raises(PreregistrationError, match="exhausted"):
        budget.consume(
            candidate_hash="b" * 64, purpose="second query",
            occurred_at=datetime(2026, 7, 18, tzinfo=timezone.utc),
        )

