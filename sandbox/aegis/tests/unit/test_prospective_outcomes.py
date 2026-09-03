from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aegis.domain import Candle
from aegis.prospective import (
    ActivationContract,
    ProspectiveOutcomeError,
    ProspectiveOutcomeJournal,
    ProspectiveOutcomeMaturator,
    ProspectiveSignalEvidence,
    HistoricalE5ClosureError,
    deny_historical_e5_stage,
    load_historical_closure,
)
from aegis.utils import sha256_file


UTC = timezone.utc


def activation(**changes) -> ActivationContract:
    values = {
        "active": True,
        "activated_at_utc": datetime(2026, 7, 21, 11, 0, tzinfo=UTC),
        "cohort_id": "cohort-v1",
        "model_artifact_hash": "a" * 64,
        "configuration_hash": "b" * 64,
        "python_commit": "c" * 40,
        "typescript_commit": "d" * 40,
    }
    values.update(changes)
    return ActivationContract(**values)


def envelope(**changes):
    values = {
        "schema_id": "aegis-prospective-signal-evidence-v1",
        "prospective_signal_id": "e" * 64,
        "cohort_id": "cohort-v1",
        "protocol_version": "aegis-prospective-validation-v1",
        "symbol": "ADAUSDT",
        "side": "SHORT",
        "signal_timestamp_utc": "2026-07-21T12:00:00.000Z",
        "information_cutoff_utc": "2026-07-21T12:00:00.000Z",
        "model_artifact_hash": "a" * 64,
        "configuration_hash": "b" * 64,
        "source_python_commit": "c" * 40,
        "source_typescript_commit": "d" * 40,
    }
    values.update(changes)
    return values


def candle(open_time: datetime, *, high=100.1, low=99.5, close=99.6) -> Candle:
    return Candle(open_time, open_time + timedelta(minutes=5), 100.0, high, low, close, 1000.0, True, "PROSPECTIVE_FIXTURE")


def path():
    signal = candle(datetime(2026, 7, 21, 11, 55, tzinfo=UTC), high=100.1, low=99.9, close=100.0)
    future = tuple(candle(signal.close_time + timedelta(minutes=5 * index)) for index in range(12))
    return signal, future


def test_signal_contract_rejects_preactivation_and_manifest_drift() -> None:
    with pytest.raises(ProspectiveOutcomeError, match="PROSPECTIVE_PREACTIVATION_SIGNAL"):
        ProspectiveSignalEvidence.parse(envelope(), activation(activated_at_utc=datetime(2026, 7, 21, 13, tzinfo=UTC)))
    with pytest.raises(ProspectiveOutcomeError, match="PROSPECTIVE_MODEL_HASH_MISMATCH"):
        ProspectiveSignalEvidence.parse(envelope(model_artifact_hash="f" * 64), activation())
    with pytest.raises(ProspectiveOutcomeError, match="PROSPECTIVE_CONFIG_HASH_MISMATCH"):
        ProspectiveSignalEvidence.parse(envelope(configuration_hash="f" * 64), activation())
    with pytest.raises(ProspectiveOutcomeError, match="PROSPECTIVE_CODE_HASH_MISMATCH"):
        ProspectiveSignalEvidence.parse(envelope(source_python_commit="f" * 40), activation())
    with pytest.raises(ProspectiveOutcomeError, match="PROSPECTIVE_PROTOCOL_MISMATCH"):
        ProspectiveSignalEvidence.parse(envelope(protocol_version="v2"), activation())


def test_outcome_cannot_mature_before_horizon_and_missing_data_fails(tmp_path: Path) -> None:
    signal, future = path()
    maturator = ProspectiveOutcomeMaturator(activation(), ProspectiveOutcomeJournal(tmp_path / "outcomes.jsonl"))
    with pytest.raises(ProspectiveOutcomeError, match="PROSPECTIVE_OUTCOME_PREMATURE"):
        maturator.mature_and_persist(envelope(), signal, future, as_of_utc=future[-1].open_time)
    with pytest.raises(ProspectiveOutcomeError, match="PROSPECTIVE_MARKET_DATA_INCOMPLETE"):
        maturator.mature_and_persist(envelope(), signal, future[:-1], as_of_utc=future[-1].close_time)
    assert not (tmp_path / "outcomes.jsonl").exists()


def test_mature_outcome_persists_every_target_field_row_level(tmp_path: Path) -> None:
    signal, future = path()
    path_out = tmp_path / "outcomes.jsonl"
    maturator = ProspectiveOutcomeMaturator(activation(), ProspectiveOutcomeJournal(path_out))
    outcome = maturator.mature_and_persist(envelope(), signal, future, as_of_utc=future[-1].close_time)
    persisted = json.loads(path_out.read_text(encoding="utf-8"))
    assert persisted == outcome
    assert set((
        "gross_return_fraction", "net_return_fraction", "mfe_fraction", "mae_fraction", "tail_event",
        "qmae", "clean_quality", "net_quality_after_costs", "label_valid", "fees_fraction",
        "slippage_fraction", "funding_fraction", "horizon_bars", "termination_reason",
        "missingness", "validation_state",
    )).issubset(outcome)
    assert outcome["funding_fraction"] == 0.0
    assert outcome["fees_fraction"] == pytest.approx(0.0008)
    assert outcome["slippage_fraction"] == pytest.approx(0.0002)
    assert outcome["net_return_fraction"] == pytest.approx(outcome["gross_return_fraction"] - 0.001)
    assert outcome["qmae"] == outcome["mae_fraction"]


def test_one_to_one_journal_rejects_duplicate_and_conflict(tmp_path: Path) -> None:
    signal, future = path()
    path_out = tmp_path / "outcomes.jsonl"
    maturator = ProspectiveOutcomeMaturator(activation(), ProspectiveOutcomeJournal(path_out))
    outcome = maturator.mature_and_persist(envelope(), signal, future, as_of_utc=future[-1].close_time)
    with pytest.raises(ProspectiveOutcomeError, match="PROSPECTIVE_OUTCOME_DUPLICATE"):
        maturator.journal.append(outcome)
    changed = {**outcome, "gross_return_fraction": 0.5}
    with pytest.raises(ProspectiveOutcomeError, match="PROSPECTIVE_OUTCOME_CONFLICT"):
        maturator.journal.append(changed)
    assert len(path_out.read_text(encoding="utf-8").splitlines()) == 1


def test_restart_is_deterministic_and_tampered_duplicate_fails_closed(tmp_path: Path) -> None:
    signal, future = path()
    path_out = tmp_path / "outcomes.jsonl"
    first = ProspectiveOutcomeMaturator(activation(), ProspectiveOutcomeJournal(path_out))
    outcome = first.mature_and_persist(envelope(), signal, future, as_of_utc=future[-1].close_time)
    first_hash = first.journal.sha256
    recovered = ProspectiveOutcomeJournal(path_out)
    assert recovered.sha256 == first_hash
    with pytest.raises(ProspectiveOutcomeError, match="PROSPECTIVE_OUTCOME_DUPLICATE"):
        recovered.append(outcome)
    path_out.write_text(path_out.read_text(encoding="utf-8") * 2, encoding="utf-8")
    with pytest.raises(ProspectiveOutcomeError, match="PROSPECTIVE_OUTCOME_DUPLICATE"):
        ProspectiveOutcomeJournal(path_out)


def test_signal_and_market_join_are_exact_and_no_trade_cannot_mature(tmp_path: Path) -> None:
    signal, future = path()
    maturator = ProspectiveOutcomeMaturator(activation(), ProspectiveOutcomeJournal(tmp_path / "outcomes.jsonl"))
    with pytest.raises(ProspectiveOutcomeError, match="PROSPECTIVE_SIGNAL_CANDLE_MISMATCH"):
        shifted = candle(signal.open_time + timedelta(minutes=5), high=100.1, low=99.9, close=100.0)
        maturator.mature_and_persist(envelope(), shifted, future, as_of_utc=future[-1].close_time)
    with pytest.raises(ProspectiveOutcomeError, match="PROSPECTIVE_OUTCOME_NO_TRADE_PROHIBITED"):
        maturator.mature_and_persist(envelope(side="NO_TRADE"), signal, future, as_of_utc=future[-1].close_time)


def test_gap_partial_and_nonfinal_market_data_fail_closed(tmp_path: Path) -> None:
    signal, future = path()
    maturator = ProspectiveOutcomeMaturator(activation(), ProspectiveOutcomeJournal(tmp_path / "outcomes.jsonl"))
    gap = list(future)
    gap[5] = candle(gap[5].open_time + timedelta(minutes=5))
    with pytest.raises(ProspectiveOutcomeError, match="PROSPECTIVE_MARKET_DATA_INCOMPLETE"):
        maturator.mature_and_persist(envelope(), signal, gap, as_of_utc=gap[-1].close_time)
    partial = list(future)
    original = partial[-1]
    partial[-1] = Candle(original.open_time, original.close_time, original.open, original.high, original.low, original.close, original.volume, False, original.source)
    with pytest.raises(ProspectiveOutcomeError, match="PROSPECTIVE_MARKET_DATA_INCOMPLETE"):
        maturator.mature_and_persist(envelope(), signal, partial, as_of_utc=partial[-1].close_time)


def test_historical_e5_closure_denies_discovery_and_confirmation_without_mutation() -> None:
    root = Path(__file__).resolve().parents[2]
    closure_path = root / "reports/governance/aegis_prospective_validation/historical_e5_closure_report.json"
    entry_path = root / "reports/governance/e5_signal_edge_protocol/phase1a/sealed/fold12_historical_entry_manifest_v1.jsonl"
    entry_hash = sha256_file(entry_path)
    closure = load_historical_closure(closure_path)
    assert closure["classification"] == "HISTORICAL_E5_NON_EXECUTABLE_MISSING_CONTEMPORANEOUS_ROW_TARGETS"
    assert closure["lockbox"] == "NOT_CONSUMED"
    assert closure["consumed_queries"] == [] and closure["budget_remaining"] == 1
    for stage in ("DISCOVERY", "CONFIRMATION"):
        with pytest.raises(HistoricalE5ClosureError, match="E5_HISTORICAL_EXECUTION_CLOSED"):
            deny_historical_e5_stage(stage)
    assert sha256_file(entry_path) == entry_hash
