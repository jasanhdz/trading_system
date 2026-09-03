from pathlib import Path

from aegis.research.shadow_evidence import audit_entry_quality_v2_evidence
from aegis.research.shadow_runtime import EntryQualityV2ShadowRuntime

from test_entry_quality_v2_shadow_runtime import _batch, _runtime_config


def test_shadow_evidence_audit_tracks_paper_outcomes_without_promotion(
    tmp_path: Path,
) -> None:
    config = _runtime_config(tmp_path)
    runtime = EntryQualityV2ShadowRuntime(config)
    for cycle in range(config.horizon_bars + 1):
        runtime.observe_batch(_batch(cycle))

    result = audit_entry_quality_v2_evidence(
        config.signal_journal,
        config.outcome_journal,
    )

    assert result["signal_records"] == 11 * (config.horizon_bars + 1)
    assert result["decision_cycles"] == config.horizon_bars + 1
    assert result["matured_episode_count"] == 11
    assert result["non_overlapping_episode_count"] == 11
    assert result["global_performance"]["matured_episodes"] == 11
    assert result["exchange_mutations"] == 0
    assert result["automatic_promotion"] is False
    assert result["evidence_state"] == "COLLECTING_SHADOW_EVIDENCE"
