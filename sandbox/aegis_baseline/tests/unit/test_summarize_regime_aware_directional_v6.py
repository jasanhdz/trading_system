from __future__ import annotations

from summarize_regime_aware_directional_v6 import compact_summary
from test_regime_aware_directional_v6_gate import report


def test_compact_summary_preserves_gate_and_directional_fold_evidence() -> None:
    source = report()
    source.update(
        {
            "experiment_id": "v6-test",
            "source_evidence_start": "2025-01-01T00:00:00+00:00",
            "source_evidence_end": "2026-01-01T00:00:00+00:00",
            "content_hash": "a" * 64,
        }
    )
    for side in ("LONG", "SHORT"):
        source["sides"][side].update(  # type: ignore[index]
            {
                "rows": 100,
                "verdict": "ELIGIBLE_FOR_PROSPECTIVE_SHADOW_DEPLOYMENT",
                "folds": [
                    {
                        "fold": 1,
                        "status": "EVALUATED",
                        "passed": True,
                        "baseline": {"count": 20},
                        "selected": {"count": 5},
                    }
                ],
            }
        )
    summary = compact_summary(source)
    assert summary["shadow_gate"]["eligible"] is True
    assert summary["sides"]["LONG"]["folds"][0]["selected"]["count"] == 5
    assert summary["runtime_effect"] == "NONE"
