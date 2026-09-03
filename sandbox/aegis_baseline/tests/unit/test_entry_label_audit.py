from __future__ import annotations

from aegis.research.entry_label_audit import audit_entry_labels


def row(*, clean: bool, utility: float, outcome: str, event_bar: int | None, mae_ratio: float):
    return {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "symbol": "BTCUSDT",
        "side": "LONG",
        "independent": True,
        "v11_clean_entry_label": clean,
        "clean_fast_success": clean,
        "target_before_stop": outcome == "FAVORABLE_FIRST",
        "mae_fraction": 0.001,
        "mfe_fraction": 0.004,
        "time_underwater_bars": 2,
        "v11_causal_regime": "RANGE_LOW_VOL",
        "v11_path_diagnostics": {
            "pre_event_mae_as_adverse_barrier_fraction": mae_ratio,
        },
        "v10_contract_outcomes": {
            "ROE_10_H12": {
                "outcome": outcome,
                "event_bar": event_bar,
                "realized_utility": utility,
            }
        },
        "protection_profiles": {"CURRENT_TS": {"worst_net_return": utility / 2}},
    }


def test_audit_quantifies_economic_alignment_and_v11_exclusions():
    result = audit_entry_labels(
        [
            row(clean=True, utility=0.004, outcome="FAVORABLE_FIRST", event_bar=2, mae_ratio=0.2),
            row(clean=False, utility=0.002, outcome="FAVORABLE_FIRST", event_bar=8, mae_ratio=0.2),
            row(clean=False, utility=-0.006, outcome="ADVERSE_FIRST", event_bar=3, mae_ratio=1.0),
        ],
        v18_clean_average_precision={"LONG": 0.40},
    )
    clean = result["label_alignment"]["v11_clean"]
    assert clean["prevalence"] == 1 / 3
    assert clean["selected"]["mean_utility"] == 0.004
    assert clean["positive_utility_recall"] == 0.5
    assert result["v11_clean_exclusion_counts"] == {
        "FAVORABLE_BUT_AFTER_SIX_BARS": 1,
        "NOT_FAVORABLE_FIRST": 1,
    }
    assert result["holdout_accesses"] == 0


def test_audit_is_descriptive_and_keeps_side_groups_separate():
    long_row = row(clean=True, utility=0.001, outcome="FAVORABLE_FIRST", event_bar=1, mae_ratio=0.1)
    short_row = {**long_row, "side": "SHORT", "symbol": "ETHUSDT"}
    result = audit_entry_labels([long_row, short_row])
    assert result["selection_effect"] == "NONE"
    assert set(result["by_side"]) == {"LONG", "SHORT"}
