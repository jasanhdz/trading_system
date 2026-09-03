from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def test_replay_reports_candidate_frequency_without_mutation(tmp_path: Path) -> None:
    decisions = tmp_path / "decisions.jsonl"
    intelligence = tmp_path / "intelligence.jsonl"
    output = tmp_path / "report.json"
    outcomes = tmp_path / "outcomes.jsonl"
    candidate = {
        "market_timestamp": "2026-08-08T00:00:00Z",
        "symbol": "BTCUSDT",
        "side": "SHORT",
        "opportunity_probability": 0.7,
        "danger_probability": 0.2,
        "mae_q90": 0.004,
        "mfe_q50": 0.012,
        "net_return_mean": 0.003,
        "confirmation": {"state": "CONFIRMED", "components_passed": 4},
        "confirmation_features": {
            "signed_ret_1_atr": 1.0,
            "signed_ret_3_atr": 1.0,
            "signed_ret_6_atr": 1.0,
            "signed_ret_12_atr": -1.0,
        },
    }
    write_jsonl(decisions, [candidate])
    write_jsonl(
        intelligence,
        [
            {
                "market_timestamp": candidate["market_timestamp"],
                "symbol": candidate["symbol"],
                "regime_v3_shadow": {
                    "volatility": "NORMAL",
                    "structure": "TREND",
                },
            }
        ],
    )
    write_jsonl(
        outcomes,
        [
            {
                "market_timestamp": candidate["market_timestamp"],
                "symbol": candidate["symbol"],
                "directional_outcomes": {
                    "SHORT": {
                        "net_return_after_costs": 0.002,
                        "mae_fraction": 0.001,
                        "mfe_fraction": 0.004,
                    }
                },
            }
        ],
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/replay_seven_point_entry_shadow.py",
            "--decisions",
            str(decisions),
            "--intelligence",
            str(intelligence),
            "--output",
            str(output),
            "--outcomes",
            str(outcomes),
        ],
        check=True,
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["cycles_with_quality_candidate"] == 1
    assert report["candidate_cycle_fraction"] == 1.0
    assert report["selection_effect"] == "NONE"
    assert report["exchange_mutations"] == 0
    metrics = report["matured_outcome_metrics"][
        "COUNTERFACTUAL_QUALITY_CANDIDATE|SHORT"
    ]
    assert metrics["rows"] == 1
    assert metrics["mean_net_return_after_costs"] == 0.002
