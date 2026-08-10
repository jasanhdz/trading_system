from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def timestamp(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def test_replay_labels_clean_paths_and_preserves_zero_authority(tmp_path: Path) -> None:
    decisions = tmp_path / "decisions.jsonl"
    signals = tmp_path / "signals.jsonl"
    intelligence = tmp_path / "intelligence.jsonl"
    output = tmp_path / "report.json"
    start = datetime(2026, 8, 9, tzinfo=timezone.utc)
    decision = {
        "market_timestamp": timestamp(start),
        "symbol": "BTCUSDT",
        "side": "LONG",
        "selected": True,
        "opportunity_probability": 0.70,
        "danger_probability": 0.20,
        "mae_q90": 0.002,
        "mfe_q50": 0.009,
        "net_return_mean": 0.003,
        "confirmation": {"state": "CONFIRMED", "components_passed": 4},
        "confirmation_features": {
            "signed_ret_1_atr": 1.0,
            "signed_ret_3_atr": 1.0,
            "signed_ret_6_atr": 1.0,
            "signed_ret_12_atr": 1.0,
        },
    }
    write_jsonl(decisions, [decision])
    write_jsonl(
        intelligence,
        [
            {
                "market_timestamp": timestamp(start),
                "symbol": "BTCUSDT",
                "regime_v3_shadow": {
                    "volatility": "NORMAL",
                    "structure": "TREND",
                },
            }
        ],
    )
    market_rows = []
    for index in range(13):
        base = 100.0 + index * 0.10
        market_rows.append(
            {
                "market_timestamp": timestamp(start + timedelta(minutes=5 * index)),
                "symbol": "BTCUSDT",
                "market_bar": {
                    "open": base,
                    "high": base + 0.35,
                    "low": base - 0.05,
                    "close": base + 0.25,
                },
            }
        )
    write_jsonl(signals, market_rows)

    subprocess.run(
        [
            sys.executable,
            "scripts/replay_entry_methodology_v2_shadow.py",
            "--decisions",
            str(decisions),
            "--signals",
            str(signals),
            "--intelligence",
            str(intelligence),
            "--output",
            str(output),
        ],
        check=True,
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["tier_counts"] == {"A": 1}
    assert report["outcomes_by_tier"]["A"]["clean_path_success_rate"] == 1.0
    assert report["current_selected_control"]["rows"] == 1
    assert report["selection_effect"] == "NONE"
    assert report["exchange_authority"] is False
    assert report["exchange_mutations"] == 0
