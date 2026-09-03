#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aegis.research.recent_short_w14 import (
    build_dataset,
    evaluate_short_selector,
    evaluate_volume_navigation,
    load_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("config/experiments/aegis_recent_short_w14.yaml"))
    parser.add_argument("--output", type=Path, default=Path("data/recent_short_w14/run_01"))
    args = parser.parse_args()
    config = load_config(args.root / args.config)
    dataset = build_dataset(args.root, config["universe"])
    short_result, selected = evaluate_short_selector(dataset, config)
    volume_result = evaluate_volume_navigation(dataset, config)
    verdict = {
        "schema_version": "aegis-recent-short-w14-verdict-v1",
        "experiment_id": config["experiment_id"],
        "data_end_utc": config["data_end_utc"],
        "splits": config["splits"],
        "w14a": short_result,
        "w14b": volume_result,
        "W14_EDGE_FOUND": bool(short_result["W14A_RECENT_SHORT_EDGE_FOUND"] or volume_result["W14B_VOLUME_NAVIGATION_EDGE_FOUND"]),
        "W14_READY_FOR_SHADOW": False,
        "W14_READY_FOR_LIVE": False,
        "production_modified": False,
    }
    output = args.root / args.output
    output.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(output / "w14a_validation_decisions.parquet")
    (output / "w14_verdict.json").write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    print(json.dumps(verdict, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
