#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import yaml

from aegis.research.recent_strategy_atlas_w16 import walk_forward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/experiments/aegis_recent_strategy_atlas_w16.yaml"))
    parser.add_argument("--output", type=Path, default=Path("data/recent_strategy_atlas_w16/run_01"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    result, folds = walk_forward(Path.cwd(), config)
    verdict = {"verdict": "AEGIS_W16_RECENT_STRATEGY_EDGE" if result["W16_RECENT_STRATEGY_EDGE_FOUND"] else "AEGIS_W16_NO_RECENT_CLASSIC_STRATEGY_EDGE",
               "result": result, "W16_READY_FOR_PROSPECTIVE_OBSERVATION": result["W16_RECENT_STRATEGY_EDGE_FOUND"],
               "W16_READY_FOR_SHADOW": False, "W16_READY_FOR_LIVE": False, "production_modified": False}
    args.output.mkdir(parents=True, exist_ok=True)
    folds.to_csv(args.output / "folds.csv", index=False)
    (args.output / "verdict.json").write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    print(json.dumps(verdict, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
