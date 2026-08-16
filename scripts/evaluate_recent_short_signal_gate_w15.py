#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import yaml

from aegis.research.recent_short_signal_gate_w15 import evaluate, load_evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/experiments/aegis_recent_short_signal_gate_w15.yaml"))
    parser.add_argument("--output", type=Path, default=Path("data/recent_short_signal_gate_w15/run_01"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    frame = load_evidence(Path(config["source"]))
    result = evaluate(frame, config)
    verdict = {"verdict": "AEGIS_W15_RECENT_SHORT_SIGNAL_EDGE" if result["W15_RECENT_SHORT_SIGNAL_EDGE_FOUND"] else "AEGIS_W15_NO_RECENT_SHORT_SIGNAL_EDGE",
               "result": result, "W15_READY_FOR_SHADOW": False, "W15_READY_FOR_LIVE": False,
               "final_holdout": "FUTURE_ONLY_SEALED", "production_modified": False}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "verdict.json").write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    print(json.dumps(verdict, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
