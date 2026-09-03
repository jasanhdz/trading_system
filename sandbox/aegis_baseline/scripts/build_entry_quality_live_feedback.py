"""Build the immutable Shadow feedback dataset for future challenger training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aegis.research.live_feedback import (
    build_live_feedback_evidence,
    load_live_feedback_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/experiments/aegis_entry_quality_v3_live_feedback.yaml"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = load_live_feedback_config(config_path, repo_root=root)
    report = build_live_feedback_evidence(config)
    print(json.dumps(report["training_readiness"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

