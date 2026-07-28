"""Evaluate frozen entry-condition hypotheses using Shadow evidence only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aegis.research.entry_condition_study import (
    evaluate_entry_condition_study,
    load_entry_condition_study_config,
    write_entry_condition_study_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "config/experiments/aegis_entry_condition_shadow_v1.yaml"
        ),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = load_entry_condition_study_config(config_path, repo_root=root)
    report = evaluate_entry_condition_study(config)
    report_sha256 = write_entry_condition_study_report(
        report, config.report_path
    )
    print(
        json.dumps(
            {
                **report["readiness"],
                "report_path": str(config.report_path),
                "report_sha256": report_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
