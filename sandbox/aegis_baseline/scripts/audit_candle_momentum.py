"""Audit same-color candle continuation without touching runtime or exchange."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aegis.research.candle_momentum_audit import (
    load_candle_momentum_audit_config,
    run_candle_momentum_audit,
    write_candle_momentum_reports,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "config/experiments/aegis_candle_momentum_audit_v1.yaml"
        ),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = load_candle_momentum_audit_config(
        config_path,
        repo_root=root,
    )
    report = run_candle_momentum_audit(config)
    digests = write_candle_momentum_reports(
        report,
        json_path=config.json_report,
        markdown_path=config.markdown_report,
    )
    print(
        json.dumps(
            {
                "conclusion": report["conclusion"],
                "validated_opportunities": report[
                    "validated_opportunities"
                ],
                "reports": digests,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
