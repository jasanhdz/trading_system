#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.tools.shadow_status_report import shadow_status_report  # noqa: E402

DEFAULT_CONFIG = "aegis_alpha/configs/base.yaml"
DEFAULT_LOG_GLOB = "aegis_alpha/logs/shadow/*.jsonl"
DEFAULT_OUTPUT_DIR = Path("aegis_alpha/logs/shadow")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--logs", default=DEFAULT_LOG_GLOB)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--status-only", action="store_true")
    args = parser.parse_args()

    shadow_status_report(args.config, args.logs)
    if not args.status_only:
        from aegis_alpha.tools.evaluate_shadow_log import evaluate_shadow_log

        evaluate_shadow_log(args.config, args.logs, Path(args.output_dir))


if __name__ == "__main__":
    main()
