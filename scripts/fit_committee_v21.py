#!/usr/bin/env python3
"""Fit the preregistered V2.1 calibrated-risk Shadow artifact offline."""

from __future__ import annotations

import argparse
from pathlib import Path

from aegis.research.committee_v21_fit import (
    build_committee_v21_dataset,
    fit_committee_v21,
    load_committee_v21_fit_config,
    report_summary,
    write_committee_v21_fit_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/experiments/aegis_committee_v21_preregistered_v1.yaml"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_committee_v21_fit_config(
        (
            (root / args.config).resolve()
            if not args.config.is_absolute()
            else args.config
        ),
        repo_root=root,
    )

    def progress(current: int, total: int) -> None:
        print(f"committee-v2.1-fit {current}/{total}", flush=True)

    rows, duplicate_audit = build_committee_v21_dataset(
        config,
        repo_root=root,
        progress=progress,
    )
    artifact, report = fit_committee_v21(
        config,
        rows,
        duplicate_audit=duplicate_audit,
    )
    outputs = write_committee_v21_fit_outputs(config, artifact, report)
    print(report_summary(report))
    for name, value in outputs.items():
        print(f"{name}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
