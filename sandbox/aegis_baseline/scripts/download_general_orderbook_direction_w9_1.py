#!/usr/bin/env python3
"""Download only preregistered public TRAIN/VALIDATION samples for W9.1."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import yaml
from tardis_dev import download_datasets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--concurrency", type=int, default=6)
    args = parser.parse_args()
    root = args.root.resolve()
    config = yaml.safe_load((root / "config/experiments/aegis_general_orderbook_direction_w9_1.yaml").read_text())
    months = config["partitions"]["train_months"] + config["partitions"]["validation_months"]
    destination = root / "data/historical_orderbook_direction_w9_1/raw"
    for month in months:
        from_date = f"{month}-01"
        to_date = (date.fromisoformat(from_date) + timedelta(days=1)).isoformat()
        download_datasets(
            exchange=config["data"]["exchange"],
            data_types=config["data"]["data_types"],
            symbols=config["data"]["symbols"],
            from_date=from_date,
            to_date=to_date,
            api_key="",
            download_dir=str(destination),
            concurrency=args.concurrency,
            skip_if_exists=True,
        )


if __name__ == "__main__":
    main()
