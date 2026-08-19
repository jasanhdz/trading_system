#!/usr/bin/env python3
"""Build the frozen E4 dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "sandbox/aegis_strategy_router/src"), str(EXPERIMENT / "src")]

from aegis_e4.contracts import load_config  # noqa: E402
from aegis_e4.dataset import build_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=EXPERIMENT / "config/preregistration_v1.json")
    parser.add_argument("--output", type=Path, default=EXPERIMENT / "artifacts/dataset_v1")
    args = parser.parse_args()
    print(json.dumps(build_dataset(load_config(args.config), args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
