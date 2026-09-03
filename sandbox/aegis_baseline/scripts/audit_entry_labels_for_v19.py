#!/usr/bin/env python3
"""Audit existing entry labels before defining any V19 experiment."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from aegis.research.entry_label_audit import audit_entry_labels
from aegis.utils import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/calibrated_horizon_v11/canonical_dataset.jsonl.gz"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/v19_design/entry_label_audit.json")
    )
    args = parser.parse_args()
    with gzip.open(args.dataset, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    result = audit_entry_labels(
        rows,
        v18_clean_average_precision={"LONG": 0.059718, "SHORT": 0.063987},
    )
    result["source_dataset"] = str(args.dataset)
    result["source_dataset_sha256"] = sha256_file(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "independent_rows": result["independent_rows"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
