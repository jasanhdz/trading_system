#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aegis_range_v1.readiness import SourceIntegrityError, build_derived_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Build non-economic Aegis Range R2 readiness data")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = build_derived_dataset(args.repo_root.resolve(), args.output_root.resolve())
    except SourceIntegrityError as exc:
        manifest_path = args.output_root.resolve() / "derived_dataset_manifest.json"
        detail = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        print(
            json.dumps(
                {
                    "funding_events_missing_mark_price": detail.get("funding_events_missing_mark_price"),
                    "status": str(exc),
                },
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "logical_sha256": manifest["logical_sha256"],
                "funding_events_missing_mark_price": manifest["funding_events_missing_mark_price"],
                "status": "AEGIS_RANGE_R2_DATA_READINESS_READY_FOR_REVIEW",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
