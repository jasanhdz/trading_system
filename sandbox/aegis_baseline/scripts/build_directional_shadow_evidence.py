"""Build fail-closed LONG Shadow evidence from current append-only journals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aegis.research.directional_shadow_evidence import (
    build_directional_shadow_evidence,
    load_directional_shadow_evidence_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/entry_quality_v3_dual_shadow.yaml"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = load_directional_shadow_evidence_config(
        config_path,
        repo_root=root,
    )
    report = build_directional_shadow_evidence(config)
    print(json.dumps(report["readiness"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

