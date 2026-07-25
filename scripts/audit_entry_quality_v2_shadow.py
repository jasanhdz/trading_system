"""Print a sanitized Entry Quality V2 Shadow evidence summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aegis.research.shadow_evidence import audit_entry_quality_v2_evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/entry_quality_v2_shadow"),
    )
    parser.add_argument("--minimum-matured-episodes", type=int, default=300)
    args = parser.parse_args()
    result = audit_entry_quality_v2_evidence(
        args.root / "signals.jsonl",
        args.root / "outcomes.jsonl",
        minimum_matured_episodes=args.minimum_matured_episodes,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

