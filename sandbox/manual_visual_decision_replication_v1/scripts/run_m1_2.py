from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo_root / "sandbox/manual_visual_decision_replication_v1/src"))
    from mvdr_v1 import execute_m1_2

    summary = execute_m1_2(args.repo_root, args.output_root)
    print(json.dumps({"status": summary["STATUS"], "interpretation": summary["interpretation"], "flags": summary["flags"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
