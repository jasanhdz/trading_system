from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="One-shot, read-only prospective corridor shadow recorder")
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo_root / "sandbox/manual_visual_decision_replication_v1/src"))
    from mvdr_v1.m1 import fetch_klines
    from mvdr_v1.m1_2 import ProspectiveCorridorShadowRecorder, compact_corridor_frame

    decision_at = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = decision_at - timedelta(hours=4)
    sui = fetch_klines("SUIUSDT", start, decision_at)
    btc = fetch_klines("BTCUSDT", start, decision_at)
    frame = compact_corridor_frame(decision_at, sui, btc)
    record = ProspectiveCorridorShadowRecorder(args.output_root).append(frame)
    print(json.dumps({"timestamp": record["timestamp"], "final_action": record["final_action"], "orders_enabled": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
