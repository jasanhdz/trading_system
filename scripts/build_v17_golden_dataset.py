#!/usr/bin/env python3
"""Build closed historical V17 vectors and frozen Python outputs for TS parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from aegis.config import CANONICAL_SYMBOLS
from aegis.utils import Sha256HashProvider, sha256_file
from aegis.v17_feature_contract import V17_DTYPE, V17_FEATURE_SCHEMA, select_v17_features
from aegis.v17_research_artifact import V17ResearchArtifact
from train_directional_contract_v15_research import _partition
from train_economic_ranker_v16_research import _load_v16


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=Path("config/bundles/aegis-v17-research-artifact-v1.json"))
    parser.add_argument("--output", type=Path, default=Path("binance-futures-bot-ts/src/challengers/fixtures/v17-golden-dataset.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    artifact_path = args.artifact if args.artifact.is_absolute() else root / args.artifact
    output = args.output if args.output.is_absolute() else root / args.output
    artifact = V17ResearchArtifact.load(artifact_path)
    config = yaml.safe_load((root / "config/experiments/aegis_safety_gated_ranker_v17_research.yaml").read_text())
    rows = _load_v16(root / config["authority"]["source_dataset"])
    fold = config["validation"]["folds"][-1]
    events = []
    for side in ("LONG", "SHORT"):
        _, _, test = _partition([row for row in rows if row["side"] == side], fold, int(config["validation"]["embargo_minutes"]))
        for symbol in CANONICAL_SYMBOLS:
            row = next(item for item in test if item["symbol"] == symbol)
            vector = select_v17_features(side, row["features"])
            result = artifact.sides[side].score(vector)
            events.append({
                "event_id": f"{row['timestamp']}::{symbol}::{side}",
                "timestamp": row["timestamp"],
                "symbol": symbol,
                "side": side,
                "feature_vector": {
                    "side": side,
                    "schemaVersion": V17_FEATURE_SCHEMA,
                    "schemaHash": vector.schema_hash,
                    "names": list(vector.names),
                    "values": list(vector.values),
                    "dtype": V17_DTYPE,
                },
                "python": result,
            })
    payload = {
        "schema_id": "aegis-v17-python-typescript-golden-v1",
        "artifact_sha256": sha256_file(artifact_path),
        "artifact_content_hash": artifact.content_hash,
        "closed_historical_events": True,
        "event_count": len(events),
        "events": events,
    }
    payload["content_hash"] = Sha256HashProvider().digest_value(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "events": len(events), "content_hash": payload["content_hash"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
