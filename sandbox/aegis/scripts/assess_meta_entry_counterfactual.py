"""Assess frozen Meta-Entry predictions without refitting the model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from aegis.research.meta_entry import (
    MetaEntryCounterfactual,
    assess_counterfactual_predictions,
    counterfactual_mapping,
)
from aegis.training.run_state import atomic_write_json
from aegis.utils import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "data/meta_entry_v3_research/live_counterfactual_predictions.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/meta_entry_v3_research/counterfactual_assessment.json"
        ),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    predictions_path = (
        args.predictions
        if args.predictions.is_absolute()
        else root / args.predictions
    )
    output_path = args.output if args.output.is_absolute() else root / args.output
    rows = []
    with predictions_path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            value = json.loads(line)
            rows.append(
                MetaEntryCounterfactual(
                    event_id=str(value["event_id"]),
                    symbol=str(value["symbol"]),
                    selected=bool(value["selected"]),
                    favorable=bool(value["favorable"]),
                    net_return=float(value["net_return"]),
                    mae=float(value["mae"]),
                    probability=float(value["probability"]),
                    percentile=float(value["percentile"]),
                    actual_trade=bool(value["actual_trade"]),
                )
            )
    report = {
        "schema_id": "aegis-meta-entry-v3-counterfactual-assessment-v1",
        "source_predictions_path": str(predictions_path.relative_to(root)),
        "source_predictions_sha256": sha256_file(predictions_path),
        "assessment": assess_counterfactual_predictions(rows),
        "actual_trade_predictions": [
            counterfactual_mapping(row) for row in rows if row.actual_trade
        ],
        "training_or_refit_performed": False,
        "exchange_mutations": 0,
    }
    atomic_write_json(output_path, report)
    os.chmod(output_path, 0o600)
    print(json.dumps(report["assessment"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
