#!/usr/bin/env python3
"""Build the causal Aegis-conditioned dataset without opening August holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

EXPERIMENT = Path(__file__).resolve().parents[1]
SANDBOX = EXPERIMENT.parents[1]
REPOSITORY = SANDBOX.parents[1]
ENTRY_V1 = SANDBOX / "experiments/independent_entry_quality_discovery_v1"
DIRECTIONAL_V1 = SANDBOX / "experiments/directional_alpha_v1"
for path in (EXPERIMENT / "src", ENTRY_V1 / "src", DIRECTIONAL_V1 / "src", SANDBOX / "src", REPOSITORY / "src"):
    sys.path.insert(0, str(path))

from aegis_entry_enhancement_v1.dataset import build_neutral_panel, load_signals, score_signals, sha256  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=EXPERIMENT / "config/preregistration_v1.json")
    parser.add_argument("--candles", type=Path, default=REPOSITORY / "data/aegis_entry_enhancement_v1/candles_1m")
    parser.add_argument("--output", type=Path, default=EXPERIMENT / "artifacts/dataset_v1")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    signals, signal_audit = load_signals(REPOSITORY, config)
    panel, candles = build_neutral_panel(signals, args.candles, config["symbols"])
    scored = score_signals(signals=signals, panel=panel, candles=candles, config=config, repository=REPOSITORY)
    holdout = scored.loc[scored.split.eq("FINAL_HOLDOUT")].drop(columns=[column for column in scored if column.startswith("target__")], errors="ignore")
    development = scored.loc[~scored.split.isin(["FINAL_HOLDOUT", "EMBARGO_1", "EMBARGO_2", "EMBARGO_3"])].copy()
    embargo = scored.loc[scored.split.str.startswith("EMBARGO")].drop(columns=[column for column in scored if column.startswith("target__")], errors="ignore")
    if any(column.startswith("target__") for column in holdout):
        raise RuntimeError("FINAL_HOLDOUT_LABEL_LEAK")
    development.to_parquet(args.output / "development_labeled.parquet", index=False, compression="zstd")
    holdout.to_parquet(args.output / "final_holdout_features_sealed.parquet", index=False, compression="zstd")
    embargo.to_parquet(args.output / "embargo_features.parquet", index=False, compression="zstd")
    (args.output / "aegis_signal_audit.json").write_text(json.dumps(signal_audit, indent=2, sort_keys=True, default=str) + "\n")
    feature_columns = sorted(column for column in scored if column.startswith("feature__") or column in (
        "opportunity_score", "directional_probability_aegis_side", "directional_probability_opposite_side",
        "predicted_net_aegis_side_bps", "predicted_net_opposite_side_bps", "predicted_aegis_advantage_bps",
        "quality_score", "ood_feature_fraction", "ood",
    ))
    manifest = {
        "schema": "aegis-entry-enhancement-v1-dataset-manifest",
        "config_sha256": sha256(args.config), "source_manifest_sha256": sha256(args.candles / "dataset_manifest.json"),
        "rows_by_split": scored.split.value_counts().sort_index().to_dict(),
        "development_rows": len(development), "holdout_rows": len(holdout), "holdout_labels_built": False,
        "sides": scored.side.value_counts().to_dict(), "symbols": scored.symbol.value_counts().to_dict(),
        "feature_columns": feature_columns,
        "max_feature_after_signal": int((pd.to_datetime(scored.max_feature_available_at, utc=True) > pd.to_datetime(scored.signal_timestamp, utc=True)).sum()),
        "opportunity_retrained": False, "directional_retrained": False, "aegis_side_changed": False,
        "aegis_signal_audit_sha256": sha256(args.output / "aegis_signal_audit.json"),
    }
    (args.output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
