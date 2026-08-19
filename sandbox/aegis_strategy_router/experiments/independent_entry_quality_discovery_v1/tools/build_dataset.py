#!/usr/bin/env python3
"""Build the frozen causal entry-quality dataset; never label FINAL_HOLDOUT."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


EXPERIMENT = Path(__file__).resolve().parents[1]
SANDBOX = EXPERIMENT.parents[1]
REPOSITORY = SANDBOX.parents[1]
for path in (EXPERIMENT / "src", SANDBOX / "src", REPOSITORY / "src"):
    sys.path.insert(0, str(path))

from independent_entry_quality_v1.dataset import build_symbol_rows, combine_symbol_rows  # noqa: E402
from independent_entry_quality_v1.features import assert_feature_allowlist, dictionary_payload  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=EXPERIMENT / "config/preregistration_v1.json")
    parser.add_argument("--candles", type=Path, default=REPOSITORY / "data/independent_entry_quality_discovery_v1/candles_1m")
    parser.add_argument("--output", type=Path, default=EXPERIMENT / "artifacts/dataset_v1")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    pending = []
    for symbol in config["symbols"]:
        path = args.output / "by_symbol" / f"{symbol}.audit.json"
        if path.exists() and not args.overwrite:
            print(f"SKIP_COMPLETE {symbol}", flush=True)
        else:
            pending.append(symbol)
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        jobs = {
            pool.submit(
                build_symbol_rows, symbol=symbol, candle_root=args.candles,
                config=config, output_root=args.output,
            ): symbol
            for symbol in pending
        }
        for future in as_completed(jobs):
            audit = future.result()
            print(f"COMPLETE {audit.symbol} snapshots={audit.snapshots} rows={audit.rows} rejected={audit.rejected}", flush=True)
    development, groups = combine_symbol_rows(symbols=config["symbols"], output_root=args.output)
    features = sorted(column for column in development if column.startswith("feature__"))
    assert_feature_allowlist(features)
    holdout_path = args.output / "final_holdout_features_sealed.parquet"
    import pandas as pd
    holdout = pd.read_parquet(holdout_path)
    if any(column.startswith("target__") for column in holdout):
        raise RuntimeError("FINAL_HOLDOUT_LABEL_LEAK")
    if not holdout.label_state.eq("SEALED").all():
        raise RuntimeError("FINAL_HOLDOUT_STATE_INVALID")
    feature_dictionary = dictionary_payload(groups)
    dictionary_path = args.output / "feature_dictionary.json"
    dictionary_path.write_text(json.dumps(feature_dictionary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    source_manifest = args.candles / "dataset_manifest.json"
    audits = [json.loads((args.output / "by_symbol" / f"{symbol}.audit.json").read_text()) for symbol in config["symbols"]]
    manifest = {
        "schema": "independent-entry-quality-dataset-manifest-v1",
        "experiment": config["experiment"],
        "config_sha256": sha256(args.config),
        "source_manifest": str(source_manifest.relative_to(REPOSITORY)),
        "source_manifest_sha256": sha256(source_manifest),
        "feature_dictionary_sha256": sha256(dictionary_path),
        "feature_count": len(features),
        "development_rows": len(development),
        "development_market_state_groups": development.market_state_group_id.nunique(),
        "rows_by_split": development.split.value_counts().sort_index().to_dict(),
        "symbols_by_split": development.groupby("split").symbol.nunique().to_dict(),
        "final_holdout_rows": len(holdout),
        "cross_market_fail_closed_rows": sum(item["rows"] for item in audits) - len(development) - len(holdout) - len(pd.read_parquet(args.output / "embargo_features.parquet")),
        "final_holdout_feature_sha256": sha256(holdout_path),
        "final_holdout_labels_built": False,
        "aegis_fields_loaded": False,
        "phase2_candidate_fields_loaded": False,
        "symbol_audits": audits,
    }
    (args.output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("development_rows", "development_market_state_groups", "rows_by_split", "final_holdout_rows", "feature_count")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
