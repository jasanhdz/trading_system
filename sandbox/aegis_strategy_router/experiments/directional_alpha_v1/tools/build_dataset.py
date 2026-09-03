#!/usr/bin/env python3
"""Build Directional Alpha V1 while preserving the sealed holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


EXPERIMENT = Path(__file__).resolve().parents[1]
SANDBOX = EXPERIMENT.parents[1]
REPOSITORY = SANDBOX.parents[1]
V1 = SANDBOX / "experiments/independent_entry_quality_discovery_v1"
for path in (EXPERIMENT / "src", V1 / "src", SANDBOX / "src", REPOSITORY / "src"):
    sys.path.insert(0, str(path))

from aegis_strategy_router.domain.serialization import content_hash  # noqa: E402
from aegis_strategy_router.domain.types import DataStatus, Side  # noqa: E402
from aegis_strategy_router.replay.precomputed_snapshot_builder import PrecomputedSnapshotBuilder  # noqa: E402
from independent_entry_quality_v1.dataset import _targets, build_symbol_rows, combine_symbol_rows, split_for  # noqa: E402
from independent_entry_quality_v1.features import extract_features  # noqa: E402
from directional_alpha_v1.features import (  # noqa: E402
    add_directional_features, assert_allowlist, dictionary_payload, feature_hash,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Audit:
    symbol: str
    snapshots: int
    rows: int
    rejected: int
    holdout_rows: int
    labeled_rows: int


def build_symbol_rows_gap_aware(*, symbol: str, candle_root: Path, config: dict, output_root: Path) -> Audit:
    frame = pd.read_parquet(candle_root / f"{symbol}_1m.parquet").sort_values("open_time_ms", kind="mergesort")
    builder = PrecomputedSnapshotBuilder()
    first = min(pd.Timestamp(bounds[0]) for bounds in config["splits"].values())
    last = max(pd.Timestamp(bounds[1]) for bounds in config["splits"].values()) - pd.Timedelta(hours=1)
    anchors = pd.date_range(first, last, freq="1h", tz="UTC")
    open_ms = frame.open_time_ms.to_numpy(dtype="int64", copy=False)
    rows, rejected, groups = [], [], {}
    for timestamp in anchors:
        partition = split_for(timestamp, config["splits"])
        latest_open = int(timestamp.timestamp() * 1000) - 60_000
        position = int(np.searchsorted(open_ms, latest_open, side="left"))
        if position >= len(frame) or int(open_ms[position]) != latest_open:
            rejected.append({"decision_at": timestamp.isoformat(), "reason": "REFERENCE_CANDLE_MISSING"})
            continue
        try:
            source_hash = builder.causal_source_hash(symbol, frame, timestamp.to_pydatetime())
            snapshot = builder.build(
                symbol=symbol, decision_at=timestamp.to_pydatetime(), built_at=timestamp.to_pydatetime(),
                reference_price=float(frame.iloc[position].close), one_minute=frame,
                source_versions={"directional_alpha_source_hash": source_hash},
            )
            incomplete = [
                state.timeframe.value for state in snapshot.timeframes
                if state.status is not DataStatus.AVAILABLE
                or (state.structural is not None and state.structural.status is not DataStatus.AVAILABLE)
            ]
            if incomplete:
                raise ValueError("INCOMPLETE_SNAPSHOT:" + ",".join(incomplete))
            group_id = content_hash({"experiment": config["experiment"], "symbol": symbol, "decision_at": timestamp.to_pydatetime()})
            for side in (Side.LONG, Side.SHORT):
                features, local_groups, available_at = extract_features(snapshot, side)
                groups.update(local_groups)
                row = {
                    "row_id": content_hash({"group": group_id, "side": side.value}),
                    "market_state_group_id": group_id, "temporal_block_id": timestamp.strftime("%Y-%m-%dT%H"),
                    "symbol": symbol, "decision_at": timestamp, "side": side.value,
                    "snapshot_id": snapshot.snapshot_id, "snapshot_schema_version": snapshot.schema_version,
                    "snapshot_schema_hash": snapshot.schema_hash, "source_hash": source_hash,
                    "max_feature_available_at": available_at, "split": partition,
                    "label_state": "SEALED" if partition == "FINAL_HOLDOUT" else ("EMBARGO" if partition.startswith("EMBARGO") else "LABELED"),
                }
                row.update({f"feature__{name}": value for name, value in features.items()})
                if row["label_state"] == "LABELED":
                    future = frame.iloc[position + 1:position + 61]
                    expected = np.arange(
                        int(timestamp.timestamp() * 1000),
                        int(timestamp.timestamp() * 1000) + 60 * 60_000,
                        60_000,
                        dtype="int64",
                    )
                    if len(future) != 60 or not np.array_equal(future.open_time_ms.to_numpy(dtype="int64"), expected):
                        raise ValueError("OUTCOME_HORIZON_GAP")
                    row.update(_targets(future, snapshot, side, config))
                rows.append(row)
        except ValueError as error:
            rejected.append({"decision_at": timestamp.isoformat(), "reason": str(error)})
    destination = output_root / "by_symbol"
    destination.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_parquet(destination / f"{symbol}.parquet", index=False, compression="zstd")
    audit = Audit(symbol, result.snapshot_id.nunique(), len(result), len(rejected), int(result.label_state.eq("SEALED").sum()), int(result.label_state.eq("LABELED").sum()))
    source_manifest = json.loads((candle_root / "dataset_manifest.json").read_text())["symbols"][symbol]
    (destination / f"{symbol}.audit.json").write_text(json.dumps({
        **audit.__dict__, "source_gaps": source_manifest["gaps"], "rejections": rejected,
        "aegis_loaded": False, "phase2_candidates_loaded": False, "holdout_labels_built": False,
    }, indent=2, sort_keys=True) + "\n")
    (destination / f"{symbol}.groups.json").write_text(json.dumps(groups, indent=2, sort_keys=True) + "\n")
    return audit


def add_pair_targets(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    utility = result.pivot(index="market_state_group_id", columns="side", values="target__net_common_payoff_bps")
    result["target__utility_long_bps"] = result.market_state_group_id.map(utility.LONG)
    result["target__utility_short_bps"] = result.market_state_group_id.map(utility.SHORT)
    result["target__directional_advantage_long_minus_short_bps"] = result.target__utility_long_bps - result.target__utility_short_bps
    result["target__side_advantage_bps"] = result["feature__context__side_sign"] * result.target__directional_advantage_long_minus_short_bps
    margin = 20.0
    result["target__economic_side_label"] = "SKIP"
    result.loc[(result.target__utility_long_bps > 0) & (result.target__directional_advantage_long_minus_short_bps >= margin), "target__economic_side_label"] = "LONG"
    result.loc[(result.target__utility_short_bps > 0) & (result.target__directional_advantage_long_minus_short_bps <= -margin), "target__economic_side_label"] = "SHORT"
    return result


def enrich(frame: pd.DataFrame, *, opportunity_bundle: dict, config: dict, thresholds: dict[str, float] | None) -> tuple[pd.DataFrame, dict[str, str], dict[str, float]]:
    result = frame.copy()
    result["market_state_group_id"] = [
        content_hash({"experiment": config["experiment"], "symbol": symbol, "decision_at": timestamp.to_pydatetime()})
        for symbol, timestamp in zip(result.symbol, pd.to_datetime(result.decision_at, utc=True))
    ]
    result["row_id"] = [content_hash({"group": group, "side": side}) for group, side in zip(result.market_state_group_id, result.side)]
    result, groups = add_directional_features(result)
    model_features = opportunity_bundle["features"]
    missing = sorted(set(model_features) - set(result.columns))
    if missing:
        raise RuntimeError(f"OPPORTUNITY_SCHEMA_MISSING:{missing}")
    long = result.loc[result.side.eq("LONG")]
    state_scores = opportunity_bundle["opportunity"].predict_proba(long[model_features])[:, 1]
    score_map = dict(zip(long.market_state_group_id, state_scores))
    result["opportunity_score_frozen"] = result.market_state_group_id.map(score_map)
    if result.opportunity_score_frozen.isna().any():
        raise RuntimeError("OPPORTUNITY_SCORE_JOIN_MISSING")
    if thresholds is None:
        train_scores = result.loc[result.split.eq("TRAIN") & result.side.eq("LONG"), "opportunity_score_frozen"]
        thresholds = {
            str(quantile): float(train_scores.quantile(quantile))
            for quantile in config["opportunity_model"]["diagnostic_population_quantiles"]
        }
    for quantile, threshold in thresholds.items():
        result[f"opportunity_top_{int(float(quantile) * 100)}"] = result.opportunity_score_frozen.ge(threshold)
    directional_columns = sorted(groups)
    result["directional_feature_hash"] = [feature_hash(row, directional_columns) for row in result[directional_columns].to_dict("records")]
    return result, groups, thresholds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=EXPERIMENT / "config/preregistration_v1.json")
    parser.add_argument("--candles", type=Path, default=REPOSITORY / "data/directional_alpha_v1/candles_1m")
    parser.add_argument("--output", type=Path, default=EXPERIMENT / "artifacts/dataset_v1")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    model_path = REPOSITORY / config["opportunity_model"]["artifact"]
    if sha256(model_path) != config["opportunity_model"]["artifact_sha256"]:
        raise RuntimeError("FROZEN_OPPORTUNITY_ARTIFACT_HASH_MISMATCH")
    bundle = joblib.load(model_path)
    args.output.mkdir(parents=True, exist_ok=True)
    pending = [
        symbol for symbol in config["symbols"]
        if args.overwrite or not (args.output / "by_symbol" / f"{symbol}.audit.json").exists()
    ]
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        jobs = {
            executor.submit(build_symbol_rows, symbol=symbol, candle_root=args.candles, config=config, output_root=args.output): symbol
            for symbol in pending
        }
        for future in as_completed(jobs):
            audit = future.result()
            print(f"COMPLETE {audit.symbol} snapshots={audit.snapshots} rejected={audit.rejected}", flush=True)
    development, _ = combine_symbol_rows(symbols=config["symbols"], output_root=args.output)
    holdout = pd.read_parquet(args.output / "final_holdout_features_sealed.parquet")
    embargo = pd.read_parquet(args.output / "embargo_features.parquet")
    development, groups, thresholds = enrich(development, opportunity_bundle=bundle, config=config, thresholds=None)
    development = add_pair_targets(development)
    holdout, holdout_groups, _ = enrich(holdout, opportunity_bundle=bundle, config=config, thresholds=thresholds)
    embargo, embargo_groups, _ = enrich(embargo, opportunity_bundle=bundle, config=config, thresholds=thresholds)
    groups.update(holdout_groups)
    groups.update(embargo_groups)
    if any(column.startswith("target__") for column in holdout):
        raise RuntimeError("FINAL_HOLDOUT_LABEL_LEAK")
    directional_columns = sorted(groups)
    assert_allowlist(directional_columns)
    development.to_parquet(args.output / "development_labeled.parquet", index=False, compression="zstd")
    holdout.to_parquet(args.output / "final_holdout_features_sealed.parquet", index=False, compression="zstd")
    embargo.to_parquet(args.output / "embargo_features.parquet", index=False, compression="zstd")
    dictionary = dictionary_payload(groups)
    dictionary_path = args.output / "feature_dictionary.json"
    dictionary_path.write_text(json.dumps(dictionary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema": "directional-alpha-v1-dataset-manifest", "experiment": config["experiment"],
        "config_sha256": sha256(args.config), "source_manifest_sha256": sha256(args.candles / "dataset_manifest.json"),
        "opportunity_model_sha256": sha256(model_path), "opportunity_retrained": False,
        "opportunity_thresholds_from_train_scores_only": thresholds,
        "rows_by_split": development.split.value_counts().sort_index().to_dict(),
        "states_by_split": development.groupby("split").market_state_group_id.nunique().to_dict(),
        "primary_population_rows_by_split": development.loc[development.opportunity_top_90].split.value_counts().sort_index().to_dict(),
        "directional_feature_count": len(directional_columns), "feature_dictionary_sha256": sha256(dictionary_path),
        "final_holdout_rows": len(holdout), "final_holdout_labels_built": False,
        "aegis_fields_loaded": False, "phase2_candidate_fields_loaded": False,
    }
    (args.output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
