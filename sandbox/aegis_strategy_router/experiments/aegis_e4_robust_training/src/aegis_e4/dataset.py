"""Build the E4 5-minute causal panel and sealed-holdout dataset."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import feature_schema, split_for, stable_hash
from .features import add_cross_market, assert_causal_availability, build_neutral_symbol_panel, orient_sides


def load_candles(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).sort_values("open_time_ms", kind="mergesort").reset_index(drop=True)
    required = {"open_time_ms", "open", "high", "low", "close", "volume", "taker_buy_volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"SOURCE_COLUMNS_MISSING:{sorted(missing)}")
    if frame.open_time_ms.duplicated().any():
        raise ValueError("DUPLICATE_MINUTE")
    gaps = np.diff(frame.open_time_ms.to_numpy(np.int64))
    if len(gaps) and not np.all(gaps == 60_000):
        raise ValueError("SOURCE_MINUTE_GAP")
    return frame


def build_dataset(config: dict[str, Any], output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    source_root = Path(config["source"]["root"])
    first = min(pd.Timestamp(bounds[0]) for bounds in config["splits"].values())
    last = max(pd.Timestamp(bounds[1]) for bounds in config["splits"].values())
    anchors = pd.date_range(first, last, freq="5min", inclusive="left", tz="UTC")
    neutral_parts, candle_map, family_map = [], {}, {}
    source_manifest = json.loads((source_root / "dataset_manifest.json").read_text())
    audits = []
    for symbol in config["symbols"]:
        candles = load_candles(source_root / f"{symbol}_1m.parquet")
        candle_map[symbol] = candles
        panel, families = build_neutral_symbol_panel(
            candles, anchors, list(config["source"]["timeframes_minutes"])
        )
        panel["symbol"] = symbol
        panel["split"] = panel.decision_at.map(lambda value: split_for(value, config["splits"]))
        neutral_parts.append(panel)
        family_map.update(families)
        audits.append({
            "symbol": symbol, "source_rows": len(candles),
            "first_open_ms": int(candles.open_time_ms.iloc[0]),
            "last_open_ms": int(candles.open_time_ms.iloc[-1]), "gaps": 0, "duplicates": 0,
        })
    neutral = pd.concat(neutral_parts, ignore_index=True)
    neutral, cross_families = add_cross_market(neutral)
    family_map.update(cross_families)
    sided, sided_families = orient_sides(neutral, family_map)
    family_map = sided_families
    sided["split"] = sided.decision_at.map(lambda value: split_for(value, config["splits"]))
    sided["market_state_id"] = [stable_hash(["E4", symbol, str(timestamp)]) for symbol, timestamp in zip(sided.symbol, sided.decision_at)]
    sided["row_id"] = [stable_hash([state, side]) for state, side in zip(sided.market_state_id, sided.side)]
    sided["episode_id"] = [
        stable_hash(["E4_EPISODE", symbol, pd.Timestamp(timestamp).floor("60min").isoformat()])
        for symbol, timestamp in zip(sided.symbol, sided.decision_at)
    ]
    sided["temporal_block_id"] = sided.decision_at.dt.floor("60min").astype(str)
    sided["label_state"] = np.where(
        sided.split.eq("FINAL_HOLDOUT"), "SEALED",
        np.where(sided.split.str.startswith("EMBARGO"), "EMBARGO", "LABELED"),
    )
    feature_columns = [name for name in sided.columns if name.startswith("feature__")]
    complete = np.isfinite(sided[feature_columns].to_numpy(float)).all(axis=1)
    ineligible_feature_rows = int((~complete).sum())
    sided = sided.loc[complete].reset_index(drop=True)
    assert_causal_availability(sided)
    labeled_parts, sealed_parts, embargo_parts = [], [], []
    for symbol, group in sided.groupby("symbol", sort=True):
        labels = build_targets(candle_map[symbol], group, config)
        labeled = group.loc[group.label_state.eq("LABELED")].merge(labels, on="row_id", how="left", validate="one_to_one")
        if labeled.filter(like="target__").isna().all(axis=1).any():
            raise ValueError(f"TARGETS_MISSING:{symbol}")
        labeled_parts.append(labeled)
        sealed_parts.append(group.loc[group.label_state.eq("SEALED")].copy())
        embargo_parts.append(group.loc[group.label_state.eq("EMBARGO")].copy())
    development = pd.concat(labeled_parts, ignore_index=True)
    sealed = pd.concat(sealed_parts, ignore_index=True)
    embargo = pd.concat(embargo_parts, ignore_index=True)
    if any(column.startswith("target__") for column in sealed.columns):
        raise ValueError("SEALED_HOLDOUT_TARGET_LEAK")
    development.to_parquet(output / "development_labeled.parquet", index=False, compression="zstd")
    sealed.to_parquet(output / "final_holdout_features_sealed.parquet", index=False, compression="zstd")
    embargo.to_parquet(output / "embargo_features.parquet", index=False, compression="zstd")
    schema = feature_schema(family_map)
    (output / "feature_schema.json").write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema": "aegis-e4-dataset-manifest-v1",
        "classification": config["classification"],
        "source_manifest_sha256": stable_hash(source_manifest),
        "feature_schema_sha256": schema["sha256"],
        "decision_cadence_minutes": 5,
        "raw_rows": int(len(development)),
        "market_states": int(development.market_state_id.nunique()),
        "effective_symbol_episodes": int(development.episode_id.nunique()),
        "effective_temporal_blocks": int(development.temporal_block_id.nunique()),
        "rows_by_split": development.groupby("split").size().astype(int).to_dict(),
        "episodes_by_split": development.groupby("split").episode_id.nunique().astype(int).to_dict(),
        "sealed_holdout_rows": int(len(sealed)),
        "sealed_holdout_targets_built": False,
        "ineligible_feature_rows_fail_closed": ineligible_feature_rows,
        "source_audit": audits,
    }
    manifest["manifest_sha256"] = stable_hash(manifest)
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def build_targets(candles: pd.DataFrame, rows: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    eligible = rows.loc[rows.label_state.eq("LABELED")].copy()
    if eligible.empty:
        return pd.DataFrame({"row_id": []})
    horizon = int(config["source"]["horizon_minutes"])
    open_ms = candles.open_time_ms.to_numpy(np.int64)
    starts = np.searchsorted(open_ms, (eligible.decision_at.astype("int64") // 1_000_000).to_numpy(np.int64))
    valid = starts + horizon <= len(candles)
    if not valid.all():
        eligible = eligible.loc[valid].copy()
        starts = starts[valid]
    index = starts[:, None] + np.arange(horizon)[None, :]
    highs = candles.high.to_numpy(float)[index]
    lows = candles.low.to_numpy(float)[index]
    closes = candles.close.to_numpy(float)[index]
    entries = candles.open.to_numpy(float)[starts]
    direction = eligible.side.map({"LONG": 1.0, "SHORT": -1.0}).to_numpy(float)
    atr_bps = eligible["feature__base__tf15m__atr_pct_bps"].to_numpy(float)
    barrier_bps = float(config["labels"]["barrier_atr"]) * atr_bps
    favorable_path = np.where(direction[:, None] > 0, (highs / entries[:, None] - 1.0), (1.0 - lows / entries[:, None])) * 10_000.0
    adverse_path = np.where(direction[:, None] > 0, (1.0 - lows / entries[:, None]), (highs / entries[:, None] - 1.0)) * 10_000.0
    favorable_hit = favorable_path >= barrier_bps[:, None]
    adverse_hit = adverse_path >= barrier_bps[:, None]
    f_any, a_any = favorable_hit.any(axis=1), adverse_hit.any(axis=1)
    f_first = np.where(f_any, favorable_hit.argmax(axis=1), horizon)
    a_first = np.where(a_any, adverse_hit.argmax(axis=1), horizon)
    favorable_first = f_any & (f_first < a_first)
    adverse_first = a_any & (a_first <= f_first)
    neither = ~(favorable_first | adverse_first)
    mfe = np.maximum(0.0, favorable_path.max(axis=1))
    mae = np.maximum(0.0, adverse_path.max(axis=1))
    fixed_return = direction * (closes[:, -1] / entries - 1.0) * 10_000.0
    gross = np.where(favorable_first, barrier_bps, np.where(adverse_first, -barrier_bps, fixed_return))
    severe = mae >= float(config["labels"]["severe_mae_bps"])
    late = (fixed_return > 0.0) & (mfe < float(config["labels"]["late_entry_remaining_mfe_bps"]))
    close_at_decision = candles.close.to_numpy(float)[np.maximum(0, starts - 1)]
    implementation_shortfall = direction * (entries / close_at_decision - 1.0) * 10_000.0
    return pd.DataFrame({
        "row_id": eligible.row_id.to_numpy(),
        "target__favorable_first": favorable_first.astype(int),
        "target__adverse_first": adverse_first.astype(int),
        "target__neither": neither.astype(int),
        "target__tail_risk": severe.astype(int),
        "target__entry_quality": ((mfe > mae) & (gross > 0.0)).astype(int),
        "target__late_entry_risk": late.astype(int),
        "target__mfe_bps": mfe,
        "target__mae_bps": mae,
        "target__mfe_minus_mae_bps": mfe - mae,
        "target__fixed_return_bps": fixed_return,
        "target__gross_common_payoff_bps": gross,
        "target__net_14bps": gross - float(config["labels"]["cost_bps"]),
        "target__net_20bps": gross - float(config["labels"]["stress_cost_bps"]),
        "target__time_to_favorable_minutes": np.where(f_any, f_first + 1, np.nan),
        "target__time_to_adverse_minutes": np.where(a_any, a_first + 1, np.nan),
        "target__entry_implementation_shortfall_bps": implementation_shortfall,
        "target__realistic_net_14bps": gross - float(config["labels"]["cost_bps"]) - implementation_shortfall,
        "target__barrier_bps": barrier_bps,
    })
