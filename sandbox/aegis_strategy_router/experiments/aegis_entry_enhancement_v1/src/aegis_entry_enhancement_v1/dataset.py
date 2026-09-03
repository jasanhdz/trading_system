"""Causal Aegis-signal panel construction and frozen module scoring."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import numpy as np
import pandas as pd

from aegis_strategy_router.domain.serialization import canonical_json_bytes, content_hash
from aegis_strategy_router.domain.types import DataStatus, Side, Timeframe
from aegis_strategy_router.replay.precomputed_snapshot_builder import PrecomputedSnapshotBuilder
from independent_entry_quality_v1.dataset import _targets, split_for
from independent_entry_quality_v1.features import add_cross_market_features, extract_features
from directional_alpha_v1.features import add_directional_features


TRADE_ID_TIME = re.compile(r"^AEGIS-TURBO-[A-Z0-9]+-(\d{8})-(\d{6})-(\d{3})$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def signal_timestamp(trade_id: str) -> pd.Timestamp:
    match = TRADE_ID_TIME.match(trade_id)
    if match is None:
        raise ValueError(f"UNPARSABLE_TRADE_ID:{trade_id}")
    day, clock, milliseconds = match.groups()
    return pd.Timestamp(f"{day[:4]}-{day[4:6]}-{day[6:]}T{clock[:2]}:{clock[2:4]}:{clock[4:]}.{milliseconds}Z")


def load_open_events(log_root: Path, required: set[str], config: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    events: dict[str, dict[str, Any]] = {}
    source_hashes: dict[str, str] = {}
    for path in sorted(log_root.glob("turbo_trades_*.jsonl")):
        used = False
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                trade_id = row.get("trade_id")
                if trade_id not in required or row.get("status") != "OPEN":
                    continue
                if row.get("strategy") != config["aegis_baseline"]["strategy"] or row.get("mode") != config["aegis_baseline"]["mode"]:
                    continue
                events[trade_id] = row
                used = True
        if used:
            source_hashes[str(path)] = sha256(path)
    return events, source_hashes


def load_signals(repository: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    csv_path = repository / config["aegis_baseline"]["source_csv"]
    if sha256(csv_path) != config["aegis_baseline"]["source_csv_sha256"]:
        raise RuntimeError("AEGIS_SOURCE_CSV_HASH_MISMATCH")
    source = pd.read_csv(csv_path)
    source["signal_timestamp"] = [signal_timestamp(value) for value in source.trade_id]
    source["opened_at"] = pd.to_datetime(source.opened_at, utc=True, format="mixed")
    source["split"] = [split_for(value, config["splits"]) for value in source.signal_timestamp]
    relevant = source.loc[source.split.ne("OUTSIDE")].copy()
    events, hashes = load_open_events(repository / "binance-futures-bot-ts/logs/aegis", set(relevant.trade_id), config)
    missing = sorted(set(relevant.trade_id) - set(events))
    if missing:
        raise RuntimeError(f"MISSING_OPEN_EVENTS:{len(missing)}")
    rows = []
    for item in relevant.to_dict("records"):
        event = events[item["trade_id"]]
        if event["side"] != item["side"] or event["symbol"] != item["symbol"]:
            raise RuntimeError(f"AEGIS_OPEN_MISMATCH:{item['trade_id']}")
        metadata = event.get("metadata") or {}
        policy = metadata.get("entryPolicy") or {}
        clean = metadata.get("cleanEntryGuard") or {}
        rows.append({
            "aegis_signal_episode_id": hashlib.sha256(item["trade_id"].encode()).hexdigest(),
            "trade_id": item["trade_id"], "symbol": item["symbol"], "side": item["side"],
            "signal_timestamp": item["signal_timestamp"], "opened_at": item["opened_at"],
            "entry_price": float(event["entry_price"]), "split": item["split"],
            "turbo_score": float(event.get("turbo_score", np.nan)),
            "votes_long": int((event.get("votes") or {}).get("long", 0)),
            "votes_short": int((event.get("votes") or {}).get("short", 0)),
            "raw_reason": metadata.get("rawReason"), "final_reason": policy.get("finalReason"),
            "aegis_entry_model_version": clean.get("entryQualityModelVersion") or clean.get("modelVersion"),
            "aegis_policy_metadata_hash": hashlib.sha256(canonical_json_bytes(metadata)).hexdigest(),
            "opened_minus_signal_ms": (item["opened_at"] - item["signal_timestamp"]).total_seconds() * 1000.0,
        })
    result = pd.DataFrame(rows).sort_values(["signal_timestamp", "trade_id"], kind="mergesort")
    invalid_timestamp = result.opened_minus_signal_ms.gt(config["aegis_baseline"]["maximum_open_minus_signal_ms"])
    excluded_timestamp_ids = result.loc[invalid_timestamp, "trade_id"].tolist()
    result = result.loc[~invalid_timestamp].copy()
    audit = {
        "source_rows": len(source), "in_scope_rows": len(result), "matched_open_events": len(events),
        "missing_open_events": len(missing), "source_log_hashes": hashes,
        "sides": result.side.value_counts().to_dict(), "splits": result.split.value_counts().to_dict(),
        "policy_hashes": result.aegis_policy_metadata_hash.nunique(),
        "timestamp_integrity_excluded": len(excluded_timestamp_ids),
        "timestamp_integrity_excluded_trade_ids": excluded_timestamp_ids,
        "entry_model_versions": {
            "MISSING_IN_SOURCE_EVENT" if pd.isna(key) else str(key): int(value)
            for key, value in result.aegis_entry_model_version.value_counts(dropna=False).items()
        },
    }
    return result, audit


def build_neutral_panel(signals: pd.DataFrame, candle_root: Path, symbols: list[str]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    times = sorted(signals.signal_timestamp.unique())
    builder = PrecomputedSnapshotBuilder()
    rows: list[dict[str, Any]] = []
    candles: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        frame = pd.read_parquet(candle_root / f"{symbol}_1m.parquet").sort_values("open_time_ms", kind="mergesort")
        candles[symbol] = frame
        opens = frame.open_time_ms.to_numpy("int64")
        for timestamp_value in times:
            timestamp = pd.Timestamp(timestamp_value)
            latest_open = int(timestamp.floor("min").timestamp() * 1000) - 60_000
            position = int(np.searchsorted(opens, latest_open, side="left"))
            if position >= len(frame) or int(opens[position]) != latest_open:
                continue
            try:
                snapshot = builder.build(
                    symbol=symbol, decision_at=timestamp.to_pydatetime(), built_at=timestamp.to_pydatetime(),
                    reference_price=float(frame.iloc[position].close), one_minute=frame,
                    source_versions={"aegis_entry_enhancement_source": builder.causal_source_hash(symbol, frame, timestamp.to_pydatetime())},
                )
                if any(state.status is not DataStatus.AVAILABLE or (state.structural and state.structural.status is not DataStatus.AVAILABLE) for state in snapshot.timeframes):
                    continue
                for side in (Side.LONG, Side.SHORT):
                    features, _, available_at = extract_features(snapshot, side)
                    row = {
                        "decision_at": timestamp, "symbol": symbol, "side": side.value,
                        "snapshot_id": snapshot.snapshot_id, "snapshot_schema_hash": snapshot.schema_hash,
                        "max_feature_available_at": available_at,
                    }
                    row.update({f"feature__{name}": value for name, value in features.items()})
                    rows.append(row)
            except ValueError:
                continue
    panel = pd.DataFrame(rows)
    panel, _ = add_cross_market_features(panel)
    feature_columns = [column for column in panel if column.startswith("feature__")]
    panel = panel.dropna(subset=feature_columns).copy()
    panel, _ = add_directional_features(panel)
    return panel, candles


def score_signals(
    *, signals: pd.DataFrame, panel: pd.DataFrame, candles: dict[str, pd.DataFrame],
    config: dict[str, Any], repository: Path,
) -> pd.DataFrame:
    opportunity_path = repository / config["frozen_modules"]["opportunity"]["artifact"]
    directional_path = repository / config["frozen_modules"]["directional"]["artifact"]
    if sha256(opportunity_path) != config["frozen_modules"]["opportunity"]["sha256"]:
        raise RuntimeError("OPPORTUNITY_ARTIFACT_HASH_MISMATCH")
    if sha256(directional_path) != config["frozen_modules"]["directional"]["sha256"]:
        raise RuntimeError("DIRECTIONAL_ARTIFACT_HASH_MISMATCH")
    opportunity_bundle = joblib.load(opportunity_path)
    directional_bundle = joblib.load(directional_path)
    v1_dataset = pd.read_parquet(repository / "sandbox/aegis_strategy_router/experiments/independent_entry_quality_discovery_v1/artifacts/dataset_v1/development_labeled.parquet")
    reference = v1_dataset.loc[v1_dataset.split.eq("TRAIN"), opportunity_bundle["features"]]
    lower, upper = reference.quantile(0.01), reference.quantile(0.99)
    keyed = panel.set_index(["decision_at", "symbol", "side"], drop=False)
    rows = []
    for signal in signals.to_dict("records"):
        timestamp, symbol, side = pd.Timestamp(signal["signal_timestamp"]), signal["symbol"], signal["side"]
        opposite = "SHORT" if side == "LONG" else "LONG"
        try:
            actual = keyed.loc[(timestamp, symbol, side)].copy()
            counter = keyed.loc[(timestamp, symbol, opposite)].copy()
            long_state = keyed.loc[(timestamp, symbol, "LONG")]
        except KeyError:
            continue
        opportunity_values = long_state[opportunity_bundle["features"]].to_frame().T
        opportunity_score = float(opportunity_bundle["opportunity"].predict_proba(opportunity_values)[:, 1][0])
        actual_values = actual[directional_bundle["features"]].to_frame().T
        counter_values = counter[directional_bundle["features"]].to_frame().T
        actual_net = float(directional_bundle["ridge"].predict(actual_values)[0])
        counter_net = float(directional_bundle["ridge"].predict(counter_values)[0])
        actual_probability = float(directional_bundle["logistic"].predict_proba(actual_values)[:, 1][0])
        counter_probability = float(directional_bundle["logistic"].predict_proba(counter_values)[:, 1][0])
        ood_fraction = float(((opportunity_values.iloc[0] < lower) | (opportunity_values.iloc[0] > upper)).mean())
        row = dict(signal)
        row.update({
            "snapshot_id": actual.snapshot_id, "snapshot_schema_hash": actual.snapshot_schema_hash,
            "max_feature_available_at": actual.max_feature_available_at,
            "opportunity_score": opportunity_score,
            "directional_probability_aegis_side": actual_probability,
            "directional_probability_opposite_side": counter_probability,
            "predicted_net_aegis_side_bps": actual_net,
            "predicted_net_opposite_side_bps": counter_net,
            "predicted_aegis_advantage_bps": actual_net - counter_net,
            "quality_score": opportunity_score * actual_probability,
            "ood_feature_fraction": ood_fraction,
            "ood": ood_fraction > config["population_shift"]["ood_feature_fraction_threshold"],
            "feature__tf15m__atr_percentile_96": actual["feature__tf15m__atr_percentile_96"],
            "feature__btc_1h_directional_return_bps": actual["feature__cross__btcusdt__tf1h__directional_return_3_bps"] * actual["feature__context__side_sign"],
        })
        if signal["split"] != "FINAL_HOLDOUT":
            frame = candles[symbol]
            latest_close = float(frame.loc[frame.open_time_ms.lt(int(timestamp.floor("min").timestamp() * 1000)), "close"].iloc[-1])
            state15_atr = float(actual["feature__tf15m__structural_atr_bps"]) * latest_close / 10_000.0
            target_snapshot = SimpleNamespace(
                decision_at=timestamp.to_pydatetime(), reference_price=float(signal["entry_price"]),
                timeframes=(SimpleNamespace(timeframe=Timeframe.M15, structural=SimpleNamespace(atr14=state15_atr)),),
            )
            row.update(_targets(frame, target_snapshot, Side(side), config))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["signal_timestamp", "trade_id"], kind="mergesort")
