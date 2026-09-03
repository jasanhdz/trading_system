"""Dataset loading, audit, and causal feature building for E4 Risk Guard Validation."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aegis_strategy_router.domain.serialization import canonical_json_bytes
from aegis_e4.features import build_neutral_symbol_panel, add_cross_market, orient_sides, assert_causal_availability


TRADE_ID_TIME = re.compile(r"^AEGIS-TURBO-[A-Z0-9]+-(\d{8})-(\d{6})-(\d{3})$")


def sha256_file(path: Path) -> str:
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


def split_for(timestamp: pd.Timestamp, splits: dict[str, list[str]]) -> str:
    val = pd.to_datetime(timestamp, utc=True)
    for name, bounds in splits.items():
        if pd.Timestamp(bounds[0]) <= val < pd.Timestamp(bounds[1]):
            return name
    return "OUTSIDE"


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
            source_hashes[str(path)] = sha256_file(path)
    return events, source_hashes


def load_and_audit_signals(repository: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    csv_path = repository / config["aegis_baseline"]["source_csv"]
    expected_hash = config["aegis_baseline"]["source_csv_sha256"]
    actual_hash = sha256_file(csv_path)
    if actual_hash != expected_hash:
        raise RuntimeError(f"AEGIS_SOURCE_CSV_HASH_MISMATCH: expected {expected_hash}, got {actual_hash}")

    source = pd.read_csv(csv_path)
    source["signal_timestamp"] = [signal_timestamp(val) for val in source.trade_id]
    source["opened_at_dt"] = pd.to_datetime(source.opened_at, utc=True, format="mixed")
    source["split"] = [split_for(val, config["splits"]) for val in source.signal_timestamp]

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
        
        # Economic values from classification CSV and trade logs
        pnl_usdt = float(item.get("pnl_usdt", 0.0))
        roe = float(item.get("roe", 0.0))
        leverage = float(item.get("leverage", 15.0))
        mfe_bps = float(item.get("mfe_bps_underlying", 0.0))
        mae_bps = float(item.get("mae_bps_underlying", 0.0))
        exit_type = str(item.get("exit_type", "UNKNOWN"))
        duration_min = float(item.get("duration_minutes", 0.0))
        
        # Realized net bps (unleveraged return from entry to exit minus baseline costs)
        # roe = unleveraged_return * leverage => unleveraged_return = roe / leverage
        # in bps: (roe / leverage) * 10,000
        gross_bps_realized = (roe / max(1.0, leverage)) * 10_000.0
        
        rows.append({
            "trade_id": item["trade_id"],
            "signal_timestamp": item["signal_timestamp"],
            "opened_at": item["opened_at_dt"],
            "symbol": item["symbol"],
            "side": item["side"],
            "split": item["split"],
            "entry_price": float(event["entry_price"]),
            "leverage": leverage,
            "pnl_usdt": pnl_usdt,
            "roe": roe,
            "gross_bps": gross_bps_realized,
            "mfe_bps": mfe_bps,
            "mae_bps": mae_bps,
            "mfe_mae_ratio": float(mfe_bps / max(1e-6, mae_bps)),
            "exit_type": exit_type,
            "duration_minutes": duration_min,
            "bad_entry": int(item.get("bad_entry", 1 if roe <= 0 else 0)),
            "good_entry": int(item.get("good_entry", 1 if roe > 0 else 0)),
            "turbo_score": float(event.get("turbo_score", item.get("turbo_score", np.nan))),
            "entry_quality_score": float(item.get("entry_quality_score", np.nan)),
            "tail_risk_score_e3": float(item.get("tail_risk_score", np.nan)),
            "chop_risk": float(item.get("chop_risk", np.nan)),
            "exhaustion_risk": float(item.get("exhaustion_risk", np.nan)),
            "aegis_entry_model_version": clean.get("entryQualityModelVersion") or clean.get("modelVersion"),
            "aegis_policy_metadata_hash": hashlib.sha256(canonical_json_bytes(metadata)).hexdigest(),
            "opened_minus_signal_ms": (item["opened_at_dt"] - item["signal_timestamp"]).total_seconds() * 1000.0,
        })

    result = pd.DataFrame(rows).sort_values(["signal_timestamp", "trade_id"], kind="mergesort")
    invalid_timestamp = result.opened_minus_signal_ms.gt(config["aegis_baseline"]["maximum_open_minus_signal_ms"])
    excluded_ids = result.loc[invalid_timestamp, "trade_id"].tolist()
    result = result.loc[~invalid_timestamp].copy()

    audit = {
        "source_rows": len(source),
        "in_scope_rows": len(result),
        "matched_open_events": len(events),
        "missing_open_events": len(missing),
        "source_log_hashes": hashes,
        "sides": result.side.value_counts().to_dict(),
        "splits": result.split.value_counts().to_dict(),
        "policy_hashes": int(result.aegis_policy_metadata_hash.nunique()),
        "timestamp_integrity_excluded": len(excluded_ids),
        "timestamp_integrity_excluded_trade_ids": excluded_ids,
        "symbols": result.symbol.value_counts().to_dict(),
    }
    return result, audit


def build_e4_panel_for_signals(signals: pd.DataFrame, candle_root: Path, symbols: list[str]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    times = pd.DatetimeIndex(sorted(signals.signal_timestamp.unique()))
    candles: dict[str, pd.DataFrame] = {}
    panel_pieces = []
    families: dict[str, str] = {}

    for sym in symbols:
        c_path = candle_root / f"{sym}_1m.parquet"
        frame = pd.read_parquet(c_path).sort_values("open_time_ms", kind="mergesort")
        candles[sym] = frame
        p, fam = build_neutral_symbol_panel(frame, times, [5, 15, 60, 240])
        p["symbol"] = sym
        panel_pieces.append(p)
        families.update(fam)

    panel = pd.concat(panel_pieces, ignore_index=True)
    panel, cross_fam = add_cross_market(panel)
    families.update(cross_fam)
    side_panel, side_fam = orient_sides(panel, families)
    families.update(side_fam)

    # Check causality
    assert_causal_availability(side_panel)

    return side_panel, candles
