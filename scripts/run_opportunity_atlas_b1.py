#!/usr/bin/env python3
"""Build and evaluate the preregistered Opportunity Atlas B1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aegis.config import CANONICAL_SYMBOLS
from aegis.research.market_event_economic_path_m1b import load_funding
from aegis.research.opportunity_atlas_b1 import (
    EVENT_FEATURES,
    SYMBOL_FEATURES,
    attach_path_targets,
    build_event_targets,
    combined_policy,
    component_metrics,
    economic_summary,
    event_features,
    feature_contract_hash,
    fit_models,
    partition_mask,
    symbol_side_rows,
)
from aegis.utils import sha256_file


PARTITIONS = {
    "TRAIN": ("2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"),
    "CALIBRATION": ("2025-01-02T00:00:00Z", "2025-07-01T00:00:00Z"),
    "VALIDATION": ("2025-07-02T00:00:00Z", "2026-01-01T00:00:00Z"),
    "PSEUDO_FORWARD": ("2026-01-02T00:00:00Z", "2026-08-01T00:00:00Z"),
}
HORIZONS = (60, 240)


def _safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return _safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return "INF" if value > 0 else "-INF"
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _bootstrap(rows: pd.DataFrame, repetitions: int = 1000) -> dict[str, float] | None:
    if rows.empty:
        return None
    grouped = rows.assign(
        day=pd.to_datetime(rows.timestamp_ms, unit="ms", utc=True).dt.floor("1D")
    ).groupby("day")["net_primary"].apply(np.asarray)
    days = list(grouped)
    random = np.random.default_rng(1806)
    values = []
    for _ in range(repetitions):
        sample = np.concatenate([days[index] for index in random.integers(0, len(days), len(days))])
        values.append(float(sample.mean()))
    return {"lower_95": float(np.quantile(values, .025)), "upper_95": float(np.quantile(values, .975))}


def _control_rows(selected: pd.DataFrame, rows: pd.DataFrame, kind: str) -> pd.DataFrame:
    output = []
    for event in selected.itertuples(index=False):
        pool = rows.loc[rows.timestamp_ms.eq(event.timestamp_ms)].copy()
        if pool.empty:
            continue
        if kind == "RANDOM":
            digest = hashlib.sha256(f"b1:{event.timestamp_ms}".encode("ascii")).digest()
            candidate = pool.sort_values(["symbol", "side"]).iloc[int.from_bytes(digest[:8], "big") % len(pool)]
        elif kind == "BTC":
            candidate = pool.loc[pool.symbol.eq("BTCUSDT") & pool.side.eq(event.side)].iloc[0]
        elif kind == "MOMENTUM":
            side = "LONG" if float(pool.loc[pool.symbol.eq("BTCUSDT"), "return_4h"].iloc[0]) >= 0 else "SHORT"
            aligned = pool.loc[pool.side.eq(side)]
            candidate = aligned.sort_values(["return_4h", "symbol"], ascending=[side == "SHORT", True]).iloc[0]
        else:
            side = "SHORT" if float(pool.loc[pool.symbol.eq("BTCUSDT"), "return_4h"].iloc[0]) >= 0 else "LONG"
            aligned = pool.loc[pool.side.eq(side)]
            candidate = aligned.sort_values(["return_4h", "symbol"], ascending=[side == "SHORT", True]).iloc[0]
        output.append({
            "timestamp_ms": int(event.timestamp_ms), "symbol": str(candidate.symbol), "side": str(candidate.side),
            "gross_return": float(candidate.gross_return), "net_primary": float(candidate.gross_return - .0014),
            "net_stress": float(candidate.gross_return - .0020), "mae": float(candidate.mae), "mfe": float(candidate.mfe),
        })
    return pd.DataFrame(output)


def _component_gate(validation: dict[str, Any], forward: dict[str, Any]) -> dict[str, bool]:
    return {
        "opportunity_validation_auc": validation["opportunity"]["roc_auc"] >= .55,
        "opportunity_forward_auc": forward["opportunity"]["roc_auc"] >= .55,
        "opportunity_brier": validation["opportunity"]["brier"] < validation["opportunity"]["base_rate_brier"],
        "direction_validation": validation["direction"]["balanced_accuracy"] >= .55,
        "direction_forward": forward["direction"]["balanced_accuracy"] >= .55,
        "ranking_validation": all(value["spearman"] >= .05 for value in validation["ranking"].values()),
        "ranking_forward": all(value["spearman"] >= .05 for value in forward["ranking"].values()),
        "mae_validation": all(value["mae_spearman"] >= .10 for value in validation["path_risk"].values()),
        "mae_forward": all(value["mae_spearman"] >= .10 for value in forward["path_risk"].values()),
    }


def _combined_gate(selected: pd.DataFrame, controls: dict[str, pd.DataFrame], components: dict[str, bool]) -> dict[str, bool]:
    summary = economic_summary(selected)
    bootstrap = _bootstrap(selected)
    return {
        "minimum_events": len(selected) >= 100,
        "positive_net": summary.get("expectancy", -math.inf) > 0,
        "bootstrap_lower_positive": bool(bootstrap and bootstrap["lower_95"] > 0),
        "profit_factor": summary.get("profit_factor", 0) > 1.10,
        "stress_positive": economic_summary(selected, "net_stress").get("expectancy", -math.inf) > 0,
        "positive_thirds": sum(value > 0 for value in summary.get("temporal_thirds", ())) >= 2,
        "positive_symbols": summary.get("positive_symbols", 0) >= 7,
        "symbol_concentration": summary.get("maximum_symbol_share", math.inf) <= .25,
        "outperform_controls": all(summary.get("expectancy", -math.inf) > economic_summary(rows).get("expectancy", math.inf) for rows in controls.values()),
        "component_gates": all(components.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("data/opportunity_atlas_b1/run_01"))
    parser.add_argument("--reuse-dataset", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output_root if args.output_root.is_absolute() else root / args.output_root
    output.mkdir(parents=True, exist_ok=True)
    os.chmod(output, 0o700)
    panel_path = root / "data/economic_alpha_discovery_a1/run_01/causal_panel.parquet"
    if sha256_file(panel_path) != "a07fd4f2cafc796239e7a9c13c13037c18ed81fed8285d3ce83ddfb6f77f9797":
        raise RuntimeError("AEGIS_B1_SOURCE_PANEL_HASH_MISMATCH")
    panel = pd.read_parquet(panel_path)
    event_base = event_features(panel)
    hourly_states = panel.loc[panel.timestamp_ms.isin(event_base.timestamp_ms)].dropna(subset=list(SYMBOL_FEATURES)).copy()
    minute_root = root / "data/market_event_fast_track_m1a/full_run_01/cache"
    funding_root = root / "data/market_event_economic_path_m1b/raw"
    minute = {symbol: pd.read_parquet(minute_root / f"{symbol}.parquet") for symbol in CANONICAL_SYMBOLS}
    funding = {symbol: load_funding(funding_root, symbol) for symbol in CANONICAL_SYMBOLS}
    reports, gates, artifacts = {}, {}, []
    for horizon in HORIZONS:
        event_path = output / f"events_{horizon}m.parquet"
        row_path = output / f"symbol_sides_{horizon}m.parquet"
        if args.reuse_dataset and event_path.exists() and row_path.exists():
            events, rows = pd.read_parquet(event_path), pd.read_parquet(row_path)
        else:
            targets = attach_path_targets(hourly_states, minute, funding, horizon)
            events = build_event_targets(event_base, targets)
            rows = symbol_side_rows(targets)
            events.to_parquet(event_path, compression="zstd", index=False)
            rows.to_parquet(row_path, compression="zstd", index=False)
            os.chmod(event_path, 0o600); os.chmod(row_path, 0o600)
        masks = {name: partition_mask(events, *bounds) for name, bounds in PARTITIONS.items()}
        models = fit_models(events, rows, masks["TRAIN"], masks["CALIBRATION"])
        validation = component_metrics(models, events, rows, masks["VALIDATION"])
        forward = component_metrics(models, events, rows, masks["PSEUDO_FORWARD"])
        components = _component_gate(validation, forward)
        partition_reports = {}
        horizon_pass = True
        for name in ("VALIDATION", "PSEUDO_FORWARD"):
            selected = combined_policy(models, events, rows, masks[name])
            controls = {kind.lower(): _control_rows(selected, rows, kind) for kind in ("RANDOM", "BTC", "MOMENTUM", "REVERSAL")}
            combined = _combined_gate(selected, controls, components)
            horizon_pass &= all(combined.values())
            partition_reports[name] = {
                "selected": economic_summary(selected), "bootstrap": _bootstrap(selected),
                "controls": {key: economic_summary(value) for key, value in controls.items()},
                "combined_gate": combined, "combined_gate_pass": all(combined.values()),
            }
            if not selected.empty:
                selected.assign(partition=name).to_parquet(output / f"selected_{horizon}m_{name.lower()}.parquet", compression="zstd", index=False)
        reports[f"{horizon}m"] = {
            "event_rows": len(events), "symbol_side_rows": len(rows),
            "opportunity_threshold": models.opportunity_threshold,
            "components": {"VALIDATION": validation, "PSEUDO_FORWARD": forward},
            "component_gate": components, "component_gate_pass": all(components.values()),
            "partitions": partition_reports, "horizon_gate_pass": horizon_pass,
        }
        gates.append(horizon_pass)
        artifacts.extend([event_path, row_path])
        print(f"b1_horizon={horizon} events={len(events)} rows={len(rows)} pass={horizon_pass}", flush=True)
    result = {
        "schema_version": "aegis-opportunity-atlas-b1-result-v1", "experiment_id": "aegis-opportunity-atlas-b1-01",
        "evidence_class": "CONTAMINATED_ARCHITECTURE_DIAGNOSTIC_NO_PROMOTION_AUTHORITY",
        "feature_contracts": {"event": {"names": EVENT_FEATURES, "sha256": feature_contract_hash(EVENT_FEATURES)}, "symbol": {"names": SYMBOL_FEATURES, "sha256": feature_contract_hash(SYMBOL_FEATURES)}},
        "artifacts": {path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size} for path in artifacts},
        "results": reports, "B1_COMPONENT_SEPARABILITY_FOUND": any(all(report["component_gate"].values()) for report in reports.values()),
        "B1_COMBINED_POLICY_FOUND": any(gates), "B1_READY_FOR_FORWARD_EXPERIMENT": any(gates),
        "B1_READY_FOR_SHADOW": False, "B1_READY_FOR_LIVE": False,
        "exchange_calls": 0, "exchange_mutations": 0, "runtime_changes": "NONE",
    }
    result_path = output / "result.json"
    result_path.write_text(json.dumps(_safe(result), indent=2, sort_keys=True) + "\n")
    os.chmod(result_path, 0o600)
    print(json.dumps({"result": str(result_path), "horizon_passes": gates}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
