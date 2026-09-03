"""Closed-list compatibility replay coordinator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .historical_adapter import reproduce_historical_trades
from .manifests import atomic_write, digest, sha256_file
from .schemas import ReplayConfig


class HistoricalReproductionMismatch(RuntimeError):
    """Stage 0 produced trades but did not reproduce the frozen control."""


def _metrics(values: np.ndarray) -> dict[str, float | int]:
    wins = values[values > 0]; losses = values[values < 0]
    return {
        "trades": int(len(values)), "net_expectancy": float(values.mean()),
        "gross_expectancy": float(values.mean() + .0015),
        "profit_factor": float(wins.sum() / abs(losses.sum())),
        "win_rate": float((values > 0).mean()),
    }


def run_stage_zero(config: ReplayConfig) -> dict[str, Any]:
    inputs = config.payload["inputs"]
    for item in inputs.values():
        path = Path(item["path"])
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"COMPATIBILITY_REPLAY_BLOCKED: input hash mismatch: {path}")
    report = json.loads(Path(inputs["econ_report"]["path"]).read_text())
    base_costs = report["cost_scenarios"]["B_base"]
    generated = reproduce_historical_trades(
        Path(inputs["dataset"]["path"]),
        Path(inputs["rv2_pickle"]["path"]),
        Path(inputs["eqm_pickle"]["path"]),
        Path(inputs["series_manifest"]["path"]),
        base_costs,
    )
    frozen = pd.read_csv(inputs["trades"]["path"])
    frozen = frozen[(frozen["strategy"] == "eqm_plus_trrm") & (frozen["scenario"] == "B_base")].copy()
    frozen["ts"] = pd.to_datetime(frozen["ts"])
    keys = ["fold", "symbol", "ts"]
    common = generated.merge(frozen, on=keys, how="inner", suffixes=("_generated", "_frozen"))
    overlap = len(common) / max(1, len(frozen))
    frozen_keys = frozen[keys].astype(str).agg("|".join, axis=1)
    generated_keys = generated[keys].astype(str).agg("|".join, axis=1)
    missing = frozen.loc[~frozen_keys.isin(set(generated_keys)), keys]
    additional = generated.loc[~generated_keys.isin(set(frozen_keys)), keys]
    common["net_error"] = (common["net_generated"] - common["net_frozen"]).abs()
    maximum_net_error = float(common["net_error"].max()) if len(common) else float("inf")
    reference = report["strategies"]["eqm_plus_trrm"]["B_base"]
    metrics = _metrics(generated["net"].to_numpy(dtype=float))
    metrics["gross_expectancy"] = float(generated["gross"].mean())
    fold_values = tuple(float(generated[generated["fold"] == f"fold_{fold}"]["net"].mean()) for fold in range(1, 5))
    passed = (
        overlap >= .99 and metrics["trades"] == int(reference["trades"])
        and maximum_net_error <= 1e-9
        and abs(float(metrics["profit_factor"]) - float(reference["profit_factor"])) <= 1e-12
        and abs(float(metrics["net_expectancy"]) - float(reference["expectancy"])) <= 1e-12
        and all(abs(a - b) <= .01 for a, b in zip(fold_values, report["strategies"]["eqm_plus_trrm"]["per_fold_expectancy_base"]))
    )
    payload = {
        "schema_version": "aegis-compat-replay-stage-v1", "stage": "STAGE_0",
        "changed_axis": "NONE_HISTORICAL_REPRODUCTION_CONTROL", "passed": passed,
        "trade_overlap": overlap, "maximum_common_trade_net_error": maximum_net_error,
        **metrics, "per_fold_expectancy": fold_values,
        "selection_count": len(generated), "symbol_concentration": float(frozen["symbol"].value_counts(normalize=True).max()),
        "score_distribution": {
            "rv2_tail": {str(q): float(generated["rv2_tail"].quantile(q)) for q in (0.0, 0.5, 0.9, 1.0)},
            "eqm_score": {str(q): float(generated["eqm_score"].quantile(q)) for q in (0.0, 0.5, 0.9, 1.0)},
        },
        "input_hashes": {key: value["sha256"] for key, value in inputs.items()},
        "output_hash": digest({"generated_keys": generated[keys].astype(str).to_dict("records")}),
        "row_counts": {
            "generated": len(generated), "frozen": len(frozen), "common": len(common),
            "missing": len(missing), "additional": len(additional),
        },
        "first_divergence": None if passed else {
            "missing": missing.sort_values("ts").head(1).astype(str).to_dict("records"),
            "additional": additional.sort_values("ts").head(1).astype(str).to_dict("records"),
            "net": common.sort_values(["net_error", "ts"], ascending=[False, True]).head(1)[
                keys + ["entry", "exit", "gross", "cost_generated", "net_generated", "net_frozen", "net_error"]
            ].astype(str).to_dict("records"),
        },
        "safety_flags": {"dev_only": True, "lockbox": False, "semi_blind": False, "candidate": False},
    }
    output = Path(config.payload["output_root"]); atomic_write(output / "stage_0.json", payload)
    atomic_write(output / "manifest.json", {"replay_id": config.payload["replay_id"], "config_hash": sha256_file(config.path), "stage_hashes": {"STAGE_0": digest(payload)}})
    if not passed:
        raise HistoricalReproductionMismatch("HISTORICAL_REPRODUCTION_MISMATCH: Stage 0 tolerance failed")
    return payload
