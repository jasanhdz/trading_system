#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.edge.common import save_model_bundle  # noqa: E402
from aegis_alpha.turbo.recent_dataset import build_recent_dataset  # noqa: E402
from aegis_alpha.turbo.snapshot_utils import normalize_turbo_symbol, turbo_symbol_model_dir  # noqa: E402

SCHEMA_VERSION = "aegis_short_phase_o_prod_ready_manifest_v1"
GLOBAL_SCHEMA_VERSION = "aegis_short_phase_o_prod_ready_global_v1"
POINTER_SCHEMA_VERSION = "aegis_short_phase_o_prod_ready_pointer_v1"
REPORT_SCHEMA_VERSION = "aegis_short_shadow_artifacts_o_report_v1"
SIDE = "SHORT"
ENTRY_SYMBOLS = ["LTCUSDT", "AVAXUSDT", "ETHUSDT", "SUIUSDT", "ADAUSDT", "DOGEUSDT", "BTCUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT"]
AVOID_ONLY_SYMBOLS = ["LINKUSDT"]
ALL_SYMBOLS = ENTRY_SYMBOLS + AVOID_ONLY_SYMBOLS
ALLOWED_RUNTIME_USE = ["metadata_logging", "telegram_reporting", "offline_comparison", "yaml_shadow_runtime"]
FORBIDDEN_RUNTIME_USE = ["order_decision_without_yaml_live", "gate_override", "size_override", "leverage_override", "close_position", "open_position"]
CONTRACT = ["prod_ready_artifacts", "activation_controlled_by_bot_yaml", "no_pm2_restart_by_generator", "no_orders_by_generator"]

BASE_FEATURES = ["local_trend_down_score", "local_momentum_down_score", "local_chop_score", "btc_eth_long_contradiction", "btc_eth_short_agreement", "short_room_to_fall_12", "short_overhead_risk_12", "distance_ema25", "distance_ema99", "ema25_slope", "ema99_slope", "trend_efficiency_12", "trend_efficiency_24", "close_location_12", "upper_wick_ratio", "volume_ratio_12", "realized_vol_12", "realized_vol_24"]
REPAIR_FEATURES = BASE_FEATURES + ["short_failed_breakdown_risk_12", "short_failed_breakdown_risk_24", "short_lower_wick_sweep_risk"]
MOMENTUM_FEATURES = BASE_FEATURES + ["short_breakdown_strength_12", "short_breakdown_strength_24", "range_expansion_12", "atr_ratio_14"]
LINK_FEATURES = ["local_chop_score", "short_reclaim_range_risk", "short_failed_breakdown_risk_12", "short_lower_wick_sweep_risk", "btc_eth_long_contradiction", "short_adverse_rebound_risk", "upper_wick_ratio", "micro_danger_probability", "micro_quality_pred", "micro_hit_probability"]

SYMBOL_SPECS: Dict[str, Dict[str, Any]] = {
    "LTCUSDT": dict(shadow_type="entry_model", source_phase="phase_i_lockbox_confirmed", feature_set="operable_v3", feature_mode="selected_family", alpha_family="short_v3_confirmed", lookback_days=14, horizon_candles=12, decision_mode="metadata_shadow_entry", caution_level="normal", feature_names=BASE_FEATURES),
    "AVAXUSDT": dict(shadow_type="entry_model", source_phase="phase_i_lockbox_confirmed", feature_set="operable_v3", feature_mode="selected_family", alpha_family="short_v3_confirmed", lookback_days=14, horizon_candles=12, decision_mode="metadata_shadow_entry", caution_level="normal", feature_names=BASE_FEATURES),
    "ETHUSDT": dict(shadow_type="entry_model", source_phase="phase_i_lockbox_confirmed", feature_set="operable_v3", feature_mode="selected_family", alpha_family="short_v3_confirmed", lookback_days=30, horizon_candles=24, decision_mode="metadata_shadow_entry", caution_level="cautious", feature_names=BASE_FEATURES),
    "SUIUSDT": dict(shadow_type="entry_model", source_phase="phase_i_lockbox_confirmed", feature_set="combined_v3", feature_mode="selected_family", alpha_family="short_v3_confirmed", lookback_days=7, horizon_candles=12, decision_mode="metadata_shadow_entry", caution_level="normal", feature_names=BASE_FEATURES),
    "ADAUSDT": dict(shadow_type="entry_model", source_phase="phase_j1_repair_lockbox_confirmed", feature_set="operable_v2", feature_mode="selected_family", alpha_family="repaired_short_v3", lookback_days=30, horizon_candles=24, repair_mode="hit8_primary", decision_mode="metadata_shadow_entry", repaired_candidate=True, caution_level="normal", feature_names=REPAIR_FEATURES),
    "DOGEUSDT": dict(shadow_type="entry_model", source_phase="phase_j1_repair_lockbox_confirmed", feature_set="operable_v2", feature_mode="selected_family", alpha_family="repaired_short_v3", lookback_days=30, horizon_candles=12, repair_mode="hit8_primary", decision_mode="metadata_shadow_entry", repaired_candidate=True, caution_level="cautious", feature_names=REPAIR_FEATURES),
    "BTCUSDT": dict(shadow_type="entry_model", source_phase="phase_j1_repair_lockbox_confirmed", feature_set="combined_v3", feature_mode="selected_family", alpha_family="repaired_short_v3", lookback_days=30, horizon_candles=12, repair_mode="top_bucket_only", decision_mode="metadata_shadow_entry", repaired_candidate=True, caution_level="very_cautious", warning="fragile_hit8_lift", feature_names=REPAIR_FEATURES),
    "BNBUSDT": dict(shadow_type="entry_model", source_phase="phase_l3_alpha_confirmed", feature_set="combined_v3", feature_mode="selected_family", alpha_family="momentum_burst_lower_target", lookback_days=30, target_name="hit5_before_minus3", horizon_candles=12, decision_mode="hit_primary", caution_level="normal", feature_names=MOMENTUM_FEATURES),
    "XRPUSDT": dict(shadow_type="entry_model", source_phase="phase_l3_alpha_confirmed", feature_set="combined_v3", feature_mode="selected_family", alpha_family="momentum_burst_lower_target", lookback_days=30, target_name="hit5_before_minus3", horizon_candles=24, decision_mode="hit_primary", caution_level="cautious", feature_names=MOMENTUM_FEATURES),
    "SOLUSDT": dict(shadow_type="entry_model", source_phase="phase_m_final_repair_confirmed", feature_set="combined_v3", feature_mode="selected_family", alpha_family="momentum_burst_lower_target", lookback_days=30, target_name="hit5_before_minus3", horizon_candles=12, decision_mode="quality_primary", final_repair_candidate=True, caution_level="cautious", feature_names=MOMENTUM_FEATURES),
    "LINKUSDT": dict(shadow_type="avoid_only_filter", source_phase="phase_n1_micro_roe_avoid_only", feature_set="combined_v3", feature_mode="selected_family", alpha_family="avoid_only_bad_short_filter", lookback_days=30, leverage=20, target_roe=0.08, stop_roe=0.05, horizon_candles=12, decision_mode="avoid_by_micro_danger", link_micro_roe_short=False, link_avoid_only=True, affects_entry_decision=False, affects_sizing=False, affects_gating=False, caution_level="avoid_only", warning="not_entry_model", feature_names=LINK_FEATURES),
}

@dataclass(frozen=True)
class GenerationResult:
    artifact_stamp: str
    metadata_only: bool
    dry_run: bool
    base_dir: str
    global_manifest_path: str
    pointer_manifest_path: str
    report_paths: Dict[str, str]
    symbol_manifests: list[dict[str, Any]]
    active_manifest_paths: list[str]
    model_files: list[str]
    errors: list[str]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "log", "-1", "--oneline"], cwd=repo_root(), text=True).strip()
    except Exception:
        return "unknown"


def assert_live_compatible_path(path: os.PathLike[str] | str) -> None:
    text = str(Path(path)).replace("\\", "/")
    if "active_manifest.json" in text:
        return
    if "/models/turbo/" not in text and "aegis_alpha/models/turbo/" not in text:
        raise ValueError(f"Phase O prod-ready artifact must live under models/turbo: {path}")
    if "/active/phase_o_" not in text and not text.endswith("phase_o_shadow_manifest.json"):
        raise ValueError(f"Phase O model artifacts must stay under active/phase_o_<stamp>: {path}")


def feature_schema_hash(feature_names: Sequence[str], spec: Mapping[str, Any]) -> str:
    payload = {"feature_names": list(feature_names), "feature_set": spec.get("feature_set"), "feature_mode": spec.get("feature_mode"), "alpha_family": spec.get("alpha_family")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def select_symbols(symbols_arg: str, only_symbol: Optional[str]) -> list[str]:
    if only_symbol:
        symbols = [only_symbol.upper()]
    elif symbols_arg.upper() == "ALL":
        symbols = list(ALL_SYMBOLS)
    else:
        symbols = [part.strip().upper() for part in symbols_arg.split(",") if part.strip()]
    unknown = sorted(set(symbols) - set(ALL_SYMBOLS))
    if unknown:
        raise SystemExit(f"Unknown Phase O symbol(s): {', '.join(unknown)}")
    return symbols


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def train_short_edge_bundle(symbol: str, spec: Mapping[str, Any], model_path: Path, fast: bool) -> dict[str, Any]:
    assert_live_compatible_path(model_path)
    lookback_days = int(spec["lookback_days"])
    built = build_recent_dataset(symbol, lookback_days, save=True)
    dataset = built["dataset"]
    x = np.asarray(dataset["X"], dtype=np.float32)
    y = np.asarray(dataset["short_net_return_12"], dtype=np.float32)
    split = max(1, int(len(x) * 0.75))
    if split >= len(x):
        split = len(x) - 1
    x_train, x_val = x[:split], x[split:]
    y_train, y_val = y[:split], y[split:]
    estimator = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=60 if fast else 120,
        learning_rate=0.055,
        l2_regularization=0.08,
        max_leaf_nodes=15,
        early_stopping=True,
        random_state=8100 + lookback_days,
    )
    estimator.fit(x_train, y_train)
    val_pred = estimator.predict(x_val) if len(x_val) else np.array([], dtype=np.float32)
    validation_score = float(-np.mean(np.abs(y_val - val_pred))) if len(x_val) else None
    bundle = {
        "metadata": {
            "schema_version": "aegis_turbo_recent_edge_model_v1",
            "created_at": utc_stamp(),
            "phase_o_prod_ready": True,
            "yaml_shadow_expected": True,
            "symbol": symbol,
            "side": "short",
            "lookback_days": lookback_days,
            "horizon_candles": int(spec["horizon_candles"]),
            "target_key": "short_net_return_12",
            "alpha_family": spec["alpha_family"],
            "source_phase": spec["source_phase"],
            "model_kind": "regressor",
            "sample_count": int(len(x)),
            "train_samples": int(len(x_train)),
            "validation_samples": int(len(x_val)),
            "validation_score": validation_score,
            "sklearn_version": sklearn_version,
        },
        "feature_names": dataset["feature_names"].tolist() if hasattr(dataset["feature_names"], "tolist") else list(dataset["feature_names"]),
        "estimator": estimator,
    }
    save_model_bundle(model_path, bundle)
    return dict(bundle["metadata"], model_path=str(model_path), feature_count=len(bundle["feature_names"]))


def train_link_avoid_bundles(symbol: str, spec: Mapping[str, Any], artifact_dir: Path, fast: bool) -> tuple[list[str], dict[str, Any]]:
    built = build_recent_dataset(symbol, int(spec["lookback_days"]), save=True)
    dataset = built["dataset"]
    x = np.asarray(dataset["X"], dtype=np.float32)
    y_return = np.asarray(dataset["short_net_return_12"], dtype=np.float32)
    stop_threshold = -float(spec["stop_roe"]) / float(spec["leverage"])
    hit_threshold = float(spec["target_roe"]) / float(spec["leverage"])
    hit_y = (y_return >= hit_threshold).astype(int)
    danger_y = (y_return <= stop_threshold).astype(int)
    quality_y = np.clip(y_return * float(spec["leverage"]), -1.0, 1.0)
    split = max(1, int(len(x) * 0.75))
    if split >= len(x):
        split = len(x) - 1
    x_train, x_val = x[:split], x[split:]
    outputs: list[str] = []
    summaries: dict[str, Any] = {"sample_count": int(len(x)), "train_samples": int(len(x_train)), "validation_samples": int(len(x) - split), "feature_count": int(x.shape[1]) if x.ndim == 2 else 0}
    specs = [
        ("micro_hit_classifier.joblib", HistGradientBoostingClassifier(max_iter=60 if fast else 120, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=0.08, early_stopping=True, random_state=9101), hit_y),
        ("micro_quality_regressor.joblib", HistGradientBoostingRegressor(loss="absolute_error", max_iter=60 if fast else 120, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=0.08, early_stopping=True, random_state=9102), quality_y),
        ("micro_danger_classifier.joblib", HistGradientBoostingClassifier(max_iter=60 if fast else 120, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=0.08, early_stopping=True, random_state=9103), danger_y),
    ]
    for name, estimator, y in specs:
        path = artifact_dir / name
        assert_live_compatible_path(path)
        if len(np.unique(y[:split])) < 2 and "classifier" in name:
            summaries[name] = "single_class_skipped"
            continue
        estimator.fit(x_train, y[:split])
        bundle = {
            "metadata": {"schema_version": "aegis_link_micro_roe_avoid_model_v1", "created_at": utc_stamp(), "phase_o_prod_ready": True, "yaml_shadow_expected": True, "symbol": symbol, "side": "short", "avoid_only": True, "entry_enabled": False, "model_role": Path(name).stem, "source_phase": spec["source_phase"], "sklearn_version": sklearn_version},
            "feature_names": dataset["feature_names"].tolist() if hasattr(dataset["feature_names"], "tolist") else list(dataset["feature_names"]),
            "estimator": estimator,
        }
        save_model_bundle(path, bundle)
        outputs.append(str(path))
        summaries[name] = "trained"
    return outputs, summaries


def symbol_manifest(symbol: str, spec: Mapping[str, Any], artifact_stamp: str, artifact_dir: Path, metadata_only: bool, model_files: Sequence[str], train_summary: Mapping[str, Any]) -> dict[str, Any]:
    entry_enabled = spec["shadow_type"] == "entry_model"
    avoid_only = spec["shadow_type"] == "avoid_only_filter"
    feature_names = list(spec.get("feature_names", []))
    warnings = []
    if spec.get("warning"):
        warnings.append(spec["warning"])
    if metadata_only:
        warnings.append("metadata_only_no_model_files")
    else:
        warnings.append("prod_ready_artifacts_activation_controlled_by_yaml_shadow")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": now_iso(),
        "artifact_stamp": artifact_stamp,
        "symbol": symbol,
        "side": SIDE,
        "shadow_type": spec["shadow_type"],
        "source_phase": spec["source_phase"],
        "caution_level": spec["caution_level"],
        "enabled_for_shadow": True,
        "metadata_only": bool(metadata_only),
        "entry_enabled": entry_enabled,
        "avoid_only": avoid_only,
        "prod_ready": True,
        "yaml_shadow_expected": True,
        "affects_decision": bool(entry_enabled),
        "affects_gating": False,
        "affects_sizing": False,
        "affects_orders": False,
        "allowed_runtime_use": list(ALLOWED_RUNTIME_USE),
        "forbidden_runtime_use": list(FORBIDDEN_RUNTIME_USE),
        "feature_set": spec["feature_set"],
        "feature_mode": spec["feature_mode"],
        "alpha_family": spec["alpha_family"],
        "lookback_days": spec["lookback_days"],
        "horizon_candles": spec["horizon_candles"],
        "decision_mode": spec["decision_mode"],
        "model_files": list(model_files),
        "artifact_dir": str(artifact_dir),
        "feature_names": feature_names,
        "feature_schema_hash": feature_schema_hash(feature_names, spec),
        "train_samples": int(train_summary.get("train_samples", 0) or 0),
        "validation_samples": int(train_summary.get("validation_samples", 0) or 0),
        "test_samples": 0,
        "validation_summary": dict(train_summary),
        "source_reports": {spec["source_phase"].split("_")[1] if "_" in spec["source_phase"] else "phase": spec["source_phase"]},
        "warnings": warnings,
        "git_commit_at_generation": git_commit(),
        "research_only_generation": False,
        "not_active_promoted": False,
    }
    for key in ("target_name", "target_roe", "stop_roe", "leverage", "repair_mode", "repaired_candidate", "final_repair_candidate", "link_micro_roe_short", "link_avoid_only", "affects_entry_decision", "warning"):
        if key in spec:
            manifest[key] = spec[key]
    return manifest


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def update_active_manifest(symbol: str, spec: Mapping[str, Any], manifest: Mapping[str, Any], model_files: Sequence[str], stamp: str) -> Path:
    symbol_dir = turbo_symbol_model_dir(symbol)
    path = symbol_dir / "active_manifest.json"
    active = load_json(path)
    if not active:
        active = {"symbol": symbol, "model_paths": {}, "windows": []}
    active.setdefault("model_paths", {})
    active.setdefault("windows", [])
    if manifest["entry_enabled"] and model_files:
        key = f"short_{int(spec['lookback_days'])}d"
        active["model_paths"][key] = str(Path(model_files[0]).resolve())
        windows = set(int(x) for x in active.get("windows", []) if str(x).isdigit())
        windows.add(int(spec["lookback_days"]))
        active["windows"] = sorted(windows)
    active["phase_o_prod_ready"] = True
    active["yaml_shadow_expected"] = True
    active["phase_o_artifact_stamp"] = stamp
    active.setdefault("phase_o_symbols", {})[symbol] = {
        "entry_enabled": manifest["entry_enabled"],
        "avoid_only": manifest["avoid_only"],
        "shadow_type": manifest["shadow_type"],
        "caution_level": manifest["caution_level"],
        "symbol_manifest": str((Path(manifest["artifact_dir"]) / "symbol_shadow_manifest.json").resolve()),
    }
    if manifest["avoid_only"]:
        active["phase_o_avoid_only"] = True
        active["phase_o_avoid_artifacts"] = list(model_files)
    active["updated_at_phase_o"] = now_iso()
    write_json(path, active)
    return path


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_reports(out_dir: Path, stamp: str, global_manifest: Mapping[str, Any], pointer: Mapping[str, Any], manifests: Sequence[Mapping[str, Any]], metadata_only: bool, dry_run: bool, validation_status: str) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"aegis_short_shadow_artifacts_o_{stamp}"
    md_path = base.with_suffix(".md")
    json_path = base.with_suffix(".json")
    summary_csv = out_dir / f"aegis_short_shadow_artifacts_summary_o_{stamp}.csv"
    symbols_csv = out_dir / f"aegis_short_shadow_artifacts_symbols_o_{stamp}.csv"
    entry = [m["symbol"] for m in manifests if m.get("entry_enabled")]
    avoid = [m["symbol"] for m in manifests if m.get("avoid_only")]
    model_count = sum(len(m.get("model_files", [])) for m in manifests)
    payload = {"schema_version": REPORT_SCHEMA_VERSION, "created_at": now_iso(), "artifact_stamp": stamp, "dry_run": dry_run, "metadata_only": metadata_only, "prod_ready": True, "yaml_shadow_expected": True, "global_manifest_path": global_manifest.get("path"), "pointer_manifest_path": pointer.get("path"), "entry_symbols": entry, "avoid_only_symbols": avoid, "link_status": "avoid_only_filter", "model_file_count": model_count, "validation_status": validation_status, "symbol_manifests": list(manifests)}
    md = [f"# Phase O SHORT Prod-Ready Shadow Artifacts {stamp}", "", "## Safety", "- prod_ready: true", "- yaml_shadow_expected: true", "- pm2_restarted: false", "- orders_sent: false", "- env_changed: false", "", "## Paths", f"- Global manifest: {global_manifest.get('path')}", f"- Pointer manifest: {pointer.get('path')}", "", "## Entry Shadow Symbols", "- " + ", ".join(entry), "", "## Avoid-Only Symbols", "- " + ", ".join(avoid), "", "## LINK", "- LINKUSDT avoid_only_filter, entry_enabled=false, affects_orders=false", "", "## Generation", f"- metadata_only: {metadata_only}", f"- model_file_count: {model_count}", f"- validation_status: {validation_status}"]
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    write_json(json_path, payload)
    write_csv(summary_csv, [{"artifact_stamp": stamp, "metadata_only": metadata_only, "prod_ready": True, "symbol_count": len(manifests), "entry_shadow_count": len(entry), "avoid_only_count": len(avoid), "model_file_count": model_count, "validation_status": validation_status}], ["artifact_stamp", "metadata_only", "prod_ready", "symbol_count", "entry_shadow_count", "avoid_only_count", "model_file_count", "validation_status"])
    write_csv(symbols_csv, [{"symbol": m["symbol"], "shadow_type": m["shadow_type"], "entry_enabled": m["entry_enabled"], "avoid_only": m["avoid_only"], "lookback_days": m["lookback_days"], "horizon_candles": m["horizon_candles"], "alpha_family": m["alpha_family"], "model_file_count": len(m.get("model_files", [])), "manifest": str(Path(m["artifact_dir"]) / "symbol_shadow_manifest.json")} for m in manifests], ["symbol", "shadow_type", "entry_enabled", "avoid_only", "lookback_days", "horizon_candles", "alpha_family", "model_file_count", "manifest"])
    return {"md": str(md_path), "json": str(json_path), "summary_csv": str(summary_csv), "symbols_csv": str(symbols_csv)}


def generate(args: argparse.Namespace) -> GenerationResult:
    symbols = select_symbols(args.symbols, args.only_symbol)
    stamp = utc_stamp()
    base_dir = Path(args.shadow_model_dir) if args.shadow_model_dir else repo_root() / "aegis_alpha/models/turbo"
    out_dir = Path(args.out_dir)
    metadata_only = bool(args.metadata_only or args.skip_training)
    manifests: list[dict[str, Any]] = []
    active_paths: list[str] = []
    model_files: list[str] = []
    if args.dry_run:
        for symbol in symbols:
            spec = SYMBOL_SPECS[symbol]
            symbol_dir = turbo_symbol_model_dir(symbol) / "active" / f"phase_o_{stamp}"
            manifests.append(symbol_manifest(symbol, spec, stamp, symbol_dir, metadata_only, [], {"status": "dry_run"}))
        return GenerationResult(stamp, metadata_only, True, str(base_dir), "", "", {}, manifests, [], [], [])

    for symbol in symbols:
        symbol = normalize_turbo_symbol(symbol)
        spec = SYMBOL_SPECS[symbol]
        artifact_dir = turbo_symbol_model_dir(symbol) / "active" / f"phase_o_{stamp}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        files: list[str] = []
        summary: dict[str, Any] = {"status": "metadata_only" if metadata_only else "trained"}
        if not metadata_only:
            if spec["shadow_type"] == "entry_model":
                model_path = artifact_dir / f"turbo_short_edge_{int(spec['lookback_days'])}d_phase_o_{stamp}.joblib"
                summary = train_short_edge_bundle(symbol, spec, model_path, args.fast)
                files = [str(model_path.resolve())]
            else:
                files, summary = train_link_avoid_bundles(symbol, spec, artifact_dir, args.fast)
        manifest = symbol_manifest(symbol, spec, stamp, artifact_dir, metadata_only, files, summary)
        write_json(artifact_dir / "symbol_shadow_manifest.json", manifest)
        write_json(artifact_dir / "metadata.json", {"schema_version": "aegis_short_phase_o_symbol_metadata_v1", "created_at": now_iso(), "symbol": symbol, "prod_ready": True, "yaml_shadow_expected": True, "manifest": str((artifact_dir / "symbol_shadow_manifest.json").resolve())})
        if manifest["avoid_only"]:
            write_json(artifact_dir / "avoid_filter_metadata.json", {"schema_version": "aegis_short_phase_o_avoid_filter_v1", "created_at": now_iso(), "symbol": symbol, "entry_enabled": False, "avoid_only": True, "target_roe": spec.get("target_roe"), "stop_roe": spec.get("stop_roe"), "leverage": spec.get("leverage"), "affects_orders": False})
        active_path = update_active_manifest(symbol, spec, manifest, files, stamp)
        active_paths.append(str(active_path.resolve()))
        model_files.extend(files)
        manifests.append(manifest)

    entry = [m["symbol"] for m in manifests if m["entry_enabled"]]
    avoid = [m["symbol"] for m in manifests if m["avoid_only"]]
    global_manifest_path = base_dir / f"phase_o_global_short_manifest_{stamp}.json"
    pointer_manifest_path = base_dir / "phase_o_short_manifest.json"
    global_manifest = {"schema_version": GLOBAL_SCHEMA_VERSION, "path": str(global_manifest_path.resolve()), "created_at": now_iso(), "artifact_stamp": stamp, "symbol_count": len(manifests), "entry_shadow_count": len(entry), "avoid_only_count": len(avoid), "symbols": [m["symbol"] for m in manifests], "entry_symbols": entry, "avoid_only_symbols": avoid, "excluded_entry_symbols": ["LINKUSDT"], "prod_ready": True, "yaml_shadow_expected": True, "shadow_runtime_contract": CONTRACT, "artifact_paths": {m["symbol"]: str((Path(m["artifact_dir"]) / "symbol_shadow_manifest.json").resolve()) for m in manifests}, "active_manifest_paths": active_paths, "model_files": model_files}
    pointer = {"schema_version": POINTER_SCHEMA_VERSION, "path": str(pointer_manifest_path.resolve()), "created_at": now_iso(), "latest_artifact_stamp": stamp, "latest_manifest": str(global_manifest_path.resolve()), "enabled_for_shadow": True, "prod_ready": True, "yaml_shadow_expected": True, "entry_symbols": entry, "avoid_only_symbols": avoid, "active_manifest_touched": True}
    write_json(global_manifest_path, global_manifest)
    write_json(pointer_manifest_path, pointer)
    reports = write_reports(out_dir, stamp, global_manifest, pointer, manifests, metadata_only, False, "pending_validator")
    return GenerationResult(stamp, metadata_only, False, str(base_dir), str(global_manifest_path.resolve()), str(pointer_manifest_path.resolve()), reports, manifests, active_paths, model_files, [])


def payload(result: GenerationResult) -> dict[str, Any]:
    return {"artifact_stamp": result.artifact_stamp, "metadata_only": result.metadata_only, "dry_run": result.dry_run, "base_dir": result.base_dir, "global_manifest_path": result.global_manifest_path, "pointer_manifest_path": result.pointer_manifest_path, "report_paths": result.report_paths, "symbol_count": len(result.symbol_manifests), "entry_shadow_count": sum(1 for m in result.symbol_manifests if m.get("entry_enabled")), "avoid_only_count": sum(1 for m in result.symbol_manifests if m.get("avoid_only")), "entry_symbols": [m["symbol"] for m in result.symbol_manifests if m.get("entry_enabled")], "avoid_only_symbols": [m["symbol"] for m in result.symbol_manifests if m.get("avoid_only")], "active_manifest_paths": result.active_manifest_paths, "model_files": result.model_files, "errors": result.errors}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Phase O SHORT prod-ready artifacts controlled by YAML shadow mode.")
    parser.add_argument("--symbols", default="ALL")
    parser.add_argument("--only-symbol", default=None)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--out-dir", default="/home/jasan/Develop")
    parser.add_argument("--shadow-model-dir", default=str(repo_root() / "aegis_alpha/models/turbo"))
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.validate_only:
        raise SystemExit("Use aegis_alpha/tools/validate_short_shadow_artifacts_o.py")
    result = generate(args)
    print(json.dumps(payload(result), indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
