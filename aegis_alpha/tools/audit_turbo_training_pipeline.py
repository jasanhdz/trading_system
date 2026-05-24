#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aegis_alpha.edge.common import WINDOW_FEATURE_NAMES, edge_feature_names
from aegis_alpha.features.feature_builder import FEATURE_COLUMNS
from aegis_alpha.turbo.config import DEFAULT_TURBO_CONFIG
from aegis_alpha.turbo.recent_dataset import (
    MAX_TARGET_HORIZON,
    OPERABLE_TARGET_NAMES,
    OPERABLE_TARGET_SCHEMA_VERSION,
    TRADE_QUALITY_FORMULA,
    build_recent_dataset,
)
from aegis_alpha.turbo.snapshot_utils import (
    load_turbo_snapshot_status,
    normalize_turbo_symbol,
    turbo_snapshot_path,
    turbo_symbol_model_dir,
)
from aegis_alpha.turbo.train_recent_edge import MIN_TRAIN_SAMPLES, VALIDATION_PCT


EXPECTED_SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "SUIUSDT",
    "LTCUSDT",
)

SOURCE_FILES = (
    "aegis_alpha/turbo/train_recent_edge.py",
    "aegis_alpha/turbo/recent_dataset.py",
    "aegis_alpha/turbo/turbo_signal.py",
    "aegis_alpha/inference/server.py",
    "aegis_alpha/signals/common.py",
    "aegis_alpha/edge/common.py",
    "aegis_alpha/features/feature_builder.py",
    "aegis_alpha/tools/run_turbo_scheduled_retrain.py",
)

FEATURE_CSV_COLUMNS = (
    "feature_name",
    "base_feature",
    "window_lookback",
    "source",
    "live_availability",
    "possible_leakage_risk",
    "symbols_observed",
    "nan_rate",
    "inf_rate",
    "zero_rate",
    "constant_rate",
    "duplicate_correlation_risk",
    "comments",
)

LEAKAGE_CSV_COLUMNS = (
    "area",
    "risk",
    "severity",
    "evidence",
    "mitigation",
)


@dataclass
class FeatureAuditRow:
    feature_name: str
    base_feature: str
    window_lookback: str
    source: str
    live_availability: str
    possible_leakage_risk: str
    symbols_observed: int = 0
    nan_rate: float | None = None
    inf_rate: float | None = None
    zero_rate: float | None = None
    constant_rate: float | None = None
    duplicate_correlation_risk: str = "unknown"
    comments: str = ""


@dataclass
class LeakageRiskRow:
    area: str
    risk: str
    severity: str
    evidence: str
    mitigation: str


@dataclass
class SymbolDatasetAudit:
    symbol: str
    dataset_loaded: bool
    sample_count: int | None = None
    feature_count: int | None = None
    feature_timestamp: str | None = None
    snapshot_paths: dict[str, Any] = field(default_factory=dict)
    model_paths: dict[str, str] = field(default_factory=dict)
    manifest_path: str | None = None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_stamp() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%SZ")


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def parse_symbols(raw: str | None) -> list[str]:
    if not raw:
        return list(EXPECTED_SYMBOLS)
    return list(dict.fromkeys(normalize_turbo_symbol(item) for item in raw.split(",") if item.strip()))


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def source_code(path: str) -> str:
    try:
        return (repo_root() / path).read_text(encoding="utf-8")
    except OSError:
        return ""


def assigned_names_from_file(path: str) -> list[str]:
    code = source_code(path)
    if not code:
        return []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            names.append(node.name)
        elif isinstance(node, ast.ClassDef):
            names.append(node.name)
    return sorted(set(names))


def infer_feature_source(base_feature: str) -> str:
    if base_feature in {"log_ret", "high_norm", "low_norm", "trend_efficiency", "vol_regime"}:
        return "OHLC price path in build_feature_frame"
    if base_feature.startswith("ema_") or base_feature in {"rsi_norm", "adx_norm"}:
        return "technical indicator from historical close/high/low"
    if base_feature.startswith("cvd"):
        return "buy_volume/volume derived CVD feature"
    if base_feature.startswith("vol"):
        return "volume rolling statistic"
    if base_feature == "candle_progress":
        return "timestamp modulo candle interval"
    return "feature_builder.FEATURE_COLUMNS"


def infer_window(prefix: str) -> str:
    mapping = {
        "last": "current row at signal step",
        "mean_6": "mean over last 6 feature rows",
        "mean_12": "mean over last 12 feature rows",
        "mean_64": "mean over configured model window",
        "std_12": "std over last 12 feature rows",
        "std_64": "std over configured model window",
        "delta_6": "current row minus row 6 steps back",
        "delta_12": "current row minus row 12 steps back",
    }
    return mapping.get(prefix, "unknown")


def split_edge_feature_name(name: str) -> tuple[str, str]:
    for prefix in sorted(WINDOW_FEATURE_NAMES, key=len, reverse=True):
        needle = f"{prefix}_"
        if name.startswith(needle):
            return prefix, name[len(needle):]
    return "unknown", name


def feature_stats_matrix(x: np.ndarray, feature_names: list[str]) -> dict[str, dict[str, float | str]]:
    if x.ndim != 2:
        raise ValueError(f"expected 2d feature matrix, got shape={x.shape}")
    stats: dict[str, dict[str, float | str]] = {}
    finite = np.isfinite(x)
    for idx, name in enumerate(feature_names):
        col = np.asarray(x[:, idx], dtype=np.float64)
        col_finite = finite[:, idx]
        finite_values = col[col_finite]
        nan_rate = float(np.isnan(col).mean()) if len(col) else 0.0
        inf_rate = float(np.isinf(col).mean()) if len(col) else 0.0
        zero_rate = float((col_finite & (col == 0.0)).mean()) if len(col) else 0.0
        constant_rate = 1.0 if len(finite_values) > 1 and float(np.nanstd(finite_values)) <= 1e-12 else 0.0
        stats[name] = {
            "nan_rate": nan_rate,
            "inf_rate": inf_rate,
            "zero_rate": zero_rate,
            "constant_rate": constant_rate,
        }
    duplicate_groups = duplicate_or_correlated_features(x, feature_names)
    for name, reason in duplicate_groups.items():
        stats.setdefault(name, {})["duplicate_correlation_risk"] = reason
    return stats


def duplicate_or_correlated_features(x: np.ndarray, feature_names: list[str], corr_threshold: float = 0.999) -> dict[str, str]:
    risks: dict[str, str] = {}
    if x.ndim != 2 or x.shape[0] < 3 or x.shape[1] < 2:
        return risks
    finite_x = np.nan_to_num(x.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    seen: dict[bytes, str] = {}
    for idx, name in enumerate(feature_names):
        rounded = np.round(finite_x[:, idx], 8).tobytes()
        if rounded in seen:
            risks[name] = f"duplicate_of:{seen[rounded]}"
        else:
            seen[rounded] = name
    std = finite_x.std(axis=0)
    non_constant = np.where(std > 1e-12)[0]
    if len(non_constant) <= 1:
        return risks
    limited = non_constant[: min(len(non_constant), 220)]
    corr = np.corrcoef(finite_x[:, limited], rowvar=False)
    for i, left_idx in enumerate(limited):
        for j in range(i + 1, len(limited)):
            right_idx = limited[j]
            value = corr[i, j]
            if math.isfinite(float(value)) and abs(float(value)) >= corr_threshold:
                risks.setdefault(feature_names[right_idx], f"correlated_{float(value):.4f}_with:{feature_names[left_idx]}")
    return risks


def merge_feature_stats(rows: list[dict[str, dict[str, float | str]]]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for stat_map in rows:
        for name, values in stat_map.items():
            current = merged.setdefault(
                name,
                {
                    "symbols_observed": 0,
                    "nan_rate": [],
                    "inf_rate": [],
                    "zero_rate": [],
                    "constant_rate": [],
                    "duplicate_correlation_risk": [],
                },
            )
            current["symbols_observed"] += 1
            for key in ("nan_rate", "inf_rate", "zero_rate", "constant_rate"):
                numeric = finite_float(values.get(key))
                if numeric is not None:
                    current[key].append(numeric)
            risk = values.get("duplicate_correlation_risk")
            if risk:
                current["duplicate_correlation_risk"].append(str(risk))
    for values in merged.values():
        for key in ("nan_rate", "inf_rate", "zero_rate", "constant_rate"):
            series = values.get(key) or []
            values[key] = round(float(sum(series) / len(series)), 6) if series else None
        risks = values.get("duplicate_correlation_risk") or []
        values["duplicate_correlation_risk"] = ";".join(sorted(set(risks))[:3]) if risks else "none_detected"
    return merged


def build_feature_rows(feature_stats: dict[str, dict[str, Any]] | None = None) -> list[FeatureAuditRow]:
    feature_stats = feature_stats or {}
    rows: list[FeatureAuditRow] = []
    for name in edge_feature_names(list(FEATURE_COLUMNS)):
        prefix, base = split_edge_feature_name(name)
        stats = feature_stats.get(name, {})
        leakage = "low"
        comments = "built from current/past rows by build_edge_feature_matrix"
        if base == "candle_progress":
            comments = "time-of-candle feature; verify timestamp units stay consistent in live snapshots"
        rows.append(
            FeatureAuditRow(
                feature_name=name,
                base_feature=base,
                window_lookback=infer_window(prefix),
                source=infer_feature_source(base),
                live_availability="same snapshot live_X generated by build_recent_dataset",
                possible_leakage_risk=leakage,
                symbols_observed=int(stats.get("symbols_observed") or 0),
                nan_rate=finite_float(stats.get("nan_rate")),
                inf_rate=finite_float(stats.get("inf_rate")),
                zero_rate=finite_float(stats.get("zero_rate")),
                constant_rate=finite_float(stats.get("constant_rate")),
                duplicate_correlation_risk=str(stats.get("duplicate_correlation_risk") or "not_measured"),
                comments=comments,
            )
        )
    return rows


def build_leakage_rows() -> list[LeakageRiskRow]:
    return [
        LeakageRiskRow(
            area="feature_alignment",
            risk="past_only_feature_window",
            severity="LOW",
            evidence="build_edge_feature_matrix uses features[step] plus windows ending before/current step; targets use idx+1..idx+horizon.",
            mitigation="Keep tests around step alignment when adding V2 targets.",
        ),
        LeakageRiskRow(
            area="target_alignment",
            risk="return_target_not_trade_path",
            severity="HIGH",
            evidence="recent_dataset trains long_net_return_12/short_net_return_12 from future close return only; MFE/MAE are recorded but not trained.",
            mitigation="Fase B: add hit-before-stop, MAE danger, MFE/MAE and trade_quality targets using high/low path.",
        ),
        LeakageRiskRow(
            area="validation",
            risk="single_chronological_split_no_walk_forward",
            severity="MEDIUM",
            evidence=f"train_recent_edge.py uses fixed VALIDATION_PCT={VALIDATION_PCT} and reports MAE/RMSE, not trading expectancy.",
            mitigation="Fase D: temporal walk-forward with trading metrics and promotion gates.",
        ),
        LeakageRiskRow(
            area="score_calibration",
            risk="regressor_magnitude_used_as_probability_like_score",
            severity="HIGH",
            evidence="runtime maps max predicted return / 0.003 plus vote agreement into turbo_score; no calibration curve or bucket gate.",
            mitigation="Fase D/E: score bucket calibration and V2 shadow probabilities.",
        ),
        LeakageRiskRow(
            area="runtime_artifacts",
            risk="ETH_legacy_global_snapshot_path",
            severity="MEDIUM",
            evidence="turbo_snapshot_path returns processed/turbo_recent_*d.npz for DEFAULT_TURBO_CONFIG.symbol when legacy path exists.",
            mitigation="Document/migrate ETH snapshots to symbol-specific path after parity checks.",
        ),
        LeakageRiskRow(
            area="live_parity",
            risk="feature_schema_not_hash_validated_in_v1_runtime",
            severity="MEDIUM",
            evidence="V1 joblib bundles contain feature_names, but turbo_signal only loads estimator and predicts live_X shape.",
            mitigation="Fase C/E: manifest feature_schema_hash and runtime validation before shadow prediction.",
        ),
    ]


def active_model_paths(symbol: str) -> dict[str, str]:
    symbol_dir = turbo_symbol_model_dir(symbol)
    manifest_path = symbol_dir / "active_manifest.json"
    paths: dict[str, str] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            model_paths = manifest.get("model_paths") if isinstance(manifest, dict) else None
            if isinstance(model_paths, dict):
                return {str(key): str(value) for key, value in model_paths.items()}
        except Exception:
            pass
    for lookback in DEFAULT_TURBO_CONFIG.lookback_days:
        for side in ("long", "short"):
            paths[f"{side}_{int(lookback)}d"] = str(symbol_dir / "active" / f"turbo_{side}_edge_{int(lookback)}d_v010.joblib")
    return paths


def audit_symbol_dataset(symbol: str, lookback_days: int, load_dataset: bool) -> tuple[SymbolDatasetAudit, dict[str, dict[str, float | str]] | None]:
    normalized = normalize_turbo_symbol(symbol)
    manifest_path = turbo_symbol_model_dir(normalized) / "active_manifest.json"
    audit = SymbolDatasetAudit(
        symbol=normalized,
        dataset_loaded=False,
        manifest_path=str(manifest_path),
        model_paths=active_model_paths(normalized),
    )
    for lookback in DEFAULT_TURBO_CONFIG.lookback_days:
        path = turbo_snapshot_path(int(lookback), normalized)
        status = load_turbo_snapshot_status(path, include_sample_count=True)
        audit.snapshot_paths[f"{int(lookback)}d"] = status
    if not load_dataset:
        audit.warnings.append("dataset_stats_skipped")
        return audit, None
    try:
        payload = build_recent_dataset(normalized, lookback_days=lookback_days, save=False)
        dataset = payload.get("dataset", payload)
        x = np.asarray(dataset["X"], dtype=np.float32)
        feature_names = [str(item) for item in dataset.get("feature_names", [])]
        audit.dataset_loaded = True
        audit.sample_count = int(len(x))
        audit.feature_count = int(x.shape[1]) if x.ndim == 2 else None
        feature_timestamp = dataset.get("feature_timestamp")
        audit.feature_timestamp = str(feature_timestamp) if feature_timestamp is not None else None
        if audit.sample_count is not None and audit.sample_count < MIN_TRAIN_SAMPLES:
            audit.warnings.append("sample_count_below_min_train_samples")
        if len(feature_names) != (audit.feature_count or 0):
            audit.warnings.append("feature_name_count_mismatch")
        return audit, feature_stats_matrix(x, feature_names)
    except Exception as exc:
        audit.error = repr(exc)
        audit.warnings.append("dataset_load_failed")
        return audit, None


def build_pipeline_map() -> list[dict[str, Any]]:
    return [
        {
            "stage": "ohlcv_data",
            "implementation": "aegis_alpha.signals.common.load_signal_market -> DatabaseManager.get_ohlcv_data",
            "output": "OHLCV dataframe for symbol/timeframe",
            "notes": "Uses DEFAULT_TURBO_CONFIG.config_path and optional symbol_override.",
        },
        {
            "stage": "base_features",
            "implementation": "aegis_alpha.features.feature_builder.build_feature_frame",
            "output": f"{len(FEATURE_COLUMNS)} base columns after warmup trimming",
            "notes": "Rolling/EMA indicators, CVD, ADX, volatility regime.",
        },
        {
            "stage": "edge_feature_matrix",
            "implementation": "aegis_alpha.edge.common.build_edge_feature_matrix",
            "output": f"{len(edge_feature_names(list(FEATURE_COLUMNS)))} edge features",
            "notes": "last/mean/std/delta windows are nan_to_num clipped to [-10,10].",
        },
        {
            "stage": "dataset_creation",
            "implementation": "aegis_alpha.turbo.recent_dataset.build_recent_dataset",
            "output": "X, live_X, timestamps, future_return, MFE/MAE, long/short net return targets",
            "notes": f"Valid mask excludes last MAX_TARGET_HORIZON={MAX_TARGET_HORIZON} rows.",
        },
        {
            "stage": "current_targets",
            "implementation": "recent_dataset._target_stats",
            "output": "long_net_return_12 and short_net_return_12 for training",
            "notes": "Future close return minus fee; not hit-before-stop and not bot exit logic.",
        },
        {
            "stage": "training",
            "implementation": "aegis_alpha.turbo.train_recent_edge._train_one",
            "output": "HistGradientBoostingRegressor per side/lookback",
            "notes": "Absolute-error regressor trained separately for long and short.",
        },
        {
            "stage": "validation",
            "implementation": "train_recent_edge._train_one",
            "output": "validation_score=-MAE, validation_rmse, avg_predicted_return",
            "notes": "Single chronological split, no walk-forward or trading expectancy gate.",
        },
        {
            "stage": "save_candidate",
            "implementation": "save_model_bundle + train report",
            "output": "joblib model bundles and turbo_train_report_*.json",
            "notes": "Bundles include metadata and feature_names.",
        },
        {
            "stage": "scheduled_retrain_promotion",
            "implementation": "aegis_alpha.tools.run_turbo_scheduled_retrain",
            "output": "active/ models plus active_manifest.json after validation",
            "notes": "Validation checks model existence, sample counts, finite snapshots, and live prediction.",
        },
        {
            "stage": "runtime_load",
            "implementation": "aegis_alpha.turbo.turbo_signal._load_model_set/_current_feature",
            "output": "active estimators plus latest snapshot live_X",
            "notes": "Runtime reads active_manifest if present and newest fresh snapshot.",
        },
        {
            "stage": "runtime_signal",
            "implementation": "aegis_alpha.turbo.turbo_signal.evaluate_turbo_shadow",
            "output": "action, votes, turbo_score, recent_scores, gated decision",
            "notes": "Votes compare long/short regressors by lookback; score is agreement plus magnitude, not calibrated probability.",
        },
        {
            "stage": "inference_endpoint",
            "implementation": "aegis_alpha.inference.server /ml-v2/predict",
            "output": "LegacyMlV2Response with aegis.turbo metadata",
            "notes": "TS bot consumes this service and applies additional guards/orchestrator logic.",
        },
    ]


def build_train_live_parity(symbols: list[str], audits: list[SymbolDatasetAudit]) -> dict[str, Any]:
    expected_feature_names = edge_feature_names(list(FEATURE_COLUMNS))
    return {
        "feature_names_source_train": "dataset['feature_names'] from edge_feature_names(FEATURE_COLUMNS)",
        "feature_names_source_runtime": "live_X inside selected turbo_recent_*d.npz generated by the same build_recent_dataset",
        "expected_feature_count": len(expected_feature_names),
        "runtime_model_loading": "active_manifest model_paths -> active/turbo_<side>_edge_<lookback>d_v010.joblib -> legacy ETH fallback",
        "runtime_snapshot_loading": "turbo_snapshot_path + freshest feature_timestamp; ETH default may use legacy global snapshot",
        "nan_inf_handling": "build_edge_feature_matrix applies nan_to_num and clips values to [-10, 10]",
        "schema_validation_gap": "V1 runtime does not enforce feature_names/schema hash before estimator.predict",
        "symbols": {
            audit.symbol: {
                "dataset_loaded": audit.dataset_loaded,
                "feature_count": audit.feature_count,
                "feature_count_matches_expected": audit.feature_count == len(expected_feature_names) if audit.feature_count is not None else None,
                "manifest_path": audit.manifest_path,
                "model_paths": audit.model_paths,
                "snapshots": audit.snapshot_paths,
                "warnings": audit.warnings,
                "error": audit.error,
            }
            for audit in audits
            if audit.symbol in symbols
        },
    }


def likely_phase2_red_explanation() -> list[str]:
    return [
        "The live Turbo score is derived from regressor agreement and predicted return magnitude, while Phase 2 measured trade-path directional outcomes.",
        "The training target is future close-to-close net return at 12 candles; it does not require target-before-stop, controlled MAE, or survival through bot exits.",
        "MFE/MAE are computed in the dataset but not optimized by the current model, so high scores can still have bad adverse excursion.",
        "Validation reports MAE/RMSE of return regression, not hit rate, expectancy, p90 MAE, or score-bucket calibration.",
        "The score formula clips predicted return magnitude by a fixed 0.003 divisor and is not calibrated by symbol/side/regime.",
        "Actual entries are filtered by Aegis/Probe/EventRisk/CleanEntry/Regime contexts, so raw signals can be directionally RED even if guarded trades sometimes improve.",
    ]


def recommendations_for_phase_b() -> list[str]:
    return [
        "Add path-aware hit-before-stop targets for LONG and SHORT using high/low candles, not close-only future returns.",
        "Add MAE danger and MFE/MAE ratio targets to penalize entries that go deeply adverse before any possible win.",
        "Add bounded trade_quality targets that reward target-before-stop and penalize adverse excursion, fees, and slippage.",
        "Keep current return regressors as secondary signals so V2 can compare against V1 without replacing live behavior.",
        "Persist target distribution reports per symbol/side/lookback before training any V2 model.",
    ]


def load_operable_targets_context(path_text: str | None) -> dict[str, Any]:
    context: dict[str, Any] = {
        "operable_v2_targets_present": True,
        "schema_version": OPERABLE_TARGET_SCHEMA_VERSION,
        "target_names": list(OPERABLE_TARGET_NAMES),
        "trade_quality_formula": TRADE_QUALITY_FORMULA,
        "distribution_report_path": path_text,
        "distribution_report_loaded": False,
        "global_horizon_12": [],
    }
    if not path_text:
        return context
    try:
        payload = json.loads(Path(path_text).read_text(encoding="utf-8"))
        context["distribution_report_loaded"] = True
        context["global_horizon_12"] = payload.get("global_horizon_12", [])
        context["best_symbol_sides_30d_h12"] = payload.get("best_symbol_sides_30d_h12", [])
        context["worst_symbol_sides_30d_h12"] = payload.get("worst_symbol_sides_30d_h12", [])
    except Exception as exc:
        context["distribution_report_error"] = repr(exc)
    return context


def feature_quality_summary(feature_rows: list[FeatureAuditRow]) -> dict[str, Any]:
    constant = [row.feature_name for row in feature_rows if (row.constant_rate or 0.0) >= 0.5]
    high_zero = [row.feature_name for row in feature_rows if (row.zero_rate or 0.0) > 0.9]
    non_finite = [
        row.feature_name
        for row in feature_rows
        if (row.nan_rate or 0.0) > 0.0 or (row.inf_rate or 0.0) > 0.0
    ]
    duplicate = [
        row.feature_name
        for row in feature_rows
        if row.duplicate_correlation_risk not in {"none_detected", "not_measured"}
    ]
    warnings: list[str] = []
    if any("candle_progress" in name for name in high_zero):
        warnings.append("candle_progress_family_all_zero_in_observed_datasets")
    if constant:
        warnings.append("constant_features_present")
    if duplicate:
        warnings.append("duplicate_or_near_duplicate_features_present")
    if non_finite:
        warnings.append("non_finite_features_present")
    return {
        "feature_count": len(feature_rows),
        "constant_feature_count": len(constant),
        "constant_features": constant,
        "over_90_pct_zero_feature_count": len(high_zero),
        "over_90_pct_zero_features": high_zero,
        "non_finite_feature_count": len(non_finite),
        "non_finite_features": non_finite,
        "duplicate_or_correlated_feature_count": len(duplicate),
        "warnings": warnings,
    }


def build_report(
    symbols: list[str],
    lookback_days: int,
    load_dataset: bool,
    operable_targets_report: str | None = None,
) -> tuple[dict[str, Any], list[FeatureAuditRow], list[LeakageRiskRow]]:
    audits: list[SymbolDatasetAudit] = []
    stat_maps: list[dict[str, dict[str, float | str]]] = []
    for symbol in symbols:
        audit, stats = audit_symbol_dataset(symbol, lookback_days, load_dataset)
        audits.append(audit)
        if stats:
            stat_maps.append(stats)
    merged_stats = merge_feature_stats(stat_maps)
    feature_rows = build_feature_rows(merged_stats)
    leakage_rows = build_leakage_rows()
    report = {
        "schema_version": "aegis_turbo_training_pipeline_audit_v1",
        "created_at": utc_now().isoformat(),
        "symbols": symbols,
        "lookback_days": lookback_days,
        "load_dataset": load_dataset,
        "source_files": {path: {"functions_or_classes": assigned_names_from_file(path)} for path in SOURCE_FILES},
        "pipeline_map": build_pipeline_map(),
        "current_targets": {
            "trained_targets": ["long_net_return_12", "short_net_return_12"],
            "reported_but_not_trained_targets": [
                "future_return_6",
                "future_return_12",
                "future_return_24",
                "mfe_12",
                "mae_12",
                "mfe_24",
                "mae_24",
                "long_good_12",
                "short_good_12",
            ],
            "target_gap": "Return target is close-to-close and does not model SL/TP/BE/trailing or hit-before-stop order.",
            "operable_targets": load_operable_targets_context(operable_targets_report),
        },
        "training_settings": {
            "model_type": "HistGradientBoostingRegressor",
            "loss": "absolute_error",
            "min_train_samples": MIN_TRAIN_SAMPLES,
            "validation_pct": VALIDATION_PCT,
            "lookback_days": list(DEFAULT_TURBO_CONFIG.lookback_days),
            "timeframe": DEFAULT_TURBO_CONFIG.timeframe,
            "feature_count_expected": len(edge_feature_names(list(FEATURE_COLUMNS))),
        },
        "train_live_parity": build_train_live_parity(symbols, audits),
        "symbol_audits": [asdict(audit) for audit in audits],
        "feature_summary": [asdict(row) for row in feature_rows],
        "feature_quality_summary": feature_quality_summary(feature_rows),
        "leakage_risks": [asdict(row) for row in leakage_rows],
        "likely_phase2_red_explanation": likely_phase2_red_explanation(),
        "phase_b_recommendations": recommendations_for_phase_b(),
        "safety": {
            "trained_models": False,
            "modified_live_config": False,
            "modified_live_logic": False,
            "promoted_models": False,
        },
    }
    return report, feature_rows, leakage_rows


def write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    status_lines = []
    for audit in report.get("symbol_audits", []):
        status = "loaded" if audit.get("dataset_loaded") else "not_loaded"
        issue = ",".join(audit.get("warnings") or []) or audit.get("error") or "none"
        status_lines.append(f"| {audit.get('symbol')} | {status} | {audit.get('sample_count')} | {audit.get('feature_count')} | {issue} |")
    leakage_lines = [
        f"| {row['area']} | {row['severity']} | {row['risk']} | {row['evidence']} |"
        for row in report.get("leakage_risks", [])
    ]
    feature_quality = report.get("feature_quality_summary", {})
    operable_targets = report.get("current_targets", {}).get("operable_targets", {})
    lines = [
        f"# Aegis Turbo Training Pipeline Audit {report['created_at']}",
        "",
        "## Executive Summary",
        "",
        "- Current V1 training is operationally coherent but optimizes return regression, not bot-operable trade survival.",
        "- Train/live feature parity is mostly direct because runtime consumes `live_X` from the same snapshot builder.",
        "- Main risk is target/validation misalignment, not obvious future leakage in feature construction.",
        "- V1 runtime score is not calibrated probability; this likely explains Phase 2 RED directional results.",
        "",
        "## Pipeline Map",
        "",
    ]
    for item in report.get("pipeline_map", []):
        lines.append(f"- `{item['stage']}`: {item['implementation']} -> {item['output']}. {item['notes']}")
    lines.extend([
        "",
        "## Symbol Dataset Audit",
        "",
        "| Symbol | Dataset | Samples | Features | Issues |",
        "|---|---:|---:|---:|---|",
        *status_lines,
        "",
        "## Feature Quality Summary",
        "",
        f"- Feature columns audited: `{feature_quality.get('feature_count')}`",
        f"- Non-finite feature columns observed: `{feature_quality.get('non_finite_feature_count')}`",
        f"- Constant feature columns observed: `{feature_quality.get('constant_feature_count')}`",
        f"- Over 90% zero feature columns observed: `{feature_quality.get('over_90_pct_zero_feature_count')}`",
        f"- Warnings: `{feature_quality.get('warnings') or []}`",
        f"- Constant features: `{', '.join(feature_quality.get('constant_features') or []) or 'none'}`",
        "",
        "## Current Target Gap",
        "",
        f"- Trained targets: `{', '.join(report['current_targets']['trained_targets'])}`",
        f"- Gap: {report['current_targets']['target_gap']}",
        "",
        "## Fase B Operable Targets",
        "",
        f"- Operable V2 targets present: `{operable_targets.get('operable_v2_targets_present')}`",
        f"- Schema: `{operable_targets.get('schema_version')}`",
        f"- New target column count: `{len(operable_targets.get('target_names') or [])}`",
        f"- Distribution report: `{operable_targets.get('distribution_report_path') or 'not supplied'}`",
        f"- Distribution report loaded: `{operable_targets.get('distribution_report_loaded')}`",
        f"- Formula: `{operable_targets.get('trade_quality_formula')}`",
        "",
        "## Leakage And Parity Risks",
        "",
        "| Area | Severity | Risk | Evidence |",
        "|---|---|---|---|",
        *leakage_lines,
        "",
        "## Why Phase 2 Raw Directional Metrics Were RED",
        "",
    ])
    for item in report.get("likely_phase2_red_explanation", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Recommended Fase B Scope", ""])
    for item in report.get("phase_b_recommendations", []):
        lines.append(f"- {item}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only audit of Aegis Turbo V1 training pipeline.")
    parser.add_argument("--symbols", help="Comma-separated symbols. Defaults to expected Aegis Turbo symbols.")
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument("--out-dir", default="/home/jasan/Develop")
    parser.add_argument("--skip-dataset", action="store_true", help="Only static audit; skip loading build_recent_dataset.")
    parser.add_argument("--operable-targets-report", help="Optional JSON output from audit_turbo_operable_targets.py.")
    args = parser.parse_args()

    symbols = parse_symbols(args.symbols)
    stamp = utc_stamp()
    out_dir = Path(args.out_dir)
    report, feature_rows, leakage_rows = build_report(
        symbols,
        args.lookback_days,
        load_dataset=not args.skip_dataset,
        operable_targets_report=args.operable_targets_report,
    )

    json_path = out_dir / f"aegis_turbo_training_pipeline_audit_{stamp}.json"
    md_path = out_dir / f"aegis_turbo_training_pipeline_audit_{stamp}.md"
    features_path = out_dir / f"aegis_turbo_training_features_{stamp}.csv"
    leakage_path = out_dir / f"aegis_turbo_training_leakage_risks_{stamp}.csv"

    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(md_path, report)
    write_csv(features_path, [asdict(row) for row in feature_rows], FEATURE_CSV_COLUMNS)
    write_csv(leakage_path, [asdict(row) for row in leakage_rows], LEAKAGE_CSV_COLUMNS)

    print(json.dumps({
        "report": str(md_path),
        "json": str(json_path),
        "features_csv": str(features_path),
        "leakage_csv": str(leakage_path),
        "symbols": symbols,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
