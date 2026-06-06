#!/usr/bin/env python3
"""Research-only profiler for LONG training pipeline performance."""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.tools.profile_long_alpha_families_a import add_features, json_safe, mean, quantile, top_fraction_mask
from aegis_alpha.tools.train_long_alpha_candidates_c import (
    DEFAULT_DB_PATH,
    DEFAULT_MODEL_DIR,
    add_long_c_features,
    assert_research_only_path,
    build_probability_buckets,
    compute_labels,
    feature_hash,
    load_candles_research,
    model_predict_proba,
    regressor,
    safe_ap,
    safe_auc,
    safe_corr,
    safe_spearman,
    select_long_family_features,
    train_or_skip_classifier,
)
from aegis_alpha.tools.walk_forward_long_alpha_family_d import build_expanding_folds, selection_mask
from aegis_alpha.turbo.long_research_cache import LongResearchCache

DEFAULT_OUT_DIR = Path("/home/jasan/Develop")


@dataclass
class SectionTimer:
    sections: list[dict[str, Any]] = field(default_factory=list)

    @contextmanager
    def section(self, name: str, **meta: Any):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            row = {"section": name, "seconds": elapsed}
            row.update(meta)
            self.sections.append(row)

    def total(self, prefix: str | None = None) -> float:
        rows = self.sections if prefix is None else [r for r in self.sections if str(r["section"]).startswith(prefix)]
        return float(sum(float(r["seconds"]) for r in rows))


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_cmd(args: list[str], timeout: float = 5.0) -> dict[str, Any]:
    if not args or shutil.which(args[0]) is None:
        return {"available": False, "stdout": "", "stderr": "command not found", "returncode": None}
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
        return {"available": True, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip(), "returncode": proc.returncode}
    except Exception as exc:
        return {"available": True, "stdout": "", "stderr": str(exc), "returncode": None}


def detect_gpu() -> dict[str, Any]:
    rocm = run_cmd(["rocm-smi"], timeout=3)
    radeontop = run_cmd(["radeontop", "-d", "-", "-l", "1"], timeout=2)
    nvidia = run_cmd(["nvidia-smi"], timeout=2)
    torch_probe = (
        "import json\n"
        "try:\n"
        " import torch\n"
        " print(json.dumps({'available': True, 'version': getattr(torch, '__version__', None), "
        "'cuda_is_available': bool(torch.cuda.is_available()), 'device_count': int(torch.cuda.device_count()), "
        "'hip': getattr(torch.version, 'hip', None)}))\n"
        "except Exception as exc:\n"
        " print(json.dumps({'available': False, 'error': str(exc)}))\n"
    )
    probe = run_cmd([sys.executable, "-c", torch_probe], timeout=3)
    try:
        torch_info = json.loads(probe.get("stdout") or "{}")
    except Exception:
        torch_info = {"available": False, "error": probe.get("stderr") or "torch probe timeout/invalid output"}
    return {"rocm_smi": rocm, "radeontop": radeontop, "nvidia_smi": nvidia, "torch": torch_info}


def memory_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    try:
        import psutil  # type: ignore

        proc = psutil.Process()
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        snapshot.update({
            "rss_bytes": proc.memory_info().rss,
            "system_used_bytes": vm.used,
            "system_available_bytes": vm.available,
            "swap_used_bytes": swap.used,
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "loadavg": list(getattr(__import__("os"), "getloadavg")()),
        })
    except Exception as exc:
        snapshot["error"] = str(exc)
    return snapshot


def classify_bottleneck(timings: dict[str, float], *, loadavg: float | None = None, cores: int | None = None, swap_used_bytes: int | None = None) -> str:
    total = max(float(timings.get("total_wall_time", 0.0)), 1e-9)
    if swap_used_bytes and swap_used_bytes > 0:
        return "BOTTLENECK_MEMORY_PRESSURE"
    if loadavg is not None and cores and loadavg > cores * 1.25:
        return "BOTTLENECK_PROCESS_CONTENTION"
    sqlite = timings.get("load_candles_symbol_time", 0.0) + timings.get("load_btc_context_time", 0.0) + timings.get("load_eth_context_time", 0.0)
    feature = timings.get("feature_build_base_time", 0.0) + timings.get("feature_build_long_c_time", 0.0)
    fit = sum(v for k, v in timings.items() if k.startswith("train_"))
    report = timings.get("report_write_time", 0.0) + timings.get("bucket_time", 0.0)
    if sqlite / total > 0.30:
        return "BOTTLENECK_SQLITE_IO"
    if feature / total > 0.30:
        return "BOTTLENECK_FEATURE_BUILD"
    if fit / total > 0.40:
        return "BOTTLENECK_SKLEARN_FIT"
    if report / total > 0.20:
        return "BOTTLENECK_REPORT_BUCKETS"
    return "BOTTLENECK_UNKNOWN"


def estimate_experiment_cost(
    *,
    symbols: int,
    configs_per_symbol: int = 1,
    targets: int = 1,
    horizons: int = 1,
    repair_modes: int = 1,
    folds: int = 4,
    models_per_fold: int = 4,
    avg_time_per_model_fit: float = 0.0,
    avg_time_per_config: float = 0.0,
) -> dict[str, Any]:
    config_count = symbols * configs_per_symbol * targets * horizons * repair_modes
    total_model_fits = config_count * folds * models_per_fold
    estimated_fit_seconds = total_model_fits * max(avg_time_per_model_fit, 0.0)
    estimated_config_seconds = config_count * max(avg_time_per_config, 0.0)
    estimated_seconds = max(estimated_fit_seconds, estimated_config_seconds)
    if estimated_seconds / 60.0 > 45:
        recommendation = "too_large_for_codex_session"
    elif estimated_seconds / 60.0 > 15:
        recommendation = "medium_run"
    else:
        recommendation = "safe_small_run"
    return {
        "symbols": symbols,
        "configs_per_symbol": configs_per_symbol,
        "targets": targets,
        "horizons": horizons,
        "repair_modes": repair_modes,
        "folds": folds,
        "models_per_fold": models_per_fold,
        "config_count": config_count,
        "total_model_fits": total_model_fits,
        "avg_time_per_model_fit_seconds": max(avg_time_per_model_fit, 0.0),
        "avg_time_per_config_seconds": max(avg_time_per_config, 0.0),
        "estimated_fit_minutes": estimated_fit_seconds / 60.0,
        "estimated_config_minutes": estimated_config_seconds / 60.0,
        "estimated_total_minutes": estimated_seconds / 60.0,
        "recommendation": recommendation,
    }


def _valid_indices(frame, labels: dict[str, np.ndarray], features: list[str], horizon: int) -> np.ndarray:
    valid = frame[["open", "high", "low", "close", "volume"]].notna().all(axis=1).to_numpy().copy()
    valid[:220] = False
    valid[-horizon:] = False
    for name in features:
        valid &= np.isfinite(frame[name].to_numpy(float))
    valid &= np.isfinite(labels["quality"])
    return np.flatnonzero(valid)


def _cache(args: argparse.Namespace) -> LongResearchCache | None:
    return getattr(args, "_long_research_cache", None) if getattr(args, "use_cache", False) else None


def cached_value(args: argparse.Namespace, namespace: str, parts: tuple[Any, ...], factory):
    cache = _cache(args)
    if cache is None:
        return factory()
    return cache.get_or_set(namespace, parts, factory)


def valid_idx_hash(idx: np.ndarray) -> str:
    import hashlib
    arr = np.asarray(idx, dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def _fit_classifier(name: str, x_train: np.ndarray, y_train: np.ndarray, seed: int, max_iter: int) -> tuple[Any | None, bool]:
    model, _ = train_or_skip_classifier(name, x_train, y_train, seed, max_iter)
    return model, model is not None


def profile_once(args: argparse.Namespace, timer: SectionTimer) -> dict[str, Any]:
    db_path = Path(args.db_path)
    max_iter = 30 if args.fast else 120
    total_model_fit_count = 0
    total_prediction_count = 0
    model_fit_rows: list[dict[str, Any]] = []
    cache = _cache(args)
    with timer.section("load_candles_symbol_time"):
        parts = cache.ohlcv_key(args.symbol, args.lookback_days, db_path) if cache else ()
        df = cached_value(args, "ohlcv", parts, lambda: load_candles_research(db_path, args.symbol, args.lookback_days))
    with timer.section("load_btc_context_time"):
        if args.symbol == "BTCUSDT":
            btc = df
        else:
            parts = cache.ohlcv_key("BTCUSDT", args.lookback_days, db_path) if cache else ()
            btc = cached_value(args, "ohlcv", parts, lambda: load_candles_research(db_path, "BTCUSDT", args.lookback_days))
    with timer.section("load_eth_context_time"):
        if args.symbol == "ETHUSDT":
            eth = df
        else:
            parts = cache.ohlcv_key("ETHUSDT", args.lookback_days, db_path) if cache else ()
            eth = cached_value(args, "ohlcv", parts, lambda: load_candles_research(db_path, "ETHUSDT", args.lookback_days))
    with timer.section("feature_build_base_time"):
        parts = cache.feature_key(args.symbol, args.lookback_days, f"base:{args.feature_mode}", db_path) if cache else ()
        base = cached_value(args, "feature_base", parts, lambda: add_features(df, btc, eth))
    with timer.section("feature_build_long_c_time"):
        parts = cache.feature_key(args.symbol, args.lookback_days, f"long_c:{args.family}:{args.feature_mode}", db_path) if cache else ()
        frame = cached_value(args, "feature_long_c", parts, lambda: add_long_c_features(base).reset_index(drop=True))
    with timer.section("target_label_time"):
        parts = cache.labels_key(args.symbol, args.target, args.horizon, args.lookback_days, db_path) if cache else ()
        labels = cached_value(args, "labels", parts, lambda: compute_labels(frame["close"].to_numpy(float), frame["high"].to_numpy(float), frame["low"].to_numpy(float), frame, args.target, args.horizon))
    with timer.section("feature_select_time"):
        features, missing, proxies = select_long_family_features(frame, args.family, args.feature_mode)
    with timer.section("valid_mask_time"):
        idx = _valid_indices(frame, labels, features, args.horizon)
    with timer.section("fold_build_time"):
        parts = cache.folds_key(len(idx), args.fold_count, args.min_train_samples, args.min_test_samples) if cache else ()
        folds = cached_value(args, "folds", parts, lambda: build_expanding_folds(len(idx), args.fold_count, args.min_train_samples, args.min_test_samples))
    fold_summaries: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    cache = _cache(args)
    if cache is not None:
        schema = feature_hash(features)
        x_all = cached_value(args, "x_matrix", cache.x_key(args.symbol, schema, valid_idx_hash(idx)), lambda: frame[features].to_numpy(dtype=float))
    else:
        x_all = frame[features].to_numpy(dtype=float)
    for fold_no, (train_local, test_local) in enumerate(folds, start=1):
        train_idx = idx[train_local]
        test_idx = idx[test_local]
        x_train, x_test = x_all[train_idx], x_all[test_idx]
        seed = abs(hash((args.symbol, args.family, args.target, args.horizon, fold_no))) % 2_000_000_000
        y_hit_train = labels["hit"][train_idx]
        y_danger_train = labels["mae_danger"][train_idx]
        y_exh_train = labels["late_entry_exhaustion"][train_idx]
        y_quality_train = labels["quality"][train_idx]
        with timer.section("train_hit_classifier_time", fold=fold_no):
            hit_model, did_fit = _fit_classifier("hit", x_train, y_hit_train, seed, max_iter)
        total_model_fit_count += int(did_fit)
        model_fit_rows.append({"fold": fold_no, "model": "long_hit_classifier", "fit": did_fit, "samples": len(train_idx), "features": len(features)})
        with timer.section("train_quality_regressor_time", fold=fold_no):
            quality_model = regressor(seed + 3, max_iter).fit(x_train, y_quality_train)
        total_model_fit_count += 1
        model_fit_rows.append({"fold": fold_no, "model": "long_quality_regressor", "fit": True, "samples": len(train_idx), "features": len(features)})
        with timer.section("train_danger_classifier_time", fold=fold_no):
            danger_model, did_fit = _fit_classifier("danger", x_train, y_danger_train, seed + 1, max_iter)
        total_model_fit_count += int(did_fit)
        model_fit_rows.append({"fold": fold_no, "model": "long_mae_danger_classifier", "fit": did_fit, "samples": len(train_idx), "features": len(features)})
        with timer.section("train_exhaustion_classifier_time", fold=fold_no):
            exhaustion_model, did_fit = _fit_classifier("exhaustion", x_train, y_exh_train, seed + 2, max_iter)
        total_model_fit_count += int(did_fit)
        model_fit_rows.append({"fold": fold_no, "model": "long_exhaustion_classifier", "fit": did_fit, "samples": len(train_idx), "features": len(features)})
        with timer.section("predict_hit_time", fold=fold_no):
            hit_pred = model_predict_proba(hit_model, x_test)
        with timer.section("predict_quality_time", fold=fold_no):
            quality_pred = quality_model.predict(x_test)
        with timer.section("predict_danger_time", fold=fold_no):
            danger_pred = model_predict_proba(danger_model, x_test)
        with timer.section("predict_exhaustion_time", fold=fold_no):
            exhaustion_pred = model_predict_proba(exhaustion_model, x_test)
        total_prediction_count += len(test_idx) * 4
        with timer.section("selection_mask_time", fold=fold_no):
            selected = selection_mask(hit_pred, quality_pred, danger_pred, exhaustion_pred)
        with timer.section("metrics_time", fold=fold_no):
            y_hit = labels["hit"][test_idx]
            y_quality = labels["quality"][test_idx]
            y_stop = labels["stop"][test_idx]
            y_mae = labels["mae"][test_idx]
            baseline_hit = mean(y_hit) or 0.0
            selected_hit = mean(y_hit[selected]) if selected.any() else None
            baseline_quality = mean(y_quality) or 0.0
            selected_quality = mean(y_quality[selected]) if selected.any() else None
            baseline_stop = mean(y_stop) or 0.0
            selected_stop = mean(y_stop[selected]) if selected.any() else None
            baseline_mae = quantile(y_mae, 0.90) or 0.0
            selected_mae = quantile(y_mae[selected], 0.90) if selected.any() else None
            fold_summaries.append({
                "fold": fold_no,
                "train_samples": int(len(train_idx)),
                "test_samples": int(len(test_idx)),
                "baseline_hit_rate": baseline_hit,
                "selected_hit_rate": selected_hit,
                "hit_lift": (selected_hit - baseline_hit) if selected_hit is not None else None,
                "baseline_quality": baseline_quality,
                "selected_quality": selected_quality,
                "net_quality_lift_after_costs": (selected_quality - baseline_quality) if selected_quality is not None else None,
                "stop_rate_delta": (selected_stop - baseline_stop) if selected_stop is not None else None,
                "p90_mae_delta": ((selected_mae - baseline_mae) / max(baseline_mae, 1e-12)) if selected_mae is not None else None,
                "hit_auc": safe_auc(y_hit, np.nan_to_num(hit_pred, nan=baseline_hit)) if hit_model is not None else None,
                "hit_average_precision": safe_ap(y_hit, np.nan_to_num(hit_pred, nan=baseline_hit)) if hit_model is not None else None,
                "quality_corr": safe_corr(y_quality, quality_pred),
                "quality_spearman_corr": safe_spearman(y_quality, quality_pred),
                "selected_fraction": float(selected.mean()) if len(selected) else 0.0,
            })
        with timer.section("bucket_time", fold=fold_no):
            bucket_rows.extend(build_probability_buckets(
                np.nan_to_num(hit_pred, nan=float(baseline_hit)),
                labels["hit"][test_idx],
                labels["quality"][test_idx],
                labels["mae_danger"][test_idx],
                labels["late_entry_exhaustion"][test_idx],
                symbol=args.symbol,
                family=args.family,
                target_name=args.target,
                horizon=args.horizon,
                bucket_source=f"perf_fold_{fold_no}_hit_probability",
            ))
    return {
        "symbol": args.symbol,
        "family": args.family,
        "target": args.target,
        "horizon": args.horizon,
        "lookback_days": args.lookback_days,
        "fold_count_requested": args.fold_count,
        "fold_count_built": len(folds),
        "samples_total": int(len(frame)),
        "samples_valid": int(len(idx)),
        "feature_count": int(len(features)),
        "missing_features": missing,
        "proxy_features_used": proxies,
        "feature_schema_hash": feature_hash(features),
        "total_model_fit_count": total_model_fit_count,
        "total_prediction_count": total_prediction_count,
        "fold_summaries": fold_summaries,
        "bucket_rows": bucket_rows,
        "model_fit_rows": model_fit_rows,
    }


def aggregate_timings(timer: SectionTimer, total_wall_time: float) -> dict[str, float]:
    timings: dict[str, float] = {"total_wall_time": total_wall_time}
    for row in timer.sections:
        timings[row["section"]] = timings.get(row["section"], 0.0) + float(row["seconds"])
    return timings


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys or ["empty"], extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(row.get(key)) for key in keys})


def recommendations(bottleneck: str, gpu: dict[str, Any]) -> list[str]:
    base = [
        "cache OHLCV por symbol/lookback y reutilizar BTC/ETH context",
        "cache feature frame por symbol/lookback/family y labels por target/horizon",
        "cache folds e X matrix para evitar recálculo entre repair modes",
        "desactivar buckets/reportes detallados en smoke",
        "early-stop de configs con folds negativos tempranos",
        "mantener max_iter bajo en smoke y subirlo solo para finalists",
        "paralelizar por símbolo/config con workers=max(1, ncpu//3) si load y PM2 están estables",
    ]
    if bottleneck == "BOTTLENECK_SKLEARN_FIT":
        base.insert(0, "priorizar cache de matrices y reducir fits por matriz antes de ampliar grids")
    if gpu.get("torch", {}).get("cuda_is_available"):
        base.append("probar backend torch_gpu MLP research-only en 1 símbolo; sklearn HistGradientBoosting no usa ROCm")
    else:
        base.append("GPU no acelera sklearn actual; evaluar torch ROCm solo como backend experimental futuro")
    return base


def write_reports(profile: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    args = profile.get("args", {})
    if "use_cache" in args:
        mode = "cache" if args.get("use_cache") else "nocache"
        prefix = f"aegis_long_perf_b_profile_{mode}"
    else:
        prefix = "aegis_long_perf_profile"
    paths = {
        "md": str(out_dir / f"{prefix}_{stamp}.md"),
        "json": str(out_dir / f"{prefix}_{stamp}.json"),
        "timing_sections": str(out_dir / f"aegis_long_perf_timing_sections_{stamp}.csv"),
        "model_fits": str(out_dir / f"aegis_long_perf_model_fits_{stamp}.csv"),
        "cost_estimates": str(out_dir / f"aegis_long_perf_cost_estimates_{stamp}.csv"),
        "recommendations": str(out_dir / f"aegis_long_perf_b_recommendations_{stamp}.md" if "use_cache" in args else out_dir / f"aegis_long_perf_recommendations_{stamp}.md"),
        "cache_stats": str(out_dir / f"aegis_long_perf_b_cache_stats_{stamp}.csv"),
        "speedup": str(out_dir / f"aegis_long_perf_b_speedup_{stamp}.csv"),
    }
    Path(paths["json"]).write_text(json.dumps(json_safe(profile), indent=2, sort_keys=True) + "\n")
    write_csv(Path(paths["timing_sections"]), profile["timing_sections"])
    write_csv(Path(paths["model_fits"]), profile["result"]["model_fit_rows"])
    write_csv(Path(paths["cost_estimates"]), profile["cost_estimates"])
    if "cache_stats" in profile:
        write_csv(Path(paths["cache_stats"]), [profile["cache_stats"]])
    if "speedup" in profile:
        write_csv(Path(paths["speedup"]), [profile["speedup"]] + profile.get("repeat_runs", []))
    rec_lines = ["# LONG Training Performance Recommendations", ""]
    rec_lines.extend(f"- {item}" for item in profile["recommendations"])
    Path(paths["recommendations"]).write_text("\n".join(rec_lines) + "\n")
    timings = profile["timings"]
    lines = [
        "# Aegis LONG Training Performance Profile",
        "",
        "## Safety",
        "- research-only",
        "- no live changes",
        "- no active_manifest",
        "- no YAML",
        "- no PM2",
        "- no orders",
        "",
        "## Benchmark",
        f"- Config: `{profile['args']['symbol']} {profile['args']['family']} {profile['args']['target']} h{profile['args']['horizon']}`",
        f"- Total wall time: `{timings.get('total_wall_time', 0.0):.3f}s`",
        f"- Bottleneck: `{profile['bottleneck']}`",
        f"- Model fits: `{profile['result']['total_model_fit_count']}`",
        f"- Predictions: `{profile['result']['total_prediction_count']}`",
        f"- Cache enabled: `{profile.get('cache_enabled', False)}` hits=`{profile.get('cache_stats', {}).get('hits')}` misses=`{profile.get('cache_stats', {}).get('misses')}`",
        f"- Internal repeat speedup: `{profile.get('speedup', {}).get('internal_speedup')}`",
        f"- GPU used: `False` (sklearn HistGradientBoosting CPU-bound)",
        f"- Torch ROCm available: `{profile['gpu'].get('torch', {}).get('cuda_is_available')}` hip=`{profile['gpu'].get('torch', {}).get('hip')}`",
        "",
        "## Timing Sections",
        "| section | seconds |",
        "|---|---:|",
    ]
    for row in sorted(profile["timing_sections"], key=lambda r: -float(r["seconds"]))[:30]:
        lines.append(f"| {row['section']} | {float(row['seconds']):.4f} |")
    lines += [
        "",
        "## D3.1 Cost Estimate",
        "| config_count | total_model_fits | fit_minutes | config_minutes | estimated_minutes | recommendation |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    d31 = profile["cost_estimates"][0]
    lines.append(f"| {d31['config_count']} | {d31['total_model_fits']} | {float(d31['estimated_fit_minutes']):.2f} | {float(d31['estimated_config_minutes']):.2f} | {float(d31['estimated_total_minutes']):.2f} | {d31['recommendation']} |")
    Path(paths["md"]).write_text("\n".join(lines) + "\n")
    return paths


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    assert_research_only_path(args.out_dir)
    assert_research_only_path(args.model_dir)
    args._long_research_cache = LongResearchCache(args.cache_max_items) if getattr(args, "use_cache", False) else None
    gpu_before = detect_gpu() if args.profile_gpu else {}
    mem_before = memory_snapshot() if args.profile_memory else {}
    timer = SectionTimer()
    start = time.perf_counter()
    result: dict[str, Any] = {}
    repeat_runs: list[dict[str, Any]] = []
    for repeat_index in range(args.repeat):
        before_hits = args._long_research_cache.hits if getattr(args, "_long_research_cache", None) is not None else 0
        before_misses = args._long_research_cache.misses if getattr(args, "_long_research_cache", None) is not None else 0
        repeat_start = time.perf_counter()
        result = profile_once(args, timer)
        repeat_seconds = time.perf_counter() - repeat_start
        after_hits = args._long_research_cache.hits if getattr(args, "_long_research_cache", None) is not None else 0
        after_misses = args._long_research_cache.misses if getattr(args, "_long_research_cache", None) is not None else 0
        repeat_runs.append({
            "repeat_index": repeat_index + 1,
            "seconds": repeat_seconds,
            "cache_hits_delta": after_hits - before_hits,
            "cache_misses_delta": after_misses - before_misses,
            "model_fits": result.get("total_model_fit_count", 0),
        })
    total_wall = time.perf_counter() - start
    timings = aggregate_timings(timer, total_wall)
    avg_fit = sum(v for k, v in timings.items() if k.startswith("train_")) / max(result.get("total_model_fit_count", 0), 1)
    mem_after = memory_snapshot() if args.profile_memory else {}
    loadavg = None
    cores = None
    swap = None
    try:
        loadavg = float(mem_after.get("loadavg", [None])[0]) if mem_after else None
        cores = int(__import__("os").cpu_count() or 1)
        swap = int(mem_after.get("swap_used_bytes") or 0)
    except Exception:
        pass
    bottleneck = classify_bottleneck(timings, loadavg=loadavg, cores=cores, swap_used_bytes=swap)
    avg_config = total_wall / max(args.repeat, 1)
    cost_estimates = [
        estimate_experiment_cost(symbols=1, targets=4, horizons=1, repair_modes=6, folds=4, models_per_fold=4, avg_time_per_model_fit=avg_fit, avg_time_per_config=avg_config),
        estimate_experiment_cost(symbols=3, targets=1, horizons=1, repair_modes=1, folds=args.fold_count, models_per_fold=4, avg_time_per_model_fit=avg_fit, avg_time_per_config=avg_config),
    ]
    profile = {
        "schema_version": "aegis_long_perf_profile_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "RESEARCH_ONLY",
        "safety": {"no_live_changes": True, "no_active_manifest": True, "no_yaml": True, "no_pm2": True, "no_orders": True},
        "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        "timings": timings,
        "timing_sections": timer.sections,
        "result": result,
        "memory_before": mem_before,
        "memory_after": mem_after,
        "gpu": gpu_before,
        "bottleneck": bottleneck,
        "cost_estimates": cost_estimates,
        "recommendations": recommendations(bottleneck, gpu_before),
        "gpu_used": False,
        "sklearn_uses_gpu": False,
        "cache_enabled": bool(getattr(args, "use_cache", False)),
        "cache_stats": args._long_research_cache.summary() if getattr(args, "_long_research_cache", None) is not None else {"enabled": False},
        "repeat_runs": repeat_runs,
        "speedup": {
            "repeat": args.repeat,
            "cache_hits": args._long_research_cache.hits if getattr(args, "_long_research_cache", None) is not None else 0,
            "first_repeat_seconds": repeat_runs[0]["seconds"] if repeat_runs else None,
            "last_repeat_seconds": repeat_runs[-1]["seconds"] if repeat_runs else None,
            "internal_speedup": (repeat_runs[0]["seconds"] / repeat_runs[-1]["seconds"]) if len(repeat_runs) > 1 and repeat_runs[-1]["seconds"] > 0 else None,
        },
    }
    with timer.section("report_write_time"):
        paths = write_reports(profile, Path(args.out_dir))
    profile["timing_sections"] = timer.sections
    profile["timings"] = aggregate_timings(timer, total_wall + profile["timing_sections"][-1]["seconds"])
    profile["reports"] = paths
    profile["bottleneck"] = classify_bottleneck(profile["timings"], loadavg=loadavg, cores=cores, swap_used_bytes=swap)
    profile["recommendations"] = recommendations(profile["bottleneck"], gpu_before)
    Path(paths["json"]).write_text(json.dumps(json_safe(profile), indent=2, sort_keys=True) + "\n")
    return profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="AVAXUSDT")
    parser.add_argument("--family", default="micro_roe_momentum_long")
    parser.add_argument("--target", default="long_roe12_before_minus8")
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--lookback-days", type=int, default=60)
    parser.add_argument("--fold-count", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--feature-mode", choices=("selected_family", "combined_v3_all"), default="selected_family")
    parser.add_argument("--min-train-samples", type=int, default=1000)
    parser.add_argument("--min-test-samples", type=int, default=250)
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--profile-gpu", action="store_true")
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument("--use-cache", action="store_true")
    cache_group.add_argument("--no-cache", action="store_false", dest="use_cache")
    parser.set_defaults(use_cache=False)
    parser.add_argument("--cache-max-items", type=int, default=64)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--no-save-models", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = run_profile(args)
    print(json.dumps(json_safe({"reports": profile["reports"], "bottleneck": profile["bottleneck"], "total_wall_time": profile["timings"]["total_wall_time"]}), indent=2))


if __name__ == "__main__":
    main()
