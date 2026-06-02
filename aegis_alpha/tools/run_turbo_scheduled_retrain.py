#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn import __version__ as sklearn_version

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from config.settings import settings  # noqa: E402
from data.storage.database_manager import DatabaseManager  # noqa: E402
from aegis_alpha.turbo.config import DEFAULT_TURBO_CONFIG, TURBO_VERSION  # noqa: E402
from aegis_alpha.turbo.recent_dataset import build_recent_dataset  # noqa: E402
from aegis_alpha.turbo.snapshot_utils import (  # noqa: E402
    load_turbo_snapshot_status,
    normalize_turbo_symbol,
    turbo_snapshot_path,
    turbo_symbol_model_dir,
)
from aegis_alpha.turbo.train_recent_edge import MIN_TRAIN_SAMPLES, train_recent_edge_models  # noqa: E402
from aegis_alpha.turbo.phase_o_overlay import (  # noqa: E402
    PHASE_O_SYMBOLS,
    apply_phase_o_overlay_to_active_manifest,
    validate_phase_o_overlay,
)
from aegis_alpha.tools.refresh_turbo_snapshots import refresh_features_only  # noqa: E402


LOCK_PATH = Path("/tmp/aegis_turbo_retrain.lock")
REPORT_DIR = Path("aegis_alpha/logs/turbo_retrain")
WINDOWS = (7, 14, 30)
SIDES = ("long", "short")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_stamp() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%SZ")


def utc_iso() -> str:
    return utc_now().isoformat()


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def parse_symbols(raw: str) -> list[str]:
    symbols = [normalize_turbo_symbol(item) for item in raw.split(",") if item.strip()]
    return list(dict.fromkeys(symbols))


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def retrain_lock() -> Any:
    if LOCK_PATH.exists():
        try:
            payload = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
            pid = int(payload.get("pid") or 0)
        except Exception:
            payload = {}
            pid = 0
        if pid_alive(pid):
            print("retrain already running", file=sys.stderr)
            raise SystemExit(0)
        print(f"stale retrain lock removed: {LOCK_PATH} payload={payload}", file=sys.stderr)
        LOCK_PATH.unlink(missing_ok=True)

    payload = {"pid": os.getpid(), "started_at": utc_iso()}
    LOCK_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    try:
        yield
    finally:
        try:
            current = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
            if int(current.get("pid") or 0) == os.getpid():
                LOCK_PATH.unlink(missing_ok=True)
        except FileNotFoundError:
            pass


def db_symbol(symbol: str) -> str:
    return symbol if "/" in symbol else symbol.replace("USDT", "/USDT")


def candle_metrics(symbol: str, timeframe: str) -> dict[str, Any]:
    db = DatabaseManager(settings.DATABASE_URL)
    normalized = db_symbol(symbol)
    df = db.get_ohlcv_data(normalized, timeframe)
    gaps = db.get_data_gaps(normalized, timeframe)
    duplicates = 0
    if not df.empty:
        duplicates = int(df.index.duplicated().sum())
    return {
        "rows": int(len(df)),
        "first_timestamp": df.index.min().isoformat() if not df.empty else None,
        "last_candle_ts": df.index.max().isoformat() if not df.empty else None,
        "gaps": int(len(gaps)),
        "duplicates": duplicates,
    }


def update_candles(symbol: str) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/update_candles.py",
        "--symbol",
        symbol,
        "--timeframe",
        DEFAULT_TURBO_CONFIG.timeframe,
    ]
    started = time.time()
    proc = subprocess.run(cmd, cwd=Path.cwd(), text=True, capture_output=True, check=False)
    result = {
        "command": cmd,
        "returncode": int(proc.returncode),
        "duration_seconds": round(time.time() - started, 3),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "success": proc.returncode == 0,
    }
    result.update(candle_metrics(symbol, DEFAULT_TURBO_CONFIG.timeframe))
    if proc.returncode != 0:
        raise RuntimeError(f"update_candles failed for {symbol}: {proc.stderr[-500:] or proc.stdout[-500:]}")
    return result


def expected_model_path(directory: Path, side: str, window: int) -> Path:
    return directory / f"turbo_{side}_edge_{window}d_v010.joblib"


def validate_candidate_models(symbol: str, candidate_dir: Path, train_report: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    model_paths: dict[str, str] = {}
    predictions: dict[str, float] = {}
    sample_counts: dict[str, int] = {}
    dataset_nan_checks: dict[str, dict[str, Any]] = {}

    snapshots: dict[int, dict[str, Any]] = {}
    for window in WINDOWS:
        path = turbo_snapshot_path(window, symbol)
        try:
            with np.load(path, allow_pickle=True) as data:
                x = np.asarray(data["X"], dtype=np.float32)
                live_x = np.asarray(data.get("live_X", x[-1:]), dtype=np.float32)
                snapshots[window] = {"x": x, "live_x": live_x[-1:].astype(np.float32)}
                dataset_nan_checks[f"{window}d"] = {
                    "path": str(path),
                    "sample_count": int(len(x)),
                    "feature_count": int(x.shape[1]) if x.ndim == 2 else 0,
                    "x_all_finite": bool(np.isfinite(x).all()),
                    "live_x_all_finite": bool(np.isfinite(live_x).all()),
                }
                if len(x) < MIN_TRAIN_SAMPLES:
                    errors.append(f"{window}d sample_count {len(x)} < {MIN_TRAIN_SAMPLES}")
                if not np.isfinite(x).all() or not np.isfinite(live_x).all():
                    errors.append(f"{window}d contains non-finite features")
        except Exception as exc:
            errors.append(f"{window}d snapshot validation failed: {exc!r}")

    for window in WINDOWS:
        for side in SIDES:
            key = f"{side}_{window}d"
            path = expected_model_path(candidate_dir, side, window)
            model_paths[key] = str(path)
            if not path.exists():
                errors.append(f"missing model {path}")
                continue
            try:
                bundle = joblib.load(path)
                estimator = bundle.get("estimator") if isinstance(bundle, dict) else bundle
                metadata = bundle.get("metadata", {}) if isinstance(bundle, dict) else {}
                if estimator is None:
                    errors.append(f"{key} estimator missing")
                    continue
                sample_count = int(metadata.get("sample_count") or 0)
                sample_counts[key] = sample_count
                if sample_count < MIN_TRAIN_SAMPLES:
                    errors.append(f"{key} sample_count {sample_count} < {MIN_TRAIN_SAMPLES}")
                live_x = snapshots.get(window, {}).get("live_x")
                if live_x is None:
                    continue
                pred = estimator.predict(live_x)
                pred_value = float(np.asarray(pred).reshape(-1)[0])
                predictions[key] = pred_value
                if not np.isfinite(pred_value):
                    errors.append(f"{key} prediction is not finite")
            except Exception as exc:
                errors.append(f"{key} joblib/predict failed: {exc!r}")

    trained = [
        model for model in train_report.get("models", [])
        if isinstance(model, dict) and model.get("model_status") == "trained"
    ]
    if len(trained) < len(WINDOWS) * len(SIDES):
        errors.append(f"trained model count {len(trained)} < {len(WINDOWS) * len(SIDES)}")

    validation = {
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "model_paths": model_paths,
        "sample_counts": sample_counts,
        "dataset_nan_checks": dataset_nan_checks,
        "local_predictions": predictions,
        "sklearn_version": sklearn_version,
    }
    return len(errors) == 0, validation


def validate_snapshot_refresh(symbol: str) -> dict[str, Any]:
    payload = refresh_features_only(symbol)
    statuses = {
        f"{window}d": load_turbo_snapshot_status(turbo_snapshot_path(window, symbol), include_sample_count=True)
        for window in WINDOWS
    }
    is_fresh = all(bool(status.get("is_fresh")) for status in statuses.values())
    return {
        "refresh_payload": payload,
        "snapshot_statuses": statuses,
        "is_fresh": is_fresh,
    }


def active_manifest_path(symbol: str) -> Path:
    return turbo_symbol_model_dir(symbol) / "active_manifest.json"


def active_manifest_is_fresh(symbol: str) -> bool:
    manifest_path = active_manifest_path(symbol)
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if manifest.get("validation_status") != "passed":
        return False
    for window in WINDOWS:
        status = load_turbo_snapshot_status(turbo_snapshot_path(window, symbol), include_sample_count=False)
        if not status.get("is_fresh"):
            return False
    return True


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def promote_candidate(
    symbol: str,
    candidate_dir: Path,
    train_report_path: str | None,
    validation: dict[str, Any],
    started_stamp: str,
    report_path: Path,
    disable_phase_o_overlay: bool = False,
) -> dict[str, Any]:
    symbol_dir = turbo_symbol_model_dir(symbol)
    active_dir = symbol_dir / "active"
    active_tmp = symbol_dir / f"active_tmp_{started_stamp}"
    backup_dir = symbol_dir / "backups" / started_stamp
    manifest_path = active_manifest_path(symbol)

    if active_tmp.exists():
        shutil.rmtree(active_tmp)
    shutil.copytree(candidate_dir, active_tmp)

    if active_dir.exists():
        backup_dir.parent.mkdir(parents=True, exist_ok=True)
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        os.replace(active_dir, backup_dir)
        if manifest_path.exists():
            shutil.copy2(manifest_path, backup_dir / "active_manifest.json")
    else:
        backup_dir = None

    os.replace(active_tmp, active_dir)
    model_paths = {
        f"{side}_{window}d": str(expected_model_path(active_dir, side, window))
        for window in WINDOWS
        for side in SIDES
    }
    manifest = {
        "symbol": symbol,
        "version": f"{TURBO_VERSION}-{started_stamp}",
        "created_at": started_stamp,
        "promoted_at": utc_iso(),
        "model_paths": model_paths,
        "windows": list(WINDOWS),
        "sklearn_version": sklearn_version,
        "feature_version": "aegis_turbo_recent_dataset_v1",
        "training_report_path": train_report_path,
        "scheduled_retrain_report_path": str(report_path),
        "validation_status": "passed" if validation.get("passed") else "failed",
    }
    overlay_applied = False
    overlay_reason = "disabled_by_cli" if disable_phase_o_overlay else "symbol_not_in_phase_o"
    overlay_error = None
    phase_o_model_path = None
    link_avoid_only = False
    if not disable_phase_o_overlay and symbol in PHASE_O_SYMBOLS:
        try:
            manifest = apply_phase_o_overlay_to_active_manifest(symbol, manifest, symbol_dir.parent)
            overlay_errors = validate_phase_o_overlay(symbol, manifest, symbol_dir.parent)
            if overlay_errors:
                raise ValueError(f"Phase O overlay invalid for {symbol}: {overlay_errors}")
            overlay_applied = True
            overlay_reason = "persistent_phase_o_overlay_applied"
            phase_o_model_path = manifest.get("phase_o_model_path")
            link_avoid_only = bool(manifest.get("phase_o_avoid_only"))
        except Exception as exc:
            overlay_reason = "phase_o_overlay_failed_base_manifest_preserved"
            overlay_error = repr(exc)
    atomic_write_json(manifest_path, manifest)
    return {
        "active_dir": str(active_dir),
        "backup_dir": str(backup_dir) if backup_dir is not None else None,
        "active_manifest_path": str(manifest_path),
        "active_manifest": manifest,
        "phase_o_overlay_applied": overlay_applied,
        "phase_o_overlay_reason": overlay_reason,
        "phase_o_overlay_error": overlay_error,
        "phase_o_model_path": phase_o_model_path,
        "link_avoid_only": link_avoid_only,
    }


def run_symbol(symbol: str, args: argparse.Namespace, started_stamp: str, report_path: Path) -> dict[str, Any]:
    symbol_started = time.time()
    symbol = normalize_turbo_symbol(symbol)
    symbol_dir = turbo_symbol_model_dir(symbol)
    candidate_dir = symbol_dir / "candidates" / started_stamp
    result: dict[str, Any] = {
        "symbol": symbol,
        "started_at": utc_iso(),
        "candles_updated": False,
        "rows": None,
        "last_candle_ts": None,
        "training_duration_seconds": None,
        "candidate_dir": str(candidate_dir),
        "validation_passed": False,
        "promoted": False,
        "active_dir": None,
        "backup_dir": None,
        "errors": [],
        "warnings": [],
    }

    if args.skip_existing_fresh and active_manifest_is_fresh(symbol):
        result.update({
            "skipped": True,
            "skip_reason": "active_manifest_and_snapshots_fresh",
            "finished_at": utc_iso(),
            "duration_seconds": round(time.time() - symbol_started, 3),
        })
        return result

    try:
        candle_report = update_candles(symbol)
        result["candles_updated"] = True
        result["candle_update"] = candle_report
        result["rows"] = candle_report.get("rows")
        result["last_candle_ts"] = candle_report.get("last_candle_ts")

        train_started = time.time()
        train_report = train_recent_edge_models(symbol, tuple(WINDOWS), output_dir=candidate_dir)
        result["training_duration_seconds"] = round(time.time() - train_started, 3)
        result["train_report_path"] = train_report.get("report_path")
        result["train_report"] = train_report

        validation_passed, validation = validate_candidate_models(symbol, candidate_dir, train_report)
        result["validation"] = validation
        result["validation_passed"] = validation_passed
        result["errors"].extend(validation.get("errors", []))
        result["warnings"].extend(validation.get("warnings", []))

        snapshot_validation = validate_snapshot_refresh(symbol)
        result["snapshot_validation"] = snapshot_validation
        if not snapshot_validation.get("is_fresh"):
            result["validation_passed"] = False
            result["errors"].append("snapshot refresh did not produce fresh snapshots")

        if result["validation_passed"] and args.promote_if_valid:
            promotion = promote_candidate(
                symbol,
                candidate_dir,
                train_report.get("report_path"),
                validation,
                started_stamp,
                report_path,
                disable_phase_o_overlay=args.disable_phase_o_overlay,
            )
            result.update(promotion)
            if promotion.get("phase_o_overlay_error"):
                result["errors"].append(f"phase_o_overlay_failed: {promotion['phase_o_overlay_error']}")
            result["promoted"] = True
    except Exception as exc:
        result["errors"].append(repr(exc))
    finally:
        result["finished_at"] = utc_iso()
        result["duration_seconds"] = round(time.time() - symbol_started, 3)
    return result


def write_markdown_report(path: Path, report: dict[str, Any]) -> Path:
    md_path = path.with_suffix(".md")
    lines = [
        f"# Aegis Turbo Scheduled Retrain {report['started_at']}",
        "",
        f"- Mode: `{report['mode']}`",
        f"- Promote if valid: `{report['promote_if_valid']}`",
        f"- Phase O overlay enabled: `{report.get('phase_o_overlay_enabled')}`",
        f"- Failed Phase O overlays: `{', '.join(report.get('failed_overlay_symbols', [])) or 'none'}`",
        f"- Duration: `{report.get('duration_seconds')}` seconds",
        f"- Succeeded: `{', '.join(report.get('symbols_succeeded', [])) or 'none'}`",
        f"- Failed: `{', '.join(report.get('symbols_failed', [])) or 'none'}`",
        f"- Promoted: `{', '.join(report.get('promoted_symbols', [])) or 'none'}`",
        "",
        "## Per Symbol",
        "",
    ]
    for symbol, payload in report.get("per_symbol", {}).items():
        lines.extend([
            f"### {symbol}",
            "",
            f"- Validation passed: `{payload.get('validation_passed')}`",
            f"- Promoted: `{payload.get('promoted')}`",
            f"- Rows: `{payload.get('rows')}`",
            f"- Last candle: `{payload.get('last_candle_ts')}`",
            f"- Candidate: `{payload.get('candidate_dir')}`",
            f"- Active: `{payload.get('active_dir')}`",
            f"- Backup: `{payload.get('backup_dir')}`",
            f"- Errors: `{payload.get('errors') or []}`",
            f"- Warnings: `{payload.get('warnings') or []}`",
            "",
        ])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def build_report(args: argparse.Namespace, symbols: list[str], started_at: str) -> tuple[dict[str, Any], Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"turbo_retrain_{started_at}.json"
    report = {
        "schema_version": "aegis_turbo_scheduled_retrain_v1",
        "started_at": utc_iso(),
        "started_stamp": started_at,
        "mode": args.mode,
        "promote_if_valid": bool(args.promote_if_valid),
        "skip_existing_fresh": bool(args.skip_existing_fresh),
        "phase_o_overlay_enabled": not bool(args.disable_phase_o_overlay),
        "failed_overlay_symbols": [],
        "symbols_requested": symbols,
        "symbols_succeeded": [],
        "symbols_failed": [],
        "promoted_symbols": [],
        "failed_symbols": [],
        "per_symbol": {},
    }
    return report, report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols")
    parser.add_argument("--mode", choices=("safe",), default="safe")
    parser.add_argument("--promote-if-valid", action="store_true")
    parser.add_argument("--skip-existing-fresh", type=parse_bool, default=False)
    parser.add_argument("--max-symbols-per-run", type=int)
    parser.add_argument("--disable-phase-o-overlay", action="store_true", help="Disable persistent Phase O overlay reapplication")
    args = parser.parse_args()

    symbols = parse_symbols(args.symbols)
    if args.max_symbols_per_run is not None and args.max_symbols_per_run > 0:
        symbols = symbols[: args.max_symbols_per_run]
    started_monotonic = time.time()
    started_stamp = utc_stamp()
    report, report_path = build_report(args, symbols, started_stamp)

    with retrain_lock():
        for symbol in symbols:
            payload = run_symbol(symbol, args, started_stamp, report_path)
            report["per_symbol"][symbol] = payload
            if payload.get("validation_passed") and not payload.get("errors"):
                report["symbols_succeeded"].append(symbol)
            else:
                report["symbols_failed"].append(symbol)
                report["failed_symbols"].append(symbol)
            if payload.get("promoted"):
                report["promoted_symbols"].append(symbol)
            if payload.get("phase_o_overlay_error"):
                report["failed_overlay_symbols"].append(symbol)

        report["finished_at"] = utc_iso()
        report["duration_seconds"] = round(time.time() - started_monotonic, 3)
        atomic_write_json(report_path, report)
        md_path = write_markdown_report(report_path, report)
        report["markdown_report_path"] = str(md_path)
        atomic_write_json(report_path, report)

    print(json.dumps(report, indent=2, sort_keys=True))
    if report["symbols_failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
