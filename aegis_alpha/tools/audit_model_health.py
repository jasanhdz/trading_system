from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aegis_alpha.config import REPO_ROOT
from aegis_alpha.turbo.config import DEFAULT_TURBO_CONFIG
from aegis_alpha.turbo.snapshot_utils import (
    TURBO_MAX_FEATURE_AGE_SECONDS,
    load_turbo_snapshot_status,
    normalize_turbo_symbol,
    turbo_snapshot_path,
    turbo_symbol_model_dir,
)


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

REPORT_COLUMNS = (
    "symbol",
    "operationalStatus",
    "directionalStatus",
    "modelExists",
    "manifestExists",
    "snapshotSource",
    "snapshotFresh",
    "featureAgeMin",
    "latestRetrainOk",
    "apiPredictOk",
    "validationScoreAvg",
    "rmseAvg",
    "directionalMetricsAvailable",
    "directionalConfidence",
    "operationalWarnings",
    "directionalWarnings",
    "legacyWarnings",
    "recommendedAction",
)


@dataclass
class ApiPredictResult:
    ok: bool | None = None
    error: str | None = None
    latency_ms: float | None = None


@dataclass
class ModelHealthRow:
    symbol: str
    operationalStatus: str
    directionalStatus: str
    modelExists: bool
    manifestExists: bool
    snapshotSource: str | None
    snapshotFresh: bool
    featureAgeMin: float | None
    latestRetrainOk: bool
    apiPredictOk: bool | None
    validationScoreAvg: float | None
    rmseAvg: float | None
    directionalMetricsAvailable: bool
    directionalConfidence: float | None = None
    operationalWarnings: list[str] = field(default_factory=list)
    directionalWarnings: list[str] = field(default_factory=list)
    legacyWarnings: list[str] = field(default_factory=list)
    recommendedAction: list[str] = field(default_factory=list)
    directionalPhase2Actions: list[str] = field(default_factory=list)
    directionalMetricsSummary: dict[str, Any] | None = None
    directionalReportPath: str | None = None
    modelPaths: list[str] = field(default_factory=list)
    manifestPath: str | None = None
    snapshotPath: str | None = None
    selectedSnapshotSource: str | None = None
    selectedSnapshotLookbackDays: int | None = None
    featureTimestamp: str | None = None
    snapshotMtime: str | None = None
    sampleCount: int | None = None
    lastTs: str | None = None
    trainStart: str | None = None
    trainEnd: str | None = None
    featureCount: int | None = None
    modelVersion: str | None = None
    validationStatus: str | None = None
    promotedAt: str | None = None
    trainingReportPath: str | None = None
    apiPredictError: str | None = None
    apiLatencyMs: float | None = None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    candidates = (text, text.replace("Z", "+00:00"))
    for candidate in candidates:
        try:
            parsed = datetime.fromisoformat(candidate)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            continue
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y%m%dT%H%M%SZ"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result or result in {float("inf"), float("-inf")}:
        return None
    return result


def iso_from_mtime(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    except OSError:
        return None


def list_model_paths(symbol: str, manifest: dict[str, Any] | None) -> list[Path]:
    model_paths = manifest.get("model_paths") if isinstance(manifest, dict) else None
    if isinstance(model_paths, dict):
        paths = [Path(str(path)) for path in model_paths.values()]
        return [path if path.is_absolute() else REPO_ROOT / path for path in paths]

    active_dir = turbo_symbol_model_dir(symbol) / "active"
    result: list[Path] = []
    for lookback_days in DEFAULT_TURBO_CONFIG.lookback_days:
        for side in ("long", "short"):
            result.append(active_dir / f"turbo_{side}_edge_{int(lookback_days)}d_v010.joblib")
    return result


def latest_retrain_report(log_dir: Path | None = None) -> dict[str, Any] | None:
    report_dir = log_dir or (REPO_ROOT / "aegis_alpha" / "logs" / "turbo_retrain")
    reports = sorted(report_dir.glob("turbo_retrain_*.json"))
    if not reports:
        return None
    latest_path = reports[-1]
    data = read_json(latest_path)
    if data is None:
        return {"path": str(latest_path), "read_error": True}
    data["path"] = str(latest_path)
    return data


def latest_retrain_ok_for_symbol(latest: dict[str, Any] | None, symbol: str) -> bool:
    if not latest:
        return False
    failed = latest.get("failed_symbols")
    if isinstance(failed, list) and symbol in {str(item).upper() for item in failed}:
        return False
    promoted = latest.get("promoted_symbols")
    if isinstance(promoted, list):
        return symbol in {str(item).upper() for item in promoted}
    return not bool(failed)


def snapshot_source(path: str | None, symbol: str) -> str | None:
    if not path:
        return None
    normalized = normalize_turbo_symbol(symbol)
    snapshot_path = Path(path)
    symbol_dir = DEFAULT_TURBO_CONFIG.data_dir / "turbo" / normalized
    try:
        snapshot_path.relative_to(symbol_dir)
        return "symbol"
    except ValueError:
        pass
    legacy_names = {f"turbo_recent_{int(days)}d.npz" for days in DEFAULT_TURBO_CONFIG.lookback_days}
    if snapshot_path.parent == DEFAULT_TURBO_CONFIG.data_dir and snapshot_path.name in legacy_names:
        return "legacy_global"
    return "unknown"


def select_runtime_snapshot(symbol: str) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for lookback_days in DEFAULT_TURBO_CONFIG.lookback_days:
        path = turbo_snapshot_path(int(lookback_days), symbol)
        status = load_turbo_snapshot_status(path, include_sample_count=True)
        status["lookback_days"] = int(lookback_days)
        status["source"] = snapshot_source(status.get("path"), symbol)
        candidates.append(status)

    existing = [candidate for candidate in candidates if candidate.get("exists")]
    if not existing:
        return {
            "path": None,
            "source": None,
            "freshness": {
                "exists": False,
                "is_fresh": False,
                "feature_age_seconds": None,
                "feature_timestamp": None,
                "error": "missing_all_snapshots",
            },
            "candidates": candidates,
        }

    existing.sort(
        key=lambda item: (
            item.get("feature_timestamp") is not None,
            item.get("feature_timestamp") or "",
            item.get("snapshot_mtime") or "",
            -int(item.get("lookback_days") or 0),
        ),
        reverse=True,
    )
    selected = existing[0]
    return {
        "path": selected.get("path"),
        "source": selected.get("source"),
        "freshness": selected,
        "candidates": candidates,
    }


def training_report_from_manifest(manifest: dict[str, Any] | None) -> Path | None:
    if not isinstance(manifest, dict):
        return None
    raw = manifest.get("training_report_path")
    if not raw:
        return None
    path = Path(str(raw))
    return path if path.is_absolute() else REPO_ROOT / path


def training_summary(report: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    models = report.get("models") if isinstance(report.get("models"), list) else []
    dataset_reports = report.get("dataset_reports") if isinstance(report.get("dataset_reports"), list) else []

    validation_scores = [value for value in (finite_float(item.get("validation_score")) for item in models if isinstance(item, dict)) if value is not None]
    rmses = [value for value in (finite_float(item.get("validation_rmse")) for item in models if isinstance(item, dict)) if value is not None]
    directional_metric_keys = (
        "validation_long_precision",
        "validation_short_precision",
        "validation_neutral_precision",
        "long_precision",
        "short_precision",
        "neutral_precision",
    )
    directional_metrics_available = any(
        any(finite_float(item.get(key)) is not None for key in directional_metric_keys)
        for item in models
        if isinstance(item, dict)
    )

    starts = [parse_dt(item.get("date_start")) for item in dataset_reports if isinstance(item, dict)]
    ends = [parse_dt(item.get("date_end")) for item in dataset_reports if isinstance(item, dict)]
    starts = [item for item in starts if item is not None]
    ends = [item for item in ends if item is not None]
    feature_counts = [int(item.get("feature_count")) for item in dataset_reports if isinstance(item, dict) and isinstance(item.get("feature_count"), int)]
    sample_counts = [int(item.get("sample_count")) for item in dataset_reports if isinstance(item, dict) and isinstance(item.get("sample_count"), int)]

    return {
        "validationScoreAvg": mean(validation_scores),
        "rmseAvg": mean(rmses),
        "directionalMetricsAvailable": directional_metrics_available,
        "trainStart": min(starts).isoformat() if starts else None,
        "trainEnd": max(ends).isoformat() if ends else None,
        "featureCount": max(feature_counts) if feature_counts else None,
        "sampleCount": max(sample_counts) if sample_counts else None,
    }


def status_counts(rows: list[ModelHealthRow], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(getattr(row, key))
        counts[value] = counts.get(value, 0) + 1
    return counts


def classify_directional(summary: dict[str, Any], directional_warnings: list[str]) -> str:
    if not summary.get("directionalMetricsAvailable"):
        directional_warnings.append("no_class_precision_metrics_long_short_neutral")
        return "UNKNOWN"
    if directional_warnings:
        return "YELLOW"
    return "GREEN"


def classify_operational(
    *,
    model_exists: bool,
    manifest_exists: bool,
    snapshot_fresh: bool,
    latest_retrain_ok: bool,
    operational_warnings: list[str],
) -> str:
    critical_missing = not model_exists or not manifest_exists or not snapshot_fresh
    if critical_missing:
        return "RED"
    if operational_warnings:
        return "YELLOW"
    if not latest_retrain_ok:
        return "YELLOW"
    return "GREEN"


def recommended_actions(row: ModelHealthRow) -> list[str]:
    actions: list[str] = []
    if row.operationalStatus == "GREEN" and row.directionalStatus == "UNKNOWN":
        actions.append("operational_ok_waiting_directional_metrics")
    if row.snapshotSource == "legacy_global":
        actions.append("document_or_migrate_legacy_global_snapshot")
    if "weakest_relative_metrics" in row.operationalWarnings or "weakest_relative_metrics" in row.directionalWarnings:
        actions.append("reduce_confidence_until_directional_metrics")
    if row.legacyWarnings:
        actions.append("legacy_check_document_or_remove_from_turbo_health")
    if row.operationalStatus == "RED":
        actions.append("fix_missing_or_stale_runtime_artifacts")
    actions.extend(row.directionalPhase2Actions)
    if not actions:
        actions.append("no_action_phase1")
    return sorted(set(actions))


def load_directional_report(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    report_path = Path(path)
    data = read_json(report_path)
    if data is not None:
        data["_path"] = str(report_path)
    return data


def apply_directional_report(rows: list[ModelHealthRow], directional_report: dict[str, Any] | None) -> None:
    if not directional_report:
        return
    summaries = directional_report.get("symbolSummaries")
    if not isinstance(summaries, list):
        return
    by_symbol = {str(item.get("symbol", "")).upper(): item for item in summaries if isinstance(item, dict)}
    for row in rows:
        summary = by_symbol.get(row.symbol)
        if not summary:
            continue
        row.directionalStatus = str(summary.get("directionalStatus") or row.directionalStatus)
        row.directionalConfidence = finite_float(summary.get("directionalConfidence"))
        row.directionalMetricsAvailable = True
        row.directionalWarnings = [warning for warning in row.directionalWarnings if warning != "no_class_precision_metrics_long_short_neutral"]
        for warning in summary.get("directionalWarnings") or []:
            warning_text = str(warning)
            if warning_text not in row.directionalWarnings:
                row.directionalWarnings.append(warning_text)
        row.directionalPhase2Actions = [str(action) for action in (summary.get("recommendedAction") or [])]
        row.directionalMetricsSummary = {
            "sampleCount": summary.get("sampleCount"),
            "scoreCalibration": summary.get("scoreCalibration"),
            "longCount": (summary.get("long") or {}).get("count") if isinstance(summary.get("long"), dict) else None,
            "longExpectancy60m": (summary.get("long") or {}).get("netExpectancy60m") if isinstance(summary.get("long"), dict) else None,
            "longHit8BeforeMinus5": (summary.get("long") or {}).get("hit8BeforeMinus5") if isinstance(summary.get("long"), dict) else None,
            "shortCount": (summary.get("short") or {}).get("count") if isinstance(summary.get("short"), dict) else None,
            "shortExpectancy60m": (summary.get("short") or {}).get("netExpectancy60m") if isinstance(summary.get("short"), dict) else None,
            "shortHit8BeforeMinus5": (summary.get("short") or {}).get("hit8BeforeMinus5") if isinstance(summary.get("short"), dict) else None,
        }
        row.directionalReportPath = directional_report.get("_path")


def ping_predict(symbol: str, base_url: str, timeout_seconds: float = 2.0) -> ApiPredictResult:
    url = base_url.rstrip("/") + "/ml-v2/predict"
    payload = json.dumps({"symbol": symbol}).encode("utf-8")
    request = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read()
            latency_ms = (time.perf_counter() - started) * 1000
            data = json.loads(body.decode("utf-8"))
            valid = isinstance(data, dict) and str(data.get("symbol", "")).upper() == symbol
            return ApiPredictResult(ok=bool(valid), latency_ms=round(latency_ms, 2), error=None if valid else "invalid_predict_payload")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        latency_ms = (time.perf_counter() - started) * 1000
        return ApiPredictResult(ok=False, latency_ms=round(latency_ms, 2), error=str(exc))


def audit_symbols(
    symbols: list[str] | tuple[str, ...] = EXPECTED_SYMBOLS,
    *,
    check_api: bool = False,
    api_base_url: str | None = None,
    api_timeout_seconds: float = 5.0,
    directional_report_path: str | Path | None = None,
) -> dict[str, Any]:
    normalized_symbols = [normalize_turbo_symbol(symbol) for symbol in symbols]
    latest = latest_retrain_report()
    rows: list[ModelHealthRow] = []

    base_url = api_base_url or os.environ.get("ML_SERVICE_URL") or "http://127.0.0.1:8001"

    for symbol in normalized_symbols:
        manifest_path = turbo_symbol_model_dir(symbol) / "active_manifest.json"
        manifest = read_json(manifest_path)
        manifest_exists = manifest is not None
        model_paths = list_model_paths(symbol, manifest)
        model_exists = bool(model_paths) and all(path.exists() and path.stat().st_size > 0 for path in model_paths)
        snapshot = select_runtime_snapshot(symbol)
        freshness = snapshot.get("freshness") or {}
        source = snapshot.get("source")
        snapshot_fresh = bool(freshness.get("exists")) and bool(freshness.get("is_fresh"))
        feature_age_seconds = finite_float(freshness.get("feature_age_seconds"))
        training_path = training_report_from_manifest(manifest)
        training_report = read_json(training_path) if training_path else None
        summary = training_summary(training_report)

        operational_warnings: list[str] = []
        directional_warnings: list[str] = []
        legacy_warnings: list[str] = []

        if source == "legacy_global":
            operational_warnings.append("runtime_uses_legacy_global_snapshots")
        if not latest_retrain_ok_for_symbol(latest, symbol) and model_exists:
            operational_warnings.append("latest_retrain_not_confirmed_but_active_model_exists")
        if not model_exists:
            operational_warnings.append("missing_or_empty_model_files")
        if not manifest_exists:
            operational_warnings.append("missing_active_manifest")
        if not snapshot_fresh:
            operational_warnings.append("missing_or_stale_runtime_snapshot")

        legacy_path = REPO_ROOT / "models" / "v2_ensemble" / symbol / "metadata.json"
        if not legacy_path.exists():
            legacy_warnings.append("legacy_ConfigLoader_v2_ensemble_metadata_missing")

        directional_status = classify_directional(summary, directional_warnings)
        api_result = ping_predict(symbol, base_url, timeout_seconds=api_timeout_seconds) if check_api else ApiPredictResult()

        row = ModelHealthRow(
            symbol=symbol,
            operationalStatus="UNKNOWN",
            directionalStatus=directional_status,
            modelExists=model_exists,
            manifestExists=manifest_exists,
            snapshotSource=source,
            snapshotFresh=snapshot_fresh,
            featureAgeMin=round(feature_age_seconds / 60.0, 1) if feature_age_seconds is not None else None,
            latestRetrainOk=latest_retrain_ok_for_symbol(latest, symbol),
            apiPredictOk=api_result.ok,
            validationScoreAvg=summary.get("validationScoreAvg"),
            rmseAvg=summary.get("rmseAvg"),
            directionalMetricsAvailable=bool(summary.get("directionalMetricsAvailable")),
            operationalWarnings=operational_warnings,
            directionalWarnings=directional_warnings,
            legacyWarnings=legacy_warnings,
            modelPaths=[str(path) for path in model_paths],
            manifestPath=str(manifest_path),
            snapshotPath=str(snapshot.get("path")) if snapshot.get("path") else None,
            selectedSnapshotSource=source,
            selectedSnapshotLookbackDays=freshness.get("lookback_days") if isinstance(freshness.get("lookback_days"), int) else None,
            featureTimestamp=freshness.get("feature_timestamp"),
            snapshotMtime=freshness.get("snapshot_mtime"),
            sampleCount=int(summary.get("sampleCount") or freshness.get("sample_count") or 0) or None,
            lastTs=freshness.get("last_ts"),
            trainStart=summary.get("trainStart"),
            trainEnd=summary.get("trainEnd"),
            featureCount=summary.get("featureCount"),
            modelVersion=manifest.get("version") if isinstance(manifest, dict) else None,
            validationStatus=manifest.get("validation_status") if isinstance(manifest, dict) else None,
            promotedAt=manifest.get("promoted_at") if isinstance(manifest, dict) else None,
            trainingReportPath=str(training_path) if training_path else None,
            apiPredictError=api_result.error,
            apiLatencyMs=api_result.latency_ms,
        )
        rows.append(row)

    score_values = [(row.symbol, row.validationScoreAvg) for row in rows if row.validationScoreAvg is not None]
    rmse_values = [(row.symbol, row.rmseAvg) for row in rows if row.rmseAvg is not None]
    weakest_symbols: set[str] = set()
    if len(score_values) > 1:
        weakest_symbols.add(min(score_values, key=lambda item: item[1])[0])
    if len(rmse_values) > 1:
        weakest_symbols.add(max(rmse_values, key=lambda item: item[1])[0])

    directional_report = load_directional_report(directional_report_path)
    apply_directional_report(rows, directional_report)

    for row in rows:
        if row.symbol in weakest_symbols:
            if "weakest_relative_metrics" not in row.operationalWarnings:
                row.operationalWarnings.append("weakest_relative_metrics")
            if "weakest_relative_metrics" not in row.directionalWarnings:
                row.directionalWarnings.append("weakest_relative_metrics")
        row.operationalStatus = classify_operational(
            model_exists=row.modelExists,
            manifest_exists=row.manifestExists,
            snapshot_fresh=row.snapshotFresh,
            latest_retrain_ok=row.latestRetrainOk,
            operational_warnings=row.operationalWarnings,
        )
        row.recommendedAction = recommended_actions(row)

    latest_summary = None
    if latest:
        latest_summary = {
            "path": latest.get("path"),
            "started_at": latest.get("started_at"),
            "finished_at": latest.get("finished_at"),
            "failed_symbols": latest.get("failed_symbols"),
            "promoted_symbols": latest.get("promoted_symbols"),
            "mode": latest.get("mode"),
        }

    return {
        "generated_at": utc_now().isoformat(),
        "symbols_expected": normalized_symbols,
        "max_feature_age_seconds": TURBO_MAX_FEATURE_AGE_SECONDS,
        "check_api": check_api,
        "api_base_url": base_url if check_api else None,
        "api_timeout_seconds": api_timeout_seconds if check_api else None,
        "directional_report_path": str(directional_report_path) if directional_report_path else None,
        "latest_retrain": latest_summary,
        "operationalStatusCounts": status_counts(rows, "operationalStatus"),
        "directionalStatusCounts": status_counts(rows, "directionalStatus"),
        "rows": [asdict(row) for row in rows],
    }


def csv_value(value: Any) -> Any:
    if isinstance(value, list):
        return "|".join(str(item) for item in value)
    return value


def write_csv(report: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        for row in report["rows"]:
            writer.writerow({column: csv_value(row.get(column)) for column in REPORT_COLUMNS})


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Aegis Turbo Model Health Phase 1",
        "",
        f"Generated: {report['generated_at']}",
        f"API check: {report['check_api']}",
        f"Operational counts: {report['operationalStatusCounts']}",
        f"Directional counts: {report['directionalStatusCounts']}",
        "",
        "## Latest Retrain",
        "",
        "```json",
        json.dumps(report.get("latest_retrain"), indent=2, sort_keys=True),
        "```",
        "",
        "## Symbols",
        "",
        "| Symbol | Operational | Directional | Snapshot | Fresh | Feature age | Validation score | RMSE | Warnings | Recommended |",
        "|---|---:|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for row in report["rows"]:
        warnings = []
        for key in ("operationalWarnings", "directionalWarnings", "legacyWarnings"):
            values = row.get(key) or []
            if values:
                warnings.append(f"{key}=" + ",".join(values))
        lines.append(
            "| {symbol} | {operationalStatus} | {directionalStatus} | {snapshotSource} | {snapshotFresh} | {featureAgeMin} | {validationScoreAvg} | {rmseAvg} | {warnings} | {recommendedAction} |".format(
                symbol=row.get("symbol"),
                operationalStatus=row.get("operationalStatus"),
                directionalStatus=row.get("directionalStatus"),
                snapshotSource=row.get("snapshotSource"),
                snapshotFresh=row.get("snapshotFresh"),
                featureAgeMin=row.get("featureAgeMin"),
                validationScoreAvg=row.get("validationScoreAvg"),
                rmseAvg=row.get("rmseAvg"),
                warnings="<br>".join(warnings) if warnings else "",
                recommendedAction=", ".join(row.get("recommendedAction") or []),
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_reports(report: dict[str, Any], out_dir: Path, timestamp: str) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"aegis_model_health_phase1_report_{timestamp}.md"
    json_path = out_dir / f"aegis_model_health_phase1_report_{timestamp}.json"
    csv_path = out_dir / f"aegis_model_health_phase1_summary_{timestamp}.csv"

    md_path.write_text("", encoding="utf-8")
    write_markdown(report, md_path)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(report, csv_path)
    return {"md": str(md_path), "json": str(json_path), "csv": str(csv_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Aegis Turbo model operational health separately from directional confidence.")
    parser.add_argument("--symbols", default=",".join(EXPECTED_SYMBOLS), help="Comma-separated symbols to audit.")
    parser.add_argument("--out-dir", default="/home/jasan/Develop", help="Directory for report outputs.")
    parser.add_argument("--check-api", "--ping-predict", action="store_true", dest="check_api", help="Optionally call /ml-v2/predict for each symbol.")
    parser.add_argument("--api-url", default=os.environ.get("ML_SERVICE_URL", "http://127.0.0.1:8001"), help="Base URL for --check-api.")
    parser.add_argument("--api-timeout-seconds", type=float, default=5.0, help="Timeout per /ml-v2/predict call when --check-api is enabled.")
    parser.add_argument("--directional-report", default=None, help="Optional Phase 2 directional metrics JSON to merge into this report.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
    report = audit_symbols(
        symbols,
        check_api=bool(args.check_api),
        api_base_url=args.api_url,
        api_timeout_seconds=float(args.api_timeout_seconds),
        directional_report_path=args.directional_report,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    paths = write_reports(report, Path(args.out_dir), timestamp)
    print(json.dumps({"paths": paths, "operationalStatusCounts": report["operationalStatusCounts"], "directionalStatusCounts": report["directionalStatusCounts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
