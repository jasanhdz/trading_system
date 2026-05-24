from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aegis_alpha.tools import audit_model_health as audit


def _write_symbol_artifacts(tmp_path: Path, symbol: str, *, score: float = -0.001, rmse: float = 0.002) -> Path:
    symbol_dir = tmp_path / "models" / symbol
    active = symbol_dir / "active"
    active.mkdir(parents=True)
    model_paths: dict[str, str] = {}
    for lookback_days in (7, 14, 30):
        for side in ("long", "short"):
            key = f"{side}_{lookback_days}d"
            path = active / f"turbo_{side}_edge_{lookback_days}d_v010.joblib"
            path.write_bytes(b"model")
            model_paths[key] = str(path)

    report_path = tmp_path / f"{symbol}_train_report.json"
    report_path.write_text(
        json.dumps(
            {
                "symbol": symbol,
                "models": [
                    {"validation_score": score, "validation_rmse": rmse, "side": "long", "lookback_days": 7},
                    {"validation_score": score, "validation_rmse": rmse, "side": "short", "lookback_days": 7},
                ],
                "dataset_reports": [
                    {
                        "date_start": "2026-05-01 00:00:00",
                        "date_end": "2026-05-23 10:00:00",
                        "feature_count": 168,
                        "sample_count": 1000,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "symbol": symbol,
        "version": "0.1.0-test",
        "validation_status": "passed",
        "promoted_at": "2026-05-23T12:00:00+00:00",
        "training_report_path": str(report_path),
        "model_paths": model_paths,
    }
    (symbol_dir / "active_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return symbol_dir


def _patch_common(monkeypatch, tmp_path: Path, snapshot_by_symbol: dict[str, dict], scores: dict[str, tuple[float, float]] | None = None) -> None:
    scores = scores or {}
    dirs: dict[str, Path] = {}
    for symbol in snapshot_by_symbol:
        score, rmse = scores.get(symbol, (-0.001, 0.002))
        dirs[symbol] = _write_symbol_artifacts(tmp_path, symbol, score=score, rmse=rmse)

    monkeypatch.setattr(audit, "turbo_symbol_model_dir", lambda symbol: dirs[str(symbol).upper()])
    monkeypatch.setattr(
        audit,
        "latest_retrain_report",
        lambda: {
            "started_at": "2026-05-23T12:20:00+00:00",
            "finished_at": "2026-05-23T12:48:00+00:00",
            "failed_symbols": [],
            "promoted_symbols": list(snapshot_by_symbol.keys()),
        },
    )
    monkeypatch.setattr(audit, "select_runtime_snapshot", lambda symbol: snapshot_by_symbol[str(symbol).upper()])


def _snapshot(symbol: str, *, source: str = "symbol", fresh: bool = True) -> dict:
    return {
        "path": f"/tmp/{symbol}/turbo_recent_30d.npz",
        "source": source,
        "freshness": {
            "exists": True,
            "is_fresh": fresh,
            "feature_age_seconds": 120,
            "feature_timestamp": "2026-05-24T00:05:00+00:00",
            "snapshot_mtime": "2026-05-24T00:06:00+00:00",
            "sample_count": 1000,
            "last_ts": "2026-05-23 22:05:00",
            "lookback_days": 30,
        },
    }


def test_eth_legacy_fallback_is_operational_yellow_not_red(monkeypatch, tmp_path):
    _patch_common(monkeypatch, tmp_path, {"ETHUSDT": _snapshot("ETHUSDT", source="legacy_global")})

    report = audit.audit_symbols(["ETHUSDT"])
    row = report["rows"][0]

    assert row["operationalStatus"] == "YELLOW"
    assert row["directionalStatus"] == "UNKNOWN"
    assert row["snapshotSource"] == "legacy_global"
    assert "runtime_uses_legacy_global_snapshots" in row["operationalWarnings"]
    assert "document_or_migrate_legacy_global_snapshot" in row["recommendedAction"]


def test_missing_snapshots_without_fallback_is_operational_red(monkeypatch, tmp_path):
    missing = {
        "path": None,
        "source": None,
        "freshness": {"exists": False, "is_fresh": False, "feature_age_seconds": None, "error": "missing_all_snapshots"},
    }
    _patch_common(monkeypatch, tmp_path, {"ADAUSDT": missing})

    report = audit.audit_symbols(["ADAUSDT"])
    row = report["rows"][0]

    assert row["operationalStatus"] == "RED"
    assert "missing_or_stale_runtime_snapshot" in row["operationalWarnings"]
    assert "fix_missing_or_stale_runtime_artifacts" in row["recommendedAction"]


def test_legacy_v2_missing_is_legacy_warning_only_not_operational_red(monkeypatch, tmp_path):
    _patch_common(monkeypatch, tmp_path, {"BTCUSDT": _snapshot("BTCUSDT")})

    report = audit.audit_symbols(["BTCUSDT"])
    row = report["rows"][0]

    assert row["operationalStatus"] == "GREEN"
    assert "legacy_ConfigLoader_v2_ensemble_metadata_missing" in row["legacyWarnings"]
    assert "legacy_ConfigLoader_v2_ensemble_metadata_missing" not in row["operationalWarnings"]


def test_missing_class_precision_sets_directional_unknown_not_operational_red(monkeypatch, tmp_path):
    _patch_common(monkeypatch, tmp_path, {"SOLUSDT": _snapshot("SOLUSDT")})

    report = audit.audit_symbols(["SOLUSDT"])
    row = report["rows"][0]

    assert row["operationalStatus"] == "GREEN"
    assert row["directionalStatus"] == "UNKNOWN"
    assert row["directionalMetricsAvailable"] is False
    assert "no_class_precision_metrics_long_short_neutral" in row["directionalWarnings"]


def test_weakest_relative_metrics_warns_without_red(monkeypatch, tmp_path):
    _patch_common(
        monkeypatch,
        tmp_path,
        {
            "BTCUSDT": _snapshot("BTCUSDT"),
            "SUIUSDT": _snapshot("SUIUSDT"),
        },
        scores={
            "BTCUSDT": (-0.001, 0.002),
            "SUIUSDT": (-0.009, 0.011),
        },
    )

    report = audit.audit_symbols(["BTCUSDT", "SUIUSDT"])
    rows = {row["symbol"]: row for row in report["rows"]}

    assert rows["SUIUSDT"]["operationalStatus"] == "YELLOW"
    assert rows["SUIUSDT"]["directionalStatus"] == "UNKNOWN"
    assert "weakest_relative_metrics" in rows["SUIUSDT"]["operationalWarnings"]
    assert "weakest_relative_metrics" in rows["SUIUSDT"]["directionalWarnings"]
    assert rows["SUIUSDT"]["operationalStatus"] != "RED"


def test_directional_report_overrides_directional_status(monkeypatch, tmp_path):
    _patch_common(monkeypatch, tmp_path, {"ETHUSDT": _snapshot("ETHUSDT", source="legacy_global")})
    directional_path = tmp_path / "directional.json"
    directional_path.write_text(
        json.dumps(
            {
                "symbolSummaries": [
                    {
                        "symbol": "ETHUSDT",
                        "directionalStatus": "RED",
                        "directionalConfidence": 0.25,
                        "sampleCount": 50,
                        "scoreCalibration": "NOT_CALIBRATED",
                        "directionalWarnings": ["score_not_calibrated"],
                        "recommendedAction": ["reduce_confidence_until_directional_metrics"],
                        "long": {"count": 40, "netExpectancy60m": -0.01, "hit8BeforeMinus5": 0.2},
                        "short": {"count": 10, "netExpectancy60m": 0.002, "hit8BeforeMinus5": 0.5},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = audit.audit_symbols(["ETHUSDT"], directional_report_path=directional_path)
    row = report["rows"][0]

    assert row["directionalStatus"] == "RED"
    assert row["directionalMetricsAvailable"] is True
    assert row["directionalConfidence"] == 0.25
    assert "score_not_calibrated" in row["directionalWarnings"]
    assert "no_class_precision_metrics_long_short_neutral" not in row["directionalWarnings"]
    assert row["directionalMetricsSummary"]["longExpectancy60m"] == -0.01
