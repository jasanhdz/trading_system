#!/usr/bin/env python3
"""GEN2-D3 shared contract helpers (spec: aegis_alpha/docs/gen2/GEN2_D3_SPEC.md v1.1).

Research-only. Immutable-artifact conventions: every artifact directory gets a
manifest with sha256 checksums, code commit, config hash and environment info.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GEN2_ROOT = Path("/home/jasan/Develop/aegis_gen2")
D3_ROOT = GEN2_ROOT / "d3"
SNAPSHOT_ROOT = D3_ROOT / "market_snapshots"
SERIES_ROOT = D3_ROOT / "canonical_series"
DATASET_ROOT = D3_ROOT / "datasets"
REPORTS_ROOT = D3_ROOT / "reports"
LOCKBOX_MANIFEST_PATH = GEN2_ROOT / "GEN2_LOCKBOX_MANIFEST.json"

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT",
    "ADAUSDT", "AVAXUSDT", "SUIUSDT", "LTCUSDT", "LINKUSDT",
]
TIMEFRAME = "5m"
BAR_MINUTES = 5
TARGET_RANGE_DAYS = 730
MIN_DAYS_PER_SYMBOL = 350
RAW_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_asset_volume", "number_of_trades", "taker_buy_base_volume", "taker_buy_quote_volume",
]
REPO = Path(__file__).resolve().parents[2]


def read_csv_exact(path: Path, **kwargs: Any):
    """Read a canonical CSV with exact float round-trip.

    pandas' default fast parser is not guaranteed to round-trip the shortest
    float repr (verified on pandas 3.0: '100.49999999999997' parses to ...96),
    which would break the bit-identity contract of this spec.
    """
    import pandas as pd

    return pd.read_csv(path, float_precision="round_trip", **kwargs)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_stamp() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_commit() -> str:
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, text=True, capture_output=True, timeout=8, check=False)
        return proc.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
    }
    for mod in ("numpy", "pandas", "sklearn", "joblib"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = "missing"
    return info


def validate_gen2_path(path: Path) -> None:
    resolved = path.resolve()
    text = str(resolved)
    if "active" in resolved.parts or "active_manifest" in text:
        raise ValueError(f"refusing live/active path: {resolved}")
    allowed_roots = (GEN2_ROOT.resolve(), Path("/tmp"))
    if not any(root == resolved or root in resolved.parents for root in allowed_roots):
        raise ValueError(f"refusing non-gen2 output path: {resolved}")


def write_manifest(directory: Path, manifest: dict[str, Any], name: str) -> Path:
    manifest = dict(manifest)
    manifest.setdefault("created_at_utc", utc_now().isoformat())
    manifest.setdefault("code_commit", git_commit())
    manifest.setdefault("environment", environment_info())
    path = directory / f"{name}.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    checksums = {f.name: sha256_file(f) for f in sorted(directory.iterdir()) if f.is_file() and f.suffix == ".csv"}
    checksums[path.name] = sha256_file(path)
    (directory / f"{name}.sha256").write_text(
        "\n".join(f"{v}  {k}" for k, v in sorted(checksums.items())) + "\n", encoding="utf-8"
    )
    return path


def verify_manifest_checksums(directory: Path, name: str) -> tuple[bool, list[str]]:
    sha_path = directory / f"{name}.sha256"
    if not sha_path.exists():
        return False, [f"missing {sha_path.name}"]
    errors: list[str] = []
    for line in sha_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, filename = line.split(None, 1)
        target = directory / filename.strip()
        if not target.exists():
            errors.append(f"missing file {filename}")
        elif sha256_file(target) != digest:
            errors.append(f"checksum mismatch {filename}")
    return not errors, errors


def new_artifact_dir(root: Path, artifact_id: str | None = None) -> tuple[Path, str]:
    artifact_id = artifact_id or utc_stamp()
    directory = root / artifact_id
    validate_gen2_path(directory)
    if directory.exists():
        raise FileExistsError(f"artifact_id already exists (immutability): {directory}")
    directory.mkdir(parents=True)
    return directory, artifact_id
