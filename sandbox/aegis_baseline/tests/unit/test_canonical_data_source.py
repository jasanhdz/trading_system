import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aegis.config import CANONICAL_SYMBOLS
from aegis.data import CanonicalDataError, CanonicalSeriesSource, DataPurpose


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(root: Path, *, status: str = "OK") -> str:
    root.mkdir()
    included = {}
    start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    for symbol_index, symbol in enumerate(CANONICAL_SYMBOLS):
        path = root / f"{symbol}_5m.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle); writer.writerow(("timestamp", "open", "high", "low", "close", "volume"))
            for index in range(3):
                value = 10.0 + symbol_index + index * 0.01
                writer.writerow(((start + timedelta(minutes=5 * index)).replace(tzinfo=None).isoformat(sep=" "),
                                 value, value + 0.02, value - 0.02, value + 0.01, 100.0))
        included[symbol] = {"sha256": _sha(path), "rows": 3}
    passes = [{"symbol": symbol, "passes": True} for symbol in CANONICAL_SYMBOLS]
    manifest = {"schema": "gen2_d3_series_v1", "status": status, "artifact_id": "fixture",
                "excluded_symbols": [], "included_symbols": included,
                "gates": {"g4_gaps": passes, "g5_coverage": passes}}
    manifest_path = root / "series_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    manifest_hash = _sha(manifest_path)
    lines = [f"{included[symbol]['sha256']}  {symbol}_5m.csv" for symbol in CANONICAL_SYMBOLS]
    lines.append(f"{manifest_hash}  series_manifest.json")
    (root / "series_manifest.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_hash


def test_canonical_source_is_hash_verified_read_only_and_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "canonical"; expected = _fixture(root)
    source = CanonicalSeriesSource(root, DataPurpose.TRAINING, expected)
    first = source.audit(); second = source.audit()
    assert first == second
    assert first.read_only and first.finality_verified and first.gap_free
    start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    before = {path.name: _sha(path) for path in root.iterdir()}
    bars = source.load(start=start, end=start + timedelta(minutes=15))
    assert tuple(bars) == CANONICAL_SYMBOLS and all(len(rows) == 3 for rows in bars.values())
    assert {path.name: _sha(path) for path in root.iterdir()} == before


def test_canonical_source_rejects_pending_finality_and_content_mutation(tmp_path: Path) -> None:
    pending = tmp_path / "pending"; pending_hash = _fixture(pending, status="PENDING")
    with pytest.raises(CanonicalDataError, match="not final"):
        CanonicalSeriesSource(pending, DataPurpose.TRAINING, pending_hash).audit()
    root = tmp_path / "mutated"; expected = _fixture(root)
    with (root / "BTCUSDT_5m.csv").open("a", encoding="utf-8") as handle:
        handle.write("corruption\n")
    with pytest.raises(CanonicalDataError, match="hash mismatch"):
        CanonicalSeriesSource(root, DataPurpose.TRAINING, expected).audit()


def test_existing_gen2_canonical_manifest_contract_is_read_only() -> None:
    root = Path("/home/jasan/Develop/aegis_gen2/d3/canonical_series/v1")
    expected = "00177a1b8e9e9db9b0cb105b63034bd4b3e5a9c859be3053e59d16be04e52916"
    audit = CanonicalSeriesSource(root, DataPurpose.BENCHMARK, expected).audit(verify_content=False)
    assert audit.artifact_id == "v1"
    assert audit.symbols == CANONICAL_SYMBOLS
    assert audit.read_only and audit.finality_verified
