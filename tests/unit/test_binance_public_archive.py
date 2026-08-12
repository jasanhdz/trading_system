import hashlib
import zipfile

import pytest

from aegis.research.binance_public_archive import (
    ArchiveRequest,
    BinancePublicArchiveClient,
    PublicArchiveError,
    append_manifest,
    month_range,
)


def test_archive_urls_are_exact_and_reject_unknown_surfaces() -> None:
    assert ArchiveRequest(
        "futures/um", "aggTrades", "BTCUSDT", "2026-07"
    ).url.endswith(
        "/futures/um/monthly/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2026-07.zip"
    )
    assert ArchiveRequest(
        "spot", "klines", "ADAUSDT", "2025-01", "1m"
    ).url.endswith("/spot/monthly/klines/ADAUSDT/1m/ADAUSDT-1m-2025-01.zip")
    with pytest.raises(PublicArchiveError, match="TYPE_INVALID"):
        ArchiveRequest("futures/um", "orders", "BTCUSDT", "2026-07").validate()
    with pytest.raises(PublicArchiveError, match="INTERVAL_CONTRACT"):
        ArchiveRequest("spot", "klines", "BTCUSDT", "2026-07").validate()
    assert month_range("2025-11", "2026-02") == (
        "2025-11", "2025-12", "2026-01", "2026-02"
    )
    with pytest.raises(PublicArchiveError, match="REVERSED"):
        month_range("2026-02", "2025-11")


def test_checksum_and_zip_validation_fail_closed(tmp_path) -> None:
    filename = "BTCUSDT-aggTrades-2026-07.zip"
    digest = "a" * 64
    assert (
        BinancePublicArchiveClient.parse_checksum(
            f"{digest}  {filename}\n".encode(), filename
        )
        == digest
    )
    with pytest.raises(PublicArchiveError, match="MANIFEST_INVALID"):
        BinancePublicArchiveClient.parse_checksum(
            f"{digest}  wrong.zip\n".encode(), filename
        )
    archive = tmp_path / filename
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.csv", "a,b\n1,2\n")
    with pytest.raises(PublicArchiveError, match="MEMBER_INVALID"):
        BinancePublicArchiveClient.validate_zip(archive)


def test_manifest_is_idempotent_and_rejects_changed_archive(tmp_path) -> None:
    request = ArchiveRequest("spot", "aggTrades", "BTCUSDT", "2026-07")
    archive = tmp_path / request.filename
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("rows.csv", "1,2\n")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    from aegis.research.binance_public_archive import ArchiveEvidence

    row = ArchiveEvidence(
        "aegis-binance-public-archive-evidence-v1",
        request,
        request.url,
        request.url + ".CHECKSUM",
        str(archive),
        digest,
        digest,
        archive.stat().st_size,
        ("rows.csv",),
        True,
    )
    manifest = tmp_path / "manifest.jsonl"
    append_manifest(manifest, (row,))
    append_manifest(manifest, (row,))
    assert len(manifest.read_text().splitlines()) == 1
    with pytest.raises(PublicArchiveError, match="CONFLICT"):
        append_manifest(manifest, (row.__class__(**{**row.__dict__, "actual_sha256": "0" * 64}),))
