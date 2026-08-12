"""Checksum-verified, public-only Binance archive boundary for M1A."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

from ..utils import canonical_json, sha256_file


ARCHIVE_HOST = "data.binance.vision"
ARCHIVE_ROOT = f"https://{ARCHIVE_HOST}/data"
SYMBOL_PATTERN = re.compile(r"^[A-Z0-9]{5,20}$")
MONTH_PATTERN = re.compile(r"^20\d{2}-(0[1-9]|1[0-2])$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class PublicArchiveError(RuntimeError):
    pass


@dataclass(frozen=True)
class ArchiveRequest:
    market: str
    data_type: str
    symbol: str
    month: str
    interval: str | None = None

    def validate(self) -> None:
        allowed = {
            "spot": {"aggTrades", "klines"},
            "futures/um": {"aggTrades", "klines", "fundingRate", "markPriceKlines"},
        }
        if self.market not in allowed or self.data_type not in allowed[self.market]:
            raise PublicArchiveError("AEGIS_M1A_ARCHIVE_TYPE_INVALID")
        if not SYMBOL_PATTERN.fullmatch(self.symbol):
            raise PublicArchiveError("AEGIS_M1A_SYMBOL_INVALID")
        if not MONTH_PATTERN.fullmatch(self.month):
            raise PublicArchiveError("AEGIS_M1A_MONTH_INVALID")
        interval_required = self.data_type in {"klines", "markPriceKlines"}
        if interval_required != (self.interval is not None):
            raise PublicArchiveError("AEGIS_M1A_INTERVAL_CONTRACT_INVALID")
        if self.interval is not None and self.interval not in {
            "1m", "5m", "15m", "1h", "4h", "1d"
        }:
            raise PublicArchiveError("AEGIS_M1A_INTERVAL_INVALID")

    @property
    def filename(self) -> str:
        self.validate()
        stem = (
            f"{self.symbol}-{self.interval}-{self.month}"
            if self.data_type in {"klines", "markPriceKlines"}
            else f"{self.symbol}-{self.data_type}-{self.month}"
        )
        return f"{stem}.zip"

    @property
    def url(self) -> str:
        self.validate()
        pieces = [ARCHIVE_ROOT, self.market, "monthly", self.data_type, self.symbol]
        if self.interval is not None:
            pieces.append(self.interval)
        pieces.append(self.filename)
        return "/".join(pieces)


@dataclass(frozen=True)
class ArchiveEvidence:
    schema_version: str
    request: ArchiveRequest
    url: str
    checksum_url: str
    file: str
    expected_sha256: str
    actual_sha256: str
    byte_size: int
    zip_members: tuple[str, ...]
    downloaded: bool


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        raise PublicArchiveError("AEGIS_M1A_REDIRECT_PROHIBITED")


class BinancePublicArchiveClient:
    def __init__(self, *, timeout_seconds: float = 60.0) -> None:
        if timeout_seconds <= 0.0:
            raise ValueError("timeout must be positive")
        self.timeout_seconds = timeout_seconds
        self.opener = urllib.request.build_opener(_NoRedirect())

    def _read(self, url: str) -> bytes:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https" or parsed.hostname != ARCHIVE_HOST:
            raise PublicArchiveError("AEGIS_M1A_ARCHIVE_HOST_INVALID")
        request = urllib.request.Request(url, method="GET")
        with self.opener.open(request, timeout=self.timeout_seconds) as response:
            resolved = urllib.parse.urlparse(response.geturl())
            if resolved.scheme != "https" or resolved.hostname != ARCHIVE_HOST:
                raise PublicArchiveError("AEGIS_M1A_ARCHIVE_RESPONSE_HOST_INVALID")
            return response.read()

    @staticmethod
    def parse_checksum(payload: bytes, expected_filename: str) -> str:
        try:
            line = payload.decode("ascii").strip()
        except UnicodeDecodeError as exc:
            raise PublicArchiveError("AEGIS_M1A_CHECKSUM_ENCODING_INVALID") from exc
        parts = line.split()
        if len(parts) != 2 or parts[1].lstrip("*") != expected_filename:
            raise PublicArchiveError("AEGIS_M1A_CHECKSUM_MANIFEST_INVALID")
        digest = parts[0].lower()
        if not SHA256_PATTERN.fullmatch(digest):
            raise PublicArchiveError("AEGIS_M1A_CHECKSUM_INVALID")
        return digest

    @staticmethod
    def validate_zip(path: Path) -> tuple[str, ...]:
        try:
            with zipfile.ZipFile(path) as archive:
                corrupt = archive.testzip()
                if corrupt is not None:
                    raise PublicArchiveError("AEGIS_M1A_ZIP_CRC_INVALID")
                members = tuple(item.filename for item in archive.infolist() if not item.is_dir())
        except zipfile.BadZipFile as exc:
            raise PublicArchiveError("AEGIS_M1A_ZIP_INVALID") from exc
        if not members or any(
            PurePosixPath(name).is_absolute()
            or ".." in PurePosixPath(name).parts
            or PurePosixPath(name).suffix.lower() != ".csv"
            for name in members
        ):
            raise PublicArchiveError("AEGIS_M1A_ZIP_MEMBER_INVALID")
        return members

    def download(self, request: ArchiveRequest, root: Path) -> ArchiveEvidence:
        request.validate()
        target = root / request.market / "monthly" / request.data_type / request.symbol
        if request.interval is not None:
            target /= request.interval
        target.mkdir(parents=True, exist_ok=True)
        os.chmod(target, 0o700)
        destination = target / request.filename
        checksum_url = request.url + ".CHECKSUM"
        expected = self.parse_checksum(self._read(checksum_url), request.filename)
        downloaded = False
        if destination.exists():
            if sha256_file(destination) != expected:
                raise PublicArchiveError("AEGIS_M1A_IMMUTABLE_ARCHIVE_CONFLICT")
        else:
            payload = self._read(request.url)
            actual = hashlib.sha256(payload).hexdigest()
            if actual != expected:
                raise PublicArchiveError("AEGIS_M1A_ARCHIVE_HASH_MISMATCH")
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{request.filename}.", dir=target
            )
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.chmod(temporary_name, 0o400)
                os.replace(temporary_name, destination)
                downloaded = True
            finally:
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)
        os.chmod(destination, 0o400)
        actual = sha256_file(destination)
        members = self.validate_zip(destination)
        return ArchiveEvidence(
            schema_version="aegis-binance-public-archive-evidence-v1",
            request=request,
            url=request.url,
            checksum_url=checksum_url,
            file=str(destination.resolve()),
            expected_sha256=expected,
            actual_sha256=actual,
            byte_size=destination.stat().st_size,
            zip_members=members,
            downloaded=downloaded,
        )


def month_range(start: str, end: str) -> tuple[str, ...]:
    if not MONTH_PATTERN.fullmatch(start) or not MONTH_PATTERN.fullmatch(end):
        raise PublicArchiveError("AEGIS_M1A_MONTH_RANGE_INVALID")
    start_year, start_month = map(int, start.split("-"))
    end_year, end_month = map(int, end.split("-"))
    cursor = start_year * 12 + start_month - 1
    finish = end_year * 12 + end_month - 1
    if cursor > finish:
        raise PublicArchiveError("AEGIS_M1A_MONTH_RANGE_REVERSED")
    values = []
    while cursor <= finish:
        values.append(f"{cursor // 12:04d}-{cursor % 12 + 1:02d}")
        cursor += 1
    return tuple(values)


def append_manifest(path: Path, evidence: Iterable[ArchiveEvidence]) -> None:
    """Append evidence once; reject changed identities or archive hashes."""

    existing: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            existing[str(row["url"])] = row
    additions = []
    for item in evidence:
        row = json.loads(canonical_json(asdict(item)))
        prior = existing.get(item.url)
        if prior is not None:
            comparable = {**row, "downloaded": prior.get("downloaded", False)}
            if prior != comparable:
                raise PublicArchiveError("AEGIS_M1A_MANIFEST_CONFLICT")
            continue
        existing[item.url] = row
        additions.append(row)
    if not additions:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in additions:
            handle.write(canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o600)
