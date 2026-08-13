#!/usr/bin/env python3
"""Import checksum-identified Binance USD-M aggregate-trade archives into C2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable

from aegis.config import CANONICAL_SYMBOLS
from aegis.research.market_event_lab_m1 import MarketEventContractError
from aegis.research.microstructure_events_c2 import AggregateTrade, C2Archive


ARCHIVE_NAME = re.compile(
    r"^(?P<symbol>[A-Z0-9]+)-aggTrades-(?P<month>20\d{2}-(?:0[1-9]|1[0-2]))\.zip$"
)
EXPECTED_HEADER = (
    "agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id",
    "transact_time", "is_buyer_maker",
)


def archive_identity(path: Path) -> tuple[str, str]:
    match = ARCHIVE_NAME.fullmatch(path.name)
    if match is None or match.group("symbol") not in CANONICAL_SYMBOLS:
        raise MarketEventContractError("AEGIS_C2_ARCHIVE_IDENTITY_INVALID")
    return match.group("symbol"), match.group("month")


def archive_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_aggregate_trades(path: Path) -> Iterable[AggregateTrade]:
    symbol, _ = archive_identity(path)
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as error:
        raise MarketEventContractError("AEGIS_C2_ARCHIVE_ZIP_INVALID") from error
    with archive:
        members = tuple(item for item in archive.infolist() if not item.is_dir())
        if len(members) != 1:
            raise MarketEventContractError("AEGIS_C2_ARCHIVE_MEMBER_COUNT_INVALID")
        member = members[0]
        member_path = PurePosixPath(member.filename)
        expected = f"{path.stem}.csv"
        if (
            member_path.is_absolute() or ".." in member_path.parts
            or member_path.name != expected
        ):
            raise MarketEventContractError("AEGIS_C2_ARCHIVE_MEMBER_INVALID")
        with archive.open(member) as raw:
            rows = csv.DictReader((line.decode("utf-8-sig") for line in raw))
            if tuple(rows.fieldnames or ()) != EXPECTED_HEADER:
                raise MarketEventContractError("AEGIS_C2_ARCHIVE_HEADER_INVALID")
            for item in rows:
                try:
                    price = float(item["price"])
                    quantity = float(item["quantity"])
                    timestamp = int(item["transact_time"])
                    trade_id = int(item["agg_trade_id"])
                    maker = item["is_buyer_maker"].lower()
                except (KeyError, TypeError, ValueError) as error:
                    raise MarketEventContractError("AEGIS_C2_ARCHIVE_ROW_INVALID") from error
                if (
                    price <= 0 or quantity <= 0 or timestamp <= 0 or trade_id <= 0
                    or maker not in {"true", "false"}
                ):
                    raise MarketEventContractError("AEGIS_C2_ARCHIVE_ROW_INVALID")
                yield AggregateTrade(
                    symbol=symbol,
                    aggregate_trade_id=trade_id,
                    event_time_ms=timestamp,
                    trade_time_ms=timestamp,
                    price=price,
                    quantity=quantity,
                    quote_notional=price * quantity,
                    buyer_is_maker=maker == "true",
                )


def import_archives(paths: Iterable[Path], output: Path) -> dict[str, object]:
    archive = C2Archive(output)
    reports = []
    try:
        for path in sorted(paths):
            symbol, month = archive_identity(path)
            accepted = duplicates = 0
            first_timestamp = last_timestamp = None
            for row in iter_aggregate_trades(path):
                if archive.insert(row):
                    accepted += 1
                else:
                    duplicates += 1
                first_timestamp = (
                    row.trade_time_ms if first_timestamp is None
                    else min(first_timestamp, row.trade_time_ms)
                )
                last_timestamp = (
                    row.trade_time_ms if last_timestamp is None
                    else max(last_timestamp, row.trade_time_ms)
                )
            digest = archive_sha256(path)
            manifest_hash = archive.append_manifest({
                "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "source": f"BINANCE_PUBLIC_ARCHIVE:{symbol}:{month}:sha256:{digest}",
                "first_timestamp_ms": first_timestamp,
                "last_timestamp_ms": last_timestamp,
                "accepted_rows": accepted,
                "duplicate_rows": duplicates,
                "rejected_rows": 0,
            })
            reports.append({
                "file": str(path.resolve()), "sha256": digest, "symbol": symbol,
                "month": month, "accepted_rows": accepted,
                "duplicate_rows": duplicates, "manifest_hash": manifest_hash,
            })
        archive.validate_manifest_chain()
    finally:
        archive.close()
    return {
        "schema_version": "aegis-market-microstructure-c2-import-v1",
        "output": str(output.resolve()), "archives": reports,
        "authenticated_requests": 0, "exchange_mutations": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/market_microstructure_events_c2/c2_archive.db"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output if args.output.is_absolute() else root / args.output
    paths = tuple(path if path.is_absolute() else root / path for path in args.archives)
    print(json.dumps(import_archives(paths, output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
