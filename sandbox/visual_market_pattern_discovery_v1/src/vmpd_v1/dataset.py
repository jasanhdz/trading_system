from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image

from .core import (artifact_manifest, audit_candles, complete_window, frame_id, iso_utc,
                   load_candles, read_json, sha256_file, write_json, write_jsonl_gz)
from .render import image_sha256, render_frame


def locate_sources(repo: Path) -> dict[str, Path]:
    candidates = [
        repo / "data/aegis_entry_enhancement_v1/candles_1m",
        repo / "data/live_entry_quality_audit_20260815/candles_1m",
        repo / "data/independent_entry_quality_discovery_v1/candles_1m",
    ]
    valid = [p for p in candidates if all((p / f"{s}_1m.parquet").exists() for s in ("SUIUSDT", "BTCUSDT"))]
    if not valid:
        raise FileNotFoundError("No common local SUIUSDT/BTCUSDT 1m source; materialize official Binance USD-M klines first")
    return {s: valid[0] / f"{s}_1m.parquet" for s in ("SUIUSDT", "BTCUSDT")}


def generate(repo: Path, output: Path, config_path: Path, limit: int | None = None) -> dict:
    cfg = read_json(config_path)
    sources = locate_sources(repo)
    dfs = {s: load_candles(p) for s, p in sources.items()}
    start_ms, end_ms = complete_window(dfs, cfg["window_complete_days"])
    source_records = {}
    for symbol, path in sources.items():
        audit = audit_candles(dfs[symbol])
        in_window = dfs[symbol][(dfs[symbol].open_time_ms >= start_ms) & (dfs[symbol].close_time_ms <= end_ms)]
        window_audit = audit_candles(in_window)
        source_records[symbol] = {"path": str(path.relative_to(repo)), "sha256": sha256_file(path),
                                  "full_source": audit, "frozen_window": window_audit}
        if window_audit["gaps"] or window_audit["duplicates"] or window_audit["invalid_close_times"]:
            raise ValueError(f"Non-contiguous frozen source for {symbol}: {window_audit}")
    manifest = {"schema_version": 1, "source": cfg["source"], "start_at": iso_utc(start_ms),
                "end_at": iso_utc(end_ms), "window_complete_days": cfg["window_complete_days"],
                "sources": source_records}
    write_json(output / "data_manifest.json", manifest)
    images = output / "images" / "raw"
    images.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(output / "data_manifest.json")
    first_decision = start_ms + cfg["snapshot_minutes"] * 60_000 - 1
    decisions = list(range(first_decision, end_ms + 1, cfg["snapshot_minutes"] * 60_000))
    if limit is not None:
        decisions = decisions[:limit]
    rows = []
    for i, t in enumerate(decisions):
        fid = frame_id(t)
        rel = Path("images/raw") / f"{fid}.png"
        target = output / rel
        if target.exists():
            image = Image.open(target).convert("RGB")
            if image.size != tuple(cfg["layout"]["resolution"]):
                raise ValueError(f"Existing frame has wrong resolution: {target}")
        else:
            image = render_frame(dfs, t, tuple(cfg["layout"]["resolution"]))
            image.save(target, format="PNG", optimize=False, compress_level=6)
        rows.append({"frame_id": fid, "decision_at": iso_utc(t), "symbol": "SUIUSDT",
                     "image_path": str(rel), "view": "RAW_VIEW", "source_hash": source_hash,
                     "pixel_sha256": image_sha256(image), "file_sha256": sha256_file(output / rel)})
        if (i + 1) % 1000 == 0:
            print(json.dumps({"rendered": i + 1, "total": len(decisions)}), flush=True)
    write_jsonl_gz(output / "frames_manifest.jsonl.gz", rows)
    write_json(output / "diagnostic_summary.json", {"frames": len(rows), "stage": "VISUAL_DATASET_READY", "trading_authority": False})
    write_json(output / "m1_manifest.json", artifact_manifest(output, sha256_file(config_path), "visual_dataset"))
    return {"frames": len(rows), "start_at": manifest["start_at"], "end_at": manifest["end_at"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate causal canonical visual frames")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--limit", type=int, help="Test-only prefix limit")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    output = args.output or project / "artifacts/m1"
    config = args.config or project / "config/m1_frozen.json"
    print(json.dumps(generate(args.repo.resolve(), output.resolve(), config.resolve(), args.limit), indent=2))


if __name__ == "__main__":
    main()
