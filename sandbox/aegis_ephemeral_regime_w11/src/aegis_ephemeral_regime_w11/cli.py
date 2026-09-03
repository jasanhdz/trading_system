"""Command-line entry points for W11 audit and execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .data import build_data_panel, load_frozen_config, load_selected_candles
from .experiment import run_experiment
from .reporting import sha256_file, write_results


def sandbox_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "config" / "w11_frozen.json").is_file() and (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("could not resolve W11 sandbox root")


def repository_root(root: Path | None = None) -> Path:
    start = (root or sandbox_root()).resolve()
    for parent in (start, *start.parents):
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("could not resolve repository root")


def audit_sources(root: Path | None = None, repository: Path | None = None) -> dict[str, Any]:
    sandbox = (root or sandbox_root()).resolve()
    repo = (repository or repository_root(sandbox)).resolve()
    config_path = sandbox / "config" / "w11_frozen.json"
    config = load_frozen_config(config_path)
    manifest_path = (repo / config["source"]["manifest"]).resolve()
    candle_dir = (repo / config["source"]["candle_dir"]).resolve()
    if manifest_path.parent != candle_dir:
        raise ValueError("configured manifest is outside the configured source")
    source = json.loads(manifest_path.read_text(encoding="utf-8"))
    symbols: dict[str, Any] = {}
    for symbol in config["source"]["symbols"]:
        record = source["symbols"][symbol]
        path = (repo / record["parquet"]).resolve()
        if path != candle_dir / f"{symbol}_1m.parquet":
            raise ValueError(f"unexpected source path for {symbol}")
        actual = sha256_file(path)
        if actual != record["parquet_sha256"]:
            raise ValueError(f"SHA-256 mismatch for {symbol}")
        symbols[symbol] = {"path": path.relative_to(repo).as_posix(), "rows": record["rows"], "sha256": actual}
    payload = {
        "status": "PASS", "config_sha256": sha256_file(config_path),
        "source_manifest": manifest_path.relative_to(repo).as_posix(),
        "source_manifest_sha256": sha256_file(manifest_path), "symbols": symbols,
        "external_holdouts_accessed": False, "model_results_produced": False,
    }
    artifacts = sandbox / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "data_audit_runtime.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aegis-ephemeral-regime-w11")
    parser.add_argument("command", choices=("audit", "run"))
    args = parser.parse_args(argv)
    root = sandbox_root()
    repo = repository_root(root)
    if args.command == "audit":
        audit_sources(root, repo)
        return 0
    config = load_frozen_config(root / "config" / "w11_frozen.json")
    candles = load_selected_candles(config, repository_dir=repo, verify_hashes=True)
    panel = build_data_panel(candles, config, repository_dir=repo)
    result = run_experiment(panel, config)
    write_results(result, config, root, repository_dir=repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
