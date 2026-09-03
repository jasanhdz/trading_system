#!/usr/bin/env python3
"""Build causal scored dataset for AEGIS_E4_RISK_GUARD_VALIDATION_V1."""

import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "sandbox/aegis_strategy_router/src"))
sys.path.insert(0, str(repo_root / "sandbox/aegis_strategy_router/experiments/aegis_e4_robust_training/src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aegis_e4_risk_guard_validation_v1.dataset import load_and_audit_signals, build_e4_panel_for_signals, sha256_file
from aegis_e4_risk_guard_validation_v1.scoring import score_signals_with_e4

def main():
    exp_dir = Path(__file__).resolve().parents[1]
    config_path = exp_dir / "config/preregistration_v1.json"
    config = json.loads(config_path.read_text())

    print("[1/4] Loading and auditing Aegis live signals...")
    signals, audit = load_and_audit_signals(repo_root, config)
    print(f"      Loaded {len(signals)} signals. Splits: {audit['splits']}")

    print("[2/4] Building causal E4 feature panel from 1m candles...")
    candle_root = repo_root / config["sources"]["live_candles"]
    side_panel, candles = build_e4_panel_for_signals(signals, candle_root, config["symbols"])
    print(f"      Side panel constructed: {side_panel.shape}")

    print("[3/4] Scoring signals with frozen E4 models (Tail Risk, Late Entry, Quality)...")
    scored_df = score_signals_with_e4(signals, side_panel, config, repo_root)
    print(f"      Scored dataset shape: {scored_df.shape}")

    print("[4/4] Saving dataset artifacts...")
    ds_dir = exp_dir / "artifacts/dataset_v1"
    ds_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = ds_dir / "causal_scored_signals.parquet"
    scored_df.to_parquet(parquet_path, index=False)

    audit_path = ds_dir / "aegis_signal_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    manifest = {
        "schema": "aegis-e4-risk-guard-dataset-manifest-v1",
        "total_signals": len(scored_df),
        "splits": audit["splits"],
        "symbols": audit["symbols"],
        "sides": audit["sides"],
        "causal_scored_signals_parquet_sha256": sha256_file(parquet_path),
        "aegis_signal_audit_sha256": sha256_file(audit_path),
    }
    manifest_path = ds_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"Dataset build complete! Artifacts saved to {ds_dir}")

if __name__ == "__main__":
    main()
