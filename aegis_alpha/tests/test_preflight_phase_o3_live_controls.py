#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.tools.preflight_phase_o3_live_controls import ENTRY_SYMBOLS, LOOKBACK, validate


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_tree() -> Path:
    root = Path(tempfile.mkdtemp(prefix="phase_o3_preflight_"))
    write(root / "aegis_alpha/configs/turbo.yaml", """
enabled: true
sizing:
  conservative: {leverage: 15, position_fraction: 0.01}
  normal: {leverage: 20, position_fraction: 0.01}
  premium: {leverage: 25, position_fraction: 0.01}
  max_allowed_leverage: 30
  max_allowed_position_fraction: 0.01
""")
    write(root / "binance-futures-bot-ts/regime_config.live.yaml", """
aegis:
  phase_o_short_live:
    enabled: true
    mode: EXPERIMENTAL_LIVE
    side: SHORT_ONLY
    capital_profile: TEST_CAPITAL
    allow_orders: true
    require_brackets: true
    max_position_fraction_default: 0.01
    max_open_phase_o_positions: 1
    max_phase_o_trades_per_day: 3
    align_short_leverage_with_long: true
    remove_legacy_short_leverage_downshift: true
    allow_link_entry: false
    link_avoid_only: true
    leverage: {conservative: 15, normal: 20, premium: 25, max_allowed_leverage: 30}
    guard_modes:
      phase_o_ml: ENFORCE
      momentum_ride: ENFORCE
      clean_entry: SHADOW
      event_risk: SHADOW
      entry_quality: SHADOW
      decision_brain: SHADOW
      regime_engine: SHADOW
      short_gate_legacy: SHADOW
      risk_shadow_guards: SHADOW
    hard_safety:
      brackets: ENFORCE
      max_open_positions: ENFORCE
      max_trades_per_day: ENFORCE
      daily_loss_stop: ENFORCE
      exchange_min_notional: ENFORCE
      exchange_order_errors: ENFORCE
      link_no_entry: ENFORCE
    symbols:
      LTCUSDT: {enabled: true, max_position_fraction: 0.01, caution_level: normal}
      AVAXUSDT: {enabled: true, max_position_fraction: 0.01, caution_level: normal}
      ETHUSDT: {enabled: true, max_position_fraction: 0.0075, caution_level: cautious}
      SUIUSDT: {enabled: true, max_position_fraction: 0.0075, caution_level: normal}
      ADAUSDT: {enabled: true, max_position_fraction: 0.01, caution_level: normal}
      DOGEUSDT: {enabled: true, max_position_fraction: 0.0075, caution_level: cautious}
      BTCUSDT: {enabled: true, max_position_fraction: 0.005, caution_level: very_cautious}
      BNBUSDT: {enabled: true, max_position_fraction: 0.01, caution_level: normal}
      XRPUSDT: {enabled: true, max_position_fraction: 0.0075, caution_level: cautious}
      SOLUSDT: {enabled: true, max_position_fraction: 0.0075, caution_level: final_repair_candidate}
      LINKUSDT: {enabled: true, entry_enabled: false, avoid_only: true}
  entry_policy:
    guards:
      momentum_ride: {enabled: true, mode: ENFORCE}
      clean_entry: {enabled: true, mode: ENFORCE}
      event_risk: {enabled: true, mode: ENFORCE}
      entry_quality: {enabled: true, mode: ENFORCE}
      decision_brain: {enabled: true, mode: ENFORCE}
      regime: {enabled: true, mode: SHADOW}
      short_gate: {enabled: true, mode: ENFORCE}
      long_risk_shadow: {enabled: true, mode: ENFORCE_PROBE_LONG_CRITICAL}
  turbo:
    enabled: true
    live_enabled: true
    allow_short: true
    position_fraction_cap: 0.01
    max_trades_per_day: 3
    max_consecutive_losses: 2
    require_brackets: true
  momentum_ride:
    safety_caps: {max_leverage: 30, max_position_fraction: 0.01}
    profiles:
      major_short: {leverage: 25, position_fraction: 0.01}
SYMBOL_OVERRIDES:
  LTCUSDT: {AEGIS_TURBO: {}}
""")

    write(root / "binance-futures-bot-ts/src/app/services/TradingService.ts", "phase_o_short_guard_modes_applied withPhaseOShortGuardModes")
    for symbol in ENTRY_SYMBOLS:
        short_key = f"short_{LOOKBACK[symbol]}d"
        manifest = {
            "phase_o_live_enabled": True,
            "model_paths": {short_key: f"/tmp/{symbol}/phase_o_model.joblib", "long_30d": f"/tmp/{symbol}/long.joblib"},
            "pre_phase_o_live_model_paths": {"long_30d": f"/tmp/{symbol}/long.joblib"},
        }
        write(root / "aegis_alpha/models/turbo" / symbol / "active_manifest.json", json.dumps(manifest))
    link_manifest = {"phase_o_link_entry_enabled": False, "phase_o_avoid_only": True, "model_paths": {"short_30d": "/tmp/LINK/old_short.joblib"}}
    write(root / "aegis_alpha/models/turbo/LINKUSDT/active_manifest.json", json.dumps(link_manifest))
    return root


def test_passes_valid_tree() -> None:
    report = validate(make_tree(), 20)
    assert report["status"] == "PASSED", report["errors"]


def test_fails_if_link_entry_enabled() -> None:
    root = make_tree()
    path = root / "aegis_alpha/models/turbo/LINKUSDT/active_manifest.json"
    data = json.loads(path.read_text())
    data["phase_o_link_entry_enabled"] = True
    path.write_text(json.dumps(data))
    report = validate(root, 20)
    assert report["status"] == "FAILED"
    assert any("LINKUSDT" in e for e in report["errors"])


def test_fails_if_secondary_guard_enforced() -> None:
    root = make_tree()
    path = root / "binance-futures-bot-ts/regime_config.live.yaml"
    text = path.read_text().replace("clean_entry: SHADOW", "clean_entry: ENFORCE", 1)
    path.write_text(text)
    report = validate(root, 20)
    assert "guard_modes.clean_entry_not_SHADOW" in report["errors"]


def test_fails_if_position_cap_too_large() -> None:
    root = make_tree()
    path = root / "binance-futures-bot-ts/regime_config.live.yaml"
    path.write_text(path.read_text().replace("position_fraction_cap: 0.01", "position_fraction_cap: 0.02"))
    report = validate(root, 20)
    assert "aegis.turbo.position_fraction_cap_gt_0.01" in report["errors"]


def test_fails_if_long_model_path_changed() -> None:
    root = make_tree()
    path = root / "aegis_alpha/models/turbo/LTCUSDT/active_manifest.json"
    data = json.loads(path.read_text())
    data["model_paths"]["long_30d"] = "/tmp/changed_long.joblib"
    path.write_text(json.dumps(data))
    report = validate(root, 20)
    assert any("long_model_path_changed" in e for e in report["errors"])


if __name__ == "__main__":
    test_passes_valid_tree()
    test_fails_if_link_entry_enabled()
    test_fails_if_secondary_guard_enforced()
    test_fails_if_position_cap_too_large()
    test_fails_if_long_model_path_changed()
    print("manual_preflight_phase_o3_live_controls_tests_passed")
