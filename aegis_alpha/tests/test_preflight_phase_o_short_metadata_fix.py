#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.tools.preflight_phase_o_short_metadata_fix import (  # noqa: E402
    ENTRY_SYMBOLS,
    audit_manifests,
    inspect_ts_source,
)


def robust_source() -> str:
    return """
    private extractPhaseOTurboMetadata(signal: unknown) {
      return [
        'signal.metadata.aegis.turbo.phase_o',
        'signal.metadata.aegis.turbo.raw.phase_o',
        'signal.metadata.turbo.phase_o',
        'signal.aegis.turbo.phase_o',
        'signal.metadata.rawPrediction.aegis.turbo.phase_o',
        'signal.phase_o',
        'signal.phaseO'
      ];
    }
    private isPhaseOShortLiveSignal(signal, side) {
      const metadata = this.extractPhaseOTurboMetadata(signal, side);
      console.log('phase_o_short_guard_modes_applied', 'phase_o_metadata_source_path');
    }
    """


def write_manifest(root: Path, symbol: str, ok: bool = True, link: bool = False) -> None:
    active = root / symbol
    active.mkdir(parents=True, exist_ok=True)
    if link:
        manifest = {"phase_o_avoid_only": True, "phase_o_link_entry_enabled": False, "model_paths": {}}
    else:
        model_dir = root / symbol / "phase_o"
        model_dir.mkdir(exist_ok=True)
        model = model_dir / "short_phase_o.joblib"
        model.write_text("x")
        manifest = {
            "phase_o_live_enabled": ok,
            "phase_o_overlay_persistence_enabled": ok,
            "model_paths": {"short_30d": str(model) if ok else "/tmp/base.joblib"},
        }
    (active / "active_manifest.json").write_text(json.dumps(manifest))


def test_preflight_detects_metadata_phase_o_source() -> None:
    checks = inspect_ts_source(robust_source())
    assert checks["ok"] is True
    assert checks["extractor_present"] is True
    assert checks["multiple_metadata_paths"] is True


def test_preflight_fails_if_extractor_missing() -> None:
    checks = inspect_ts_source("const phaseO = turbo.phase_o ?? turbo.raw?.phase_o ?? {};")
    assert checks["ok"] is False
    assert checks["single_rigid_path_only"] is True


def test_preflight_fails_if_manifests_drifted() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for symbol in ENTRY_SYMBOLS:
            write_manifest(root, symbol, ok=(symbol != "ETHUSDT"))
        write_manifest(root, "LINKUSDT", link=True)
        audit = audit_manifests(root)
        assert audit["ok"] is False
        assert any("ETHUSDT" in err for err in audit["errors"])


def test_preflight_fails_if_link_entry_enabled() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for symbol in ENTRY_SYMBOLS:
            write_manifest(root, symbol, ok=True)
        active = root / "LINKUSDT"
        active.mkdir(parents=True)
        model = active / "phase_o_link.joblib"
        model.write_text("x")
        (active / "active_manifest.json").write_text(json.dumps({
            "phase_o_avoid_only": False,
            "phase_o_link_entry_enabled": True,
            "model_paths": {"short_30d": str(model)},
        }))
        audit = audit_manifests(root)
        assert audit["ok"] is False
        assert any("LINKUSDT" in err for err in audit["errors"])


def test_json_serializes() -> None:
    payload = {"checks": inspect_ts_source(robust_source())}
    json.dumps(payload, sort_keys=True)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("preflight_phase_o_short_metadata_fix tests passed")
