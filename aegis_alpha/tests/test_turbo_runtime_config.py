from __future__ import annotations

import os

from aegis_alpha.turbo.config import (
    DEFAULT_TURBO_CONFIG,
    clear_runtime_turbo_config_cache,
    get_runtime_turbo_config,
)
from aegis_alpha.turbo.turbo_signal import _sizing


def _touch_newer(path):
    now = os.stat(path).st_mtime_ns + 10_000_000
    os.utime(path, ns=(now, now))


def test_runtime_turbo_config_falls_back_when_missing(monkeypatch, tmp_path):
    clear_runtime_turbo_config_cache()
    monkeypatch.setenv("AEGIS_TURBO_CONFIG", str(tmp_path / "missing.yaml"))

    cfg = get_runtime_turbo_config()

    assert cfg.position_fraction.normal == DEFAULT_TURBO_CONFIG.position_fraction.normal
    assert cfg.leverage.normal == DEFAULT_TURBO_CONFIG.leverage.normal
    clear_runtime_turbo_config_cache()


def test_runtime_turbo_config_hot_reloads_from_yaml(monkeypatch, tmp_path):
    path = tmp_path / "turbo.yaml"
    monkeypatch.setenv("AEGIS_TURBO_CONFIG", str(path))
    clear_runtime_turbo_config_cache()
    path.write_text(
        """
enabled: true
sizing:
  normal:
    leverage: 11
    position_fraction: 0.33
""",
        encoding="utf-8",
    )

    first = get_runtime_turbo_config()
    assert first.leverage.normal == 11
    assert first.position_fraction.normal == 0.33

    path.write_text(
        """
enabled: true
sizing:
  normal:
    leverage: 12
    position_fraction: 0.44
""",
        encoding="utf-8",
    )
    _touch_newer(path)

    second = get_runtime_turbo_config()
    assert second.leverage.normal == 12
    assert second.position_fraction.normal == 0.44
    clear_runtime_turbo_config_cache()


def test_sizing_uses_yaml_buckets(monkeypatch, tmp_path):
    path = tmp_path / "turbo.yaml"
    monkeypatch.setenv("AEGIS_TURBO_CONFIG", str(path))
    clear_runtime_turbo_config_cache()
    path.write_text(
        """
enabled: true
sizing:
  conservative:
    leverage: 5
    position_fraction: 0.05
  normal:
    leverage: 10
    position_fraction: 0.20
  premium:
    leverage: 15
    position_fraction: 0.40
thresholds:
  min_turbo_score_conservative: 0.45
  min_turbo_score_shadow: 0.55
  min_turbo_score_premium: 0.70
""",
        encoding="utf-8",
    )

    assert _sizing(0.50) == ("conservative", 5, 0.05)
    assert _sizing(0.60) == ("normal", 10, 0.20)
    assert _sizing(0.80) == ("premium", 15, 0.40)
    clear_runtime_turbo_config_cache()
