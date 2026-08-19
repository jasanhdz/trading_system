"""Feature flags for the Risk Guard system.

The tail_risk_threshold is FROZEN at 0.4522452210875323 (V1).
It cannot be changed at runtime. Only mode (disabled/observe_only/enforce)
and fail_closed can be toggled.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .domain import RiskGuardConfig

FROZEN_TAIL_RISK_THRESHOLD = 0.4522452210875323


class RiskGuardMode:
    """Mode constants."""
    DISABLED = "disabled"
    OBSERVE_ONLY = "observe_only"
    ENFORCE = "enforce"


class RiskGuardFlags:
    """Thread-safe feature flags for the risk guard system.

    Flags are loaded from a JSON file and can be updated at runtime.
    Default state: DISABLED (risk guard off, no enforcement).

    The tail_risk_threshold is FROZEN and cannot be changed at runtime.
    Only mode and fail_closed can be toggled.

    Usage:
        flags = RiskGuardFlags()
        flags.load_from_file("config/risk_guard_flags.json")

        config = flags.to_config()
        if config.enabled:
            guard = E4TailRiskGuard(config)
    """

    _instance: RiskGuardFlags | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._enabled = False
        self._mode = RiskGuardMode.DISABLED
        self._fail_closed = True
        self._models_joblib_path = ""
        self._models_joblib_sha256 = ""
        self._feature_schema_path = ""
        self._feature_schema_sha256 = ""
        self._candle_data_root = ""
        self._update_lock = threading.Lock()

    @classmethod
    def instance(cls) -> RiskGuardFlags:
        """Get or create the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        with cls._lock:
            cls._instance = None

    def load_from_file(self, path: str | Path) -> None:
        """Load flags from a JSON file."""
        p = Path(path)
        if not p.exists():
            return
        data = json.loads(p.read_text())
        with self._update_lock:
            self._enabled = data.get("enabled", False)
            self._mode = data.get("mode", RiskGuardMode.DISABLED)
            self._fail_closed = data.get("fail_closed", True)
            self._models_joblib_path = data.get("models_joblib_path", "")
            self._models_joblib_sha256 = data.get("models_joblib_sha256", "")
            self._feature_schema_path = data.get("feature_schema_path", "")
            self._feature_schema_sha256 = data.get("feature_schema_sha256", "")
            self._candle_data_root = data.get("candle_data_root", "")

    def update(
        self,
        enabled: bool | None = None,
        mode: str | None = None,
        fail_closed: bool | None = None,
    ) -> None:
        """Update flags at runtime (thread-safe).

        NOTE: tail_risk_threshold is FROZEN at V1 value and cannot be changed.
        """
        with self._update_lock:
            if enabled is not None:
                self._enabled = enabled
            if mode is not None:
                self._mode = mode
            if fail_closed is not None:
                self._fail_closed = fail_closed

    def to_config(self) -> RiskGuardConfig:
        """Convert current flags to a RiskGuardConfig."""
        with self._update_lock:
            return RiskGuardConfig(
                enabled=self._enabled,
                mode=self._mode,
                tail_risk_threshold=FROZEN_TAIL_RISK_THRESHOLD,
                fail_closed=self._fail_closed,
                models_joblib_path=self._models_joblib_path,
                models_joblib_sha256=self._models_joblib_sha256,
                feature_schema_path=self._feature_schema_path,
                feature_schema_sha256=self._feature_schema_sha256,
                candle_data_root=self._candle_data_root,
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize current flags."""
        with self._update_lock:
            return {
                "enabled": self._enabled,
                "mode": self._mode,
                "tail_risk_threshold": FROZEN_TAIL_RISK_THRESHOLD,
                "fail_closed": self._fail_closed,
                "models_joblib_path": self._models_joblib_path,
                "models_joblib_sha256": self._models_joblib_sha256,
                "feature_schema_path": self._feature_schema_path,
                "feature_schema_sha256": self._feature_schema_sha256,
                "candle_data_root": self._candle_data_root,
            }

    def save_to_file(self, path: str | Path) -> None:
        """Save current flags to a JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def enforce(self) -> bool:
        return self._enabled and self._mode == RiskGuardMode.ENFORCE

    @property
    def observe_only(self) -> bool:
        return self._enabled and self._mode == RiskGuardMode.OBSERVE_ONLY
