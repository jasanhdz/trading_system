"""Unit tests for Risk Guard feature flags."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from aegis.risk_guard.domain import FROZEN_TAIL_RISK_THRESHOLD
from aegis.risk_guard.flags import RiskGuardFlags, RiskGuardMode


class TestRiskGuardFlags:
    def setup_method(self):
        RiskGuardFlags.reset()

    def teardown_method(self):
        RiskGuardFlags.reset()

    def test_default_state(self):
        flags = RiskGuardFlags()
        assert not flags.enabled
        assert flags.mode == RiskGuardMode.DISABLED
        assert not flags.enforce
        assert not flags.observe_only

    def test_singleton(self):
        f1 = RiskGuardFlags.instance()
        f2 = RiskGuardFlags.instance()
        assert f1 is f2

    def test_reset(self):
        f1 = RiskGuardFlags.instance()
        RiskGuardFlags.reset()
        f2 = RiskGuardFlags.instance()
        assert f1 is not f2

    def test_update_mode(self):
        flags = RiskGuardFlags()
        flags.update(enabled=True, mode=RiskGuardMode.ENFORCE)
        assert flags.enabled
        assert flags.mode == RiskGuardMode.ENFORCE
        assert flags.enforce
        assert not flags.observe_only

    def test_observe_only(self):
        flags = RiskGuardFlags()
        flags.update(enabled=True, mode=RiskGuardMode.OBSERVE_ONLY)
        assert flags.observe_only
        assert not flags.enforce

    def test_threshold_frozen(self):
        """Threshold is FROZEN at V1 value and cannot be changed."""
        flags = RiskGuardFlags()
        config = flags.to_config()
        assert config.tail_risk_threshold == FROZEN_TAIL_RISK_THRESHOLD
        assert config.tail_risk_threshold == 0.4522452210875323

    def test_to_config(self):
        flags = RiskGuardFlags()
        flags.update(enabled=True, mode=RiskGuardMode.ENFORCE)
        config = flags.to_config()
        assert config.enabled
        assert config.mode == RiskGuardMode.ENFORCE
        assert config.tail_risk_threshold == FROZEN_TAIL_RISK_THRESHOLD

    def test_to_dict(self):
        flags = RiskGuardFlags()
        flags.update(enabled=True, mode=RiskGuardMode.OBSERVE_ONLY)
        d = flags.to_dict()
        assert d["enabled"] is True
        assert d["mode"] == "observe_only"
        assert d["tail_risk_threshold"] == FROZEN_TAIL_RISK_THRESHOLD

    def test_save_and_load(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            flags = RiskGuardFlags()
            flags.update(enabled=True, mode=RiskGuardMode.ENFORCE)
            flags.save_to_file(path)

            RiskGuardFlags.reset()
            flags2 = RiskGuardFlags()
            flags2.load_from_file(path)

            assert flags2.enabled
            assert flags2.mode == RiskGuardMode.ENFORCE
            # Threshold is always frozen, regardless of what was in the file
            config = flags2.to_config()
            assert config.tail_risk_threshold == FROZEN_TAIL_RISK_THRESHOLD
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_nonexistent_file(self):
        flags = RiskGuardFlags()
        flags.load_from_file("/nonexistent/path.json")
        assert not flags.enabled

    def test_thread_safety(self):
        import threading

        flags = RiskGuardFlags()
        errors = []

        def updater():
            try:
                for _ in range(100):
                    flags.update(enabled=True, mode=RiskGuardMode.ENFORCE)
                    flags.update(enabled=False, mode=RiskGuardMode.DISABLED)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=updater) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors

    def test_update_rejects_invalid_mode(self):
        flags = RiskGuardFlags()
        with pytest.raises(ValueError, match="Invalid mode"):
            flags.update(mode="enfroce")

    def test_update_rejects_enabled_with_disabled(self):
        flags = RiskGuardFlags()
        with pytest.raises(ValueError, match="Contradictory"):
            flags.update(enabled=True, mode=RiskGuardMode.DISABLED)
