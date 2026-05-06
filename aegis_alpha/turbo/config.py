from __future__ import annotations

from dataclasses import dataclass, field, replace
import logging
from pathlib import Path
import os
import threading
from typing import Any

from aegis_alpha.config import load_config
from aegis_alpha.config import REPO_ROOT


LOGGER = logging.getLogger(__name__)
TURBO_VERSION = "0.1.0"
TURBO_MODE = "TURBO_SHADOW"
DEFAULT_RUNTIME_CONFIG_PATH = REPO_ROOT / "aegis_alpha" / "configs" / "turbo.yaml"


@dataclass(frozen=True)
class TurboLeverageConfig:
    conservative: float = 15.0
    normal: float = 20.0
    premium: float = 25.0
    max_allowed: float = 30.0


@dataclass(frozen=True)
class TurboPositionConfig:
    conservative: float = 0.08
    normal: float = 0.12
    premium: float = 0.18
    max_allowed: float = 0.25


@dataclass(frozen=True)
class TurboExitConfig:
    hard_stop_roe: float = -15.0
    take_profit_roe: float = 25.0
    trailing_activation_roe: float = 15.0
    trailing_callback_roe: float = 8.0


@dataclass(frozen=True)
class TurboRiskGuardConfig:
    max_turbo_trades_per_day: int = 3
    max_consecutive_losses: int = 2
    daily_loss_stop_pct: float = 10.0
    daily_profit_lock_pct: float = 25.0
    cooldown_after_loss_steps: int = 144
    cooldown_after_two_losses_steps: int = 288


@dataclass(frozen=True)
class TurboThresholdConfig:
    min_turbo_score_conservative: float = 0.0
    min_turbo_score_shadow: float = 0.55
    min_turbo_score_premium: float = 0.70
    min_agreement_count: int = 2
    block_if_safe_regime_toxic: bool = True
    experimental_short: bool = False


@dataclass(frozen=True)
class TurboConfig:
    version: str = TURBO_VERSION
    mode: str = TURBO_MODE
    symbol: str = "ETHUSDT"
    timeframe: str = "5m"
    lookback_days: tuple[int, ...] = (7, 14, 30)
    config_path: str = "aegis_alpha/configs/base.yaml"
    model_dir: Path = REPO_ROOT / "aegis_alpha" / "models" / "turbo"
    log_dir: Path = REPO_ROOT / "aegis_alpha" / "logs" / "turbo"
    data_dir: Path = REPO_ROOT / "aegis_alpha" / "data" / "processed"
    leverage: TurboLeverageConfig = field(default_factory=TurboLeverageConfig)
    position_fraction: TurboPositionConfig = field(default_factory=TurboPositionConfig)
    exits: TurboExitConfig = field(default_factory=TurboExitConfig)
    risk: TurboRiskGuardConfig = field(default_factory=TurboRiskGuardConfig)
    thresholds: TurboThresholdConfig = field(default_factory=TurboThresholdConfig)


def _load_yaml_experimental_short() -> bool:
    cfg = load_config(os.environ.get("AEGIS_CONFIG"))
    turbo_cfg = getattr(cfg, "turbo", None)
    return bool(getattr(turbo_cfg, "experimental_short", False))


def _load_yaml_max_turbo_trades_per_day() -> int:
    cfg = load_config(os.environ.get("AEGIS_CONFIG"))
    turbo_cfg = getattr(cfg, "turbo", None)
    value = getattr(turbo_cfg, "max_turbo_trades_per_day", None)
    return int(value) if isinstance(value, int) else TurboRiskGuardConfig.max_turbo_trades_per_day


DEFAULT_TURBO_CONFIG = TurboConfig(
    thresholds=TurboThresholdConfig(experimental_short=_load_yaml_experimental_short()),
    risk=TurboRiskGuardConfig(max_turbo_trades_per_day=_load_yaml_max_turbo_trades_per_day()),
)


_RUNTIME_CONFIG_LOCK = threading.Lock()
_RUNTIME_CONFIG_CACHE: TurboConfig | None = None
_RUNTIME_CONFIG_CACHE_KEY: tuple[str, int | None, int | None] | None = None
_RUNTIME_CONFIG_WARNING_CACHE_KEY: tuple[str, int | None, int | None] | None = None


def _runtime_config_path() -> Path:
    raw = os.environ.get("AEGIS_TURBO_CONFIG")
    return Path(raw).expanduser() if raw else DEFAULT_RUNTIME_CONFIG_PATH


def _cache_key(path: Path) -> tuple[str, int | None, int | None]:
    try:
        stat = path.stat()
        return str(path.resolve()), stat.st_mtime_ns, stat.st_size
    except OSError:
        return str(path.resolve()), None, None


def clear_runtime_turbo_config_cache() -> None:
    global _RUNTIME_CONFIG_CACHE, _RUNTIME_CONFIG_CACHE_KEY, _RUNTIME_CONFIG_WARNING_CACHE_KEY
    with _RUNTIME_CONFIG_LOCK:
        _RUNTIME_CONFIG_CACHE = None
        _RUNTIME_CONFIG_CACHE_KEY = None
        _RUNTIME_CONFIG_WARNING_CACHE_KEY = None


def runtime_turbo_config_status() -> dict[str, Any]:
    path = _runtime_config_path()
    key = _cache_key(path)
    return {
        "path": str(path),
        "exists": key[1] is not None,
        "mtime_ns": key[1],
        "size": key[2],
        "hot_reload": True,
    }


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result and result not in {float("inf"), float("-inf")} else None


def _float_field(section: dict[str, Any], key: str, default: float, *, minimum: float | None = None, maximum: float | None = None) -> float:
    value = _finite_float(section.get(key))
    if value is None:
        return default
    if minimum is not None and value < minimum:
        return default
    if maximum is not None and value > maximum:
        return default
    return value


def _int_field(section: dict[str, Any], key: str, default: int, *, minimum: int | None = None) -> int:
    value = _finite_float(section.get(key))
    if value is None:
        return default
    result = int(value)
    if minimum is not None and result < minimum:
        return default
    return result


def _bool_field(section: dict[str, Any], key: str, default: bool) -> bool:
    value = section.get(key)
    return value if isinstance(value, bool) else default


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    return value if isinstance(value, dict) else {}


def _sizing_value(sizing: dict[str, Any], tier: str, key: str) -> Any:
    value = sizing.get(tier)
    if isinstance(value, dict):
        return value.get(key)
    return None


def _merge_leverage(base: TurboLeverageConfig, raw: dict[str, Any]) -> TurboLeverageConfig:
    direct = _section(raw, "leverage")
    sizing = _section(raw, "sizing")
    merged = {
        "conservative": _float_field({"value": _sizing_value(sizing, "conservative", "leverage")}, "value", _float_field(direct, "conservative", base.conservative, minimum=1.0), minimum=1.0),
        "normal": _float_field({"value": _sizing_value(sizing, "normal", "leverage")}, "value", _float_field(direct, "normal", base.normal, minimum=1.0), minimum=1.0),
        "premium": _float_field({"value": _sizing_value(sizing, "premium", "leverage")}, "value", _float_field(direct, "premium", base.premium, minimum=1.0), minimum=1.0),
        "max_allowed": _float_field(
            {"value": sizing.get("max_allowed_leverage")},
            "value",
            _float_field(direct, "max_allowed", base.max_allowed, minimum=1.0),
            minimum=1.0,
        ),
    }
    return replace(base, **merged)


def _merge_position_fraction(base: TurboPositionConfig, raw: dict[str, Any]) -> TurboPositionConfig:
    direct = _section(raw, "position_fraction")
    sizing = _section(raw, "sizing")
    merged = {
        "conservative": _float_field({"value": _sizing_value(sizing, "conservative", "position_fraction")}, "value", _float_field(direct, "conservative", base.conservative, minimum=0.0, maximum=1.0), minimum=0.0, maximum=1.0),
        "normal": _float_field({"value": _sizing_value(sizing, "normal", "position_fraction")}, "value", _float_field(direct, "normal", base.normal, minimum=0.0, maximum=1.0), minimum=0.0, maximum=1.0),
        "premium": _float_field({"value": _sizing_value(sizing, "premium", "position_fraction")}, "value", _float_field(direct, "premium", base.premium, minimum=0.0, maximum=1.0), minimum=0.0, maximum=1.0),
        "max_allowed": _float_field(
            {"value": sizing.get("max_allowed_position_fraction")},
            "value",
            _float_field(direct, "max_allowed", base.max_allowed, minimum=0.0, maximum=1.0),
            minimum=0.0,
            maximum=1.0,
        ),
    }
    return replace(base, **merged)


def _merge_thresholds(base: TurboThresholdConfig, raw: dict[str, Any]) -> TurboThresholdConfig:
    section = _section(raw, "thresholds")
    return replace(
        base,
        min_turbo_score_conservative=_float_field(section, "min_turbo_score_conservative", base.min_turbo_score_conservative, minimum=0.0, maximum=1.0),
        min_turbo_score_shadow=_float_field(section, "min_turbo_score_shadow", base.min_turbo_score_shadow, minimum=0.0, maximum=1.0),
        min_turbo_score_premium=_float_field(section, "min_turbo_score_premium", base.min_turbo_score_premium, minimum=0.0, maximum=1.0),
        min_agreement_count=_int_field(section, "min_agreement_count", base.min_agreement_count, minimum=1),
        block_if_safe_regime_toxic=_bool_field(section, "block_if_safe_regime_toxic", base.block_if_safe_regime_toxic),
        experimental_short=_bool_field(section, "experimental_short", base.experimental_short),
    )


def _merge_exits(base: TurboExitConfig, raw: dict[str, Any]) -> TurboExitConfig:
    section = _section(raw, "exits")
    return replace(
        base,
        hard_stop_roe=_float_field(section, "hard_stop_roe", base.hard_stop_roe),
        take_profit_roe=_float_field(section, "take_profit_roe", base.take_profit_roe),
        trailing_activation_roe=_float_field(section, "trailing_activation_roe", base.trailing_activation_roe, minimum=0.0),
        trailing_callback_roe=_float_field(section, "trailing_callback_roe", base.trailing_callback_roe, minimum=0.0),
    )


def _merge_risk(base: TurboRiskGuardConfig, raw: dict[str, Any]) -> TurboRiskGuardConfig:
    section = _section(raw, "risk")
    return replace(
        base,
        max_turbo_trades_per_day=_int_field(section, "max_turbo_trades_per_day", base.max_turbo_trades_per_day, minimum=0),
        max_consecutive_losses=_int_field(section, "max_consecutive_losses", base.max_consecutive_losses, minimum=1),
        daily_loss_stop_pct=_float_field(section, "daily_loss_stop_pct", base.daily_loss_stop_pct, minimum=0.0),
        daily_profit_lock_pct=_float_field(section, "daily_profit_lock_pct", base.daily_profit_lock_pct, minimum=0.0),
        cooldown_after_loss_steps=_int_field(section, "cooldown_after_loss_steps", base.cooldown_after_loss_steps, minimum=0),
        cooldown_after_two_losses_steps=_int_field(section, "cooldown_after_two_losses_steps", base.cooldown_after_two_losses_steps, minimum=0),
    )


def _coerce_runtime_config(raw: dict[str, Any]) -> TurboConfig:
    if isinstance(raw.get("turbo"), dict):
        raw = raw["turbo"]
    if raw.get("enabled") is False:
        return DEFAULT_TURBO_CONFIG
    return replace(
        DEFAULT_TURBO_CONFIG,
        leverage=_merge_leverage(DEFAULT_TURBO_CONFIG.leverage, raw),
        position_fraction=_merge_position_fraction(DEFAULT_TURBO_CONFIG.position_fraction, raw),
        thresholds=_merge_thresholds(DEFAULT_TURBO_CONFIG.thresholds, raw),
        exits=_merge_exits(DEFAULT_TURBO_CONFIG.exits, raw),
        risk=_merge_risk(DEFAULT_TURBO_CONFIG.risk, raw),
    )


def _load_runtime_turbo_config(path: Path, key: tuple[str, int | None, int | None]) -> TurboConfig:
    global _RUNTIME_CONFIG_WARNING_CACHE_KEY
    if key[1] is None:
        return DEFAULT_TURBO_CONFIG
    try:
        import yaml
    except ImportError:
        if _RUNTIME_CONFIG_WARNING_CACHE_KEY != key:
            LOGGER.warning("turbo_yaml_unavailable path=%s fallback=defaults", path)
            _RUNTIME_CONFIG_WARNING_CACHE_KEY = key
        return DEFAULT_TURBO_CONFIG
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError("turbo yaml root must be a mapping")
        return _coerce_runtime_config(raw)
    except Exception as exc:
        if _RUNTIME_CONFIG_WARNING_CACHE_KEY != key:
            LOGGER.warning("turbo_yaml_invalid path=%s fallback=defaults error=%r", path, exc)
            _RUNTIME_CONFIG_WARNING_CACHE_KEY = key
        return DEFAULT_TURBO_CONFIG


def get_runtime_turbo_config() -> TurboConfig:
    global _RUNTIME_CONFIG_CACHE, _RUNTIME_CONFIG_CACHE_KEY
    path = _runtime_config_path()
    key = _cache_key(path)
    with _RUNTIME_CONFIG_LOCK:
        if _RUNTIME_CONFIG_CACHE is not None and _RUNTIME_CONFIG_CACHE_KEY == key:
            return _RUNTIME_CONFIG_CACHE
        _RUNTIME_CONFIG_CACHE = _load_runtime_turbo_config(path, key)
        _RUNTIME_CONFIG_CACHE_KEY = key
        return _RUNTIME_CONFIG_CACHE
