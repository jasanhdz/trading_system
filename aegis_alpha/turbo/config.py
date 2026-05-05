from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from aegis_alpha.config import REPO_ROOT


TURBO_VERSION = "0.1.0"
TURBO_MODE = "TURBO_SHADOW"


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


DEFAULT_TURBO_CONFIG = TurboConfig()
