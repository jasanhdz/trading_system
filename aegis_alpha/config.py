from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent


@dataclass(frozen=True)
class RiskConfig:
    initial_balance: float = 20.0
    leverage: float = 5.0
    position_fraction: float = 0.25
    hard_stop_roe: float = 0.15
    min_hold_steps: int = 6
    min_flat_steps: int = 12
    commission_rate: float = 0.0004
    slippage: float = 0.0001

    @property
    def total_fee(self) -> float:
        return self.commission_rate + self.slippage


@dataclass(frozen=True)
class SignalGateConfig:
    min_top_prob: float = 0.65
    min_long_short_gap: float = 0.15
    chop_min_top_prob: float = 0.72
    chop_min_gap: float = 0.20


@dataclass(frozen=True)
class ModelConfig:
    name: str = "aegis_alpha"
    version: str = "0.1.0"
    champion_path: str = "aegis_alpha/models/champion/aegis_champion.zip"
    window_size: int = 64
    n_features: int = 21


@dataclass(frozen=True)
class TurboConfig:
    experimental_short: bool = False
    max_turbo_trades_per_day: int = 3
    max_consecutive_losses: int = 2


@dataclass(frozen=True)
class AegisConfig:
    symbol: str = "ETHUSDT"
    timeframe: str = "5m"
    database_url: str = "sqlite:////home/jasan/Develop/trading_system/data/binance_candles.db"
    risk: RiskConfig = RiskConfig()
    gates: SignalGateConfig = SignalGateConfig()
    model: ModelConfig = ModelConfig()
    turbo: TurboConfig = TurboConfig()


def _coerce_config(raw: dict[str, Any]) -> AegisConfig:
    risk = RiskConfig(**raw.get("risk", {}))
    gates = SignalGateConfig(**raw.get("gates", {}))
    model = ModelConfig(**raw.get("model", {}))
    turbo = TurboConfig(**raw.get("turbo", {}))
    return AegisConfig(
        symbol=raw.get("symbol", "ETHUSDT"),
        timeframe=raw.get("timeframe", "5m"),
        database_url=raw.get("database_url", AegisConfig.database_url),
        risk=risk,
        gates=gates,
        model=model,
        turbo=turbo,
    )


def load_config(path: str | Path | None = None) -> AegisConfig:
    config_path = Path(path) if path else PROJECT_ROOT / "configs" / "base.yaml"
    if not config_path.exists():
        return AegisConfig()

    try:
        import yaml
    except ImportError:
        return AegisConfig()

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _coerce_config(raw)
