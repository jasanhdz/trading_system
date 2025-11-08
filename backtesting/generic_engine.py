from __future__ import annotations

import importlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Optional, Type
import asyncio
import inspect

import pandas as pd

from data.storage.database_manager import db_manager
from backtesting.lazy_exchange import LazySimExchange


def _to_utc_naive(ts_like) -> pd.Timestamp:
    """Normaliza cualquier timestamp a pandas.Timestamp *UTC-naive* (sin tz)."""
    t = pd.Timestamp(ts_like)
    if t.tz is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t


@dataclass
class EngineParams:
    symbol: str
    timeframe: str
    leverage: int = 1
    days: int = 30
    warmup_bars: int = 300
    start: Optional[datetime] = None
    end: Optional[datetime] = None


def _build_config_fallback(leverage: int, timeframe: str):
    """
    Config mínima para backtest si no podemos importar la oficial.
    Ajusta aquí defaults razonables usados por mean_reversion.
    """
    return SimpleNamespace(
        # Core
        LEVERAGE=leverage,
        ENTRY_TIMEFRAME=timeframe,
        VOL_AVG_LEN=20,

        # Filtros de rango
        MR_ADX_MAX=28.0,
        MR_BB_WIDTH_MAX=0.020,
        MR_SPIKE_VOL_FACTOR=3.0,

        # Touch bands
        MR_TOUCH_EPS=0.005,

        # Streaks
        MR_MIN_STREAK=2,

        # RSI gates
        MR_RSI_LOW=32.0,
        MR_RSI_HIGH=68.0,

        # Sides
        ALLOW_LONGS=True,
        ALLOW_SHORTS=True,

        # Shorts estrictos (confirmación 1H opcional)
        MR_STRICT_SHORTS=False,
        MR_SHORT_CONFIRM_1H=False,
        MR_SHORT_1H_ADX_MIN=18.0,
    )


class _AsyncExchangeAdapter:
    """
    Adaptador 'async' sobre el exchange simulado. Si el core es async,
    se hace await; si es sync, se devuelve directo.
    """
    def __init__(self, core):
        self._core = core

    def seek_to(self, ts):
        res = self._core.seek_to(ts)
        # Si algún día seek_to fuera async, resolvemos aquí:
        if inspect.isawaitable(res):
            # Estamos en un contexto síncrono; ejecutamos la corutina.
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                # No deberíamos entrar aquí en este motor, pero por si acaso:
                return asyncio.run(res)
            else:
                return asyncio.run(res)

    async def get_candles(self, symbol: str, timeframe: str, limit: int = 300):
        res = self._core.get_candles(symbol, timeframe, limit)
        return await res if inspect.isawaitable(res) else res

    async def get_mark_price(self, symbol: str):
        res = self._core.get_mark_price(symbol)
        return await res if inspect.isawaitable(res) else res

    # Métodos no críticos para este backtest: devolvemos algo seguro
    async def read_active_position(self, *_, **__):
        return None

    async def get_symbol_filters(self, *_, **__):
        return SimpleNamespace(tick_size=0.0001, price_precision=6)

    async def open_stop_for_side(self, *_, **__):
        return None

    async def close_side_market_safe(self, *_, **__):
        return None

    async def cancel_order_by_id(self, *_, **__):
        return None

    async def place_stop_close(self, *_, **__):
        return None

class GenericBacktester:
    """
    Backtester genérico por estrategia (válido para *cualquier* clase con .evaluate()).
    Soporta estrategias sync o async.
    """

    def __init__(self, strategy_dotted: str, params: EngineParams):
        self.params = params

        # 1) Normaliza start/end a UTC-naive
        end = params.end or datetime.now(timezone.utc).replace(tzinfo=None)
        start = params.start or (end - timedelta(days=params.days))
        self.start = _to_utc_naive(start)
        self.end = _to_utc_naive(end)

        # 2) Carga DF desde la base y asegura índice UTC-naive
        df = db_manager.get_ohlcv_data(params.symbol, params.timeframe)
        if df.empty:
            raise SystemExit(f"No hay datos para {params.symbol} {params.timeframe} en la DB.")

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)

        df = df.sort_index()
        df = df[(df.index >= self.start) & (df.index <= self.end)]
        if df.empty:
            raise SystemExit(
                f"No hay datos en rango para {params.symbol} {params.timeframe} "
                f"entre {self.start} y {self.end}"
            )
        self.df = df

        # 3) Instancia estrategia
        self.strategy = self._load_strategy(strategy_dotted)

        # 4) Exchange simulado y adaptador async
        core_ex = LazySimExchange(
            symbol=params.symbol,
            main_timeframe=params.timeframe,
            start=self.start.to_pydatetime(),
            end=self.end.to_pydatetime(),
            warmup_bars=params.warmup_bars,
        )
        self.exchange = _AsyncExchangeAdapter(core_ex)

        # 5) Config: intenta importar la oficial; si no, usa fallback
        self.config = self._load_config_or_fallback(leverage=params.leverage, timeframe=params.timeframe)

        # 6) Estado/Logger mínimos
        self.state = SimpleNamespace(mode=None, last_side=None, last_entry_price=None, peak_roe=None)
        self.logger = SimpleNamespace(
            debug=lambda *a, **k: None,
            info=lambda *a, **k: None,
            warn=lambda *a, **k: None,
            error=lambda *a, **k: None,
        )

    def _load_strategy(self, dotted: str):
        mod_name, cls_name = dotted.rsplit(".", 1)
        mod = importlib.import_module(mod_name)
        cls: Type = getattr(mod, cls_name)
        return cls()

    def _load_config_or_fallback(self, leverage: int, timeframe: str):
        try:
            # Si está tu CONFIG real, úsala
            from binance_futures_bot_py.src.infra.config import CONFIG as REAL
            # Clona en SimpleNamespace para evitar efectos colaterales
            d = {k: getattr(REAL, k) for k in dir(REAL) if k.isupper()}
            d["LEVERAGE"] = leverage  # respeta parámetro de CLI
            d["ENTRY_TIMEFRAME"] = timeframe
            return SimpleNamespace(**d)
        except Exception:
            # Fallback minimalista
            return _build_config_fallback(leverage, timeframe)

    def run(self) -> pd.DataFrame:
        results = []

        for ts in self.df.index:
            # Mueve el exchange al tiempo actual (sincrónico en el core)
            self.exchange.seek_to(ts)

            # Llama a la estrategia (sync o async)
            signal = self._call_strategy(ts)

            if isinstance(signal, dict) and signal.get("action") and signal["action"] != "IDLE":
                results.append(
                    {
                        "timestamp": ts,
                        "action": signal.get("action"),
                        "reason": signal.get("reason", ""),
                        "price": float(self.df.loc[ts, "close"]),
                    }
                )

        if not results:
            # Devuelve DF vacío coherente
            return pd.DataFrame(columns=["action", "reason", "price"])

        return pd.DataFrame(results).set_index("timestamp")

    def _call_strategy(self, ts):
        now_ms = int(pd.Timestamp(ts).timestamp() * 1000)
        maybe = self.strategy.evaluate(
            self.params.symbol,
            self.exchange,
            self.config,      # <— ya no es None
            self.state,
            now_ms,
            self.logger,
        )
        if inspect.isawaitable(maybe):
            return asyncio.run(maybe)
        return maybe
