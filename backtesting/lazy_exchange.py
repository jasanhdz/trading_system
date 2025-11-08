from __future__ import annotations
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Dict, List, Optional

import pandas as pd

from data.storage.database_manager import db_manager


def _to_utc_naive(ts_like) -> pd.Timestamp:
    """
    Normaliza cualquier timestamp a pandas.Timestamp UTC-naive (sin tz).
    - Si viene aware: convierte a UTC y le quita tz.
    - Si viene naive: lo deja como está (asumimos ya está en UTC).
    """
    t = pd.Timestamp(ts_like)
    if t.tz is not None:
        # asegurar UTC y luego quitar tz
        t = t.tz_convert("UTC").tz_localize(None)
    # t.tz es None => ya es naive
    return t


class LazySimExchange:
    """
    Exchange simulado con carga perezosa desde tu DB.
    Satisface lo que tus estrategias esperan:
      - get_candles(symbol, timeframe, n)
      - get_mark_price(symbol)
    No gestiona órdenes reales: el backtester hace los fills.
    """

    def __init__(
        self,
        symbol: str,
        main_timeframe: str,
        start: datetime,
        end: datetime,
        warmup_bars: int = 300,
    ):
        self.symbol = symbol
        self.main_tf = main_timeframe
        # Normaliza rangos a UTC-naive para que comparen con df.index naive
        self.start = _to_utc_naive(start)
        self.end = _to_utc_naive(end)
        self.warmup_bars = warmup_bars

        self._cache: Dict[str, pd.DataFrame] = {}
        self._idx_by_tf: Dict[str, int] = {}
        self.current_ts: Optional[pd.Timestamp] = None

        # Carga inicial del TF principal (con warmup hacia atrás)
        self._ensure_timeframe(self.main_tf)

        # Arranca el cursor en el primer timestamp del rango
        main_df = self._cache[self.main_tf]
        if main_df.empty:
            raise ValueError(f"No hay datos en DB para {self.symbol} {self.main_tf} en el rango")
        self.current_ts = main_df.index[0]
        self._sync_indices_to_ts(self.current_ts)

    def _ensure_timeframe(self, timeframe: str):
        if timeframe in self._cache:
            return
        tf_minutes = {
            "1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440,
        }.get(timeframe)
        if not tf_minutes:
            raise ValueError(f"Timeframe no soportado: {timeframe}")

        warmup_delta = timedelta(minutes=tf_minutes * self.warmup_bars)
        # start_warm en UTC-naive
        start_warm = _to_utc_naive(self.start - warmup_delta)
        end_naive = _to_utc_naive(self.end)

        df = db_manager.get_ohlcv_data(
            symbol=self.symbol,
            timeframe=timeframe,
            start_date=None,  # traemos más y filtramos aquí
            end_date=None,
        )
        if df.empty:
            raise SystemExit(f"No hay datos en DB para {self.symbol} {timeframe}")

        # Asegurar DatetimeIndex y que sea naive
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        # Si por alguna razón viniera aware, lo pasamos a UTC-naive
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)

        df = df.sort_index()

        # Filtro por rango (todo naive)
        df = df[(df.index >= start_warm) & (df.index <= end_naive)]
        if df.empty:
            raise SystemExit(
                f"No hay datos en rango para {self.symbol} {timeframe} "
                f"entre {start_warm} y {end_naive}"
            )

        self._cache[timeframe] = df
        self._idx_by_tf[timeframe] = 0

    def _sync_indices_to_ts(self, ts: datetime):
        ts_naive = _to_utc_naive(ts)
        for tf, df in self._cache.items():
            i = df.index.searchsorted(ts_naive, side="right") - 1
            self._idx_by_tf[tf] = max(0, i)

    def seek_to(self, ts: datetime):
        self.current_ts = _to_utc_naive(ts)
        self._sync_indices_to_ts(self.current_ts)

    async def get_candles(self, symbol: str, timeframe: str, n: int) -> List:
        if timeframe not in self._cache:
            self._ensure_timeframe(timeframe)
            if self.current_ts is not None:
                self._sync_indices_to_ts(self.current_ts)

        df = self._cache[timeframe]
        i = self._idx_by_tf[timeframe]
        lo = max(0, i - (n - 1))
        part = df.iloc[lo : i + 1]
        # devolvemos objetos con los campos que usa tu estrategia
        out = []
        for ts, r in part.iterrows():
            ts_py = pd.Timestamp(ts).to_pydatetime()  # naive
            out.append(
                SimpleNamespace(
                    time=ts_py,
                    timestamp=ts_py,
                    ts=ts_py,
                    open=float(r.open),
                    high=float(r.high),
                    low=float(r.low),
                    close=float(r.close),
                    volume=float(r.volume),
                )
            )
        return out

    async def get_mark_price(self, symbol: str) -> float:
        df = self._cache[self.main_tf]
        i = self._idx_by_tf[self.main_tf]
        i = min(max(0, i), len(df) - 1)
        return float(df.iloc[i]["close"])

    # Stubs para compatibilidad (no usados por el motor en esta versión)
    async def read_active_position(self, *a, **k): return None
    async def get_symbol_filters(self, *a, **k):
        return SimpleNamespace(tick_size=0.0001, price_precision=6)
    async def open_stop_for_side(self, *a, **k): return None
    async def cancel_order_by_id(self, *a, **k): return None
    async def place_stop_close(self, *a, **k): return None
    async def close_side_market_safe(self, *a, **k): return None

    def main_df(self) -> pd.DataFrame:
        return self._cache[self.main_tf]
