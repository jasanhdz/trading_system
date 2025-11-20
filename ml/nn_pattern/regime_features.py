"""
Features de régimen de mercado para mejorar predicciones.

Estas features ayudan al modelo a identificar:
- Si el mercado está en tendencia o en rango
- Volatilidad alta vs baja
- Sesión de trading (Asia/Europa/US)
- Patrones temporales (día de semana)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List


def calculate_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula features de régimen de mercado.

    Args:
        df: DataFrame con columnas OHLCV

    Returns:
        DataFrame con features de régimen
    """
    regime = pd.DataFrame(index=df.index)

    # 1. Fuerza de Tendencia (ya existe ADX, pero agregamos variaciones)
    # ADX > 25 = tendencia fuerte, < 20 = rango
    if 'adx' in df.columns:
        regime['adx_strength'] = df['adx']
    else:
        # Calcular ADX si no existe
        regime['adx_strength'] = _calculate_adx(df)

    # Clasificación de régimen de tendencia
    regime['is_trending'] = (regime['adx_strength'] > 25).astype(float)
    regime['is_ranging'] = (regime['adx_strength'] < 20).astype(float)

    # 2. Consistencia de Tendencia
    # % de barras que se mueven en la misma dirección
    returns = df['close'].pct_change()
    regime['trend_consistency_10'] = (
        returns.rolling(10).apply(lambda x: (x > 0).sum() / len(x))
    )
    regime['trend_consistency_20'] = (
        returns.rolling(20).apply(lambda x: (x > 0).sum() / len(x))
    )

    # Tendencia alcista/bajista fuerte
    regime['strong_uptrend'] = (regime['trend_consistency_20'] > 0.65).astype(float)
    regime['strong_downtrend'] = (regime['trend_consistency_20'] < 0.35).astype(float)

    # 3. Régimen de Volatilidad
    # Volatilidad histórica en diferentes ventanas
    vol_10 = returns.rolling(10).std()
    vol_30 = returns.rolling(30).std()
    vol_60 = returns.rolling(60).std()

    # Volatilidad relativa (comparada con promedio de 60 períodos)
    regime['vol_regime_10'] = vol_10 / (vol_60 + 1e-9)
    regime['vol_regime_30'] = vol_30 / (vol_60 + 1e-9)

    # Clasificación de volatilidad
    vol_percentile = vol_30.rolling(100).apply(
        lambda x: pd.Series(x).rank().iloc[-1] / len(x)
    )
    regime['high_volatility'] = (vol_percentile > 0.75).astype(float)
    regime['low_volatility'] = (vol_percentile < 0.25).astype(float)

    # 4. Distancia desde VWAP
    # Indica si el precio está estirado o cerca del valor justo
    vwap = _calculate_vwap(df)
    regime['distance_from_vwap'] = (df['close'] - vwap) / vwap
    regime['extended_above_vwap'] = (regime['distance_from_vwap'] > 0.01).astype(float)
    regime['extended_below_vwap'] = (regime['distance_from_vwap'] < -0.01).astype(float)

    # 5. Microestructura del Mercado
    # Balance de presión compradora/vendedora
    regime['buy_pressure'] = _calculate_buy_pressure(df)
    regime['sell_pressure'] = 1 - regime['buy_pressure']

    # Imbalance (positivo = más compradores)
    regime['microstructure_imbalance'] = regime['buy_pressure'] - 0.5

    # 6. Sesión de Trading (para timeframes intraday)
    # Esto es útil para 5m y 15m
    if df.index.tz is None:
        # Asumir UTC si no hay timezone
        dt_index = pd.to_datetime(df.index, utc=True)
    else:
        dt_index = df.index

    hour_utc = dt_index.hour

    # Sesión Asiática (00:00 - 08:00 UTC)
    regime['asian_session'] = ((hour_utc >= 0) & (hour_utc < 8)).astype(float)

    # Sesión Europea (08:00 - 16:00 UTC)
    regime['european_session'] = ((hour_utc >= 8) & (hour_utc < 16)).astype(float)

    # Sesión US (16:00 - 24:00 UTC)
    regime['us_session'] = (hour_utc >= 16).astype(float)

    # Overlap Europa-US (12:00 - 16:00 UTC) - más volumen
    regime['high_volume_session'] = ((hour_utc >= 12) & (hour_utc < 16)).astype(float)

    # 7. Patrones Temporales
    day_of_week = dt_index.dayofweek

    # Día de la semana (0 = Lunes, 6 = Domingo)
    regime['day_of_week'] = day_of_week.astype(float) / 6.0  # Normalizado 0-1

    # Lunes (efecto Monday)
    regime['is_monday'] = (day_of_week == 0).astype(float)

    # Fin de semana (viernes tarde, domingo) - menos volumen en cripto
    regime['is_weekend'] = ((day_of_week == 5) | (day_of_week == 6)).astype(float)

    # 8. Momentum del Mercado
    # Comparación de precio actual vs promedios
    sma_20 = df['close'].rolling(20).mean()
    sma_50 = df['close'].rolling(50).mean()

    regime['price_vs_sma20'] = (df['close'] - sma_20) / sma_20
    regime['price_vs_sma50'] = (df['close'] - sma_50) / sma_50

    # Cruces de medias móviles (señal de cambio de régimen)
    regime['sma_cross_bullish'] = (sma_20 > sma_50).astype(float)
    regime['sma_cross_bearish'] = (sma_20 < sma_50).astype(float)

    # 9. Expansión/Contracción de Volatilidad
    # Bollinger Band width como % del precio
    if 'bb_width' in df.columns:
        regime['bb_width_pct'] = df['bb_width'] / df['close']
    else:
        bb_width = _calculate_bb_width(df)
        regime['bb_width_pct'] = bb_width / df['close']

    # Squeeze de volatilidad (BB estrecho = posible ruptura próxima)
    bb_width_ma = regime['bb_width_pct'].rolling(20).mean()
    regime['volatility_squeeze'] = (
        regime['bb_width_pct'] < bb_width_ma * 0.7
    ).astype(float)

    # Expansión de volatilidad (BB ancho = posible agotamiento)
    regime['volatility_expansion'] = (
        regime['bb_width_pct'] > bb_width_ma * 1.3
    ).astype(float)

    # 10. Fuerza Relativa del Mercado
    # ROC (Rate of Change) para medir momentum
    roc_10 = df['close'].pct_change(10)
    roc_20 = df['close'].pct_change(20)

    regime['roc_10'] = roc_10
    regime['roc_20'] = roc_20

    # Aceleración del precio (segunda derivada)
    regime['price_acceleration'] = roc_10.diff()

    # Limpiar NaN e infinitos
    regime = regime.replace([np.inf, -np.inf], np.nan)
    regime = regime.ffill()
    regime = regime.fillna(0)

    return regime


def _calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calcula Average Directional Index."""
    high = df['high']
    low = df['low']
    close = df['close']

    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    plus_dm = pd.Series(plus_dm, index=df.index).rolling(period).sum()
    minus_dm = pd.Series(minus_dm, index=df.index).rolling(period).sum()

    # Directional Indicators
    plus_di = 100 * plus_dm / atr
    minus_di = 100 * minus_dm / atr

    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    adx = dx.rolling(period).mean()

    return adx


def _calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Calcula Volume Weighted Average Price."""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    return (typical_price * df['volume']).cumsum() / df['volume'].cumsum()


def _calculate_buy_pressure(df: pd.DataFrame) -> pd.Series:
    """
    Calcula presión compradora usando la posición del cierre en el rango.

    Close cerca del High = más presión compradora
    Close cerca del Low = más presión vendedora
    """
    range_hl = df['high'] - df['low']
    range_hl = range_hl.replace(0, np.nan)  # Evitar división por cero

    buy_pressure = (df['close'] - df['low']) / range_hl
    buy_pressure = buy_pressure.fillna(0.5)  # Neutral si no hay rango

    # Suavizar con media móvil
    return buy_pressure.rolling(5).mean().fillna(0.5)


def _calculate_bb_width(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """Calcula ancho de Bollinger Bands."""
    sma = df['close'].rolling(period).mean()
    std = df['close'].rolling(period).std()

    upper = sma + (std * num_std)
    lower = sma - (std * num_std)

    return upper - lower


def get_regime_feature_names() -> List[str]:
    """Retorna lista de nombres de features de régimen."""
    return [
        # Tendencia
        'adx_strength',
        'is_trending',
        'is_ranging',
        'trend_consistency_10',
        'trend_consistency_20',
        'strong_uptrend',
        'strong_downtrend',

        # Volatilidad
        'vol_regime_10',
        'vol_regime_30',
        'high_volatility',
        'low_volatility',
        'bb_width_pct',
        'volatility_squeeze',
        'volatility_expansion',

        # Microestructura
        'distance_from_vwap',
        'extended_above_vwap',
        'extended_below_vwap',
        'buy_pressure',
        'sell_pressure',
        'microstructure_imbalance',

        # Sesiones
        'asian_session',
        'european_session',
        'us_session',
        'high_volume_session',

        # Temporal
        'day_of_week',
        'is_monday',
        'is_weekend',

        # Momentum
        'price_vs_sma20',
        'price_vs_sma50',
        'sma_cross_bullish',
        'sma_cross_bearish',
        'roc_10',
        'roc_20',
        'price_acceleration',
    ]
