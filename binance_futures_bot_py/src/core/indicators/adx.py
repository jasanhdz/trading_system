# src/core/indicators/adx.py
import math
from typing import List, Union, Dict

Number = Union[float, int]


def _rma_tail(values: List[Number], n: int) -> float:
    """
    RMA (Wilder) como en TS:
      let v = arr[0]; k = 1/n; for i>0: v = v*(1-k) + arr[i]*k
    Se asume 'values' ya recortado a los últimos n elementos.
    """
    if not values:
        return 0.0
    k = 1.0 / float(n)
    v = float(values[0])
    for x in values[1:]:
        v = v * (1.0 - k) + float(x) * k
    return v


def adx(high: List[Number], low: List[Number], close: List[Number], length: int = 14) -> Dict[str, float]:
    """
    Replica exacta de tu TS:
      - Construye TR, +DM, -DM en bucle (i = 1..)
      - RMA de las últimas 'length' muestras
      - plusDI/minusDI = 100 * (p14/tr14)
      - ADX = DX instantáneo (no re-smooth): 100 * |+DI - -DI| / (+DI + -DI)
    Con guardas anti-NaN: si hay divisiones por 0 -> regresamos 0.0.
    """
    n = min(len(high), len(low), len(close))
    if n < length + 2:
        return {"adx": 0.0, "plus_di": 0.0, "minus_di": 0.0}

    tr: List[float] = []
    plus_dm: List[float] = []
    minus_dm: List[float] = []

    for i in range(1, n):
        up = float(high[i]) - float(high[i - 1])
        dn = float(low[i - 1]) - float(low[i])
        plus_dm.append(up if (up > dn and up > 0) else 0.0)
        minus_dm.append(dn if (dn > up and dn > 0) else 0.0)

        a = float(high[i]) - float(low[i])
        b = abs(float(high[i]) - float(close[i - 1]))
        c = abs(float(low[i]) - float(close[i - 1]))
        tr.append(max(a, b, c))

    # Suavizados (últimos 'length' elementos)
    tr14 = _rma_tail(tr[-length:], length)
    p14 = _rma_tail(plus_dm[-length:], length)
    m14 = _rma_tail(minus_dm[-length:], length)

    # Evita NaN si tr14 == 0 o no es finito
    if tr14 <= 0.0 or not math.isfinite(tr14):
        return {"adx": 0.0, "plus_di": 0.0, "minus_di": 0.0}

    plus_di = 100.0 * (p14 / tr14)
    minus_di = 100.0 * (m14 / tr14)

    denom = plus_di + minus_di
    if denom <= 0.0 or not math.isfinite(denom):
        adx_val = 0.0
    else:
        adx_val = 100.0 * abs(plus_di - minus_di) / denom

    return {"adx": float(adx_val), "plus_di": float(plus_di), "minus_di": float(minus_di)}


def sma(arr: List[Number], n: int) -> float:
    k = min(len(arr), n)
    if k <= 0:
        return float("nan")
    s = 0.0
    start = len(arr) - k
    for i in range(start, len(arr)):
        s += float(arr[i])
    return s / float(k)
