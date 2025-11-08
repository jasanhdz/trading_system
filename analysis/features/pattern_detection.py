# =============================================================================
# analysis/features/pattern_detection.py
"""
Sistema de detección de patrones técnicos
"""
import pandas as pd
import numpy as np
from typing import Dict
from utils.logger import setup_logger
logger = setup_logger("pattern_detection")

class PatternDetection:
    """Detección de patrones técnicos clásicos"""
    
    def __init__(self, data_with_indicators: pd.DataFrame):
        self.data = data_with_indicators
        
    def support_resistance_levels(self, window: int = 20, min_touches: int = 2) -> Dict:
        """Detectar niveles de soporte y resistencia"""
        logger.info("Detecting support and resistance levels...")
        
        df = self.data.copy()
        
        # Encontrar picos y valles
        highs = df['high'].rolling(window, center=True).max() == df['high']
        lows = df['low'].rolling(window, center=True).min() == df['low']
        
        resistance_levels = []
        support_levels = []
        
        # Agrupar niveles similares
        high_prices = df[highs]['high'].values
        low_prices = df[lows]['low'].values
        
        # Simplificado: tomar niveles más relevantes
        if len(high_prices) > 0:
            resistance_levels = [np.percentile(high_prices, p) for p in [75, 85, 95]]
        
        if len(low_prices) > 0:
            support_levels = [np.percentile(low_prices, p) for p in [25, 15, 5]]
        
        return {
            'resistance_levels': resistance_levels,
            'support_levels': support_levels,
            'current_price': df['close'].iloc[-1]
        }
    
    def breakout_detection(self) -> pd.Series:
        """Detectar breakouts de rangos"""
        df = self.data.copy()
        
        # Usar Bollinger Bands para detectar breakouts
        bb_breakout_up = (df['close'] > df['bb_upper']) & (df['close'].shift(1) <= df['bb_upper'].shift(1))
        bb_breakout_down = (df['close'] < df['bb_lower']) & (df['close'].shift(1) >= df['bb_lower'].shift(1))
        
        breakouts = pd.Series(0, index=df.index)
        breakouts[bb_breakout_up] = 1
        breakouts[bb_breakout_down] = -1
        
        return breakouts
    
    def trend_reversal_signals(self) -> pd.DataFrame:
        """Detectar señales de reversión de tendencia"""
        df = self.data.copy()
        
        signals = pd.DataFrame(index=df.index)
        
        # Divergencia RSI
        signals['rsi_bull_divergence'] = self._rsi_divergence(df, bullish=True)
        signals['rsi_bear_divergence'] = self._rsi_divergence(df, bullish=False)
        
        # MACD señales
        signals['macd_bull_cross'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        signals['macd_bear_cross'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        
        return signals
    
    def _rsi_divergence(self, df: pd.DataFrame, bullish: bool = True, period: int = 14) -> pd.Series:
        """Detectar divergencias RSI (simplificado)"""
        rsi = df['rsi_14'] if 'rsi_14' in df.columns else pd.Series(50, index=df.index)
        price = df['close']
        
        divergence = pd.Series(False, index=df.index)
        
        # Implementación simplificada
        if bullish:
            # Precio hace mínimos más bajos, RSI hace mínimos más altos
            price_low = price.rolling(period).min() == price
            rsi_rising = rsi > rsi.shift(period)
            divergence = price_low & rsi_rising
        else:
            # Precio hace máximos más altos, RSI hace máximos más bajos
            price_high = price.rolling(period).max() == price
            rsi_falling = rsi < rsi.shift(period)
            divergence = price_high & rsi_falling
        
        return divergence