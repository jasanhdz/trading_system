# =============================================================================
# FEATURE ENGINEERING SYSTEM - INDICADORES TÉCNICOS AVANZADOS
# =============================================================================

# analysis/features/technical_indicators.py
"""
Sistema completo de indicadores técnicos para trading algorítmico
"""
import pandas as pd
import numpy as np
from scipy.stats import zscore
from typing import Tuple
import talib
from utils.logger import setup_logger
from config.settings import settings

logger = setup_logger("technical_indicators")

class TechnicalIndicators:
    """Clase principal para cálculo de indicadores técnicos"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Inicializar con datos OHLCV
        
        Args:
            data: DataFrame con columnas ['open', 'high', 'low', 'close', 'volume']
        """
        self.data = data.copy()
        self.validate_data()
        
    def validate_data(self):
        """Validar que los datos tienen las columnas necesarias"""
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if len(self.data) < 50:
            logger.warning("Dataset has less than 50 records. Some indicators may not be reliable.")
    
    def momentum_indicators(self) -> pd.DataFrame:
        """Calcular indicadores de momentum"""
        logger.info("Calculating momentum indicators...")
        
        df = self.data.copy()
        
        # RSI - Relative Strength Index
        df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
        df['rsi_7'] = talib.RSI(df['close'], timeperiod=7)
        df['rsi_21'] = talib.RSI(df['close'], timeperiod=21)
        
        # Stochastic
        df['stoch_k'], df['stoch_d'] = talib.STOCH(
            df['high'], df['low'], df['close'],
            fastk_period=14, slowk_period=3, slowd_period=3
        )
        
        # Williams %R
        df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # ROC - Rate of Change
        df['roc_10'] = talib.ROC(df['close'], timeperiod=10)
        df['roc_20'] = talib.ROC(df['close'], timeperiod=20)
        
        # Momentum
        df['momentum_10'] = talib.MOM(df['close'], timeperiod=10)
        
        # CCI - Commodity Channel Index
        df['cci_14'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
        df['cci_20'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=20)
        
        return df
    
    def trend_indicators(self) -> pd.DataFrame:
        """Calcular indicadores de tendencia"""
        logger.info("Calculating trend indicators...")
        
        df = self.data.copy()
        
        # Moving Averages
        df['sma_10'] = talib.SMA(df['close'], timeperiod=10)
        df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
        df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
        df['sma_100'] = talib.SMA(df['close'], timeperiod=100)
        df['sma_200'] = talib.SMA(df['close'], timeperiod=200)
        
        # Exponential Moving Averages
        df['ema_10'] = talib.EMA(df['close'], timeperiod=10)
        df['ema_20'] = talib.EMA(df['close'], timeperiod=20)
        df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_histogram'] = talib.MACD(
            df['close'], fastperiod=12, slowperiod=26, signalperiod=9
        )
        
        # ADX - Average Directional Index
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        df['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        df['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Parabolic SAR
        df['sar'] = talib.SAR(df['high'], df['low'], acceleration=0.02, maximum=0.2)
        
        # Aroon
        df['aroon_up'], df['aroon_down'] = talib.AROON(
            df['high'], df['low'], timeperiod=14
        )
        df['aroon_oscillator'] = df['aroon_up'] - df['aroon_down']
        
        # TRIX
        df['trix'] = talib.TRIX(df['close'], timeperiod=14)
        
        return df
    
    def volatility_indicators(self) -> pd.DataFrame:
        """Calcular indicadores de volatilidad"""
        logger.info("Calculating volatility indicators...")
        
        df = self.data.copy()
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            df['close'], timeperiod=20, nbdevup=2, nbdevdn=2
        )
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Keltner Channels
        df['kc_upper'], df['kc_middle'], df['kc_lower'] = self._keltner_channels(df)
        df['kc_width'] = (df['kc_upper'] - df['kc_lower']) / df['kc_middle']
        
        # Average True Range
        df['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['atr_20'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=20)
        
        # Normalized ATR (for risk management)
        df['natr_14'] = df['atr_14'] / df['close']
        df['natr_20'] = df['atr_20'] / df['close']
        
        # Historical Volatility
        df['returns'] = df['close'].pct_change()
        df['hist_vol_10'] = df['returns'].rolling(10).std() * np.sqrt(252)
        df['hist_vol_20'] = df['returns'].rolling(20).std() * np.sqrt(252)
        df['hist_vol_50'] = df['returns'].rolling(50).std() * np.sqrt(252)
        
        # Volatility Ratio
        df['vol_ratio'] = df['hist_vol_10'] / df['hist_vol_50']
        
        return df
    
    def volume_indicators(self) -> pd.DataFrame:
        """Calcular indicadores de volumen"""
        logger.info("Calculating volume indicators...")
        
        df = self.data.copy()
        
        # Volume Moving Averages
        df['volume_sma_10'] = talib.SMA(df['volume'], timeperiod=10)
        df['volume_sma_20'] = talib.SMA(df['volume'], timeperiod=20)
        df['volume_sma_50'] = talib.SMA(df['volume'], timeperiod=50)
        
        # Volume Ratios
        df['volume_ratio_10'] = df['volume'] / df['volume_sma_10']
        df['volume_ratio_20'] = df['volume'] / df['volume_sma_20']
        
        # On Balance Volume
        df['obv'] = talib.OBV(df['close'], df['volume'])
        df['obv_sma'] = talib.SMA(df['obv'], timeperiod=20)
        
        # Accumulation/Distribution Line
        df['ad_line'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
        
        # Chaikin Money Flow
        df['cmf'] = self._chaikin_money_flow(df, period=20)
        
        # Volume Price Trend
        df['vpt'] = self._volume_price_trend(df)
        
        # Money Flow Index
        df['mfi'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
        
        return df
    
    def price_action_indicators(self) -> pd.DataFrame:
        """Calcular indicadores de price action"""
        logger.info("Calculating price action indicators...")
        
        df = self.data.copy()
        
        # Typical Price
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Weighted Close Price
        df['weighted_close'] = (df['high'] + df['low'] + 2 * df['close']) / 4
        
        # Price ranges
        df['high_low_range'] = df['high'] - df['low']
        df['close_open_range'] = df['close'] - df['open']
        
        # Candle patterns (simplified)
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        df['body_ratio'] = df['body_size'] / df['high_low_range']
        
        # Price position within range
        df['price_position'] = (df['close'] - df['low']) / df['high_low_range']
        
        # Gap detection
        df['gap_up'] = (df['low'] > df['high'].shift(1)).astype(int)
        df['gap_down'] = (df['high'] < df['low'].shift(1)).astype(int)
        
        # Inside/Outside bars
        df['inside_bar'] = ((df['high'] <= df['high'].shift(1)) & 
                           (df['low'] >= df['low'].shift(1))).astype(int)
        df['outside_bar'] = ((df['high'] >= df['high'].shift(1)) & 
                            (df['low'] <= df['low'].shift(1))).astype(int)
        
        return df
    
    def statistical_indicators(self) -> pd.DataFrame:
        """Calcular indicadores estadísticos avanzados - CORREGIDO"""
        logger.info("Calculating statistical indicators...")
        
        df = self.data.copy()
        
        # Z-Score
        df['zscore_20'] = df['close'].rolling(20).apply(lambda x: zscore(x)[-1] if len(x) == 20 else np.nan)
        df['zscore_50'] = df['close'].rolling(50).apply(lambda x: zscore(x)[-1] if len(x) == 50 else np.nan)
        
        # Percentile ranks
        df['percentile_20'] = df['close'].rolling(20).rank(pct=True)
        df['percentile_50'] = df['close'].rolling(50).rank(pct=True)
        
        # Linear regression - CORREGIDO: Calcular cada métrica por separado
        df['lr_slope_20'] = df['close'].rolling(20).apply(lambda x: self._calculate_slope(x), raw=False)
        df['lr_slope_50'] = df['close'].rolling(50).apply(lambda x: self._calculate_slope(x), raw=False)
        df['lr_r2_20'] = df['close'].rolling(20).apply(lambda x: self._calculate_r2(x), raw=False)
        df['lr_r2_50'] = df['close'].rolling(50).apply(lambda x: self._calculate_r2(x), raw=False)
        
        # Intercept (simplified)
        df['lr_intercept_20'] = df['close'].rolling(20).mean()  # Approximación
        df['lr_intercept_50'] = df['close'].rolling(50).mean()  # Approximación
        
        # Hurst Exponent (simplified)
        df['hurst_20'] = df['close'].rolling(20).apply(self._hurst_exponent, raw=False)
        
        # Skewness and Kurtosis
        if 'returns' not in df.columns:
            df['returns'] = df['close'].pct_change()
            
        df['skewness_20'] = df['returns'].rolling(20).skew()
        df['kurtosis_20'] = df['returns'].rolling(20).kurt()
        
        # Autocorrelation
        df['autocorr_1'] = df['returns'].rolling(20).apply(lambda x: x.autocorr(lag=1) if len(x) > 1 else np.nan)
        df['autocorr_5'] = df['returns'].rolling(50).apply(lambda x: x.autocorr(lag=5) if len(x) > 5 else np.nan)
        
        return df
    
    def risk_management_indicators(self) -> pd.DataFrame:
      """Calcular indicadores específicos para risk management - LEVERAGE DINÁMICO"""
      logger.info("Calculating risk management indicators...")
      
      df = self.data.copy()
      
      # Calcular ATR internamente
      df['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
      df['atr_20'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=20)
      df['natr_14'] = df['atr_14'] / df['close']
      df['natr_20'] = df['atr_20'] / df['close']
      
      # Calcular returns para VaR
      df['returns'] = df['close'].pct_change()
      
      # Dynamic ATR-based stops
      df['atr_stop_long'] = df['close'] - (2 * df['atr_14'])
      df['atr_stop_short'] = df['close'] + (2 * df['atr_14'])
      
      # Volatility-adjusted position sizing
      df['vol_position_size'] = 0.01 / df['natr_14']  # 1% risk per trade
      df['vol_position_size'] = df['vol_position_size'].clip(upper=1.0)  # Max 100% allocation
      
      # Risk-Reward Ratios
      df['rr_ratio_2'] = df['atr_14'] * 2 / df['atr_14']  # 2:1 RR target
      df['rr_ratio_3'] = df['atr_14'] * 3 / df['atr_14']  # 3:1 RR target
      
      # Value at Risk approximation
      df['var_1pct'] = df['close'] * df['returns'].rolling(20).quantile(0.01)
      df['var_5pct'] = df['close'] * df['returns'].rolling(20).quantile(0.05)
      
      # Maximum Adverse Excursion estimation
      df['mae_estimate'] = df['atr_14'] * 1.5  # Conservative MAE estimate
      
      # CORREGIDO: Leverage dinámico desde configuración
      leverage = settings.LEVERAGE  # Lee dinámicamente del .env
      logger.info(f"Using dynamic leverage: {leverage}x from configuration")
      
      df['leverage_risk'] = df['natr_14'] * leverage
      df['liquidation_distance'] = 1 / leverage
      df['risk_warning'] = (df['natr_14'] * leverage > 0.5).astype(int)  # Warning when risk > 50%
      
      return df

    def regime_detection_indicators(self) -> pd.DataFrame:
        """Indicadores para detección de regímenes de mercado - CORREGIDO"""
        logger.info("Calculating regime detection indicators...")
        
        df = self.data.copy()
        
        # CORREGIDO: Calcular métricas necesarias internamente
        # Calcular returns y volatilidad histórica
        df['returns'] = df['close'].pct_change()
        df['hist_vol_20'] = df['returns'].rolling(20).std() * np.sqrt(252)
        
        # Calcular ADX internamente
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Calcular EMAs internamente
        df['ema_20'] = talib.EMA(df['close'], timeperiod=20)
        df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
        
        # Trend strength
        df['trend_strength'] = abs(df['adx'])
        
        # Volatility regime
        vol_median = df['hist_vol_20'].rolling(100).median()
        df['vol_regime'] = (df['hist_vol_20'] > vol_median).astype(int)
        
        # Price regime (bull/bear/sideways)
        ma_short = df['ema_20']
        ma_long = df['ema_50']
        
        conditions = [
            (df['close'] > ma_short) & (ma_short > ma_long),
            (df['close'] < ma_short) & (ma_short < ma_long)
        ]
        df['price_regime'] = np.select(conditions, [1, -1], default=0)  # 1=bull, -1=bear, 0=sideways
        
        # Market efficiency (random walk test)
        df['efficiency_ratio'] = self._efficiency_ratio(df['close'], period=20)
        
        return df
    
    def calculate_all_indicators(self) -> pd.DataFrame:
        """Calcular todos los indicadores de una vez"""
        logger.info("Calculating all technical indicators...")
        
        # Empezar con datos base
        result = self.data.copy()
        
        # Calcular todos los grupos de indicadores
        momentum_df = self.momentum_indicators()
        trend_df = self.trend_indicators() 
        volatility_df = self.volatility_indicators()
        volume_df = self.volume_indicators()
        price_action_df = self.price_action_indicators()
        statistical_df = self.statistical_indicators()
        risk_df = self.risk_management_indicators()
        regime_df = self.regime_detection_indicators()
        
        # Combinar todos los DataFrames
        dfs_to_merge = [
            momentum_df, trend_df, volatility_df, volume_df, 
            price_action_df, statistical_df, risk_df, regime_df
        ]
        
        for df in dfs_to_merge:
            # Merge solo las columnas nuevas
            new_cols = [col for col in df.columns if col not in result.columns]
            if new_cols:
                result = result.join(df[new_cols])
        
        logger.info(f"Calculated {len(result.columns) - 6} technical indicators")  # -6 for OHLCV + returns
        
        return result
    
    # Helper methods
    def _keltner_channels(self, df: pd.DataFrame, period: int = 20, multiplier: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calcular Keltner Channels"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        middle = typical_price.rolling(period).mean()
        atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=period)
        
        upper = middle + (multiplier * atr)
        lower = middle - (multiplier * atr)
        
        return upper, middle, lower
    
    def _chaikin_money_flow(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calcular Chaikin Money Flow"""
        mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        mf_volume = mf_multiplier * df['volume']
        cmf = mf_volume.rolling(period).sum() / df['volume'].rolling(period).sum()
        return cmf
    
    def _volume_price_trend(self, df: pd.DataFrame) -> pd.Series:
        """Calcular Volume Price Trend"""
        price_change = df['close'].pct_change()
        vpt = (price_change * df['volume']).cumsum()
        return vpt
    
    def _calculate_slope(self, y):
        """Calcular slope de regresión lineal - NUEVO"""
        try:
            if len(y) < 2:
                return np.nan
            x = np.arange(len(y))
            slope = np.polyfit(x, y, 1)[0]
            return slope
        except:
            return np.nan
    
    def _calculate_r2(self, y):
        """Calcular R² de regresión lineal - NUEVO"""
        try:
            if len(y) < 2:
                return np.nan
            x = np.arange(len(y))
            correlation_matrix = np.corrcoef(x, y)
            correlation = correlation_matrix[0, 1]
            r2 = correlation ** 2
            return r2 if not np.isnan(r2) else 0.0
        except:
            return 0.0
    
    def _hurst_exponent(self, ts):
        """Calcular Hurst Exponent simplificado"""
        try:
            if len(ts) < 10:
                return np.nan
            
            lags = range(2, min(len(ts)//2, 20))
            tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        except:
            return np.nan
    
    def _efficiency_ratio(self, series: pd.Series, period: int) -> pd.Series:
        """Calcular Efficiency Ratio de Kaufman"""
        change = abs(series.diff(period))
        volatility = abs(series.diff()).rolling(period).sum()
        er = change / volatility
        return er.fillna(0)