# =============================================================================
# ANÁLISIS EXPLORATORIO DE DATOS - SISTEMA COMPLETO
# =============================================================================

# 1. analysis/statistics/exploratory.py
"""
Sistema completo de análisis exploratorio para datos de trading
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import jarque_bera, shapiro, normaltest
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from data.storage.database_manager import db_manager
from utils.logger import setup_logger

logger = setup_logger("eda")

class ExploratoryDataAnalysis:
    """Clase principal para análisis exploratorio de datos"""
    
    def __init__(self, symbol: str = "XRP/USDT"):
        self.symbol = symbol
        self.data = {}
        self.results = {}
        
    def load_data(self, timeframes: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Cargar datos de múltiples timeframes"""
        if timeframes is None:
            timeframes = ['1m', '5m', '15m', '1h', '4h']
            
        logger.info(f"Loading data for {self.symbol} across timeframes: {timeframes}")
        
        for tf in timeframes:
            df = db_manager.get_ohlcv_data(self.symbol, tf)
            if not df.empty:
                # Calcular returns básicos
                df['returns'] = df['close'].pct_change()
                df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
                df['volatility'] = df['returns'].rolling(20).std()
                df['volume_ma'] = df['volume'].rolling(20).mean()
                
                self.data[tf] = df
                logger.info(f"Loaded {len(df):,} records for {tf}")
            else:
                logger.warning(f"No data found for {tf}")
                
        return self.data
    
    def basic_statistics(self) -> Dict:
        """Calcular estadísticas básicas para cada timeframe"""
        logger.info("Calculating basic statistics...")
        
        stats_summary = {}
        
        for tf, df in self.data.items():
            stats_dict = {
                'records': len(df),
                'date_range': (df.index.min(), df.index.max()),
                'price_stats': {
                    'min': df['close'].min(),
                    'max': df['close'].max(),
                    'mean': df['close'].mean(),
                    'std': df['close'].std(),
                    'median': df['close'].median()
                },
                'returns_stats': {
                    'mean': df['returns'].mean(),
                    'std': df['returns'].std(),
                    'skewness': df['returns'].skew(),
                    'kurtosis': df['returns'].kurtosis(),
                    'min': df['returns'].min(),
                    'max': df['returns'].max()
                },
                'volume_stats': {
                    'mean': df['volume'].mean(),
                    'std': df['volume'].std(),
                    'median': df['volume'].median()
                }
            }
            stats_summary[tf] = stats_dict
            
        self.results['basic_stats'] = stats_summary
        return stats_summary
    
    def returns_distribution_analysis(self) -> Dict:
        """Análisis detallado de distribución de returns"""
        logger.info("Analyzing returns distribution...")
        
        distribution_results = {}
        
        for tf, df in self.data.items():
            returns = df['returns'].dropna()
            
            # Tests de normalidad
            jb_stat, jb_pvalue = jarque_bera(returns)
            shapiro_stat, shapiro_pvalue = shapiro(returns[:5000] if len(returns) > 5000 else returns)
            
            # Estadísticas descriptivas
            percentiles = np.percentile(returns, [1, 5, 10, 25, 50, 75, 90, 95, 99])
            
            distribution_results[tf] = {
                'normality_tests': {
                    'jarque_bera': {'statistic': jb_stat, 'pvalue': jb_pvalue, 'normal': jb_pvalue > 0.05},
                    'shapiro': {'statistic': shapiro_stat, 'pvalue': shapiro_pvalue, 'normal': shapiro_pvalue > 0.05}
                },
                'percentiles': {
                    '1%': percentiles[0], '5%': percentiles[1], '10%': percentiles[2],
                    '25%': percentiles[3], '50%': percentiles[4], '75%': percentiles[5],
                    '90%': percentiles[6], '95%': percentiles[7], '99%': percentiles[8]
                },
                'tail_analysis': {
                    'extreme_positive': len(returns[returns > percentiles[7]]),  # >95%
                    'extreme_negative': len(returns[returns < percentiles[1]]),  # <5%
                    'outliers_iqr': self._detect_outliers_iqr(returns)
                }
            }
            
        self.results['distribution'] = distribution_results
        return distribution_results
    
    def volatility_analysis(self) -> Dict:
        """Análisis de volatilidad y clustering"""
        logger.info("Analyzing volatility patterns...")
        
        volatility_results = {}
        
        for tf, df in self.data.items():
            returns = df['returns'].dropna()
            abs_returns = np.abs(returns)
            
            # Volatility clustering (ARCH test approximation)
            # Autocorrelación de returns cuadrados
            squared_returns = returns ** 2
            autocorr_lags = [1, 2, 5, 10, 20]
            autocorr_squared = [squared_returns.autocorr(lag) for lag in autocorr_lags]
            
            # Rolling volatility analysis
            vol_20 = returns.rolling(20).std()
            vol_50 = returns.rolling(50).std()
            
            volatility_results[tf] = {
                'volatility_stats': {
                    'mean_vol': vol_20.mean(),
                    'std_vol': vol_20.std(),
                    'max_vol': vol_20.max(),
                    'min_vol': vol_20.min()
                },
                'clustering_evidence': {
                    'autocorr_squared_returns': dict(zip(autocorr_lags, autocorr_squared)),
                    'arch_effect': any(abs(corr) > 0.1 for corr in autocorr_squared if not pd.isna(corr))
                },
                'volatility_regimes': self._identify_volatility_regimes(vol_20)
            }
            
        self.results['volatility'] = volatility_results
        return volatility_results
    
    def temporal_patterns(self) -> Dict:
        """Análisis de patrones temporales (estacionalidad)"""
        logger.info("Analyzing temporal patterns...")
        
        temporal_results = {}
        
        for tf, df in self.data.items():
            df_copy = df.copy()
            df_copy['hour'] = df_copy.index.hour
            df_copy['day_of_week'] = df_copy.index.dayofweek
            df_copy['month'] = df_copy.index.month
            
            temporal_results[tf] = {
                'hourly_patterns': {
                    'mean_returns_by_hour': df_copy.groupby('hour')['returns'].mean().to_dict(),
                    'volatility_by_hour': df_copy.groupby('hour')['returns'].std().to_dict(),
                    'volume_by_hour': df_copy.groupby('hour')['volume'].mean().to_dict()
                },
                'daily_patterns': {
                    'mean_returns_by_day': df_copy.groupby('day_of_week')['returns'].mean().to_dict(),
                    'volatility_by_day': df_copy.groupby('day_of_week')['returns'].std().to_dict()
                },
                'monthly_patterns': {
                    'mean_returns_by_month': df_copy.groupby('month')['returns'].mean().to_dict(),
                    'volatility_by_month': df_copy.groupby('month')['returns'].std().to_dict()
                }
            }
            
        self.results['temporal'] = temporal_results
        return temporal_results
    
    def correlation_analysis(self) -> Dict:
        """Análisis de correlación entre timeframes"""
        logger.info("Analyzing cross-timeframe correlations...")
        
        if len(self.data) < 2:
            logger.warning("Need at least 2 timeframes for correlation analysis")
            return {}
        
        # Preparar datos para correlación
        timeframes = list(self.data.keys())
        correlation_data = {}
        
        # Resample todos los timeframes a frecuencia común (hora)
        for tf in timeframes:
            df_hourly = self.data[tf].resample('1H').last()
            correlation_data[tf] = df_hourly
        
        # Encontrar período común
        common_start = max(df.index.min() for df in correlation_data.values())
        common_end = min(df.index.max() for df in correlation_data.values())
        
        # Calcular correlaciones
        correlations = {}
        returns_corr = pd.DataFrame()
        
        for tf in timeframes:
            df_common = correlation_data[tf][common_start:common_end]
            returns_corr[f'{tf}_returns'] = df_common['returns']
        
        correlation_matrix = returns_corr.corr()
        
        correlations = {
            'returns_correlation_matrix': correlation_matrix.to_dict(),
            'summary': {
                'highest_correlation': correlation_matrix.max().max(),
                'lowest_correlation': correlation_matrix.min().min(),
                'mean_correlation': correlation_matrix.mean().mean()
            }
        }
        
        self.results['correlations'] = correlations
        return correlations
    
    def market_regime_analysis(self) -> Dict:
        """Identificar regímenes de mercado (bull/bear/sideways)"""
        logger.info("Analyzing market regimes...")
        
        regime_results = {}
        
        for tf, df in self.data.items():
            # Calcular moving averages para identificar tendencias
            df_copy = df.copy()
            df_copy['ma_20'] = df_copy['close'].rolling(20).mean()
            df_copy['ma_50'] = df_copy['close'].rolling(50).mean()
            
            # Definir regímenes basados en MA
            conditions = [
                (df_copy['close'] > df_copy['ma_20']) & (df_copy['ma_20'] > df_copy['ma_50']),
                (df_copy['close'] < df_copy['ma_20']) & (df_copy['ma_20'] < df_copy['ma_50'])
            ]
            choices = ['bullish', 'bearish']
            df_copy['regime'] = np.select(conditions, choices, default='sideways')
            
            # Estadísticas por régimen
            regime_stats = {}
            for regime in ['bullish', 'bearish', 'sideways']:
                regime_data = df_copy[df_copy['regime'] == regime]
                if len(regime_data) > 0:
                    regime_stats[regime] = {
                        'count': len(regime_data),
                        'percentage': len(regime_data) / len(df_copy) * 100,
                        'mean_return': regime_data['returns'].mean(),
                        'volatility': regime_data['returns'].std(),
                        'avg_volume': regime_data['volume'].mean()
                    }
                else:
                    regime_stats[regime] = {'count': 0, 'percentage': 0}
            
            regime_results[tf] = regime_stats
            
        self.results['regimes'] = regime_results
        return regime_results
    
    def gap_analysis(self) -> Dict:
        """Identificar gaps en los datos"""
        logger.info("Analyzing data gaps...")
        
        gap_results = {}
        
        for tf in self.data.keys():
            gaps = db_manager.get_data_gaps(self.symbol, tf)
            gap_results[tf] = {
                'gap_count': len(gaps),
                'gaps': [(str(start), str(end)) for start, end in gaps]
            }
            
        self.results['gaps'] = gap_results
        return gap_results
    
    def generate_comprehensive_report(self) -> str:
        """Generar reporte completo de análisis"""
        logger.info("Generating comprehensive EDA report...")
        
        # Ejecutar todos los análisis si no se han ejecutado
        if not self.results:
            self.basic_statistics()
            self.returns_distribution_analysis()
            self.volatility_analysis()
            self.temporal_patterns()
            self.correlation_analysis()
            self.market_regime_analysis()
            self.gap_analysis()
        
        report = []
        report.append("="*80)
        report.append(f"ANÁLISIS EXPLORATORIO DE DATOS - {self.symbol}")
        report.append("="*80)
        
        # Estadísticas básicas
        report.append("\n1. ESTADÍSTICAS BÁSICAS")
        report.append("-" * 40)
        for tf, stats in self.results['basic_stats'].items():
            report.append(f"\n{tf.upper()} Timeframe:")
            report.append(f"  Registros: {stats['records']:,}")
            report.append(f"  Período: {stats['date_range'][0].strftime('%Y-%m-%d')} a {stats['date_range'][1].strftime('%Y-%m-%d')}")
            report.append(f"  Precio: ${stats['price_stats']['min']:.4f} - ${stats['price_stats']['max']:.4f}")
            report.append(f"  Returns promedio: {stats['returns_stats']['mean']:.6f}")
            report.append(f"  Volatilidad: {stats['returns_stats']['std']:.6f}")
            report.append(f"  Skewness: {stats['returns_stats']['skewness']:.4f}")
            report.append(f"  Kurtosis: {stats['returns_stats']['kurtosis']:.4f}")
        
        # Distribución de returns
        report.append("\n2. ANÁLISIS DE DISTRIBUCIÓN")
        report.append("-" * 40)
        for tf, dist in self.results['distribution'].items():
            report.append(f"\n{tf.upper()}:")
            jb_test = dist['normality_tests']['jarque_bera']
            report.append(f"  Normalidad (Jarque-Bera): {'NORMAL' if jb_test['normal'] else 'NO NORMAL'} (p={jb_test['pvalue']:.4f})")
            report.append(f"  VaR 5%: {dist['percentiles']['5%']:.4f}")
            report.append(f"  VaR 1%: {dist['percentiles']['1%']:.4f}")
        
        # Volatilidad
        report.append("\n3. ANÁLISIS DE VOLATILIDAD")
        report.append("-" * 40)
        for tf, vol in self.results['volatility'].items():
            report.append(f"\n{tf.upper()}:")
            report.append(f"  Volatilidad promedio: {vol['volatility_stats']['mean_vol']:.6f}")
            report.append(f"  ARCH effect: {'SÍ' if vol['clustering_evidence']['arch_effect'] else 'NO'}")
        
        # Regímenes de mercado
        report.append("\n4. REGÍMENES DE MERCADO")
        report.append("-" * 40)
        for tf, regimes in self.results['regimes'].items():
            report.append(f"\n{tf.upper()}:")
            for regime, stats in regimes.items():
                if stats['count'] > 0:
                    report.append(f"  {regime.capitalize()}: {stats['percentage']:.1f}% ({stats['count']} períodos)")
        
        # Gaps en datos
        report.append("\n5. CALIDAD DE DATOS")
        report.append("-" * 40)
        for tf, gap_info in self.results['gaps'].items():
            report.append(f"{tf.upper()}: {gap_info['gap_count']} gaps detectados")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)
    
    def _detect_outliers_iqr(self, data: pd.Series) -> Dict:
        """Detectar outliers usando método IQR"""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        
        return {
            'count': len(outliers),
            'percentage': len(outliers) / len(data) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    
    def _identify_volatility_regimes(self, volatility: pd.Series) -> Dict:
        """Identificar regímenes de alta/baja volatilidad"""
        vol_clean = volatility.dropna()
        if len(vol_clean) == 0:
            return {'high_vol_periods': 0, 'low_vol_periods': 0}
        
        median_vol = vol_clean.median()
        high_vol_periods = len(vol_clean[vol_clean > median_vol])
        low_vol_periods = len(vol_clean[vol_clean <= median_vol])
        
        return {
            'high_vol_periods': high_vol_periods,
            'low_vol_periods': low_vol_periods,
            'median_volatility': median_vol
        }
