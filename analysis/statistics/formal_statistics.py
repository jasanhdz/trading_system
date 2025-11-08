# =============================================================================
# ANÁLISIS ESTADÍSTICO FORMAL - VERSIÓN CORREGIDA
# analysis/statistics/formal_statistics.py
# =============================================================================
"""
Análisis estadístico formal para series temporales financieras - Versión corregida
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import jarque_bera, normaltest
import warnings
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')

# Importaciones statsmodels con manejo de errores
try:
    from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: statsmodels not fully available: {e}")
    print("Install with: pip install statsmodels>=0.14.0")
    STATSMODELS_AVAILABLE = False

from data.storage.database_manager import db_manager
from utils.logger import setup_logger

logger = setup_logger("formal_statistics")

class FormalStatisticalAnalysis:
    """Análisis estadístico formal para series temporales financieras"""
    
    def __init__(self, symbol: str = "XRP/USDT"):
        self.symbol = symbol
        self.data = {}
        self.results = {}
        
        if not STATSMODELS_AVAILABLE:
            logger.error("statsmodels is required for formal statistical analysis")
            logger.error("Install with: pip install statsmodels>=0.14.0")
            raise ImportError("statsmodels package is required")
        
    def load_and_prepare_data(self, timeframes: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Cargar y preparar datos para análisis estadístico"""
        if timeframes is None:
            timeframes = ['1m', '5m', '15m', '1h', '4h']
            
        logger.info(f"Loading data for formal statistical analysis...")
        
        for tf in timeframes:
            df = db_manager.get_ohlcv_data(self.symbol, tf)
            if not df.empty:
                # Calcular returns y log returns
                df['returns'] = df['close'].pct_change()
                df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
                df['abs_returns'] = np.abs(df['returns'])
                df['squared_returns'] = df['returns'] ** 2
                
                # Remover outliers extremos (>5 std) para análisis más robusto
                returns_clean = df['returns'].dropna()
                threshold = 5 * returns_clean.std()
                df['returns_clean'] = df['returns'].clip(-threshold, threshold)
                df['log_returns_clean'] = np.log((1 + df['returns_clean']).clip(lower=0.0001))
                
                self.data[tf] = df
                logger.info(f"Loaded {len(df):,} records for {tf}")
            else:
                logger.warning(f"No data found for {tf}")
                
        return self.data
    
    def autocorrelation_analysis(self, max_lags: int = 100) -> Dict:
        """Análisis formal de autocorrelación (ACF/PACF)"""
        logger.info(f"Performing autocorrelation analysis up to {max_lags} lags...")
        
        autocorr_results = {}
        
        for tf, df in self.data.items():
            logger.info(f"Analyzing {tf} timeframe...")
            
            # Preparar series limpias
            returns = df['returns_clean'].dropna()
            abs_returns = df['abs_returns'].dropna()
            squared_returns = df['squared_returns'].dropna()
            
            if len(returns) < max_lags + 50:
                logger.warning(f"Insufficient data for {tf}. Using available data.")
                max_lags_tf = min(max_lags, len(returns) // 4)
            else:
                max_lags_tf = max_lags
            
            try:
                # ACF y PACF para returns
                acf_returns, acf_confint = acf(returns, nlags=max_lags_tf, alpha=0.05)
                pacf_returns, pacf_confint = pacf(returns, nlags=max_lags_tf, alpha=0.05)
                
                # ACF para returns absolutos (test de dependencia)
                acf_abs_returns, acf_abs_confint = acf(abs_returns, nlags=max_lags_tf, alpha=0.05)
                
                # ACF para returns cuadrados (test ARCH)
                acf_squared_returns, acf_squared_confint = acf(squared_returns, nlags=max_lags_tf, alpha=0.05)
                
                # Ljung-Box test para autocorrelación
                lb_returns = acorr_ljungbox(returns, lags=min(20, max_lags_tf//5), return_df=True)
                lb_squared = acorr_ljungbox(squared_returns, lags=min(20, max_lags_tf//5), return_df=True)
                
                # Identificar lags significativos
                significant_acf_lags = self._find_significant_lags(acf_returns, acf_confint)
                significant_pacf_lags = self._find_significant_lags(pacf_returns, pacf_confint)
                significant_arch_lags = self._find_significant_lags(acf_squared_returns, acf_squared_confint)
                
                autocorr_results[tf] = {
                    'acf_returns': acf_returns,
                    'pacf_returns': pacf_returns,
                    'acf_abs_returns': acf_abs_returns,
                    'acf_squared_returns': acf_squared_returns,
                    'acf_confint': acf_confint,
                    'pacf_confint': pacf_confint,
                    'acf_abs_confint': acf_abs_confint,
                    'acf_squared_confint': acf_squared_confint,
                    'ljung_box_returns': lb_returns,
                    'ljung_box_squared': lb_squared,
                    'significant_acf_lags': significant_acf_lags,
                    'significant_pacf_lags': significant_pacf_lags,
                    'significant_arch_lags': significant_arch_lags,
                    'max_lags': max_lags_tf,
                    'summary': {
                        'has_autocorr_structure': len(significant_acf_lags) > 0,
                        'has_arch_effects': len(significant_arch_lags) > 0,
                        'suggested_ar_order': self._suggest_ar_order(significant_pacf_lags),
                        'suggested_ma_order': self._suggest_ma_order(significant_acf_lags),
                    }
                }
                
                logger.info(f"{tf} - Significant ACF lags: {significant_acf_lags[:5]}...")
                logger.info(f"{tf} - Significant PACF lags: {significant_pacf_lags[:5]}...")
                logger.info(f"{tf} - Significant ARCH lags: {significant_arch_lags[:5]}...")
                
            except Exception as e:
                logger.error(f"Error in autocorrelation analysis for {tf}: {e}")
                autocorr_results[tf] = {'error': str(e)}
        
        self.results['autocorrelation'] = autocorr_results
        return autocorr_results
    
    def arch_test_analysis(self, max_lags: int = 20) -> Dict:
        """Test formal de efectos ARCH (heterocedasticidad)"""
        logger.info("Performing formal ARCH tests...")
        
        arch_results = {}
        
        for tf, df in self.data.items():
            logger.info(f"ARCH testing for {tf}...")
            
            returns = df['returns_clean'].dropna()
            
            # Test ARCH para diferentes lags
            arch_tests = {}
            for lag in [1, 5, 10, 15, 20]:
                if lag < len(returns) // 10:  # Ensure sufficient data
                    try:
                        lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(returns, nlags=lag)
                        arch_tests[lag] = {
                            'lm_statistic': lm_stat,
                            'lm_pvalue': lm_pvalue,
                            'f_statistic': f_stat,
                            'f_pvalue': f_pvalue,
                            'has_arch_effects': lm_pvalue < 0.05
                        }
                    except Exception as e:
                        logger.warning(f"ARCH test failed for lag {lag} in {tf}: {e}")
                        continue
            
            # Engle's original ARCH test (lag 1)
            engle_test = arch_tests.get(1, {})
            
            # Test de autocorrelación en returns cuadrados (método alternativo)
            squared_returns = returns ** 2
            try:
                lb_squared_extended = acorr_ljungbox(squared_returns, lags=max_lags, return_df=True)
            except:
                lb_squared_extended = pd.DataFrame()
                logger.warning(f"Extended Ljung-Box test failed for {tf}")
            
            # Análisis de volatility clustering
            vol_periods = self._identify_volatility_periods(returns)
            
            arch_results[tf] = {
                'arch_tests': arch_tests,
                'engle_test': engle_test,
                'ljung_box_squared_extended': lb_squared_extended,
                'volatility_periods': vol_periods,
                'summary': {
                    'has_strong_arch_effects': any(test.get('has_arch_effects', False) 
                                                 for test in arch_tests.values()),
                    'optimal_arch_lag': self._find_optimal_arch_lag(arch_tests),
                    'arch_strength': self._assess_arch_strength(arch_tests),
                    'volatility_clustering_score': vol_periods['clustering_score']
                }
            }
            
            # Log resultados principales
            strong_arch = arch_results[tf]['summary']['has_strong_arch_effects']
            optimal_lag = arch_results[tf]['summary']['optimal_arch_lag']
            cluster_score = vol_periods['clustering_score']
            
            logger.info(f"{tf} - Strong ARCH effects: {strong_arch}")
            logger.info(f"{tf} - Optimal ARCH lag: {optimal_lag}")
            logger.info(f"{tf} - Volatility clustering score: {cluster_score:.3f}")
            
            # ADVERTENCIA DE RIESGO para leverage alto
            if strong_arch and cluster_score > 0.3:
                logger.warning(f"{tf} - HIGH VOLATILITY CLUSTERING DETECTED!")
                logger.warning(f"{tf} - Your 30x leverage is EXTREMELY DANGEROUS during volatility clusters")
                logger.warning(f"{tf} - Consider reducing leverage to 5x or lower")
        
        self.results['arch_tests'] = arch_results
        return arch_results
    
    def stationarity_tests(self) -> Dict:
        """Tests de estacionariedad (ADF, KPSS)"""
        logger.info("Performing stationarity tests...")
        
        stationarity_results = {}
        
        for tf, df in self.data.items():
            logger.info(f"Stationarity testing for {tf}...")
            
            # Test en prices (levels)
            prices = df['close'].dropna()
            returns = df['returns_clean'].dropna()
            log_prices = np.log(prices)
            
            try:
                # Augmented Dickey-Fuller test
                adf_prices = adfuller(prices, autolag='AIC')
                adf_returns = adfuller(returns, autolag='AIC')
                adf_log_prices = adfuller(log_prices, autolag='AIC')
                
                # KPSS test
                kpss_prices = kpss(prices, regression='ct')
                kpss_returns = kpss(returns, regression='c')
                kpss_log_prices = kpss(log_prices, regression='ct')
                
                stationarity_results[tf] = {
                    'adf_tests': {
                        'prices': {
                            'statistic': adf_prices[0],
                            'pvalue': adf_prices[1],
                            'critical_values': adf_prices[4],
                            'is_stationary': adf_prices[1] < 0.05
                        },
                        'returns': {
                            'statistic': adf_returns[0],
                            'pvalue': adf_returns[1],
                            'critical_values': adf_returns[4],
                            'is_stationary': adf_returns[1] < 0.05
                        },
                        'log_prices': {
                            'statistic': adf_log_prices[0],
                            'pvalue': adf_log_prices[1],
                            'critical_values': adf_log_prices[4],
                            'is_stationary': adf_log_prices[1] < 0.05
                        }
                    },
                    'kpss_tests': {
                        'prices': {
                            'statistic': kpss_prices[0],
                            'pvalue': kpss_prices[1],
                            'critical_values': kpss_prices[3],
                            'is_stationary': kpss_prices[1] > 0.05
                        },
                        'returns': {
                            'statistic': kpss_returns[0],
                            'pvalue': kpss_returns[1],
                            'critical_values': kpss_returns[3],
                            'is_stationary': kpss_returns[1] > 0.05
                        },
                        'log_prices': {
                            'statistic': kpss_log_prices[0],
                            'pvalue': kpss_log_prices[1],
                            'critical_values': kpss_log_prices[3],
                            'is_stationary': kpss_log_prices[1] > 0.05
                        }
                    }
                }
                
                # Interpretación combinada ADF + KPSS
                prices_stationary = (adf_prices[1] < 0.05) and (kpss_prices[1] > 0.05)
                returns_stationary = (adf_returns[1] < 0.05) and (kpss_returns[1] > 0.05)
                
                stationarity_results[tf]['summary'] = {
                    'prices_stationary': prices_stationary,
                    'returns_stationary': returns_stationary,
                    'differencing_required': not prices_stationary
                }
                
                logger.info(f"{tf} - Prices stationary: {prices_stationary}")
                logger.info(f"{tf} - Returns stationary: {returns_stationary}")
                
            except Exception as e:
                logger.error(f"Stationarity tests failed for {tf}: {e}")
                stationarity_results[tf] = {'error': str(e)}
        
        self.results['stationarity'] = stationarity_results
        return stationarity_results
    
    def generate_comprehensive_report(self) -> str:
        """Generar reporte completo del análisis estadístico formal"""
        if not self.results:
            return "No results available. Run analyses first."
        
        report = []
        report.append("="*80)
        report.append("ANÁLISIS ESTADÍSTICO FORMAL - XRP/USDT")
        report.append("="*80)
        
        # ADVERTENCIA DE LEVERAGE
        report.append("\n⚠️  ADVERTENCIA CRÍTICA DE RIESGO ⚠️")
        report.append("-" * 40)
        report.append("Este análisis revela patrones de volatilidad en XRP futuros.")
        report.append("Su leverage actual (30x) es EXTREMADAMENTE PELIGROSO.")
        report.append("Considere reducir a 5x o menor para trading seguro.")
        
        # Resumen ejecutivo
        report.append("\nRESUMEN EJECUTIVO")
        report.append("-" * 40)
        
        for tf in self.data.keys():
            autocorr = self.results.get('autocorrelation', {}).get(tf, {})
            arch = self.results.get('arch_tests', {}).get(tf, {})
            stationarity = self.results.get('stationarity', {}).get(tf, {})
            
            if 'error' in autocorr or 'error' in arch or 'error' in stationarity:
                report.append(f"\n{tf.upper()} - ERROR EN ANÁLISIS")
                continue
            
            report.append(f"\n{tf.upper()} TIMEFRAME:")
            
            if autocorr and 'summary' in autocorr:
                has_structure = autocorr['summary']['has_autocorr_structure']
                has_arch = autocorr['summary']['has_arch_effects']
                report.append(f"  • Estructura temporal: {'SÍ' if has_structure else 'NO'}")
                report.append(f"  • Efectos ARCH detectados: {'SÍ' if has_arch else 'NO'}")
                
                if has_structure:
                    ar_order = autocorr['summary']['suggested_ar_order']
                    ma_order = autocorr['summary']['suggested_ma_order']
                    report.append(f"  • Modelo sugerido: ARIMA({ar_order}, 1, {ma_order})")
            
            if arch and 'summary' in arch:
                arch_strength = arch['summary']['arch_strength']
                optimal_lag = arch['summary']['optimal_arch_lag']
                clustering = arch['summary']['volatility_clustering_score']
                report.append(f"  • Fuerza ARCH: {arch_strength}")
                report.append(f"  • Lag óptimo ARCH: {optimal_lag}")
                report.append(f"  • Clustering volatilidad: {clustering:.3f}")
                
                # Advertencia específica para clustering alto
                if clustering > 0.3:
                    report.append(f"  ⚠️  ALTA CLUSTERING - LEVERAGE 30x MUY PELIGROSO")
            
            if stationarity and 'summary' in stationarity:
                prices_stat = stationarity['summary']['prices_stationary']
                returns_stat = stationarity['summary']['returns_stationary']
                report.append(f"  • Precios estacionarios: {'SÍ' if prices_stat else 'NO'}")
                report.append(f"  • Returns estacionarios: {'SÍ' if returns_stat else 'NO'}")
        
        # Recomendaciones específicas de risk management
        report.append("\nRECOMENDACIONES CRÍTICAS")
        report.append("-" * 40)
        
        for tf in self.data.keys():
            arch = self.results.get('arch_tests', {}).get(tf, {})
            if arch and 'summary' in arch:
                clustering = arch['summary']['volatility_clustering_score']
                has_strong_arch = arch['summary']['has_strong_arch_effects']
                
                report.append(f"\n{tf.upper()}:")
                if has_strong_arch:
                    report.append(f"  ⚠️  EFECTOS ARCH FUERTES DETECTADOS")
                    report.append(f"  • La volatilidad NO es constante")
                    report.append(f"  • Períodos de alta volatilidad se agrupan")
                    report.append(f"  • Su leverage 30x puede causar pérdidas del 100%")
                    report.append(f"  • RECOMENDACIÓN: Reducir leverage a 3-5x")
                
                if clustering > 0.5:
                    report.append(f"  🚨 CLUSTERING EXTREMO (score: {clustering:.3f})")
                    report.append(f"  • Alta probabilidad de volatilidad persistente")
                    report.append(f"  • PELIGRO CRÍTICO con leverage alto")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)
    
    # Helper methods (same as before but with error handling)
    def _find_significant_lags(self, correlations: np.ndarray, conf_intervals: np.ndarray) -> List[int]:
        """Encontrar lags con autocorrelación significativa"""
        significant_lags = []
        
        try:
            for i in range(1, len(correlations)):  # Skip lag 0
                lower_bound = conf_intervals[i, 0]
                upper_bound = conf_intervals[i, 1]
                
                if correlations[i] < lower_bound or correlations[i] > upper_bound:
                    significant_lags.append(i)
        except Exception as e:
            logger.warning(f"Error finding significant lags: {e}")
        
        return significant_lags
    
    def _suggest_ar_order(self, significant_pacf_lags: List[int]) -> int:
        """Sugerir orden AR basado en PACF"""
        if not significant_pacf_lags:
            return 0
        return min(3, len([lag for lag in significant_pacf_lags if lag <= 5]))
    
    def _suggest_ma_order(self, significant_acf_lags: List[int]) -> int:
        """Sugerir orden MA basado en ACF"""
        if not significant_acf_lags:
            return 0
        return min(3, len([lag for lag in significant_acf_lags if lag <= 5]))
    
    def _find_optimal_arch_lag(self, arch_tests: Dict) -> int:
        """Encontrar lag óptimo para modelo ARCH"""
        if not arch_tests:
            return 1
        
        best_lag = 1
        best_pvalue = 1.0
        
        for lag, test_result in arch_tests.items():
            pvalue = test_result.get('lm_pvalue', 1.0)
            if pvalue < best_pvalue and pvalue < 0.05:
                best_pvalue = pvalue
                best_lag = lag
        
        return best_lag
    
    def _assess_arch_strength(self, arch_tests: Dict) -> str:
        """Evaluar la fuerza de los efectos ARCH"""
        if not arch_tests:
            return "Desconocida"
        
        significant_tests = [test for test in arch_tests.values() 
                           if test.get('lm_pvalue', 1.0) < 0.05]
        
        if len(significant_tests) == 0:
            return "Débil"
        elif len(significant_tests) <= 2:
            return "Moderada"
        else:
            return "Fuerte"
    
    def _identify_volatility_periods(self, returns: pd.Series) -> Dict:
        """Identificar períodos de alta y baja volatilidad"""
        vol_window = min(20, len(returns) // 10)
        rolling_vol = returns.rolling(vol_window).std()
        
        vol_median = rolling_vol.median()
        vol_75 = rolling_vol.quantile(0.75)
        vol_25 = rolling_vol.quantile(0.25)
        
        high_vol_periods = (rolling_vol > vol_75).sum()
        low_vol_periods = (rolling_vol < vol_25).sum()
        
        # Clustering score
        high_vol_binary = (rolling_vol > vol_75).astype(int)
        clustering_score = high_vol_binary.autocorr(lag=1) if len(high_vol_binary) > 1 else 0
        clustering_score = max(0, clustering_score) if not pd.isna(clustering_score) else 0
        
        return {
            'high_vol_periods': high_vol_periods,
            'low_vol_periods': low_vol_periods,
            'total_periods': len(rolling_vol.dropna()),
            'clustering_score': clustering_score,
            'volatility_75th': vol_75,
            'volatility_median': vol_median,
            'volatility_25th': vol_25
        }
