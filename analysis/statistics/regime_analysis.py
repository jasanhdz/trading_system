# =============================================================================
# analysis/statistics/regime_analysis.py
# =============================================================================

"""
Sistema completo para análisis de regímenes de mercado incluyendo:
- Modelos GARCH para clustering de volatilidad
- Hidden Markov Models para identificar estados
- Correlaciones externas (BTC, índices, DXY)  
- Detección de anomalías y crisis
- LEVERAGE DINÁMICO desde configuración
"""
import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
from scipy.stats import jarque_bera
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# Importaciones condicionales para modelos avanzados
try:
    from arch import arch_model
    from hmmlearn.hmm import GaussianHMM
    ADVANCED_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Advanced models not available: {e}")
    print("Install with: pip install arch-py hmmlearn")
    ADVANCED_MODELS_AVAILABLE = False

from data.storage.database_manager import db_manager
from config.settings import settings  # IMPORTAR SETTINGS PARA LEVERAGE DINÁMICO
from utils.logger import setup_logger

logger = setup_logger("regime_analysis")

class AdvancedRegimeAnalysis:
    """Sistema avanzado para análisis de regímenes de mercado"""
    
    def __init__(self, symbol: str = "XRP/USDT"):
        self.symbol = symbol
        self.data = {}
        self.external_data = {}
        self.models = {}
        self.results = {}
        # LEVERAGE DINÁMICO desde configuración
        self.current_leverage = getattr(settings, 'LEVERAGE', 10)  # Default 10x si no está definido
        logger.info(f"Using leverage: {self.current_leverage}x from configuration")
        
    def load_data(self, timeframes: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Cargar datos de XRP para análisis"""
        if timeframes is None:
            timeframes = ['1m', '5m', '15m', '1h', '4h']
            
        logger.info("Loading XRP data for regime analysis...")
        
        for tf in timeframes:
            df = db_manager.get_ohlcv_data(self.symbol, tf)
            if not df.empty:
                # Preparar datos para análisis de regímenes
                df['returns'] = df['close'].pct_change()
                df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
                df['abs_returns'] = np.abs(df['returns'])
                df['squared_returns'] = df['returns'] ** 2
                df['realized_vol'] = df['returns'].rolling(20).std()
                
                # Limpiar datos
                df = df.dropna()
                
                self.data[tf] = df
                logger.info(f"Loaded {len(df):,} records for {tf}")
            else:
                logger.warning(f"No data found for {tf}")
                
        return self.data
    
    def collect_external_data(self) -> Dict[str, pd.DataFrame]:
        """Recolectar datos externos para análisis de correlación"""
        logger.info("Collecting external market data...")
        
        # Definir símbolos externos con configuración específica
        external_symbols = {
            'BTC': {'symbol': 'BTC-USD', 'interval': '1h'},
            'ETH': {'symbol': 'ETH-USD', 'interval': '1h'}, 
            'SPY': {'symbol': 'SPY', 'interval': '1h'},  # S&P 500
            'QQQ': {'symbol': 'QQQ', 'interval': '1h'},  # NASDAQ
            'DXY': {'symbol': 'DX-Y.NYB', 'interval': '1h'},  # Dollar Index
            'VIX': {'symbol': '^VIX', 'interval': '1h'},  # Volatility Index
        }
        
        # Calcular rango de fechas basado en datos XRP
        if self.data:
            start_date = min(df.index.min() for df in self.data.values())
            end_date = max(df.index.max() for df in self.data.values())
            
            # Convertir a timezone naive para yfinance
            if hasattr(start_date, 'tz_localize'):
                start_date = start_date.tz_localize(None) if start_date.tz is None else start_date.tz_convert(None)
            if hasattr(end_date, 'tz_localize'):
                end_date = end_date.tz_localize(None) if end_date.tz is None else end_date.tz_convert(None)
        else:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=100)
        
        logger.info(f"Collecting external data from {start_date} to {end_date}")
        
        for name, config in external_symbols.items():
            try:
                symbol = config['symbol']
                interval = config.get('interval', '1h')
                
                logger.info(f"Downloading {name} ({symbol}) with {interval} interval...")
                
                # Usar yfinance con configuración específica
                ticker = yf.Ticker(symbol)
                hist = ticker.history(
                    start=start_date - timedelta(days=2), 
                    end=end_date + timedelta(days=1),
                    interval=interval,
                    auto_adjust=True,
                    prepost=False
                )
                
                if not hist.empty:
                    # Limpiar y preparar datos
                    hist = hist.dropna()
                    
                    # Calcular returns y volatilidad
                    hist['returns'] = hist['Close'].pct_change()
                    hist['log_returns'] = np.log(hist['Close'] / hist['Close'].shift(1))
                    hist['volatility'] = hist['returns'].rolling(24).std()
                    
                    # Limpiar outliers extremos
                    returns_clean = hist['returns']
                    q99 = returns_clean.quantile(0.99)
                    q01 = returns_clean.quantile(0.01)
                    hist['returns'] = hist['returns'].clip(q01, q99)
                    
                    self.external_data[name] = hist
                    logger.info(f"✅ Collected {len(hist):,} records for {name}")
                    logger.info(f"   Date range: {hist.index.min()} to {hist.index.max()}")
                else:
                    logger.warning(f"❌ No data collected for {name}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to collect {name}: {e}")
                continue
        
        return self.external_data
    
    def fit_garch_models(self) -> Dict[str, Any]:
        """Ajustar modelos GARCH para cada timeframe"""
        if not ADVANCED_MODELS_AVAILABLE:
            logger.error("GARCH models require 'arch' package. Install with: pip install arch-py")
            return {}
        
        logger.info("Fitting GARCH models...")
        
        garch_results = {}
        
        for tf, df in self.data.items():
            logger.info(f"Fitting GARCH model for {tf}...")
            
            try:
                returns = df['returns'].dropna() * 100  # Convert to percentage for numerical stability
                
                if len(returns) < 100:
                    logger.warning(f"Insufficient data for GARCH model in {tf}")
                    continue
                
                # Determinar orden GARCH basado en análisis ARCH previo
                garch_orders = {'15m': (1, 1), '1h': (1, 1), '4h': (1, 1)}
                p, q = garch_orders.get(tf, (1, 1))
                
                # Ajustar modelo GARCH
                model = arch_model(returns, vol='Garch', p=p, q=q, dist='normal')
                fitted_model = model.fit(disp='off', show_warning=False)
                
                # Extraer resultados
                conditional_volatility = fitted_model.conditional_volatility
                standardized_residuals = fitted_model.std_resid
                
                # Calcular métricas del modelo
                aic = fitted_model.aic
                bic = fitted_model.bic
                log_likelihood = fitted_model.loglikelihood
                
                # Identificar regímenes de volatilidad
                vol_regimes = self._identify_volatility_regimes(conditional_volatility)
                
                garch_results[tf] = {
                    'model': fitted_model,
                    'conditional_volatility': conditional_volatility,
                    'standardized_residuals': standardized_residuals,
                    'volatility_regimes': vol_regimes,
                    'model_stats': {
                        'aic': aic,
                        'bic': bic,
                        'log_likelihood': log_likelihood,
                        'p': p,
                        'q': q
                    },
                    'forecast': self._garch_forecast(fitted_model, steps=10)
                }
                
                logger.info(f"{tf} GARCH({p},{q}): AIC={aic:.2f}, BIC={bic:.2f}")
                logger.info(f"{tf} Volatility regimes: {vol_regimes['summary']}")
                
            except Exception as e:
                logger.error(f"GARCH model failed for {tf}: {e}")
                continue
        
        self.models['garch'] = garch_results
        return garch_results
    
    def fit_hmm_models(self, n_components: int = 3) -> Dict[str, Any]:
        """Ajustar Hidden Markov Models para identificar regímenes"""
        if not ADVANCED_MODELS_AVAILABLE:
            logger.error("HMM models require 'hmmlearn' package. Install with: pip install hmmlearn")
            return {}
        
        logger.info(f"Fitting Hidden Markov Models with {n_components} states...")
        
        hmm_results = {}
        
        for tf, df in self.data.items():
            logger.info(f"Fitting HMM for {tf}...")
            
            try:
                # Preparar features para HMM
                features = self._prepare_hmm_features(df)
                
                if features is None or len(features) < 50:
                    logger.warning(f"Insufficient data for HMM in {tf}")
                    continue
                
                # Normalizar features
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features)
                
                # Ajustar HMM
                model = GaussianHMM(n_components=n_components, covariance_type="full", random_state=42)
                model.fit(features_scaled)
                
                # Predecir estados
                hidden_states = model.predict(features_scaled)
                state_probabilities = model.predict_proba(features_scaled)
                
                # Analizar regímenes identificados
                regime_analysis = self._analyze_hmm_regimes(df, hidden_states, features)
                
                # Score del modelo
                score = model.score(features_scaled)
                
                hmm_results[tf] = {
                    'model': model,
                    'scaler': scaler,
                    'hidden_states': hidden_states,
                    'state_probabilities': state_probabilities,
                    'features': features,
                    'regime_analysis': regime_analysis,
                    'model_score': score,
                    'n_components': n_components
                }
                
                logger.info(f"{tf} HMM: Score={score:.2f}, States identified")
                logger.info(f"{tf} Regime summary: {regime_analysis['summary']}")
                
            except Exception as e:
                logger.error(f"HMM model failed for {tf}: {e}")
                continue
        
        self.models['hmm'] = hmm_results
        return hmm_results
    
    def analyze_external_correlations(self) -> Dict[str, Any]:
        """Analizar correlaciones con mercados externos"""
        logger.info("Analyzing correlations with external markets...")
        
        if not self.external_data:
            logger.warning("No external data available. Collecting now...")
            self.collect_external_data()
        
        correlation_results = {}
        
        # Usar datos de 1h como referencia principal
        if '1h' not in self.data:
            logger.error("No 1h XRP data available for correlation analysis")
            return {}
        
        xrp_data = self.data['1h'].copy()
        
        for external_name, external_df in self.external_data.items():
            logger.info(f"Analyzing correlation with {external_name}...")
            
            try:
                # Alinear timestamps
                aligned_data = self._align_timestamps(xrp_data, external_df, external_name)
                
                if aligned_data is None or len(aligned_data) < 24:  # Al menos 24 horas de datos
                    logger.warning(f"Insufficient aligned data for {external_name}")
                    continue
                
                # Calcular correlaciones
                correlations = self._calculate_correlations(aligned_data, external_name)
                
                # Analizar correlación rolling
                rolling_corr = self._rolling_correlation_analysis(aligned_data, external_name)
                
                # Detectar cambios de régimen en correlación
                corr_regimes = self._detect_correlation_regimes(rolling_corr)
                
                correlation_results[external_name] = {
                    'static_correlations': correlations,
                    'rolling_correlations': rolling_corr,
                    'correlation_regimes': corr_regimes,
                    'data_points': len(aligned_data),
                    'summary': self._summarize_correlation(correlations, rolling_corr)
                }
                
                logger.info(f"{external_name} - Static correlation: {correlations['returns']:.3f}")
                logger.info(f"{external_name} - Correlation stability: {corr_regimes['stability_score']:.3f}")
                
            except Exception as e:
                logger.error(f"Correlation analysis failed for {external_name}: {e}")
                continue
        
        self.results['correlations'] = correlation_results
        return correlation_results
    
    def detect_anomalies_and_crises(self) -> Dict[str, Any]:
        """Detectar anomalías y períodos de crisis"""
        logger.info("Detecting market anomalies and crisis periods...")
        
        anomaly_results = {}
        
        for tf, df in self.data.items():
            logger.info(f"Analyzing anomalies in {tf}...")
            
            try:
                # Múltiples métodos de detección de anomalías
                anomalies = {}
                
                # 1. Statistical outliers (Z-score)
                anomalies['statistical'] = self._detect_statistical_outliers(df)
                
                # 2. Volatility anomalies
                anomalies['volatility'] = self._detect_volatility_anomalies(df)
                
                # 3. Volume anomalies
                anomalies['volume'] = self._detect_volume_anomalies(df)
                
                # 4. Price movement anomalies
                anomalies['price_movement'] = self._detect_price_movement_anomalies(df)
                
                # 5. Combined anomaly score
                combined_score = self._combine_anomaly_scores(anomalies, df)
                
                # Identificar períodos de crisis
                crisis_periods = self._identify_crisis_periods(df, combined_score)
                
                # Análisis de contagio (si hay datos externos)
                contagion_analysis = None
                if self.external_data:
                    contagion_analysis = self._analyze_contagion_effects(df, tf)
                
                anomaly_results[tf] = {
                    'individual_anomalies': anomalies,
                    'combined_anomaly_score': combined_score,
                    'crisis_periods': crisis_periods,
                    'contagion_analysis': contagion_analysis,
                    'summary': {
                        'total_anomalies': len(combined_score[combined_score > 2]),
                        'crisis_periods_count': len(crisis_periods),
                        'most_severe_date': combined_score.idxmax(),
                        'max_anomaly_score': combined_score.max()
                    }
                }
                
                logger.info(f"{tf} - Total anomalies detected: {anomaly_results[tf]['summary']['total_anomalies']}")
                logger.info(f"{tf} - Crisis periods: {len(crisis_periods)}")
                
            except Exception as e:
                logger.error(f"Anomaly detection failed for {tf}: {e}")
                continue
        
        self.results['anomalies'] = anomaly_results
        return anomaly_results
    
    def enhanced_garch_analysis(self) -> Dict[str, Any]:
        """Análisis GARCH mejorado con interpretación de regímenes"""
        if 'garch' not in self.models:
            logger.error("GARCH models not fitted yet")
            return {}
        
        enhanced_results = {}
        
        for tf, garch_result in self.models['garch'].items():
            logger.info(f"Enhanced GARCH analysis for {tf}...")
            
            # Extraer datos del modelo
            model = garch_result['model']
            conditional_vol = garch_result['conditional_volatility']
            regimes = garch_result['volatility_regimes']
            
            # Análisis de persistencia de volatilidad
            vol_persistence = conditional_vol.autocorr(lag=1)
            
            # Análisis de clustering
            vol_changes = conditional_vol.diff().abs()
            clustering_intensity = vol_changes.rolling(20).mean().std()
            
            # Identification de crisis periods basado en GARCH
            vol_threshold = conditional_vol.quantile(0.95)
            crisis_periods_garch = conditional_vol[conditional_vol > vol_threshold]
            
            # Forecasting próximos períodos
            forecast = garch_result.get('forecast')
            
            enhanced_results[tf] = {
                'volatility_persistence': vol_persistence,
                'clustering_intensity': clustering_intensity,
                'crisis_threshold': vol_threshold,
                'crisis_periods_count': len(crisis_periods_garch),
                'current_regime': regimes['regimes'].iloc[-1] if len(regimes['regimes']) > 0 else 'Unknown',
                'forecast_volatility': forecast['volatility_forecast'] if forecast else None,
                'regime_interpretation': self._interpret_volatility_regime(regimes, conditional_vol)
            }
            
            logger.info(f"{tf} - Volatility persistence: {vol_persistence:.3f}")
            logger.info(f"{tf} - Current regime: {enhanced_results[tf]['current_regime']}")
        
        return enhanced_results
    
    def generate_leverage_risk_assessment(self) -> str:
        """Generar evaluación específica de riesgo basado en leverage actual"""
        risk_assessment = []
        
        # Liquidation distance con leverage actual
        liquidation_distance = (1 / self.current_leverage) * 100
        
        risk_assessment.append(f"\n🎯 ANÁLISIS DE RIESGO CON {self.current_leverage}X LEVERAGE")
        risk_assessment.append("-" * 60)
        risk_assessment.append(f"• Distancia a liquidación: {liquidation_distance:.1f}% movimiento adverso")
        
        if 'anomalies' in self.results:
            total_crisis_4h = self.results['anomalies'].get('4h', {}).get('summary', {}).get('crisis_periods_count', 0)
            
            if total_crisis_4h > 0:
                risk_assessment.append(f"• {total_crisis_4h} períodos de crisis detectados en 4h timeframe")
                
                if self.current_leverage >= 20:
                    risk_assessment.append("• ⚠️ ALTO RIESGO: Leverage >20x con crisis frecuentes")
                elif self.current_leverage >= 10:
                    risk_assessment.append("• ⚠️ RIESGO MODERADO: Monitorear volatilidad de cerca")
                elif self.current_leverage <= 5:
                    risk_assessment.append("• ✅ RIESGO CONSERVADOR: Leverage apropiado para XRP")
        
        # Evaluación basada en correlaciones
        if 'correlations' in self.results:
            high_corr_assets = []
            for asset, data in self.results['correlations'].items():
                corr = data['static_correlations']['returns']
                if abs(corr) > 0.6:
                    high_corr_assets.append(f"{asset} ({corr:.2f})")
            
            if high_corr_assets:
                risk_assessment.append("• Alta correlación con: " + ", ".join(high_corr_assets))
                risk_assessment.append("• IMPLICACIÓN: Sin diversificación - si crypto crash, todo crash junto")
        
        # Recomendaciones específicas
        risk_assessment.append("\n📋 RECOMENDACIONES ESPECÍFICAS:")
        
        if self.current_leverage > 10:
            risk_assessment.append("• CRÍTICO: Reducir leverage a 5x máximo para XRP futuros")
        elif self.current_leverage <= 5:
            risk_assessment.append("• BUENA PRÁCTICA: Leverage conservador apropiado")
        
        if 'garch' in self.models:
            high_vol_periods = 0
            for tf, garch in self.models['garch'].items():
                regimes = garch['volatility_regimes']['counts']
                high_vol_periods += regimes.get('High', 0)
            
            if high_vol_periods > 0:
                max_loss = (high_vol_periods / sum(len(df) for df in self.data.values())) * 100
                risk_assessment.append(f"• {high_vol_periods} períodos de alta volatilidad ({max_loss:.1f}% del tiempo)")
                
                leverage_impact = self.current_leverage * 0.05  # Asumiendo 5% volatilidad alta
                risk_assessment.append(f"• Con {self.current_leverage}x leverage, riesgo máximo: {leverage_impact:.1f}% por período")
        
        return "\n".join(risk_assessment)
    
    def generate_comprehensive_report(self) -> str:
        """Generar reporte completo del análisis de regímenes con leverage dinámico"""
        report = []
        report.append("="*80)
        report.append("ANÁLISIS AVANZADO DE REGÍMENES DE MERCADO - XRP/USDT")
        report.append("="*80)
        
        # Resumen ejecutivo
        report.append("\nRESUMEN EJECUTIVO")
        report.append("-" * 40)
        
        total_regimes = 0
        total_anomalies = 0
        
        for tf in self.data.keys():
            report.append(f"\n{tf.upper()} TIMEFRAME:")
            
            # GARCH results
            if 'garch' in self.models and tf in self.models['garch']:
                garch = self.models['garch'][tf]
                vol_regimes = garch['volatility_regimes']['summary']
                report.append(f"  GARCH Model: AIC={garch['model_stats']['aic']:.2f}")
                report.append(f"  Volatility Regimes: {vol_regimes}")
            
            # HMM results
            if 'hmm' in self.models and tf in self.models['hmm']:
                hmm = self.models['hmm'][tf]
                regime_summary = hmm['regime_analysis']['summary']
                report.append(f"  HMM States: {hmm['n_components']} regimes identified")
                report.append(f"  Regime Distribution: {regime_summary}")
                total_regimes += hmm['n_components']
            
            # Anomaly results
            if 'anomalies' in self.results and tf in self.results['anomalies']:
                anomaly = self.results['anomalies'][tf]
                anomaly_count = anomaly['summary']['total_anomalies']
                crisis_count = anomaly['summary']['crisis_periods_count']
                report.append(f"  Anomalies Detected: {anomaly_count}")
                report.append(f"  Crisis Periods: {crisis_count}")
                total_anomalies += anomaly_count
        
        # Correlaciones externas
        if 'correlations' in self.results:
            report.append("\nCORRELACIONES EXTERNAS")
            report.append("-" * 40)
            
            for asset, corr_data in self.results['correlations'].items():
                static_corr = corr_data['static_correlations']['returns']
                stability = corr_data['correlation_regimes']['stability_score']
                report.append(f"{asset}: {static_corr:.3f} (estabilidad: {stability:.3f})")
        
        # LEVERAGE RISK ASSESSMENT DINÁMICO
        leverage_risk = self.generate_leverage_risk_assessment()
        report.append(leverage_risk)
        
        # Recomendaciones estratégicas
        report.append("\nRECOMENDACIONES ESTRATÉGICAS")
        report.append("-" * 40)
        
        if total_regimes > 0:
            report.append("• Implementar estrategias adaptativas basadas en regímenes detectados")
            report.append("• Usar modelos GARCH para predicción de volatilidad")
            report.append("• Ajustar tamaño de posición según régimen actual")
        
        if 'correlations' in self.results:
            high_corr_assets = [asset for asset, data in self.results['correlations'].items() 
                              if abs(data['static_correlations']['returns']) > 0.5]
            if high_corr_assets:
                report.append(f"• Monitorear correlaciones altas con: {', '.join(high_corr_assets)}")
        
        report.append(f"\n⚙️ Configuración actual: {self.current_leverage}x leverage")
        report.append("\n" + "="*80)
        return "\n".join(report)
    
    # Helper methods - TODOS CORREGIDOS
    def _prepare_hmm_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Preparar features para HMM"""
        try:
            # Crear DataFrame temporal para evitar problemas de indexing
            temp_df = df.copy()
            
            # Calcular features individuales
            returns = temp_df['returns'].fillna(0)
            realized_vol = temp_df['realized_vol'].fillna(temp_df['realized_vol'].mean())
            
            # Volume normalizado
            if 'volume' in temp_df.columns:
                volume_norm = (temp_df['volume'] - temp_df['volume'].mean()) / temp_df['volume'].std()
                volume_norm = volume_norm.fillna(0)
            else:
                volume_norm = pd.Series(0, index=temp_df.index)
            
            # Price momentum
            price_momentum = temp_df['close'].pct_change(5).fillna(0)
            
            # Crear matriz de features
            features_df = pd.DataFrame({
                'returns': returns,
                'volatility': realized_vol,
                'volume': volume_norm,
                'momentum': price_momentum
            })
            
            # Convertir a numpy array
            features = features_df.values
            
            # Verificar que no hay NaN
            if np.isnan(features).any():
                logger.warning("NaN values found in features, filling with zeros")
                features = np.nan_to_num(features)
            
            # Verificar dimensiones
            if features.shape[0] < 50:
                logger.warning(f"Only {features.shape[0]} samples for HMM")
                return None
                
            logger.info(f"HMM features prepared: {features.shape}")
            return features
            
        except Exception as e:
            logger.error(f"Error preparing HMM features: {e}")
            return None
    
    def _align_timestamps(self, xrp_data: pd.DataFrame, external_data: pd.DataFrame, name: str) -> Optional[pd.DataFrame]:
        """Alinear timestamps entre XRP y datos externos"""
        try:
            logger.info(f"Aligning timestamps for {name}...")
            
            # Convertir ambos índices a UTC
            xrp_utc = xrp_data.copy()
            if xrp_utc.index.tz is None:
                xrp_utc.index = xrp_utc.index.tz_localize('UTC')
            else:
                xrp_utc.index = xrp_utc.index.tz_convert('UTC')
            
            external_utc = external_data.copy()
            if external_utc.index.tz is None:
                external_utc.index = external_utc.index.tz_localize('UTC')
            else:
                external_utc.index = external_utc.index.tz_convert('UTC')
            
            # Encontrar período común
            common_start = max(xrp_utc.index.min(), external_utc.index.min())
            common_end = min(xrp_utc.index.max(), external_utc.index.max())
            
            logger.info(f"Common period: {common_start} to {common_end}")
            
            # Filtrar a período común
            xrp_common = xrp_utc[common_start:common_end]
            external_common = external_utc[common_start:common_end]
            
            # Resamplear XRP a frecuencia horaria si es necesario
            if len(xrp_common) > len(external_common) * 2:  # XRP tiene más frecuencia
                xrp_resampled = xrp_common.resample('1H').agg({
                    'returns': 'sum',  # Sumar returns para agregación
                    'realized_vol': 'last'
                }).dropna()
            else:
                xrp_resampled = xrp_common
            
            # Resamplear external data a 1H
            external_resampled = external_common.resample('1H').agg({
                'returns': 'sum',
                'volatility': 'last'
            }).dropna()
            
            # Crear DataFrame combinado con inner join
            combined = pd.DataFrame()
            combined['xrp_returns'] = xrp_resampled['returns']
            combined['xrp_volatility'] = xrp_resampled['realized_vol']
            combined[f'{name}_returns'] = external_resampled['returns']
            combined[f'{name}_volatility'] = external_resampled['volatility']
            
            # Eliminar NaN
            combined = combined.dropna()
            
            logger.info(f"Final aligned data: {len(combined)} records")
            
            return combined if len(combined) >= 24 else None  # Al menos 24 horas de datos
            
        except Exception as e:
            logger.error(f"Error aligning timestamps for {name}: {e}")
            return None
    
    def _identify_volatility_regimes(self, conditional_volatility: pd.Series) -> Dict:
        """Identificar regímenes de volatilidad basados en GARCH"""
        vol_25 = conditional_volatility.quantile(0.25)
        vol_75 = conditional_volatility.quantile(0.75)
        
        regimes = pd.cut(conditional_volatility, 
                        bins=[-np.inf, vol_25, vol_75, np.inf], 
                        labels=['Low', 'Medium', 'High'])
        
        regime_counts = regimes.value_counts()
        
        return {
            'regimes': regimes,
            'thresholds': {'low': vol_25, 'high': vol_75},
            'counts': regime_counts.to_dict(),
            'summary': f"Low: {regime_counts.get('Low', 0)}, Med: {regime_counts.get('Medium', 0)}, High: {regime_counts.get('High', 0)}"
        }
    
    def _garch_forecast(self, fitted_model, steps: int = 10):
        """Generar pronóstico GARCH"""
        try:
            forecast = fitted_model.forecast(horizon=steps)
            return {
                'variance_forecast': forecast.variance.iloc[-1].values,
                'volatility_forecast': np.sqrt(forecast.variance.iloc[-1].values)
            }
        except:
            return None
    
    def _analyze_hmm_regimes(self, df: pd.DataFrame, states: np.ndarray, features: np.ndarray) -> Dict:
        """Analizar regímenes identificados por HMM"""
        df_analysis = df.iloc[:len(states)].copy()
        df_analysis['regime'] = states
        
        regime_stats = {}
        for regime in np.unique(states):
            regime_data = df_analysis[df_analysis['regime'] == regime]
            regime_stats[f'regime_{regime}'] = {
                'count': len(regime_data),
                'avg_return': regime_data['returns'].mean(),
                'avg_volatility': regime_data['realized_vol'].mean(),
                'avg_volume': regime_data['volume'].mean() if 'volume' in regime_data else np.nan
            }
        
        # Análisis de transiciones
        transitions = self._analyze_regime_transitions(states)
        
        return {
            'regime_stats': regime_stats,
            'transitions': transitions,
            'summary': f"{len(np.unique(states))} regimes, most common: {np.bincount(states).argmax()}"
        }
    
    def _analyze_regime_transitions(self, states: np.ndarray) -> Dict:
        """Analizar probabilidades de transición entre regímenes"""
        n_states = len(np.unique(states))
        transition_matrix = np.zeros((n_states, n_states))
        
        for i in range(len(states) - 1):
            current_state = states[i]
            next_state = states[i + 1]
            transition_matrix[current_state, next_state] += 1
        
        # Normalizar por filas
        transition_probs = transition_matrix / (transition_matrix.sum(axis=1, keepdims=True) + 1e-8)
        
        return {
            'transition_matrix': transition_matrix,
            'transition_probabilities': transition_probs,
            'persistence': np.diag(transition_probs)
        }
    
    def _calculate_correlations(self, aligned_data: pd.DataFrame, name: str) -> Dict:
        """Calcular correlaciones estáticas"""
        correlations = {
            'returns': aligned_data['xrp_returns'].corr(aligned_data[f'{name}_returns']),
            'volatility': aligned_data['xrp_volatility'].corr(aligned_data[f'{name}_volatility']),
        }
        return correlations
    
    def _rolling_correlation_analysis(self, aligned_data: pd.DataFrame, name: str, window: int = 24) -> pd.Series:
        """Analizar correlación rolling"""
        rolling_corr = aligned_data['xrp_returns'].rolling(window).corr(aligned_data[f'{name}_returns'])
        return rolling_corr.dropna()
    
    def _detect_correlation_regimes(self, rolling_corr: pd.Series) -> Dict:
        """Detectar cambios en regímenes de correlación"""
        if len(rolling_corr) < 10:
            return {'stability_score': 0.0}
        
        # Stability score based on standard deviation
        stability_score = 1 / (1 + rolling_corr.std())
        
        # Regime changes (simplified)
        regime_changes = np.abs(rolling_corr.diff()) > 0.2
        change_points = rolling_corr.index[regime_changes].tolist()
        
        return {
            'stability_score': stability_score,
            'regime_changes': len(change_points),
            'change_points': change_points[:5]  # Top 5
        }
    
    def _summarize_correlation(self, static_corr: Dict, rolling_corr: pd.Series) -> Dict:
        """Resumir análisis de correlación"""
        return {
            'static_returns_corr': static_corr['returns'],
            'correlation_range': (rolling_corr.min(), rolling_corr.max()),
            'correlation_std': rolling_corr.std(),
            'current_correlation': rolling_corr.iloc[-1] if len(rolling_corr) > 0 else np.nan
        }
    
    def _detect_statistical_outliers(self, df: pd.DataFrame, threshold: float = 3.0) -> pd.Series:
        """Detectar outliers estadísticos usando Z-score"""
        z_scores = np.abs(stats.zscore(df['returns'].dropna()))
        return pd.Series(z_scores > threshold, index=df.index[:len(z_scores)])
    
    def _detect_volatility_anomalies(self, df: pd.DataFrame) -> pd.Series:
        """Detectar anomalías de volatilidad"""
        vol_rolling = df['returns'].rolling(20).std()
        vol_threshold = vol_rolling.quantile(0.95)
        return vol_rolling > vol_threshold
    
    def _detect_volume_anomalies(self, df: pd.DataFrame) -> pd.Series:
        """Detectar anomalías de volumen"""
        if 'volume' not in df.columns:
            return pd.Series(False, index=df.index)
        
        vol_rolling = df['volume'].rolling(20).mean()
        vol_threshold = vol_rolling.quantile(0.95)
        return df['volume'] > vol_threshold
    
    def _detect_price_movement_anomalies(self, df: pd.DataFrame) -> pd.Series:
        """Detectar anomalías de movimiento de precio"""
        price_changes = np.abs(df['close'].pct_change())
        threshold = price_changes.quantile(0.99)
        return price_changes > threshold
    
    def _combine_anomaly_scores(self, anomalies: Dict, df: pd.DataFrame) -> pd.Series:
        """Combinar múltiples scores de anomalía"""
        combined_score = pd.Series(0.0, index=df.index)
        
        for anomaly_type, anomaly_series in anomalies.items():
            if len(anomaly_series) > 0:
                # Align indices
                aligned = anomaly_series.reindex(combined_score.index, fill_value=False)
                combined_score += aligned.astype(float)
        
        return combined_score
    
    def _identify_crisis_periods(self, df: pd.DataFrame, anomaly_score: pd.Series, threshold: float = 2.0) -> List[Tuple]:
        """Identificar períodos de crisis basados en score de anomalía"""
        crisis_mask = anomaly_score >= threshold
        
        # Encontrar períodos consecutivos
        crisis_periods = []
        start = None
        
        for date, is_crisis in crisis_mask.items():
            if is_crisis and start is None:
                start = date
            elif not is_crisis and start is not None:
                crisis_periods.append((start, date))
                start = None
        
        # Handle case where crisis continues to the end
        if start is not None:
            crisis_periods.append((start, crisis_mask.index[-1]))
        
        return crisis_periods
    
    def _analyze_contagion_effects(self, df: pd.DataFrame, timeframe: str) -> Optional[Dict]:
        """Analizar efectos de contagio con mercados externos"""
        if not self.external_data or 'correlations' not in self.results:
            return None
        
        contagion_analysis = {}
        
        # Analizar si las correlaciones aumentan durante crisis
        if timeframe in self.results.get('anomalies', {}):
            crisis_periods = self.results['anomalies'][timeframe]['crisis_periods']
            
            for asset, corr_data in self.results['correlations'].items():
                if len(crisis_periods) > 0:
                    # Simplified contagion analysis
                    normal_corr = corr_data['static_correlations']['returns']
                    contagion_analysis[asset] = {
                        'normal_correlation': normal_corr,
                        'crisis_periods_analyzed': len(crisis_periods)
                    }
        
        return contagion_analysis
    
    def _interpret_volatility_regime(self, regimes: Dict, conditional_vol: pd.Series) -> Dict:
        """Interpretar regímenes de volatilidad para trading"""
        current_vol = conditional_vol.iloc[-1]
        thresholds = regimes['thresholds']
        
        if current_vol <= thresholds['low']:
            regime = 'Low'
            trading_advice = f'Range-bound strategies, tight stops, leverage up to {min(self.current_leverage, 10)}x safe'
        elif current_vol >= thresholds['high']:
            regime = 'High'
            trading_advice = f'Trend-following, wider stops, reduce position size, max leverage {max(1, self.current_leverage // 2)}x'
        else:
            regime = 'Medium'
            trading_advice = f'Balanced approach, current {self.current_leverage}x leverage acceptable'
        
        return {
            'current_regime': regime,
            'current_volatility': current_vol,
            'trading_advice': trading_advice,
            'regime_stability': conditional_vol.rolling(20).std().iloc[-1]
        }
