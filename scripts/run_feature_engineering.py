# =============================================================================
# scripts/run_feature_engineering.py - VERSIÓN CON LEVERAGE DINÁMICO
"""
Script para ejecutar feature engineering completo con leverage dinámico
"""
import sys
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from data.storage.database_manager import db_manager
from analysis.features.technical_indicators import TechnicalIndicators
from analysis.features.pattern_detection import PatternDetection
from config.settings import settings  # AGREGADO para leverage dinámico
from utils.logger import setup_logger
import pandas as pd

def main():
    logger = setup_logger("feature_engineering")
    
    # AGREGADO: Mostrar configuración de leverage al inicio
    logger.info(f"Starting comprehensive feature engineering with {settings.LEVERAGE}x leverage...")
    
    # Verificar que TA-Lib esté instalado
    try:
        import talib
        logger.info("TA-Lib found, proceeding with feature engineering...")
    except ImportError:
        logger.error("TA-Lib not installed. Please install: pip install TA-Lib")
        print("\n" + "="*60)
        print("ERROR: TA-Lib not installed")
        print("="*60)
        print("Please install TA-Lib:")
        print("pip install TA-Lib")
        print("\nIf that fails on Mac/Linux:")
        print("brew install ta-lib")
        print("pip install TA-Lib")
        print("="*60)
        return
    
    # Cargar datos para múltiples timeframes
    timeframes = ['1m', '5m', '15m', '1h', '4h']
    
    for tf in timeframes:
        logger.info(f"Processing {tf} timeframe...")
        
        # Cargar datos
        df = db_manager.get_ohlcv_data('XRP/USDT', tf)
        
        if df.empty:
            logger.warning(f"No data found for {tf}")
            continue
        
        logger.info(f"Loaded {len(df):,} records for {tf}")
        
        try:
            # Inicializar calculadora de indicadores
            indicators = TechnicalIndicators(df)
            
            # Calcular todos los indicadores
            df_with_indicators = indicators.calculate_all_indicators()
            
            logger.info(f"Calculated {len(df_with_indicators.columns)} total features")
            
            # Detectar patrones
            pattern_detector = PatternDetection(df_with_indicators)
            
            # Soporte y resistencia
            support_resistance = pattern_detector.support_resistance_levels()
            logger.info(f"Support levels: {[f'${level:.4f}' for level in support_resistance['support_levels']]}")
            logger.info(f"Resistance levels: {[f'${level:.4f}' for level in support_resistance['resistance_levels']]}")
            logger.info(f"Current price: ${support_resistance['current_price']:.4f}")
            
            # Breakouts
            breakouts = pattern_detector.breakout_detection()
            breakout_count = len(breakouts[breakouts != 0])
            bullish_breakouts = len(breakouts[breakouts == 1])
            bearish_breakouts = len(breakouts[breakouts == -1])
            
            logger.info(f"Detected {breakout_count} total breakout signals")
            logger.info(f"  - Bullish breakouts: {bullish_breakouts}")
            logger.info(f"  - Bearish breakouts: {bearish_breakouts}")
            
            # Reversal signals
            reversals = pattern_detector.trend_reversal_signals()
            rsi_bull_signals = len(reversals[reversals['rsi_bull_divergence'] == True])
            rsi_bear_signals = len(reversals[reversals['rsi_bear_divergence'] == True])
            macd_bull_signals = len(reversals[reversals['macd_bull_cross'] == True])
            macd_bear_signals = len(reversals[reversals['macd_bear_cross'] == True])
            
            logger.info(f"Reversal signals detected:")
            logger.info(f"  - RSI Bull divergences: {rsi_bull_signals}")
            logger.info(f"  - RSI Bear divergences: {rsi_bear_signals}")
            logger.info(f"  - MACD Bull crosses: {macd_bull_signals}")
            logger.info(f"  - MACD Bear crosses: {macd_bear_signals}")
            
            # Guardar resultados
            output_file = Path("analysis_results") / f"indicators_{tf}_XRP_USDT.csv"
            output_file.parent.mkdir(exist_ok=True)
            
            # Añadir señales al DataFrame
            df_with_indicators['breakout_signal'] = breakouts
            df_with_indicators = df_with_indicators.join(reversals)
            
            # Guardar
            df_with_indicators.to_csv(output_file)
            logger.info(f"Saved {len(df_with_indicators.columns)} features to {output_file}")
            
            # CORREGIDO: Mostrar estadísticas de riesgo con leverage dinámico
            if 'leverage_risk' in df_with_indicators.columns:
                high_risk_periods = len(df_with_indicators[df_with_indicators['risk_warning'] == 1])
                total_periods = len(df_with_indicators)
                risk_percentage = (high_risk_periods / total_periods) * 100
                
                # CORREGIDO: Usar leverage dinámico en mensajes
                current_leverage = settings.LEVERAGE
                liquidation_threshold = (1 / current_leverage) * 100  # Porcentaje para liquidación
                
                if high_risk_periods > 0:
                    logger.warning(f"HIGH RISK PERIODS ({tf}): {high_risk_periods}/{total_periods} ({risk_percentage:.1f}%)")
                    logger.warning(f"With {current_leverage}x leverage, these periods have >50% loss potential")
                else:
                    logger.info(f"RISK ANALYSIS ({tf}): No high-risk periods detected with {current_leverage}x leverage")
                
                current_risk = df_with_indicators['leverage_risk'].iloc[-1]
                current_natr = df_with_indicators['natr_14'].iloc[-1]
                
                # CORREGIDO: Mensajes dinámicos basados en leverage configurado
                logger.info(f"CURRENT RISK LEVEL: {current_risk:.1f}% of account with {current_leverage}x leverage")
                logger.info(f"Current NATR: {current_natr:.4f} ({current_natr*100:.2f}%)")
                logger.info(f"Liquidation distance: {liquidation_threshold:.1f}% move against position")
                
                # Evaluar nivel de riesgo actual
                if current_risk > 10:
                    logger.warning(f"Current risk level is HIGH with {current_leverage}x leverage")
                elif current_risk > 5:
                    logger.warning(f"Current risk level is MODERATE with {current_leverage}x leverage")
                else:
                    logger.info(f"Current risk level is ACCEPTABLE with {current_leverage}x leverage")
                
                # Mostrar cuándo fue la última vez que el riesgo fue "seguro"
                safe_periods = df_with_indicators[df_with_indicators['risk_warning'] == 0]
                if not safe_periods.empty:
                    last_safe = safe_periods.index[-1]
                    logger.info(f"Last 'safe' period: {last_safe}")
                else:
                    logger.error(f"NO SAFE PERIODS FOUND in entire dataset with {current_leverage}x leverage!")
            
            # CORREGIDO: Análisis de volatilidad con leverage dinámico
            if 'hist_vol_20' in df_with_indicators.columns:
                avg_vol = df_with_indicators['hist_vol_20'].mean()
                max_vol = df_with_indicators['hist_vol_20'].max()
                current_leverage = settings.LEVERAGE
                
                logger.info(f"Volatility analysis ({tf}):")
                logger.info(f"  - Average 20-day vol: {avg_vol:.2f}")
                logger.info(f"  - Maximum 20-day vol: {max_vol:.2f}")
                
                # CORREGIDO: Cálculo dinámico de pérdida potencial
                max_loss_potential = max_vol * current_leverage
                if max_loss_potential > 50:
                    logger.warning(f"  - With {current_leverage}x leverage, max vol period = {max_loss_potential:.1f}% potential loss (HIGH RISK)")
                elif max_loss_potential > 20:
                    logger.warning(f"  - With {current_leverage}x leverage, max vol period = {max_loss_potential:.1f}% potential loss (MODERATE RISK)")
                else:
                    logger.info(f"  - With {current_leverage}x leverage, max vol period = {max_loss_potential:.1f}% potential loss (ACCEPTABLE)")
            
        except Exception as e:
            logger.error(f"Error processing {tf}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    logger.info("Feature engineering completed successfully!")
    
    # CORREGIDO: Resumen final con leverage dinámico
    print("\n" + "="*80)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*80)
    print("✅ Successfully created indicators:")
    print("   • Momentum: RSI, Stochastic, Williams %R, ROC, CCI")
    print("   • Trend: SMA/EMA, MACD, ADX, Parabolic SAR, Aroon")
    print("   • Volatility: Bollinger Bands, Keltner Channels, ATR")
    print("   • Volume: OBV, A/D Line, CMF, MFI")
    print("   • Price Action: Candlestick patterns, gaps, ranges")
    print("   • Statistical: Z-score, percentiles, regression stats")
    print("   • Risk Management: Dynamic stops, position sizing")
    print("   • Pattern Detection: Support/resistance, breakouts")
    print("\n📁 Files saved to: analysis_results/")
    print(f"\n⚙️  Configuration: {settings.LEVERAGE}x leverage from .env")
    
    # CORREGIDO: Recomendaciones dinámicas basadas en leverage
    if settings.LEVERAGE > 10:
        print("\n" + "⚠️ "*20)
        print("HIGH LEVERAGE WARNING")
        print("⚠️ "*20)
        print(f"Current leverage: {settings.LEVERAGE}x")
        print("Consider reducing leverage for safer trading")
        print("Recommended: 5x or lower for most strategies")
        print("Edit your .env file: LEVERAGE=5")
        print("⚠️ "*20)
    elif settings.LEVERAGE > 5:
        print(f"\n📊 Moderate leverage detected: {settings.LEVERAGE}x")
        print("Monitor risk warnings carefully")
    else:
        print(f"\n✅ Conservative leverage: {settings.LEVERAGE}x")
        print("Good risk management configuration")
    
    print("="*80)

if __name__ == "__main__":
    main()