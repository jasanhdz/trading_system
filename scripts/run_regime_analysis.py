# =============================================================================
# scripts/run_regime_analysis.py
# =============================================================================

#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

def main():
    try:
        from analysis.statistics.regime_analysis import AdvancedRegimeAnalysis
        from utils.logger import setup_logger
        
        logger = setup_logger("regime_analysis_runner")
        logger.info("Starting advanced regime analysis...")
        
        # Inicializar análisis
        regime_analyzer = AdvancedRegimeAnalysis("XRP/USDT")
        
        # Cargar datos XRP
        timeframes = ['1m', '5m', '15m', '1h', '4h']
        data = regime_analyzer.load_data(timeframes)
        
        if not data:
            logger.error("No XRP data loaded. Please collect data first.")
            return
        
        # Recolectar datos externos
        logger.info("Collecting external market data...")
        regime_analyzer.collect_external_data()
        
        # Ajustar modelos GARCH
        logger.info("Fitting GARCH models...")
        try:
            garch_results = regime_analyzer.fit_garch_models()
            if garch_results:
                logger.info(f"GARCH models fitted for {len(garch_results)} timeframes")
        except Exception as e:
            logger.error(f"GARCH modeling failed: {e}")
        
        # Ajustar modelos HMM
        logger.info("Fitting Hidden Markov Models...")
        try:
            hmm_results = regime_analyzer.fit_hmm_models(n_components=3)
            if hmm_results:
                logger.info(f"HMM models fitted for {len(hmm_results)} timeframes")
        except Exception as e:
            logger.error(f"HMM modeling failed: {e}")
        
        # Analizar correlaciones externas
        logger.info("Analyzing external correlations...")
        corr_results = regime_analyzer.analyze_external_correlations()
        if corr_results:
            logger.info(f"Correlations analyzed with {len(corr_results)} external assets")
        
        # Detectar anomalías
        logger.info("Detecting anomalies and crisis periods...")
        anomaly_results = regime_analyzer.detect_anomalies_and_crises()
        if anomaly_results:
            total_anomalies = sum(result['summary']['total_anomalies'] 
                                for result in anomaly_results.values())
            logger.info(f"Total anomalies detected: {total_anomalies}")
        
        # Generar reporte
        logger.info("Generating comprehensive regime analysis report...")
        report = regime_analyzer.generate_comprehensive_report()
        
        # Guardar reporte
        report_file = Path("analysis_reports") / "regime_analysis_XRP_USDT.txt"
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(report)
        print(f"\nReporte guardado en: {report_file}")
        
        logger.info("Advanced regime analysis completed!")
        
    except ImportError as e:
        print("ERROR: Missing dependencies for advanced regime analysis")
        print("Install with: pip install yfinance arch-py hmmlearn scikit-learn")
        print(f"Details: {e}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()