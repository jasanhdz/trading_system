# =============================================================================
# scripts/run_formal_statistics.py - VERSIÓN SIMPLIFICADA
# =============================================================================
#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

def main():
    try:
        from analysis.statistics.formal_statistics import FormalStatisticalAnalysis
        from utils.logger import setup_logger
        
        logger = setup_logger("formal_stats_runner")
        logger.info("Starting formal statistical analysis...")
        
        # Inicializar análisis
        formal_stats = FormalStatisticalAnalysis("XRP/USDT")
        
        # Cargar datos
        timeframes = ['1m', '5m', '15m', '1h', '4h']
        data = formal_stats.load_and_prepare_data(timeframes)
        
        if not data:
            logger.error("No data loaded. Please collect data first.")
            return
        
        # Ejecutar análisis
        logger.info("Running autocorrelation analysis (ACF/PACF)...")
        formal_stats.autocorrelation_analysis(max_lags=100)
        
        logger.info("Running formal ARCH tests...")
        formal_stats.arch_test_analysis(max_lags=20)
        
        logger.info("Running stationarity tests...")
        formal_stats.stationarity_tests()
        
        # Generar reporte
        report = formal_stats.generate_comprehensive_report()
        
        # Guardar reporte
        report_file = Path("analysis_reports") / "formal_statistical_analysis_XRP_USDT.txt"
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(report)
        print(f"\nReporte guardado en: {report_file}")
        
        logger.info("Formal statistical analysis completed!")
        
    except ImportError as e:
        print("ERROR: statsmodels not installed")
        print("Run: pip install statsmodels>=0.14.0")
        print(f"Details: {e}")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    main()