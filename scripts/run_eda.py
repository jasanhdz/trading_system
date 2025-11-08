# =============================================================================
# 3. scripts/run_eda.py
"""
Script para ejecutar análisis exploratorio completo
"""
#!/usr/bin/env python3

import sys
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from analysis.statistics.exploratory import ExploratoryDataAnalysis
from utils.logger import setup_logger

def main():
    logger = setup_logger("eda_runner")
    
    logger.info("Starting comprehensive EDA analysis...")
    
    # Inicializar análisis
    eda = ExploratoryDataAnalysis("XRP/USDT")
    
    # Cargar datos
    timeframes = ['1m', '5m', '15m', '1h', '4h']
    data = eda.load_data(timeframes)
    
    if not data:
        logger.error("No data loaded. Please collect data first.")
        return
    
    # Ejecutar análisis completo
    logger.info("Running basic statistics...")
    eda.basic_statistics()
    
    logger.info("Analyzing returns distribution...")
    eda.returns_distribution_analysis()
    
    logger.info("Analyzing volatility patterns...")
    eda.volatility_analysis()
    
    logger.info("Analyzing temporal patterns...")
    eda.temporal_patterns()
    
    logger.info("Analyzing correlations...")
    eda.correlation_analysis()
    
    logger.info("Analyzing market regimes...")
    eda.market_regime_analysis()
    
    logger.info("Checking data gaps...")
    eda.gap_analysis()
    
    # Generar reporte
    logger.info("Generating comprehensive report...")
    report = eda.generate_comprehensive_report()
    
    # Guardar reporte
    report_file = Path("analysis_reports") / f"eda_report_{eda.symbol.replace('/', '_')}.txt"
    report_file.parent.mkdir(exist_ok=True)
    
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(report)
    print(f"\nReporte guardado en: {report_file}")
    
    logger.info("EDA analysis completed successfully!")

if __name__ == "__main__":
    main()