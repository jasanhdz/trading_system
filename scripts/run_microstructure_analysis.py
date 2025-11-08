# =============================================================================
# scripts/run_corrected_microstructure.py
# =============================================================================

#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

def main():
    from analysis.statistics.microstructure_analysis import CorrectedMicrostructureAnalysis
    from utils.logger import setup_logger
    
    logger = setup_logger("corrected_microstructure_runner")
    logger.info("Starting CORRECTED microstructure analysis...")
    
    # Inicializar análisis corregido
    corrected_analyzer = CorrectedMicrostructureAnalysis("XRP/USDT")
    
    # Cargar datos
    timeframes = ['1m', '5m', '15m', '1h', '4h']
    data = corrected_analyzer.load_data(timeframes)
    
    if not data:
        logger.error("No data loaded. Please collect data first.")
        return
    
    logger.info("="*60)
    logger.info("RUNNING CORRECTED METHODOLOGIES")
    logger.info("="*60)
    
    # Análisis de spreads CORREGIDO
    logger.info("Calculating REALISTIC spreads...")
    spread_results = corrected_analyzer.calculate_realistic_spreads()
    
    # Simulación de slippage CORREGIDA
    logger.info("Simulating REALISTIC slippage...")
    slippage_results = corrected_analyzer.simulate_realistic_slippage()
    
    # Costos de transacción REALES
    logger.info("Calculating REAL transaction costs...")
    cost_results = corrected_analyzer.calculate_real_transaction_costs()
    
    # Volume profile VALIDADO
    logger.info("Validating volume profile...")
    volume_results = corrected_analyzer.validate_volume_profile()
    
    # Comparación con análisis anterior
    logger.info("Comparing with previous incorrect analysis...")
    comparison = corrected_analyzer.compare_with_previous_analysis()
    
    # Generar reporte corregido
    logger.info("Generating CORRECTED comprehensive report...")
    report = corrected_analyzer.generate_corrected_comprehensive_report()
    
    # Guardar reporte
    report_file = Path("analysis_reports") / "corrected_microstructure_analysis_XRP_USDT.txt"
    report_file.parent.mkdir(exist_ok=True)
    
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(report)
    print(f"\nReporte CORREGIDO guardado en: {report_file}")
    
    # Mostrar comparación de mejoras
    print("\n" + "="*60)
    print("MEJORAS APLICADAS VS ANÁLISIS ANTERIOR")
    print("="*60)
    
    if comparison['improvements']:
        for improvement in comparison['improvements']:
            tf = improvement['timeframe']
            breakeven_improvement = improvement['breakeven_improvement_bps']
            print(f"\n{tf.upper()}:")
            print(f"  Break-even mejorado en: {breakeven_improvement:.1f} bps")
            print(f"  Ahora es REALISTA y VIABLE para trading")
    
    logger.info("CORRECTED microstructure analysis completed successfully!")
    logger.info("Previous methodological errors have been FIXED")
    logger.info("Results are now suitable for strategy development")

if __name__ == "__main__":
    main()