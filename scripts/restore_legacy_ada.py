#!/usr/bin/env python3
import shutil
from pathlib import Path
import os
from datetime import datetime

# Configuración
ROOT = Path("/home/jasan/Develop/trading_system")
MODELS_DIR = ROOT / "models/advanced"
CURRENT_ADA = MODELS_DIR / "ADAUSDT/1h"
DEPRECATED_DIR = MODELS_DIR / "deprecated/2025_12_19_2323"
OLD_ADA = DEPRECATED_DIR / "ADAUSDT/1h"
OLD_CONFIG = DEPRECATED_DIR / "thresholds_config.json"
EXPERIMENTS_DIR = MODELS_DIR / "experiments/2025_12_19_conservative_ada"

def restore_legacy_model():
    print("🔄 Iniciando restauración del modelo ADA Legacy...")
    
    # 1. Archivar el modelo actual (El 'Nuevo' que no nos convenció por poco volumen)
    if CURRENT_ADA.exists():
        print(f"   📦 Archivando modelo actual en: {EXPERIMENTS_DIR}")
        EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        if (EXPERIMENTS_DIR / "ADAUSDT").exists():
            shutil.rmtree(EXPERIMENTS_DIR / "ADAUSDT")
        shutil.move(str(CURRENT_ADA), str(EXPERIMENTS_DIR / "ADAUSDT"))
        
        # Crear nota explicativa
        readme_content = """# Experimento: Modelo ADA Conservador (Sniper)
Fecha: 2025-12-19

## Descripción
Este modelo fue entrenado buscando maximizar el Sharpe Ratio.
- **Arquitectura:** DeepTemporalNet (3 LSTM layers, 192 hidden)
- **Training:** FP32 (No AMP), Batch 64.
- **Métricas:**
  - Sharpe: 1.88
  - Win Rate: 59.8%
  - Trades (Test): 189

## Razón de Descarte
Aunque las métricas de calidad eran excelentes, el volumen de operaciones (189 trades en test) fue considerado insuficiente para la estrategia de rotación de capital del usuario. Se prefirió restaurar el modelo anterior ("Legacy") que generaba ~1177 trades con un Sharpe de 1.59, ofreciendo más oportunidades de mercado.
"""
        (EXPERIMENTS_DIR / "README.md").write_text(readme_content)
        print("   📝 Nota README.md creada.")

    # 2. Restaurar el modelo viejo
    if OLD_ADA.exists():
        print(f"   🔙 Restaurando modelo Legacy desde: {OLD_ADA}")
        CURRENT_ADA.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(OLD_ADA), str(CURRENT_ADA))
        print("   ✅ Modelo Legacy restaurado.")
    else:
        print("   ❌ ERROR: No se encontró el modelo viejo en deprecated!")
        return

    # 3. Restaurar configuración vieja
    if OLD_CONFIG.exists():
        print(f"   🔙 Restaurando thresholds_config.json desde: {OLD_CONFIG}")
        shutil.copy(str(OLD_CONFIG), str(MODELS_DIR / "thresholds_config.json"))
        print("   ✅ Configuración restaurada.")
    else:
        print("   ⚠️  No se encontró thresholds_config.json viejo, se mantiene el actual.")

if __name__ == "__main__":
    restore_legacy_model()
