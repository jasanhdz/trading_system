#!/usr/bin/env python3
# scripts/view_diary.py
"""
Visualizador de la Bitácora de Entrenamiento.
Muestra las últimas métricas de accuracy para cada modelo.
"""
import sys
from pathlib import Path

# Hack para importar modulos desde ml/
sys.path.insert(0, str(Path(__file__).parents[1]))

from ml.utils.training_diary import TrainingDiary

def main():
    d = TrainingDiary()
    df = d.get_summary(limit=30)

    print("\n📓 ÚLTIMOS ENTRENAMIENTOS REGISTRADOS:")
    print("="*90)
    
    if not df.empty:
        cols = ['ts', 'symbol', 'model', 'accuracy', 'f1_score', 'samples']
        available_cols = [c for c in cols if c in df.columns]
        
        # Formatear timestamp
        df['ts'] = df['ts'].dt.strftime('%Y-%m-%d %H:%M')
        
        # Formatear accuracy como porcentaje
        if 'accuracy' in df.columns:
            df['accuracy'] = df['accuracy'].apply(lambda x: f"{x:.2%}" if x else "N/A")
        if 'f1_score' in df.columns:
            df['f1_score'] = df['f1_score'].apply(lambda x: f"{x:.2f}" if x else "N/A")
            
        print(df[available_cols].to_string(index=False))
    else:
        print("La bitácora está vacía. Ejecuta un entrenamiento primero.")
        
    print("="*90)

if __name__ == "__main__":
    main()
