#!/usr/bin/env python3
import sys
from pathlib import Path
import json
import shutil

def consolidate_model(model_dir):
    path = Path(model_dir)
    results_path = path / "production_training_results.json"
    
    if not results_path.exists():
        print(f"❌ No se encontró results.json en {path}")
        return

    data = json.loads(results_path.read_text())
    fold_results = data.get('fold_results', [])
    
    fold_idx = 5 # Default fallback
    
    if fold_results:
        best_fold = max(fold_results, key=lambda x: x['test_metrics']['macro_f1'])
        fold_idx = best_fold['fold']
        print(f"🏆 Mejor Fold (según métricas): {fold_idx} (F1: {best_fold['test_metrics']['macro_f1']:.4f})")
    else:
        print("⚠️  No se encontraron métricas de folds en el JSON. Buscando archivos físicos...")
        # Buscar el último fold existente
        for i in range(5, 0, -1):
            if (path / f"best_model_fold{i}.pt").exists():
                fold_idx = i
                print(f"👉 Usando Fold {fold_idx} (Último disponible)")
                break
    
    # Archivos a consolidar
    files_to_copy = {
        f"scaler_fold{fold_idx}.pkl": "scaler.pkl",
        f"feature_selector_fold{fold_idx}.pkl": "feature_selector.pkl",
        f"best_model_fold{fold_idx}.pt": "model.pt"
    }
    
    for src, dst in files_to_copy.items():
        src_path = path / src
        dst_path = path / dst
        if src_path.exists():
            shutil.copy(src_path, dst_path)
            print(f"   ✅ Copiado {src} -> {dst}")
        else:
            print(f"   ⚠️  No se encontró {src}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 consolidate_best_model.py <model_dir>")
        sys.exit(1)
        
    consolidate_model(sys.argv[1])
