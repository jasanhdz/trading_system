import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from matrix_env import PhantomMatrixEnv
from matrix_trainer import evaluate_model_single
import torch

MODELS_DIR = "/home/jasan/Develop/trading_system/models"

CHAMPIONS = {
    "V31_Monstruo_Actual": os.path.join(MODELS_DIR, "phantom_v30_champion.zip"),
    "Marzo_22_Conservador": os.path.join(MODELS_DIR, "phantom_v30_champion.zip.backup_1774198651"),
    "V11_Bear_Sniper": os.path.join(MODELS_DIR, "phantom_v30_champion_v11_sniper_backup.zip"),
    "Marzo_12_Primitivo": os.path.join(MODELS_DIR, "phantom_v30_champion.zip.backup_1773356230"),
}

def load_tensor_data_local(device="cpu", split="val"):
    base_dir = "/home/jasan/Develop/trading_system/ml/python/data/tensors"
    fpath = os.path.join(base_dir, f"{split}_data.pt")
    return torch.load(fpath, map_location=device, weights_only=True)

from tensor_loader import load_tensor_data

if __name__ == "__main__":
    print("🔥 INICIANDO EL GRAN COLISEO DE CAMPEONES 🔥\n")
    
    try:
        eval_data = load_tensor_data("cpu", split="val")
        features_np = eval_data['features'].numpy()
        close_np = eval_data['close'].numpy()
    except Exception as e:
        print(f"Error cargando matriz numpy: {e}")
        sys.exit(1)
        
    print("Market Data Cargada. Instanciando Entorno de Validacion Vectorizado...")
    
    eval_env = PhantomMatrixEnv(
        features=features_np,
        close_prices=close_np,
        num_envs=32, 
    )
    
    results = []
    
    for name, path in CHAMPIONS.items():
        print(f"\n🗡️ Evaluando: {name} ...")
        if not os.path.exists(path):
            print(f"   ⚠️ Modelo no encontrado: {path}")
            continue
            
        # evaluate_model_single corre 1 semilla y retorna balance, P95 DD, acciones y SignalQ
        bal, dd, ac, _ = evaluate_model_single(path, eval_env, seed=42)
        
        idle = ac.get(0, 0)
        longs = ac.get(1, 0)
        shorts = ac.get(2, 0)
        closes = ac.get(3, 0)
        
        total_actions = idle + longs + shorts + closes
        trades = longs + shorts
        
        tr = (trades / total_actions * 100) if total_actions > 0 else 0.0
            
        results.append({
            "name": name,
            "pnl": bal,
            "dd": dd * 100.0,
            "tr": tr,
            "longs": longs,
            "shorts": shorts,
            "idle": idle
        })

    print("\n\n🏆 RESULTADOS FINALES DEL TORNEO 🏆\n")
    print(f"| Modelo | Retorno Seguro Mín. (PnL) | Drawdown P95 (%) | Trading Rate (%) | Longs Efectuados | Shorts Efectuados | Idle (Pasividad) |")
    print("|---|---|---|---|---|---|---|")
    
    results.sort(key=lambda x: x["pnl"], reverse=True)
    
    for r in results:
        pnl = f"${r['pnl']:.2f}"
        dd = f"{r['dd']:.1f}%"
        tr = f"{r['tr']:.1f}%"
        nm = r["name"].replace("_", " ")
        print(f"| {nm} | {pnl} | {dd} | {tr} | {r['longs']} | {r['shorts']} | {r['idle']} |")
        
    eval_env.close()
