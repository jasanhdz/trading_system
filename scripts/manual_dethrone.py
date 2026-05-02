
import os
import shutil
import numpy as np
import torch
from pathlib import Path
import sys

# Añadir el path para importar el entorno
sys.path.append('/home/jasan/Develop/trading_system')
from scripts.phantom_v30.matrix_env import PhantomMatrixEnv
from scripts.phantom_v30.matrix_trainer import evaluate_model, CHAMPION_PATH, SAFE_CHECKPOINT_PATH, load_tensor_data, MAX_DD_THRESHOLD

def get_utility(pnl, dd):
    return pnl * np.power(np.maximum(1.0 - dd, 0.001), 1.5)

def manual_coliseum():
    print("🏟️ MANUAL COLISEUM: FORCED SUCCESSION")
    
    # Rutas absolutas para evitar problemas con CWD
    CHAMPION_ABS = "/home/jasan/Develop/trading_system/models/phantom_v30_champion.zip"
    SAFE_ABS = "/home/jasan/Develop/trading_system/models/phantom_v31_safe_checkpoint.zip"

    # 1. Cargar datos de validación
    eval_data = load_tensor_data("cpu", split="val")
    eval_env = PhantomMatrixEnv(
        features=eval_data['features'].numpy(),
        close_prices=eval_data['close'].numpy(),
        num_envs=32
    )

    # 2. Evaluar Campeón actual (Kamikaze)
    print(f"\n📊 Evaluating Current Champion: {CHAMPION_ABS}")
    c_pnl, c_dd, _, _ = evaluate_model(CHAMPION_ABS, eval_env)
    c_util = get_utility(c_pnl, c_dd)
    print(f"   > PnL: ${c_pnl:.2f} | DD: {c_dd*100:.1f}% | Utility: {c_util:.3f}")

    # 3. Evaluar Safe Checkpoint (Candidato Limpio)
    print(f"\n📊 Evaluating Candidate: {SAFE_ABS}")
    s_pnl, s_dd, _, _ = evaluate_model(SAFE_ABS, eval_env)
    s_util = get_utility(s_pnl, s_dd)
    print(f"   > PnL: ${s_pnl:.2f} | DD: {s_dd*100:.1f}% | Utility: {s_util:.3f}")

    # 4. Decisión de Sucesión
    print("\n⚖️ VERDICT:")
    if s_util > c_util:
        print(f"🚀 SUCCESSION APPROVED! Candidate is {((s_util/c_util)-1)*100:.1f}% more efficient.")
        backup_path = f"{CHAMPION_ABS}.kamikaze_manual_backup_{int(c_dd*100)}"
        if os.path.exists(CHAMPION_ABS):
            os.rename(CHAMPION_ABS, backup_path)
        shutil.copy2(SAFE_ABS, CHAMPION_ABS)
        print(f"✅ New Clean Champion crowned! (Legacy backup: {backup_path})")
    else:
        print("🛡️ SUCCESSION DENIED! Champion remains more efficient for now.")

    eval_env.close()

if __name__ == "__main__":
    manual_coliseum()
