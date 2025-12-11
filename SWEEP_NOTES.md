# Sweep / Entrenamientos - BTCUSDT 15m

## Hardware / GPUs
- AMD GPU1 inestable se retiró; solo quedan 2x RX 6600 (GPU0 y GPU1 actuales) + 1x NVIDIA GTX 1660.
- NVML requiere sudo en este host; usar `sudo smi-nvidia` para lectura concisa.

## Mejores hallazgos previos
- Sweep FAST (baseline):
  - Config: TR=0.002, ph=6, seq_len=48, hd=192, lstm=3, dropout=0.35, bs=128, lr=3e-4.
  - Entrenamiento completo (NVIDIA): Macro F1≈0.353, Long F1≈0.224, Short F1≈0.402. PnL positivo en sweep.
- Aggressive (parcial, 6/24 por shard antes de abortar):
  - TR=0.0015: F1≈0.315 máximo, PnL negativo.
  - TR=0.0025: PnL≈0.109 máximo, F1≈0.205; F1 bajo.
  - No superaron al FAST en F1/PnL.

## Corridas actuales
- Seeds adicionales del mejor FAST:
  - AMD GPU1 (ROCm) seed=123 → `logs/train_best_fast_seed123_gpu1.log`
  - NVIDIA seed=456 → `logs/train_best_fast_seed456_gpuN.log`
  - Mismo hiperparámetro del FAST; propósito: robustez/ensemble.
- Mini-grid focalizado en GPU0 (secuencial, 30 epochs, hd=192, bs=128, lr=3e-4, seed=42):
  - TR ∈ {0.002, 0.0025}, ph ∈ {4, 6}, dropout ∈ {0.30, 0.35}
  - Logs: `logs/train_grid_gpu0_tr{TR}_ph{PH}_dr{DR}.log`
  - Objetivo: encontrar mejoras pequeñas alrededor del FAST.

## Comandos útiles
- Procesos: `ps -ef | grep train_production_ready.py | grep -v grep`
- Logs en vivo:
  - Seeds: `tail -f logs/train_best_fast_seed123_gpu1.log` y `tail -f logs/train_best_fast_seed456_gpuN.log`
  - Grid: `tail -f logs/train_grid_gpu0_tr0.002_ph4_dr0.30.log` (y cambiar según combo)
- GPUs: `rocm-smi --showuse --showtemp` (AMD) y `sudo smi-nvidia` (NVIDIA).

## Siguientes pasos sugeridos
- Evaluar al cierre del grid y seeds: comparar Macro F1/PnL vs. baseline FAST.
- Si algún combo supera F1≈0.35 o PnL mejora, promoverlo a entrenamiento completo (50 epochs) y considerar ensemble de 2–3 seeds.
- Para despliegue como gatillo: usar umbrales conservadores, especialmente para la clase Long (históricamente F1 más bajo); validar con backtest reciente incluyendo costos/slippage y guards del bot.
