#!/usr/bin/env python3
"""
Script optimizado para entrenar todos los símbolos en paralelo usando 3 GPUs AMD RX 6600.

Distribuye automáticamente los 19 símbolos entre las 3 GPUs disponibles.
Cada GPU entrena ambos timeframes (5m y 15m) para los símbolos asignados.

Uso:
    python scripts/train_all_parallel.py
    python scripts/train_all_parallel.py --epochs 200 --batch-size 256
    python scripts/train_all_parallel.py --symbols BTCUSDT,ETHUSDT  # Solo algunos símbolos
"""

import subprocess
import sys
import time
import signal
from pathlib import Path
from datetime import datetime
import argparse

# Símbolos organizados por tier (según CLAUDE.md)
TIER1_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
TIER2_SYMBOLS = ['XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'DOTUSDT', 'AVAXUSDT', 'LINKUSDT', 'LTCUSDT']
TIER3_SYMBOLS = ['BCHUSDT', 'UNIUSDT', 'TRXUSDT', 'ETCUSDT', 'XLMUSDT', 'XMRUSDT', 'RUNEUSDT', 'ARBUSDT']

ALL_SYMBOLS = TIER1_SYMBOLS + TIER2_SYMBOLS + TIER3_SYMBOLS

# Configuración de hiperparámetros por tier
TIER_CONFIGS = {
    'tier1': {
        'hidden_dim': 192,
        'lstm_layers': 3,
        'dropout': 0.25,
        'target_return': 0.0015,
        'sequence_length': 48,
    },
    'tier2': {
        'hidden_dim': 192,
        'lstm_layers': 3,
        'dropout': 0.30,
        'target_return': 0.002,
        'sequence_length': 48,
    },
    'tier3': {
        'hidden_dim': 192,
        'lstm_layers': 3,
        'dropout': 0.35,
        'target_return': 0.0025,
        'sequence_length': 48,
    }
}

TIMEFRAMES = ['5m', '15m']

# Configuración de GPUs
N_GPUS = 3  # 3x AMD RX 6600


class GPUTrainingManager:
    """Maneja el entrenamiento paralelo en múltiples GPUs."""

    def __init__(self, n_gpus=3, verbose=True):
        self.n_gpus = n_gpus
        self.verbose = verbose
        self.processes = {}  # gpu_id -> proceso actual
        self.completed_jobs = []
        self.failed_jobs = []
        self.start_time = time.time()

        # Crear directorio de logs
        self.log_dir = Path(__file__).parent.parent / 'logs' / 'parallel_training'
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Manejar señales para limpieza
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Maneja señales de interrupción."""
        print("\n⚠️  Señal de interrupción recibida. Deteniendo entrenamientos...")
        self._cleanup()
        sys.exit(1)

    def _cleanup(self):
        """Detiene todos los procesos en ejecución."""
        for gpu_id, proc in self.processes.items():
            if proc and proc.poll() is None:
                print(f"Deteniendo proceso en GPU {gpu_id}...")
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()

    def _log(self, msg):
        """Imprime mensaje con timestamp."""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {msg}")

    def _get_tier_config(self, symbol):
        """Obtiene configuración según el tier del símbolo."""
        if symbol in TIER1_SYMBOLS:
            return 'tier1', TIER_CONFIGS['tier1']
        elif symbol in TIER2_SYMBOLS:
            return 'tier2', TIER_CONFIGS['tier2']
        else:
            return 'tier3', TIER_CONFIGS['tier3']

    def _build_training_command(self, symbol, timeframe, config, base_args):
        """Construye el comando de entrenamiento."""
        tier_name, tier_config = self._get_tier_config(symbol)

        cmd = [
            sys.executable,
            'scripts/train_production_ready.py',
            '--symbol', symbol,
            '--timeframe', timeframe,
            '--device', 'cuda:0',  # Siempre cuda:0 porque CUDA_VISIBLE_DEVICES limita la visibilidad
        ]

        # Agregar hiperparámetros del tier
        for key, value in tier_config.items():
            cmd.extend([f'--{key.replace("_", "-")}', str(value)])

        # Agregar argumentos base (pueden sobrescribir tier config)
        for arg in base_args:
            if arg.startswith('--'):
                cmd.append(arg)
            else:
                cmd.append(arg)

        return cmd

    def _start_training(self, gpu_id, symbol, timeframe, config, base_args):
        """Inicia un entrenamiento en una GPU específica."""
        # Configurar environment para usar GPU específica
        import os
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        env['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'  # Para RX 6600

        # Construir comando
        cmd = self._build_training_command(symbol, timeframe, config, base_args)

        # Archivo de log
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"gpu{gpu_id}_{symbol}_{timeframe}_{timestamp}.log"

        self._log(f"🚀 GPU {gpu_id}: Iniciando {symbol} {timeframe}")

        # Iniciar proceso
        with open(log_file, 'w') as f:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=Path(__file__).parent.parent
            )

        self.processes[gpu_id] = {
            'process': proc,
            'symbol': symbol,
            'timeframe': timeframe,
            'log_file': log_file,
            'start_time': time.time()
        }

        return proc

    def _wait_for_gpu(self):
        """Espera a que una GPU se libere. Retorna el ID de la GPU libre."""
        while True:
            # Verificar procesos completados
            for gpu_id in list(self.processes.keys()):
                job = self.processes[gpu_id]
                proc = job['process']

                if proc.poll() is not None:  # Proceso terminado
                    elapsed = time.time() - job['start_time']

                    if proc.returncode == 0:
                        self._log(f"✅ GPU {gpu_id}: {job['symbol']} {job['timeframe']} "
                                f"completado en {elapsed/60:.1f} min")
                        self.completed_jobs.append(job)
                    else:
                        self._log(f"❌ GPU {gpu_id}: {job['symbol']} {job['timeframe']} "
                                f"FALLÓ (código {proc.returncode})")
                        self._log(f"   Ver log: {job['log_file']}")
                        self.failed_jobs.append(job)

                    # GPU ahora está libre
                    del self.processes[gpu_id]
                    return gpu_id

            # Verificar si hay GPU idle
            for gpu_id in range(self.n_gpus):
                if gpu_id not in self.processes:
                    return gpu_id

            # No hay GPUs libres, esperar
            time.sleep(5)

    def train_all(self, symbols, timeframes, config, base_args):
        """Entrena todos los símbolos y timeframes en paralelo."""
        # Crear lista de trabajos
        jobs = []
        for symbol in symbols:
            for timeframe in timeframes:
                jobs.append({
                    'symbol': symbol,
                    'timeframe': timeframe
                })

        total_jobs = len(jobs)
        self._log(f"📋 Total de trabajos: {total_jobs}")
        self._log(f"🎮 GPUs disponibles: {self.n_gpus}")
        self._log(f"⚡ Speedup esperado: ~{self.n_gpus}x")
        self._log(f"⏱️  Tiempo estimado: {(total_jobs * 2.5) / self.n_gpus:.1f} horas")
        self._log("")

        # Procesar cola de trabajos
        job_idx = 0

        try:
            while job_idx < total_jobs or self.processes:
                # Iniciar nuevos trabajos en GPUs libres
                while job_idx < total_jobs:
                    # Encontrar GPU libre
                    free_gpu = None
                    for gpu_id in range(self.n_gpus):
                        if gpu_id not in self.processes:
                            free_gpu = gpu_id
                            break

                    if free_gpu is None:
                        break  # No hay GPUs libres

                    # Obtener siguiente trabajo
                    job = jobs[job_idx]
                    job_idx += 1

                    # Iniciar entrenamiento
                    self._start_training(
                        free_gpu,
                        job['symbol'],
                        job['timeframe'],
                        config,
                        base_args
                    )

                    time.sleep(2)  # Pequeña pausa entre inicios

                # Esperar a que se libere una GPU
                if self.processes:
                    self._wait_for_gpu()

        except KeyboardInterrupt:
            self._log("\n⚠️  Interrupción del usuario")
            self._cleanup()
            sys.exit(1)

        # Resumen final
        self._print_summary()

    def _print_summary(self):
        """Imprime resumen del entrenamiento."""
        elapsed_total = time.time() - self.start_time

        print("\n" + "="*80)
        print("📊 RESUMEN DE ENTRENAMIENTO")
        print("="*80)
        print(f"Tiempo total: {elapsed_total/3600:.2f} horas")
        print(f"Completados: {len(self.completed_jobs)}")
        print(f"Fallidos: {len(self.failed_jobs)}")

        if self.completed_jobs:
            print("\n✅ Trabajos completados:")
            for job in self.completed_jobs:
                print(f"  - {job['symbol']} {job['timeframe']}")

        if self.failed_jobs:
            print("\n❌ Trabajos fallidos:")
            for job in self.failed_jobs:
                print(f"  - {job['symbol']} {job['timeframe']}")
                print(f"    Log: {job['log_file']}")

        print("="*80 + "\n")

        if self.failed_jobs:
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Entrenamiento paralelo de todos los símbolos en 3 GPUs'
    )

    # Selección de símbolos
    parser.add_argument(
        '--symbols',
        type=str,
        default='all',
        help='Lista de símbolos separados por coma, o "all" para todos (default: all)'
    )

    parser.add_argument(
        '--timeframes',
        type=str,
        default='5m,15m',
        help='Timeframes separados por coma (default: 5m,15m)'
    )

    # Hiperparámetros globales (sobrescriben tier configs)
    parser.add_argument('--epochs', type=int, help='Número de épocas')
    parser.add_argument('--batch-size', type=int, help='Tamaño de batch')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, help='Dimensión oculta')
    parser.add_argument('--lstm-layers', type=int, help='Número de capas LSTM')
    parser.add_argument('--dropout', type=float, help='Dropout rate')
    parser.add_argument('--sequence-length', type=int, help='Longitud de secuencia')
    parser.add_argument('--prediction-horizon', type=int, help='Horizonte de predicción')
    parser.add_argument('--target-return', type=float, help='Return target threshold')

    args = parser.parse_args()

    # Procesar símbolos
    if args.symbols.lower() == 'all':
        symbols = ALL_SYMBOLS
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
        # Validar símbolos
        invalid = [s for s in symbols if s not in ALL_SYMBOLS]
        if invalid:
            print(f"❌ Símbolos inválidos: {invalid}")
            print(f"Símbolos válidos: {ALL_SYMBOLS}")
            sys.exit(1)

    # Procesar timeframes
    timeframes = [t.strip() for t in args.timeframes.split(',')]

    # Construir argumentos adicionales
    base_args = []
    if args.epochs:
        base_args.extend(['--epochs', str(args.epochs)])
    if args.batch_size:
        base_args.extend(['--batch-size', str(args.batch_size)])
    if args.lr:
        base_args.extend(['--lr', str(args.lr)])
    if args.hidden_dim:
        base_args.extend(['--hidden-dim', str(args.hidden_dim)])
    if args.lstm_layers:
        base_args.extend(['--lstm-layers', str(args.lstm_layers)])
    if args.dropout:
        base_args.extend(['--dropout', str(args.dropout)])
    if args.sequence_length:
        base_args.extend(['--sequence-length', str(args.sequence_length)])
    if args.prediction_horizon:
        base_args.extend(['--prediction-horizon', str(args.prediction_horizon)])
    if args.target_return:
        base_args.extend(['--target-return', str(args.target_return)])

    # Banner
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║             Entrenamiento Paralelo Multi-GPU (3x RX 6600)               ║
    ║                      Trading System - Batch Training                     ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)

    print(f"Símbolos a entrenar: {len(symbols)}")
    print(f"Timeframes: {timeframes}")
    print(f"Total de modelos: {len(symbols) * len(timeframes)}")
    print(f"GPUs: 3x AMD Radeon RX 6600")
    print("")

    if len(symbols) <= 5:
        print(f"Símbolos: {', '.join(symbols)}")
    else:
        print(f"Tier 1: {[s for s in symbols if s in TIER1_SYMBOLS]}")
        print(f"Tier 2: {[s for s in symbols if s in TIER2_SYMBOLS]}")
        print(f"Tier 3: {[s for s in symbols if s in TIER3_SYMBOLS]}")

    print("")
    input("Presiona Enter para comenzar (Ctrl+C para cancelar)...")
    print("")

    # Crear manager y entrenar
    manager = GPUTrainingManager(n_gpus=N_GPUS, verbose=True)
    manager.train_all(symbols, timeframes, TIER_CONFIGS, base_args)


if __name__ == '__main__':
    main()
