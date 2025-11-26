#!/usr/bin/env python3
"""
dispatch_training.py

Orquestador de entrenamiento para sistema híbrido Multi-GPU (AMD + NVIDIA).
Detecta GPUs disponibles y lanza procesos de entrenamiento en el entorno virtual adecuado.

Uso:
    python scripts/dispatch_training.py --symbols BTCUSDT,ETHUSDT --timeframes 5m,15m
"""

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List, Dict, Optional
import threading
from queue import Queue

@dataclass
class GPUInfo:
    id: int
    name: str
    type: str  # 'AMD' or 'NVIDIA'
    env_path: str
    python_path: str

@dataclass
class TrainingJob:
    symbol: str
    timeframe: str
    target_return: float = 0.005
    prediction_horizon: int = 6
    status: str = "PENDING"  # PENDING, RUNNING, COMPLETED, FAILED
    gpu_id: Optional[int] = None
    process: Optional[subprocess.Popen] = None

class GPUDispatcher:
    def __init__(self):
        self.gpus: List[GPUInfo] = self._detect_gpus()
        self.jobs: Queue[TrainingJob] = Queue()
        self.active_jobs: List[TrainingJob] = []
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        
    def _detect_gpus(self) -> List[GPUInfo]:
        """Detecta GPUs disponibles y asigna entornos."""
        gpus = []
        
        # 1. Detectar AMD GPUs (rocm-smi)
        try:
            result = subprocess.run(['rocm-smi', '--showid'], capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if "GPU[" in line:  # Más específico para evitar falsos positivos
                        try:
                            gpu_id = int(line.split('[')[1].split(']')[0])
                            # Verificar si ya agregamos esta GPU (por si acaso)
                            if not any(g.id == gpu_id and g.type == "AMD" for g in gpus):
                                gpus.append(GPUInfo(
                                    id=gpu_id,
                                    name=f"AMD GPU {gpu_id}",
                                    type="AMD",
                                    env_path=".venv_rocm57",
                                    python_path=".venv_rocm57/bin/python"
                                ))
                        except:
                            pass
        except FileNotFoundError:
            pass

        # 2. Detectar NVIDIA GPUs (nvidia-smi)
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,name', '--format=csv,noheader'], capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if line.strip():
                        parts = line.split(',')
                        gpu_idx = int(parts[0])
                        name = parts[1].strip()
                        
                        gpus.append(GPUInfo(
                            id=gpu_idx,
                            name=f"NVIDIA {name}",
                            type="NVIDIA",
                            env_path=".venv_cuda",
                            python_path=".venv_cuda/bin/python"
                        ))
        except FileNotFoundError:
            pass
            
        print(f"✅ GPUs Detectadas: {len(gpus)}")
        for gpu in gpus:
            print(f"  - [{gpu.type}] ID {gpu.id}: {gpu.name} (Env: {gpu.env_path})")
            
        return gpus

    def add_jobs(self, symbols: List[str], timeframes: List[str], target_return: float = 0.005, prediction_horizon: int = 6):
        for symbol in symbols:
            for tf in timeframes:
                # Check if model already exists
                results_path = f"models/advanced/{symbol}/{tf}/production_training_results.json"
                if os.path.exists(results_path):
                    print(f"⏩ Saltando {symbol} {tf} (Ya existe)")
                    continue
                    
                self.jobs.put(TrainingJob(
                    symbol=symbol, 
                    timeframe=tf,
                    target_return=target_return,
                    prediction_horizon=prediction_horizon
                ))
                
    def _run_job(self, gpu: GPUInfo, job: TrainingJob):
        print(f"🚀 Iniciando {job.symbol} {job.timeframe} en GPU {gpu.id} ({gpu.type})")
        
        # Configurar batch size según GPU
        batch_size = "96" if gpu.type == "AMD" else "128"
        
        cmd = [
            gpu.python_path,
            "-u",
            "scripts/train_production_ready.py",
            "--symbol", job.symbol,
            "--timeframe", job.timeframe,
            "--epochs", "200",
            "--batch-size", batch_size,
            "--target-return", str(job.target_return),
            "--prediction-horizon", str(job.prediction_horizon),
            "--device", "cuda:0", # Siempre cuda:0 porque aislamos la GPU
        ]
        
        # Environment variables
        env = os.environ.copy()
        
        if gpu.type == "NVIDIA":
            env["CUDA_VISIBLE_DEVICES"] = str(gpu.id)
            
        if gpu.type == "AMD":
            env["HIP_VISIBLE_DEVICES"] = str(gpu.id)
            env["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
            env["LD_LIBRARY_PATH"] = f"/opt/rocm/lib:{env.get('LD_LIBRARY_PATH', '')}"
            env["PYTORCH_HIP_ALLOC_CONF"] = "max_split_size_mb:512"
            env["MIOPEN_DISABLE_CACHE"] = "0"
            
        try:
            log_file = open(f"logs/multi_gpu/{job.symbol}_{job.timeframe}_{gpu.type}_{gpu.id}.log", "w")
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env
            )
            
            with self.lock:
                job.process = process
                job.status = "RUNNING"
                job.gpu_id = gpu.id
                self.active_jobs.append(job)
            
            process.wait()
            
            with self.lock:
                if job in self.active_jobs:
                    self.active_jobs.remove(job)
            
            if process.returncode == 0:
                job.status = "COMPLETED"
                print(f"✅ Completado: {job.symbol} {job.timeframe}")
            else:
                job.status = "FAILED"
                print(f"❌ Falló: {job.symbol} {job.timeframe}")
                
            log_file.close()
            
        except Exception as e:
            print(f"❌ Error ejecutando job: {e}")
            job.status = "FAILED"

    def run(self):
        if not self.gpus:
            print("❌ No se detectaron GPUs. Saliendo.")
            return

        threads = []
        
        for gpu in self.gpus:
            t = threading.Thread(target=self._gpu_worker, args=(gpu,))
            t.start()
            threads.append(t)
            
        try:
            # Main thread waits for queue to empty
            while any(t.is_alive() for t in threads):
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Deteniendo dispatcher... Matando procesos hijos...")
            self.stop_event.set()
            self._kill_all_jobs()
            
        for t in threads:
            t.join()
            
    def _kill_all_jobs(self):
        with self.lock:
            for job in self.active_jobs:
                if job.process:
                    print(f"   Matando {job.symbol} {job.timeframe}...")
                    try:
                        job.process.terminate()
                        job.process.wait(timeout=5)
                    except:
                        job.process.kill()

    def _gpu_worker(self, gpu: GPUInfo):
        while not self.stop_event.is_set():
            try:
                job = self.jobs.get(timeout=1)
            except:
                if self.jobs.empty():
                    break
                continue
                
            self._run_job(gpu, job)
            self.jobs.task_done()

def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Training Dispatcher")
    parser.add_argument("--symbols", type=str, required=True, help="Comma separated symbols")
    parser.add_argument("--timeframes", type=str, required=True, help="Comma separated timeframes")
    parser.add_argument("--target-return", type=float, default=0.005, help="Target return")
    parser.add_argument("--prediction-horizon", type=int, default=6, help="Prediction horizon")
    
    args = parser.parse_args()
    
    symbols = args.symbols.split(",")
    timeframes = args.timeframes.split(",")
    
    os.makedirs("logs/multi_gpu", exist_ok=True)
    
    dispatcher = GPUDispatcher()
    dispatcher.add_jobs(symbols, timeframes, args.target_return, args.prediction_horizon)
    dispatcher.run()

if __name__ == "__main__":
    main()
