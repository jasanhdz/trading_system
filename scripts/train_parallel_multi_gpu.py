#!/usr/bin/env python3
"""
Parallel Multi-GPU Training Script
Trains multiple models simultaneously across available GPUs.

Usage:
    # Train BTC on 3 GPUs (5m, 15m, ensemble)
    python scripts/train_parallel_multi_gpu.py --symbol BTCUSDT --gpus 0,1,2

    # Train multiple symbols in parallel
    python scripts/train_parallel_multi_gpu.py --symbols BTCUSDT,ETHUSDT,SOLUSDT --gpus 0,1,2

    # Train all symbols (batch mode)
    python scripts/train_parallel_multi_gpu.py --mode batch --gpus 0,1,2
"""

import argparse
import subprocess
import time
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import signal

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class MultiGPUTrainer:
    """Manages parallel training across multiple GPUs"""

    def __init__(self, gpus: List[int], verbose: bool = True):
        self.gpus = gpus
        self.verbose = verbose
        self.processes: Dict[int, subprocess.Popen] = {}
        self.training_jobs: Dict[int, Dict] = {}
        self.start_time = time.time()

    def _log(self, msg: str):
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {msg}")

    def train_single(
        self,
        gpu_id: int,
        script: str,
        symbol: str,
        timeframe: str,
        extra_args: List[str] = None
    ) -> subprocess.Popen:
        """Train a single model on a specific GPU"""

        # Set environment to use specific GPU
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        # Build command
        cmd = [
            sys.executable,
            script,
            '--symbol', symbol,
            '--timeframe', timeframe,
            '--device', 'cuda:0',  # Always use cuda:0 since CUDA_VISIBLE_DEVICES limits visibility
        ]

        if extra_args:
            cmd.extend(extra_args)

        # Log files
        log_dir = project_root / 'logs' / 'multi_gpu'
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{symbol}_{timeframe}_gpu{gpu_id}_{timestamp}.log"

        self._log(f"🚀 GPU {gpu_id}: Starting {symbol} {timeframe}")
        self._log(f"   Command: {' '.join(cmd)}")
        self._log(f"   Log: {log_file}")

        # Start process
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True
            )

        # Track job
        job_info = {
            'symbol': symbol,
            'timeframe': timeframe,
            'script': script,
            'log_file': str(log_file),
            'start_time': time.time(),
            'pid': process.pid
        }

        self.processes[gpu_id] = process
        self.training_jobs[gpu_id] = job_info

        return process

    def wait_for_gpu(self, timeout: Optional[int] = None) -> Optional[int]:
        """Wait for any GPU to become available"""

        start = time.time()

        while True:
            # Check for completed processes
            for gpu_id, process in list(self.processes.items()):
                if process.poll() is not None:  # Process finished
                    job = self.training_jobs[gpu_id]
                    elapsed = time.time() - job['start_time']

                    if process.returncode == 0:
                        self._log(f"✅ GPU {gpu_id}: {job['symbol']} {job['timeframe']} completed in {elapsed/60:.1f} min")
                    else:
                        self._log(f"❌ GPU {gpu_id}: {job['symbol']} {job['timeframe']} FAILED (exit code {process.returncode})")
                        self._log(f"   Check log: {job['log_file']}")

                    # GPU is now available
                    del self.processes[gpu_id]
                    return gpu_id

            # Check for idle GPUs
            for gpu_id in self.gpus:
                if gpu_id not in self.processes:
                    return gpu_id

            # Timeout check
            if timeout and (time.time() - start) > timeout:
                return None

            time.sleep(5)  # Check every 5 seconds

    def train_queue(self, training_queue: List[Dict]):
        """Train a queue of jobs across available GPUs"""

        self._log(f"📋 Training queue: {len(training_queue)} jobs")
        self._log(f"🎮 Available GPUs: {self.gpus}")
        self._log("")

        queue = training_queue.copy()
        completed = []
        failed = []

        try:
            while queue or self.processes:
                # Start new jobs on idle GPUs
                while queue:
                    # Find idle GPU
                    idle_gpu = None
                    for gpu_id in self.gpus:
                        if gpu_id not in self.processes:
                            idle_gpu = gpu_id
                            break

                    if idle_gpu is None:
                        break  # No idle GPUs, wait

                    # Pop next job
                    job = queue.pop(0)

                    # Start training
                    self.train_single(
                        gpu_id=idle_gpu,
                        script=job['script'],
                        symbol=job['symbol'],
                        timeframe=job['timeframe'],
                        extra_args=job.get('extra_args', [])
                    )

                    time.sleep(2)  # Small delay between starts

                # Wait for a GPU to finish
                if self.processes:
                    finished_gpu = self.wait_for_gpu(timeout=10)

                    if finished_gpu is not None:
                        job = self.training_jobs[finished_gpu]

                        # Check if successful
                        if self.processes.get(finished_gpu) and self.processes[finished_gpu].returncode == 0:
                            completed.append(job)
                        else:
                            failed.append(job)

        except KeyboardInterrupt:
            self._log("\n⚠️  Interrupt received, stopping all training...")
            self.stop_all()
            sys.exit(1)

        # Final summary
        elapsed_total = time.time() - self.start_time

        self._log("")
        self._log("="*80)
        self._log("📊 Training Summary")
        self._log("="*80)
        self._log(f"Total time: {elapsed_total/60:.1f} minutes")
        self._log(f"Completed: {len(completed)}")
        self._log(f"Failed: {len(failed)}")

        if completed:
            self._log("\n✅ Completed jobs:")
            for job in completed:
                self._log(f"  - {job['symbol']} {job['timeframe']}")

        if failed:
            self._log("\n❌ Failed jobs:")
            for job in failed:
                self._log(f"  - {job['symbol']} {job['timeframe']}")
                self._log(f"    Log: {job['log_file']}")

        self._log("="*80)

        return completed, failed

    def stop_all(self):
        """Stop all running training processes"""
        for gpu_id, process in self.processes.items():
            if process.poll() is None:  # Still running
                self._log(f"Stopping GPU {gpu_id} (PID {process.pid})...")
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()


def create_training_queue(
    mode: str,
    symbols: List[str],
    timeframes: List[str],
    script: str,
    extra_args: List[str]
) -> List[Dict]:
    """Create training job queue"""

    queue = []

    if mode == 'single':
        # Train single model per symbol/timeframe
        for symbol in symbols:
            for timeframe in timeframes:
                queue.append({
                    'script': script,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'extra_args': extra_args
                })

    elif mode == 'ensemble':
        # Train ensemble per symbol/timeframe (one job, parallel internally)
        for symbol in symbols:
            for timeframe in timeframes:
                queue.append({
                    'script': str(project_root / 'scripts' / 'train_ensemble.py'),
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'extra_args': extra_args
                })

    elif mode == 'batch':
        # Batch mode: all tier 1 + tier 2 symbols
        tier1_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
        tier2_symbols = ['XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'DOTUSDT', 'AVAXUSDT']

        for symbol in tier1_symbols + tier2_symbols:
            for timeframe in timeframes:
                queue.append({
                    'script': script,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'extra_args': extra_args
                })

    return queue


def main():
    parser = argparse.ArgumentParser(description='Multi-GPU parallel training')

    # GPU configuration
    parser.add_argument('--gpus', type=str, default='0,1,2',
                        help='Comma-separated GPU IDs (default: 0,1,2)')

    # Training configuration
    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'ensemble', 'batch'],
                        help='Training mode (default: single)')
    parser.add_argument('--symbols', type=str, default='BTCUSDT',
                        help='Comma-separated symbols (default: BTCUSDT)')
    parser.add_argument('--timeframes', type=str, default='5m,15m',
                        help='Comma-separated timeframes (default: 5m,15m)')

    # Script selection
    parser.add_argument('--script', type=str,
                        default='scripts/train_production_ready.py',
                        help='Training script to use')

    # Training hyperparameters (passed through)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--hidden-dim', type=int, default=192)
    parser.add_argument('--lstm-layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.3)

    # Dataset configuration
    parser.add_argument('--sequence-length', type=int, default=48)
    parser.add_argument('--prediction-horizon', type=int, default=6)
    parser.add_argument('--target-return', type=float, default=0.005)

    args = parser.parse_args()

    # Parse GPU list
    gpus = [int(x.strip()) for x in args.gpus.split(',')]

    # Parse symbols and timeframes
    symbols = [x.strip() for x in args.symbols.split(',')]
    timeframes = [x.strip() for x in args.timeframes.split(',')]

    # Build extra args to pass to training script
    extra_args = [
        '--epochs', str(args.epochs),
        '--batch-size', str(args.batch_size),
        '--lr', str(args.lr),
        '--hidden-dim', str(args.hidden_dim),
        '--lstm-layers', str(args.lstm_layers),
        '--dropout', str(args.dropout),
        '--sequence-length', str(args.sequence_length),
        '--prediction-horizon', str(args.prediction_horizon),
        '--target-return', str(args.target_return),
    ]

    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                   Multi-GPU Parallel Training                            ║
    ║                    Trading System - Fast Training                        ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)

    print(f"GPUs: {gpus}")
    print(f"Mode: {args.mode}")
    print(f"Symbols: {symbols}")
    print(f"Timeframes: {timeframes}")
    print(f"Script: {args.script}")
    print("")

    # Create training queue
    queue = create_training_queue(
        mode=args.mode,
        symbols=symbols,
        timeframes=timeframes,
        script=args.script,
        extra_args=extra_args
    )

    if not queue:
        print("❌ No training jobs created!")
        sys.exit(1)

    print(f"📋 Total jobs: {len(queue)}")
    print(f"⚡ Expected speedup: ~{len(gpus)}x")
    print(f"⏱️  Estimated time: {(len(queue) * 3) / len(gpus):.1f} hours (assuming 3h per job)")
    print("")

    input("Press Enter to start training (Ctrl+C to cancel)...")
    print("")

    # Start training
    trainer = MultiGPUTrainer(gpus=gpus, verbose=True)

    try:
        completed, failed = trainer.train_queue(queue)

        if failed:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        trainer.stop_all()
        sys.exit(1)


if __name__ == '__main__':
    main()
