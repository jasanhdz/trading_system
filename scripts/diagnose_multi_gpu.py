#!/usr/bin/env python3
"""
Multi-GPU Diagnostic Script
Detects all available GPUs (AMD ROCm + NVIDIA CUDA) and tests them.
"""

import os
import sys
import torch
import subprocess

def print_section(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")

def check_system_gpus():
    """Check GPUs detected by lspci"""
    print_section("System GPU Detection (lspci)")
    try:
        result = subprocess.run(['lspci', '-nn'], capture_output=True, text=True)
        lines = [l for l in result.stdout.split('\n') if 'VGA' in l or 'Display' in l or '3D' in l]

        amd_gpus = [l for l in lines if 'AMD' in l or 'ATI' in l]
        nvidia_gpus = [l for l in lines if 'NVIDIA' in l or 'nVidia' in l]

        print(f"AMD GPUs found: {len(amd_gpus)}")
        for gpu in amd_gpus:
            print(f"  - {gpu}")

        print(f"\nNVIDIA GPUs found: {len(nvidia_gpus)}")
        for gpu in nvidia_gpus:
            print(f"  - {gpu}")

        return len(amd_gpus), len(nvidia_gpus)
    except Exception as e:
        print(f"Error: {e}")
        return 0, 0

def check_pytorch():
    """Check PyTorch installation and GPU support"""
    print_section("PyTorch Configuration")

    print(f"PyTorch version: {torch.__version__}")
    print(f"Python version: {sys.version}")

    # Check CUDA
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA devices detected: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"\n  Device {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    Total memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"    Compute capability: {props.major}.{props.minor}")

    # Check ROCm
    print(f"\nROCm support: {hasattr(torch.version, 'hip') and torch.version.hip is not None}")
    if hasattr(torch.version, 'hip') and torch.version.hip:
        print(f"ROCm version: {torch.version.hip}")

    # Check environment variables
    print(f"\nEnvironment variables:")
    rocm_vars = ['HSA_OVERRIDE_GFX_VERSION', 'ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES']
    cuda_vars = ['CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER']

    for var in rocm_vars + cuda_vars:
        val = os.environ.get(var, 'Not set')
        print(f"  {var}: {val}")

def test_gpu_compute(device_id=0):
    """Test actual computation on a GPU"""
    try:
        device = torch.device(f'cuda:{device_id}')

        # Small matrix multiplication test
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)

        # Warmup
        for _ in range(5):
            z = torch.matmul(x, y)
            torch.cuda.synchronize()

        # Timed test
        import time
        start = time.time()
        for _ in range(100):
            z = torch.matmul(x, y)
            torch.cuda.synchronize()
        elapsed = time.time() - start

        return True, elapsed
    except Exception as e:
        return False, str(e)

def test_all_gpus():
    """Test computation on all detected GPUs"""
    print_section("GPU Compute Tests")

    if not torch.cuda.is_available():
        print("❌ No CUDA/ROCm GPUs available in PyTorch")
        return

    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_name(i)
        print(f"\nTesting cuda:{i} ({device_name})...")

        success, result = test_gpu_compute(i)

        if success:
            print(f"  ✅ SUCCESS - 100 matmuls (1000x1000) in {result:.3f}s")
            gflops = (100 * 2 * 1000**3) / (result * 1e9)
            print(f"     Performance: {gflops:.1f} GFLOPS")
        else:
            print(f"  ❌ FAILED - {result}")

def check_memory_available():
    """Check available memory on each GPU"""
    print_section("GPU Memory Status")

    if not torch.cuda.is_available():
        print("No GPUs available")
        return

    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_name(i)
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3

        # Reset stats and check free memory
        torch.cuda.reset_peak_memory_stats(i)
        torch.cuda.empty_cache()

        # Get allocated memory
        try:
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            free = total - reserved

            print(f"\ncuda:{i} ({device_name})")
            print(f"  Total:     {total:.2f} GB")
            print(f"  Allocated: {allocated:.2f} GB")
            print(f"  Reserved:  {reserved:.2f} GB")
            print(f"  Free:      {free:.2f} GB")

            if free < 1.0:
                print(f"  ⚠️  WARNING: Low free memory!")
            else:
                print(f"  ✅ OK")
        except Exception as e:
            print(f"  ❌ Error checking memory: {e}")

def suggest_training_config():
    """Suggest optimal training configuration"""
    print_section("Training Configuration Suggestions")

    if not torch.cuda.is_available():
        print("❌ No GPUs available - training will be VERY slow on CPU")
        print("\nRecommendation: Install ROCm or CUDA support for PyTorch")
        return

    n_gpus = torch.cuda.device_count()

    print(f"✅ {n_gpus} GPU(s) detected\n")

    if n_gpus == 1:
        print("Single GPU Training:")
        print("  python scripts/test_btc_models.py --mode all --device cuda:0")

    elif n_gpus == 4:
        print("🚀 MULTI-GPU PARALLEL TRAINING (Recommended):")
        print("\nOption 1: Train different symbols in parallel (fastest for multiple symbols)")
        print("  # Terminal 1 - BTC on GPU 0")
        print("  CUDA_VISIBLE_DEVICES=0 python scripts/train_production_ready.py --symbol BTCUSDT --device cuda:0 &")
        print("\n  # Terminal 2 - ETH on GPU 1")
        print("  CUDA_VISIBLE_DEVICES=1 python scripts/train_production_ready.py --symbol ETHUSDT --device cuda:0 &")
        print("\n  # Terminal 3 - SOL on GPU 2")
        print("  CUDA_VISIBLE_DEVICES=2 python scripts/train_production_ready.py --symbol SOLUSDT --device cuda:0 &")
        print("\n  # Terminal 4 - BNB on GPU 3")
        print("  CUDA_VISIBLE_DEVICES=3 python scripts/train_production_ready.py --symbol BNBUSDT --device cuda:0 &")

        print("\n\nOption 2: Train ensemble models in parallel (fastest for single symbol)")
        print("  python scripts/train_ensemble.py --symbol BTCUSDT --n-models 4 --parallel-gpus 0,1,2,3")

        print("\n\nOption 3: Train different timeframes in parallel")
        print("  # GPU 0: BTC 5m")
        print("  CUDA_VISIBLE_DEVICES=0 python scripts/train_production_ready.py --symbol BTCUSDT --timeframe 5m &")
        print("\n  # GPU 1: BTC 15m")
        print("  CUDA_VISIBLE_DEVICES=1 python scripts/train_production_ready.py --symbol BTCUSDT --timeframe 15m &")
        print("\n  # GPU 2: ETH 5m")
        print("  CUDA_VISIBLE_DEVICES=2 python scripts/train_production_ready.py --symbol ETHUSDT --timeframe 5m &")
        print("\n  # GPU 3: ETH 15m")
        print("  CUDA_VISIBLE_DEVICES=3 python scripts/train_production_ready.py --symbol ETHUSDT --timeframe 15m &")

    else:
        print(f"Train {n_gpus} models in parallel:")
        for i in range(n_gpus):
            print(f"  CUDA_VISIBLE_DEVICES={i} python scripts/train_production_ready.py --device cuda:0 &")

    print("\n" + "="*80)
    print("Expected speedup: ~{}x (training {} models simultaneously)".format(n_gpus, n_gpus))
    print("="*80)

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                     Multi-GPU Diagnostic Tool                            ║
    ║                  Trading System - GPU Configuration                      ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)

    # System detection
    n_amd, n_nvidia = check_system_gpus()

    # PyTorch configuration
    check_pytorch()

    # Memory status
    check_memory_available()

    # Compute tests
    test_all_gpus()

    # Training suggestions
    suggest_training_config()

    print("\n" + "="*80)
    print("Diagnostic complete!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
