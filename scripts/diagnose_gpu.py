#!/usr/bin/env python3
"""Script de diagnóstico para GPU y PyTorch."""

import sys
import torch

print("\n" + "="*70)
print("DIAGNÓSTICO DE GPU Y PYTORCH")
print("="*70)

# 1. PyTorch Info
print("\n1. PYTORCH:")
print(f"   Version: {torch.__version__}")
print(f"   ROCm version: {torch.version.hip if hasattr(torch.version, 'hip') else 'N/A'}")
print(f"   CUDA version: {torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A'}")

# 2. GPU Detection
print("\n2. GPU DETECTION:")
print(f"   torch.cuda.is_available(): {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"   Device count: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\n   GPU {i}:")
        print(f"      Name: {props.name}")
        print(f"      Memory: {props.total_memory / 1e9:.2f} GB")
        print(f"      Compute Capability: {props.major}.{props.minor}")
else:
    print("   ❌ No se detectaron GPUs")
    print("\n   POSIBLES CAUSAS:")
    print("   - ROCm no instalado")
    print("   - PyTorch instalado sin soporte de GPU")
    print("   - Drivers AMD no instalados")

# 3. Test básico
print("\n3. TEST BÁSICO DE GPU:")
try:
    if torch.cuda.is_available():
        x = torch.randn(100, 100).cuda()
        y = torch.mm(x, x)
        print("   ✓ Operaciones básicas en GPU funcionan")
    else:
        print("   ⚠️  GPU no disponible, no se puede probar")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 4. Test LSTM
print("\n4. TEST LSTM EN GPU:")
try:
    if torch.cuda.is_available():
        import torch.nn as nn
        lstm = nn.LSTM(10, 20, 2).cuda()
        x = torch.randn(5, 3, 10).cuda()
        output, _ = lstm(x)
        print("   ✓ LSTM funciona en GPU")
    else:
        print("   ⚠️  GPU no disponible, no se puede probar")
except Exception as e:
    print(f"   ❌ LSTM falla: {e}")

# 5. Verificar instalación
print("\n5. VERIFICAR INSTALACIÓN:")
print(f"   Python: {sys.version}")
print(f"   torch.cuda.is_available(): {torch.cuda.is_available()}")

is_rocm = bool(getattr(torch.version, "hip", None))
print(f"   ¿Es ROCm?: {is_rocm}")

if is_rocm:
    print("\n   ℹ️  PyTorch con soporte ROCm detectado")
elif torch.version.cuda:
    print("\n   ℹ️  PyTorch con soporte CUDA detectado")
else:
    print("\n   ⚠️  PyTorch SIN soporte de GPU (CPU only)")
    print("\n   SOLUCIÓN:")
    print("   Necesitas reinstalar PyTorch con soporte ROCm:")
    print("\n   # Para ROCm 6.0:")
    print("   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0")
    print("\n   # Para ROCm 5.7:")
    print("   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7")

print("\n" + "="*70)
print("FIN DEL DIAGNÓSTICO")
print("="*70 + "\n")
