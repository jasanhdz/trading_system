import torch
import torch.nn as nn
import sys
import os

print(f"Environment HSA_OVERRIDE_GFX_VERSION: {os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'Not Set')}", flush=True)
print(f"Testing on device: {torch.cuda.get_device_name(0)}", flush=True)

try:
    # Test simple allocation
    x = torch.randn(100, 100).cuda()
    print("✅ Allocation OK", flush=True)
    
    # Test Matrix Mul
    z = x @ x
    print("✅ Matmul OK", flush=True)
    
    # Test LSTM (que suele fallar si MIOpen no tiene los kernels)
    print("Testing LSTM...", flush=True)
    lstm = nn.LSTM(100, 50, batch_first=True).cuda()
    input = torch.randn(32, 10, 100).cuda()
    output, _ = lstm(input)
    print("✅ LSTM OK", flush=True)
    
except Exception as e:
    print(f"❌ Error: {e}", flush=True)
    sys.exit(1)
