import torch
import os
import sys

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"ROCm Version (in torch): {getattr(torch.version, 'hip', 'None')}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device Count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    try:
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        
        print("Attempting tensor operation on GPU...")
        x = torch.tensor([1.0, 2.0, 3.0]).cuda()
        y = x * 2
        print(f"Success! Result: {y}")
    except Exception as e:
        print(f"Tensor operation failed: {e}")
else:
    print("CUDA not available. Checking environment variables...")
    print(f"HSA_OVERRIDE_GFX_VERSION: {os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'Not Set')}")
    print(f"HIP_VISIBLE_DEVICES: {os.environ.get('HIP_VISIBLE_DEVICES', 'Not Set')}")
