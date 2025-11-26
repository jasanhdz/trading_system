import os
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"ROCm available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    # Test tensor allocation
    try:
        x = torch.rand(1000, 1000).cuda()
        print("✅ Tensor allocation successful")
        y = torch.matmul(x, x)
        print("✅ Matrix multiplication successful")
    except Exception as e:
        print(f"❌ Error: {e}")
else:
    print("❌ ROCm NOT available")
