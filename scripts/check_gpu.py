
import torch
import sys

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"ROCm Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Device Count: {torch.cuda.device_count()}")
    
    try:
        print("\nAttempting Tensor Operation on GPU...")
        x = torch.randn(1024, 1024, device="cuda")
        y = torch.matmul(x, x)
        print(f"✅ Success! Matmul result shape: {y.shape}")
        
        print("\nAttempting Linear Layer...")
        import torch.nn as nn
        layer = nn.Linear(1024, 512).to("cuda")
        z = layer(x)
        print(f"✅ Success! Linear layer result shape: {z.shape}")
        
    except Exception as e:
        print(f"❌ GPU Operation Failed: {e}")
else:
    print("❌ No GPU detected by PyTorch.")
