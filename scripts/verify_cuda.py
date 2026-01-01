import torch
import sys

def verify_cuda():
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA is NOT available.")
        return False
        
    device_count = torch.cuda.device_count()
    print(f"✅ CUDA is available! Found {device_count} device(s).")
    
    for i in range(device_count):
        print(f"   Device {i}: {torch.cuda.get_device_name(i)}")
        
    try:
        print("\n🧪 Testing Tensor Allocation on GPU...")
        x = torch.rand(5, 3).cuda()
        print(f"   Tensor created successfully:\n{x}")
        print("✅ Tensor allocation successful.")
        return True
    except Exception as e:
        print(f"❌ Failed to allocate tensor on GPU: {e}")
        return False

if __name__ == "__main__":
    success = verify_cuda()
    sys.exit(0 if success else 1)
