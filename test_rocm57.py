import torch
import os

print(f"PyTorch version: {torch.__version__}")
print(f"ROCm version: {torch.version.hip}")

# Intentar detectar GPUs
if torch.cuda.is_available():
    print(f"✅ CUDA/ROCm disponible! Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        
    # Test tensor
    try:
        x = torch.rand(100).cuda()
        print("✅ Tensor test passed")
    except Exception as e:
        print(f"❌ Tensor test failed: {e}")
else:
    print("❌ CUDA/ROCm NOT available")
    
# Imprimir variables de entorno relevantes
print("\nEnvironment:")
print(f"HSA_OVERRIDE_GFX_VERSION: {os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'Not set')}")
print(f"HIP_VISIBLE_DEVICES: {os.environ.get('HIP_VISIBLE_DEVICES', 'Not set')}")
