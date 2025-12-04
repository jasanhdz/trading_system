#!/bin/bash
# Test rápido de AMD GPU con entrenamiento

export HSA_OVERRIDE_GFX_VERSION=10.3.0
export LD_LIBRARY_PATH=/opt/rocm-6.2.0/lib:/opt/rocm-6.2.0/lib64:$LD_LIBRARY_PATH
export HIP_VISIBLE_DEVICES=0
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512

echo "Testing AMD GPU 0 with minimal training..."

.venv_rocm62/bin/python -c "
import torch
import torch.nn as nn

print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device: {torch.cuda.get_device_name(0)}')

# Minimal model
model = nn.Linear(10, 2).cuda()
x = torch.randn(32, 10).cuda()
y = model(x)
print(f'✅ Forward pass successful: {y.shape}')

# Backward
loss = y.sum()
loss.backward()
print(f'✅ Backward pass successful')
print('AMD GPU is ready for training!')
"
