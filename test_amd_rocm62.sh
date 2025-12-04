#!/bin/bash

echo "Testing AMD GPU with ROCm 6.2..."

export HSA_OVERRIDE_GFX_VERSION=10.3.0
export LD_LIBRARY_PATH=/opt/rocm-6.2.0/lib:/opt/rocm-6.2.0/lib64:$LD_LIBRARY_PATH
export HIP_VISIBLE_DEVICES=0
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512
export MIOPEN_DISABLE_CACHE=0

.venv_rocm62/bin/python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('Device name:', torch.cuda.get_device_name(0))
    print('Testing tensor allocation...')
    x = torch.randn(100, 100).cuda()
    y = torch.randn(100, 100).cuda()
    z = x @ y
    print('✅ Matrix multiplication successful!')
    print('Result shape:', z.shape)
else:
    print('❌ No CUDA devices available')
"
