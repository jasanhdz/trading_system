#!/bin/bash

echo "========================================="
echo "POST-REBOOT AMD GPU VERIFICATION"
echo "========================================="
echo ""

echo "1. Checking kernel module parameters..."
cat /sys/module/amdgpu/parameters/exp_hw_support
echo ""

echo "2. Checking if KFD is loaded..."
lsmod | grep amdgpu
echo ""

echo "3. Checking /dev/kfd..."
ls -l /dev/kfd
echo ""

echo "4. Running rocm-smi..."
rocm-smi
echo ""

echo "5. Testing PyTorch ROCm detection..."
cd /home/jasan/Develop/trading_system
./test_amd_rocm62.sh
echo ""

echo "========================================="
echo "If GPUs are detected above, run:"
echo "cd /home/jasan/Develop/trading_system"
echo "./launch_training.sh"
echo "========================================="
