#!/bin/bash
# Script para verificar que las GPUs AMD estén disponibles

echo "🔍 Verificando módulo amdgpu..."
if lsmod | grep -q amdgpu; then
    echo "✅ Módulo amdgpu cargado"
else
    echo "❌ Módulo amdgpu NO cargado"
    echo "Intentando cargar módulo..."
    sudo modprobe amdgpu
    if [ $? -eq 0 ]; then
        echo "✅ Módulo cargado exitosamente"
    else
        echo "❌ Error al cargar módulo"
        exit 1
    fi
fi

echo ""
echo "🔍 Verificando GPUs con rocm-smi..."
rocm-smi --showid

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ GPUs AMD detectadas correctamente"
else
    echo ""
    echo "❌ Error al detectar GPUs"
    exit 1
fi

echo ""
echo "🔍 Verificando PyTorch..."
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export LD_LIBRARY_PATH=/opt/rocm-6.2.0/lib:/opt/rocm-6.2.0/lib64:$LD_LIBRARY_PATH

.venv_rocm62/bin/python -c "
import torch
print('PyTorch Version:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())
print('Device Count:', torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ PyTorch detecta las GPUs correctamente"
else
    echo ""
    echo "❌ PyTorch no puede detectar las GPUs"
    exit 1
fi

echo ""
echo "✅ Todo configurado correctamente"
