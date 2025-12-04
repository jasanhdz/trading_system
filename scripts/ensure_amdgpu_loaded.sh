#!/bin/bash
# Script para asegurar que el módulo amdgpu esté cargado

# Verificar si el módulo está cargado
if ! lsmod | grep -q amdgpu; then
    echo "$(date): Módulo amdgpu no cargado, intentando cargar..."
    sudo modprobe amdgpu
    
    if [ $? -eq 0 ]; then
        echo "$(date): ✅ Módulo amdgpu cargado exitosamente"
    else
        echo "$(date): ❌ Error al cargar módulo amdgpu"
        exit 1
    fi
else
    echo "$(date): ✅ Módulo amdgpu ya está cargado"
fi

# Verificar que rocm-smi funcione
if rocm-smi --showid > /dev/null 2>&1; then
    echo "$(date): ✅ rocm-smi funciona correctamente"
else
    echo "$(date): ❌ rocm-smi no funciona"
    exit 1
fi
