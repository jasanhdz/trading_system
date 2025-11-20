#!/bin/bash
# Script wrapper para entrenar con ROCm configurado correctamente

# Configurar variables de entorno para RX 6600 (gfx1032)
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512
export MIOPEN_DISABLE_CACHE=0

echo "=================================================="
echo "ENTRENAMIENTO CON ROCm CONFIGURADO"
echo "=================================================="
echo "GPU: RX 6600 (gfx1032 → gfx1030)"
echo "HSA_OVERRIDE_GFX_VERSION: $HSA_OVERRIDE_GFX_VERSION"
echo "=================================================="
echo ""

# Ejecutar el comando que se pase como argumentos
exec "$@"
