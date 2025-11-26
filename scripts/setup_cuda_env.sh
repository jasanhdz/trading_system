#!/bin/bash

# setup_cuda_env.sh
# Crea un entorno virtual separado para usar la GPU NVIDIA con CUDA

ENV_NAME=".venv_cuda"

echo "🚀 Configurando entorno para NVIDIA (CUDA)..."

if [ -d "$ENV_NAME" ]; then
    echo "⚠️  El entorno $ENV_NAME ya existe."
    read -p "¿Deseas recrearlo? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelando."
        exit 1
    fi
    rm -rf "$ENV_NAME"
fi

echo "📦 Creando entorno virtual..."
python3 -m venv "$ENV_NAME"

# Activar entorno
source "$ENV_NAME/bin/activate"

echo "⬇️  Instalando PyTorch con soporte CUDA 12.1..."
# Instalar PyTorch CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "⬇️  Instalando otras dependencias..."
# Instalar resto de dependencias (excluyendo torch que ya instalamos)
# Usamos grep -v para excluir torch del requirements.txt si estuviera ahí explícitamente sin versión
# Pero como requirements.txt tiene torch>=2.0.0, pip podría intentar reinstalarlo si no tenemos cuidado.
# La mejor forma es instalar requirements.txt y dejar que pip resuelva, pero asegurándonos que torch sea el de CUDA.
# Al haber instalado torch con --index-url cu121 antes, pip debería respetar esa versión si cumple el requisito >=2.0.0

pip install -r requirements.txt

echo "✅ Entorno configurado correctamente."
echo "Para usar la GPU NVIDIA:"
echo "source $ENV_NAME/bin/activate"
echo "python scripts/train_production_ready.py ..."
