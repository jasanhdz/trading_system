#!/bin/bash
# Script de emergencia para revertir a ROCm 5.7
# Ejecutar con sudo si es necesario (pedirá password)

echo "🚨 INICIANDO ROLLBACK A ROCM 5.7..."

# 1. Desinstalar ROCm 6.2
echo "🗑️  Desinstalando ROCm 6.2..."
echo "hasanazael" | sudo -S amdgpu-install --uninstall -y

# 2. Instalar ROCm 5.7
echo "⬇️  Instalando ROCm 5.7..."
# Nota: Usamos el instalador 5.7 específico
wget https://repo.radeon.com/amdgpu-install/5.7/ubuntu/jammy/amdgpu-install_5.7.50700-1_all.deb
echo "hasanazael" | sudo -S dpkg -i amdgpu-install_5.7.50700-1_all.deb
echo "hasanazael" | sudo -S amdgpu-install --usecase=rocm --no-dkms -y

# 3. Recrear entorno virtual
echo "🐍 Recreando entorno virtual .venv_rocm57..."
rm -rf .venv_rocm57
python3 -m venv .venv_rocm57
.venv_rocm57/bin/pip install --upgrade pip

# 4. Instalar PyTorch para ROCm 5.7
echo "🔥 Instalando PyTorch 2.3.1 (ROCm 5.7)..."
.venv_rocm57/bin/pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/rocm5.7

# 5. Instalar dependencias
echo "📦 Instalando dependencias..."
.venv_rocm57/bin/pip install scikit-learn pandas click ta coloredlogs sqlalchemy psycopg2-binary python-dotenv ta-lib

# 6. Restaurar dispatcher
echo "🔄 Restaurando configuración del dispatcher..."
sed -i 's/.venv_rocm62/.venv_rocm57/g' scripts/dispatch_training.py

echo "✅ ROLLBACK COMPLETADO. Ejecuta ./launch_training.sh para reiniciar."
