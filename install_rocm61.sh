#!/bin/bash
echo "🔄 Instalando ROCm 6.1.3 (Versión estable para Ubuntu 24.04)..."

# 1. Limpiar versiones anteriores
echo "hasanazael" | sudo -S amdgpu-install --uninstall -y

# 2. Descargar e instalar instalador 6.1.3
wget https://repo.radeon.com/amdgpu-install/6.1.3/ubuntu/noble/amdgpu-install_6.1.60103-1_all.deb
echo "hasanazael" | sudo -S dpkg -i amdgpu-install_6.1.60103-1_all.deb

# 3. Instalar ROCm (userspace)
echo "hasanazael" | sudo -S amdgpu-install --usecase=rocm --no-dkms -y

# 4. Configurar entorno Python (usando el venv_rocm62 que ya teníamos o creando uno nuevo)
# Vamos a reusar el nombre .venv_rocm62 para no cambiar scripts, pero reinstalamos torch
rm -rf .venv_rocm62
python3 -m venv .venv_rocm62
.venv_rocm62/bin/pip install --upgrade pip

# Instalar PyTorch compatible con ROCm 6.1
# PyTorch oficial 2.4.1 o 2.5.1 suele ser compatible con 6.1
.venv_rocm62/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1

# Instalar dependencias
.venv_rocm62/bin/pip install scikit-learn pandas click ta coloredlogs sqlalchemy psycopg2-binary python-dotenv ta-lib

echo "✅ ROCm 6.1.3 instalado."
