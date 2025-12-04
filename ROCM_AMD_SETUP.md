# ROCm 6.2 + AMD RX 6600 (gfx1032) – Guía de reinstalación

Pasos mínimos para dejar el servidor listo si se borran los entornos o tras un formateo.

## 1) Prerrequisitos
- Usuario en grupos `video` y `render` (requiere re-login): `sudo usermod -aG video,render $USER`
- Kernel ya carga `amdgpu`; valida con `lsmod | grep amdgpu`.
- Repos habilitados: Ubuntu 24.04 (noble) y ROCm 6.2.

## 2) Repositorios APT
```bash
echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/6.2 noble main" | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
```

## 3) Instalar stack ROCm 6.2 (user-space)
Paquetes instalados:
```bash
sudo apt install -y --no-install-recommends \
  hip-runtime-amd rocm-hip-runtime rocm-hip-libraries \
  miopen-hip rocminfo rocm-smi \
  rccl rocfft rocblas rocsparse rocsolver rocrand hipblas hipfft hipsparse hipsolver hipsparselt hiptensor \
  rocm-smi-lib rocm-language-runtime roctracer rocalution
```
- `/opt/rocm` apunta a `/opt/rocm-6.2.0`.
- Si `dpkg` queda a medias: `sudo dpkg --configure -a && sudo apt -f install`.

## 4) Variables de entorno base (Navi23/gfx1032)
Usar siempre antes de probar o entrenar (ya están en scripts):
```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export LD_LIBRARY_PATH=/opt/rocm-6.2.0/lib:/opt/rocm-6.2.0/lib64:$LD_LIBRARY_PATH
# Opcionales de estabilidad en AMD:
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512
export HSA_ENABLE_SDMA=0
```

## 5) Entorno Python ROCm
Recrear venv y dependencias:
```bash
rm -rf .venv_rocm62
python3 -m venv .venv_rocm62

# PyTorch ROCm 6.2
.venv_rocm62/bin/pip install --upgrade pip
.venv_rocm62/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Dependencias del proyecto
.venv_rocm62/bin/pip install scikit-learn pandas click ta coloredlogs sqlalchemy psycopg2-binary python-dotenv ta-lib
```

## 6) Validación rápida
```bash
# Hardware
rocm-smi --showid --showproductname    # espera 2 GPUs RX 6600
rocminfo                              # debe listar nodos gfx1032

# PyTorch
HSA_OVERRIDE_GFX_VERSION=10.3.0 LD_LIBRARY_PATH=/opt/rocm-6.2.0/lib:/opt/rocm-6.2.0/lib64 \
  .venv_rocm62/bin/python - <<'PY'
import torch
print(torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY

# Scripts del repo
bash scripts/verify_amd_gpus.sh
bash test_amd_rocm62.sh
bash test_amd_training.sh
```

## 7) Lanzar entrenamientos
```bash
# Exporta entorno ROCm (si no usas los scripts)
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export LD_LIBRARY_PATH=/opt/rocm-6.2.0/lib:/opt/rocm-6.2.0/lib64:$LD_LIBRARY_PATH

bash launch_training.sh
```
Monitoreo:
- Dispatcher/log principal: `tail -f logs/training_post_reboot_*.log`
- Jobs por GPU: `tail -f logs/multi_gpu/*.log`
- Uso GPUs AMD: `rocm-smi --showid --showuse`
- Uso GPU NVIDIA: `nvidia-smi`

## 8) Notas y troubleshooting
- Si `rocm-smi`/`rocminfo` fallan con permisos en contenedores/sandbox, ejecuta con sudo o ajusta permisos de `/dev/kfd` y `/dev/dri/renderD*`.
- Warning `hipBLASLt on an unsupported architecture` en RX 6600 es normal: PyTorch cae a hipBLAS.
- Si hay OOM en AMD, reduce batch (p.ej. 64/80) o baja `max_split_size_mb`.
- NVIDIA se usa con `.venv_cuda` y `nvidia-smi`; las AMD con `.venv_rocm62`.
