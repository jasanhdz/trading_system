#!/bin/bash
set -euo pipefail

WORKSPACE_ROOT=/home/jasan/Develop/trading_system
ROOT="$WORKSPACE_ROOT/sandbox/aegis_baseline"
TS_ROOT="$WORKSPACE_ROOT/binance-futures-bot-ts"

exec env -i \
  HOME=/home/jasan \
  USER=jasan \
  PATH=/usr/local/bin:/usr/bin:/bin \
  NODE_ENV=production \
  AEGIS_SHADOW_PYTHON=/home/jasan/.venv_rocm62/bin/python \
  AEGIS_SHADOW_PYTHON_COMMIT=94174efa02e957fff95b85f98979e6e041a54d36 \
  AEGIS_SHADOW_TYPESCRIPT_COMMIT=96f4ad175dcf3c90c2f058211ed4523b804d3853 \
  AEGIS_SHADOW_ACTIVATION_PATH="$WORKSPACE_ROOT/reports/governance/aegis_prospective_validation/activation/shadow_cohort_1_activation.json" \
  AEGIS_SHADOW_MODEL_BUNDLE="$ROOT/config/bundles/aegis-prospective-shadow-candidate-v1.json" \
  AEGIS_SHADOW_DATA_ROOT="$WORKSPACE_ROOT/data/prospective_shadow/cohort_1" \
  node "$TS_ROOT/dist/prospective/shadow-service.js"
