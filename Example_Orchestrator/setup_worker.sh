#!/usr/bin/env bash
# Bootstrap a worker for City Locator distributed training.
# Run on each Linux worker (adjust ENV_NAME/PORT as needed).
set -euo pipefail

ENV_NAME=tf-mw
PYTHON_VER=3.11
PORT=12345
PROJECT_DIR=~/city_locator
DATA_DIR="$PROJECT_DIR/data"

# Ensure conda is on PATH
if ! command -v conda >/dev/null 2>&1; then
  echo "Conda not found. Install Miniconda first." >&2
  exit 1
fi

# Create env if missing
if ! conda env list | grep -q "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python="${PYTHON_VER}"
fi

# Activate env
source ~/.bashrc
conda activate "${ENV_NAME}"

# Install deps
pip install --upgrade pip
pip install tensorflow kagglehub matplotlib pandas pillow

# Project layout
mkdir -p "${PROJECT_DIR}" "${DATA_DIR}" "${PROJECT_DIR}/results"

# Optional: download dataset locally on the worker
python - <<'PY'
import kagglehub
path = kagglehub.dataset_download("amaralibey/gsv-cities")
print("Downloaded gsv-cities to:", path)
PY

cat <<EOF
Setup complete.
Next steps per worker:
  1) Copy train_city_distributed.py to ${PROJECT_DIR}
  2) Ensure data is at ${DATA_DIR} (or update orchestrator REMOTE_DATA_DIR)
  3) Open port ${PORT}/tcp if firewall is enabled (e.g., sudo ufw allow ${PORT}/tcp)
EOF
