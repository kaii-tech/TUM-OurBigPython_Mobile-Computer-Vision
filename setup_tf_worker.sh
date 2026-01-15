#!/bin/bash
# TensorFlow worker setup: clone repo (Kaii branch), enter it, create env, install TF + requirements.
# Assumes Debian/Ubuntu-like worker with conda available or installable in $HOME.

set -euo pipefail

REPO_URL="https://github.com/kaii-tech/TUM-OurBigPython_Mobile-Computer-Vision.git"
REPO_DIR="$HOME/TUM-OurBigPython_Mobile-Computer-Vision"
BRANCH="Kaii"
ENV_NAME="tf-linux"
PYTHON_VERSION="3.11"
REQ_FILE="requirements-linux.txt"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log() { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
err() { echo -e "${RED}[✗]${NC} $1"; }

# ---------------------------------------------------------------------------
# Conda bootstrap
# ---------------------------------------------------------------------------
init_conda() {
  if command -v conda >/dev/null 2>&1; then return 0; fi
  for base in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "$HOME/mambaforge" \
             "/opt/conda" "/opt/miniconda3" "/opt/anaconda3"; do
    if [ -f "$base/etc/profile.d/conda.sh" ]; then
      source "$base/etc/profile.d/conda.sh"; return 0; fi
  done
  warn "Conda not found. Installing Miniconda..."
  INST=/tmp/Miniconda3-latest-Linux-x86_64.sh
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$INST"
  bash "$INST" -b -p "$HOME/miniconda3"
  rm -f "$INST"
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
}

# ---------------------------------------------------------------------------
# Repo sync
# ---------------------------------------------------------------------------
sync_repo() {
  if [ -d "$REPO_DIR/.git" ]; then
    cd "$REPO_DIR"
    git fetch --all --prune
    git checkout "$BRANCH"
    git reset --hard origin/"$BRANCH"
  else
    git clone --branch "$BRANCH" --single-branch "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
  fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
log "Starting worker setup"
init_conda
log "Conda ready"

sync_repo
log "Repo synced at $(pwd) (branch $BRANCH)"

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

log "Updating conda"
conda update -n base -c defaults conda -y

if command -v nvidia-smi >/dev/null 2>&1; then
  log "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
else
  warn "nvidia-smi not found; GPU may be unavailable"
fi

log "Ensuring env $ENV_NAME (Python $PYTHON_VERSION)"
if conda env list | grep -q "^${ENV_NAME} "; then
  warn "Env $ENV_NAME exists; reusing"
else
  conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

log "Installing TensorFlow (GPU-capable wheels)"
pip install --upgrade pip
pip install "tensorflow[and-cuda]>=2.15.0"

if [ -f "$REQ_FILE" ]; then
  log "Installing requirements from $REQ_FILE"
  pip install -r "$REQ_FILE"
else
  warn "$REQ_FILE not found; skipping extras"
fi

log "Verifying TensorFlow import"
python - <<'PY'
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("GPUs:", gpus)
PY

log "Setup complete. To use: conda activate $ENV_NAME"
