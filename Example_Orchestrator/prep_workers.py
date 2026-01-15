#!/usr/bin/env python3
"""
Prepare all workers:
- SSH into each worker
- Clone/pull the repo
- Ensure conda env exists and deps installed
- Download dataset via kagglehub
- Report readiness
"""

import os
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WORKER_IPS: List[str] = [
    "10.157.144.9",
    "10.157.144.12",
    "10.157.144.14",
    "10.157.144.19",
    "10.157.144.24",
    "10.157.144.27",
    "10.157.144.32",
    "10.157.144.34",
]

DEFAULT_SSH_USER = "go56pic"
DEFAULT_SSH_PORT = 22
DEFAULT_TF_PORT = 12345  # for the training run later
MAX_WORKERS = int(os.environ.get("ACTIVE_WORKERS", len(WORKER_IPS)))
SECRETS_PATH = Path(__file__).parent / ".orchestrator_secrets.json"

REPO_URL = os.environ.get(
    "REPO_URL",
    "https://github.com/kaii-tech/TUM-OurBigPython_Mobile-Computer-Vision.git",
)
PROJECT_DIR = "~/city_locator"  # clone target on workers
ENV_NAME = os.environ.get("TF_ENV", "tf-m3")
PYTHON_VER = os.environ.get("PYTHON_VER", "3.11")
ENV_ACTIVATE = f"source ~/.bashrc && conda activate {ENV_NAME}"
DATASET_ID = os.environ.get("DATASET_ID", "amaralibey/gsv-cities")
DATA_LINK = "https://www.kaggle.com/datasets/amaralibey/gsv-cities"
DATA_DIR = "~/city_locator/data"  # symlink target to the downloaded dataset

SSH_OPTS = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_secrets(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r") as f:
        return json.load(f)


def build_workers() -> List[Dict[str, Any]]:
    secrets = load_secrets(SECRETS_PATH)
    user = secrets.get("user", DEFAULT_SSH_USER)
    password = secrets.get("password")
    ssh_port = int(secrets.get("ssh_port", secrets.get("port", DEFAULT_SSH_PORT)))
    tf_port = int(secrets.get("tf_port", DEFAULT_TF_PORT))
    ips = WORKER_IPS[:MAX_WORKERS]
    workers = []
    for ip in ips:
        workers.append({
            "host": f"{user}@{ip}",
            "ssh_port": ssh_port,
            "tf_port": tf_port,
            "password": password,
        })
    return workers


def ssh_command(worker: Dict[str, Any], remote_cmd: str) -> subprocess.CompletedProcess:
    cmd = ["ssh", *SSH_OPTS]
    if worker.get("ssh_port") and worker["ssh_port"] != 22:
        cmd += ["-p", str(worker["ssh_port"])]
    cmd += [worker["host"], "bash", "-lc", remote_cmd]
    if worker.get("password"):
        cmd = ["sshpass", "-p", worker["password"]] + cmd
    return subprocess.run(cmd, text=True, capture_output=True)


# ---------------------------------------------------------------------------
# Main logic per worker
# ---------------------------------------------------------------------------

def prep_worker(worker: Dict[str, Any]) -> None:
    print(f"\n=== Preparing {worker['host']} ===")
    tf_port = worker.get("tf_port", DEFAULT_TF_PORT)
    remote_script = f"""
set -e
export PATH="$HOME/miniconda3/bin:$HOME/anaconda3/bin:$PATH"

# 1) Ensure conda env
if ! conda env list | grep -q "{ENV_NAME}"; then
  conda create -y -n {ENV_NAME} python={PYTHON_VER}
fi
source ~/.bashrc
conda activate {ENV_NAME}

# 2) Clone or update repo
if [ ! -d {PROJECT_DIR} ]; then
  git clone {REPO_URL} {PROJECT_DIR}
else
  cd {PROJECT_DIR}
  git fetch --all --prune
  git reset --hard origin/main || true
  git checkout Kaii || true
  git pull --rebase || true
fi
cd {PROJECT_DIR}

# 3) Install deps
pip install --upgrade pip
pip install tensorflow kagglehub matplotlib pandas pillow

# 4) Download dataset via kagglehub
python - <<'PY'
import kagglehub
path = kagglehub.dataset_download("{DATASET_ID}")
print("Dataset path:", path)
PY

# 5) Symlink dataset to expected location
mkdir -p {DATA_DIR}
LATEST=$(ls -dt $HOME/.cache/kagglehub/datasets/*/*/* 2>/dev/null | head -n1)
if [ -n "$LATEST" ]; then
  rm -rf {DATA_DIR}
  ln -s "$LATEST" {DATA_DIR}
  echo "Linked dataset to {DATA_DIR} -> $LATEST"
fi

# 6) Reminder for training port
echo "TF_PORT={tf_port}"
"""

    result = ssh_command(worker, remote_script)
    if result.returncode != 0:
        print(f"✗ {worker['host']} failed:\n{result.stderr}")
    else:
        print(f"✓ {worker['host']} ready.\n{result.stdout}")


def main():
    workers = build_workers()
    print("Preparing workers:", [w["host"] for w in workers])
    for w in workers:
        prep_worker(w)
    print("\nAll workers processed. You can now run city_orchestrator.py or the notebook.")


if __name__ == "__main__":
    main()
