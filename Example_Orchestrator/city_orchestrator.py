#!/usr/bin/env python3
"""
City Locator distributed orchestrator using MultiWorkerMirroredStrategy.
- Launches workers over SSH (IP-based)
- Sets TF_CONFIG for each worker
- Monitors jobs
- Collects artifacts from chief worker (index 0)
"""

import os
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# Configuration (edit to match your environment)
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
DEFAULT_TF_PORT = 12345  # TensorFlow worker port
DEFAULT_SSH_PORT = 22    # SSH port; override in secrets if non-standard
MAX_WORKERS = int(os.environ.get("ACTIVE_WORKERS", len(WORKER_IPS)))
SECRETS_PATH = Path(__file__).parent / ".orchestrator_secrets.json"  # user/password container

PROJECT_DIR = "~/city_locator"  # Remote project dir containing train_city_distributed.py
REMOTE_DATA_DIR = "~/city_locator/data"  # Directory with class subfolders on each worker
REMOTE_OUTPUT_DIR = "~/city_locator/results"  # Where checkpoints/history are written on each worker
ENV_ACTIVATE = "source ~/.bashrc && conda activate tf-m3"  # Change to your env

MODEL_NAME = "mobilenetv2"
EPOCHS = 6
BATCH_SIZE = 128  # Per worker
IMG_SIZE = 224

LOCAL_RESULTS_DIR = Path("./orchestrator_results")
LOCAL_RESULTS_DIR.mkdir(exist_ok=True)

CHECK_INTERVAL = 30  # seconds
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no"]


def load_secrets(path: Path) -> Dict[str, Any]:
    """Load SSH credentials from a hidden JSON file if present."""
    if not path.exists():
        return {}
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception as exc:  # pragma: no cover
        print(f"Warning: failed to read {path}: {exc}")
        return {}


def build_workers() -> List[Dict[str, Any]]:
    """Construct worker dicts using IP list and optional secrets."""
    secrets = load_secrets(SECRETS_PATH)
    user = secrets.get("user", DEFAULT_SSH_USER)
    password = secrets.get("password")
    tf_port = int(secrets.get("tf_port", secrets.get("port", DEFAULT_TF_PORT)))
    ssh_port = int(secrets.get("ssh_port", DEFAULT_SSH_PORT))
    selected_ips = WORKER_IPS[:MAX_WORKERS]
    workers = []
    for ip in selected_ips:
        workers.append({
            "host": f"{user}@{ip}",
            "tf_port": tf_port,
            "ssh_port": ssh_port,
            "password": password,
        })
    return workers

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_tf_config(workers: List[Dict[str, Any]]) -> List[str]:
    """Return TF_CONFIG strings aligned to workers list."""
    cluster_workers = [f"{w['host'].split('@')[-1]}:{w['port']}" for w in workers]
    tf_configs = []
    for idx, _ in enumerate(workers):
        tfc = {
            "cluster": {"worker": cluster_workers},
            "task": {"type": "worker", "index": idx},
        }
        tf_configs.append(json.dumps(tfc))
    return tf_configs


def launch_worker(worker: Dict[str, Any], tf_config: str, log_path: Path):
    """Launch a single worker over SSH and stream output to log file."""
    remote_cmd = (
        f"cd {PROJECT_DIR} && "
        f"{ENV_ACTIVATE} && "
        f"TF_CONFIG='{tf_config}' PYTHONUNBUFFERED=1 "
        f"python train_city_distributed.py "
        f"--data-dir {REMOTE_DATA_DIR} "
        f"--output-dir {REMOTE_OUTPUT_DIR} "
        f"--epochs {EPOCHS} "
        f"--batch-size {BATCH_SIZE} "
        f"--img-size {IMG_SIZE} "
        f"--model {MODEL_NAME}"
    )

    log_file = log_path.open("w")

    ssh_cmd = ["ssh", *SSH_OPTS, worker["host"], "bash", "-lc", remote_cmd]
    if worker.get("password"):
        ssh_cmd = ["sshpass", "-p", worker["password"]] + ssh_cmd

    cmd = ssh_cmd
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc, log_file


def monitor(processes):
    status = {name: "running" for name in processes}
    start = time.time()
    while any(v == "running" for v in status.values()):
        elapsed = (time.time() - start) / 60
        print(f"\n[{elapsed:.1f}m] Status:")
        for name, proc in processes.items():
            rc = proc.poll()
            if rc is None:
                print(f"  {name}: training...")
            else:
                status[name] = "completed" if rc == 0 else f"failed ({rc})"
                print(f"  {name}: {status[name]}")
        if any(v == "running" for v in status.values()):
            time.sleep(CHECK_INTERVAL)
    return status


def fetch_from_chief(chief_worker: Dict[str, Any]):
    """Fetch artifacts from chief worker (index 0)."""
    files = [
        f"{REMOTE_OUTPUT_DIR}/{MODEL_NAME}_history.json",
        f"{REMOTE_OUTPUT_DIR}/{MODEL_NAME}_final.keras",
        f"{REMOTE_OUTPUT_DIR}/{MODEL_NAME}_float16.tflite",
    ]
    for remote_file in files:
        local_path = LOCAL_RESULTS_DIR / Path(remote_file).name
        scp_cmd = ["scp", *SSH_OPTS, f"{chief_worker['host']}:{remote_file}", str(local_path)]
        if chief_worker.get("password"):
            scp_cmd = ["sshpass", "-p", chief_worker["password"]] + scp_cmd
        try:
            subprocess.run(scp_cmd, check=True, capture_output=True, text=True)
            print(f"Fetched {remote_file} -> {local_path}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to fetch {remote_file}: {e.stderr.strip()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    workers = build_workers()
    print("=" * 70)
    print("City Locator Distributed Orchestrator")
    print("=" * 70)
    print(f"Workers: {[w['host'] for w in workers]}")
    print(f"Model: {MODEL_NAME} | Epochs: {EPOCHS} | Batch/worker: {BATCH_SIZE}")
    print(f"Remote project: {PROJECT_DIR}")
    print(f"Remote data dir: {REMOTE_DATA_DIR}")
    print(f"Remote output dir: {REMOTE_OUTPUT_DIR}")
    print(f"Logs -> {LOCAL_RESULTS_DIR}")
    print("=" * 70)

    tf_configs = build_tf_config(workers)
    processes = {}
    logs = {}

    # Launch all workers
    for idx, worker in enumerate(workers):
        log_path = LOCAL_RESULTS_DIR / f"worker_{idx}.log"
        print(f"Launching worker {idx} on {worker['host']} (log: {log_path})")
        proc, log_file = launch_worker(worker, tf_configs[idx], log_path)
        processes[f"worker_{idx}"] = proc
        logs[f"worker_{idx}"] = log_file
        time.sleep(2)

    print("\nAll workers launched. Monitoring...")
    status = monitor(processes)

    # Close log files
    for lf in logs.values():
        lf.close()

    print("\nFinal status:")
    for name, state in status.items():
        print(f"  {name}: {state}")

    # Fetch artifacts from chief (index 0) if succeeded
    if status.get("worker_0", "failed").startswith("completed"):
        print("\nFetching artifacts from chief worker...")
        fetch_from_chief(workers[0])
    else:
        print("Chief worker did not complete successfully; skipping fetch.")

    print("\nDone at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
