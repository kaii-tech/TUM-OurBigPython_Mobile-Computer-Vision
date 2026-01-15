#!/usr/bin/env python3
"""
VPR Distributed Training Orchestrator
Run this as: jupyter notebook orchestrator.ipynb

This script:
1. Launches training on 3 Linux workstations in parallel
2. Monitors their progress
3. Collects results and visualizes them

Configure WORKSTATIONS dict with your actual SSH hosts/credentials
"""

# ============================================================================
# Cell 1: Imports and Configuration
# ============================================================================

import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, Markdown, clear_output

print("="*70)
print("VPR Distributed Training Orchestrator")
print("="*70)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Configuration
WORKSTATIONS = {
    'ws1': {
        'host': 'user@ws1.uni.local',
        'model': 'resnet50',
        'gpu': 'RTX 3060'
    },
    'ws2': {
        'host': 'user@ws2.uni.local',
        'model': 'efficientnetb0',
        'gpu': 'RTX 3060'
    },
    'ws3': {
        'host': 'user@ws3.uni.local',
        'model': 'vit',
        'gpu': 'RTX 3060'
    }
}

PROJECT_DIR = '~/vpr_training'
REMOTE_DATA_DIR = '~/vpr_training/data'
OUTPUT_DIR = './results'

# Create output directory
Path(OUTPUT_DIR).mkdir(exist_ok=True)

print("Configuration:")
for ws_name, config in WORKSTATIONS.items():
    print(f"  {ws_name}: {config['host']} → {config['model']}")
print()

# ============================================================================
# Cell 2: Launch Training Jobs
# ============================================================================

def launch_training_job(ws_name, config):
    """Launch training on a remote workstation via SSH"""
    
    host = config['host']
    model = config['model']
    
    # SSH command to run training
    cmd = [
        'ssh',
        host,
        f'cd {PROJECT_DIR} && '
        f'source venv/bin/activate && '
        f'python train_base.py {model} {REMOTE_DATA_DIR}'
    ]
    
    print(f"\n{'='*70}")
    print(f"Launching {model.upper()} on {ws_name}")
    print(f"Host: {host}")
    print(f"{'='*70}")
    
    # Start process (non-blocking)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    return proc

# Launch all jobs
print("\n" + "="*70)
print("LAUNCHING TRAINING JOBS")
print("="*70)
print(f"\nLaunching {len(WORKSTATIONS)} independent training jobs...\n")

processes = {}
launch_times = {}

for ws_name, config in WORKSTATIONS.items():
    processes[ws_name] = launch_training_job(ws_name, config)
    launch_times[ws_name] = time.time()
    time.sleep(2)  # Small delay between launches

print("\n✓ All training jobs launched!")

# ============================================================================
# Cell 3: Monitor Progress
# ============================================================================

def monitor_jobs(processes, check_interval=30, timeout=86400):
    """Monitor all running jobs"""
    
    status = {ws: 'running' for ws in processes}
    start_time = time.time()
    
    while any(s == 'running' for s in status.values()):
        elapsed = (time.time() - start_time) / 60
        
        print(f"\n[{elapsed:.1f}m elapsed] Status:")
        
        for ws_name, proc in processes.items():
            returncode = proc.poll()
            
            if returncode is not None:
                status[ws_name] = 'completed' if returncode == 0 else 'failed'
                status_str = "✓ COMPLETED" if returncode == 0 else "✗ FAILED"
                print(f"  {ws_name}: {status_str}")
            else:
                print(f"  {ws_name}: ⏳ training...")
        
        # Check for timeout
        if elapsed > timeout / 60:
            print("\n⚠ Timeout reached, terminating jobs...")
            for proc in processes.values():
                proc.terminate()
            break
        
        if any(s == 'running' for s in status.values()):
            time.sleep(check_interval)
    
    return status

print("\n" + "="*70)
print("MONITORING PROGRESS")
print("="*70)

final_status = monitor_jobs(processes, check_interval=60)

print("\n" + "="*70)
print("TRAINING SUMMARY")
print("="*70)

for ws_name, status in final_status.items():
    model = WORKSTATIONS[ws_name]['model']
    print(f"{ws_name} ({model}): {status}")

print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# Cell 4: Collect Results
# ============================================================================

def fetch_results(ws_name, config):
    """Fetch training results from a workstation"""
    
    host = config['host']
    model = config['model']
    results_dir = f'{PROJECT_DIR}/results'
    
    # Fetch history file
    history_file = f'{model}_history.json'
    local_path = f'{OUTPUT_DIR}/{ws_name}_{history_file}'
    
    cmd = [
        'scp',
        f'{host}:{results_dir}/{history_file}',
        local_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=30)
        
        with open(local_path, 'r') as f:
            history = json.load(f)
        
        return {
            'ws': ws_name,
            'model': model,
            'final_train_acc': history['accuracy'][-1],
            'final_val_acc': history['val_accuracy'][-1],
            'best_val_acc': max(history['val_accuracy']),
            'final_train_loss': history['loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'epochs': len(history['accuracy']),
            'history': history
        }
    except Exception as e:
        print(f"Error fetching results from {ws_name}: {e}")
        return None

# Fetch results
print("\n" + "="*70)
print("COLLECTING RESULTS")
print("="*70 + "\n")

results = []
for ws_name, config in WORKSTATIONS.items():
    if final_status.get(ws_name) == 'completed':
        print(f"Fetching results from {ws_name}...")
        result = fetch_results(ws_name, config)
        if result:
            results.append(result)
            print(f"  ✓ {result['model']}: Best val_acc = {result['best_val_acc']:.4f}\n")

# Create summary dataframe
if results:
    df = pd.DataFrame([
        {k: v for k, v in r.items() if k != 'history'}
        for r in results
    ])
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70 + "\n")
    print(df.to_string(index=False))
    print()

# ============================================================================
# Cell 5: Visualize Results
# ============================================================================

if results:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Validation Accuracy Comparison
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, result in enumerate(results):
        axes[0].plot(
            result['history']['val_accuracy'],
            label=f"{result['ws']} ({result['model']})",
            linewidth=2,
            color=colors[i % len(colors)]
        )
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Validation Accuracy', fontsize=12)
    axes[0].set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Final Metrics Bar Chart
    models = [r['ws'] for r in results]
    val_accs = [r['best_val_acc'] for r in results]
    
    bars = axes[1].bar(models, val_accs, color=colors[:len(models)])
    axes[1].set_ylabel('Best Validation Accuracy', fontsize=12)
    axes[1].set_title('Best Validation Accuracy by Model', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, 1])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/training_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✓ Results visualization saved to {OUTPUT_DIR}/training_results.png")

print("\n" + "="*70)
print("ORCHESTRATION COMPLETE")
print("="*70)
