# VPR Distributed Training Setup Guide

## Architecture

```
MacOS Laptop (Orchestrator)
  └─ Jupyter Notebook
      ├─ SSH → Linux WS1 (ResNet50)
      ├─ SSH → Linux WS2 (EfficientNet)
      └─ SSH → Linux WS3 (ViT)
```

Each workstation runs independently, no distributed TensorFlow complexity.

---

## PART 1: Setup on Linux Workstations (do once per machine)

### 1.1 SSH Access Setup

Ensure you can SSH into each workstation without password:

```bash
# On your Mac, generate SSH key if you don't have one
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519

# Copy public key to each workstation
ssh-copy-id -i ~/.ssh/id_ed25519 user@ws1.uni.local
ssh-copy-id -i ~/.ssh/id_ed25519 user@ws2.uni.local
ssh-copy-id -i ~/.ssh/id_ed25519 user@ws3.uni.local

# Test
ssh user@ws1.uni.local "echo 'SSH works!'"
```

### 1.2 Python Environment Setup

On **each workstation**, run:

```bash
# Create project directory
mkdir -p ~/vpr_training
cd ~/vpr_training

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install tensorflow==2.15.0
pip install numpy pillow scipy matplotlib pandas scikit-learn
pip install tensorboard
```

### 1.3 Prepare Dataset Structure

On **each workstation**:

```bash
# Create data directories
mkdir -p ~/vpr_training/data/train
mkdir -p ~/vpr_training/data/val

# Copy your dataset here
# Expected structure:
# data/train/
#   class_0/
#     img1.jpg
#     img2.jpg
#   class_1/
#     ...
# data/val/
#   class_0/
#   class_1/
```

---

## PART 2: Training Scripts on Linux Workstations

### 2.1 Base Training Script

Create `~/vpr_training/train_base.py` on **each workstation**:

```python
import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
from pathlib import Path

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

def load_data(data_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """Load images from directory structure: data/class_name/*.jpg"""
    
    train_ds = keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, 'train'),
        seed=42,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode='categorical',
    )
    
    val_ds = keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, 'val'),
        seed=42,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode='categorical',
    )
    
    return train_ds, val_ds

def create_model(num_classes, base_model_name='resnet50'):
    """Create transfer learning model"""
    
    if base_model_name == 'resnet50':
        base = keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
    elif base_model_name == 'efficientnetb0':
        base = keras.applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
    elif base_model_name == 'vit':
        # Using a smaller ViT variant
        base = keras.applications.DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
    else:
        raise ValueError(f"Unknown model: {base_model_name}")
    
    # Freeze base model
    base.trainable = False
    
    # Add custom top layers
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = keras.applications.resnet50.preprocess_input(inputs) if base_model_name == 'resnet50' else \
        keras.applications.efficientnet.preprocess_input(inputs) if base_model_name == 'efficientnetb0' else \
        keras.applications.densenet.preprocess_input(inputs)
    
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    return model, base

def train_model(model, train_ds, val_ds, model_name, output_dir='./results'):
    """Train the model with checkpointing"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    checkpoint_path = os.path.join(output_dir, f'{model_name}_best.keras')
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(output_dir, f'{model_name}_logs'),
            histogram_freq=1
        )
    ]
    
    # Train
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}\n")
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_path = os.path.join(output_dir, f'{model_name}_final.keras')
    model.save(final_path)
    
    # Save history
    history_path = os.path.join(output_dir, f'{model_name}_history.json')
    history_dict = {
        'accuracy': [float(x) for x in history.history['accuracy']],
        'loss': [float(x) for x in history.history['loss']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'val_loss': [float(x) for x in history.history['val_loss']]
    }
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    return history

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python train_base.py <model_name> [data_dir]")
        print("  model_name: resnet50, efficientnetb0, vit")
        sys.exit(1)
    
    model_name = sys.argv[1]
    data_dir = sys.argv[2] if len(sys.argv) > 2 else './data'
    
    # Load data
    print(f"Loading data from {data_dir}...")
    train_ds, val_ds = load_data(data_dir)
    num_classes = len(train_ds.class_names)
    
    # Create model
    print(f"Creating {model_name} model...")
    model, base = create_model(num_classes, model_name)
    model.summary()
    
    # Train
    train_model(model, train_ds, val_ds, model_name)
```

---

## PART 3: Jupyter Notebook on MacOS (Orchestrator)

Create `~/vpr_training/orchestrator.ipynb` on **your Mac**:

```python
# ============================================================================
# VPR Distributed Training Orchestrator
# Run this on your MacOS laptop to launch training on all workstations
# ============================================================================

import subprocess
import json
import time
from pathlib import Path
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

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

# ============================================================================
# Cell 1: Launch all training jobs
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
    print(f"Launching {model.upper()} on {ws_name} ({host})")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}")
    
    # Start process (non-blocking)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    return proc

# Launch all jobs
print("\n" + "="*70)
print("VPR Distributed Training - Parallel Experiments")
print("="*70)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Launching {len(WORKSTATIONS)} independent training jobs...\n")

processes = {}
for ws_name, config in WORKSTATIONS.items():
    processes[ws_name] = launch_training_job(ws_name, config)
    time.sleep(2)  # Small delay between launches

print("\n✓ All training jobs launched!")
print("\nMonitoring processes...\n")

# ============================================================================
# Cell 2: Monitor progress
# ============================================================================

def monitor_jobs(processes, check_interval=30):
    """Monitor all running jobs"""
    
    status = {ws: 'running' for ws in processes}
    start_time = time.time()
    
    while any(s == 'running' for s in status.values()):
        elapsed = (time.time() - start_time) / 60
        
        print(f"\n[{elapsed:.1f}m elapsed]")
        
        for ws_name, proc in processes.items():
            returncode = proc.poll()
            
            if returncode is not None:
                status[ws_name] = 'completed' if returncode == 0 else 'failed'
                status_str = "✓ COMPLETED" if returncode == 0 else "✗ FAILED"
                print(f"  {ws_name}: {status_str}")
            else:
                print(f"  {ws_name}: ⏳ training...")
        
        if any(s == 'running' for s in status.values()):
            time.sleep(check_interval)
    
    return status

# Monitor the jobs
final_status = monitor_jobs(processes)

print("\n" + "="*70)
print("Training Summary")
print("="*70)
for ws_name, status in final_status.items():
    print(f"{ws_name}: {status}")

print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# Cell 3: Collect results from all workstations
# ============================================================================

import pandas as pd
import matplotlib.pyplot as plt

def fetch_results(ws_name, config):
    """Fetch training results from a workstation"""
    
    host = config['host']
    model = config['model']
    results_dir = f'{PROJECT_DIR}/results'
    
    # Fetch history file
    history_file = f'{model}_history.json'
    local_path = f'./results/{ws_name}_{history_file}'
    
    cmd = [
        'scp',
        f'{host}:{results_dir}/{history_file}',
        local_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        
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
print("Collecting Results")
print("="*70)

results = []
for ws_name, config in WORKSTATIONS.items():
    print(f"Fetching results from {ws_name}...")
    result = fetch_results(ws_name, config)
    if result:
        results.append(result)

# Create summary dataframe
if results:
    df = pd.DataFrame([
        {k: v for k, v in r.items() if k != 'history'}
        for r in results
    ])
    
    print("\nResults Summary:")
    print(df.to_string(index=False))
    
    # ========================================================================
    # Cell 4: Visualize Results
    # ========================================================================
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Validation Accuracy Comparison
    for result in results:
        axes[0].plot(
            result['history']['val_accuracy'],
            label=f"{result['ws']} ({result['model']})",
            linewidth=2
        )
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Validation Accuracy')
    axes[0].set_title('Validation Accuracy Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Final Metrics Bar Chart
    models = [r['ws'] for r in results]
    val_accs = [r['best_val_acc'] for r in results]
    
    bars = axes[1].bar(models, val_accs, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[1].set_ylabel('Best Validation Accuracy')
    axes[1].set_title('Best Validation Accuracy by Model')
    axes[1].set_ylim([0, 1])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150)
    plt.show()
    
    print("\n✓ Results visualization saved to training_results.png")
```

---

## PART 4: Files Structure

Ensure you have this structure **on each Linux workstation**:

```
~/vpr_training/
├── venv/                          # Virtual environment
├── train_base.py                  # Main training script
├── data/
│   ├── train/
│   │   ├── class_0/
│   │   │   ├── img1.jpg
│   │   │   └── img2.jpg
│   │   ├── class_1/
│   │   └── ...
│   └── val/
│       ├── class_0/
│       └── ...
└── results/                       # Created automatically
    ├── resnet50_best.keras
    ├── resnet50_history.json
    └── ...
```

On **your Mac**:

```
~/vpr_training/
├── orchestrator.ipynb             # Main Jupyter notebook
├── results/                       # Fetched results
│   ├── ws1_resnet50_history.json
│   ├── ws2_efficientnetb0_history.json
│   └── ws3_vit_history.json
└── training_results.png           # Visualization output
```

---

## PART 5: Running the Training

### Step 1: Test SSH access

```bash
ssh user@ws1.uni.local "echo 'Ready'"
ssh user@ws2.uni.local "echo 'Ready'"
ssh user@ws3.uni.local "echo 'Ready'"
```

### Step 2: Verify data is on workstations

```bash
ssh user@ws1.uni.local "ls ~/vpr_training/data/train/"
```

### Step 3: Run Jupyter notebook on Mac

```bash
cd ~/vpr_training
jupyter notebook orchestrator.ipynb
```

### Step 4: Execute cells in order

- Cell 1: Launch training jobs
- Cell 2: Monitor progress
- Cell 3 & 4: Collect and visualize results

---

## Troubleshooting

### Issue: SSH permission denied

```bash
# Add your public key to each workstation
ssh-copy-id -i ~/.ssh/id_ed25519 user@ws1.uni.local
```

### Issue: TensorFlow not found on workstation

```bash
ssh user@ws1.uni.local "source ~/vpr_training/venv/bin/activate && python -c 'import tensorflow; print(tensorflow.__version__)'"
```

### Issue: Data not found on workstation

```bash
ssh user@ws1.uni.local "ls -lah ~/vpr_training/data/train/"
```

### Issue: GPU not detected

```bash
ssh user@ws1.uni.local "source ~/vpr_training/venv/bin/activate && python -c 'import tensorflow as tf; print(tf.config.list_physical_devices(\"GPU\"))'"
```

---

## What Runs Where

| Component | Location | Purpose |
|-----------|----------|---------|
| `orchestrator.ipynb` | **MacOS** | Launch jobs, monitor, visualize |
| `train_base.py` | **Each Linux WS** | Train models independently |
| Dataset | **Each Linux WS** | Local data for training |
| Results | **Each Linux WS** | Training logs/checkpoints |
| SSH commands | **MacOS** | Remote job control |
