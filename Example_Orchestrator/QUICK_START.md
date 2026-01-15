# VPR Distributed Training - Quick Start Guide

## What Runs Where

```
YOUR MACOS LAPTOP (Orchestrator)
├─ orchestrator.py/ipynb
│  └─ SSH launches training on workstations
│  └─ Monitors progress
│  └─ Collects results & visualizes
│
LINUX WORKSTATIONS (Workers)
├─ WS1: train_base.py → ResNet50 training
├─ WS2: train_base.py → EfficientNet training  
└─ WS3: train_base.py → ViT training
```

---

## Setup Checklist (15 minutes)

### On EACH Linux Workstation (WS1, WS2, WS3)

- [ ] **1. SSH Setup** (on your Mac first)
  ```bash
  ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519
  ssh-copy-id -i ~/.ssh/id_ed25519 user@ws1.uni.local
  ssh-copy-id -i ~/.ssh/id_ed25519 user@ws2.uni.local
  ssh-copy-id -i ~/.ssh/id_ed25519 user@ws3.uni.local
  ```

- [ ] **2. Create project structure** (SSH into each workstation)
  ```bash
  mkdir -p ~/vpr_training/data/{train,val}
  cd ~/vpr_training
  python3 -m venv venv
  source venv/bin/activate
  pip install tensorflow==2.15.0 numpy pillow matplotlib pandas scikit-learn
  ```

- [ ] **3. Copy training script**
  ```bash
  # Copy train_base.py to ~/vpr_training/ on each workstation
  scp train_base.py user@ws1.uni.local:~/vpr_training/
  scp train_base.py user@ws2.uni.local:~/vpr_training/
  scp train_base.py user@ws3.uni.local:~/vpr_training/
  ```

- [ ] **4. Upload your dataset**
  ```bash
  # Upload to each workstation
  # Should have structure:
  # data/train/class_0/*.jpg, data/train/class_1/*.jpg, ...
  # data/val/class_0/*.jpg, data/val/class_1/*.jpg, ...
  
  scp -r local_path_to_data/train user@ws1.uni.local:~/vpr_training/data/
  scp -r local_path_to_data/val user@ws1.uni.local:~/vpr_training/data/
  # ... repeat for ws2, ws3
  ```

### On YOUR MacOS Laptop

- [ ] **1. Install Jupyter** 
  ```bash
  pip install jupyter pandas matplotlib
  ```

- [ ] **2. Copy orchestrator script**
  ```bash
  mkdir -p ~/vpr_training
  cp orchestrator.py ~/vpr_training/
  ```

- [ ] **3. Edit WORKSTATIONS config in orchestrator.py**
  ```python
  WORKSTATIONS = {
      'ws1': {'host': 'your_username@ws1.uni.local', 'model': 'resnet50', ...},
      'ws2': {'host': 'your_username@ws2.uni.local', 'model': 'efficientnetb0', ...},
      'ws3': {'host': 'your_username@ws3.uni.local', 'model': 'vit', ...}
  }
  ```

---

## Running Training

### Option A: Jupyter Notebook (Recommended)

```bash
cd ~/vpr_training

# Convert .py to .ipynb if needed
jupyter notebook orchestrator.ipynb

# Or create new notebook and copy cells from orchestrator.py
```

### Option B: Plain Python Script

```bash
cd ~/vpr_training
python orchestrator.py
```

---

## During Training

The script will:

1. **Launch** all 3 training jobs simultaneously via SSH (takes ~30 seconds)
2. **Monitor** progress every 60 seconds (shows which are done/running)
3. **Auto-collect** results when jobs complete
4. **Visualize** results in a comparison plot

Expected training time per model:
- **ResNet50**: 2-4 hours
- **EfficientNet**: 1.5-3 hours  
- **ViT (DenseNet121)**: 2-4 hours

Total time: ~3-4 hours (because they run in parallel!)

---

## Troubleshooting

### SSH Connection Fails
```bash
# Test SSH
ssh user@ws1.uni.local "pwd"

# If it fails, check:
# 1. Username is correct
# 2. Workstation hostname/IP is correct
# 3. SSH key was copied: ssh-copy-id ...
```

### TensorFlow Import Error on Workstation
```bash
# SSH into workstation
ssh user@ws1.uni.local

# Check TensorFlow
source ~/vpr_training/venv/bin/activate
python -c "import tensorflow; print(tensorflow.__version__)"

# If fails, reinstall
pip install --upgrade tensorflow
```

### GPU Not Detected
```bash
# SSH into workstation
source ~/vpr_training/venv/bin/activate
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# If empty list, check NVIDIA drivers
nvidia-smi
```

### Data Not Found
```bash
# SSH into workstation
ls -la ~/vpr_training/data/train/
ls -la ~/vpr_training/data/val/

# Should see subdirectories for each class
```

---

## File Structure (After Setup)

**On each Linux Workstation:**
```
~/vpr_training/
├── venv/
├── train_base.py
├── data/
│   ├── train/
│   │   ├── class_0/
│   │   ├── class_1/
│   │   └── ...
│   └── val/
│       ├── class_0/
│       ├── class_1/
│       └── ...
└── results/          (created during training)
    ├── resnet50_best.keras
    ├── resnet50_history.json
    ├── efficientnetb0_best.keras
    ├── efficientnetb0_history.json
    └── ...
```

**On your MacOS:**
```
~/vpr_training/
├── orchestrator.py
├── orchestrator.ipynb (if using Jupyter)
└── results/          (created after training)
    ├── ws1_resnet50_history.json
    ├── ws2_efficientnetb0_history.json
    ├── ws3_vit_history.json
    └── training_results.png
```

---

## Next Steps After Training

1. **Compare results** - Look at `training_results.png` to see which model performed best
2. **Fine-tune best model** - Unfreeze more layers and continue training
3. **Deploy** - Use the best model for inference

```python
# Load best model and use for inference
import tensorflow as tf
model = tf.keras.models.load_model('~/vpr_training/results/resnet50_best.keras')

# Make predictions
predictions = model.predict(your_image)
```

---

## Support for Your Specific Setup

You mentioned:
- **OS**: MacOS (orchestrator) + Linux workstations (workers)
- **GPUs**: RTX 3060 on each workstation (12GB VRAM)
- **Network**: Gigabit Ethernet, IP-only
- **Dataset**: 27 classes, ~6000 images total
- **Status**: Changed datasets, ready to go!

This setup is **perfect** for your requirements. Each model trains independently on its own GPU with no communication overhead.

---

## Still Have Questions?

- Check the full guide in `vpr_setup_guide.md`
- SSH into workstation and manually run: `python train_base.py resnet50 ~/vpr_training/data`
- Check GPU usage: `nvidia-smi` (on Linux workstation)
- Monitor training: `tensorboard --logdir ~/vpr_training/results/`
