#!/bin/bash

# TensorFlow Linux Setup Script for Debian
# This script sets up a conda environment with TensorFlow and GPU support

set -e  # Exit on error

echo "=========================================="
echo "TensorFlow Linux Environment Setup"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Initialize conda
initialize_conda() {
    # Check if conda is already available
    if command -v conda &> /dev/null; then
        return 0
    fi
    
    # Try to find and source conda from common locations
    POSSIBLE_CONDA_PATHS=(
        "$HOME/miniconda3"
        "$HOME/anaconda3"
        "$HOME/miniforge3"
        "$HOME/mambaforge"
        "/opt/conda"
        "/opt/miniconda3"
        "/opt/anaconda3"
        "/usr/local/miniconda3"
        "/usr/local/anaconda3"
    )
    
    for conda_path in "${POSSIBLE_CONDA_PATHS[@]}"; do
        if [ -f "${conda_path}/etc/profile.d/conda.sh" ]; then
            print_status "Found conda at: ${conda_path}"
            source "${conda_path}/etc/profile.d/conda.sh"
            return 0
        fi
    done
    
    # If still not found, install Miniconda
    print_warning "Conda not found. Installing Miniconda..."
    echo ""
    
    MINICONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
    MINICONDA_URL="https://repo.anaconda.com/miniconda/${MINICONDA_INSTALLER}"
    INSTALL_PATH="$HOME/miniconda3"
    
    # Download Miniconda installer
    print_status "Downloading Miniconda installer..."
    if ! wget -q --show-progress "${MINICONDA_URL}" -O "/tmp/${MINICONDA_INSTALLER}"; then
        print_error "Failed to download Miniconda installer!"
        echo "Please install manually from: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    
    # Install Miniconda
    print_status "Installing Miniconda to ${INSTALL_PATH}..."
    if ! bash "/tmp/${MINICONDA_INSTALLER}" -b -p "${INSTALL_PATH}"; then
        print_error "Miniconda installation failed!"
        rm -f "/tmp/${MINICONDA_INSTALLER}"
        exit 1
    fi
    
    # Clean up installer
    rm -f "/tmp/${MINICONDA_INSTALLER}"
    print_status "Miniconda installed successfully!"
    
    # Initialize conda for bash
    print_status "Initializing conda..."
    "${INSTALL_PATH}/bin/conda" init bash
    
    print_status "Miniconda installed successfully!"
    print_status "Restarting script in new bash session with conda available..."
    echo ""
    
    # Restart the script in a new bash session with conda initialized
    exec bash "$0" "$@"
}

# Initialize conda first
print_status "Checking for conda installation..."
initialize_conda

# Accept Conda Terms of Service
echo ""
echo "=========================================="
echo "Accepting Conda Terms of Service"
echo "=========================================="
print_status "Accepting Conda TOS..."
# Accept TOS for default channels
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
conda tos accept --override-channels --channel defaults 2>/dev/null || true
print_status "Conda TOS accepted!"

# 1. Update Conda
echo ""
echo "=========================================="
echo "Step 1: Updating Conda"
echo "=========================================="
print_status "Updating conda to the latest version..."
conda update -n base -c defaults conda -y
print_status "Conda updated successfully!"

# 2. Check CUDA drivers (no sudo - info only)
echo ""
echo "=========================================="
echo "Step 2: Checking CUDA Drivers"
echo "=========================================="

if command -v nvidia-smi &> /dev/null; then
    NVIDIA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1)
    print_status "NVIDIA driver detected: ${NVIDIA_VERSION}"
    nvidia-smi --query-gpu=name --format=csv,noheader | while read -r gpu; do
        print_status "GPU found: ${gpu}"
    done
else
    print_warning "NVIDIA driver not found - TensorFlow will run on CPU only"
    echo ""
    echo "  To install NVIDIA drivers (requires sudo), run:"
    echo "    sudo apt-get update && sudo apt-get install -y nvidia-driver-535 nvidia-cuda-toolkit"
    echo ""
fi

# 3. Create conda environment
echo ""
echo "=========================================="
echo "Step 3: Creating Conda Environment"
echo "=========================================="

ENV_NAME="tf-linux"
PYTHON_VERSION="3.11"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    print_warning "Environment '${ENV_NAME}' already exists."
    read -p "Do you want to remove it and create a new one? (y/n): " remove_env
    if [[ "$remove_env" == "y" || "$remove_env" == "Y" ]]; then
        print_status "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
        print_status "Creating new conda environment: ${ENV_NAME} with Python ${PYTHON_VERSION}..."
        conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
    else
        print_status "Using existing environment."
    fi
else
    print_status "Creating new conda environment: ${ENV_NAME} with Python ${PYTHON_VERSION}..."
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
fi

print_status "Environment '${ENV_NAME}' created successfully!"

# 4. Install requirements
echo ""
echo "=========================================="
echo "Step 4: Installing Requirements"
echo "=========================================="

# Activate environment and install packages
print_status "Activating environment and installing packages..."

# Get the conda base path
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

print_status "Installing TensorFlow with GPU support..."
# Install TensorFlow with GPU support first
pip install tensorflow[and-cuda]>=2.15.0

print_status "Installing additional requirements from requirements-linux.txt..."
if [ -f "requirements-linux.txt" ]; then
    pip install -r requirements-linux.txt
    print_status "Requirements installed successfully!"
else
    print_error "requirements-linux.txt not found!"
    exit 1
fi

# 5. Test and Enable NVIDIA GPU Support
echo ""
echo "=========================================="
echo "Step 5: Testing NVIDIA GPU Support"
echo "=========================================="

print_status "Checking TensorFlow installation..."
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"

print_status "Checking for GPU availability..."
python << EOF
import tensorflow as tf
import sys

print("\n" + "="*50)
print("GPU Detection Test")
print("="*50)

# Check if GPU is available
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f"\n✓ SUCCESS: Found {len(gpus)} GPU(s)!")
    print("\nGPU Details:")
    print("-" * 50)
    for i, gpu in enumerate(gpus):
        print(f"\nGPU {i}: {gpu.name}")
        print(f"  Device Type: {gpu.device_type}")
        
    # Test GPU with a simple operation
    print("\n" + "="*50)
    print("Running GPU Test Operation...")
    print("="*50)
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
            print("\n✓ GPU computation successful!")
            print(f"  Test result:\n{c.numpy()}")
    except Exception as e:
        print(f"\n✗ GPU computation failed: {e}")
        sys.exit(1)
    
    # Memory growth settings
    print("\n" + "="*50)
    print("Configuring GPU Memory Growth...")
    print("="*50)
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✓ GPU memory growth enabled!")
        print("  (Prevents TensorFlow from allocating all GPU memory at once)")
    except Exception as e:
        print(f"⚠ Warning: Could not set memory growth: {e}")
    
else:
    print("\n✗ WARNING: No GPU detected!")
    print("\nTensorFlow will run on CPU only.")
    print("\nPossible solutions:")
    print("  1. Ensure NVIDIA drivers are installed: nvidia-smi")
    print("  2. Verify CUDA toolkit is installed")
    print("  3. Reboot the system if drivers were just installed")
    print("  4. Install TensorFlow GPU: pip install tensorflow[and-cuda]")

print("\n" + "="*50)
print("CUDA and cuDNN Information")
print("="*50)

cuda_available = tf.test.is_built_with_cuda()
print(f"Built with CUDA: {cuda_available}")

if cuda_available:
    print(f"CUDA Version: {tf.sysconfig.get_build_info().get('cuda_version', 'Unknown')}")
    print(f"cuDNN Version: {tf.sysconfig.get_build_info().get('cudnn_version', 'Unknown')}")

print("\n" + "="*50)
EOF

print_status "GPU test complete!"

# List available GPUs using nvidia-smi if available
echo ""
echo "=========================================="
echo "NVIDIA GPU System Information"
echo "=========================================="
if command -v nvidia-smi &> /dev/null; then
    print_status "Detailed GPU information from nvidia-smi:"
    echo ""
    nvidia-smi
else
    print_warning "nvidia-smi not available. Install NVIDIA drivers to see GPU information."
fi

# Final summary
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
print_status "Conda environment '${ENV_NAME}' is ready!"
echo ""
echo "Activating the '${ENV_NAME}' environment..."
echo ""
print_status "Environment activated!"
echo ""
echo "To deactivate later, run:"
echo "  conda deactivate"
echo ""
echo "To test TensorFlow with GPU:"
echo "  python -c 'import tensorflow as tf; print(tf.config.list_physical_devices(\"GPU\"))'"
echo ""
print_status "All done! Happy coding! 🚀"
echo ""

# Activate the environment and start a new shell
exec bash --init-file <(echo "source ~/.bashrc; conda activate ${ENV_NAME}; echo ''; echo 'Conda environment ${ENV_NAME} is now active!'; echo ''")
