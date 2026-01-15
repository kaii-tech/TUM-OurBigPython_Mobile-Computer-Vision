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

# 1. Update Conda
echo ""
echo "=========================================="
echo "Step 1: Updating Conda"
echo "=========================================="
print_status "Updating conda to the latest version..."
conda update -n base -c defaults conda -y
print_status "Conda updated successfully!"

# 2. Update CUDA drivers (requires sudo)
echo ""
echo "=========================================="
echo "Step 2: Checking CUDA Drivers"
echo "=========================================="
print_warning "CUDA driver updates require sudo privileges."
print_warning "Current NVIDIA driver version:"

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=driver_version --format=csv,noheader
    print_status "NVIDIA driver detected!"
    
    read -p "Do you want to update NVIDIA/CUDA drivers? (y/n): " update_drivers
    if [[ "$update_drivers" == "y" || "$update_drivers" == "Y" ]]; then
        print_status "Updating NVIDIA CUDA drivers..."
        # Add NVIDIA package repositories
        sudo apt-get update
        sudo apt-get install -y software-properties-common
        
        # Install/Update NVIDIA drivers and CUDA toolkit
        print_status "Installing NVIDIA drivers and CUDA toolkit..."
        sudo apt-get install -y nvidia-driver-535 nvidia-cuda-toolkit
        
        print_warning "NVIDIA driver update complete. A reboot may be required."
        print_warning "Run 'sudo reboot' after this script completes if needed."
    else
        print_status "Skipping NVIDIA driver update."
    fi
else
    print_warning "NVIDIA driver not found. Please install NVIDIA drivers manually:"
    echo "   sudo apt-get update"
    echo "   sudo apt-get install -y nvidia-driver-535 nvidia-cuda-toolkit"
    echo ""
    read -p "Continue anyway? (y/n): " continue_anyway
    if [[ "$continue_anyway" != "y" && "$continue_anyway" != "Y" ]]; then
        print_error "Exiting script."
        exit 1
    fi
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
echo "To activate the environment, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To deactivate, run:"
echo "  conda deactivate"
echo ""
echo "To test TensorFlow with GPU:"
echo "  python -c 'import tensorflow as tf; print(tf.config.list_physical_devices(\"GPU\"))'"
echo ""
print_status "All done! Happy coding! 🚀"
