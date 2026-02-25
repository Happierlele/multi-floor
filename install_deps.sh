#!/bin/bash
set -e

# Define environment name and path
ENV_NAME="habitat_gpu"
ENV_PATH="$HOME/miniconda3/envs/$ENV_NAME"

echo "----------------------------------------------------------------"
echo "🔧 Installing Remaining Dependencies for Habitat (Vanda HPC)"
echo "----------------------------------------------------------------"

# Ensure Conda is available
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
else
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

echo "🚀 Activating environment: $ENV_NAME"
conda activate "$ENV_NAME"

# Check Python version
PY_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "ℹ️  Python Version: $PY_VERSION"

# Install PyTorch
# For Python 3.7 (which is likely what is installed if using older habitat-sim),
# we need to use pip with specific versions as conda channels might be broken for old PyTorch.
echo "📦 Installing PyTorch (CUDA 11.7/11.8 compatible)..."

if [[ "$PY_VERSION" == "3.7" ]]; then
    echo "⚠️  Detected Python 3.7. Using pip to install PyTorch 1.13.1 (Last version supporting Py3.7)..."
    pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
else
    echo "ℹ️  Python > 3.7 detected. Using Conda-Forge for PyTorch (Safe & Fast)..."
    conda install -p "$ENV_PATH" --override-channels -c conda-forge pytorch torchvision torchaudio pytorch-cuda=11.8 -y
fi

echo "📦 Installing Project Dependencies (Ray, Tensorboard, etc.)..."
pip install scikit-image matplotlib ray tensorboard opencv-python

echo "----------------------------------------------------------------"
echo "✅ All dependencies installed successfully!"
echo "----------------------------------------------------------------"
