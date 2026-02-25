#!/bin/bash
# =============================================================================
# Vanda HPC - Habitat-Sim GPU Environment Setup Script
# Created for: Vanda GPU Node (A100/V100)
# =============================================================================

echo "🚀 Starting Habitat-Sim GPU Setup for Vanda HPC..."

# 1. Basic Configuration
USER_ID=$(whoami)
BASE_DIR="$HOME"
ENV_NAME="habitat_gpu"
ENV_PATH="$BASE_DIR/envs/$ENV_NAME"

echo "📍 Install Path: $ENV_PATH"

# 2. Check & Install Miniconda (if missing)
if ! command -v conda &> /dev/null; then
    if [ -d "$HOME/miniconda3" ]; then
        echo "🔄 Found existing Miniconda at ~/miniconda3, sourcing..."
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    else
        echo "⚠️ Conda not found. Installing Miniconda3..."
        INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
        wget "https://repo.anaconda.com/miniconda/$INSTALLER" -O "$HOME/$INSTALLER"
        bash "$HOME/$INSTALLER" -b -p "$HOME/miniconda3"
        rm "$HOME/$INSTALLER"
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
        "$HOME/miniconda3/bin/conda" init bash
        echo "✅ Miniconda installed."
    fi
fi

# Hack: Create empty .anaconda/navigator/anaconda-navigator.ini to potentially bypass some checks
# or ensure channels are set to conda-forge to avoid defaults if possible.
# But the error is specific to 'conda tos'.
# We will try to set channel_priority to strict and use conda-forge first to avoid main channel issues if possible,
# or simply attempt to remove defaults if that's the blocker.
# However, the most robust fix for this specific Anaconda ToS error in automated scripts is usually
# to just use conda-forge which doesn't have this ToS restriction, OR manually accept.
# Since we can't manually interact, we will try to configure conda to use conda-forge primarily.

echo "🔧 Configuring Conda Channels to prefer conda-forge (bypassing Anaconda Main ToS issues)..."
conda config --add channels conda-forge
conda config --set channel_priority flexible

# 3. Create Environment (Python 3.7 recommended for older Habitat compatibility)
if [ ! -d "$ENV_PATH" ]; then
    echo "📦 Creating Conda environment '$ENV_NAME'..."
    # Explicitly use conda-forge for creation too
    conda create -p "$ENV_PATH" -c conda-forge python=3.7 cmake -y
else
    echo "✅ Environment '$ENV_NAME' already exists."
fi

# Activate Environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"

# 4. Install Dependencies (GPU Optimized)
echo "⬇️ Installing Dependencies..."

# 4.1 Basic Science Stack
# Using --override-channels -c conda-forge explicitly to avoid 'defaults' channel ToS
conda install -p "$ENV_PATH" --override-channels -c conda-forge numpy cython imageio -y

# 4.2 Habitat-Sim (with Bullet Physics & Headless support)
# 'withbullet' enables physics. Conda usually handles CUDA deps automatically.
echo "Installing Habitat-Sim (with Bullet Physics)..."
conda install -p "$ENV_PATH" --override-channels -c conda-forge -c aihabitat habitat-sim withbullet -y

# 4.3 Headless Rendering Support (Critical for Server)
echo "Installing Headless Rendering libraries..."
conda install -p "$ENV_PATH" --override-channels -c conda-forge libglvnd mesalib -y

# 4.4 PyTorch (GPU - CUDA 11.8)
# Explicitly installing CUDA version to ensure GPU usage
echo "Installing PyTorch (CUDA 11.8)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 5. Verification
echo "🔍 Verifying Installation..."
python -c "import torch; print('PyTorch CUDA Available:', torch.cuda.is_available())"
python -c "import habitat_sim; print('Habitat-Sim Version:', habitat_sim.__version__)"

echo "🎉 Setup Complete! To use the environment, run:"
echo "   conda activate $ENV_PATH"
