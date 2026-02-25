#!/bin/bash
# =============================================================================
# Vanda GPU Cluster Setup Script
# =============================================================================
# This script sets up the Habitat environment specifically for Vanda (GPU).
# It uses the shared Conda cache to avoid re-downloading packages where possible.

echo "🚀 Setting up Environment for Vanda (GPU)..."

# 1. Initialize Conda/Mamba
source ~/.bashrc
CONDA_BIN="conda"
if command -v mamba >/dev/null 2>&1; then
    CONDA_BIN="mamba"
    echo "   ✅ Found Mamba (Faster!)"
fi

# 2. Define Path (Using /hpctmp on Vanda)
USER_ID=$(whoami)
ENV_PATH="/hpctmp/$USER_ID/envs/habitat_vanda"

echo "   📂 Target Environment: $ENV_PATH"

# 3. Create Environment (if not exists)
if [ ! -d "$ENV_PATH" ]; then
    echo "   📦 Creating environment..."
    # We create a python 3.7 env first
    "$CONDA_BIN" create -p "$ENV_PATH" python=3.7 cmake -y
else
    echo "   ✅ Environment directory exists."
fi

# 4. Activate
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"

# 5. Install Dependencies (Leveraging Cache)
echo "   ⬇️ Installing Dependencies..."

# 5.1 PyTorch with CUDA support (Essential for Vanda)
# Vanda usually runs CUDA 11.x or 12.x. We'll pick a safe 11.7 or 11.8 compatible version.
# Or just let conda resolve the best one for the driver.
echo "      Installing PyTorch (GPU)..."
"$CONDA_BIN" install -p "$ENV_PATH" pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y

# 5.2 Habitat-Sim (GPU Version)
# Note: 'with-bullet' enables physics.
echo "      Installing Habitat-Sim (GPU)..."
"$CONDA_BIN" install -p "$ENV_PATH" habitat-sim with-bullet -c conda-forge -c aihabitat -y

# 5.3 Other Python Deps
echo "      Installing PIP dependencies..."
# Using pip from the environment
PIP_BIN="$ENV_PATH/bin/pip"
"$PIP_BIN" install gym==0.22.0 tensorboard scikit-image scipy matplotlib ray

# 6. Check/Download Habitat Test Data
echo "   📂 Checking Habitat Test Data..."
DATA_DIR="data/versioned_data/habitat_test_scenes"
SCENE_FILE="$DATA_DIR/skokloster-castle.glb"

if [ ! -f "$SCENE_FILE" ]; then
    echo "      Downloading Habitat Test Scenes..."
    # Activate env to use python
    "$ENV_PATH/bin/python" -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data/ --no-replace
else
    echo "      ✅ Test scenes found."
fi

echo "✅ Setup Complete for Vanda!"
echo "   To run: qsub run_vanda.pbs"
