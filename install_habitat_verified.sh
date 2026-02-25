#!/bin/bash
# =============================================================================
# Install Habitat-Sim (Verified Method)
# Installs ONLY habitat-sim and essential rendering libs
# Bypasses Anaconda TOS by using conda-forge exclusively
# =============================================================================

USER_ID=$(whoami)
BASE_DIR="$HOME"
ENV_NAME="habitat_gpu"
ENV_PATH="$BASE_DIR/envs/$ENV_NAME"

echo "🚀 Starting Targeted Installation..."

# 1. Source Conda
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"

# 2. Install Habitat-Sim (The Missing Piece)
echo "📦 Installing Habitat-Sim (with Bullet Physics)..."
# strict channel override to avoid TOS errors
conda install -p "$ENV_PATH" \
    --override-channels \
    -c conda-forge \
    -c aihabitat \
    habitat-sim withbullet \
    -y

# 3. Install Headless Rendering Libs (Prevent next error)
# Without these, you will likely get an EGL/OpenGL error immediately after installation
echo "🎨 Installing Headless Rendering Support..."
conda install -p "$ENV_PATH" \
    --override-channels \
    -c conda-forge \
    libglvnd mesalib \
    -y

echo "✅ Installation actions complete."
echo "🔄 Running check again..."

# 4. Immediate Verification
python -c "import habitat_sim; print('🎉 SUCCESS: Habitat-Sim Imported! Version:', habitat_sim.__version__)"
