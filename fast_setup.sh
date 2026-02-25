#!/bin/bash

# =============================================================================
# Simplest Habitat-Sim CPU/Headless Environment Setup Script
# VERSION: 24.0 (Reverted to CPU/Headless by User Request)
# =============================================================================
echo "Running fast_setup.sh V24.0 (CPU/Headless Revert)..."

# 0. Cleanup Windows Line Endings (Just in case)
# sed -i 's/\r$//' "$0"

# 1. Initialize Conda/Mamba
source ~/.bashrc

# Try to locate and source conda if not available
if ! command -v conda &> /dev/null; then
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    fi
fi

# Final check and Install if missing
if ! command -v conda &> /dev/null; then
    # Check if Miniconda directory already exists
    if [ -d "$HOME/miniconda3" ]; then
        echo "⚠️ Conda command not found, but ~/miniconda3 exists."
        echo "🔄 Sourcing existing Miniconda..."
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    else
        echo "⚠️ Conda not found. Installing Miniconda3 automatically..."
        
        # Define paths
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        INSTALLER_PATH="$HOME/miniconda_installer.sh"
        INSTALL_DIR="$HOME/miniconda3"
        
        # Download (Skip if installer already exists)
        if [ ! -f "$INSTALLER_PATH" ]; then
            echo "⬇️ Downloading Miniconda..."
            if command -v wget >/dev/null 2>&1; then
                wget "$MINICONDA_URL" -O "$INSTALLER_PATH"
            elif command -v curl >/dev/null 2>&1; then
                curl -L "$MINICONDA_URL" -o "$INSTALLER_PATH"
            else
                echo "❌ Error: Neither wget nor curl found. Cannot download Miniconda."
                exit 1
            fi
        else
            echo "⬇️ Installer found, skipping download."
        fi
        
        # Install
        echo "📦 Installing Miniconda to $INSTALL_DIR..."
        bash "$INSTALLER_PATH" -b -p "$INSTALL_DIR"
        # Don't delete installer immediately in case of failure, user can clean up later
        # rm "$INSTALLER_PATH"
        
        # Activate for this session
        source "$INSTALL_DIR/etc/profile.d/conda.sh"
        
        # Init for future sessions
        "$INSTALL_DIR/bin/conda" init bash
        
        echo "✅ Miniconda installed and activated!"
    fi
fi

CONDA_BIN="conda"
if command -v mamba >/dev/null 2>&1; then
    CONDA_BIN="mamba"
    echo "🐍 Mamba found! Using for installation commands."
fi

# Ensure Conda base environment is sourced correctly
if [ -n "$CONDA_EXE" ] && [ -f "${CONDA_EXE%/*}/../etc/profile.d/conda.sh" ]; then
    source "${CONDA_EXE%/*}/../etc/profile.d/conda.sh"
fi

USER_ID=$(whoami)
# Vanda HPC Note: /hpctmp is often node-local or temporary. 
# Using Home directory for persistence across sessions and nodes.
BASE_DIR="$HOME"
ENV_PATH="$BASE_DIR/envs/habitat"
REPO_DIR="$BASE_DIR/repos"

# Define explicit paths to binaries
PYTHON_BIN="$ENV_PATH/bin/python"
PIP_BIN="$ENV_PATH/bin/pip"

# Activate Environment (Safe Method)
echo "🔌 Activating environment..."
# Check if environment exists before trying to activate
if [ -d "$ENV_PATH" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_PATH"
else
    echo "Environment not found at $ENV_PATH. Will create it in Phase 1."
fi

echo "Using Python: $PYTHON_BIN"

# Helper function to check python package
check_pkg() {
    "$PYTHON_BIN" -c "import $1" >/dev/null 2>&1
}

check_pkg_ver() {
    "$PYTHON_BIN" -c "import $1; assert $1.__version__ == '$2'" >/dev/null 2>&1
}

# =============================================================================
# PHASE 1: CREATE ENVIRONMENT (If not exists)
# =============================================================================
if [ ! -d "$ENV_PATH" ]; then
    echo "Creating environment in $ENV_PATH..."
    # Standard habitat env: python 3.7, cmake
    "$CONDA_BIN" create -p "$ENV_PATH" python=3.7 cmake -y
else
    echo "Environment exists. Skipping creation."
fi

# Re-activate to ensure paths are set
# Check if environment exists before trying to activate
if [ -d "$ENV_PATH" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_PATH"
else
    echo "Environment not found at $ENV_PATH. Will create it in Phase 1."
fi

# =============================================================================
# PHASE 2: INSTALL DEPENDENCIES (GPU Focus)
# =============================================================================

# 1. Basic Scientific Stack
if ! check_pkg "numpy" || ! check_pkg "cython" || ! check_pkg "imageio"; then
    echo "Installing numpy, cython, imageio..."
    "$CONDA_BIN" install -p "$ENV_PATH" numpy cython imageio -y
fi

# 2. Habitat-Sim (GPU + Bullet Physics)
# Installing from conda-forge and aihabitat channel. 
# We request 'habitat-sim' with 'withbullet' feature for physics support.
# Ideally, conda resolves to a GPU-compatible build if CUDA is present, but we force it via meta-packages if needed.
if ! check_pkg "habitat_sim"; then
    echo "Installing Habitat-Sim (GPU + Bullet)..."
    # Installing habitat-sim with bullet physics support
    "$CONDA_BIN" install -p "$ENV_PATH" -c conda-forge -c aihabitat habitat-sim withbullet -y
fi

# 2.1 Essential System Libraries for Headless Rendering
# Even on GPU, we might need these for EGL context creation if system libs are old
if [ ! -f "$ENV_PATH/lib/libGL.so.1" ]; then
    echo "Ensuring libglvnd and mesalib are installed..."
    "$CONDA_BIN" install -p "$ENV_PATH" -c conda-forge libglvnd mesalib -y
fi

# 2.1.5 PyTorch (GPU/CUDA)
# Installing PyTorch with CUDA 11.x support (Common for Vanda/A100/V100 nodes)
if ! check_pkg "torch"; then
    echo "Installing PyTorch (GPU/CUDA)..."
    # Using pip for better control over CUDA version. Assuming CUDA 11.8 compatibility.
    "$PIP_BIN" install torch torchvision --index-url https://download.pytorch.org/whl/cu118
fi

# 2.2 Symlink Hack for libOpenGL.so.0
# Many NUS HPC nodes have missing/mismatched system libraries.
if [ -f "$ENV_PATH/lib/libGL.so.1" ] && [ ! -f "$ENV_PATH/lib/libOpenGL.so.0" ]; then
    echo "Applying libOpenGL.so.0 symlink hack (Creating link from libGL.so.1)..."
    ln -sf "$ENV_PATH/lib/libGL.so.1" "$ENV_PATH/lib/libOpenGL.so.0"
fi

# 3. Gym (Legacy 0.21.0 or 0.22.0)
# Habitat-Lab 0.3.1 usually wants Gym 0.22.0+ for typing, but we'll stick to what works.
# Let's install 0.22.0 to be safe against ActType errors.
if ! check_pkg "gym"; then
    echo "Installing Gym 0.22.0..."
    "$PIP_BIN" install "gym==0.22.0"
fi

# 4. Habitat-Lab (From GitHub, pinned to v0.3.1)
if ! check_pkg "habitat"; then
    echo "Installing Habitat-Lab v0.3.1..."
    mkdir -p "$REPO_DIR"
    cd "$REPO_DIR"
    if [ ! -d "habitat-lab" ]; then
        git clone --branch v0.3.1 https://github.com/facebookresearch/habitat-lab.git
    fi
    cd habitat-lab
    # Install with --no-deps to avoid gym version conflict if any
    "$PIP_BIN" install . --no-deps
fi

echo "✅ Environment Setup Complete (CPU/Headless Mode)."
echo "   Environment Location: $ENV_PATH"
