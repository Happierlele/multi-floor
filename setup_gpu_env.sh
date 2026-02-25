#!/bin/bash

# =============================================================================
# Habitat-Sim GPU Environment Setup Script for NUS HPC
# =============================================================================

# 1. Initialize Conda
source ~/.bashrc
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "Error: Conda not found. Please load conda module or install Miniconda."
    exit 1
fi

# 2. Create Environment
ENV_NAME="habitat_gpu"
echo "Creating Conda environment: $ENV_NAME..."
conda create -n $ENV_NAME python=3.9 cmake=3.14.0 -y

# 3. Activate Environment
conda activate $ENV_NAME

# 4. Install PyTorch (CUDA 11.8 compatible)
echo "Installing PyTorch (CUDA 11.8)..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# 5. Install Habitat-Sim (Headless with GPU support)
# We use the 'headless' build which supports EGL on GPU
echo "Installing Habitat-Sim..."
conda install habitat-sim=0.3.1 withbullet headless -c conda-forge -c aihabitat -y

# 6. Install Habitat-Lab
echo "Installing Habitat-Lab..."
pip install habitat-lab==0.3.1 habitat-baselines==0.3.1

# 7. Install Other Dependencies
echo "Installing additional dependencies..."
pip install gym==0.21.0
pip install numpy<2.0
pip install opencv-python
pip install matplotlib
pip install yacs
pip install imageio
pip install imageio-ffmpeg
pip install tensorboard
pip install ray[default]

echo "============================================================================="
echo "Setup Complete!"
echo "To activate: conda activate $ENV_NAME"
echo "============================================================================="
