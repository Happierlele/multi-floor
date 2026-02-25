#!/bin/bash

# =============================================================================
# Clean Up and Reinstall Script (Fixes Disk Quota Issue)
# =============================================================================

echo "Checking disk usage..."
hpc s

echo "Cleaning up conda cache..."
conda clean --all -y
pip cache purge

echo "Cleaning up pip cache directory..."
rm -rf ~/.cache/pip

# Check if environment exists and remove it if partial
if conda env list | grep -q "habitat_gpu"; then
    echo "Removing partial environment 'habitat_gpu'..."
    conda env remove -n habitat_gpu -y
fi

echo "Moving Miniconda pkgs to scratch (if not already linking)..."
# This is a common trick on HPC: move heavy folders to scratch and symlink back
if [ ! -L ~/.conda/pkgs ]; then
    mkdir -p /hpctmp/e1616048/conda_pkgs
    mkdir -p ~/.conda
    # If pkgs exists, move it. If not, just link.
    if [ -d ~/.conda/pkgs ]; then
        mv ~/.conda/pkgs/* /hpctmp/e1616048/conda_pkgs/
        rm -rf ~/.conda/pkgs
    fi
    ln -s /hpctmp/e1616048/conda_pkgs ~/.conda/pkgs
    echo "Linked ~/.conda/pkgs to /hpctmp/e1616048/conda_pkgs"
fi

echo "============================================================================="
echo "Cleanup Done. Now re-running setup..."
echo "============================================================================="

# Re-run the setup script
./setup_gpu_env.sh
