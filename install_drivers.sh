#!/bin/bash
# install_drivers.sh
# 这是一个用于修复缺失 3D 驱动的脚本
# 请在登录节点 (Login Node) 运行此脚本: bash install_drivers.sh

source ~/miniconda3/etc/profile.d/conda.sh

echo "=================================================="
echo "   Fixing Habitat 3D Drivers (mesa)"
echo "=================================================="

# 1. 激活环境
echo "[1/3] Activating environment: habitat_headless..."
conda activate habitat_headless
if [ $? -ne 0 ]; then
    echo "ERROR: Could not activate environment 'habitat_headless'"
    exit 1
fi

# 2. 安装驱动 (尝试通用包名 mesa)
echo "[2/3] Installing mesa (includes drivers) from conda-forge..."
echo "      Using --freeze-installed to avoid dependency hell..."

# 尝试安装 mesa 主包 (通常包含 swrast_dri.so)
# 同时也安装 libglvnd (OpenGL 分发库)
conda install -y -c conda-forge mesa libglvnd --no-deps --freeze-installed

if [ $? -ne 0 ]; then
    echo "WARNING: Fast install failed. Retrying with standard install..."
    conda install -y -c conda-forge mesa libglvnd
fi

# 3. 验证
echo "[3/3] Verifying installation..."
# 标准 Conda 路径
DRIVER_PATH="$CONDA_PREFIX/lib/dri/swrast_dri.so"

if [ -f "$DRIVER_PATH" ]; then
    echo "SUCCESS: Standard Conda Driver found at: $DRIVER_PATH"
    echo "You can now submit your job using: qsub run.pbs"
else
    echo "WARNING: Standard driver NOT found."
    echo "Checking for sysroot driver (fallback)..."
    SYSROOT_DRIVER=$(find "$CONDA_PREFIX" -name "swrast_dri.so" | grep "sysroot" | head -n 1)
    if [ ! -z "$SYSROOT_DRIVER" ]; then
        echo "Found sysroot driver at: $SYSROOT_DRIVER"
        echo "Note: Sysroot drivers might require extra dependencies (handled by run.pbs)."
    else
        echo "CRITICAL: No swrast_dri.so found anywhere!"
    fi
fi

echo "Done."
