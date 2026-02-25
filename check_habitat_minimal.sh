#!/bin/bash
# =============================================================================
# Minimal Habitat Check Script
# Only sets up environment and checks what's missing
# =============================================================================

echo "🔍 Starting Minimal Habitat Check..."

USER_ID=$(whoami)
BASE_DIR="$HOME"
ENV_NAME="habitat_gpu"
ENV_PATH="$BASE_DIR/envs/$ENV_NAME"

# 1. Ensure Miniconda is available
if [ -d "$HOME/miniconda3" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif command -v conda &> /dev/null; then
    # conda is already in path
    :
else
    echo "⚠️ Conda not found. Please install Miniconda first or run setup_vanda_gpu.sh to install it."
    exit 1
fi

# 2. Check/Create Environment (Minimal)
if [ ! -d "$ENV_PATH" ]; then
    echo "📦 Creating minimal environment '$ENV_NAME'..."
    conda create -p "$ENV_PATH" --override-channels -c conda-forge python=3.7 cmake -y
else
    echo "✅ Using existing environment '$ENV_NAME'."
fi

# Activate
conda activate "$ENV_PATH"

# 3. Try Import Check
echo "----------------------------------------------------------------"
echo "🧪 Checking Habitat-Sim Import..."
python -c "import habitat_sim; print('✅ Habitat-Sim Imported Successfully! Version:', habitat_sim.__version__)" 2>/tmp/habitat_err.log

if [ $? -eq 0 ]; then
    echo "🎉 Habitat-Sim is working!"
else
    echo "❌ Habitat-Sim Import Failed."
    echo "Here is the error message:"
    echo "----------------------------------------------------------------"
    cat /tmp/habitat_err.log
    echo "----------------------------------------------------------------"
    
    # Analyze common errors
    ERR_LOG=$(cat /tmp/habitat_err.log)
    if [[ "$ERR_LOG" == *"No module named 'habitat_sim'"* ]]; then
        echo "💡 Diagnosis: 'habitat-sim' is not installed at all."
        echo "   Action: You need to install it."
        echo "   Command: conda install -p $ENV_PATH --override-channels -c conda-forge -c aihabitat habitat-sim withbullet -y"
    elif [[ "$ERR_LOG" == *"libGL"* ]] || [[ "$ERR_LOG" == *"EGL"* ]]; then
        echo "💡 Diagnosis: Missing OpenGL/EGL libraries (Headless rendering issue)."
        echo "   Action: Install headless rendering support."
        echo "   Command: conda install -p $ENV_PATH --override-channels -c conda-forge libglvnd mesalib -y"
    elif [[ "$ERR_LOG" == *"No module named 'numpy'"* ]]; then
        echo "💡 Diagnosis: Missing Numpy."
        echo "   Command: conda install -p $ENV_PATH --override-channels -c conda-forge numpy -y"
    fi
fi
echo "----------------------------------------------------------------"
