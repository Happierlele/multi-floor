#!/bin/bash
# Helper script to check if the required library exists in the habitat_old environment

source /home/svu/e1616048/miniconda3/etc/profile.d/conda.sh

echo "Activating 'habitat_old' environment..."
conda activate habitat_old

LIB_PATH="$CONDA_PREFIX/lib/libstdc++.so.6"
echo "Checking: $LIB_PATH"

if [ -f "$LIB_PATH" ]; then
    echo "✅ File exists."
    
    echo "Checking for GLIBCXX_3.4.29..."
    if strings "$LIB_PATH" | grep -q "GLIBCXX_3.4.29"; then
        echo "✅ Version GLIBCXX_3.4.29 FOUND! The environment is ready."
    else
        echo "❌ Version GLIBCXX_3.4.29 NOT found in this file."
        echo "   (Found versions up to: $(strings "$LIB_PATH" | grep "GLIBCXX_3.4" | tail -n 1))"
        echo "   Action required: Run 'conda install -c conda-forge libstdcxx-ng'"
    fi
else
    echo "❌ File does NOT exist."
    echo "   Action required: Run 'conda install -c conda-forge libstdcxx-ng'"
fi
