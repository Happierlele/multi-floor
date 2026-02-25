#!/bin/bash
# Checks environment for 3D Habitat requirements

echo "Checking environment: $CONDA_PREFIX"

echo "1. Checking for mesa-dri-drivers (swrast_dri.so)..."
DRI_PATH="$CONDA_PREFIX/lib/dri/swrast_dri.so"
if [ -f "$DRI_PATH" ]; then
    echo "   [OK] Found at $DRI_PATH"
    echo "   Checking dependencies:"
    ldd "$DRI_PATH" | grep "not found"
    if [ $? -eq 0 ]; then
        echo "   [FAIL] Missing dependencies!"
    else
        echo "   [OK] Dependencies look good."
    fi
else
    echo "   [FAIL] Not found in standard location."
    SYSROOT_DRI=$(find "$CONDA_PREFIX" -name "swrast_dri.so" 2>/dev/null | grep sysroot | head -n 1)
    if [ ! -z "$SYSROOT_DRI" ]; then
        echo "   [WARNING] Found only in sysroot: $SYSROOT_DRI"
        echo "   This driver is usually broken. Please run install_drivers.sh."
    fi
fi

echo ""
echo "2. Checking for Xvfb..."
which Xvfb > /dev/null
if [ $? -eq 0 ]; then
    echo "   [OK] Found Xvfb at $(which Xvfb)"
else
    echo "   [FAIL] Xvfb not found in PATH."
fi

echo ""
echo "3. Checking GLIBCXX version..."
strings "$CONDA_PREFIX/lib/libstdc++.so.6" | grep GLIBCXX_3.4.29 > /dev/null
if [ $? -eq 0 ]; then
    echo "   [OK] GLIBCXX_3.4.29 found."
else
    echo "   [WARNING] GLIBCXX_3.4.29 NOT found. Might cause import errors."
fi
