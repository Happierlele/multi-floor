#!/bin/bash
# fix_headless_egl.sh - V3.0 (System Link Edition)
# Fixes EGL_NOT_INITIALIZED by linking Host System Libraries

USER_ID=$(whoami)
# Ensure we target the correct environment path
if [ -d "/hpctmp/$USER_ID/envs/habitat" ]; then
    ENV_PATH="/hpctmp/$USER_ID/envs/habitat"
elif [ -d "/hpctmp/$USER_ID/large-scale-DRL-exploration-new_version - habitat" ]; then
     # Handle the case where the user might have created env inside the repo or similar
     # But based on fast_setup.sh it is in /envs/habitat
     ENV_PATH="/hpctmp/$USER_ID/envs/habitat"
else
    # Fallback to active conda env
    ENV_PATH=$CONDA_PREFIX
fi

LIB_PATH="$ENV_PATH/lib"

echo "🔧 Running EGL Fix V3.0 for $ENV_PATH..."
echo "   Target Lib Path: $LIB_PATH"

if [ ! -d "$LIB_PATH" ]; then
    echo "❌ Library path does not exist! Check environment."
    exit 1
fi

# 1. Clean broken symlinks
echo "🧹 Cleaning broken symlinks..."
find "$LIB_PATH" -xtype l -name "libGL*" -delete
find "$LIB_PATH" -xtype l -name "libEGL*" -delete
find "$LIB_PATH" -xtype l -name "libOpenGL*" -delete
find "$LIB_PATH" -xtype l -name "libGLdispatch*" -delete

# 2. Link System Libraries (The "Silver Bullet" for HPC)
# Instead of relying on broken Conda packages, we link the robust system libraries.
echo "🔗 Linking Host System Libraries..."

link_host_lib() {
    SEARCH_NAME=$1
    TARGET_NAME=$2
    
    # Find the library in /usr/lib64 (Standard RHEL/CentOS location)
    # We use 'head -n 1' to pick the first match (usually the most relevant one)
    HOST_FILE=$(find /usr/lib64 -name "$SEARCH_NAME" -print -quit)
    
    if [ -n "$HOST_FILE" ]; then
        echo "   ✅ Found host $SEARCH_NAME: $HOST_FILE"
        ln -sf "$HOST_FILE" "$LIB_PATH/$TARGET_NAME"
        echo "      Linked to $LIB_PATH/$TARGET_NAME"
    else
        echo "   ⚠️ Host $SEARCH_NAME not found in /usr/lib64."
        
        # Try finding in /usr/lib or other locations if needed
        HOST_FILE_ALT=$(find /usr/lib -name "$SEARCH_NAME" -print -quit)
        if [ -n "$HOST_FILE_ALT" ]; then
             echo "   ✅ Found host (alt) $SEARCH_NAME: $HOST_FILE_ALT"
             ln -sf "$HOST_FILE_ALT" "$LIB_PATH/$TARGET_NAME"
        else
             echo "   ❌ Could not find $SEARCH_NAME on host system."
        fi
    fi
}

# Link key libraries required for EGL/OpenGL
link_host_lib "libEGL.so.1*" "libEGL.so.1"
link_host_lib "libGL.so.1*" "libGL.so.1"

# Special handling for libOpenGL: Fallback to libGL if not found
HOST_OPENGL=$(find /usr/lib64 -name "libOpenGL.so.0*" -print -quit)
if [ -n "$HOST_OPENGL" ]; then
    echo "   ✅ Found host libOpenGL: $HOST_OPENGL"
    ln -sf "$HOST_OPENGL" "$LIB_PATH/libOpenGL.so.0"
else
    echo "   ⚠️ Host libOpenGL not found. FALLBACK: Linking libGL.so.1 to libOpenGL.so.0"
    # Ensure libGL.so.1 is already linked or exists
    if [ -e "$LIB_PATH/libGL.so.1" ]; then
        # We REMOVE libOpenGL.so.0 link because it causes "undefined symbol: __GLXGL_CORE_FUNCTIONS"
        # The app should fallback to libGL.so.1 automatically if libOpenGL is missing, 
        # or we will force it via LD_PRELOAD in run.pbs
        if [ -L "$LIB_PATH/libOpenGL.so.0" ]; then
             echo "      ⚠️ Removing broken libOpenGL.so.0 symlink to allow fallback..."
             rm "$LIB_PATH/libOpenGL.so.0"
        fi
        echo "      ℹ️ Strategy: Relying on libGL.so.1 directly (No fake libOpenGL)"
    else
        echo "      ❌ Cannot fallback: libGL.so.1 is also missing!"
    fi
fi

# Special handling for libGLdispatch: FORCE Link to libGL.so.1 (Avoid undefined symbol)
echo "   ⚠️ FORCE FIX: Linking libGLdispatch.so.0 -> libGL.so.1 to avoid undefined symbols"
if [ -e "$LIB_PATH/libGL.so.1" ]; then
    ln -sf "$LIB_PATH/libGL.so.1" "$LIB_PATH/libGLdispatch.so.0"
fi

# Special handling for libGLX: SHADOWING STRATEGY
# If we remove it, system finds /lib64/libGLX.so.0 -> incompatible.
# If we link to libGL.so.1 -> __GLXGL_CORE_FUNCTIONS error.
#
# NEW STRATEGY: We must create a dummy libGLX.so.0 that is actually libGL.so.1,
# AND we must use LD_PRELOAD in run.pbs to force loading libGL.so.1 first.
# For now, we link it to libGL.so.1 again, but we will rely on LD_PRELOAD to fix the symbol issue.
echo "   ⚠️ SHADOW FIX: Linking libGLX.so.0 -> libGL.so.1 to shadow host library"
if [ -e "$LIB_PATH/libGL.so.1" ]; then
    ln -sf "$LIB_PATH/libGL.so.1" "$LIB_PATH/libGLX.so.0"
fi


# 3. Verify
echo "🔍 Verification:"
ls -l "$LIB_PATH/libEGL.so.1" 2>/dev/null
ls -l "$LIB_PATH/libGL.so.1" 2>/dev/null
ls -l "$LIB_PATH/libOpenGL.so.0" 2>/dev/null

echo "✅ Fix Complete. Please submit your job now."
