import sys
import os
import ctypes
import glob
import numpy as np

# -------------------------------------------------------------------------
# CRITICAL FIX: Bypass broken GLVND dispatcher by preloading NVIDIA GLES lib
# AND Preload Conda libstdc++ to satisfy LLVM/PyTorch
# -------------------------------------------------------------------------
preloads = []

# 1. Conda libstdc++ (High Priority)
conda_prefix = os.environ.get("CONDA_PREFIX")
if conda_prefix:
    conda_stdc = os.path.join(conda_prefix, "lib", "libstdc++.so.6")
    if os.path.exists(conda_stdc):
        preloads.append(conda_stdc)
        print(f"[Test] Found Conda libstdc++: {conda_stdc}", flush=True)

# 2. NVIDIA GLES lib (System)
nvidia_gles_lib = "/usr/lib64/libGLESv2_nvidia.so.2"
if os.path.exists(nvidia_gles_lib):
    preloads.append(nvidia_gles_lib)
    print(f"[Test] Found NVIDIA GLES lib: {nvidia_gles_lib}", flush=True)

# 3. Apply LD_PRELOAD
current_preload = os.environ.get("LD_PRELOAD", "")
new_preloads = []
for p in preloads:
    if p not in current_preload:
        new_preloads.append(p)

if new_preloads:
    preload_str = ":".join(new_preloads)
    if current_preload:
        preload_str = f"{preload_str}:{current_preload}"
    
    print(f"[Test] Setting LD_PRELOAD: {preload_str}", flush=True)
    os.environ["LD_PRELOAD"] = preload_str
    
    # Re-exec
    if "HABITAT_TEST_REEXEC" not in os.environ:
        os.environ["HABITAT_TEST_REEXEC"] = "1"
        print("[Test] Re-executing to apply LD_PRELOAD...", flush=True)
        try:
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception as e:
            print(f"[Test] Re-exec failed: {e}", flush=True)

# Configure EGL
os.environ["MAGNUM_TARGET_GLES"] = "1"
os.environ["MAGNUM_TARGET_HEADLESS"] = "1"
os.environ["MAGNUM_TARGET_EGL"] = "1"
# Try explicit device platform
os.environ["EGL_PLATFORM"] = "device"
print("[Test] Env: GLES=1, HEADLESS=1, EGL=1, EGL_PLATFORM=device", flush=True)

# Preload libstdc++ (Always needed)
def preload_std_libs():
    libs = [
        "/usr/lib64/libstdc++.so.6",
        "/usr/lib64/libomp.so"
    ]
    for lib in libs:
        if os.path.exists(lib):
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                print(f"[Test] Preloaded {lib}", flush=True)
            except Exception as e:
                print(f"[Test] Failed to preload {lib}: {e}", flush=True)

preload_std_libs()


# Configure EGL/Headless Environment BEFORE importing habitat_sim or torch
print("[Test] Setting Headless Env Vars...", flush=True)

# Set Magnum logging to verbose
os.environ["MAGNUM_LOG"] = "verbose"
os.environ["MAGNUM_GPU_VALIDATION"] = "ON"
os.environ["GLOG_minloglevel"] = "0"
os.environ["HABITAT_SIM_LOG"] = "verbose"

# NOTE: MAGNUM_TARGET_GLES is already set above if needed.
# Do not override it here.

print(f"[Test] Environment variables set: HEADLESS=1, EGL=1, PLATFORM=surfaceless", flush=True)

# Ensure CUDA_VISIBLE_DEVICES is unset to allow EGL to see all GPUs
if "CUDA_VISIBLE_DEVICES" in os.environ:
    del os.environ["CUDA_VISIBLE_DEVICES"]
    print("[Test] Unset CUDA_VISIBLE_DEVICES for EGL visibility", flush=True) 

# FORCE NVIDIA DRIVER (Crucial for GLVND)
nvidia_json = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
if os.path.exists(nvidia_json):
    os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = nvidia_json
    print(f"[Test] Forced EGL vendor to NVIDIA: {nvidia_json}", flush=True)

if "DISPLAY" in os.environ:
    del os.environ["DISPLAY"]
    print("[Test] Removed DISPLAY environment variable.", flush=True)

# LD_LIBRARY_PATH: Do NOT force /usr/lib64 first as it breaks Conda libstdc++
# We rely on LD_PRELOAD to load the correct GL libs from /usr/lib64
print(f"[Test] Using LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '')}", flush=True)

# Test Persistent HabitatEnv
# CRITICAL: Initialize HabitatEnv (EGL) BEFORE Torch
print("[Test] Initializing HabitatEnv...", flush=True)
try:
    from habitat_env import HabitatEnv
    
    # 1. Initialize once
    env = HabitatEnv(0, plot=False)
    print("[Test] HabitatEnv initialized successfully!", flush=True)
    
    # NOW import torch
    import torch
    print(f"[Test] Torch available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print("[Test] Initializing Torch CUDA...", flush=True)
        t = torch.tensor([1.0]).cuda()
        print(f"[Test] Torch CUDA initialized: {t}", flush=True)
    
    # 2. Reset (Episode 1)
    print("[Test] Resetting Env (Episode 1)...", flush=True)
    obs = env.reset()
    print(f"[Test] Episode 1 Reset complete. Observation shape: {obs.shape}", flush=True)
    
    # 3. Step Loop
    print("[Test] Stepping Env (Loop 10 times)...", flush=True)
    current_loc = env.robot_location
    for i in range(10):
        target = current_loc + np.array([0.1 * (i+1), 0.0])
        reward = env.step(target)
        print(f"[Test] Step {i+1} complete. Reward: {reward}", flush=True)
    
    # 4. Reset (Episode 2)
    print("[Test] Resetting Env (Episode 2) - REUSE TEST...", flush=True)
    env.episode_index = 1
    obs = env.reset()
    print(f"[Test] Episode 2 Reset complete. Observation shape: {obs.shape}", flush=True)
    
    print("[Test] ALL TESTS PASSED. Persistent environment works.", flush=True)
    
except Exception as e:
    print(f"[Test] Test failed: {e}", flush=True)
    import traceback
    traceback.print_exc()
