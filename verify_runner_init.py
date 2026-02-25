
import os
import sys

# CRITICAL FIX: Ensure Conda's lib folder is FIRST in LD_LIBRARY_PATH
# This forces the dynamic linker to find the newer libstdc++.so.6 from Conda
# instead of the system one, fixing the "GLIBCXX_3.4.30 not found" error.
conda_prefix = os.environ.get('CONDA_PREFIX')
if conda_prefix:
    lib_path = os.path.join(conda_prefix, "lib")
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_path not in current_ld:
        # Prepend to ensure priority
        os.environ["LD_LIBRARY_PATH"] = f"{lib_path}:{current_ld}"
        print(f"[Verify] Prepended Conda lib to LD_LIBRARY_PATH: {lib_path}", flush=True)
        # We must re-exec the process for LD_LIBRARY_PATH to take effect for the main process
        try:
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception as e:
            print(f"[Verify] Failed to re-exec: {e}", flush=True)

import time
import torch
if torch.cuda.is_available():
    print("[Verify] Initializing Torch CUDA...", flush=True)
    t = torch.tensor([1.0]).cuda()
    print(f"[Verify] Torch CUDA initialized: {t}", flush=True)

print(f"[Verify] Process ID: {os.getpid()}", flush=True)

# Simulate what runner.py does: import worker_habitat first
print("[Verify] Importing worker_habitat...", flush=True)
try:
    import worker_habitat
    print("[Verify] worker_habitat imported successfully.", flush=True)
except Exception as e:
    print(f"[Verify] Failed to import worker_habitat: {e}", flush=True)
    sys.exit(1)

# Now try the pre-initialization logic from runner.py
print("[Verify] Starting Habitat-Sim pre-initialization...", flush=True)

try:
    import habitat_sim
    
    # Use a minimal configuration for init check
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = "data/versioned_data/habitat_test_scenes/skokloster-castle.glb"
    sim_cfg.gpu_device_id = 0
    sim_cfg.enable_physics = False
    
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    
    print("[Verify] Calling habitat_sim.Simulator(cfg)...", flush=True)
    # Initialize and immediately close
    sim = habitat_sim.Simulator(cfg)
    print("[Verify] Simulator created!", flush=True)
    sim.close()
    del sim
    print(f"[Verify] Habitat-Sim pre-initialization successful!", flush=True)

except Exception as e:
    print(f"[Verify] Habitat-Sim pre-init failed: {e}", flush=True)
    import traceback
    traceback.print_exc()

print("[Verify] Done.", flush=True)
