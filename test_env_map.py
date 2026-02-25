
import os
import sys
import numpy as np
import habitat_sim

# Ensure environment is configured for headless
os.environ["MAGNUM_TARGET_GLES"] = "1"
os.environ["MAGNUM_TARGET_HEADLESS"] = "1"
os.environ["MAGNUM_TARGET_EGL"] = "1"
os.environ["EGL_PLATFORM"] = "surfaceless"

from habitat_env import HabitatEnv

def main():
    print("Testing HabitatEnv Ground Truth Map Generation...")
    try:
        env = HabitatEnv(0, plot=False)
        print("HabitatEnv initialized.")
        
        gt_map = env.ground_truth
        free_cells = np.sum(gt_map == 255)
        print(f"Ground Truth Map Free Cells: {free_cells}")
        
        if free_cells > 0:
            print("SUCCESS: Map generation works!")
        else:
            print("FAILURE: Map is empty!")
            
        env.sim.close()
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
