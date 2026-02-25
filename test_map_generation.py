
import habitat_sim
import numpy as np
import os
import matplotlib.pyplot as plt
from habitat_mapper import SimpleMapper

# Configuration
SCENE_ID = "data/versioned_data/habitat_test_scenes/skokloster-castle.glb"
CELL_SIZE = 0.4
MAP_SIZE_METERS = 100.0

def get_topdown_map(sim, cell_size, map_size_meters):
    map_size_pixels = int(map_size_meters / cell_size)
    map_center = map_size_pixels // 2
    
    # Initialize map with UNKNOWN
    topdown_map = np.ones((map_size_pixels, map_size_pixels), dtype=np.uint8) * 127
    
    # Get PathFinder
    pf = sim.pathfinder
    if not pf.is_loaded:
        print("Error: PathFinder not loaded!")
        return topdown_map

    # Get bounds
    bounds = pf.get_bounds()
    print(f"PathFinder Bounds: min={bounds[0]}, max={bounds[1]}")
    
    # Height for navigability check (usually slightly above floor)
    # The scene might have multiple floors. We'll start with y ~ 0.
    # Habitat coordinates are (x, y, z) where y is up.
    height = bounds[0][1] + 0.2 # 0.2m above floor
    
    print(f"Generating map at height {height}...")
    
    count_free = 0
    
    # Iterate over pixels
    # We scan a grid. 
    # Map pixel (px, py) -> World (x, z)
    # x = (px - center) * cell_size
    # z = (py - center) * cell_size
    
    # Optimization: Only scan within bounds
    # Convert bounds to pixels
    min_x_world = bounds[0][0]
    max_x_world = bounds[1][0]
    min_z_world = bounds[0][2]
    max_z_world = bounds[1][2]
    
    min_px = int(min_x_world / cell_size) + map_center
    max_px = int(max_x_world / cell_size) + map_center
    min_py = int(min_z_world / cell_size) + map_center
    max_py = int(max_z_world / cell_size) + map_center
    
    # Clamp to map size
    min_px = max(0, min_px)
    max_px = min(map_size_pixels, max_px)
    min_py = max(0, min_py)
    max_py = min(map_size_pixels, max_py)
    
    print(f"Scanning pixels x:[{min_px}, {max_px}], y:[{min_py}, {max_py}]")
    
    for py in range(min_py, max_py):
        for px in range(min_px, max_px):
            x = (px - map_center) * cell_size
            z = (py - map_center) * cell_size
            
            point = np.array([x, height, z])
            
            if pf.is_navigable(point):
                topdown_map[py, px] = 255 # FREE
                count_free += 1
                
    print(f"Generated map with {count_free} free cells.")
    return topdown_map

def main():
    if not os.path.exists(SCENE_ID):
        print(f"Scene file not found: {SCENE_ID}")
        return

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = SCENE_ID
    sim_cfg.gpu_device_id = 0
    sim_cfg.enable_physics = False
    
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    
    print("Initializing Simulator...")
    try:
        sim = habitat_sim.Simulator(cfg)
    except Exception as e:
        print(f"Failed to init sim: {e}")
        return

    print("Simulator initialized.")
    
    topdown_map = get_topdown_map(sim, CELL_SIZE, MAP_SIZE_METERS)
    
    if np.sum(topdown_map == 255) == 0:
        print("FAILURE: No free cells found!")
    else:
        print("SUCCESS: Free cells found.")
        # plt.imshow(topdown_map, cmap='gray')
        # plt.savefig("debug_map.png")
        # print("Map saved to debug_map.png")

    sim.close()

if __name__ == "__main__":
    main()
