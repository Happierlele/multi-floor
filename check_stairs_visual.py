
import matplotlib.pyplot as plt
from env import Env
import numpy as np
import os

def check_stairs_visual():
    # Force map index to point to 100.png
    # Env uses: map_index = episode_index % np.size(map_list)
    # BUT, Env has a loop that skips maps without robot start (208).
    # '100.png' does not have 208? Wait, setup_maps.py said: "Warning: No robot start (208) found in 100.png"
    # That is why Env is skipping 100.png and going to 104.png!
    
    # We need to fix 100.png to have a robot start position (208) if we want to test it.
    # Or find a map that HAS a robot start and ADD stairs to it.
    
    map_list = sorted(os.listdir('maps'))
    # Now that we fixed 100.png, we can test it directly.
    target_map = '100.png'
    if target_map in map_list:
        idx = map_list.index(target_map)
        print(f"Target map {target_map} is at index {idx}")
        env = Env(idx, plot=True)
        print(f"Robot cell: {env.robot_cell}")
        print(f"Stairs cell: {env.stairs_cell}")
        if np.array_equal(env.robot_cell, env.stairs_cell):
            print("ALERT: Robot and Stairs are at the SAME location!")
        else:
            dist = np.linalg.norm(np.array(env.robot_cell) - np.array(env.stairs_cell))
            print(f"Distance between robot and stairs: {dist}")
            
        print(f"Stairs coordinates found.")
        env.plot_env(0)
        plt.savefig('debug_stairs_visual.png')
    
    print(f"Current map: {env.map_list[env.map_index]}")
    print(f"Stairs cell: {env.stairs_cell}")
    print(f"Stairs coords: {env.stairs_coords}")
    
    if env.stairs_coords is None:
        print("ERROR: Stairs coordinates are None!")
    else:
        print("Stairs coordinates found.")
        
    plt.figure(figsize=(15, 5))
    env.plot_env(0)
    plt.savefig('debug_stairs_visual.png')
    print("Saved debug_stairs_visual.png")
    plt.close()

if __name__ == "__main__":
    check_stairs_visual()
