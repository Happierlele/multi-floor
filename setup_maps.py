
import os
import numpy as np
import cv2
from parameter_clean import STAIRS_VALUE, MAX_STAIRS_PER_MAP

def add_stairs(map_name):
    map_path = os.path.join('maps', map_name)
    if not os.path.exists(map_path):
        print(f"Map {map_name} not found.")
        return

    # Read image as grayscale
    img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to read {map_name}")
        return

    print(f"Processing {map_name}. Min: {img.min()}, Max: {img.max()}")
    
    # Check for robot start (208)
    if not np.any(img == 208):
        print(f"Warning: No robot start (208) found in {map_name}. Found: {np.unique(img)}")
    
    rows, cols = img.shape
    added = 0
    # Find free space (around 194 or > 150 but not 208)
    # Env logic: ground_truth > 150.
    # In the raw map, 194 is common free space.
    
    for r in range(rows // 4, rows - 10):
        if added >= MAX_STAIRS_PER_MAP:
            break
        for c in range(cols // 4, cols - 10):
            if added >= MAX_STAIRS_PER_MAP:
                break
            # Look for a 4x4 block of free space
            block = img[r:r+4, c:c+4]
            # Check if block is roughly uniform and "free" (e.g. 194)
            # We avoid 208 (robot) and 127 (unknown/wall?)
            if np.all(block == 194):
                img[r:r+4, c:c+4] = STAIRS_VALUE
                print(f"Added stairs to {map_name} at ({r}, {c})")
                added += 1
            
    if added > 0:
        cv2.imwrite(map_path, img)
        print(f"Saved {map_name}")
    else:
        print(f"Could not find free space in {map_name}")

if __name__ == "__main__":
    # Modify 100.png and 101.png
    add_stairs('100.png')
    add_stairs('101.png')
