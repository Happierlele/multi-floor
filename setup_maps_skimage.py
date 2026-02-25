
import os
import numpy as np
from skimage import io
from parameter_clean import STAIRS_VALUE, MAX_STAIRS_PER_MAP

def add_stairs(map_name):
    map_path = os.path.join('maps', map_name)
    if not os.path.exists(map_path):
        print(f"Map {map_name} not found.")
        return

    # Read image using skimage as Env does
    # Handle both uint8 and float images
    img_raw = io.imread(map_path, 1)
    if img_raw.dtype == np.uint8:
        img = img_raw.astype(float)
    else:
        img = img_raw * 255.0
    
    print(f"Processing {map_name}. Min: {img.min()}, Max: {img.max()}")
    
    img[img == 180] = 194
    rows, cols = np.where(img == 208)
    if len(rows) == 0:
        robot_pos = None
    else:
        robot_pos = np.array([np.mean(rows), np.mean(cols)])
    free_rows, free_cols = np.where(img > 185)
    if len(free_rows) == 0:
        print(f"Could not find free space in {map_name}")
        return
    added = 0
    centers = []
    min_center_dist = 60.0
    segments = MAX_STAIRS_PER_MAP
    h, w = img.shape
    seg_w = max(1, w // segments)
    for seg in range(segments):
        x_start = seg * seg_w
        x_end = w if seg == segments - 1 else (seg + 1) * seg_w
        cx = (x_start + x_end) // 2
        cy = h // 2
        free_mask = img > 185
        free_rows, free_cols = np.where(free_mask[:, x_start:x_end])
        if len(free_rows) == 0:
            continue
        free_cols = free_cols + x_start
        free_points = np.column_stack((free_rows, free_cols))
        dists = np.linalg.norm(free_points - np.array([cy, cx]), axis=1)
        sorted_indices = np.argsort(dists)
        found = False
        rr, cc = 0, 0
        for idx in sorted_indices:
            r_try, c_try = free_points[idx]
            if r_try > h - 12 or c_try > w - 12 or r_try < 6 or c_try < 6:
                continue
            block = img[r_try-5:r_try+5, c_try-5:c_try+5]
            if not np.all(block > 185):
                continue
            if centers:
                dists_centers = np.linalg.norm(np.array(centers) - np.array([r_try, c_try]), axis=1)
                if np.min(dists_centers) < min_center_dist:
                    continue
            rr, cc = int(r_try), int(c_try)
            centers.append([rr, cc])
            found = True
            break
        if not found:
            continue
        img[rr:rr+4, cc:cc+4] = STAIRS_VALUE
        print(f"Added stairs to {map_name} at ({rr}, {cc})")
        added += 1

    # Save as uint8
    io.imsave(map_path, img.astype(np.uint8), check_contrast=False)
    print(f"Saved {map_name}")

if __name__ == "__main__":
    for name in sorted(os.listdir('maps')):
        if name.lower().endswith('.png'):
            add_stairs(name)
