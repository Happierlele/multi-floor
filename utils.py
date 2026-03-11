import numpy as np
import imageio
import os
from skimage.morphology import label, binary_dilation, disk
import heapq

from parameter_clean import *


def get_frontier_in_map(map_info, location=None):
    """
    Extract global frontiers from the occupancy map.
    Frontiers are boundaries between FREE space and UNKNOWN space.
    """
    map_data = map_info.map
    # Free: > 200, Unknown: ~127 (100-150), Occupied: < 50
    
    # Binary masks
    free_mask = (map_data > 200)
    unknown_mask = (map_data > 100) & (map_data < 150)
    
    # Dilate free space to find overlap with unknown
    # Use a small disk kernel (radius 1)
    dilated_free = binary_dilation(free_mask, disk(1))
    
    # Frontier is the intersection
    frontier_mask = dilated_free & unknown_mask
    
    # Get coordinates
    frontier_indices = np.where(frontier_mask)
    frontier_cells = np.asarray([frontier_indices[1], frontier_indices[0]]).T
    
    # Convert to world coords
    frontier_coords = get_coords_from_cell_position(frontier_cells, map_info)
    
    # Return as set of tuples
    return set(map(tuple, frontier_coords))


def get_cell_position_from_coords(coords, map_info, check_negative=True):
    if coords is None:
        return np.empty((0, 2), dtype=int)
    coords = np.array(coords)
    single_cell = False
    
    # Robust handling for 2D/3D coordinates
    if coords.ndim == 1:
        if coords.shape[0] >= 2:
            coords = coords[:2].reshape(1, 2)
            single_cell = True
    elif coords.ndim == 2:
        if coords.shape[1] >= 2:
            coords = coords[:, :2]

    # Fallback/Safety check
    if coords.shape[1] != 2:
        # Try reshaping if it was flattened 2D array
        try:
             coords = coords.reshape(-1, 2)
        except:
             return np.empty((0, 2), dtype=int)

    coords_x = coords[:, 0]
    coords_y = coords[:, 1]
    cell_x = ((coords_x - map_info.map_origin_x) / map_info.cell_size)
    cell_y = ((coords_y - map_info.map_origin_y) / map_info.cell_size)

    cell_position = np.around(np.stack((cell_x, cell_y), axis=-1)).astype(int)

    if check_negative:
        # assert sum(cell_position.flatten() >= 0) == cell_position.flatten().shape[0], print(cell_position, coords, map_info.map_origin_x, map_info.map_origin_y)
        pass # Relax assertion for robustness
    if single_cell:
        return cell_position[0]
    else:
        return cell_position


def get_coords_from_cell_position(cell_position, map_info):
    cell_position = cell_position.reshape(-1, 2)
    cell_x = cell_position[:, 0]
    cell_y = cell_position[:, 1]
    coords_x = cell_x * map_info.cell_size + map_info.map_origin_x
    coords_y = cell_y * map_info.cell_size + map_info.map_origin_y
    coords = np.stack((coords_x, coords_y), axis=-1)
    coords = np.around(coords, 1)
    if coords.shape[0] == 1:
        return coords[0]
    else:
        return coords


def get_free_area_coords(map_info):
    free_indices = np.where(map_info.map == FREE)
    free_cells = np.asarray([free_indices[1], free_indices[0]]).T
    free_coords = get_coords_from_cell_position(free_cells, map_info)
    return free_coords


def get_free_and_connected_map(location, map_info):
    # a binary map for free and connected areas
    free = (map_info.map == FREE).astype(float)
    labeled_free = label(free, connectivity=2)
    cell = get_cell_position_from_coords(location, map_info)
    
    # Safety check: if cell is out of bounds or not in free area
    h, w = labeled_free.shape
    cx, cy = cell[0], cell[1]
    
    if 0 <= cy < h and 0 <= cx < w:
        label_number = labeled_free[cy, cx]
        if label_number > 0:
            connected_free_map = (labeled_free == label_number)
            return connected_free_map

    # Fallback: Return all free areas if robot is somehow in obstacle/unknown
    return free > 0


def get_updating_node_coords(location, updating_map_info, check_connectivity=True):
    x_min = updating_map_info.map_origin_x
    y_min = updating_map_info.map_origin_y
    x_max = updating_map_info.map_origin_x + (updating_map_info.map.shape[1] - 1) * CELL_SIZE
    y_max = updating_map_info.map_origin_y + (updating_map_info.map.shape[0] - 1) * CELL_SIZE

    if x_min % NODE_RESOLUTION != 0:
        x_min = (x_min // NODE_RESOLUTION + 1) * NODE_RESOLUTION
    if x_max % NODE_RESOLUTION != 0:
        x_max = x_max // NODE_RESOLUTION * NODE_RESOLUTION
    if y_min % NODE_RESOLUTION != 0:
        y_min = (y_min // NODE_RESOLUTION + 1) * NODE_RESOLUTION
    if y_max % NODE_RESOLUTION != 0:
        y_max = y_max // NODE_RESOLUTION * NODE_RESOLUTION

    x_coords = np.arange(x_min, x_max + 0.1, NODE_RESOLUTION)
    y_coords = np.arange(y_min, y_max + 0.1, NODE_RESOLUTION)
    t1, t2 = np.meshgrid(x_coords, y_coords)
    nodes = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
    nodes = np.around(nodes, 1)

    free_connected_map = None

    if not check_connectivity:

        indices = []
        nodes_cells = get_cell_position_from_coords(nodes, updating_map_info).reshape(-1, 2)
        for i, cell in enumerate(nodes_cells):
            assert 0 <= cell[1] < updating_map_info.map.shape[0] and 0 <= cell[0] < updating_map_info.map.shape[1]
            if updating_map_info.map[cell[1], cell[0]] > 200:
                indices.append(i)
        indices = np.array(indices, dtype=int)
        nodes = nodes[indices].reshape(-1, 2)

    else:
        free_connected_map = get_free_and_connected_map(location, updating_map_info)
        free_connected_map = np.array(free_connected_map)
        
        indices = []
        nodes_cells = get_cell_position_from_coords(nodes, updating_map_info).reshape(-1, 2)
        h, w = free_connected_map.shape
        
        for i, cell in enumerate(nodes_cells):
            cx, cy = cell[0], cell[1]
            if 0 <= cx < w and 0 <= cy < h:
                if free_connected_map[cy, cx]:
                    indices.append(i)
                    
        if len(indices) > 0:
            indices = np.array(indices, dtype=int)
            nodes = nodes[indices].reshape(-1, 2)
        else:
            nodes = np.empty((0, 2))

    return nodes, free_connected_map


def check_collision(x0, y0, x1, y1, map_data, threshold=128):
    """
    Bresenham's line algorithm to check for obstacles between two points.
    Returns True if collision detected.
    """
    x0, y0 = int(round(x0)), int(round(y0))
    x1, y1 = int(round(x1)), int(round(y1))
    
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            if 0 <= y < map_data.shape[0] and 0 <= x < map_data.shape[1]:
                # Assume values < threshold are occupied (0=occupied, 255=free)
                # UNKNOWN is 127. So < 100 is definitely occupied.
                if map_data[y, x] < 100: 
                    return True
            x += sx
            err -= dy
            if err < 0:
                y += sy
                err += dx
    else:
        err = dy / 2.0
        while y != y1:
            if 0 <= y < map_data.shape[0] and 0 <= x < map_data.shape[1]:
                if map_data[y, x] < 100:
                    return True
            y += sy
            err -= dx
            if err < 0:
                x += sx
                err += dy
                
    return False

def get_grid_path(map_data, start, end, free_thresh=100):
    """
    A* algorithm for grid map.
    map_data: 2D numpy array (0=Occupied, 255=Free, 127=Unknown)
    start: (x, y) tuple (indices)
    end: (x, y) tuple (indices)
    Returns: list of (x, y) tuples or None
    """
    rows, cols = map_data.shape
    start = (int(start[0]), int(start[1]))
    end = (int(end[0]), int(end[1]))
    
    if not (0 <= start[0] < cols and 0 <= start[1] < rows): return None
    if not (0 <= end[0] < cols and 0 <= end[1] < rows): return None
    
    # Priority Queue: (f_score, x, y)
    open_set = []
    heapq.heappush(open_set, (0, start[0], start[1]))
    
    came_from = {}
    g_score = {start: 0}
    f_score = {start: abs(start[0]-end[0]) + abs(start[1]-end[1])} # Manhattan heuristic
    
    visited = set()
    
    while open_set:
        current_f, cx, cy = heapq.heappop(open_set)
        current = (cx, cy)
        
        if current == end:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1] # Reverse
        
        if current in visited:
            continue
        visited.add(current)
        
        # 8-connectivity
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
            nx, ny = cx + dx, cy + dy
            neighbor = (nx, ny)
            
            if 0 <= nx < cols and 0 <= ny < rows:
                cell = map_data[ny, nx]
                if cell < 100:
                    continue
                base = 1.414 if dx!=0 and dy!=0 else 1.0
                terrain = 1.0 if cell >= 200 else 3.0
                penalty = 0.0
                if nx-1 >= 0 and map_data[ny, nx-1] < 100: penalty += 2.0
                if nx+1 < cols and map_data[ny, nx+1] < 100: penalty += 2.0
                if ny-1 >= 0 and map_data[ny-1, nx] < 100: penalty += 2.0
                if ny+1 < rows and map_data[ny+1, nx] < 100: penalty += 2.0
                tentative_g = g_score[current] + base * terrain + penalty
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + abs(nx-end[0]) + abs(ny-end[1])
                    f_score[neighbor] = f
                    heapq.heappush(open_set, (f, nx, ny))
                    
    return None # No path found


def make_gif(path, n, frame_files, rate):
    if not frame_files:
        return
    try:
        with imageio.get_writer('{}/{}_explored_rate_{:.4g}.gif'.format(path, n, rate), mode='I', duration=0.5) as writer:
            for frame in frame_files:
                if os.path.exists(frame):
                    image = imageio.imread(frame)
                    writer.append_data(image)
        print('gif complete\n')
    except Exception as e:
        print(f"Error making GIF: {e}")

    # Remove files
    for filename in frame_files[:-1]:
        if os.path.exists(filename):
            os.remove(filename)


class MapInfo:
    def __init__(self, map, map_origin_x, map_origin_y, cell_size):
        self.map = map
        self.map_origin_x = map_origin_x
        self.map_origin_y = map_origin_y
        self.cell_size = cell_size

    def update_map_info(self, map, map_origin_x, map_origin_y):
        self.map = map
        self.map_origin_x = map_origin_x
        self.map_origin_y = map_origin_y
