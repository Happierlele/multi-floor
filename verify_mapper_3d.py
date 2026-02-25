import numpy as np
import matplotlib.pyplot as plt
from habitat_mapper import SimpleMapper
from utils import get_grid_path

class MockState:
    def __init__(self, pos, rot_quat):
        self.position = pos
        self.rotation = rot_quat

class MockQuat:
    def __init__(self, x, y, z, w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

def test_mapper_and_pathfinding():
    print("Initializing SimpleMapper...")
    # Map size 20m x 20m, origin (-10, -10)
    mapper = SimpleMapper(cell_size=0.1, map_size_meters=20.0, origin_x=-10.0, origin_y=-10.0)
    global_map = mapper.reset()
    
    # 1. Simulate an obstacle
    # Add a wall at x=0, z=2.0 (Map coords)
    # Convert to indices
    # Map origin is (-10, -10). Center (0,0) is at index (100, 100)
    # Wall at (0, 2) -> index (100, 120)
    
    # Let's manually draw a wall in the map
    # Wall from (-5, 2) to (5, 2)
    start_wall_x = int((-5 - mapper.origin_x) / mapper.cell_size)
    end_wall_x = int((5 - mapper.origin_x) / mapper.cell_size)
    wall_y = int((2 - mapper.origin_y) / mapper.cell_size)
    
    global_map[wall_y, start_wall_x:end_wall_x] = 0 # Occupied
    
    print(f"Added wall at Y index {wall_y}, X range {start_wall_x}-{end_wall_x}")
    
    # 2. Test Pathfinding
    # Start: (0, 0) -> Index (100, 100)
    # End: (0, 4) -> Index (100, 140) (Behind the wall)
    
    start_idx = (100, 100)
    end_idx = (100, 140)
    
    print(f"Planning path from {start_idx} to {end_idx}...")
    path = get_grid_path(global_map, start_idx, end_idx)
    
    if path:
        print(f"Path found! Length: {len(path)}")
        # Verify path goes around the wall
        # Check mid-point of path
        mid_point = path[len(path)//2]
        print(f"Mid point of path: {mid_point}")
        
        # Visualize simplified
        # Expect X to deviate from 100 to go around wall (50 or 150)
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        print(f"Path X range: {min(xs)} - {max(xs)}")
        print(f"Path Y range: {min(ys)} - {max(ys)}")
        
        if min(xs) < start_wall_x or max(xs) >= end_wall_x:
             print("SUCCESS: Path goes around the wall.")
        else:
             print("WARNING: Path might be cutting through wall? (Check threshold)")
             print(f"Wall X range: {start_wall_x}-{end_wall_x}")
             print(f"Path X range: {min(xs)}-{max(xs)}")
             
             # Detailed check
             conflict = False
             for p in path:
                 if p[1] == wall_y and start_wall_x <= p[0] < end_wall_x:
                     print(f"CONFLICT: Point {p} is inside wall!")
                     conflict = True
                     break
             if not conflict:
                 print("Double Check: No actual point inside wall indices. Warning was boundary sensitivity.")
             
    else:
        print("FAILURE: No path found!")

if __name__ == "__main__":
    test_mapper_and_pathfinding()
