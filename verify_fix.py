
import numpy as np
import torch
import os
import sys

# Mock classes to avoid full Habitat dependency
class MockNodeManager:
    def __init__(self):
        import quads
        self.nodes_dict = quads.QuadTree((0, 0), 1000, 1000)

class MockMapInfo:
    def __init__(self):
        self.map = np.zeros((100, 100))
        self.map_origin_x = 0
        self.map_origin_y = 0
        self.cell_size = 0.1

# Test GroundTruthNodeManager init
try:
    print("Testing GroundTruthNodeManager initialization...")
    from ground_truth_node_manager import GroundTruthNodeManager
    
    nm = MockNodeManager()
    gt_info = MockMapInfo() # Should be ignored
    
    gtnm = GroundTruthNodeManager(nm, gt_info, device='cpu')
    print("GroundTruthNodeManager initialized successfully.")
    
    # Test observation generation (should be empty/dummy but not crash)
    obs = gtnm.get_ground_truth_observation([0, 0])
    print(f"Observation shape: {len(obs)}")
    
except Exception as e:
    print(f"GroundTruthNodeManager Test Failed: {e}")
    import traceback
    traceback.print_exc()

# Test Utils Frontier
try:
    print("\nTesting get_frontier_in_map...")
    from utils import get_frontier_in_map, MapInfo
    
    # Create a simple map: 10x10
    # Center (5,5) is free (255)
    # Surround is unknown (127)
    map_data = np.ones((10, 10)) * 127
    map_data[4:7, 4:7] = 255
    
    info = MapInfo(map_data, 0, 0, 1.0)
    
    # Test without location
    f1 = get_frontier_in_map(info)
    print(f"Frontiers (no loc): {len(f1)}")
    
    # Test with location (center)
    f2 = get_frontier_in_map(info, robot_location=np.array([5.5, 5.5]))
    print(f"Frontiers (with loc): {len(f2)}")
    
except Exception as e:
    print(f"Utils Test Failed: {e}")
    import traceback
    traceback.print_exc()
