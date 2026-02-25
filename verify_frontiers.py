import numpy as np
from utils import get_frontier_in_map, get_coords_from_cell_position
from parameter_clean import FREE, UNKNOWN, OCCUPIED, CELL_SIZE

class MockMapInfo:
    def __init__(self, map_data):
        self.map = map_data
        self.cell_size = CELL_SIZE
        self.map_origin_x = 0.0
        self.map_origin_y = 0.0

def test_frontier_extraction():
    # Create a 20x20 map
    # Center 10x10 is FREE
    # Surroundings are UNKNOWN
    map_data = np.full((20, 20), UNKNOWN, dtype=np.uint8)
    map_data[5:15, 5:15] = FREE
    
    # Add an obstacle inside
    map_data[8, 8] = OCCUPIED
    
    map_info = MockMapInfo(map_data)
    
    print("Testing get_frontier_in_map...")
    try:
        frontiers = get_frontier_in_map(map_info)
        print(f"Found {len(frontiers)} frontier points.")
        
        # Check if we have any points
        if len(frontiers) > 0:
            print("SUCCESS: Frontiers found.")
            # Print a few
            print("Sample frontiers:", list(frontiers)[:5])
        else:
            print("FAILURE: No frontiers found!")
            
    except NameError as e:
        print(f"CRITICAL FAILURE: NameError encountered: {e}")
        return
    except Exception as e:
        print(f"FAILURE: Other error: {e}")
        return

    # Test Agent integration
    try:
        from agent import Agent
        print("Testing Agent integration...")
        agent = Agent(policy_net=None, device='cpu')
        agent.updating_map_info = map_info
        agent.update_frontiers()
        print(f"Agent frontier count: {len(agent.frontier)}")
        if len(agent.frontier) == len(frontiers):
             print("SUCCESS: Agent integration works.")
        else:
             print("FAILURE: Agent frontier count mismatch.")
    except Exception as e:
        print(f"FAILURE: Agent integration crashed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_frontier_extraction()
