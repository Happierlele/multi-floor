import torch
import numpy as np
import os
from unittest.mock import MagicMock
from vlm_adapter import VLMAdapter

# Mock classes to simulate Agent and MapInfo
class MockMapInfo:
    def __init__(self):
        self.map = np.zeros((100, 100), dtype=int)
        self.map_origin_x = 0
        self.map_origin_y = 0
        self.cell_size = 0.4

class MockAgent:
    def __init__(self):
        self.map_info = MockMapInfo()
        self.location = np.array([5.0, 5.0])
        self.node_coords = np.array([[5.0, 5.0], [6.0, 6.0], [4.0, 4.0]]) # 3 nodes
        self.neighbor_indices = [1, 2] # Candidates are node 1 and 2

def test_vlm_adapter():
    agent = MockAgent()
    adapter = VLMAdapter()
    
    # Create a dummy observation (structure matches what agent.py expects roughly, but adapter only needs agent)
    # Adapter uses observation[4] (current_edge)
    # current_edge should be shape (1, K, 1) or similar.
    # But wait, in my vlm_adapter implementation:
    # current_edge = observation[4]
    # next_node_index = current_edge[0, chosen_index, 0].item()
    # So I need to mock observation[4] correctly.
    
    # Mocking observation list
    # Let's say we have 2 candidates.
    # current_edge tensor.
    # agent.neighbor_indices = [1, 2]
    # So current_edge should contain [1, 2] at the beginning.
    
    current_edge = torch.zeros((1, 25, 1), dtype=torch.long)
    current_edge[0, 0, 0] = 1
    current_edge[0, 1, 0] = 2
    
    observation = [None, None, None, None, current_edge, None]
    
    print("Testing VLM Adapter...")
    
    # Mock call_vlm_api to return "0" deterministically
    adapter.call_vlm_api = MagicMock(return_value="0")
    
    next_pos, action_idx = adapter.get_vlm_action(agent, observation)
    
    print(f"Action Index: {action_idx.item()}")
    print(f"Next Position: {next_pos}")
    
    assert action_idx.item() == 1 # VLM returns the node index (which is 1), not the index in candidate list
    assert np.allclose(next_pos, agent.node_coords[1]) # Node 1 is at index 0 of candidates
    
    print("Test Passed!")

if __name__ == "__main__":
    test_vlm_adapter()
