
import torch
import os
import shutil
from worker import Worker
from model import PolicyNet
from parameter_clean import *

def run_multi_floor_demo():
    # Clean up previous gifs
    if os.path.exists(gifs_path):
        shutil.rmtree(gifs_path)
    os.makedirs(gifs_path)

    # Initialize device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # Initialize PolicyNet (random weights for demo)
    policy_net = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM).to(device)

    # Initialize Worker
    # meta_agent_id=0, global_step=0, save_image=True
    print(f"Using VLM: {USE_VLM}, model: {VLM_MODEL_NAME if USE_VLM else 'N/A'}")
    worker = Worker(
        0,
        policy_net,
        0,
        device=device,
        save_image=True,
        use_vlm=USE_VLM,
        vlm_model_name=VLM_MODEL_NAME,
    )

    print(f"Starting exploration on map: {worker.env.map_list[worker.env.map_index]}")
    print(f"Goal: Explore > {FLOOR_SWITCH_EXP_RATE*100}% and find stairs.")

    # Run episode
    worker.run_episode()

    print("Demo completed.")

if __name__ == "__main__":
    run_multi_floor_demo()
