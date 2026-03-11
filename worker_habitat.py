import os
import sys
import ctypes

# -------------------------------------------------------------------------
# CRITICAL FIX: Environment Setup BEFORE any habitat_sim import
# This ensures EGL is selected and libraries are visible to the loader.
# -------------------------------------------------------------------------

# 1. Preload libstdc++ (Handled by Driver/Ray LD_PRELOAD)
# No manual action needed here.

# 2. Configure EGL/Headless Environment
print("[Worker] Configuring Environment for GLES/EGL Headless Rendering...", flush=True)

# Force EGL/GLES environment
os.environ["MAGNUM_TARGET_GLES"] = "1"
os.environ["MAGNUM_TARGET_HEADLESS"] = "1"
os.environ["MAGNUM_TARGET_EGL"] = "1"

# Try explicit device platform to fix RGB black screen
os.environ["EGL_PLATFORM"] = "device"
print("[Worker] Env: GLES=1, HEADLESS=1, EGL=1, EGL_PLATFORM=device", flush=True)

# Enable Verbose Logging
os.environ["MAGNUM_LOG"] = "verbose"
os.environ["GLOG_minloglevel"] = "0"
os.environ["HABITAT_SIM_LOG"] = "verbose"

# FORCE NVIDIA DRIVER
nvidia_json = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
if os.path.exists(nvidia_json):
    os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = nvidia_json

# CRITICAL FIX: Ray sets CUDA_VISIBLE_DEVICES.
# We MUST unset it so EGL can see the physical GPU.
if "CUDA_VISIBLE_DEVICES" in os.environ:
   cvd = os.environ["CUDA_VISIBLE_DEVICES"]
   print(f"[Worker] Ray assigned CUDA_VISIBLE_DEVICES: {cvd}", flush=True)
   
   # Attempt to preserve the physical device ID for Habitat
   parts = cvd.split(',')
   if len(parts) > 0 and parts[0].strip().isdigit():
       phys_id = parts[0].strip()
       os.environ["HABITAT_DEVICE_ID"] = phys_id
       print(f"[Worker] Inferred Physical GPU ID for Habitat: {phys_id}", flush=True)
       
   del os.environ["CUDA_VISIBLE_DEVICES"]
   print("[Worker] Unset CUDA_VISIBLE_DEVICES to allow EGL to discover all GPUs.", flush=True)

# Disable X11 usage entirely
if "DISPLAY" in os.environ:
    del os.environ["DISPLAY"]
    print("[Worker] Removed DISPLAY environment variable.", flush=True)

# LD_LIBRARY_PATH: Handled by Driver/Ray. Do NOT modify here to avoid conflicts.
print(f"[Worker] Using LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '')}", flush=True)

# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# Imports that depend on the above configuration
import habitat_sim # NOW it is safe to import
import torch
import matplotlib
matplotlib.use('Agg') # Force headless backend for matplotlib
import matplotlib.pyplot as plt
import random
import numpy as np
import quads
from copy import deepcopy

from habitat_env import HabitatEnv
from agent import Agent
from utils import *
from model import PolicyNet, StairSwitchNet
from ground_truth_node_manager import GroundTruthNodeManager
from vlm_adapter import VLMAdapter
from parameter_clean import *

if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)


class Worker:
    def __init__(self, meta_agent_id, policy_net, stair_switch_net, global_step=0, device='cpu', save_image=False, use_vlm=False, vlm_model_name="gpt-4o", vlm_api_key=None, vlm_base_url=None, env=None):
        # Configuration already handled at module level
        self.meta_agent_id = meta_agent_id
        self.device = device
        self.use_vlm = use_vlm
        print(f"[Worker {self.meta_agent_id}] Initialized with USE_VLM={self.use_vlm}", flush=True)
        
        # Initialize Environment ONCE
        # We pass global_step=0 initially, it will be updated in reset_episode
        if env is not None:
            self.env = env
            # Ensure env is configured correctly if passed in
            self.env.episode_index = global_step
            self.env.plot = False # Force disable plotting to prevent core dumps
        else:
            self.env = HabitatEnv(0, plot=save_image) # Enable plotting if requested
            
        # Do not overwrite save_image to False
        self.save_image = save_image
        if not self.save_image:
             print("[Worker] Plotting/GIF disabled.", flush=True)
        else:
             print("[Worker] Plotting/GIF ENABLED.", flush=True)
        
        self.robot = Agent(policy_net, self.device, save_image)
        
        self.stair_switch_net = stair_switch_net
        self.stair_switch_net.eval()

        # Move models to CPU within the worker to avoid CUDA initialization in subprocess
        # This is a critical fix for "Worker unexpectedly exits with a connection error code 2"
        # when Ray tries to serialize/deserialize CUDA tensors across processes.
        if self.device == 'cpu' or self.device == torch.device('cpu'):
            self.robot.policy_net.to('cpu')
            self.stair_switch_net.to('cpu')

        self.ground_truth_node_manager = GroundTruthNodeManager(self.robot.node_manager, self.env.ground_truth_info,
                                                                device=self.device, plot=save_image)

        self.multi_floor_enabled = True
        self.inter_floor_active = False
        self.inter_floor_start_step = None
        self.target_stairs_index = None
        self.position_history = []
        self.stuck_events = 0
        self.stair_transitions = 0
        self.steps_on_current_floor = 0
        self.tabu_frontiers = [] # Stores recently visited frontiers to prevent cycles

        if self.use_vlm:
            key = vlm_api_key or os.getenv("QWEN_API_KEY")
            default_base = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1" 
            base = vlm_base_url or os.getenv("QWEN_BASE_URL") or default_base
            if not key:
                print("[Worker] QWEN_API_KEY not set. Disabling VLM.", flush=True)
                self.use_vlm = False
                self.vlm = None
            else:
                print(f"[Worker] Initializing VLM with Key: {key[:6]}...{key[-4:]}, Base URL: {base}")
                self.vlm = VLMAdapter(model_name=vlm_model_name, api_key=key, base_url=base)

        self.episode_buffer = []
        self.perf_metrics = dict()
        for i in range(27):
            self.episode_buffer.append([])
            
        # Initialize state for first run
        self.reset_episode(global_step, save_image)

    def reset_episode(self, global_step, save_image):
        """
        Reset the worker state for a new episode without re-initializing the Simulator.
        """
        self.global_step = global_step
        self.save_image = save_image
        
        # Update Environment config
        self.env.episode_index = global_step
        self.env.plot = save_image
        
        # Reset Environment (re-uses existing Simulator)
        self.env.reset()
        
        # Reset Robot
        self.robot.save_image = save_image
        self.robot.reset_for_new_map()
        self.robot.update_planning_state(self.env.belief_info, self.env.robot_location, self.env.floor_id)
        
        # Reset GroundTruthNodeManager
        self.ground_truth_node_manager.plot = save_image
        # Re-init GT manager with new env info if needed, or just clear its state
        # Usually GT manager holds reference to robot.node_manager, so it might need a harder reset if map changes
        # But here we assume same map/scene for now, just new start pos.
        # If map changes, we might need deeper reset.
        
        # Reset Worker State
        self.inter_floor_active = False
        self.inter_floor_start_step = None
        self.target_stairs_index = None
        self.position_history = []
        self.stuck_events = 0
        self.stair_transitions = 0
        self.steps_on_current_floor = 0
        self.zero_utility_steps = 0 # Allow model to try for a while before forcing
        self.tabu_frontiers = []
        self.tabu_exploration_targets = [] # Tabu list for forced targets to prevent loops
        self.current_exploration_target = None # Persist target for multi-step travel
        self.current_vlm_target = None # Persist VLM target for consistent navigation
        self.last_target_distance = np.inf
        self.min_target_distance = np.inf
        
        self.episode_buffer = []
        for i in range(27):
            self.episode_buffer.append([])
        self.perf_metrics = dict()
        
        # Exploration Gain Tracking
        self.last_explored_pixel_count = 0
        self.last_check_step = 0
        self.recent_exploration_gain = 1.0

    def select_target_stairs(self):
        """
        Select the best stairs to transition to.
        Prioritizes discovered stairs that are closest.
        """
        if not self.env.stairs_coords_list:
            return None
            
        # Filter for discovered stairs
        candidates = []
        if hasattr(self.env, 'discovered_stairs') and self.env.discovered_stairs:
            candidates = list(self.env.discovered_stairs)
        else:
            # If no stairs discovered but model wants to switch, 
            # we could cheat and pick the closest one (or return None to force exploration)
            # For robustness, let's pick the closest known one if we are debugging,
            # but strict logic should require discovery.
            # Let's return None to force agent to find stairs first.
            # print("Worker: Model wants to switch floors but no stairs discovered yet.")
            return None

        # Find closest candidate
        best_idx = None
        min_dist = 1e9
        
        for idx in candidates:
            coords = self.env.stairs_coords_list[idx]
            dist = np.linalg.norm(self.env.robot_location - np.array(coords))
            if dist < min_dist:
                min_dist = dist
                best_idx = idx
                
        return best_idx

    def perform_stair_transition(self, step, stairs_idx):
        """
        Execute floor transition logic.
        """
        print(f"\n{'='*50}\n[Worker {self.meta_agent_id}] >>> FLOOR TRANSITION INITIATED <<<\nStep: {step} | Stairs Index: {stairs_idx}\n{'='*50}\n", flush=True)
        print(f"[Worker {self.meta_agent_id}] Performing Stair Transition at Step {step} via Stair {stairs_idx}!", flush=True)
        self.stair_transitions += 1
        
        # Capture old state for graph connection
        old_floor_id = self.env.floor_id
        old_location = np.copy(self.env.robot_location)
        
        # 1. Update Floor ID
        self.env.floor_id += 1
        self.steps_on_current_floor = 0
        
        # 2. Reset Semantic Map (Clear 2D memory of previous floor)
        self.env.reset_semantic_map()
        
        # 3. Teleport Robot to New Floor (Simulated)
        # In a real multi-floor mesh, we would move to the linked exit point.
        # Here we simulate by randomizing position on the new "floor" 
        # (which is just a re-run on the navmesh with cleared memory, or we could try to shift height)
        
        # We try to shift height to simulate physical floor change if the mesh supports it
        # But since we use random navigable points, let's just teleport to a random point
        # to simulate arriving at a new area.
        if self.env.sim and self.env.sim.pathfinder.is_loaded:
             try:
                 target_pos = self.env.sim.pathfinder.get_random_navigable_point()
                 # Optional: Bias height?
                 # target_pos[1] += 3.0 * self.env.floor_id 
                 # (Only if mesh is actually multi-floor tall, otherwise we might go out of bounds)
                 
                 new_state = habitat_sim.AgentState()
                 new_state.position = target_pos
                 import quaternion
                 angle = np.random.uniform(0, 2 * np.pi)
                 new_state.rotation = quaternion.from_rotation_vector(np.array([0, angle, 0]))
                 self.env.sim.get_agent(0).set_state(new_state)
                 
                 # Update env state
                 obs = self.env.sim.get_sensor_observations()
                 agent_state = self.env.sim.get_agent(0).get_state()
                 self.env.robot_belief = self.env.mapper.reset() # Ensure mapper is clean
                 self.env.robot_belief = self.env.mapper.update(obs['depth_sensor'], agent_state.sensor_states['depth_sensor'], agent_state)
                 self.env.update_robot_location_from_sim(agent_state)
                 
                 print(f"[Worker] Teleported to new floor {self.env.floor_id} at {target_pos}", flush=True)
                 
                 # 5. Regenerate Ground Truth for the new floor
                 # Use the robot's new height to slice the NavMesh
                 self.env.update_ground_truth_for_floor(height=agent_state.position[1])
                 
                 # Add explicit graph connection (Stair Edge)
                 self.robot.node_manager.add_stair_connection(old_floor_id, old_location, self.env.floor_id, self.env.robot_location)
             except Exception as e:
                 print(f"[Worker] Error during floor transition: {e}", flush=True)

        # 4. Reset Robot Planning State
        # Preserve memory (graph) for multi-floor navigation
        self.robot.reset_for_new_map(preserve_memory=True) 
        self.robot.update_planning_state(self.env.belief_info, self.env.robot_location, self.env.floor_id)
        
        # 5. Regenerate Ground Truth for the new floor (Simulated)
        # We pass a fictitious height or just regenerate
        # Assuming single-level mesh for now, we just regenerate to match current area
        # If we had a real multi-story mesh, we would pass agent_state.position[1]
        current_height = self.env.sim.get_agent(0).get_state().position[1]
        self.env.ground_truth = self.env._get_ground_truth_map(floor_height=current_height)
        self.env.ground_truth_info.update_map_info(self.env.ground_truth, self.env.belief_origin_x, self.env.belief_origin_y)
        
        # 6. Re-init Ground Truth Manager
        self.ground_truth_node_manager = GroundTruthNodeManager(self.robot.node_manager, self.env.ground_truth_info,
                                                                device=self.device, plot=self.save_image)

        # 7. Generate new stairs for this floor
        self.env._generate_stairs()
        
        # Reset control flags
        self.inter_floor_active = False
        self.target_stairs_index = None
        self.env.current_target_stairs_index = None
        self.position_history = []
        
    def run_episode(self):
        # reset_episode should be called before run_episode by the runner
        # But we also do some init here just in case
        
        done = False
        self.stair_switch_buffer = [] 
        observation = self.robot.get_observation()
        ground_truth_observation = self.ground_truth_node_manager.get_ground_truth_observation(self.env.robot_location)
        
        # Initialize curr_max_util here to avoid UnboundLocalError
        curr_max_util = 0.0

        if self.save_image:
            # self.robot.plot_env() # Agent does not have plot_env
            self.ground_truth_node_manager.plot_ground_truth_env(self.env.robot_location, self.position_history, self.env.floor_id, self.env.stairs_coords_list)
            self.env.plot_env(0)

        for i in range(MAX_EPISODE_STEP):
            if i % 10 == 0:
                print(f"Step {i}/{MAX_EPISODE_STEP} on floor {self.env.floor_id}", flush=True)
            self.steps_on_current_floor += 1
            self.save_observation(observation, ground_truth_observation)

            # Periodic Plotting (Matches 2D behavior)
            if self.save_image and i % SAVE_IMG_GAP == 0:
                 # 1. Plot and Save Ground Truth Graph (Debug View)
                 try:
                     plt.switch_backend('agg')
                     plt.figure(figsize=(10, 10))
                     self.ground_truth_node_manager.plot_ground_truth_env(self.env.robot_location, self.position_history, self.env.floor_id, self.env.stairs_coords_list)
                     graph_filename = '{}/{}_{}_graph.png'.format(gifs_path, self.global_step, i)
                     plt.savefig(graph_filename, dpi=100)
                     plt.close()
                 except Exception as e:
                     print(f"Warning: Graph plotting failed: {e}")

                 # 2. Plot and Save Environment View (Map + Camera)
                 self.env.plot_env(i)

            force_global_plan = False
            is_rescue_move = False
            next_location = None
            action_index = torch.tensor([[0]]).long().to(self.device)
            
            if len(self.position_history) >= 20: 
                recent_pos = self.position_history[-20:]
                xs = [p[0] for p in recent_pos]
                ys = [p[1] for p in recent_pos]
                if (max(xs) - min(xs) < 2.0) and (max(ys) - min(ys) < 2.0):
                     force_global_plan = True
                     print(f"Worker {self.meta_agent_id}: Detected position stagnation (20 steps in <2m box). Forcing rescue.", flush=True)

            if i > 10 and len(self.position_history) > 0:
                last_pos = np.array(self.position_history[-1])
                curr_loc = np.array(self.robot.location)
                # Compare 2D distance only, ignoring Z or floor_id
                if np.linalg.norm(curr_loc[:2] - last_pos[:2]) < 0.05:
                    self.stuck_events += 1
                else:
                    self.stuck_events = 0
                if self.stuck_events > 5: 
                    force_global_plan = True
                    print(f"Worker {self.meta_agent_id}: Detected movement stagnation (5 steps no move). Forcing rescue.", flush=True)
                    self.stuck_events = 0
            else:
                self.stuck_events = 0

            global_override_success = False
            
            if force_global_plan:
                target_node = None
                rescue_mode = ""
                
                candidates = []
                for node in self.robot.node_manager.nodes_dict.__iter__():
                    node_floor = getattr(node.data, 'floor_id', self.env.floor_id)
                    if node_floor != self.env.floor_id:
                        continue
                    candidates.append(node.data)

                if not candidates:
                    print(f"Worker {self.meta_agent_id}: No rescue candidates found!", flush=True)
                else:
                    rand_val = random.random()
                    
                    if rand_val < 0.5:
                        frontiers = [n for n in candidates if n.utility > 0]
                        if frontiers:
                            target_node = random.choice(frontiers)
                            rescue_mode = "RandomFrontier"
                    
                    if target_node is None and rand_val < 0.8:
                        target_node = max(candidates, key=lambda x: x.utility)
                        if target_node.utility <= 0: target_node = None 
                        else: rescue_mode = "MaxUtil"
                    
                    if target_node is None:
                        target_node = max(candidates, key=lambda x: np.linalg.norm(x.coords - self.robot.location))
                        rescue_mode = "Farthest"

                if target_node is not None:
                    start_key = (round(float(self.robot.location[0]), 1), round(float(self.robot.location[1]), 1))
                    start_node = self.robot.node_manager.nodes_dict.find(start_key)
                    if start_node is None:
                        gf = getattr(self.env, "global_frontiers", [])
                        gf_coords = []
                        try:
                            for it in gf:
                                if isinstance(it, (list, tuple)) and len(it) == 2 and isinstance(it[0], (list, tuple, np.ndarray)) and np.isscalar(it[1]):
                                    gf_coords.append(np.array(it[0], dtype=float))
                                elif isinstance(it, (list, tuple)) and len(it) == 2 and np.isscalar(it[0]) and np.isscalar(it[1]):
                                    gf_coords.append(np.array([float(it[0]), float(it[1])], dtype=float))
                        except Exception:
                            gf_coords = []
                        self.robot.node_manager.add_node_to_dict(self.robot.location, gf_coords, self.env.belief_info, self.env.floor_id)
                        bbox = quads.BoundingBox(
                            min_x=self.robot.location[0]-3.0, min_y=self.robot.location[1]-3.0,
                            max_x=self.robot.location[0]+3.0, max_y=self.robot.location[1]+3.0
                        )
                        nearby = self.robot.node_manager.nodes_dict.within_bb(bbox)
                        start_node = self.robot.node_manager.nodes_dict.find(start_key)
                        if start_node:
                            curr = start_node.data
                            cand = []
                            for w in nearby:
                                n = w.data
                                if n.coords.tolist() == curr.coords.tolist():
                                    continue
                                d = np.linalg.norm(n.coords - curr.coords)
                                cand.append((d, n))
                            cand.sort(key=lambda x: x[0])
                            for d, n in cand[:3]:
                                curr.neighbor_set.add((n.coords[0], n.coords[1]))
                                n.neighbor_set.add((curr.coords[0], curr.coords[1]))
                    path, length = self.robot.node_manager.a_star(self.robot.location, target_node.coords)
                    if path and length < 1e8:
                        next_target = np.array(path[0])
                        if next_target.shape[0] == 3:
                            next_target = next_target[1:]

                        neighbor_indices = self.robot.neighbor_indices
                        neighbor_coords = self.robot.node_coords[neighbor_indices]
                        dists = np.linalg.norm(neighbor_coords - next_target, axis=1)
                        best_idx = np.argmin(dists)

                        if dists[best_idx] < 2.0:
                            next_location = neighbor_coords[best_idx]
                            action_index = torch.tensor([[best_idx]]).long()
                            global_override_success = True
                            is_rescue_move = True
                        else:
                            print(f"Worker {self.meta_agent_id}: Rescue A* path blocked (next step too far). Trying fallback.", flush=True)

                if not global_override_success:
                    key = (round(float(self.robot.location[0]), 1), round(float(self.robot.location[1]), 1))
                    curr_node = self.robot.node_manager.nodes_dict.find(key)
                    
                    if curr_node is None:
                        bbox_size = 0.5
                        found_nodes = self.robot.node_manager.nodes_dict.within_bb(quads.BoundingBox(
                            min_x=self.robot.location[0]-bbox_size, min_y=self.robot.location[1]-bbox_size,
                            max_x=self.robot.location[0]+bbox_size, max_y=self.robot.location[1]+bbox_size
                        ))
                        if found_nodes:
                            curr_node = min(found_nodes, key=lambda n: np.linalg.norm(np.array(n.data.coords) - self.robot.location))
                    
                    if curr_node:
                        neighbors = list(curr_node.data.neighbor_set)
                        recent_positions = [np.array(p[:2]) for p in self.position_history[-20:]] if len(self.position_history) > 0 else []
                        def not_recent(nc):
                            if not recent_positions: return True
                            for rp in recent_positions:
                                if np.linalg.norm(np.array(nc) - rp) < 1.5:
                                    return False
                            return True
                        valid_neighbors = [n for n in neighbors if np.linalg.norm(np.array(n) - self.robot.location) > 0.1 and not_recent(n)]
                        random.shuffle(valid_neighbors)
                        
                        for neighbor in valid_neighbors:
                            rn_pt = np.array(neighbor)
                            path_check = self.env.get_shortest_path(self.robot.location, rn_pt)
                            if path_check and len(path_check) > 1:
                                print(f"Worker {self.meta_agent_id}: Rescue Fallback - Moving to neighbor {rn_pt}", flush=True)
                                
                                target_pt = path_check[1]
                                neighbor_indices = self.robot.neighbor_indices
                                neighbor_coords = self.robot.node_coords[neighbor_indices]
                                dists = np.linalg.norm(neighbor_coords - target_pt, axis=1)
                                best_idx = np.argmin(dists)
                                
                                if dists[best_idx] < 1.0:
                                    next_location = neighbor_coords[best_idx]
                                    action_index = torch.tensor([[best_idx]]).long()
                                    global_override_success = True
                                    is_rescue_move = True
                                    break

            # Fast Follow for Persistent Exploration Target (Frontier Push / Forced / Random)
            if not global_override_success and self.current_exploration_target is not None:
                try:
                    dist_to_target = np.linalg.norm(self.robot.location - self.current_exploration_target)
                    if dist_to_target < 2.0:
                        self.tabu_exploration_targets.append(self.current_exploration_target)
                        if len(self.tabu_exploration_targets) > 5:
                            self.tabu_exploration_targets.pop(0)
                        print(f"Worker {self.meta_agent_id}: Reached Persistent Exploration Target (Dist < 2.0m). Clearing.", flush=True)
                        self.current_exploration_target = None
                    else:
                        path = self.env.get_shortest_path(self.robot.location, self.current_exploration_target)
                        if path and len(path) > 1:
                            next_location = path[1]
                            is_rescue_move = True
                            global_override_success = True
                        else:
                            print(f"Worker {self.meta_agent_id}: Path to Persistent Target blocked. Tabu and clear.", flush=True)
                            self.tabu_exploration_targets.append(self.current_exploration_target)
                            if len(self.tabu_exploration_targets) > 5:
                                self.tabu_exploration_targets.pop(0)
                            self.current_exploration_target = None
                except Exception as e:
                    print(f"Worker {self.meta_agent_id}: Persistent Target follow failed: {e}", flush=True)
            
            # VLM Logic with Persistence (User Request: Fix "Ridiculous Line" by avoiding split-brain)
            if not global_override_success and self.use_vlm:
                # 1. Check if we have a valid persistent target
                if self.current_vlm_target is not None:
                    dist = np.linalg.norm(self.robot.location - self.current_vlm_target)
                    if dist < 1.0:
                        print(f"Worker {self.meta_agent_id}: Reached VLM Target. Clearing.", flush=True)
                        self.current_vlm_target = None
                
                # 2. Call VLM if no target (or target reached/invalid)
                if self.current_vlm_target is None:
                    rgb_img = getattr(self.env, "rgb_image", None)
                    
                    # Only save debug image on new VLM calls
                    if i == 0 and rgb_img is not None:
                        try:
                            from PIL import Image
                            debug_img_path = f"debug_view_agent{self.meta_agent_id}_step0.png"
                            Image.fromarray(rgb_img).save(debug_img_path)
                            print(f"Worker {self.meta_agent_id}: SAVED DEBUG 3D VIEW to {debug_img_path}", flush=True)
                        except Exception as e:
                            print(f"Worker {self.meta_agent_id}: Failed to save debug image: {e}", flush=True)

                    print(f"Worker {self.meta_agent_id}: Calling VLM navigation logic (Step {i})...", flush=True)
                    debug_path = f"{gifs_path}/vlm_nav_{self.meta_agent_id}_step_{i}.png"
                    
                    next_location, action_index = self.vlm.get_vlm_action(
                        self.robot, 
                        observation, 
                        stairs_coords=self.env.stairs_coords_list, 
                        rgb_image=rgb_img, 
                        debug_save_path=debug_path,
                        position_history=self.position_history,
                        current_floor=self.env.floor_id
                    )
                    
                    if next_location is not None:
                        self.current_vlm_target = next_location
                    else:
                        # VLM failed to return a target. Fallback to existing heuristic logic.
                        neighbor_indices = self.robot.neighbor_indices
                        if neighbor_indices is not None and len(neighbor_indices) > 0:
                            candidate_indices = np.array(neighbor_indices, dtype=int)
                            coords_arr = np.array([self.robot.node_coords[int(i)] for i in candidate_indices])
                            mask = np.linalg.norm(coords_arr - self.robot.location, axis=1) > 0.1
                            candidate_indices = candidate_indices[mask]
                            if candidate_indices.size > 0:
                                node_manager = getattr(self.robot, "node_manager", None)
                                visit_counts = {}
                                for idx2 in candidate_indices:
                                    c = self.robot.node_coords[int(idx2)]
                                    key = (round(float(c[0]), 1), round(float(c[1]), 1))
                                    node = node_manager.nodes_dict.find(key) if node_manager else None
                                    visit_counts[int(idx2)] = node.data.visit_count if node else 0
                                recent_xy = [tuple(np.array(p).flatten()[:2]) for p in self.position_history[:-1][-10:]]
                                def is_recent_xy(c):
                                    cx, cy = c[0], c[1]
                                    for p in recent_xy:
                                        if np.linalg.norm(np.array([cx, cy]) - np.array(p)) < 1.0:
                                            return True
                                    return False
                                filtered = []
                                for idx2 in candidate_indices:
                                    c = self.robot.node_coords[int(idx2)]
                                    if not is_recent_xy(c):
                                        filtered.append(int(idx2))
                                if len(filtered) == 0:
                                    filtered = [int(candidate_indices[0])]
                                filtered.sort(key=lambda k: (visit_counts.get(k, 0), -np.linalg.norm(self.robot.node_coords[int(k)] - self.robot.location)))
                                chosen = None
                                for k in filtered:
                                    cand_loc = self.robot.node_coords[int(k)]
                                    path_check = self.env.get_shortest_path(self.robot.location, cand_loc)
                                    if path_check and len(path_check) > 1:
                                        chosen = int(k)
                                        break
                                if chosen is None:
                                    chosen = int(filtered[0])
                                next_location = self.robot.node_coords[chosen]
                                action_index = torch.tensor([[chosen]]).long()
                                self.current_vlm_target = next_location

                # 3. Execute Move towards current_vlm_target
                if self.current_vlm_target is not None:
                    path = self.env.get_shortest_path(self.robot.location, self.current_vlm_target)
                    if path and len(path) > 1:
                        next_location = path[1]
                        
                        neighbor_indices = self.robot.neighbor_indices
                        action_index = torch.tensor([[0]]).long().to(self.device)
                        
                        if neighbor_indices is not None and len(neighbor_indices) > 0:
                            try:
                                neighbor_coords = self.robot.node_coords[neighbor_indices]
                                if len(neighbor_coords) > 0:
                                    dists = np.linalg.norm(neighbor_coords - next_location, axis=1)
                                    best_idx = np.argmin(dists)
                                    action_index = torch.tensor([[neighbor_indices[best_idx]]]).long().to(self.device)
                            except Exception as e:
                                print(f"Worker {self.meta_agent_id}: Warning - Failed to map path to neighbor index: {e}", flush=True)
                        
                        is_rescue_move = True
                        global_override_success = True
                    else:
                        print(f"Worker {self.meta_agent_id}: Path to VLM Target blocked. Clearing.", flush=True)
                        self.current_vlm_target = None
            
            # RL Model Fallback (only if VLM is off or failed completely)
            max_utility = 0
            if self.robot.utility is not None and len(self.robot.utility) > 0:
                max_utility = np.max(self.robot.utility)
            
            time_ratio = i / MAX_EPISODE_STEP
            switch_input = torch.tensor([self.env.explored_rate, max_utility / 100.0, time_ratio], dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                switch_logits = self.stair_switch_net(switch_input.unsqueeze(0))
                switch_probs = torch.softmax(switch_logits, dim=-1)
                switch_action = torch.argmax(switch_probs).item() 
                
            # Update exploration gain stats
            self.check_exploration_gain(i)

            min_steps_for_switch = MAX_EPISODE_STEP // 8 
            enough_steps_for_switch = self.steps_on_current_floor >= min_steps_for_switch
            
            # Frontier Push: If exploration is slow and we still have a lot of unknown,
            # proactively set a frontier as the forced exploration target even when utility > 0.
            # This prevents the agent from lingering in already explored regions.
            if (self.current_exploration_target is None
                and hasattr(self.env, "global_frontiers")
                and self.env.global_frontiers
                and self.recent_exploration_gain < 0.3
                and self.env.explored_rate < 0.90):
                
                candidates = []
                for item in self.env.global_frontiers:
                    if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], tuple):
                        f_coords, f_area = item
                    else:
                        f_coords = item
                        f_area = 10.0
                    
                    is_tabu = False
                    for t_coord in self.tabu_frontiers:
                        if np.linalg.norm(np.array(f_coords) - np.array(t_coord)) < 2.0:
                            is_tabu = True
                            break
                    if is_tabu:
                        continue
                    
                    dist_e = np.linalg.norm(np.array(f_coords) - self.robot.location)
                    coarse = f_area / (1.0 + dist_e * 0.1)
                    candidates.append((coarse, f_coords, float(f_area)))
                
                candidates.sort(key=lambda x: x[0], reverse=True)
                top_k = candidates[: min(10, len(candidates))]
                
                best_frontier = None
                best_score = -1.0
                best_geo = None
                
                for _, f_coords, f_area in top_k:
                    if hasattr(self.env, "get_shortest_path_with_distance"):
                        path_check, geo = self.env.get_shortest_path_with_distance(self.robot.location, np.array(f_coords))
                    else:
                        path_check = self.env.get_shortest_path(self.robot.location, np.array(f_coords))
                        if path_check and len(path_check) > 1:
                            arr = np.array(path_check, dtype=float)
                            geo = float(np.sum(np.linalg.norm(arr[1:] - arr[:-1], axis=1)))
                        else:
                            geo = float('inf')
                    
                    if not path_check or len(path_check) <= 1 or not np.isfinite(geo):
                        continue
                    
                    score = f_area / (1.0 + geo * 0.2)
                    if score > best_score:
                        best_score = score
                        best_frontier = f_coords
                        best_geo = geo
                
                if best_frontier is not None:
                    self.current_exploration_target = np.array(best_frontier, dtype=float)
                    self.tabu_frontiers.append(best_frontier)
                    if len(self.tabu_frontiers) > 8:
                        self.tabu_frontiers.pop(0)
                    if best_geo is None:
                        print(f"Worker {self.meta_agent_id}: Frontier Push activated. Targeting {best_frontier} (Score {best_score:.2f})", flush=True)
                    else:
                        print(f"Worker {self.meta_agent_id}: Frontier Push activated. Targeting {best_frontier} (Score {best_score:.2f}, Geo {best_geo:.1f}m)", flush=True)
            
            # Heuristic 1: Low Utility + Low Exploration Gain (Stagnation)
            # "Buy Oil" Strategy: Switch when marginal return (gain) is low AND utility is low.
            # Independent of total floor size.
            
            # Thresholds:
            # - Utility < Threshold (e.g. 20)
            # - Exploration Gain < 0.5 pixels/step (approx 4m^2 per 50 steps) -> Very slow progress
            
            # cond_stagnation = (max_utility < UTILITY_SWITCH_THRESHOLD) and (self.recent_exploration_gain < 0.5)
            
            # Safety: If utility is extremely low (< 5) and we've been here a long time, force switch
            # This handles cases where gain is 0 because we are stuck in a loop
            # cond_force = (max_utility < 5.0) and (self.steps_on_current_floor > 300)
            
            # should_switch = enough_steps_for_switch and (cond_stagnation or cond_force)

            # --- UPDATED LOGIC per User Request: Finish exploring one map first ---
            cond_finished = self.env.explored_rate > FLOOR_SWITCH_EXP_RATE
            
            # Or if we have no valid frontiers left, but only after we have explored enough
            # (avoid switching too early when mapping/frontiers are still unreliable)
            cond_exhausted = (
                (self.env.explored_rate > MIN_EXP_RATE_FOR_UTILITY_SWITCH)
                and (max_utility < UTILITY_SWITCH_THRESHOLD)
                and (self.recent_exploration_gain < 0.5)
            )
            
            # Or if we spent too long (80% of max steps) without much gain
            cond_stuck = (
                (self.steps_on_current_floor > MAX_EPISODE_STEP * 0.8)
                and (self.recent_exploration_gain < 0.1)
                and (max_utility < UTILITY_SWITCH_THRESHOLD)
            )
            
            # Early escape hatch: if we are repeatedly stuck early (low explored rate) and making little progress,
            # allow switching floors to avoid wasting the entire episode on a broken/local failure mode.
            cond_early_stuck = (
                (self.steps_on_current_floor > 250)
                and (self.env.explored_rate < 0.20)
                and (self.recent_exploration_gain < 0.20)
                and (max_utility < UTILITY_SWITCH_THRESHOLD)
                and (self.stuck_events >= 5)
            )

            # CRITICAL FIX: Allow switching if utility is high enough, even if we had a forced target.
            # This prevents being stuck in a "Forced Exploration" loop when a better target (e.g. from VLM) is available.
            switch_allowed = (
                (self.stuck_events > 15)
                or (
                    (not force_global_plan)
                    and (self.zero_utility_steps <= 5)
                    and ((self.current_exploration_target is None) or (max_utility > UTILITY_SWITCH_THRESHOLD))
                )
            )
            
            should_switch = enough_steps_for_switch and switch_allowed and (cond_finished or cond_exhausted or cond_stuck or cond_early_stuck)
            actual_action = 1 if should_switch else 0
            
            self.stair_switch_buffer.append({
                'state': switch_input,
                'action': torch.tensor([actual_action], device=self.device),
                'step': i
            })
            
            if i % 5 == 0:
                print(f"Step {i}: ExpRate={self.env.explored_rate:.2f}, Gain={self.recent_exploration_gain:.2f}, MaxUtil={max_utility:.2f}, Switch={should_switch} (Prob={switch_probs[0][1]:.2f})")

            if self.multi_floor_enabled and should_switch:
                best_idx = self.select_target_stairs()
                if best_idx is not None:
                    self.target_stairs_index = best_idx
                    self.env.current_target_stairs_index = self.target_stairs_index
                    
                    if not self.inter_floor_active:
                        self.inter_floor_active = True
                        self.inter_floor_start_step = i
                else:
                    # We want to switch (bored/stuck) but don't know where stairs are.
                    # Instead of lazily wandering, FORCE exploration to find stairs.
                    print(f"Step {i}: Wants to switch floors (Gain={self.recent_exploration_gain:.2f}) but NO stairs known! Activating Forced Exploration.", flush=True)
                    self.zero_utility_steps = 10 # Force trigger 'Zero Utility' logic immediately

            if should_switch and self.multi_floor_enabled and self.target_stairs_index is not None:
                if self.env.stairs_coords_list is not None and len(self.env.stairs_coords_list) > 0:
                    target_coords = self.env.stairs_coords_list[self.target_stairs_index]
                    if np.linalg.norm(self.env.robot_location - np.array(target_coords)) > STAIRS_DIST_TOLERANCE:
                        all_nodes = [n.data.coords for n in self.robot.node_manager.nodes_dict.__iter__()]
                        path_found = False
                        
                        if all_nodes:
                            all_nodes_arr = np.array(all_nodes)
                            current_floor_nodes = []
                            for n in self.robot.node_manager.nodes_dict.__iter__():
                                if hasattr(n.data, 'floor_id') and n.data.floor_id == self.env.floor_id:
                                    current_floor_nodes.append(n.data.coords)
                                elif not hasattr(n.data, 'floor_id'): 
                                     current_floor_nodes.append(n.data.coords)
                            
                            if current_floor_nodes:
                                all_nodes_arr = np.array(current_floor_nodes)
                                dists_to_target = np.linalg.norm(all_nodes_arr - np.array(target_coords), axis=1)
                                best_idx = np.argmin(dists_to_target)
                                best_node_coords = all_nodes_arr[best_idx]
                                
                                path, length = self.robot.node_manager.a_star(self.robot.location, best_node_coords)
                                
                                # --- NEW: Robust Fallback for Graph Disconnectivity ---
                                # If Graph A* fails (path is None) but we have a valid physical target (current_exploration_target),
                                # we MUST NOT give up. We should use the physical NavMesh path to "walk" the agent.
                                # This handles the case where the graph is broken/sparse but the world is navigable.
                                if (path is None or length > 1e8):
                                    print(f"Worker {self.meta_agent_id}: Graph A* failed to target {target_coords}. Trying Physical NavMesh Path...", flush=True)
                                    try:
                                        # 1. Compute Physical Path
                                        sim_path = habitat_sim.ShortestPath()
                                        sim_path.requested_start = self.env.sim.get_agent(0).get_state().position
                                        # Ensure target is 3D (y from current pos)
                                        target_y = self.env.sim.get_agent(0).get_state().position[1]
                                        sim_path.requested_end = np.array([target_coords[0], target_y, target_coords[1]])
                                        found_sim_path = self.env.sim.pathfinder.find_path(sim_path)
                                        
                                        if found_sim_path and len(sim_path.points) > 1:
                                            # 2. Get next waypoint (interpolate if too far)
                                            # path.points[0] is start, [1] is next corner
                                            p0 = np.array(sim_path.points[0])
                                            p1 = np.array(sim_path.points[1])
                                            
                                            # Flatten to 2D for calculation
                                            p0_2d = np.array([p0[0], p0[2]])
                                            p1_2d = np.array([p1[0], p1[2]])
                                            
                                            vec = p1_2d - p0_2d
                                            dist = np.linalg.norm(vec)
                                            step_size = 0.5 # 50cm step
                                            
                                            if dist > step_size:
                                                # Interpolate
                                                next_step_2d = p0_2d + vec * (step_size / dist)
                                            else:
                                                # Just go to the corner
                                                next_step_2d = p1_2d
                                            
                                            print(f"Worker {self.meta_agent_id}: Physical Path Found. Walking to intermediate point {next_step_2d}", flush=True)
                                            
                                            # 3. Set as next location
                                            next_location = next_step_2d
                                            
                                            # 4. CRITICAL: Bypass Neighbor Check
                                            # Since this point might not be in the graph yet, we must treat it as a "rescue/forced" move
                                            is_rescue_move = True 
                                            path_found = True
                                            
                                            # 5. Dummy action index (we will rely on next_location to drive movement)
                                            # We need to find the closest 'action' index if we were using discrete actions,
                                            # but here we are overriding next_location directly.
                                            # We just set a dummy index to satisfy the return type.
                                            action_index = torch.tensor([[0]]).long().to(self.device)
                                            
                                    except Exception as e:
                                        print(f"Worker {self.meta_agent_id}: Physical Path Fallback Error: {e}", flush=True)
                                # -------------------------------------------------------

                                if path and length < 1e8:
                                    next_target_loc = np.array(path[0])
                                    
                                    # Handle cross-floor path (3D coordinates: floor, x, y)
                                    if next_target_loc.shape[0] == 3:
                                        target_floor = int(next_target_loc[0])
                                        next_target_loc = next_target_loc[1:]
                                        
                                        # Detect floor switch
                                        if target_floor != self.env.floor_id:
                                            print(f"[Worker] Path requires floor switch: F{self.env.floor_id} -> F{target_floor}")
                                            # Find which stair connects to this?
                                            # We need to execute the transition.
                                            # Since we don't have the stair index directly from A*, 
                                            # we can look up the closest stair to our current location.
                                            stair_idx = self.select_target_stairs()
                                            if stair_idx is not None:
                                                self.perform_stair_transition(i, stair_idx)
                                                # After transition, we need to break/continue to update state
                                                # Update state for rest of loop
                                                next_location = self.robot.location # Stay at new location
                                                action_index = torch.tensor([[0]]).long().to(self.device) # Dummy action
                                                path_found = True # Handled
                                            else:
                                                print("[Worker] Error: A* planned floor switch but no stairs found!")
                                    
                                    if 'target_floor' in locals() and target_floor == self.env.floor_id:
                                        pass
                                    
                                    # Standard move logic (only if on same floor as target)
                                    target_is_same_floor = True
                                    if isinstance(path[0], tuple):
                                         if len(path[0]) == 3:
                                             target_is_same_floor = (self.env.floor_id == int(path[0][0]))
                                    elif isinstance(path[0], np.ndarray):
                                         if path[0].shape[0] == 3:
                                             target_is_same_floor = (self.env.floor_id == int(path[0][0]))
                                    
                                    if target_is_same_floor:
                                        neighbor_indices = self.robot.neighbor_indices
                                        neighbor_coords = self.robot.node_coords[neighbor_indices]
                                        
                                        diff = np.linalg.norm(neighbor_coords - next_target_loc, axis=1)
                                        idx = np.argmin(diff)
                                        
                                        if diff[idx] < 1.0: 
                                            next_location = neighbor_coords[idx]
                                            action_index = torch.tensor([[idx]]).long()
                                            path_found = True
                        
                        if not path_found:
                            if self.robot.frontier is not None and len(self.robot.frontier) > 0:
                                frontier_coords = list(self.robot.frontier)
                                frontier_coords_arr = np.array(frontier_coords)
                                dists_to_stair = np.linalg.norm(frontier_coords_arr - np.array(target_coords), axis=1)
                                best_frontier_idx = np.argmin(dists_to_stair)
                                best_frontier = frontier_coords[best_frontier_idx]
                                
                                path, length = self.robot.node_manager.a_star(self.robot.location, best_frontier)
                                if path and length < 1e8:
                                    next_target_loc = np.array(path[0])
                                    if next_target_loc.shape[0] == 3:
                                        next_target_loc = next_target_loc[1:]
                                    neighbor_indices = self.robot.neighbor_indices
                                    neighbor_coords = self.robot.node_coords[neighbor_indices]
                                    diff = np.linalg.norm(neighbor_coords - next_target_loc, axis=1)
                                    idx = np.argmin(diff)
                                    if diff[idx] < 2.0: 
                                        next_location = neighbor_coords[idx]
                                        action_index = torch.tensor([[idx]]).long()
                                        path_found = True
                                        print(f"Path to stairs blocked, rerouting to closest frontier at {best_frontier}")

            self.save_action(action_index)

            # Robust node lookup
            node_key = (self.robot.location[0], self.robot.location[1])
            node = self.robot.node_manager.nodes_dict.find(node_key)
            
            if node is None:
                # Fallback: Find closest node within reasonable range
                bbox_size = 2.0 # Half of NODE_RESOLUTION (4.0)
                candidates = self.robot.node_manager.nodes_dict.within_bb(quads.BoundingBox(
                    min_x=node_key[0]-bbox_size, min_y=node_key[1]-bbox_size,
                    max_x=node_key[0]+bbox_size, max_y=node_key[1]+bbox_size
                ))
                if candidates:
                    node = min(candidates, key=lambda n: np.linalg.norm(np.array(n.data.coords) - self.robot.location))
            
            if node is None:
                # Attempt to recover: update planning state to force graph refresh and off-grid node creation
                try:
                    self.robot.update_planning_state(self.env.belief_info, self.env.robot_location, self.env.floor_id)
                    # Try lookup again after refresh
                    node = self.robot.node_manager.nodes_dict.find(node_key)
                    if node is None:
                        # Force insert an off-grid node at robot location and connect it to nearest neighbors
                        try:
                            rounded = np.around(self.robot.location, 1)
                            self.robot.node_manager.add_node_to_dict(rounded, getattr(self.env, "global_frontiers", []), self.env.belief_info, self.env.floor_id)
                            # Connect to up to 3 nearest nodes to avoid isolation
                            search_radius = 8.0
                            bbox = quads.BoundingBox(
                                min_x=rounded[0]-search_radius, min_y=rounded[1]-search_radius,
                                max_x=rounded[0]+search_radius, max_y=rounded[1]+search_radius
                            )
                            nearby = self.robot.node_manager.nodes_dict.within_bb(bbox)
                            # Find inserted node wrapper
                            node = self.robot.node_manager.nodes_dict.find((round(float(rounded[0]), 1), round(float(rounded[1]), 1)))
                            if node:
                                current_node = node.data
                                candidates_nn = []
                                for wrapper in nearby:
                                    n = wrapper.data
                                    if n.coords.tolist() == current_node.coords.tolist():
                                        continue
                                    d = np.linalg.norm(n.coords - current_node.coords)
                                    candidates_nn.append((d, n))
                                candidates_nn.sort(key=lambda x: x[0])
                                for d, n in candidates_nn[:3]:
                                    current_node.neighbor_set.add((n.coords[0], n.coords[1]))
                                    n.neighbor_set.add((current_node.coords[0], current_node.coords[1]))
                        except Exception as e:
                            print(f"Emergency: Failed to insert off-grid node at {self.robot.location}: {e}", flush=True)
                except Exception as e:
                    print(f"Warning: Planning state refresh failed during node recovery: {e}", flush=True)
                
                # Final check after recovery attempts
                if node is None:
                    print(f"Critical Warning: No node found near robot at {self.robot.location}. Skipping neighbor check.", flush=True)
                    check = np.empty((0, 2))
                else:
                    check = np.array(list(node.data.neighbor_set)).reshape(-1, 2)
            else:
                check = np.array(list(node.data.neighbor_set)).reshape(-1, 2)
            
            if next_location is None:
                valid_neighbors = []
                if len(check) > 0:
                    for n in check:
                        if n[0] != self.robot.location[0] or n[1] != self.robot.location[1]:
                            valid_neighbors.append(n)
                if valid_neighbors:
                    next_location = np.array(random.choice(valid_neighbors))
                else:
                    print("Warning: Next location is None and no valid neighbors. Forcing exploration to random reachable point.", flush=True)
                    # Force find a random reachable point on the map to break the loop
                    try:
                        # 1. Get all free space indices from mapper
                        h, w = self.env.mapper.global_map.shape
                        free_indices = np.argwhere(self.env.mapper.global_map == 255)
                        
                        if len(free_indices) > 0:
                            # 2. Pick a random target
                            # Try up to 10 times to find a reachable point
                            found_target = False
                            for _ in range(10):
                                idx = random.choice(free_indices)
                                py, px = idx
                                # Convert pixel to world coordinates
                                tx = px * self.env.mapper.cell_size + self.env.mapper.origin_x
                                ty = py * self.env.mapper.cell_size + self.env.mapper.origin_y
                                target_pos = np.array([tx, ty])
                                
                                # Check distance (don't pick something too close)
                                if np.linalg.norm(target_pos - self.robot.location) < 2.0:
                                    continue
                                    
                                # Check reachability
                                if self.env.get_grid_path_check(self.robot.location, target_pos):
                                    next_location = target_pos
                                    print(f"Stagnation Rescue: Found random reachable target at {next_location}", flush=True)
                                    found_target = True
                                    # Create a temporary node here to aid future navigation
                                    # Use add_node_to_dict with empty frontiers since we just want a navigable point
                                    self.robot.node_manager.add_node_to_dict(target_pos, [], self.env.belief_info, self.env.floor_id)
                                    break
                            
                            if not found_target:
                                print("Stagnation Rescue: Failed to find reachable random target. Skipping step.", flush=True)
                                continue
                        else:
                             print("Stagnation Rescue: No free space found in map?! Skipping step.", flush=True)
                             continue
                    except Exception as e:
                        print(f"Stagnation Rescue Error: {e}", flush=True)
                        continue
            
            target_complex = next_location[0] + next_location[1] * 1j
            check_complex = check[:, 0] + check[:, 1] * 1j
            
            # Skip neighbor check if this is a rescue/VLM move
            # This allows the agent to 'jump' to a VLM-selected node even if not strictly connected in the current graph
            if not is_rescue_move and len(check) > 0 and target_complex not in check_complex:
                print(f"Warning: Target {next_location} not in neighbors. Correcting...")
                dists = np.linalg.norm(check - next_location, axis=1)
                idx = np.argmin(dists)
                next_location = check[idx]
            
            if next_location[0] == self.robot.location[0] and next_location[1] == self.robot.location[1]:
                print("Warning: Agent selected current location. Forcing random move.")
                # Filter out current location from neighbors if present
                valid_neighbors = []
                if node is not None:
                     # Filter neighbors
                     valid_neighbors = [n for n in list(node.data.neighbor_set) if n != (self.robot.location[0], self.robot.location[1])]
                
                # If no valid neighbors in graph, try to find ANY reachable node in belief map
                if not valid_neighbors:
                    print(f"Error: No valid neighbors in graph for node {node.data.coords if node else 'None'}! Attempting emergency recovery...", flush=True)
                    # Recovery: Look for closest node in all nodes that is reachable
                    all_nodes = [n.data for n in self.robot.node_manager.nodes_dict.__iter__()]
                    candidates = []
                    candidates_unsafe = []
                    
                    # Optimization: Filter by distance first to avoid checking all nodes
                    nearby_nodes = [n for n in all_nodes if np.linalg.norm(n.coords - self.robot.location) < 15.0]
                    
                    for n in nearby_nodes:
                        if n.coords.tolist() == self.robot.location.tolist():
                             continue
                        
                        # STRICT VALIDITY CHECK: Ensure candidate is within map bounds and navigable
                        # This prevents jumping to "void" areas or out-of-bounds
                        if hasattr(self.env, 'is_valid_location') and not self.env.is_valid_location(n.coords):
                            continue
                        
                        # Check reachability
                        # Check if node is actually reachable via A* on belief map
                        # This prevents picking nodes behind walls that look "close" in Euclidean distance
                        path_check, _ = self.robot.node_manager.a_star(self.robot.location, n.coords)
                        if not path_check:
                             candidates_unsafe.append(n.coords) # Store as unsafe fallback
                             continue
                        
                        candidates.append(n.coords)
                    
                    if candidates:
                         print(f"Recovery: Found {len(candidates)} reachable nodes nearby. Picking closest.", flush=True)
                         # Pick closest valid neighbor
                         candidates.sort(key=lambda c: np.linalg.norm(c - self.robot.location))
                         valid_neighbors = [candidates[0]]
                         
                         # Force add neighbor connection to prevent future stuck
                         if node:
                             neighbor_coords = candidates[0]
                             node.data.neighbor_set.add((neighbor_coords[0], neighbor_coords[1]))
                             # Also update the neighbor to point back to us
                             neighbor_node_wrapper = self.robot.node_manager.nodes_dict.find((neighbor_coords[0], neighbor_coords[1]))
                             if neighbor_node_wrapper:
                                 neighbor_node_wrapper.data.neighbor_set.add((node.data.coords[0], node.data.coords[1]))
                    
                    # If NO candidates found with path check, try unsafe candidates as last resort
                    # but only if they are close enough AND we can use physical pathfinder
                    elif candidates_unsafe:
                         print(f"Recovery: No path-reachable nodes found. Checking {len(candidates_unsafe)} unsafe candidates...", flush=True)
                         # Filter unsafe candidates by distance - only pick very close ones (e.g. < 5.0m) to risk a "jump"
                         # Increased from 2.0 to 5.0 to handle slightly larger gaps in sparse graphs
                         candidates_unsafe_close = [c for c in candidates_unsafe if np.linalg.norm(c - self.robot.location) < 5.0]
                         
                         if candidates_unsafe_close:
                             # Sort by distance
                             candidates_unsafe_close.sort(key=lambda c: np.linalg.norm(c - self.robot.location))
                             best_unsafe = candidates_unsafe_close[0]
                             
                             # Try to validate with Physical PathFinder if available
                             if self.env.sim and self.env.sim.pathfinder.is_loaded:
                                 try:
                                     sim_path = habitat_sim.ShortestPath()
                                     sim_path.requested_start = self.env.sim.get_agent(0).get_state().position
                                     target_y = self.env.sim.get_agent(0).get_state().position[1]
                                     sim_path.requested_end = np.array([best_unsafe[0], target_y, best_unsafe[1]])
                                     found_sim_path = self.env.sim.pathfinder.find_path(sim_path)
                                     
                                     if found_sim_path:
                                         print(f"Recovery: Physical Path VALIDATED to unsafe node {best_unsafe}. Using it.", flush=True)
                                         valid_neighbors = [best_unsafe]
                                         # Add connection
                                         if node:
                                             node.data.neighbor_set.add((best_unsafe[0], best_unsafe[1]))
                                             neighbor_node_wrapper = self.robot.node_manager.nodes_dict.find((best_unsafe[0], best_unsafe[1]))
                                             if neighbor_node_wrapper:
                                                 neighbor_node_wrapper.data.neighbor_set.add((node.data.coords[0], node.data.coords[1]))
                                     else:
                                         print(f"Recovery: Physical Path FAILED to unsafe node {best_unsafe}. Will fall back to Ultimate Rescue.", flush=True)
                                 except Exception as e:
                                     print(f"Recovery: Physical Path Check Error: {e}", flush=True)

                    
                    # --- CRITICAL FALLBACK: TELEPORT TO RANDOM NAVIGABLE POINT ---
                    if not valid_neighbors:
                         print("Critical Error: No valid candidates found for recovery! Agent is ISOLATED.", flush=True)
                         print("Attempting ULTIMATE RESCUE: Teleporting to random navigable point.", flush=True)
                         if self.env.sim and self.env.sim.pathfinder.is_loaded:
                             try:
                                 target_pos = self.env.sim.pathfinder.get_random_navigable_point()
                                 # Convert back to 2D
                                 next_location = np.array([target_pos[0], target_pos[2]])
                                 
                                 # Teleport SIM AGENT directly (since step() might fail if path is blocked)
                                 new_state = habitat_sim.AgentState()
                                 new_state.position = target_pos
                                 self.env.sim.get_agent(0).set_state(new_state)
                                 
                                 # Update Python Env State
                                 self.env.robot_location = next_location
                                 valid_neighbors = [next_location] # Just to satisfy logic below
                                 print(f"ULTIMATE RESCUE SUCCESS: Teleported to {next_location}", flush=True)
                                 
                                 # Force a continue to next step to refresh observations
                                 # (We manually set next_location, but let's let the loop handle step update)
                             except Exception as e:
                                 print(f"ULTIMATE RESCUE FAILED: {e}", flush=True)

                if valid_neighbors:
                    chosen = random.choice(valid_neighbors)
                    next_location = np.array(chosen)
                else:
                    print("Error: No valid neighbors to move to!")
                    pass

            if next_location[0] == self.robot.location[0] and next_location[1] == self.robot.location[1]:
                 print(f"Critical Warning: Still at current location after correction at step {i}. Skipping step.")
                 continue

            if hasattr(self.env, "get_grid_path_check") and not self.env.get_grid_path_check(self.robot.location, next_location):
                if node is not None:
                    fallback_neighbors = [np.array(n) for n in list(node.data.neighbor_set) if n != (self.robot.location[0], self.robot.location[1])]
                    fallback_neighbors = [n for n in fallback_neighbors if self.env.get_grid_path_check(self.robot.location, n)]
                    if fallback_neighbors:
                        fallback_neighbors.sort(key=lambda n: np.linalg.norm(n - next_location))
                        next_location = fallback_neighbors[0]

            use_force = is_rescue_move or (curr_max_util < 0.1)
            if use_force:
                 # print(f"Low Utility ({curr_max_util:.2f}). Enabling Force Move for standard step.", flush=True)
                 reward = self.env.step(next_location, force=True)
            else:
                 reward = self.env.step(next_location)
            if reward < -0.05 and is_rescue_move:
                print(f"Worker {self.meta_agent_id}: Rescue move failed to {next_location} (reward={reward}).", flush=True)

            # In Habitat, stairs discovery might be handled internally or via collision/proximity
            # For compatibility, we check for stairs in env
            # self.env.discover_stairs() # HabitatEnv handles this in step()

            self.robot.update_planning_state(self.env.belief_info, self.env.robot_location, self.env.floor_id)
            
            curr_max_util = 0
            if self.robot.utility is not None and len(self.robot.utility) > 0:
                curr_max_util = np.max(self.robot.utility)
            
            do_switch = should_switch or (
                enough_steps_for_switch
                and switch_allowed
                and (self.env.explored_rate > MIN_EXP_RATE_FOR_UTILITY_SWITCH)
                and (curr_max_util < UTILITY_SWITCH_THRESHOLD)
                and (self.recent_exploration_gain < 0.5)
            )

            if do_switch and self.multi_floor_enabled and self.target_stairs_index is not None:
                if self.env.stairs_coords_list is not None and len(self.env.stairs_coords_list) > 0:
                    target_coords = self.env.stairs_coords_list[self.target_stairs_index]
                    dist_to_stairs = np.linalg.norm(self.env.robot_location - np.array(target_coords))
                    if dist_to_stairs <= STAIRS_DIST_TOLERANCE * 1.5:
                        self.perform_stair_transition(i, self.target_stairs_index)
            
            self.position_history.append((round(self.env.robot_location[0], 2), round(self.env.robot_location[1], 2), self.env.floor_id))
            if len(self.position_history) > STUCK_WINDOW:
                self.position_history.pop(0)
            if len(self.position_history) == STUCK_WINDOW:
                unique_positions = set(self.position_history)
                if len(unique_positions) <= STUCK_UNIQUE_POS_THRESHOLD:
                    print(f"Stuck detected at step {i}, initiating ROBUST recovery.", flush=True)
                    self.stuck_events += 1
                    
                    # 1. Clear History & State to prevent immediate re-trigger
                    self.position_history = []
                    self.current_exploration_target = None
                    self.target_stagnation_counter = 0
                    self.min_target_distance = np.inf
                    self.inter_floor_active = False
                    self.target_stairs_index = None
                    self.env.current_target_stairs_index = None
                    
                    # 2. Try to find a valid navigable point on the SAME floor
                    target_pos_3d = None
                    
                    # Method A: Use Habitat-Sim Pathfinder (God Mode) - Most Robust
                    if self.env.sim and self.env.sim.pathfinder.is_loaded:
                        try:
                            # Get current agent state
                            agent_state = self.env.sim.get_agent(0).get_state()
                            current_height = agent_state.position[1]
                            
                            # Try up to 50 times to find a point on the same floor
                            for _ in range(50):
                                # Get random navigable point from NavMesh
                                rnd_pt = self.env.sim.pathfinder.get_random_navigable_point()
                                # Check height difference (allow 0.5m variance for slopes/steps)
                                # This ensures we stay on the current floor/connected component
                                if abs(rnd_pt[1] - current_height) < 0.5:
                                    # Check distance from current position
                                    dist = np.linalg.norm(rnd_pt - agent_state.position)
                                    if dist > 3.0: # Minimum jump distance to break loop
                                        target_pos_3d = rnd_pt
                                        print(f"Stuck Recovery: Found valid random point {target_pos_3d} (dist={dist:.2f})", flush=True)
                                        break
                        except Exception as e:
                            print(f"Stuck Recovery Pathfinder Error: {e}", flush=True)
                    
                    # Method B: Fallback to belief map free space (if Pathfinder fails/unavailable)
                    if target_pos_3d is None:
                        try:
                            free_coords = get_free_area_coords(self.env.belief_info)
                            if len(free_coords) > 0:
                                # Filter points > 3.0m away
                                curr_loc = self.env.robot_location
                                dists = np.linalg.norm(free_coords - curr_loc, axis=1)
                                valid_indices = np.where(dists > 3.0)[0]
                                if len(valid_indices) > 0:
                                    idx = np.random.choice(valid_indices)
                                    target_2d = free_coords[idx]
                                    print(f"Stuck Recovery: Teleporting to free space {target_2d} (dist={dists[idx]:.2f}) via step(force=True)", flush=True)
                                    self.env.step(target_2d, force=True)
                                    self.robot.update_planning_state(self.env.belief_info, self.env.robot_location, self.env.floor_id)
                                    
                                    # Reset exploration target to prevent returning to the stuck location
                                    if self.current_exploration_target is not None:
                                        print(f"Stuck Recovery: Clearing Forced Target {self.current_exploration_target} after teleport.", flush=True)
                                        self.current_exploration_target = None
                                        self.zero_utility_steps = 0
                                    
                                    # Re-initialize graph to clear bad state
                                    self.ground_truth_node_manager = GroundTruthNodeManager(self.robot.node_manager, self.env.ground_truth_info,
                                                                                            device=self.device, plot=self.save_image)
                                    continue # Skip rest of loop
                        except Exception as e:
                            print(f"Stuck Recovery Belief Map Error: {e}", flush=True)

                    # Execute Recovery (Navigation or Teleport)
                    if target_pos_3d is not None:
                        try:
                            # Verify path existence first (Physical Validity Check)
                            # We want to see if we can actually WALK there instead of teleporting
                            path = habitat_sim.ShortestPath()
                            path.requested_start = self.env.sim.get_agent(0).get_state().position
                            path.requested_end = target_pos_3d
                            found_path = self.env.sim.pathfinder.find_path(path)
                            
                            if found_path:
                                # Path exists! We must FORCE move the agent along this path to escape the trap.
                                # Relying on Belief Map A* (standard step) often fails if the map is bad.
                                print(f"Stuck Recovery: Found valid path to {target_pos_3d} (dist={path.geodesic_distance:.2f}). EXECUTING FORCE MOVE.", flush=True)
                                
                                target_2d = np.array([target_pos_3d[0], target_pos_3d[2]])
                                self.env.step(target_2d, force=True)
                                self.robot.update_planning_state(self.env.belief_info, self.env.robot_location, self.env.floor_id)

                                # Reset exploration target to prevent returning to the stuck location
                                if self.current_exploration_target is not None:
                                    print(f"Stuck Recovery: Clearing Forced Target {self.current_exploration_target} after Force Move.", flush=True)
                                    self.current_exploration_target = None
                                    self.zero_utility_steps = 0
                                
                                # Re-initialize graph to clear bad state
                                self.ground_truth_node_manager = GroundTruthNodeManager(self.robot.node_manager, self.env.ground_truth_info,
                                                                                        device=self.device, plot=self.save_image)
                                
                                self.stuck_events = 0
                                continue # Skip rest of loop
                                
                            else:
                                # Path DOES NOT exist. Physical Trap.
                                # User hates teleport, but if we are physically trapped (Isolated Island), we must teleport or die.
                                print(f"Stuck Recovery: Target {target_pos_3d} is physically unreachable (Isolated Island).", flush=True)
                                # Fallback to Teleport ONLY if pathfinding fails (Last Resort)
                                
                                # Teleport agent physically
                                new_state = self.env.sim.get_agent(0).get_state()
                                new_state.position = target_pos_3d
                                # Keep rotation or randomize? Keep for now.
                                self.env.sim.get_agent(0).set_state(new_state)
                                
                                # Update Python Env State
                                # Habitat-Sim uses (x, y, z) where y is up.
                                # self.env.robot_location is (x, z) usually (2D projection).
                                self.env.robot_location = np.array([target_pos_3d[0], target_pos_3d[2]])
                                
                                # Reset exploration target to prevent returning to the stuck location
                                if self.current_exploration_target is not None:
                                    print(f"Stuck Recovery: Clearing Forced Target {self.current_exploration_target} after physical teleport.", flush=True)
                                    self.current_exploration_target = None
                                    self.zero_utility_steps = 0
                                
                                # Force update of internal state
                                self.robot.update_planning_state(self.env.belief_info, self.env.robot_location, self.env.floor_id)
                                
                                # Re-initialize GT Graph to match new location and clear any graph artifacts
                                self.ground_truth_node_manager = GroundTruthNodeManager(self.robot.node_manager, self.env.ground_truth_info,
                                                                                        device=self.device, plot=self.save_image)
                                
                                print(f"Stuck Recovery: FORCED TELEPORT to {target_pos_3d} due to physical trap.", flush=True)
                        except Exception as e:
                            print(f"Stuck Recovery Execution Failed: {e}", flush=True)
                    else:
                        print("Stuck Recovery FAILED: Could not find valid target.", flush=True)

            
            if self.robot.utility.sum() <= 0.05: # Threshold to ignore sensor noise
                self.zero_utility_steps += 1
            else:
                self.zero_utility_steps = 0
                self.tabu_exploration_targets = [] # Reset tabu list when we find SIGNIFICANT utility
                if self.current_exploration_target is not None:
                    print(f"Worker {self.meta_agent_id}: High Utility ({self.robot.utility.sum():.2f}). Clearing Forced Target.", flush=True)
                    self.current_exploration_target = None


            # Only force exploration if utility is 0 for a while (give model a chance)
            if self.zero_utility_steps > 5:
                print(f"Warning: Utility sum is 0 for {self.zero_utility_steps} steps. Model inputs are empty. Activating Forced Exploration to find new frontiers...", flush=True)
                # Debug: Print node info
                print(f"Node count: {len(self.robot.node_manager.nodes_dict)}")
                
                # RELAX TERMINATION CONDITION:
                # If explored rate is low (< 0.9), assume we are just stuck in a local minimum
                # and shouldn't quit. Instead, force a random move or keep going.
                if self.env.explored_rate < 0.9:
                    print("Explored rate is < 90%, preventing early termination. Forcing random exploration...", flush=True)
                    # Force a random nearby node as target (even if utility is 0)
                    if len(self.robot.node_manager.nodes_dict) > 0:
                        target_coords = None
                        
                        # Check if we have a persistent target
                        if self.current_exploration_target is not None:
                            dist_to_target = np.linalg.norm(self.robot.location - self.current_exploration_target)
                            
                            # CRITICAL FIX: Stagnation check MUST happen here, not just at the end.
                            # If we have been chasing this target for too long without success, drop it.
                            # We use self.target_stagnation_counter which is updated at the end of loop.
                            if self.target_stagnation_counter > 5:
                                 print(f"Worker {self.meta_agent_id}: Stagnated while reaching Forced Target (Early Check). Clearing and Tabu-ing.", flush=True)
                                 self.tabu_exploration_targets.append(self.current_exploration_target)
                                 if len(self.tabu_exploration_targets) > 5:
                                     self.tabu_exploration_targets.pop(0)
                                 self.current_exploration_target = None
                                 target_coords = None # Reset to trigger new search
                            elif dist_to_target < 2.5: # WIDENED THRESHOLD: 1.0m -> 2.5m to avoid micro-oscillations
                                print(f"Worker {self.meta_agent_id}: Reached Forced Exploration Target (Dist < 2.5m). Clearing.", flush=True)
                                # Add to tabu list to prevent immediate re-selection
                                self.tabu_exploration_targets.append(self.current_exploration_target)
                                if len(self.tabu_exploration_targets) > 5:
                                    self.tabu_exploration_targets.pop(0)
                                self.current_exploration_target = None
                                target_coords = None # Reset to trigger new search
                            else:
                                print(f"Worker {self.meta_agent_id}: Continuing to Forced Exploration Target (Dist: {dist_to_target:.2f})", flush=True)
                                target_coords = self.current_exploration_target

                        # 0. Try Global Frontiers (Best for Exploration)
                        if target_coords is None and hasattr(self.env, "global_frontiers") and self.env.global_frontiers:
                            frontiers = self.env.global_frontiers
                            if frontiers:
                                candidates = []
                                for item in frontiers:
                                    if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], tuple):
                                        f_coords, f_area = item
                                    else:
                                        f_coords = item
                                        f_area = 10.0
                                    
                                    is_tabu = False
                                    for t_coord in self.tabu_frontiers:
                                        if np.linalg.norm(np.array(f_coords) - np.array(t_coord)) < 2.0:
                                            is_tabu = True
                                            break
                                    if is_tabu:
                                        continue
                                    
                                    dist_e = np.linalg.norm(np.array(f_coords) - self.robot.location)
                                    coarse = float(f_area) / (1.0 + dist_e * 0.1)
                                    candidates.append((coarse, f_coords, float(f_area)))
                                
                                candidates.sort(key=lambda x: x[0], reverse=True)
                                top_k = candidates[: min(10, len(candidates))]
                                
                                best_frontier = None
                                best_score = -1.0
                                best_geo = None
                                
                                for _, f_coords, f_area in top_k:
                                    if hasattr(self.env, "get_shortest_path_with_distance"):
                                        path_check, geo = self.env.get_shortest_path_with_distance(self.robot.location, np.array(f_coords))
                                    else:
                                        path_check = self.env.get_shortest_path(self.robot.location, np.array(f_coords))
                                        if path_check and len(path_check) > 1:
                                            arr = np.array(path_check, dtype=float)
                                            geo = float(np.sum(np.linalg.norm(arr[1:] - arr[:-1], axis=1)))
                                        else:
                                            geo = float('inf')
                                    
                                    if not path_check or len(path_check) <= 1 or not np.isfinite(geo):
                                        continue
                                    
                                    score = f_area / (1.0 + geo * 0.2)
                                    if score > best_score:
                                        best_score = score
                                        best_frontier = f_coords
                                        best_geo = geo
                                
                                if best_frontier:
                                    target_coords = best_frontier
                                    self.tabu_frontiers.append(best_frontier)
                                    if len(self.tabu_frontiers) > 8:
                                        self.tabu_frontiers.pop(0)
                                    if best_geo is None:
                                        print(f"Worker {self.meta_agent_id}: Zero Utility. Moving to best Frontier (Score {best_score:.2f}): {target_coords}", flush=True)
                                    else:
                                        print(f"Worker {self.meta_agent_id}: Zero Utility. Moving to best Frontier (Score {best_score:.2f}, Geo {best_geo:.1f}m): {target_coords}", flush=True)

                        # 1. Try VLM Rescue (User Request: "My VLM")
                        if target_coords is None and self.use_vlm:
                            print(f"Worker {self.meta_agent_id}: Zero Utility detected. Calling VLM for visual rescue...", flush=True)
                            try:
                                # Prepare RGB image if available
                                rgb_img = self.env.rgb_image
                                
                                # Use get_vlm_action to pick a direction from neighbors
                                # Note: get_vlm_action internally calls VLM API to select best neighbor
                                debug_path = f"{gifs_path}/vlm_rescue_{self.meta_agent_id}_step_{self.zero_utility_steps}.png"

                                # observation arg is unused in VLM adapter, passing None to avoid NameError
                                next_loc, _ = self.vlm.get_vlm_action(self.robot, None, self.env.stairs_coords_list, rgb_image=rgb_img, debug_save_path=debug_path, position_history=self.position_history, current_floor=self.env.floor_id)
                                
                                # TABU CHECK for VLM
                                if next_loc is not None:
                                    is_tabu_target = False
                                    for t in self.tabu_exploration_targets:
                                        if np.linalg.norm(np.array(next_loc) - np.array(t)) < 2.0:
                                            is_tabu_target = True
                                            break
                                    if is_tabu_target:
                                        print(f"Worker {self.meta_agent_id}: VLM suggested Tabu target {next_loc}. Ignoring.", flush=True)
                                        next_loc = None # Force fallback
                                
                                if next_loc is not None:
                                    # Verify it's not current location
                                    dist_curr = np.linalg.norm(next_loc - self.robot.location)
                                    
                                    # CHECK POSITION HISTORY: Avoid recently visited nodes in Zero Utility mode
                                    is_recent = False
                                    if len(self.position_history) > 0:
                                        recent_positions = []
                                        # Handle list of tuples vs numpy array, robustly
                                        for p in self.position_history[-10:]: # Check last 10 steps
                                            if isinstance(p, (list, tuple, np.ndarray)):
                                                 # Extract only x, y for distance check
                                                 recent_positions.append(np.array(p).flatten()[:2])
                                        
                                        if recent_positions:
                                            recent_positions = np.array(recent_positions)
                                            next_loc_flat = np.array(next_loc).flatten()[:2]
                                            
                                            dists = np.linalg.norm(recent_positions - next_loc_flat, axis=1)
                                            if np.any(dists < 1.0):
                                                is_recent = True
                                    
                                    if dist_curr > 0.1 and not is_recent:
                                        path_check = self.env.get_shortest_path(self.robot.location, next_loc)
                                        if not path_check or len(path_check) <= 1:
                                            self.tabu_exploration_targets.append(np.array(next_loc, dtype=float).flatten()[:2])
                                            if len(self.tabu_exploration_targets) > 5:
                                                self.tabu_exploration_targets.pop(0)
                                            next_loc = None
                                            target_coords = None
                                        else:
                                            print(f"Worker {self.meta_agent_id}: VLM suggested move to {next_loc}", flush=True)
                                            target_coords = next_loc
                                    elif is_recent:
                                        print(f"Worker {self.meta_agent_id}: VLM suggested recently visited location {next_loc}. Ignoring to force exploration.", flush=True)
                                        # target_coords remains None, falling through to Step 2
                                    else:
                                        print(f"Worker {self.meta_agent_id}: VLM suggested current location. Ignoring.", flush=True)
                                else:
                                    # Count consecutive NONE to escalate random jump distance
                                    if not hasattr(self, 'vlm_none_counter'):
                                        self.vlm_none_counter = 0
                                    self.vlm_none_counter += 1
                            except Exception as e:
                                print(f"Worker {self.meta_agent_id}: VLM Rescue failed: {e}", flush=True)

                        # 2. Force a random nearby node OR random free point as target (if VLM didn't pick one)
                        if target_coords is None:
                            # Improved: Use BFS to find REACHABLE nodes only to prevent "No path" errors
                            key = (round(float(self.robot.location[0]), 1), round(float(self.robot.location[1]), 1))
                            start_node_obj = self.robot.node_manager.nodes_dict.find(key)
                            
                            # Fuzzy fallback
                            if start_node_obj is None:
                                bbox_size = 3.0
                                found = self.robot.node_manager.nodes_dict.within_bb(quads.BoundingBox(
                                    min_x=self.robot.location[0]-bbox_size, min_y=self.robot.location[1]-bbox_size,
                                    max_x=self.robot.location[0]+bbox_size, max_y=self.robot.location[1]+bbox_size
                                ))
                                if found:
                                    start_node_obj = min(found, key=lambda n: np.linalg.norm(n.data.coords - self.robot.location))

                            reachable_coords = []
                            if start_node_obj:
                                # BFS
                                start_node = start_node_obj.data
                                visited = set()
                                visited.add((start_node.coords[0], start_node.coords[1]))
                                queue = [start_node]
                                
                                while queue:
                                    curr = queue.pop(0)
                                    reachable_coords.append(curr.coords)
                                    
                                    for n_coords in curr.neighbor_set:
                                        if n_coords not in visited:
                                            visited.add(n_coords)
                                            n_key = (round(float(n_coords[0]), 1), round(float(n_coords[1]), 1))
                                            n_node = self.robot.node_manager.nodes_dict.find(n_key)
                                            if n_node:
                                                queue.append(n_node.data)
                            
                            if not reachable_coords:
                                # Fallback if disconnected or start node not found
                                reachable_coords = [n.data.coords for n in self.robot.node_manager.nodes_dict.__iter__()]

                            # Pick a node slightly far away
                            candidates = [c for c in reachable_coords if np.linalg.norm(c - self.robot.location) > 2.0]
                            if not candidates: candidates = reachable_coords
                            
                            # SMART SELECTION:
                            # Instead of random choice, prioritize nodes that have been visited the least
                            # and are reasonably far away.
                            candidate_nodes = []
                            for c in candidates:
                                # Find the node object for this coordinate
                                key = (round(float(c[0]), 1), round(float(c[1]), 1))
                                node = self.robot.node_manager.nodes_dict.find(key)
                                if node:
                                    candidate_nodes.append(node.data)
                            
                            if candidate_nodes:
                                # Sort by visit_count (ascending), then by distance (descending) as tie-breaker?
                                # Actually, we want low visit count. Random among the lowest is good to avoid loops.
                                min_visits = min(n.visit_count for n in candidate_nodes)
                                best_candidates = [n for n in candidate_nodes if n.visit_count == min_visits]
                                
                                # TABU LOGIC: Filter out recently visited targets AND persistent tabu targets
                                # Use position_history as a rough proxy for tabu list
                                recent_positions = [tuple(p[:2]) for p in self.position_history[-10:]] # Check last 10 steps, only XY
                                
                                non_tabu_candidates = []
                                for n in best_candidates:
                                    n_coords = (n.coords[0], n.coords[1])
                                    # Check if we were near this node recently (fuzzy match)
                                    is_tabu = False
                                    for p in recent_positions:
                                        if np.linalg.norm(np.array(n_coords) - np.array(p)) < 2.5:
                                            is_tabu = True
                                            break
                                    
                                    # Also check persistent tabu targets
                                    if not is_tabu:
                                        for t in self.tabu_exploration_targets:
                                            if np.linalg.norm(np.array(n_coords) - np.array(t)) < 2.5:
                                                is_tabu = True
                                                break

                                    if not is_tabu:
                                        non_tabu_candidates.append(n)
                                
                                if non_tabu_candidates:
                                    best_candidates = non_tabu_candidates
                                    # If multiple nodes have same low visit count, pick one that is far away
                                    # to encourage expansion
                                    target_node = max(best_candidates, key=lambda n: np.linalg.norm(n.coords - self.robot.location))
                                    target_coords = target_node.coords
                                    print(f"Forced Exploration Target (Smart): {target_coords} (Visits: {target_node.visit_count})", flush=True)
                                else:
                                    # If ALL best candidates are tabu (recently visited), we must escape!
                                    # Trying to go to a visited node will just continue the loop.
                                    # Instead, pick a RANDOM FREE POINT (white space) to force expansion.
                                    print("Forced Exploration: All candidate nodes are recently visited (Tabu). Switching to Random Free Point search.", flush=True)
                                    try:
                                        free_coords = get_free_area_coords(self.env.belief_info)
                                        if len(free_coords) > 0:
                                            # Improved Random Jump: 
                                            # 1. Prefer points > base_min away (jump out of local minimum)
                                            # 2. If not possible, > 2.0m
                                            curr_loc = self.robot.location
                                            dists = np.linalg.norm(free_coords - curr_loc, axis=1)
                                            
                                            base_min = 4.0
                                            if hasattr(self, 'vlm_none_counter'):
                                                base_min = min(8.0, 4.0 + 1.0 * self.vlm_none_counter)
                                            valid_indices = np.where(dists > base_min)[0]
                                            if len(valid_indices) == 0:
                                                 valid_indices = np.where(dists > 2.0)[0]
                                            
                                            if len(valid_indices) > 0:
                                                # Try up to 20 samples to ensure navigable and path-exists
                                                attempts = min(20, len(valid_indices))
                                                chosen = None
                                                for _ in range(attempts):
                                                    idx = np.random.choice(valid_indices)
                                                    candidate = free_coords[idx]
                                                    if hasattr(self.env, 'is_valid_location') and not self.env.is_valid_location(candidate):
                                                        continue
                                                    nav_path = self.env.get_shortest_path(self.robot.location, candidate)
                                                    if nav_path and len(nav_path) > 1:
                                                        chosen = (idx, candidate)
                                                        break
                                                if chosen is None:
                                                    # Fallback: pick one and let downstream path check handle
                                                    idx = np.random.choice(valid_indices)
                                                    target = free_coords[idx]
                                                else:
                                                    idx, target = chosen
                                                print(f"Forced Exploration: Moving to random distant free point {target} (dist={dists[idx]:.2f})", flush=True)
                                                target_coords = target
                                            else:
                                                idx = np.random.randint(0, len(free_coords))
                                                target = free_coords[idx]
                                                print(f"Forced Exploration: Moving to random free point {target} (dist={dists[idx]:.2f})", flush=True)
                                                target_coords = target
                                        else:
                                            print("Forced Exploration: No free space found either. Using random node fallback.", flush=True)
                                            # Fallback to random node even if tabu
                                            idx = np.random.randint(0, len(candidates))
                                            target_coords = candidates[idx]
                                    except Exception as e:
                                        print(f"Forced Exploration Random Free Point failed: {e}", flush=True)
                                        idx = np.random.randint(0, len(candidates))
                                        target_coords = candidates[idx]
                            else:
                                # Fallback to random if lookup failed
                                target_coords = random.choice(candidates)
                                print(f"Forced Exploration Target (Random Fallback): {target_coords}", flush=True)
                        
                        # INJECTED FIX: Manually force robot to move towards this target to break stagnation
                        # This bypasses the neural network which might be outputting "stay" or invalid actions due to 0 utility
                        if target_coords is not None:
                            # Persist target
                            if self.current_exploration_target is None:
                                self.current_exploration_target = target_coords
                                self.target_stagnation_counter = 0 # Initialize counter
                                self.last_target_distance = np.inf
                                self.min_target_distance = np.inf # Track global minimum distance for this target

                            # Check for progress towards target
                            current_dist = np.linalg.norm(self.robot.location - self.current_exploration_target)
                            
                            # Initialize min_target_distance if not set (for existing targets)
                            if not hasattr(self, 'min_target_distance'):
                                self.min_target_distance = np.inf

                            # Update distance tracking
                            if self.last_target_distance == np.inf:
                                self.last_target_distance = current_dist
                            
                            # SIMPLIFIED ROBUST STAGNATION LOGIC:
                            # Only reset counter if we significantly improved the GLOBAL MINIMUM distance (> 0.1m).
                            # This prevents resetting due to oscillation near a local minimum.
                            # We require CONTINUOUS improvement of the "best so far" distance.
                            
                            improved_min = False
                            if current_dist < self.min_target_distance - 0.1:
                                improved_min = True
                                self.min_target_distance = current_dist
                            elif current_dist < self.min_target_distance:
                                # Update min but don't reset counter if tiny improvement
                                self.min_target_distance = current_dist

                            if improved_min:
                                self.target_stagnation_counter = 0
                            else:
                                self.target_stagnation_counter += 1
                                
                            self.last_target_distance = current_dist

                            # If stuck trying to reach target for >10 steps (increased from 5 to allow for some wiggle), clear it
                            if self.target_stagnation_counter > 10:
                                print(f"Worker {self.meta_agent_id}: Stagnated while reaching Forced Target (Oscillation Detected). Clearing and Tabu-ing.", flush=True)
                                self.tabu_exploration_targets.append(self.current_exploration_target)
                                if len(self.tabu_exploration_targets) > 5:
                                    self.tabu_exploration_targets.pop(0)
                                self.current_exploration_target = None
                                self.target_stagnation_counter = 0
                                self.min_target_distance = np.inf
                                # Don't return here, let the loop continue to next step logic
                                return # Skip move this step to allow re-planning

                            try:
                                rounded_tgt = np.around(target_coords, 1)
                                tgt_key = (round(float(rounded_tgt[0]), 1), round(float(rounded_tgt[1]), 1))
                                tgt_node = self.robot.node_manager.nodes_dict.find(tgt_key)
                                if tgt_node is None:
                                    gf = getattr(self.env, "global_frontiers", [])
                                    gf_coords = []
                                    try:
                                        for it in gf:
                                            if isinstance(it, (list, tuple)) and len(it) == 2 and isinstance(it[0], (list, tuple, np.ndarray)) and np.isscalar(it[1]):
                                                gf_coords.append(np.array(it[0], dtype=float))
                                            elif isinstance(it, (list, tuple)) and len(it) == 2 and np.isscalar(it[0]) and np.isscalar(it[1]):
                                                gf_coords.append(np.array([float(it[0]), float(it[1])], dtype=float))
                                    except Exception:
                                        gf_coords = []
                                    self.robot.node_manager.add_node_to_dict(rounded_tgt, gf_coords, self.env.belief_info, self.env.floor_id)
                                    search_radius = 8.0
                                    bbox = quads.BoundingBox(
                                        min_x=rounded_tgt[0]-search_radius, min_y=rounded_tgt[1]-search_radius,
                                        max_x=rounded_tgt[0]+search_radius, max_y=rounded_tgt[1]+search_radius
                                    )
                                    nearby = self.robot.node_manager.nodes_dict.within_bb(bbox)
                                    tgt_node = self.robot.node_manager.nodes_dict.find(tgt_key)
                                    if tgt_node:
                                        curr = tgt_node.data
                                        cand = []
                                        for w in nearby:
                                            n = w.data
                                            if n.coords.tolist() == curr.coords.tolist():
                                                continue
                                            d = np.linalg.norm(n.coords - curr.coords)
                                            cand.append((d, n))
                                        cand.sort(key=lambda x: x[0])
                                        for d, n in cand[:3]:
                                            curr.neighbor_set.add((n.coords[0], n.coords[1]))
                                            n.neighbor_set.add((curr.coords[0], curr.coords[1]))
                                # Use get_shortest_path from HabitatEnv
                                path = self.env.get_shortest_path(self.robot.location, target_coords)
                                if path and len(path) > 1:
                                    chosen_pt = None
                                    for idx in range(1, len(path)):
                                        cand = path[idx]
                                        if np.linalg.norm(cand - self.robot.location) > 0.2:
                                            chosen_pt = cand
                                            break
                                    if chosen_pt is not None:
                                        print(f"Forced Exploration Move to {chosen_pt}", flush=True)
                                        self.env.step(chosen_pt, force=True)
                                        self.robot.update_planning_state(self.env.belief_info, self.env.robot_location, self.env.floor_id)
                                    else:
                                        print("Forced Move produced current location for all waypoints. Skipping.", flush=True)
                                        self.current_exploration_target = None
                                else:
                                    print(f"Forced Move Failed: No path to {target_coords}. Clearing target.", flush=True)
                                    self.tabu_exploration_targets.append(np.array(target_coords, dtype=float).flatten()[:2])
                                    if len(self.tabu_exploration_targets) > 5:
                                        self.tabu_exploration_targets.pop(0)
                                    self.current_exploration_target = None 
                                    self.target_stagnation_counter = 0
                                    
                                    # Fallback: Random valid neighbor (Only if path failed)
                                    # Get current node neighbors

                                    key = (round(float(self.robot.location[0]), 1), round(float(self.robot.location[1]), 1))
                                    curr_node = self.robot.node_manager.nodes_dict.find(key)
                                    
                                    # Fuzzy search if exact match fails
                                    if curr_node is None:
                                        bbox_size = 3.0 # Synced with agent.py and node_manager.py
                                        found_nodes = self.robot.node_manager.nodes_dict.within_bb(quads.BoundingBox(
                                            min_x=self.robot.location[0]-bbox_size, min_y=self.robot.location[1]-bbox_size,
                                            max_x=self.robot.location[0]+bbox_size, max_y=self.robot.location[1]+bbox_size
                                        ))
                                        if found_nodes:
                                            curr_node = min(found_nodes, key=lambda n: np.linalg.norm(np.array(n.data.coords) - self.robot.location))

                                    if curr_node:
                                        neighbors = list(curr_node.data.neighbor_set)
                                        # Filter out self-loops (current robot location) to prevent stagnation
                                        valid_neighbors = []
                                        for n in neighbors:
                                            if np.linalg.norm(np.array(n) - self.robot.location) > 0.1:
                                                valid_neighbors.append(n)
                                        
                                        if valid_neighbors:
                                            # Anti-Ping-Pong: Avoid going back to the immediate previous location if possible
                                            if len(self.position_history) >= 2 and len(valid_neighbors) > 1:
                                                prev_pos = self.position_history[-2][:2]
                                                non_return_neighbors = [n for n in valid_neighbors if np.linalg.norm(np.array(n) - np.array(prev_pos)) > 1.5]
                                                if len(self.position_history) >= 10:
                                                    window = [np.array(p[:2]) for p in self.position_history[-10:]]
                                                    def away_from_window(nc):
                                                        for rp in window:
                                                            if np.linalg.norm(np.array(nc) - rp) < 1.5:
                                                                return False
                                                        return True
                                                    non_return_neighbors = [n for n in non_return_neighbors if away_from_window(n)]
                                                if non_return_neighbors:
                                                    valid_neighbors = non_return_neighbors

                                            # Try neighbors in random order until one works
                                            random.shuffle(valid_neighbors)
                                            move_successful = False
                                            
                                            for neighbor in valid_neighbors:
                                                rn_pt = np.array(neighbor)
                                                # Try to verify reachability first
                                                path_check = self.env.get_shortest_path(self.robot.location, rn_pt)
                                                if path_check and len(path_check) > 1:
                                                    print(f"Forced Exploration Fallback: Moving to neighbor {rn_pt} (Path Valid)", flush=True)
                                                    step_pt = None
                                                    for j in range(1, len(path_check)):
                                                        cand2 = path_check[j]
                                                        if np.linalg.norm(cand2 - self.robot.location) > 0.2:
                                                            step_pt = cand2
                                                            break
                                                    if step_pt is not None:
                                                        self.env.step(step_pt, force=True) 
                                                        move_successful = True
                                                        break
                                                    else:
                                                        # If path exists but only returns current location, skip this neighbor
                                                        continue
                                            
                                            if not move_successful:
                                                # If all path checks fail, try direct step to a random one as last resort
                                                rn_pt = np.array(valid_neighbors[0])
                                                print(f"Forced Exploration Fallback: All paths failed. Forcing direct step to {rn_pt}", flush=True)
                                                self.env.step(rn_pt, force=True)
                                                
                                            self.robot.update_planning_state(self.env.belief_info, self.env.robot_location, self.env.floor_id)
                                        else:
                                             print(f"Forced Move Failed: No valid neighbors available (found {len(neighbors)} neighbors but all too close).", flush=True)
                                    else:
                                        print("Forced Move Failed: Current node not found.", flush=True)
                                        
                            except Exception as e:
                                print(f"Forced Move Error: {e}", flush=True)
                                import traceback
                                traceback.print_exc()
                else:
                    done = True
                    reward += 20 
            self.save_reward_done(reward, done)

            observation = self.robot.get_observation()
            ground_truth_observation = self.ground_truth_node_manager.get_ground_truth_observation(
                self.env.robot_location)
            self.save_next_observations(observation, ground_truth_observation)

            next_max_utility = 0
            if self.robot.utility is not None and len(self.robot.utility) > 0:
                next_max_utility = np.max(self.robot.utility)
            next_time_ratio = (i + 1) / MAX_EPISODE_STEP
            next_switch_input = torch.tensor([self.env.explored_rate, next_max_utility / 100.0, next_time_ratio], dtype=torch.float32).to(self.device)

            if len(self.stair_switch_buffer) > 0:
                self.stair_switch_buffer[-1]['next_state'] = next_switch_input

            if self.save_image:
                try:
                    plt.switch_backend('agg')
                    plt.close('all')
                    plt.figure(figsize=(12, 12))
                    # self.robot.plot_env() # Removed invalid call
                    # self.ground_truth_node_manager.plot_ground_truth_env(self.env.robot_location) # This might also use plt, so be careful
                    self.env.plot_env(i+1)
                    
                    filename = '{}/{}_{}.png'.format(gifs_path, self.global_step, i)
                    # plt.savefig(filename, dpi=100) # plot_env already saves
                    # plt.close()
                    # if hasattr(self.env, 'frame_files'):
                    #     self.env.frame_files.append(filename)
                except Exception as e:
                    print(f"Warning: Plotting failed at step {i}: {e}", flush=True)
                    import traceback
                    traceback.print_exc()

            if self.multi_floor_enabled and self.inter_floor_active:
                if i - self.inter_floor_start_step >= INTER_FLOOR_TIMEOUT_STEPS:

                    print(f"Step {i}: Inter-floor timeout. Resetting map (soft reset).", flush=True)
                    # Soft reset: Keep robot knowledge but refresh map state
                    self.env.reset_semantic_map() 
                    # self.robot.reset_for_new_map() # Don't clear robot memory completely if we can avoid it
                    # Instead of full reset, maybe just clear transient state?
                    # For now, stick to original logic but ensure it's logged
                    self.robot.reset_for_new_map(preserve_memory=True) # Assuming we add this flag or it exists
                    self.robot.update_planning_state(self.env.belief_info, self.env.robot_location, self.env.floor_id)
                    self.ground_truth_node_manager = GroundTruthNodeManager(self.robot.node_manager, self.env.ground_truth_info,
                                                                            device=self.device, plot=self.save_image)
                    self.inter_floor_active = False
                    self.multi_floor_enabled = False
                    self.target_stairs_index = None

            if done:
                break

        self.perf_metrics['travel_dist'] = self.env.travel_dist
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = done
        self.perf_metrics['num_stuck_events'] = self.stuck_events
        self.perf_metrics['num_stair_transitions'] = self.stair_transitions
        print(f"Episode metrics: {self.perf_metrics}")

        if len(self.stair_switch_buffer) > 0:
            valid_entries = [e for e in self.stair_switch_buffer if 'next_state' in e]
            if valid_entries:
                ss_state = torch.stack([e['state'] for e in valid_entries])
                ss_action = torch.stack([e['action'] for e in valid_entries])
                ss_reward = torch.stack([e['reward'] for e in valid_entries])
                ss_next_state = torch.stack([e['next_state'] for e in valid_entries])
                ss_done = torch.stack([e['done'] for e in valid_entries])
                
                self.episode_buffer.append([ss_state, ss_action, ss_reward, ss_next_state, ss_done])
            else:
                self.episode_buffer.append([])
        else:
            self.episode_buffer.append([])

        if self.save_image:
            make_gif(gifs_path, self.global_step, self.env.frame_files, self.env.explored_rate)

    def plot_stair_step(self, episode_step, stair_step_index, start_coords, end_coords, current_coords):
        if not self.save_image:
            return
        plt.switch_backend('agg')
        plt.close('all')
        plt.figure(figsize=(6, 6))
        xs = [start_coords[0], end_coords[0]]
        ys = [start_coords[1], end_coords[1]]
        plt.plot(xs, ys, 'k--')
        plt.plot(current_coords[0], current_coords[1], 'ro')
        plt.axis('equal')
        plt.axis('off')
        filename = '{}/{}_stair_{}_{}.png'.format(gifs_path, self.global_step, episode_step, stair_step_index)
        plt.savefig(filename, dpi=150)
        plt.close()
        if hasattr(self.env, 'frame_files'):
            self.env.frame_files.append(filename)

    def check_exploration_gain(self, step):
        """
        Check how many new pixels were explored since last check.
        Used for 'Buy Oil' heuristic without knowing total floor area.
        """
        if not hasattr(self.env, 'robot_belief'):
            return 0.0
            
        current_pixels = np.sum(self.env.robot_belief > 200) + np.sum(self.env.robot_belief < 50)
        
        if self.last_check_step == 0:
            self.last_explored_pixel_count = current_pixels
            self.last_check_step = step
            return 1.0
            
        steps_delta = step - self.last_check_step
        if steps_delta >= 50: # Check every 50 steps
            pixel_gain = current_pixels - self.last_explored_pixel_count
            # Normalize by steps to get a rate (pixels per step)
            self.recent_exploration_gain = pixel_gain / steps_delta
            
            self.last_explored_pixel_count = current_pixels
            self.last_check_step = step
            # print(f"Exploration Gain: {self.recent_exploration_gain:.2f} pixels/step", flush=True)
            
        return self.recent_exploration_gain

    def perform_stair_transition(self, episode_step, stairs_index):
        if self.env.stairs_coords_list is None or len(self.env.stairs_coords_list) == 0:
            return
        if stairs_index < 0 or stairs_index >= len(self.env.stairs_coords_list):
            return
        
        # Simplified Habitat transition
        if self.save_image:
            self.env.current_target_stairs_index = stairs_index
            print(f"Plotting pre-transition frame at step {episode_step} with target stair {stairs_index}")
            self.env.plot_env(f"{episode_step}_entering_stairs")
            
        print(f'Step {episode_step}: Reached stairs {stairs_index}, switching floor in Habitat...')
        self.env.switch_floor()
        
        self.robot.reset_for_new_map(preserve_memory=True)
        self.robot.update_planning_state(self.env.belief_info, self.env.robot_location, self.env.floor_id)
        self.ground_truth_node_manager = GroundTruthNodeManager(self.robot.node_manager, self.env.ground_truth_info,
                                                                device=self.device, plot=self.save_image)
        self.inter_floor_active = False
        self.target_stairs_index = None
        self.env.current_target_stairs_index = None
        self.stair_transitions += 1
        self.steps_on_current_floor = 0
        self.position_history = []
        self.stuck_events = 0  # Reset stuck counter on floor switch

        # Reset exploration gain tracking
        self.last_explored_pixel_count = 0
        self.last_check_step = episode_step
        self.recent_exploration_gain = 100.0 # High initial value to prevent immediate switch back

    def select_target_stairs(self):
        if not hasattr(self.env, "stairs_coords_list"):
            return None
        if self.env.stairs_coords_list is None or len(self.env.stairs_coords_list) == 0:
            return None
        coords = np.array(self.env.stairs_coords_list)
        robot = np.array(self.env.robot_location)
        dists_current = np.linalg.norm(coords - robot, axis=-1)
        best_idx = int(np.argmin(dists_current))
        return best_idx

    def save_observation(self, observation, ground_truth_observation):
        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask = observation
        self.episode_buffer[0].append(torch.as_tensor(node_inputs))
        self.episode_buffer[1].append(torch.as_tensor(node_padding_mask).bool())
        self.episode_buffer[2].append(torch.as_tensor(edge_mask).bool())
        self.episode_buffer[3].append(torch.as_tensor(current_index))
        self.episode_buffer[4].append(torch.as_tensor(current_edge))
        self.episode_buffer[5].append(torch.as_tensor(edge_padding_mask).bool())

        critic_node_inputs, critic_node_padding_mask, critic_edge_mask, critic_current_index, critic_current_edge, critic_edge_padding_mask = ground_truth_observation
        self.episode_buffer[15].append(torch.as_tensor(critic_node_inputs))
        self.episode_buffer[16].append(torch.as_tensor(critic_node_padding_mask).bool())
        self.episode_buffer[17].append(torch.as_tensor(critic_edge_mask).bool())
        self.episode_buffer[18].append(torch.as_tensor(critic_current_index))
        self.episode_buffer[19].append(torch.as_tensor(critic_current_edge))
        self.episode_buffer[20].append(torch.as_tensor(critic_edge_padding_mask).bool())

    def save_action(self, action_index):
        self.episode_buffer[6].append(action_index.reshape(1, 1, 1))

    def save_reward_done(self, reward, done):
        self.episode_buffer[7].append(torch.FloatTensor([reward]).reshape(1, 1, 1).to(self.device))
        self.episode_buffer[8].append(torch.tensor([int(done)]).reshape(1, 1, 1).to(self.device))
        
        if len(self.stair_switch_buffer) > 0:
             self.stair_switch_buffer[-1]['reward'] = torch.tensor([reward], dtype=torch.float32, device=self.device)
             self.stair_switch_buffer[-1]['done'] = torch.tensor([int(done)], dtype=torch.float32, device=self.device)

    def save_next_observations(self, observation, ground_truth_observation):
        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask = observation
        self.episode_buffer[9].append(torch.as_tensor(node_inputs))
        self.episode_buffer[10].append(torch.as_tensor(node_padding_mask).bool())
        self.episode_buffer[11].append(torch.as_tensor(edge_mask).bool())
        self.episode_buffer[12].append(torch.as_tensor(current_index))
        self.episode_buffer[13].append(torch.as_tensor(current_edge))
        self.episode_buffer[14].append(torch.as_tensor(edge_padding_mask).bool())

        critic_node_inputs, critic_node_padding_mask, critic_edge_mask, critic_current_index, critic_current_edge, critic_edge_padding_mask = ground_truth_observation
        self.episode_buffer[21].append(torch.as_tensor(critic_node_inputs))
        self.episode_buffer[22].append(torch.as_tensor(critic_node_padding_mask).bool())
        self.episode_buffer[23].append(torch.as_tensor(critic_edge_mask).bool())
        self.episode_buffer[24].append(torch.as_tensor(critic_current_index))
        self.episode_buffer[25].append(torch.as_tensor(critic_current_edge))
        self.episode_buffer[26].append(torch.as_tensor(critic_edge_padding_mask).bool())

if __name__ == "__main__":
    pass
