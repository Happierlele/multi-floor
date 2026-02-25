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
            key = vlm_api_key or os.getenv("QWEN_API_KEY") or "sk-fd3a29d3e96c43b9905c8bbdb36d3f48" 
            default_base = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1" 
            base = vlm_base_url or os.getenv("QWEN_BASE_URL") or default_base
            
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
        
        self.episode_buffer = []
        for i in range(27):
            self.episode_buffer.append([])
        self.perf_metrics = dict()

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
                 self.env.robot_belief = self.env.mapper.update(obs['depth_sensor'], agent_state)
                 self.env.update_robot_location_from_sim(agent_state)
                 
                 print(f"[Worker] Teleported to new floor {self.env.floor_id} at {target_pos}", flush=True)
                 
                 # Add explicit graph connection (Stair Edge)
                 self.robot.node_manager.add_stair_connection(old_floor_id, old_location, self.env.floor_id, self.env.robot_location)
             except:
                 pass

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

        if self.save_image:
            # self.robot.plot_env() # Agent does not have plot_env
            self.ground_truth_node_manager.plot_ground_truth_env(self.env.robot_location)
            self.env.plot_env(0)

        for i in range(MAX_EPISODE_STEP):
            if i % 10 == 0:
                print(f"Step {i}/{MAX_EPISODE_STEP} on floor {self.env.floor_id}", flush=True)
            self.steps_on_current_floor += 1
            self.save_observation(observation, ground_truth_observation)

            # Periodic Plotting (Matches 2D behavior)
            if self.save_image and i % SAVE_IMG_GAP == 0:
                 self.env.plot_env(i)

            force_global_plan = False
            is_rescue_move = False
            
            if len(self.position_history) >= 20: 
                recent_pos = self.position_history[-20:]
                xs = [p[0] for p in recent_pos]
                ys = [p[1] for p in recent_pos]
                if (max(xs) - min(xs) < 2.0) and (max(ys) - min(ys) < 2.0):
                     force_global_plan = True
                     print(f"Worker {self.meta_agent_id}: Detected position stagnation (20 steps in <2m box). Forcing rescue.", flush=True)

            if i > 10 and len(self.position_history) > 0 and np.linalg.norm(self.robot.location - self.position_history[-1]) < 0.05:
                self.stuck_events += 1
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
                        valid_neighbors = [n for n in neighbors if np.linalg.norm(np.array(n) - self.robot.location) > 0.1]
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

            if not global_override_success and self.use_vlm and (i % 10 == 0 or self.inter_floor_active):
                rgb_img = getattr(self.env, "rgb_image", None)
                
                # --- VERIFICATION: Save first image to verify 3D works ---
                if i == 0 and rgb_img is not None:
                    try:
                        from PIL import Image
                        debug_img_path = f"debug_view_agent{self.meta_agent_id}_step0.png"
                        Image.fromarray(rgb_img).save(debug_img_path)
                        print(f"Worker {self.meta_agent_id}: SAVED DEBUG 3D VIEW to {debug_img_path}", flush=True)
                    except Exception as e:
                        print(f"Worker {self.meta_agent_id}: Failed to save debug image: {e}", flush=True)
                # ---------------------------------------------------------

                print(f"Worker {self.meta_agent_id}: Calling VLM navigation logic (Step {i})...", flush=True)
                debug_path = f"{gifs_path}/vlm_nav_{self.meta_agent_id}_step_{i}.png"
                next_location, action_index = self.vlm.get_vlm_action(self.robot, observation, stairs_coords=self.env.stairs_coords_list, rgb_image=rgb_img, debug_save_path=debug_path)
                # If VLM provides a location, treat it as a rescue/override move to allow graph jumps if needed
                if next_location is not None:
                    is_rescue_move = True
            else:
                next_location, action_index = self.robot.select_next_waypoint(observation)
                # print(f"Worker {self.meta_agent_id}: Model Output Action: {action_index}, Next Loc: {next_location}", flush=True)
            
            max_utility = 0
            if self.robot.utility is not None and len(self.robot.utility) > 0:
                max_utility = np.max(self.robot.utility)
            
            time_ratio = i / MAX_EPISODE_STEP
            switch_input = torch.tensor([self.env.explored_rate, max_utility / 100.0, time_ratio], dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                switch_logits = self.stair_switch_net(switch_input.unsqueeze(0))
                switch_probs = torch.softmax(switch_logits, dim=-1)
                switch_action = torch.argmax(switch_probs).item() 
                
            min_steps_for_switch = MAX_EPISODE_STEP // 8 
            enough_steps_for_switch = self.steps_on_current_floor >= min_steps_for_switch
            
            cond_utility = enough_steps_for_switch and (max_utility < UTILITY_SWITCH_THRESHOLD)
            cond_explored = self.env.explored_rate > FLOOR_SWITCH_EXP_RATE
            
            should_switch = cond_utility or cond_explored
            actual_action = 1 if should_switch else 0
            
            self.stair_switch_buffer.append({
                'state': switch_input,
                'action': torch.tensor([actual_action], device=self.device),
                'step': i
            })
            
            if i % 5 == 0:
                print(f"Step {i}: ExpRate={self.env.explored_rate:.2f}, MaxUtil={max_utility:.2f}, SwitchUtil={cond_utility}, SwitchExp={cond_explored}, NetSwitch={switch_action==1} (Prob={switch_probs[0][1]:.2f})")

            if self.multi_floor_enabled and should_switch:
                best_idx = self.select_target_stairs()
                if best_idx is not None:
                    self.target_stairs_index = best_idx
                    self.env.current_target_stairs_index = self.target_stairs_index
                
                if not self.inter_floor_active:
                    self.inter_floor_active = True
                    self.inter_floor_start_step = i

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
                                                self.perform_stair_transition(self.episode_step, stair_idx)
                                                # After transition, we need to break/continue to update state
                                                # Update state for rest of loop
                                                next_location = self.robot.location # Stay at new location
                                                action_index = torch.tensor([[0]]).long().to(self.device) # Dummy action
                                                path_found = True # Handled
                                            else:
                                                print("[Worker] Error: A* planned floor switch but no stairs found!")
                                    
                                    if target_floor == self.env.floor_id: # Only move if on same floor (or after transition logic handled elsewhere?)
                                        # Actually if we just transitioned, we are now on target_floor (maybe).
                                        # But perform_stair_transition updates self.env.floor_id.
                                        # So if we transitioned, self.env.floor_id IS NOW target_floor.
                                        # But we should probably replan from new location.
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
                print(f"Critical Warning: No node found near robot at {self.robot.location}. Skipping neighbor check.", flush=True)
                check = np.empty((0, 2))
            else:
                check = np.array(list(node.data.neighbor_set)).reshape(-1, 2)
            
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
                    print(f"Error: No valid neighbors in graph for node {node.data.coords if node else 'None'}! Attempting emergency recovery...")
                    # Recovery: Look for closest node in all nodes that is reachable
                    all_nodes = [n.data for n in self.robot.node_manager.nodes_dict.__iter__()]
                    candidates = []
                    candidates_unsafe = []
                    
                    for n in all_nodes:
                        if n.coords.tolist() == self.robot.location.tolist():
                             continue
                        
                        dist = np.linalg.norm(n.coords - self.robot.location)
                        # Increased search range to handle sparse graphs (15m radius)
                        if dist < 15.0:
                             # STRICT VALIDITY CHECK: Ensure candidate is within map bounds and navigable
                             # This prevents jumping to "void" areas or out-of-bounds
                             if hasattr(self.env, 'is_valid_location') and not self.env.is_valid_location(n.coords):
                                 continue

                             candidates_unsafe.append(n.coords)
                             # Check collision
                             if not check_collision(self.robot.location, n.coords, self.robot.updating_map_info):
                                 candidates.append(n.coords)
                    
                    if candidates:
                         print(f"Recovery: Found {len(candidates)} reachable nodes nearby. Picking closest.")
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

                    elif candidates_unsafe:
                         print(f"Recovery: No collision-free nodes found. FORCING unsafe move to closest VALID node to break loop.")
                         candidates_unsafe.sort(key=lambda c: np.linalg.norm(c - self.robot.location))
                         valid_neighbors = [candidates_unsafe[0]]
                         
                         # Force add neighbor connection
                         if node:
                             neighbor_coords = candidates_unsafe[0]
                             node.data.neighbor_set.add((neighbor_coords[0], neighbor_coords[1]))
                             neighbor_node_wrapper = self.robot.node_manager.nodes_dict.find((neighbor_coords[0], neighbor_coords[1]))
                             if neighbor_node_wrapper:
                                 neighbor_node_wrapper.data.neighbor_set.add((node.data.coords[0], node.data.coords[1]))
                    else:
                        print("Critical Error: No valid candidates found for recovery! Agent might be completely isolated or out of bounds.")

                if valid_neighbors:
                    chosen = random.choice(valid_neighbors)
                    next_location = np.array(chosen)
                else:
                    print("Error: No valid neighbors to move to!")
                    pass

            if next_location[0] == self.robot.location[0] and next_location[1] == self.robot.location[1]:
                 print(f"Critical Warning: Still at current location after correction at step {i}. Skipping step.")
                 continue

            reward = self.env.step(next_location)

            # In Habitat, stairs discovery might be handled internally or via collision/proximity
            # For compatibility, we check for stairs in env
            # self.env.discover_stairs() # HabitatEnv handles this in step()

            self.robot.update_planning_state(self.env.belief_info, self.env.robot_location, self.env.floor_id)
            
            curr_max_util = 0
            if self.robot.utility is not None and len(self.robot.utility) > 0:
                curr_max_util = np.max(self.robot.utility)
            
            do_switch = should_switch or (enough_steps_for_switch and (curr_max_util < UTILITY_SWITCH_THRESHOLD))

            if do_switch and self.multi_floor_enabled and self.target_stairs_index is not None:
                if self.env.stairs_coords_list is not None and len(self.env.stairs_coords_list) > 0:
                    target_coords = self.env.stairs_coords_list[self.target_stairs_index]
                    dist_to_stairs = np.linalg.norm(self.env.robot_location - np.array(target_coords))
                    if dist_to_stairs <= STAIRS_DIST_TOLERANCE * 1.5:
                        self.perform_stair_transition(i, self.target_stairs_index)
            
            self.position_history.append((round(self.env.robot_location[0], 2), round(self.env.robot_location[1], 2)))
            if len(self.position_history) > STUCK_WINDOW:
                self.position_history.pop(0)
            if len(self.position_history) == STUCK_WINDOW:
                unique_positions = set(self.position_history)
                if len(unique_positions) <= STUCK_UNIQUE_POS_THRESHOLD:
                    print(f"Stuck detected at step {i}, resetting semantic map (NOT a floor switch).", flush=True)
                    self.stuck_events += 1
                    self.env.reset_semantic_map()
                    self.robot.reset_for_new_map(preserve_memory=True)
                    self.robot.update_planning_state(self.env.belief_info, self.env.robot_location, self.env.floor_id)
                    self.ground_truth_node_manager = GroundTruthNodeManager(self.robot.node_manager, self.env.ground_truth_info,
                                                                            device=self.device, plot=self.save_image)
                    self.inter_floor_active = False
                    self.target_stairs_index = None
                    self.env.current_target_stairs_index = None
                    self.position_history = []
            
            if self.robot.utility.sum() == 0:
                self.zero_utility_steps += 1
            else:
                self.zero_utility_steps = 0

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
                        
                        # 0. Try Global Frontiers (Best for Exploration)
                        if hasattr(self.env, "global_frontiers") and self.env.global_frontiers:
                            frontiers = self.env.global_frontiers
                            if frontiers:
                                # Optimized: Select frontier based on Utility (Size) and Cost (Distance)
                                best_frontier = None
                                best_score = -1.0
                                
                                for item in frontiers:
                                    # Handle both new format ((x,y), area) and old format (x,y)
                                    if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], tuple):
                                        f_coords, f_area = item
                                    else:
                                        f_coords = item
                                        f_area = 10.0 # Default fallback
                                    
                                    # TABU CHECK: Skip recently visited frontiers
                                    is_tabu = False
                                    for t_coord in self.tabu_frontiers:
                                        if np.linalg.norm(np.array(f_coords) - np.array(t_coord)) < 2.0: # 2.0m radius to be safe
                                            is_tabu = True
                                            break
                                    
                                    if is_tabu:
                                        continue

                                    dist = np.linalg.norm(np.array(f_coords) - self.robot.location)
                                    
                                    # Score function: Favor large frontiers, penalize distance
                                    # alpha controls distance penalty. 
                                    score = f_area / (1.0 + dist * 0.5)
                                    
                                    if score > best_score:
                                        best_score = score
                                        best_frontier = f_coords
                                
                                if best_frontier:
                                    target_coords = best_frontier
                                    # Add to tabu list
                                    self.tabu_frontiers.append(best_frontier)
                                    if len(self.tabu_frontiers) > 8: # Keep last 8
                                        self.tabu_frontiers.pop(0)
                                        
                                    print(f"Worker {self.meta_agent_id}: Zero Utility. Moving to best Frontier (Score {best_score:.2f}): {target_coords}", flush=True)

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
                                next_loc, _ = self.vlm.get_vlm_action(self.robot, None, self.env.stairs_coords_list, rgb_image=rgb_img, debug_save_path=debug_path)
                                
                                if next_loc is not None:
                                    # Verify it's not current location
                                    if np.linalg.norm(next_loc - self.robot.location) > 0.1:
                                        print(f"Worker {self.meta_agent_id}: VLM suggested move to {next_loc}", flush=True)
                                        target_coords = next_loc
                                    else:
                                        print(f"Worker {self.meta_agent_id}: VLM suggested current location. Ignoring.", flush=True)
                            except Exception as e:
                                print(f"Worker {self.meta_agent_id}: VLM Rescue failed: {e}", flush=True)

                        # 2. Force a random nearby node as target (if VLM didn't pick one)
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
                                
                                # TABU LOGIC: Filter out recently visited targets to prevent loops
                                # Use position_history as a rough proxy for tabu list
                                recent_positions = [tuple(p) for p in self.position_history[-10:]] # Check last 10 steps
                                
                                non_tabu_candidates = []
                                for n in best_candidates:
                                    n_coords = (n.coords[0], n.coords[1])
                                    # Check if we were near this node recently (fuzzy match)
                                    is_tabu = False
                                    for p in recent_positions:
                                        if np.linalg.norm(np.array(n_coords) - np.array(p)) < 1.0:
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
                            else:
                                # Fallback to random if lookup failed
                                target_coords = random.choice(candidates)
                        
                        # INJECTED FIX: Manually force robot to move towards this target to break stagnation
                        # This bypasses the neural network which might be outputting "stay" or invalid actions due to 0 utility
                        if target_coords is not None:
                            try:
                                # Use get_shortest_path from HabitatEnv
                                path = self.env.get_shortest_path(self.robot.location, target_coords)
                                if path and len(path) > 1:
                                    next_pt = path[1] # First step towards target
                                    # Execute move
                                    print(f"Forced Exploration Move to {next_pt}", flush=True)
                                    self.env.step(next_pt)
                                    self.robot.update_planning_state(self.env.belief_info, self.env.robot_location, self.env.floor_id)
                                else:
                                    print(f"Forced Move Failed: No path to {target_coords}. Trying random neighbor...", flush=True)
                                    # Fallback: Random valid neighbor
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
                                                prev_pos = self.position_history[-2] # Last recorded position
                                                # Filter out neighbors that are too close to previous position
                                                non_return_neighbors = [n for n in valid_neighbors if np.linalg.norm(np.array(n) - np.array(prev_pos)) > 1.0]
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
                                                    # Use the next waypoint from path to ensure valid movement
                                                    self.env.step(path_check[1]) 
                                                    move_successful = True
                                                    break
                                            
                                            if not move_successful:
                                                # If all path checks fail, try direct step to a random one as last resort
                                                rn_pt = np.array(valid_neighbors[0])
                                                print(f"Forced Exploration Fallback: All paths failed. Forcing direct step to {rn_pt}", flush=True)
                                                self.env.step(rn_pt)
                                                
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
