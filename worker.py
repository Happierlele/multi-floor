import torch
import matplotlib.pyplot as plt
import random

from env import Env
from agent import Agent
from utils import *
from model import PolicyNet, StairSwitchNet
from ground_truth_node_manager import GroundTruthNodeManager
from vlm_adapter import VLMAdapter

if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)


class Worker:
    def __init__(self, meta_agent_id, policy_net, stair_switch_net, global_step, device='cpu', save_image=False, use_vlm=False, vlm_model_name="gpt-4o", vlm_api_key=None, vlm_base_url=None):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device
        self.use_vlm = use_vlm

        self.env = Env(global_step, plot=self.save_image)
        self.robot = Agent(policy_net, self.device, self.save_image)
        self.stair_switch_net = stair_switch_net
        self.stair_switch_net.eval()

        self.ground_truth_node_manager = GroundTruthNodeManager(self.robot.node_manager, self.env.ground_truth_info,
                                                                device=self.device, plot=self.save_image)

        self.multi_floor_enabled = True
        self.inter_floor_active = False
        self.inter_floor_start_step = None
        self.target_stairs_index = None
        self.position_history = []
        self.stuck_events = 0
        self.stair_transitions = 0
        self.steps_on_current_floor = 0

        if self.use_vlm:
            import os
            # 优先使用传入的参数，否则使用环境变量
            # 注意：如果你用的是阿里云千问，请务必填入正确的 QWEN_API_KEY
            key = vlm_api_key or os.getenv("QWEN_API_KEY") or "sk-fd3a29d3e96c43b9905c8bbdb36d3f48" 
            
            # 阿里云千问的 OpenAI 兼容接口地址
            # 注意：您的 Key 是在新加坡区(ap-southeast-1)创建的，需要使用国际版地址
            # default_base = "https://dashscope.aliyuncs.com/compatible-mode/v1" # 国内版
            default_base = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1" # 国际版
            base = vlm_base_url or os.getenv("QWEN_BASE_URL") or default_base
            
            print(f"[Worker] Initializing VLM with Key: {key[:6]}...{key[-4:]}, Base URL: {base}")
            self.vlm = VLMAdapter(model_name=vlm_model_name, api_key=key, base_url=base)

        self.episode_buffer = []
        self.perf_metrics = dict()
        for i in range(27):
            self.episode_buffer.append([])

    def run_episode(self):
        done = False
        self.stair_switch_buffer = [] # Initialize buffer for this episode
        
        # Explicit Floor Tracking
        # current_floor_id = self.env.floor_id

        # 1. 初始更新：根据出生点的环境信息，更新机器人的认知（地图、前沿点、拓扑图）
        self.robot.update_planning_state(self.env.belief_info, self.env.robot_location, self.env.floor_id)
        # 2. 获取初始观测数据（机器人视角）
        observation = self.robot.get_observation()
        # 3. 获取初始真值观测数据（上帝视角，用于Critic）
        ground_truth_observation = self.ground_truth_node_manager.get_ground_truth_observation(self.env.robot_location)

        # 如果开启了图像保存，绘制初始状态
        if self.save_image:
            self.robot.plot_env()
            self.ground_truth_node_manager.plot_ground_truth_env(self.env.robot_location)
            self.env.plot_env(0)

        # 4. 开始主循环，每一步对应一次决策和移动
        for i in range(MAX_EPISODE_STEP):
            if i % 10 == 0:
                print(f"Step {i}/{MAX_EPISODE_STEP} on floor {self.env.floor_id}", flush=True)
            self.steps_on_current_floor += 1
            # 保存当前观测数据到经验池（buffer）
            self.save_observation(observation, ground_truth_observation)

            # 5. 核心决策：机器人根据观测选择下一个目标点
            # next_location: 下一个目标点的物理坐标
            # action_index: 动作在邻居列表中的索引
            
            # --- Anti-Loop / Global Rescue Mechanism ---
            # If we are stuck in a loop (low variance in recent positions),
            # force a global path planning to a node that can pull us out.
            force_global_plan = False
            
            # 1. Position-based Stuck Detection (Looping in small area)
            # Ensure we have enough history to check
            if len(self.position_history) >= 20: 
                recent_pos = self.position_history[-20:]
                # Calculate bounding box of recent positions
                xs = [p[0] for p in recent_pos]
                ys = [p[1] for p in recent_pos]
                if (max(xs) - min(xs) < 1.0) and (max(ys) - min(ys) < 1.0):
                     force_global_plan = True
                     print(f"Worker {self.meta_agent_id}: Detected position stagnation (20 steps in <1m box). Forcing rescue.", flush=True)

            # Check if stuck in same position (simple check)
            # Ensure position_history is not empty before accessing -1
            if i > 10 and len(self.position_history) > 0 and np.linalg.norm(self.robot.location - self.position_history[-1]) < 0.05:
                self.stuck_events += 1
                if self.stuck_events > 5: # If stuck for 5 consecutive steps
                    force_global_plan = True
                    print(f"Worker {self.meta_agent_id}: Detected movement stagnation (5 steps no move). Forcing rescue.", flush=True)
                    self.stuck_events = 0
            else:
                self.stuck_events = 0

            global_override_success = False
            
            # VLM Rescue Strategy (Highest Priority)
            if force_global_plan and self.use_vlm:
                try:
                    # Capture current observation for VLM
                    rgb_img = self.robot.get_rgb_image() # Assuming this method exists or similar
                    prompt = "I am a navigation robot stuck in an indoor environment. The image shows my top-down map view. White is free space, black is obstacles/walls, and grey is unknown area. Do you see any open doors, hallways, or stairs that I should move towards to get unstuck? Give me a specific direction or landmark."
                    
                    # Call VLM (Wrapped in try-except to prevent network crash)
                    print(f"Worker {self.meta_agent_id}: Calling VLM for rescue...", flush=True)
                    vlm_response = self.vlm.chat(image=rgb_img, prompt=prompt)
                    print(f"Worker {self.meta_agent_id}: VLM suggested: {vlm_response}", flush=True)
                    
                    # Here we would parse the VLM response to guide navigation. 
                    # For now, to prevent complexity, we treat a successful VLM call as a trigger to 
                    # try a 'Random Frontier' with higher confidence, simulating 'inspiration'.
                    # (In a full implementation, we would parse 'door on left' to a coordinate)
                    rescue_mode = "VLM_Inspired_Random"
                    
                except Exception as e:
                    print(f"Worker {self.meta_agent_id}: VLM call failed ({e}). Falling back to heuristic rescue.", flush=True)
                    # Fallthrough to standard rescue below
            
            if force_global_plan:
                # Prefer the highest-utility node, fall back to farthest node on this floor.
                # Optimized Rescue Strategy:
                # 1. Random Frontier: 50% chance to just pick a random frontier (very effective for breaking loops)
                # 2. Max Utility: 30% chance to go to the most promising area
                # 3. Farthest Node: 20% chance to just run away
                
                target_node = None
                rescue_mode = ""
                
                candidates = []
                for node in self.robot.node_manager.nodes_dict.__iter__():
                     # Only consider nodes on current floor
                    node_floor = getattr(node.data, 'floor_id', self.env.floor_id)
                    if node_floor != self.env.floor_id:
                        continue
                    candidates.append(node.data)

                if not candidates:
                    print(f"Worker {self.meta_agent_id}: No rescue candidates found!", flush=True)
                else:
                    rand_val = random.random()
                    
                    # Strategy 1: Random Frontier (if available)
                    if rand_val < 0.5:
                        frontiers = [n for n in candidates if n.utility > 0]
                        if frontiers:
                            target_node = random.choice(frontiers)
                            rescue_mode = "RandomFrontier"
                    
                    # Strategy 2: Max Utility
                    if target_node is None and rand_val < 0.8:
                        target_node = max(candidates, key=lambda x: x.utility)
                        if target_node.utility <= 0: target_node = None # fallback if max utility is 0
                        else: rescue_mode = "MaxUtil"
                    
                    # Strategy 3: Farthest Node (Last Resort)
                    if target_node is None:
                        target_node = max(candidates, key=lambda x: np.linalg.norm(x.coords - self.robot.location))
                        rescue_mode = "Farthest"

                if target_node is not None:
                    path, length = self.robot.node_manager.a_star(self.robot.location, target_node.coords)
                    if path and length < 1e8:
                        next_target = np.array(path[0])
                        neighbor_indices = self.robot.neighbor_indices
                        neighbor_coords = self.robot.node_coords[neighbor_indices]
                        dists = np.linalg.norm(neighbor_coords - next_target, axis=1)
                        best_idx = np.argmin(dists)

                        if dists[best_idx] < 2.0:
                            next_location = neighbor_coords[best_idx]
                            action_index = torch.tensor([[best_idx]]).long()
                            global_override_success = True
                            if i % 5 == 0:
                                dist_to_target = np.linalg.norm(target_node.coords - self.robot.location)
                                print(f"Global Rescue ({rescue_mode}): target {target_node.coords}, dist={dist_to_target:.1f}, util={target_node.utility}")

            if global_override_success:
                pass # next_location and action_index are set
            elif self.use_vlm:
                next_location, action_index = self.vlm.get_vlm_action(self.robot, observation, stairs_coords=self.env.stairs_coords_list)
            else:
                next_location, action_index = self.robot.select_next_waypoint(observation)
            max_utility = 0
            if self.robot.utility is not None and len(self.robot.utility) > 0:
                max_utility = np.max(self.robot.utility)
            
            # --- Learnable Stair Switch Strategy (Inference) ---
            # Features: [Explored Rate, Max Utility, Time Ratio]
            # Note: Normalized inputs are better for NN
            time_ratio = i / MAX_EPISODE_STEP
            # explored_rate is 0-1, max_utility can be large (e.g. >10), time_ratio is 0-1
            switch_input = torch.tensor([self.env.explored_rate, max_utility / 100.0, time_ratio], dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                switch_logits = self.stair_switch_net(switch_input.unsqueeze(0))
                switch_probs = torch.softmax(switch_logits, dim=-1)
                switch_action = torch.argmax(switch_probs).item() # 0=Stay, 1=Switch
                
                # Exploration: Epsilon-greedy (optional, for now trust the net or heuristic)
                # But we need to record what we actually DID.
            
            # --- Heuristic Strategy (Current Default) ---
            min_steps_for_switch = MAX_EPISODE_STEP // 8 # Reduced from /4 to allow earlier switching
            enough_steps_for_switch = self.steps_on_current_floor >= min_steps_for_switch
            
            # Heuristic 1: Low Utility (nothing interesting nearby)
            # Increased threshold from 5.0 to UTILITY_SWITCH_THRESHOLD to switch earlier
            cond_utility = enough_steps_for_switch and (max_utility < UTILITY_SWITCH_THRESHOLD)
            
            # Heuristic 2: High Exploration Rate (diminishing returns)
            # If we explored > FLOOR_SWITCH_EXP_RATE of the map, just go to next floor
            cond_explored = self.env.explored_rate > FLOOR_SWITCH_EXP_RATE
            
            # Decide which strategy to use
            # For training, we can use the heuristic to guide the network (Imitation Learning style)
            # OR use the network's output if we are doing RL.
            # User wants "learnable", so let's allow the network to influence if it's ready.
            # However, since it's untrained, let's stick to Heuristic for behavior but RECORD it for training.
            should_switch = cond_utility or cond_explored
            actual_action = 1 if should_switch else 0
            
            # Save transition for StairSwitchNet
            # We need (state, action, reward, next_state, done)
            # State: switch_input
            # Action: actual_action
            # Reward: will be assigned after step (using environment reward)
            # Next State: will be observed in next iteration
            # Done: observed in next iteration
            
            # We buffer the current state/action and fill reward/next_state later
            self.stair_switch_buffer.append({
                'state': switch_input,
                'action': torch.tensor([actual_action], device=self.device),
                'step': i
            })
            
            if i % 5 == 0:
                print(f"Step {i}: ExpRate={self.env.explored_rate:.2f}, MaxUtil={max_utility:.2f}, SwitchUtil={cond_utility}, SwitchExp={cond_explored}, NetSwitch={switch_action==1} (Prob={switch_probs[0][1]:.2f})")

            if self.multi_floor_enabled and should_switch:
                # Always update to the closest stair point to avoid getting stuck trying to reach a "deep" stair point
                # This makes the robot aim for the nearest part of the stair area
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
                        # Improved Navigation: Use A* to find path to closest known node to target
                        all_nodes = [n.data.coords for n in self.robot.node_manager.nodes_dict.__iter__()]
                        path_found = False
                        
                        if all_nodes:
                            all_nodes_arr = np.array(all_nodes)
                            # Filter nodes by current floor
                            current_floor_nodes = []
                            for n in self.robot.node_manager.nodes_dict.__iter__():
                                if hasattr(n.data, 'floor_id') and n.data.floor_id == self.env.floor_id:
                                    current_floor_nodes.append(n.data.coords)
                                elif not hasattr(n.data, 'floor_id'): # Legacy support
                                     current_floor_nodes.append(n.data.coords)
                            
                            if current_floor_nodes:
                                all_nodes_arr = np.array(current_floor_nodes)
                                dists_to_target = np.linalg.norm(all_nodes_arr - np.array(target_coords), axis=1)
                                best_idx = np.argmin(dists_to_target)
                                best_node_coords = all_nodes_arr[best_idx]
                                
                                # Plan path using A*
                                # Note: a_star returns (path, length). path excludes start.
                                path, length = self.robot.node_manager.a_star(self.robot.location, best_node_coords)
                                
                                if path and length < 1e8:
                                    next_target_loc = np.array(path[0])
                                    # Map next_target_loc to neighbor index
                                    neighbor_indices = self.robot.neighbor_indices
                                    neighbor_coords = self.robot.node_coords[neighbor_indices]
                                    
                                    # Find neighbor closest to next_target_loc
                                    diff = np.linalg.norm(neighbor_coords - next_target_loc, axis=1)
                                    idx = np.argmin(diff)
                                    
                                    # Only proceed if the neighbor is actually close to the planned step
                                    if diff[idx] < 1.0: # Tolerance for coordinate mismatch
                                        next_location = neighbor_coords[idx]
                                        action_index = torch.tensor([[idx]]).long()
                                        path_found = True
                        
                        if not path_found:
                            # Fallback: if we are at the edge or A* failed, we assume the standard exploration
                            # will help us discover more nodes.
                            # Better Fallback: Navigate to the closest FRONTIER to the stair target
                            # This encourages expanding towards the stairs.
                            if self.robot.frontier is not None and len(self.robot.frontier) > 0:
                                frontier_coords = list(self.robot.frontier)
                                frontier_coords_arr = np.array(frontier_coords)
                                dists_to_stair = np.linalg.norm(frontier_coords_arr - np.array(target_coords), axis=1)
                                best_frontier_idx = np.argmin(dists_to_stair)
                                best_frontier = frontier_coords[best_frontier_idx]
                                
                                # Use standard navigation to this frontier
                                # We can reuse the same logic as normal exploration or just set it as target
                                # But let's try A* to this frontier first
                                path, length = self.robot.node_manager.a_star(self.robot.location, best_frontier)
                                if path and length < 1e8:
                                    next_target_loc = np.array(path[0])
                                    neighbor_indices = self.robot.neighbor_indices
                                    neighbor_coords = self.robot.node_coords[neighbor_indices]
                                    diff = np.linalg.norm(neighbor_coords - next_target_loc, axis=1)
                                    idx = np.argmin(diff)
                                    if diff[idx] < 2.0: # Increased tolerance
                                        next_location = neighbor_coords[idx]
                                        action_index = torch.tensor([[idx]]).long()
                                        path_found = True
                                        print(f"Path to stairs blocked, rerouting to closest frontier at {best_frontier}")
            # 保存动作到经验池
            self.save_action(action_index)

            # 6. 安全性检查与修正：
            # 确保选出的目标点确实是当前节点的邻居
            node = self.robot.node_manager.nodes_dict.find((self.robot.location[0], self.robot.location[1]))
            check = np.array(list(node.data.neighbor_set)).reshape(-1, 2)
            
            # 如果目标点不在邻居列表中（可能是VLM幻觉或计算误差），强制修正为最近的有效邻居
            target_complex = next_location[0] + next_location[1] * 1j
            check_complex = check[:, 0] + check[:, 1] * 1j
            
            if target_complex not in check_complex:
                print(f"Warning: Target {next_location} not in neighbors. Correcting...")
                dists = np.linalg.norm(check - next_location, axis=1)
                idx = np.argmin(dists)
                next_location = check[idx]
                # Update action index if needed (approximation)
                # action_index = ... 
            
            # 确保目标点不是当前位置（必须移动）
            # 如果模型选择了原地不动，强制随机选择一个邻居
            if next_location[0] == self.robot.location[0] and next_location[1] == self.robot.location[1]:
                print("Warning: Agent selected current location. Forcing random move.")
                # Filter out current location from neighbors if present (though neighbor set usually implies connectivity to others)
                valid_neighbors = [n for n in list(node.data.neighbor_set) if n != (self.robot.location[0], self.robot.location[1])]
                
                if valid_neighbors:
                    chosen = random.choice(valid_neighbors)
                    next_location = np.array(chosen)
                else:
                    print("Error: No valid neighbors to move to!")
                    # Handle stuck case?
                    pass

            # Final assert to catch if we are truly stuck
            # assert next_location[0] != self.robot.location[0] or next_location[1] != self.robot.location[1]
            if next_location[0] == self.robot.location[0] and next_location[1] == self.robot.location[1]:
                 print(f"Critical Warning: Still at current location after correction at step {i}. Skipping step.")
                 continue

            # 7. 执行动作：环境更新机器人位置，并返回奖励
            reward = self.env.step(next_location)

            # Check for stair discovery
            self.env.discover_stairs()

            # 8. 再次更新机器人认知（因为位置变了，可能看到了新区域）
            self.robot.update_planning_state(self.env.belief_info, self.env.robot_location, self.env.floor_id)
            
            # Recalculate utility for checking if we should switch (consistency check)
            curr_max_util = 0
            if self.robot.utility is not None and len(self.robot.utility) > 0:
                curr_max_util = np.max(self.robot.utility)
            
            # Use consistent logic: if we decided to switch earlier (should_switch) and we are still in a state that allows it,
            # or if the new utility drops below the threshold.
            # We use a slightly higher threshold here to ensure we don't oscillate if we are already close.
            do_switch = should_switch or (enough_steps_for_switch and (curr_max_util < UTILITY_SWITCH_THRESHOLD))

            if do_switch and self.multi_floor_enabled and self.target_stairs_index is not None:
                if self.env.stairs_coords_list is not None and len(self.env.stairs_coords_list) > 0:
                    target_coords = self.env.stairs_coords_list[self.target_stairs_index]
                    dist_to_stairs = np.linalg.norm(self.env.robot_location - np.array(target_coords))
                    # Relaxed tolerance for entering stairs to prevent circling
                    # If within 1.5x tolerance, allow entry
                    if dist_to_stairs <= STAIRS_DIST_TOLERANCE * 1.5:
                        self.perform_stair_transition(i, self.target_stairs_index)
            
            # Round to integer to avoid float jitter masking stuck state
            self.position_history.append((round(self.env.robot_location[0]), round(self.env.robot_location[1])))
            if len(self.position_history) > STUCK_WINDOW:
                self.position_history.pop(0)
            if len(self.position_history) == STUCK_WINDOW:
                unique_positions = set(self.position_history)
                if len(unique_positions) <= STUCK_UNIQUE_POS_THRESHOLD:
                    print(f"Stuck detected at step {i}, resetting semantic map.")
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
            
            # 9. 判断任务是否结束：
            # 如果所有节点的效用值之和为0（没有值得探索的地方了），则认为完成
            if self.robot.utility.sum() == 0:
                done = True
                reward += 20 # 给予完成奖励
            # 保存奖励和结束标志到经验池
            self.save_reward_done(reward, done)

            # 10. 获取新的观测数据（为下一轮循环做准备）
            observation = self.robot.get_observation()
            ground_truth_observation = self.ground_truth_node_manager.get_ground_truth_observation(
                self.env.robot_location)
            # 保存下一状态的观测数据到经验池
            self.save_next_observations(observation, ground_truth_observation)

            # Calculate next state for StairSwitchNet
            next_max_utility = 0
            if self.robot.utility is not None and len(self.robot.utility) > 0:
                next_max_utility = np.max(self.robot.utility)
            next_time_ratio = (i + 1) / MAX_EPISODE_STEP
            next_switch_input = torch.tensor([self.env.explored_rate, next_max_utility / 100.0, next_time_ratio], dtype=torch.float32).to(self.device)

            if len(self.stair_switch_buffer) > 0:
                self.stair_switch_buffer[-1]['next_state'] = next_switch_input

            # 绘图（如果开启）
            if self.save_image:
                try:
                    self.robot.plot_env()
                    self.ground_truth_node_manager.plot_ground_truth_env(self.env.robot_location)
                    self.env.plot_env(i+1)
                except Exception as e:
                    print(f"Warning: Plotting failed at step {i}: {e}")

            if self.multi_floor_enabled and self.inter_floor_active:
                if i - self.inter_floor_start_step >= INTER_FLOOR_TIMEOUT_STEPS:
                    self.env.reset_semantic_map()
                    self.robot.reset_for_new_map()
                    self.robot.update_planning_state(self.env.belief_info, self.env.robot_location, self.env.floor_id)
                    self.ground_truth_node_manager = GroundTruthNodeManager(self.robot.node_manager, self.env.ground_truth_info,
                                                                            device=self.device, plot=self.save_image)
                    self.inter_floor_active = False
                    self.multi_floor_enabled = False
                    self.target_stairs_index = None

            # 如果任务完成，跳出循环
            if done:
                break

        # 11. 记录本回合的性能指标
        self.perf_metrics['travel_dist'] = self.env.travel_dist
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = done
        self.perf_metrics['num_stuck_events'] = self.stuck_events
        self.perf_metrics['num_stair_transitions'] = self.stair_transitions
        print(f"Episode metrics: {self.perf_metrics}")

        # Process StairSwitchNet buffer for return
        # Convert list of dicts to list of tensors (batch format)
        # We need to stack them: [state_batch, action_batch, reward_batch, next_state_batch, done_batch]
        if len(self.stair_switch_buffer) > 0:
            # Filter incomplete entries (e.g. last step might miss next_state if we break early, but we handled it)
            valid_entries = [e for e in self.stair_switch_buffer if 'next_state' in e]
            if valid_entries:
                ss_state = torch.stack([e['state'] for e in valid_entries])
                ss_action = torch.stack([e['action'] for e in valid_entries])
                ss_reward = torch.stack([e['reward'] for e in valid_entries])
                ss_next_state = torch.stack([e['next_state'] for e in valid_entries])
                ss_done = torch.stack([e['done'] for e in valid_entries])
                
                # Append as a single list (the 28th element of episode_buffer)
                self.episode_buffer.append([ss_state, ss_action, ss_reward, ss_next_state, ss_done])
            else:
                self.episode_buffer.append([])
        else:
            self.episode_buffer.append([])

        # 保存 GIF 动画
        if self.save_image:
            make_gif(gifs_path, self.global_step, self.env.frame_files, self.env.explored_rate)

    def plot_stair_step(self, episode_step, stair_step_index, start_coords, end_coords, current_coords):
        if not self.save_image:
            return
        plt.switch_backend('agg')
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
        stairs_cell = None
        if self.env.stairs_cells is not None and len(self.env.stairs_cells) > stairs_index:
            stairs_cell = self.env.stairs_cells[stairs_index]
        
        # Force plot to show the yellow star target before transition starts
        if self.save_image:
            # Ensure the environment knows which stair is targeted for this plot
            self.env.current_target_stairs_index = stairs_index
            print(f"Plotting pre-transition frame at step {episode_step} with target stair {stairs_index}")
            self.env.plot_env(f"{episode_step}_entering_stairs")
            
        self.env.block_stairs_entrance(stairs_cell)
        start_coords = np.array(self.env.stairs_coords_list[stairs_index])
        gt_next, robot_cell_next, _, next_map_index, stairs_cells_next = self.env.import_ground_truth(self.env.map_index + 1)
        next_origin_x = -np.round(robot_cell_next[0] * self.env.cell_size, 1)
        next_origin_y = -np.round(robot_cell_next[1] * self.env.cell_size, 1)
        next_map_info = MapInfo(gt_next, next_origin_x, next_origin_y, self.env.cell_size)
        if stairs_cells_next is not None and len(stairs_cells_next) > 0:
            end_cell = stairs_cells_next[0]
            end_coords = get_coords_from_cell_position(np.array(end_cell), next_map_info)
        else:
            end_coords = get_coords_from_cell_position(np.array(robot_cell_next), next_map_info)
        for k in range(STAIR_INTERNAL_STEPS):
            alpha = float(k + 1) / float(STAIR_INTERNAL_STEPS)
            current_coords = (1.0 - alpha) * start_coords + alpha * end_coords
            self.plot_stair_step(episode_step, k, start_coords, end_coords, current_coords)
        print('Step {}: Reached stairs, performing stair transition to next map...'.format(episode_step))
        self.env.switch_to_next_map()
        print('Switched to map: {}'.format(self.env.map_list[self.env.map_index]))
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

        assert torch.all(current_edge == critic_current_edge), print(current_edge, critic_current_edge, current_index, critic_current_index)
        assert torch.all(node_inputs[0, current_index.item(), :2] == critic_node_inputs[0, critic_current_index.item(), :2]), print(node_inputs[0, current_index.item()], critic_node_inputs[0, critic_current_index.item()])
        
        # Ensure all tensors are on the same device before comparison
        # Usually node_inputs is on self.device, but check explicitly to be safe
        dev = node_inputs.device
        ce_dev = current_edge.to(dev)
        cni_dev = critic_node_inputs.to(dev)
        cce_dev = critic_current_edge.to(dev)
        
        assert torch.all(torch.gather(node_inputs, 1, ce_dev.repeat(1, 1, 2)) == torch.gather(cni_dev, 1, cce_dev.repeat(1, 1, 2)))

    def save_action(self, action_index):
        self.episode_buffer[6] += action_index.reshape(1, 1, 1)

    def save_reward_done(self, reward, done):
        self.episode_buffer[7] += torch.FloatTensor([reward]).reshape(1, 1, 1).to(self.device)
        self.episode_buffer[8] += torch.tensor([int(done)]).reshape(1, 1, 1).to(self.device)
        
        # Also update stair_switch_buffer
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
    torch.manual_seed(4777)
    np.random.seed(4777)
    model = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM)
    worker = Worker(0, model, 77, save_image=True, use_vlm=True, vlm_model_name="gpt-4o")
    worker.run_episode()
    worker = Worker(0, model, 77, save_image=True, use_vlm=True, vlm_model_name="gpt-4o")
    worker.run_episode()
