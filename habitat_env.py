import os
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# 尝试导入 habitat_sim (核心仿真器)
import habitat_sim
# try:
#     import habitat_sim
# except ImportError:
#     print("Warning: habitat_sim not installed. HabitatEnv will not work.")
#     print("Please follow instructions in habitat_migration_guide.md")

# 尝试导入 habitat (High-level API, 可选)
try:
    import habitat
except ImportError:
    pass # 暂时只需要 sim 即可运行

from habitat_mapper import SimpleMapper
from utils import MapInfo, get_grid_path, get_cell_position_from_coords, get_coords_from_cell_position
from parameter_clean import gifs_path, FREE, OCCUPIED, UNKNOWN
from habitat_env_methods import plot_env, reset_semantic_map

# 复用原有的常量
CELL_SIZE = 0.4 
SENSOR_RANGE = 10.0

class HabitatEnv:
    # Bind methods from habitat_env_methods.py
    plot_env = plot_env
    reset_semantic_map = reset_semantic_map

    def __init__(self, episode_index, plot=False):
        self.episode_index = episode_index
        self.plot = plot
        self.floor_id = 0
        
        # 1. 初始化 Habitat 仿真器
        self.sim = self._init_habitat()
        
        # 2. 初始化映射器 (3D -> 2D)
        # Calculate dynamic bounds from PathFinder if available
        if self.sim and self.sim.pathfinder.is_loaded:
            bounds = self.sim.pathfinder.get_bounds()
            min_x, min_z = bounds[0][0], bounds[0][2]
            max_x, max_z = bounds[1][0], bounds[1][2]
            
            # Add padding
            padding = 10.0
            min_x -= padding
            min_z -= padding
            max_x += padding
            max_z += padding
            
            width = max_x - min_x
            height = max_z - min_z
            map_size_meters = max(width, height)
            
            # Align origin to cell grid
            origin_x = min_x
            origin_y = min_z
            print(f"[HabitatEnv] Dynamic Map Size: {map_size_meters:.2f}m, Origin: ({origin_x:.2f}, {origin_y:.2f})", flush=True)
        else:
            map_size_meters = 100.0
            origin_x = -50.0
            origin_y = -50.0
            
        self.mapper = SimpleMapper(CELL_SIZE, map_size_meters, origin_x, origin_y)
        
        # 3. 初始化状态
        self.robot_location = np.array([0.0, 0.0]) # 虚拟的 2D 坐标
        self.robot_belief = self.mapper.reset()
        
        # 适配原有的 MapInfo 结构
        h, w = self.robot_belief.shape
        self.belief_origin_x = origin_x
        self.belief_origin_y = origin_y
        self.belief_info = MapInfo(self.robot_belief, self.belief_origin_x, self.belief_origin_y, CELL_SIZE)
        
        # Ground Truth 信息在真实探索中是不可知的，但在仿真中我们可以获取
        # 使用 PathFinder 生成真实的 TopDown Map
        self.ground_truth = self._get_ground_truth_map()
        self.ground_truth_info = MapInfo(self.ground_truth, self.belief_origin_x, self.belief_origin_y, CELL_SIZE)

        # 其它属性适配
        self.global_frontiers = []
        self.travel_dist = 0
        self.explored_rate = 0
        self.stairs_coords_list = [] # Habitat 中可以读取楼梯位置
        self.current_target_stairs_index = None
        self.stairs_cells = [] # 兼容属性
        self.map_index = 0     # 兼容属性
        self.discovered_stairs = set() # Add discovered stairs tracking
        
        # 轨迹记录 (Initialize unconditionally to prevent AttributeErrors if plot is toggled)
        self.trajectory_x = [self.robot_location[0]]
        self.trajectory_y = [self.robot_location[1]]
        self.frame_files = [] 

        # 初始观测
        self.reset()

    def _init_habitat(self):
        # 配置 Habitat
        if 'habitat_sim' not in globals(): return None # 以此避免未安装时的崩溃
        
        sim_cfg = habitat_sim.SimulatorConfiguration()
        # Use Matterport3D example scene (17DRP5sb8fy) as default
        sim_cfg.scene_id = "data/scene_datasets/scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb"
        # Force EGL or offscreen rendering
        # Use inferred physical device ID if available (from worker_habitat.py)
        sim_cfg.gpu_device_id = int(os.environ.get("HABITAT_DEVICE_ID", 0))
        print(f"[HabitatEnv] Using GPU Device ID: {sim_cfg.gpu_device_id}", flush=True)
        
        # Explicitly disable physics to reduce complexity/crash risk during init
        sim_cfg.enable_physics = False
        
        # FIX: Ensure gpu_gpu_transfer is OFF to prevent EGL/CUDA sharing crashes
        # if hasattr(sim_cfg, "gpu_gpu_transfer"):
        #     sim_cfg.gpu_gpu_transfer = False

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        
        # Handle API differences between habitat-sim versions (0.1.7 vs new)
        SensorSpecClass = getattr(habitat_sim, "CameraSensorSpec", habitat_sim.SensorSpec)

        # 修复: 使用属性赋值而不是构造函数传参，兼容不同版本的 habitat-sim
        depth_sensor_spec = SensorSpecClass()
        depth_sensor_spec.uuid = "depth_sensor"
        depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_spec.resolution = [480, 640]
        depth_sensor_spec.position = [0.0, 1.5, 0.0]

        # Add RGB Sensor for VLM
        rgb_sensor_spec = SensorSpecClass()
        rgb_sensor_spec.uuid = "color_sensor"
        rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor_spec.resolution = [480, 640]
        rgb_sensor_spec.position = [0.0, 1.5, 0.0]
        
        agent_cfg.sensor_specifications = [depth_sensor_spec, rgb_sensor_spec]
        
        cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])

        try:
            sim = habitat_sim.Simulator(cfg)
            # 确保 NavMesh 加载
            if not sim.pathfinder.is_loaded:
                 print("[HabitatEnv] Warning: PathFinder not loaded. Attempting to load...", flush=True)
                 # Try to load navmesh if exists (usually scene_id.navmesh)
                 navmesh_file = sim_cfg.scene_id.replace(".glb", ".navmesh")
                 if os.path.exists(navmesh_file):
                     sim.pathfinder.load_nav_mesh(navmesh_file)
                     print(f"[HabitatEnv] Loaded NavMesh from {navmesh_file}", flush=True)
                 else:
                     print(f"[HabitatEnv] Warning: NavMesh file {navmesh_file} not found.", flush=True)
            return sim
        except Exception as e:
            print(f"[HabitatEnv] Failed to init Habitat Simulator: {e}", flush=True)
            raise e

    def _get_ground_truth_map(self, floor_height=None):
        """
        Generates a Ground Truth TopDown Map using the Simulator's PathFinder.
        Returns a 2D numpy array where FREE=255, OCCUPIED=0.
        """
        print("[HabitatEnv] Generating Ground Truth Map from PathFinder...", flush=True)
        h, w = self.robot_belief.shape
        # Initialize as OCCUPIED (0) or UNKNOWN (127)? 
        # GroundTruthNodeManager expects FREE areas to generate nodes.
        # We initialize as OCCUPIED to be safe, then carve out FREE areas.
        gt_map = np.zeros((h, w), dtype=np.uint8) 
        
        if not self.sim or not self.sim.pathfinder.is_loaded:
            print("[HabitatEnv] Critical Warning: PathFinder not available. Returning empty Ground Truth.", flush=True)
            return gt_map

        pf = self.sim.pathfinder
        bounds = pf.get_bounds()
        
        # Height for navigability check: 
        # If floor_height is provided, use it. Otherwise default to bounds min_y.
        if floor_height is not None:
            check_height = floor_height + 0.2
        else:
            check_height = bounds[0][1] + 0.2 
        
        print(f"[HabitatEnv] Generating GT Map at height {check_height:.2f}m", flush=True)
        
        # Map parameters
        cell_size = self.mapper.cell_size 
        origin_x = self.mapper.origin_x
        origin_y = self.mapper.origin_y
        
        # Optimization: Only scan relevant area (Bounding Box of the scene)
        min_x_world = bounds[0][0]
        max_x_world = bounds[1][0]
        min_z_world = bounds[0][2]
        max_z_world = bounds[1][2]
        
        min_px = int((min_x_world - origin_x) / cell_size)
        max_px = int((max_x_world - origin_x) / cell_size) + 1
        min_py = int((min_z_world - origin_y) / cell_size)
        max_py = int((max_z_world - origin_y) / cell_size) + 1
        
        # Clamp to map size
        min_px = max(0, min_px)
        max_px = min(w, max_px)
        min_py = max(0, min_py)
        max_py = min(h, max_py)
        
        free_count = 0
        for py in range(min_py, max_py):
            for px in range(min_px, max_px):
                # Pixel to World
                x = px * cell_size + origin_x
                z = py * cell_size + origin_y
                
                point = np.array([x, check_height, z])
                
                if pf.is_navigable(point):
                    gt_map[py, px] = 255 # FREE
                    free_count += 1
        
        print(f"[HabitatEnv] Ground Truth Map generated. Free cells: {free_count}", flush=True)
        return gt_map

    def _generate_stairs(self):
        """
        Simulate stairs locations by picking random navigable points on the current floor.
        NOTE: This is for testing multi-floor navigation logic on single-floor maps.
        Real stairs should be detected from semantic annotations if available.
        """
        self.stairs_coords_list = []
        self.discovered_stairs = set()
        
        # DISABLE SIMULATED STAIRS FOR REALISM
        # The user requested to remove fake stairs in belief map if they don't exist in reality.
        # If real stairs are found (e.g. from semantic scene), they should be added here.
        # For now, we leave it empty to avoid confusion.
        return

        if not self.sim or not self.sim.pathfinder.is_loaded:
            return

        # SIMULATION ONLY: Generate virtual stairs for testing
        print("[HabitatEnv] NOTICE: Generating VIRTUAL/SIMULATED stairs for multi-floor logic testing.", flush=True)
        num_stairs = 2
        for _ in range(num_stairs):
            try:
                # Try to find a point somewhat far from robot? 
                # For now just random navigable points
                pt = self.sim.pathfinder.get_random_navigable_point()
                # Convert to 2D coords
                self.stairs_coords_list.append([pt[0], pt[2]])
            except:
                pass
        
        print(f"[HabitatEnv] Generated {len(self.stairs_coords_list)} simulated stairs locations.", flush=True)

    def discover_stairs(self):
        """
        Check if robot is close to any stairs and mark them as discovered.
        """
        if not self.stairs_coords_list:
            return
            
        robot_pos = self.robot_location
        for idx, coords in enumerate(self.stairs_coords_list):
            dist = np.linalg.norm(robot_pos - np.array(coords))
            if dist < 3.0: # Discovery radius
                if idx not in self.discovered_stairs:
                    print(f"[HabitatEnv] Discovered stairs {idx} at {coords}!", flush=True)
                    self.discovered_stairs.add(idx)

    def reset(self):
        # Reset episode-specific state variables
        self.frame_files = []
        self.travel_dist = 0
        self.explored_rate = 0
        # self.stairs_coords_list = [] # Moved to _generate_stairs
        # self.stairs_cells = []
        
        if self.sim:
            # Randomize start position on NavMesh for new episodes
            if self.sim.pathfinder.is_loaded:
                 # Try to find a navigable point
                 num_retries = 100
                 for _ in range(num_retries):
                     try:
                        target_pos = self.sim.pathfinder.get_random_navigable_point()
                     except:
                        bounds = self.sim.pathfinder.get_bounds()
                        x = np.random.uniform(bounds[0][0], bounds[1][0])
                        z = np.random.uniform(bounds[0][2], bounds[1][2])
                        y = bounds[0][1] # Floor height
                        target_pos = self.sim.pathfinder.snap_point(np.array([x, y, z]))
                     
                     if not np.isnan(target_pos[0]):
                         new_state = habitat_sim.AgentState()
                         new_state.position = target_pos
                         import quaternion
                         angle = np.random.uniform(0, 2 * np.pi)
                         new_state.rotation = quaternion.from_rotation_vector(np.array([0, angle, 0]))
                         self.sim.get_agent(0).set_state(new_state)
                         print(f"[HabitatEnv] Reset to random position: {target_pos}", flush=True)
                         break
            else:
                self.sim.reset()
                
            observations = self.sim.get_sensor_observations()
            # Update map
            agent_state = self.sim.get_agent(0).get_state()
            sensor_state = agent_state.sensor_states['depth_sensor']
            self.robot_belief = self.mapper.update(observations['depth_sensor'], sensor_state, agent_state)
            self.update_robot_location_from_sim(agent_state)
            
            # Reset trajectory with new start position
            self.trajectory_x = [self.robot_location[0]]
            self.trajectory_y = [self.robot_location[1]]
            
            # Store RGB image for VLM (remove alpha channel if present)
            if 'color_sensor' in observations:
                self.rgb_image = observations['color_sensor'][..., :3]
                if self.rgb_image.mean() < 1.0:
                     print(f"[HabitatEnv] Warning: RGB Image is very dark! Mean: {self.rgb_image.mean()}", flush=True)
            else:
                self.rgb_image = None
                print("[HabitatEnv] Warning: No color_sensor in observations!", flush=True)
        
        # Generate new stairs for the episode
        self._generate_stairs()
        
        return self.robot_belief

    def update_robot_location_from_sim(self, agent_state):
        # 将 Habitat 的 (x, y, z) 转换为我们系统的 (x, y)
        # Habitat: y is up. x, z are ground plane.
        self.robot_location = np.array([agent_state.position[0], agent_state.position[2]])

    def update_frontiers(self):
        """
        Extract global frontiers from the occupancy map.
        Frontiers are boundaries between FREE space and UNKNOWN space.
        """
        try:
            import cv2
        except ImportError:
            # Fallback if cv2 not available (though unlikely in vision project)
            return

        # 1. Create Masks
        # Free: > 200, Unknown: ~127 (100-150), Occupied: < 50
        map_gray = self.robot_belief.astype(np.uint8)
        
        mask_free = (map_gray > 200).astype(np.uint8) * 255
        mask_unknown = ((map_gray > 100) & (map_gray < 150)).astype(np.uint8) * 255
        
        # 2. Find edges of Free space that touch Unknown space
        # Dilate free space slightly to overlap with unknown
        kernel = np.ones((3, 3), np.uint8)
        dilated_free = cv2.dilate(mask_free, kernel, iterations=1)
        
        # Intersection: Dilated Free AND Unknown
        # These are unknown cells that are next to free cells -> Frontiers
        frontier_mask = cv2.bitwise_and(dilated_free, mask_unknown)
        
        # 3. Cluster/Contour to find distinct frontier points
        contours, _ = cv2.findContours(frontier_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        self.global_frontiers = []
        
        for cnt in contours:
            # Filter small noise
            area = cv2.contourArea(cnt)
            if area < 5: 
                continue
                
            # Find center of the frontier line
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Convert pixel to world coords
                wx = cx * self.mapper.cell_size + self.mapper.origin_x
                wy = cy * self.mapper.cell_size + self.mapper.origin_y
                
                self.global_frontiers.append(((wx, wy), area))
        
        # print(f"[HabitatEnv] Updated Frontiers: {len(self.global_frontiers)} found.", flush=True)

    def _move_and_update(self, target_pos_3d):
        current_state = self.sim.get_agent(0).get_state()
        new_state = habitat_sim.AgentState()
        new_state.position = target_pos_3d
        new_state.rotation = current_state.rotation
        self.sim.get_agent(0).set_state(new_state)
        
        observations = self.sim.get_sensor_observations()
        agent_state = self.sim.get_agent(0).get_state()
        sensor_state = agent_state.sensor_states['depth_sensor']
        self.robot_belief = self.mapper.update(observations['depth_sensor'], sensor_state, agent_state)
        self.belief_info.update_map_info(self.robot_belief, self.mapper.origin_x, self.mapper.origin_y)
        self.update_robot_location_from_sim(agent_state)
        
        if 'color_sensor' in observations:
            self.rgb_image = observations['color_sensor'][..., :3]
            
        if self.plot:
            self.plot_env(len(self.frame_files))
            if not os.path.exists(gifs_path):
                os.makedirs(gifs_path)
            filename = f"{gifs_path}/{self.episode_index}_{len(self.frame_files)}.png"
            plt.savefig(filename)
            plt.close()
            self.frame_files.append(filename)

    def step(self, next_waypoint):
        """
        Moves the robot to next_waypoint using Dijkstra/A* pathfinding on the belief map.
        Simulates step-by-step movement to respect obstacles and update map continuously.
        """
        if not self.sim: return 0
        
        old_position = self.robot_location.copy()
        current_state = self.sim.get_agent(0).get_state()
        height = current_state.position[1]
        
        # 1. Path Planning on Belief Map
        start_cell = get_cell_position_from_coords(self.robot_location, self.belief_info)
        end_cell = get_cell_position_from_coords(next_waypoint, self.belief_info)
        
        # Check if start/end are valid
        if start_cell is None or end_cell is None:
             print(f"[HabitatEnv] Invalid start/end for path planning: {start_cell} -> {end_cell}")
             return -1.0

        # Use A* on grid
        path = get_grid_path(self.robot_belief, start_cell, end_cell)
        
        if path is None:
            print(f"[HabitatEnv] Path planning failed to {next_waypoint}. Obstacle detected or unreachable.")
            return -0.1 # Penalty for invalid move attempt
            
        # 2. Execute Path (Simulate movement)
        # Move along path with subsampling (e.g., every 2 pixels ~= 0.8m)
        step_size = 2 
        
        # Skip start, iterate through path
        for i in range(step_size, len(path), step_size):
            target_cell = path[i]
            target_coords = get_coords_from_cell_position(np.array(target_cell), self.belief_info)
            target_pos_3d = np.array([target_coords[0], height, target_coords[1]])
            
            self._move_and_update(target_pos_3d)
            
        # Final move to ensure exact target
        if len(path) > 0:
            target_coords = get_coords_from_cell_position(np.array(path[-1]), self.belief_info)
            target_pos_3d = np.array([target_coords[0], height, target_coords[1]])
            self._move_and_update(target_pos_3d)

        # 3. Auxiliary Updates
        if self.episode_index % 5 == 0: 
            self.update_frontiers()
        self.discover_stairs()

        # 4. Calculate Reward
        dist = np.linalg.norm(self.robot_location - old_position)
        self.travel_dist += dist
        self.trajectory_x.append(self.robot_location[0])
        self.trajectory_y.append(self.robot_location[1])
        
        total_pixels = self.robot_belief.size
        explored_pixels = np.sum(self.robot_belief > 200) + np.sum(self.robot_belief < 50)
        self.explored_rate = explored_pixels / total_pixels
        
        return 0.0

    def get_shortest_path(self, start_pos, end_pos):
        """
        Computes the shortest path between two points on the NavMesh.
        Args:
            start_pos: (x, y) 2D coordinates
            end_pos: (x, y) 2D coordinates
        Returns:
            List of (x, y) points representing the path, or None if no path found.
        """
        if not self.sim or not self.sim.pathfinder.is_loaded:
            return None
            
        # Convert to 3D points (y is height)
        # We use the agent's current height for start.
        agent_state = self.sim.get_agent(0).get_state()
        height = agent_state.position[1]
        
        start_3d = np.array([start_pos[0], height, start_pos[1]])
        end_3d = np.array([end_pos[0], height, end_pos[1]])
        
        path = habitat_sim.ShortestPath()
        path.requested_start = start_3d
        path.requested_end = end_3d
        
        found = self.sim.pathfinder.find_path(path)
        
        if not found:
            return None
            
        # Convert path points back to 2D
        path_2d = []
        for p in path.points:
            path_2d.append(np.array([p[0], p[2]]))
            
        return path_2d

    def is_valid_location(self, coords_2d):
        """
        Check if a 2D coordinate (x, z) is valid:
        1. Inside map bounds
        2. Navigable on current floor (if pathfinder available)
        """
        # 1. Map Bounds Check
        if self.mapper:
            map_x = int((coords_2d[0] - self.mapper.origin_x) / self.mapper.cell_size)
            map_y = int((coords_2d[1] - self.mapper.origin_y) / self.mapper.cell_size)
            h, w = self.mapper.global_map.shape
            if not (0 <= map_x < w and 0 <= map_y < h):
                return False

        # 2. Navigability Check
        if self.sim and self.sim.pathfinder.is_loaded:
            current_state = self.sim.get_agent(0).get_state()
            height = current_state.position[1]
            target_pos = np.array([coords_2d[0], height, coords_2d[1]])
            return self.sim.pathfinder.is_navigable(target_pos)
            
        return True

    def _check_stairs_discovery(self):
        """
        模拟楼梯感知：如果机器人靠近预设的楼梯坐标，就将其加入 known list
        """
        # 硬编码的楼梯位置 (示例: Skokloster Castle 的两个楼梯点)
        # 注意: 这里需要根据具体场景修改。
        # 格式: [x, z] (Habitat Y is up)
        ground_truth_stairs = [
            np.array([2.0, 5.0]),   # 示例楼梯1
            np.array([-3.0, -4.0])  # 示例楼梯2
        ]
        
        discovery_radius = 5.0 # 5米内自动发现
        
        for gt_stair in ground_truth_stairs:
            dist = np.linalg.norm(self.robot_location - gt_stair)
            if dist < discovery_radius:
                # 检查是否已经记录过
                already_known = False
                for known_stair in self.stairs_coords_list:
                    if np.linalg.norm(np.array(known_stair) - gt_stair) < 1.0:
                        already_known = True
                        break
                
                if not already_known:
                    print(f"🌟 Discovered new stairs at {gt_stair}!")
                    self.stairs_coords_list.append(gt_stair.tolist())
                    # 同时更新 stairs_cells (用于绘图和兼容)
                    # 转换回 map pixel coords
                    mx = int((gt_stair[0] - self.belief_origin_x) / CELL_SIZE)
                    my = int((gt_stair[1] - self.belief_origin_y) / CELL_SIZE)
                    self.stairs_cells.append((mx, my))

    def calculate_reward(self, dist):
        # 直接复用原逻辑，这里简化写
        reward = -dist * 0.1
        # ... (此处应复制 env.py 中的完整奖励逻辑)
        return reward

    def evaluate_exploration_rate(self):
        # 使用“前沿点密度”或“相对增长率”来估算，而不是上帝视角的总面积
        # 方案 A (简单版): 用一个预估的地图大小常数 (例如 2000 个格子)
        # self.explored_rate = min(1.0, np.sum(self.robot_belief == 255) / 2000.0)
        
        # 方案 B (当前版): 使用 robot_belief 的总像素数作为分母
        # 注意：SimpleMapper 初始化时给了一个 map_size_meters=100.0
        # 所以 robot_belief.size 是固定的 (比如 250x250)
        # 这相当于假设机器人知道它在一个 100x100 米的盒子里
        total_pixels = self.robot_belief.size
        free_pixels = np.sum(self.robot_belief == 255)
        self.explored_rate = free_pixels / total_pixels

    def switch_floor(self):
        # 模拟楼层切换
        print(f"Switching floor in Habitat from Floor {self.floor_id}...")
        
        # 1. Update Floor ID
        self.floor_id = 1 - self.floor_id # Toggle between 0 and 1
        
        # 2. Teleport Agent to new floor height
        # Assuming Skokloster Castle or similar has floors at different heights
        # We need to find a safe navigable point on the new floor.
        # Ideally, we should teleport to the "other end" of the stairs.
        # For now, we will add/subtract a height offset to the current position
        # and snap to the nearest navigable point.
        
        if not self.sim:
            return

        current_state = self.sim.get_agent(0).get_state()
        current_pos = current_state.position
        
        # Habitat Y is Up.
        # Example height difference. This should be calibrated per scene.
        floor_height_diff = 3.5 
        
        new_height = current_pos[1] + floor_height_diff if self.floor_id == 1 else current_pos[1] - floor_height_diff
        
        target_pos = np.array([current_pos[0], new_height, current_pos[2]])
        
        # Snap to NavMesh to ensure valid position
        if self.sim.pathfinder.is_loaded:
            snap_pos = self.sim.pathfinder.snap_point(target_pos)
            if not np.isnan(snap_pos[0]):
                target_pos = snap_pos
                print(f"Snapped to NavMesh at {target_pos}")
            else:
                print("Warning: Could not snap to NavMesh on new floor!")
        
        new_state = habitat_sim.AgentState()
        new_state.position = target_pos
        new_state.rotation = current_state.rotation
        self.sim.get_agent(0).set_state(new_state)
        
        print(f"Agent teleported to Floor {self.floor_id}, Height: {target_pos[1]:.2f}")

    def block_stairs_entrance(self, stairs_cell):
        # Habitat 不需要手动阻塞楼梯入口，因为有物理碰撞
        pass

    def reset_semantic_map(self):
        print("[HabitatEnv] Resetting semantic map (clearing mapper state)...")
        if hasattr(self, 'mapper'):
            self.robot_belief = self.mapper.reset()
        
        if hasattr(self, 'belief_info'):
            self.belief_info.update_map_info(self.robot_belief, self.belief_origin_x, self.belief_origin_y)
        
        self.global_frontiers = set()
        self.explored_rate = 0

    def plot_env(self, step):
        if not self.plot:
            return

        plt.switch_backend('agg')
        plt.close('all') # Ensure no previous figures interfere
        
        # Determine if we have RGB image
        has_rgb = hasattr(self, 'rgb_image') and self.rgb_image is not None
        
        if has_rgb:
            plt.figure(figsize=(12, 6))
            # Subplot 1: Explored Map
            plt.subplot(1, 2, 1)
        else:
            plt.figure(figsize=(8, 8))
            # Only Map
            plt.gca()

        # ... (Map plotting logic remains similar)
        
        # Enhanced Map Visualization
        # Assuming robot_belief is 0-255 (0=Occupied, 255=Free, 127=Unknown)
        # We can create a colored map for better contrast
        # Use a copy to avoid modifying the original belief map for display
        map_disp = np.zeros((*self.robot_belief.shape, 3), dtype=np.uint8)
        
        # Colors
        COLOR_UNKNOWN = [200, 200, 200] # Light Gray
        COLOR_FREE = [255, 255, 255]    # White
        COLOR_OCCUPIED = [0, 0, 0]      # Black
        
        # Masking
        mask_free = (self.robot_belief > 200)
        mask_occupied = (self.robot_belief < 50)
        mask_unknown = (~mask_free & ~mask_occupied)
        
        map_disp[mask_unknown] = COLOR_UNKNOWN
        map_disp[mask_free] = COLOR_FREE
        map_disp[mask_occupied] = COLOR_OCCUPIED
        
        plt.imshow(map_disp, origin='lower')
        plt.title(f'Explored Map (Step {step})')
        plt.axis('off')
        
        # Plot robot
        if hasattr(self, 'mapper'):
            rx = (self.robot_location[0] - self.belief_origin_x) / self.mapper.cell_size
            ry = (self.robot_location[1] - self.belief_origin_y) / self.mapper.cell_size
            plt.plot(rx, ry, 'mo', markersize=6, markeredgecolor='k', zorder=10, label='Robot')
        
        # Plot trajectory
        if len(self.trajectory_x) > 1:
            traj_x = (np.array(self.trajectory_x) - self.belief_origin_x) / self.mapper.cell_size
            traj_y = (np.array(self.trajectory_y) - self.belief_origin_y) / self.mapper.cell_size
            plt.plot(traj_x, traj_y, 'b-', linewidth=1.5, alpha=0.7, zorder=5, label='Trajectory')
        
        # Plot stairs
        if hasattr(self, "stairs_coords_list") and self.stairs_coords_list is not None:
            for idx, coords in enumerate(self.stairs_coords_list):
                # Only plot if discovered or currently targeted
                is_discovered = (hasattr(self, 'discovered_stairs') and idx in self.discovered_stairs)
                is_target = (hasattr(self, "current_target_stairs_index") and 
                             self.current_target_stairs_index is not None and 
                             idx == self.current_target_stairs_index)
                
                if is_discovered or is_target:
                    sx = (coords[0] - self.belief_origin_x) / self.mapper.cell_size
                    sy = (coords[1] - self.belief_origin_y) / self.mapper.cell_size
                    
                    if is_target:
                        plt.plot(sx, sy, 'y*', markersize=12, markeredgecolor='k', zorder=8, label='Target Stair')
                    else:
                        plt.plot(sx, sy, 'r*', markersize=8, markeredgecolor='k', zorder=7, label='Discovered Stair')

        # Subplot 2: RGB View (Agent Perspective) - Only if RGB exists
        if has_rgb:
             plt.subplot(1, 2, 2)
             plt.imshow(self.rgb_image)
             plt.title('Agent View (RGB)')
             plt.axis('off')

        plt.suptitle(f'Explored: {self.explored_rate:.1%} | Dist: {self.travel_dist:.1f}m | Step: {step}', fontsize=14)
        plt.tight_layout()
        
        # Ensure gifs_path exists
        if not os.path.exists(gifs_path):
            os.makedirs(gifs_path)
            
        save_path = '{}/{}_{}_habitat.png'.format(gifs_path, self.episode_index, step)
        plt.savefig(save_path, dpi=100)
        plt.close()
        self.frame_files.append(save_path)
