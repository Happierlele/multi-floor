import os
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

habitat_sim = None

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
        self.floor_states = {} # floor_id -> mapper state
        
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
        self.belief_info = MapInfo(self.robot_belief, self.belief_origin_x, self.belief_origin_y, CELL_SIZE, vlm_rgb_map=self.mapper.get_vlm_rgb_map())
        
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
        global habitat_sim
        if habitat_sim is None:
            if os.environ.get("DISPLAY", "") == "":
                os.environ.setdefault("MAGNUM_TARGET_GLES", "1")
                os.environ.setdefault("MAGNUM_TARGET_HEADLESS", "1")
                os.environ.setdefault("MAGNUM_TARGET_EGL", "1")
                os.environ.setdefault("EGL_PLATFORM", "device")
                nvidia_json = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
                if os.path.exists(nvidia_json):
                    os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", nvidia_json)
            import habitat_sim as _habitat_sim
            habitat_sim = _habitat_sim
        
        sim_cfg = habitat_sim.SimulatorConfiguration()
        scene_id = os.environ.get("HABITAT_SCENE_ID", "").strip()
        if scene_id:
            sim_cfg.scene_id = scene_id
        else:
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
        try:
            hfov = float(os.environ.get("HABITAT_SENSOR_HFOV", "90"))
            if hasattr(depth_sensor_spec, "hfov"):
                depth_sensor_spec.hfov = hfov
        except Exception:
            pass

        # Add RGB Sensor for VLM
        rgb_sensor_spec = SensorSpecClass()
        rgb_sensor_spec.uuid = "color_sensor"
        rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor_spec.resolution = [480, 640]
        rgb_sensor_spec.position = [0.0, 1.5, 0.0]
        try:
            hfov = float(os.environ.get("HABITAT_SENSOR_HFOV", "90"))
            if hasattr(rgb_sensor_spec, "hfov"):
                rgb_sensor_spec.hfov = hfov
        except Exception:
            pass
        
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
                     recompute = os.environ.get("HABITAT_RECOMPUTE_NAVMESH", "0").strip() == "1"
                     if recompute:
                         try:
                             navmesh_settings = habitat_sim.NavMeshSettings()
                             if hasattr(navmesh_settings, "set_defaults"):
                                 navmesh_settings.set_defaults()
                             ok = sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
                             if ok and hasattr(sim.pathfinder, "save_nav_mesh"):
                                 sim.pathfinder.save_nav_mesh(navmesh_file)
                                 print(f"[HabitatEnv] Recomputed & saved NavMesh to {navmesh_file}", flush=True)
                             else:
                                 print("[HabitatEnv] NavMesh recompute failed or save_nav_mesh unavailable.", flush=True)
                         except Exception as e:
                             print(f"[HabitatEnv] NavMesh recompute failed: {e}", flush=True)
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

    def update_ground_truth_for_floor(self, height):
        """
        Regenerate the ground truth map for a specific floor height.
        This is crucial for multi-floor environments to ensure the GT map matches the current floor.
        """
        self.ground_truth = self._get_ground_truth_map(floor_height=height)
        self.ground_truth_info = MapInfo(self.ground_truth, self.belief_origin_x, self.belief_origin_y, CELL_SIZE)
        print(f"[HabitatEnv] Ground Truth updated for height {height:.2f}m", flush=True)


    def _detect_stairs_from_navmesh(self, sample_points=4000, grid_size=1.0, height_bin=0.75, max_stairs=10):
        if not self.sim or not self.sim.pathfinder.is_loaded:
            return []

        pts = []
        for _ in range(int(sample_points)):
            try:
                p = self.sim.pathfinder.get_random_navigable_point()
                if p is None:
                    continue
                pts.append([float(p[0]), float(p[1]), float(p[2])])
            except Exception:
                continue

        if len(pts) < 200:
            return []

        pts = np.asarray(pts, dtype=np.float32)
        y_bins = np.round(pts[:, 1] / float(height_bin)) * float(height_bin)
        unique_bins, counts = np.unique(y_bins, return_counts=True)
        if len(unique_bins) < 2:
            return []

        min_keep = max(50, int(0.03 * len(y_bins)))
        keep_bins = set(unique_bins[counts >= min_keep].tolist())
        if len(keep_bins) < 2:
            keep_bins = set(unique_bins[counts.argsort()[::-1][:2]].tolist())

        gx = np.floor(pts[:, 0] / float(grid_size)).astype(np.int32)
        gz = np.floor(pts[:, 2] / float(grid_size)).astype(np.int32)

        cell_bins = {}
        cell_sum = {}
        for i in range(pts.shape[0]):
            b = float(y_bins[i])
            if b not in keep_bins:
                continue
            key = (int(gx[i]), int(gz[i]))
            if key not in cell_bins:
                cell_bins[key] = {b}
                cell_sum[key] = [pts[i, 0], pts[i, 2], 1]
            else:
                cell_bins[key].add(b)
                s = cell_sum[key]
                s[0] += pts[i, 0]
                s[1] += pts[i, 2]
                s[2] += 1

        candidates = []
        for key, bins in cell_bins.items():
            if len(bins) >= 2:
                sx, sz, n = cell_sum[key]
                if n > 0:
                    candidates.append([sx / n, sz / n])

        if not candidates:
            return []

        selected = []
        min_sep = float(grid_size) * 2.0
        for c in candidates:
            if len(selected) >= int(max_stairs):
                break
            if not selected:
                selected.append(c)
                continue
            d2 = min((c[0] - s[0]) ** 2 + (c[1] - s[1]) ** 2 for s in selected)
            if d2 >= (min_sep ** 2):
                selected.append(c)

        return selected[: int(max_stairs)]


    def _generate_stairs(self):
        """
        Generate stairs candidates. If multi-floor structure exists, detect likely vertical connectors from NavMesh samples.
        """
        self.stairs_coords_list = []
        self.discovered_stairs = set()

        try:
            detected = self._detect_stairs_from_navmesh(
                sample_points=int(os.environ.get("HABITAT_STAIRS_SAMPLE_POINTS", "4000")),
                grid_size=float(os.environ.get("HABITAT_STAIRS_GRID_SIZE", "1.0")),
                height_bin=float(os.environ.get("HABITAT_STAIRS_HEIGHT_BIN", "0.75")),
                max_stairs=int(os.environ.get("HABITAT_STAIRS_MAX", "10")),
            )
        except Exception:
            detected = []

        if detected:
            self.stairs_coords_list = [list(map(float, s)) for s in detected]
            print(f"[HabitatEnv] Detected {len(self.stairs_coords_list)} stairs candidates from NavMesh.", flush=True)
            return

        if self.sim and self.sim.pathfinder.is_loaded:
            num_stairs = 2
            for _ in range(num_stairs):
                try:
                    pt = self.sim.pathfinder.get_random_navigable_point()
                    self.stairs_coords_list.append([float(pt[0]), float(pt[2])])
                except Exception:
                    pass
            if self.stairs_coords_list:
                print(f"[HabitatEnv] Generated {len(self.stairs_coords_list)} fallback stairs points.", flush=True)
        return

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
                 # Generate simulated stairs if not already present
                 if not self.stairs_coords_list:
                     self._generate_stairs()
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
            self.belief_info.update_map_info(self.robot_belief, self.mapper.origin_x, self.mapper.origin_y, vlm_rgb_map=self.mapper.get_vlm_rgb_map())
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
        
        if self.sim and self.sim.pathfinder.is_loaded and not self.stairs_coords_list:
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
        import quaternion
        dx = target_pos_3d[0] - current_state.position[0]
        dz = target_pos_3d[2] - current_state.position[2]
        yaw = np.arctan2(dz, dx)
        new_state.rotation = quaternion.from_rotation_vector(np.array([0, yaw, 0]))
        self.sim.get_agent(0).set_state(new_state)
        
        observations = self.sim.get_sensor_observations()
        agent_state = self.sim.get_agent(0).get_state()
        sensor_state = agent_state.sensor_states['depth_sensor']
        self.robot_belief = self.mapper.update(observations['depth_sensor'], sensor_state, agent_state)
        self.belief_info.update_map_info(self.robot_belief, self.mapper.origin_x, self.mapper.origin_y, vlm_rgb_map=self.mapper.get_vlm_rgb_map())
        self.update_robot_location_from_sim(agent_state)
        
        # Record trajectory at every micro-step for smooth visualization
        self.trajectory_x.append(self.robot_location[0])
        self.trajectory_y.append(self.robot_location[1])
        
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

    def step(self, next_waypoint, force=False):
        """
        Moves the robot to next_waypoint using Dijkstra/A* pathfinding on the belief map.
        Simulates step-by-step movement to respect obstacles and update map continuously.
        If force=True, falls back to NavMesh pathfinding if Belief Map path fails.
        """
        if not self.sim: return 0
        
        old_position = self.robot_location.copy()
        current_state = self.sim.get_agent(0).get_state()
        height = current_state.position[1]
        
        path_coords_list = []
        path = None

        if force:
            nav_path = self.get_shortest_path(self.robot_location, next_waypoint)
            if nav_path:
                path_coords_list = nav_path
            else:
                print(f"[HabitatEnv] Force=True but NavMesh path failed. Falling back to belief map A*.", flush=True)
                force = False

        if not force:
            start_cell = get_cell_position_from_coords(self.robot_location, self.belief_info)
            end_cell = get_cell_position_from_coords(next_waypoint, self.belief_info)
            
            if start_cell is None or end_cell is None:
                print(f"[HabitatEnv] Invalid start/end for path planning: {start_cell} -> {end_cell}")
                if self.sim and self.sim.pathfinder.is_loaded:
                    nav_path = self.get_shortest_path(self.robot_location, next_waypoint)
                    if nav_path:
                        path_coords_list = nav_path
                    else:
                        return -1.0
                else:
                    return -1.0
            else:
                path = get_grid_path(self.robot_belief, start_cell, end_cell)
                if path is None:
                    print(f"[HabitatEnv] Path planning failed to {next_waypoint}. Obstacle detected or unreachable.")
                    if self.sim and self.sim.pathfinder.is_loaded:
                        nav_path = self.get_shortest_path(self.robot_location, next_waypoint)
                        if nav_path:
                            path = None
                            path_coords_list = nav_path
                        else:
                            return -0.1
                    else:
                        return -0.1
                if path is not None:
                    for cell in path:
                        coord = get_coords_from_cell_position(np.array(cell), self.belief_info)
                        path_coords_list.append(coord)
            
        # 2. Execute Path (Simulate movement)
        # Move along path with subsampling (e.g., every 0.8m or similar)
        # Grid path is dense (pixel by pixel). NavMesh path is sparse (waypoints).
        
        if path is not None:
            # Dense grid path: subsample
            step_size = 2 
            for i in range(step_size, len(path_coords_list), step_size):
                target_coords = path_coords_list[i]
                target_pos_3d = np.array([target_coords[0], height, target_coords[1]])
                self._move_and_update(target_pos_3d)
                
            # Final point
            if len(path_coords_list) > 0:
                target_coords = path_coords_list[-1]
                target_pos_3d = np.array([target_coords[0], height, target_coords[1]])
                self._move_and_update(target_pos_3d)
        else:
            prev = self.robot_location.copy()
            for target_coords in path_coords_list:
                seg = np.array(target_coords) - prev
                dist = np.linalg.norm(seg)
                if dist < 2.0:
                    stride = float(self.mapper.cell_size)
                else:
                    stride = 0.8
                steps = max(1, int(dist / stride))
                for s in range(1, steps + 1):
                    inter = prev + (s / steps) * seg
                    target_pos_3d = np.array([inter[0], height, inter[1]])
                    self._move_and_update(target_pos_3d)
                prev = np.array(target_coords)

        # 3. Auxiliary Updates
        # Update frontiers on every step to keep utility fresh
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

    def get_grid_path_check(self, start_pos, end_pos):
        """
        Quickly check if a path exists between two points on the belief map.
        Returns True if path exists, False otherwise.
        """
        start_cell = get_cell_position_from_coords(start_pos, self.belief_info)
        end_cell = get_cell_position_from_coords(end_pos, self.belief_info)
        
        if start_cell is None or end_cell is None:
            return False
            
        path = get_grid_path(self.robot_belief, start_cell, end_cell)
        return path is not None

    def get_shortest_path(self, start_pos, end_pos):
        """
        Computes the shortest path on the Physical Network (NavMesh).
        
        This method uses Habitat-Sim's PathFinder (the "Physical Network") to calculate
        the true geodesic distance and path between two points. This serves as the 
        ground truth for movement and is used as a fallback when the high-level 
        topological graph fails.
        """
        if not self.sim or not self.sim.pathfinder.is_loaded:
            return None
            
        agent_state = self.sim.get_agent(0).get_state()
        height = float(agent_state.position[1])
        
        start_guess = np.array([float(start_pos[0]), height, float(start_pos[1])], dtype=np.float32)
        end_guess = np.array([float(end_pos[0]), height, float(end_pos[1])], dtype=np.float32)
        
        try:
            start_3d = self.sim.pathfinder.snap_point(start_guess)
            end_3d = self.sim.pathfinder.snap_point(end_guess)
        except Exception:
            start_3d = start_guess
            end_3d = end_guess
        
        if np.isnan(start_3d[0]) or np.isnan(end_3d[0]):
            return None
        
        path = habitat_sim.ShortestPath()
        path.requested_start = start_3d
        path.requested_end = end_3d
        
        found = self.sim.pathfinder.find_path(path)
        
        if not found:
            return None
            
        path_2d = []
        for p in path.points:
            path_2d.append(np.array([p[0], p[2]]))
            
        return path_2d
    
    def get_shortest_path_with_distance(self, start_pos, end_pos):
        if not self.sim or not self.sim.pathfinder.is_loaded:
            return None, float('inf')
        
        agent_state = self.sim.get_agent(0).get_state()
        height = float(agent_state.position[1])
        
        start_guess = np.array([float(start_pos[0]), height, float(start_pos[1])], dtype=np.float32)
        end_guess = np.array([float(end_pos[0]), height, float(end_pos[1])], dtype=np.float32)
        
        try:
            start_3d = self.sim.pathfinder.snap_point(start_guess)
            end_3d = self.sim.pathfinder.snap_point(end_guess)
        except Exception:
            start_3d = start_guess
            end_3d = end_guess
        
        if np.isnan(start_3d[0]) or np.isnan(end_3d[0]):
            return None, float('inf')
        
        sp = habitat_sim.ShortestPath()
        sp.requested_start = start_3d
        sp.requested_end = end_3d
        
        found = self.sim.pathfinder.find_path(sp)
        if not found:
            return None, float('inf')
        
        path_2d = []
        for p in sp.points:
            path_2d.append(np.array([p[0], p[2]]))
        return path_2d, float(sp.geodesic_distance)

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
            height = float(current_state.position[1])
            target_guess = np.array([float(coords_2d[0]), height, float(coords_2d[1])], dtype=np.float32)
            try:
                target_pos = self.sim.pathfinder.snap_point(target_guess)
            except Exception:
                target_pos = target_guess
            if np.isnan(target_pos[0]):
                return False
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
        # Save current floor map state
        if hasattr(self, 'mapper'):
            try:
                self.floor_states[self.floor_id] = self.mapper.get_state()
            except Exception:
                pass
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
        
        # Update Ground Truth for the new floor height
        try:
            self.update_ground_truth_for_floor(height=target_pos[1])
        except Exception:
            pass
        
        # 3. Restore map for target floor if we have visited before; otherwise reset
        if self.floor_id in self.floor_states:
            print(f"[HabitatEnv] Restoring belief map for Floor {self.floor_id}")
            try:
                self.mapper.set_state(self.floor_states[self.floor_id])
            except Exception:
                self.mapper.reset()
        else:
            self.mapper.reset()
        
        # 4. Refresh belief map at the new pose and sync state
        try:
            observations = self.sim.get_sensor_observations()
            agent_state = self.sim.get_agent(0).get_state()
            sensor_state = agent_state.sensor_states['depth_sensor']
            self.robot_belief = self.mapper.update(observations['depth_sensor'], sensor_state, agent_state)
            self.belief_info.update_map_info(self.robot_belief, self.mapper.origin_x, self.mapper.origin_y, vlm_rgb_map=self.mapper.get_vlm_rgb_map())
            self.update_robot_location_from_sim(agent_state)
            self.trajectory_x = [self.robot_location[0]]
            self.trajectory_y = [self.robot_location[1]]
            if 'color_sensor' in observations:
                self.rgb_image = observations['color_sensor'][..., :3]
            self.update_frontiers()
            self.discover_stairs()
        except Exception:
            pass

    def block_stairs_entrance(self, stairs_cell):
        # Habitat 不需要手动阻塞楼梯入口，因为有物理碰撞
        pass

    def reset_semantic_map(self):
        print("[HabitatEnv] Resetting semantic map (clearing mapper state)...")
        if hasattr(self, 'mapper'):
            self.robot_belief = self.mapper.reset()
        
        if hasattr(self, 'belief_info'):
            self.belief_info.update_map_info(self.robot_belief, self.belief_origin_x, self.belief_origin_y, vlm_rgb_map=self.mapper.get_vlm_rgb_map())
        
        self.global_frontiers = []
        self.explored_rate = 0
        
        # Reset trajectory to prevent old lines from persisting on new map
        self.trajectory_x = [self.robot_location[0]]
        self.trajectory_y = [self.robot_location[1]]

    def plot_env(self, step):
        if not self.plot:
            return

        plt.switch_backend('agg')
        plt.close('all') # Ensure no previous figures interfere
        
        # Determine if we have RGB image
        has_rgb = hasattr(self, 'rgb_image') and self.rgb_image is not None
        
        if has_rgb:
            plt.figure(figsize=(18, 6))
            # Subplot 1: Explored Map
            plt.subplot(1, 3, 1)
        else:
            plt.figure(figsize=(12, 6))
            # Subplot 1: Explored Map
            plt.subplot(1, 2, 1)

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
             plt.subplot(1, 3, 2)
             plt.imshow(self.rgb_image)
             plt.title('Agent View (RGB)')
             plt.axis('off')
             
             # Subplot 3: Ground Truth Map
             plt.subplot(1, 3, 3)
             gt_map_disp = np.zeros((*self.ground_truth.shape, 3), dtype=np.uint8)
             # Use same colors as belief map for consistency
             gt_mask_free = (self.ground_truth > 200)
             gt_mask_occupied = (self.ground_truth < 50)
             
             gt_map_disp[gt_mask_free] = [255, 255, 255] # White
             gt_map_disp[gt_mask_occupied] = [0, 0, 0]   # Black
             gt_map_disp[~gt_mask_free & ~gt_mask_occupied] = [127, 127, 127] # Gray
             
             plt.imshow(gt_map_disp, origin='lower')
             plt.title('Ground Truth Map (Real)')
             plt.axis('off')
             
             # Plot robot on GT map too for reference
             if hasattr(self, 'mapper'):
                rx = (self.robot_location[0] - self.belief_origin_x) / self.mapper.cell_size
                ry = (self.robot_location[1] - self.belief_origin_y) / self.mapper.cell_size
                plt.plot(rx, ry, 'mo', markersize=6, markeredgecolor='k', zorder=10)

        else:
             # Subplot 2: Ground Truth Map (if no RGB)
             plt.subplot(1, 2, 2)
             gt_map_disp = np.zeros((*self.ground_truth.shape, 3), dtype=np.uint8)
             gt_mask_free = (self.ground_truth > 200)
             gt_mask_occupied = (self.ground_truth < 50)
             
             gt_map_disp[gt_mask_free] = [255, 255, 255] # White
             gt_map_disp[gt_mask_occupied] = [0, 0, 0]   # Black
             gt_map_disp[~gt_mask_free & ~gt_mask_occupied] = [127, 127, 127] # Gray
             
             plt.imshow(gt_map_disp, origin='lower')
             plt.title('Ground Truth Map (Real)')
             plt.axis('off')
             
             if hasattr(self, 'mapper'):
                rx = (self.robot_location[0] - self.belief_origin_x) / self.mapper.cell_size
                ry = (self.robot_location[1] - self.belief_origin_y) / self.mapper.cell_size
                plt.plot(rx, ry, 'mo', markersize=6, markeredgecolor='k', zorder=10)

        consistency_text = ""
        try:
            if hasattr(self, "ground_truth") and self.ground_truth is not None and self.ground_truth.shape == self.robot_belief.shape:
                belief = self.robot_belief.astype(np.uint8)
                gt = self.ground_truth.astype(np.uint8)
                belief_free = belief > 200
                belief_occ = belief < 50
                belief_known = belief_free | belief_occ
                gt_free = gt > 200
                gt_occ = gt < 50
                valid = belief_known & (gt_free | gt_occ)
                denom = int(np.count_nonzero(valid))
                if denom > 0:
                    correct = (belief_free & gt_free) | (belief_occ & gt_occ)
                    acc = float(np.count_nonzero(correct & valid)) / float(denom)
                    ff = float(np.count_nonzero((belief_free & gt_occ) & valid)) / float(denom)
                    fo = float(np.count_nonzero((belief_occ & gt_free) & valid)) / float(denom)
                    consistency_text = f" | MapAcc: {acc:.1%} | FalseFree: {ff:.1%} | FalseOcc: {fo:.1%}"
        except Exception:
            consistency_text = ""

        plt.suptitle(f'Explored: {self.explored_rate:.1%} | Dist: {self.travel_dist:.1f}m | Step: {step}{consistency_text}', fontsize=14)
        plt.tight_layout()
        
        # Ensure gifs_path exists
        if not os.path.exists(gifs_path):
            os.makedirs(gifs_path)
            
        save_path = '{}/{}_{}_habitat.png'.format(gifs_path, self.episode_index, step)
        plt.savefig(save_path, dpi=100)
        plt.close()
        self.frame_files.append(save_path)
