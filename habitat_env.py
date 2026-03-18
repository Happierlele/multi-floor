import os
import math
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from types import SimpleNamespace

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
        try:
            if self.sim and getattr(self.sim, "pathfinder", None) is not None:
                self.mapper.pathfinder = self.sim.pathfinder
        except Exception:
            pass
        
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
        self._vlm_view_last_step = -10**9
        self._vlm_view_snap_id = 0

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
        def _infer_dataset_tag(scene_path: str) -> str:
            sp = (scene_path or "").lower()
            if "hm3d" in sp:
                return "HM3D"
            if "mp3d" in sp or "matterport" in sp:
                return "MP3D"
            if "gibson" in sp:
                return "Gibson"
            if "replica" in sp:
                return "Replica"
            if "habitat_test_scenes" in sp:
                return "HabitatTestScenes"
            return "Unknown"

        def _choose_default_scene_id() -> str:
            override = os.environ.get("HABITAT_DEFAULT_SCENE", "").strip()
            if override:
                return override

            hm3d_root = os.environ.get("HABITAT_HM3D_ROOT", "").strip()
            hm3d_candidates = []
            try:
                import glob
                roots = [p for p in [hm3d_root, "data/scene_datasets/hm3d", "data/scene_datasets/hm3d_v0.2", "data/scene_datasets/hm3d_v0.1"] if p]
                for r in roots:
                    if os.path.isdir(r):
                        hm3d_candidates.extend(glob.glob(os.path.join(r, "**", "*.scene_instance.json"), recursive=True))
                        hm3d_candidates.extend(glob.glob(os.path.join(r, "**", "*.glb"), recursive=True))
            except Exception:
                hm3d_candidates = []

            for p in hm3d_candidates:
                if os.path.exists(p):
                    return p

            fallback = "data/versioned_data/habitat_test_scenes/apartment_1.glb"
            if os.path.exists(fallback):
                return fallback

            return "data/versioned_data/habitat_test_scenes/skokloster-castle.glb"

        scene_id = os.environ.get("HABITAT_SCENE_ID", "").strip()
        sim_cfg.scene_id = scene_id if scene_id else _choose_default_scene_id()
        if sim_cfg.scene_id and os.path.isdir(sim_cfg.scene_id):
            chosen = None
            try:
                import glob
                cand = glob.glob(os.path.join(sim_cfg.scene_id, "*.scene_instance.json"))
                if not cand:
                    cand = glob.glob(os.path.join(sim_cfg.scene_id, "*.glb"))
                cand = sorted(cand)

                if cand:
                    try:
                        pick = os.environ.get("HABITAT_SCENE_PICK", "index").strip().lower()
                    except Exception:
                        pick = "index"
                    if pick == "random":
                        try:
                            seed = int(os.environ.get("HABITAT_SCENE_SEED", "0"))
                        except Exception:
                            seed = 0
                        rng = np.random.RandomState(seed)
                        chosen = cand[int(rng.randint(0, len(cand)))]
                    else:
                        try:
                            idx = int(os.environ.get("HABITAT_SCENE_INDEX", os.environ.get("HABITAT_EPISODE_INDEX", "0")))
                        except Exception:
                            idx = 0
                        chosen = cand[int(idx) % int(len(cand))]
            except Exception:
                chosen = None
            if chosen:
                sim_cfg.scene_id = chosen
        print(f"[HabitatEnv] Scene ID: {sim_cfg.scene_id}", flush=True)
        print(f"[HabitatEnv] Scene Dataset: {_infer_dataset_tag(sim_cfg.scene_id)}", flush=True)
        
        # Robust scene path handling to avoid segfaults when file does not exist
        try:
            if not os.path.exists(sim_cfg.scene_id):
                if sim_cfg.scene_id.endswith(".glb"):
                    # Try HM3D JSON variants
                    si = sim_cfg.scene_id.replace(".glb", ".scene_instance.json")
                    sc = sim_cfg.scene_id.replace(".glb", ".stage_config.json")
                    if os.path.exists(si):
                        print(f"[HabitatEnv] Scene .glb not found. Using scene_instance.json: {si}", flush=True)
                        sim_cfg.scene_id = si
                    elif os.path.exists(sc):
                        print(f"[HabitatEnv] Scene .glb not found. Using stage_config.json: {sc}", flush=True)
                        sim_cfg.scene_id = sc
                    else:
                        raise FileNotFoundError(f"Scene file not found: {sim_cfg.scene_id}. Provide a valid .scene_instance.json or .stage_config.json.")
                else:
                    raise FileNotFoundError(f"Scene file not found: {sim_cfg.scene_id}.")
        except Exception as e:
            print(f"[HabitatEnv] Scene path validation error: {e}", flush=True)
            raise
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
        try:
            if hasattr(depth_sensor_spec, "normalize_depth"):
                depth_sensor_spec.normalize_depth = False
        except Exception:
            pass
        try:
            md = float(os.environ.get("HABITAT_DEPTH_MAX", str(SENSOR_RANGE)))
            if hasattr(depth_sensor_spec, "max_depth"):
                depth_sensor_spec.max_depth = md
            if hasattr(depth_sensor_spec, "far"):
                depth_sensor_spec.far = md
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
                 scene_id = sim_cfg.scene_id
                 candidates = []
                 if scene_id.endswith(".basis.glb"):
                     candidates.append(scene_id.replace(".basis.glb", ".navmesh"))
                 if scene_id.endswith(".glb"):
                     candidates.append(scene_id.replace(".glb", ".navmesh"))
                 if scene_id.endswith(".scene_instance.json"):
                     candidates.append(scene_id.replace(".scene_instance.json", ".navmesh"))
                 if scene_id.endswith(".stage_config.json"):
                     candidates.append(scene_id.replace(".stage_config.json", ".navmesh"))
                 candidates = [c for i, c in enumerate(candidates) if c and c not in candidates[:i]]

                 navmesh_file = candidates[0] if candidates else scene_id + ".navmesh"
                 loaded = False
                 for cand in candidates:
                     if os.path.exists(cand):
                         sim.pathfinder.load_nav_mesh(cand)
                         print(f"[HabitatEnv] Loaded NavMesh from {cand}", flush=True)
                         loaded = True
                         navmesh_file = cand
                         break
                 else:
                     loaded = False

                 if not loaded:
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
            print(f"[HabitatEnv] PathFinder loaded: {bool(sim.pathfinder.is_loaded)}", flush=True)
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
        
        if floor_height is None:
            try:
                floor_height = float(self.sim.get_agent(0).get_state().position[1])
            except Exception:
                floor_height = None
            if floor_height is None or not np.isfinite(floor_height):
                try:
                    p = pf.get_random_navigable_point()
                    floor_height = float(p[1])
                except Exception:
                    floor_height = float(bounds[0][1])
        try:
            gt_height_offset = float(os.environ.get("HABITAT_GT_HEIGHT_OFFSET", "0.0"))
        except Exception:
            gt_height_offset = 0.0
        gt_height_offset = float(max(-1.0, min(gt_height_offset, 1.0)))
        check_height = float(floor_height) + gt_height_offset
        
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
        try:
            gt_use_snap = int(os.environ.get("HABITAT_GT_USE_SNAP", "1")) != 0
        except Exception:
            gt_use_snap = True
        try:
            gt_snap_dist_ratio = float(os.environ.get("HABITAT_GT_SNAP_DIST_RATIO", "0.75"))
        except Exception:
            gt_snap_dist_ratio = 0.75
        gt_snap_dist_ratio = float(max(0.0, min(gt_snap_dist_ratio, 5.0)))
        snap_tol = float(cell_size) * float(gt_snap_dist_ratio)
        snap_tol2 = float(snap_tol) * float(snap_tol)
        try:
            gt_floor_y_tol = float(os.environ.get("HABITAT_GT_FLOOR_Y_TOL", "0.6"))
        except Exception:
            gt_floor_y_tol = 0.6
        gt_floor_y_tol = float(max(0.05, min(gt_floor_y_tol, 5.0)))

        y_candidates = [float(check_height), float(check_height + 0.5), float(check_height - 0.5)]
        try:
            y_candidates.extend([float(bounds[0][1]), float(bounds[1][1])])
        except Exception:
            pass
        for py in range(min_py, max_py):
            for px in range(min_px, max_px):
                # Pixel to World
                x = px * cell_size + origin_x
                z = py * cell_size + origin_y
                
                if gt_use_snap:
                    snapped = None
                    for yy in y_candidates:
                        try:
                            sp = pf.snap_point(np.array([x, float(yy), z], dtype=np.float32))
                        except Exception:
                            sp = None
                        if sp is not None and np.all(np.isfinite(sp)):
                            snapped = sp
                            break
                    if snapped is not None:
                        dx = float(snapped[0]) - float(x)
                        dz = float(snapped[2]) - float(z)
                        dy = float(snapped[1]) - float(check_height)
                        if (dx * dx + dz * dz) <= snap_tol2 and abs(dy) <= gt_floor_y_tol:
                            gt_map[py, px] = 255
                            free_count += 1
                else:
                    point = np.array([x, check_height, z], dtype=np.float32)
                    if pf.is_navigable(point):
                        gt_map[py, px] = 255 # FREE
                        free_count += 1
        
        print(f"[HabitatEnv] Ground Truth Map generated. Free cells: {free_count}", flush=True)
        return gt_map

    def _get_ground_truth_height_map(self, floor_height=None):
        print("[HabitatEnv] Generating Ground Truth Height Map from PathFinder...", flush=True)
        h, w = self.robot_belief.shape
        height_map = np.full((h, w), np.nan, dtype=np.float32)

        if not self.sim or not self.sim.pathfinder.is_loaded:
            print("[HabitatEnv] Critical Warning: PathFinder not available. Returning empty Height Map.", flush=True)
            return height_map

        pf = self.sim.pathfinder
        bounds = pf.get_bounds()

        if floor_height is None:
            try:
                floor_height = float(self.sim.get_agent(0).get_state().position[1])
            except Exception:
                floor_height = None
            if floor_height is None or not np.isfinite(floor_height):
                try:
                    p = pf.get_random_navigable_point()
                    floor_height = float(p[1])
                except Exception:
                    floor_height = float(bounds[0][1])

        try:
            gt_height_offset = float(os.environ.get("HABITAT_GT_HEIGHT_OFFSET", "0.0"))
        except Exception:
            gt_height_offset = 0.0
        gt_height_offset = float(max(-1.0, min(gt_height_offset, 1.0)))
        check_height = float(floor_height) + gt_height_offset

        cell_size = self.mapper.cell_size
        origin_x = self.mapper.origin_x
        origin_y = self.mapper.origin_y

        min_x_world = bounds[0][0]
        max_x_world = bounds[1][0]
        min_z_world = bounds[0][2]
        max_z_world = bounds[1][2]

        min_px = int((min_x_world - origin_x) / cell_size)
        max_px = int((max_x_world - origin_x) / cell_size) + 1
        min_py = int((min_z_world - origin_y) / cell_size)
        max_py = int((max_z_world - origin_y) / cell_size) + 1

        min_px = max(0, min_px)
        max_px = min(w, max_px)
        min_py = max(0, min_py)
        max_py = min(h, max_py)

        try:
            gt_snap_dist_ratio = float(os.environ.get("HABITAT_GT_SNAP_DIST_RATIO", "0.75"))
        except Exception:
            gt_snap_dist_ratio = 0.75
        gt_snap_dist_ratio = float(max(0.0, min(gt_snap_dist_ratio, 5.0)))
        snap_tol = float(cell_size) * float(gt_snap_dist_ratio)
        snap_tol2 = float(snap_tol) * float(snap_tol)

        try:
            gt_floor_y_tol = float(os.environ.get("HABITAT_GT_FLOOR_Y_TOL", "0.6"))
        except Exception:
            gt_floor_y_tol = 0.6
        gt_floor_y_tol = float(max(0.05, min(gt_floor_y_tol, 5.0)))

        y_candidates = [float(check_height), float(check_height + 0.5), float(check_height - 0.5)]
        try:
            y_candidates.extend([float(bounds[0][1]), float(bounds[1][1])])
        except Exception:
            pass

        for py in range(min_py, max_py):
            for px in range(min_px, max_px):
                x = px * cell_size + origin_x
                z = py * cell_size + origin_y
                snapped = None
                for yy in y_candidates:
                    try:
                        sp = pf.snap_point(np.array([x, float(yy), z], dtype=np.float32))
                    except Exception:
                        sp = None
                    if sp is not None and np.all(np.isfinite(sp)):
                        snapped = sp
                        break
                if snapped is None:
                    continue
                dx = float(snapped[0]) - float(x)
                dz = float(snapped[2]) - float(z)
                dy = float(snapped[1]) - float(check_height)
                if (dx * dx + dz * dz) <= snap_tol2 and abs(dy) <= gt_floor_y_tol:
                    height_map[py, px] = float(snapped[1])

        return height_map

    def update_ground_truth_for_floor(self, height):
        """
        Regenerate the ground truth map for a specific floor height.
        This is crucial for multi-floor environments to ensure the GT map matches the current floor.
        """
        try:
            gt_floor_bin = float(os.environ.get("HABITAT_GT_FLOOR_BIN", "0.5"))
        except Exception:
            gt_floor_bin = 0.5
        gt_floor_bin = float(max(0.05, min(gt_floor_bin, 5.0)))
        key = float(round(float(height) / gt_floor_bin) * gt_floor_bin)

        try:
            cache = getattr(self, "_gt_cache", None)
        except Exception:
            cache = None
        if cache is None:
            self._gt_cache = {}
            cache = self._gt_cache

        if key in cache:
            entry = cache[key]
            if isinstance(entry, dict):
                self.ground_truth = entry.get("map", None)
                self.ground_truth_height = entry.get("height", None)
            else:
                self.ground_truth = entry
                self.ground_truth_height = None
        else:
            gt_map = self._get_ground_truth_map(floor_height=key)
            try:
                gt_height = self._get_ground_truth_height_map(floor_height=key)
            except Exception:
                gt_height = None
            cache[key] = {"map": gt_map, "height": gt_height}
            self.ground_truth = gt_map
            self.ground_truth_height = gt_height
        self.ground_truth_info = MapInfo(self.ground_truth, self.belief_origin_x, self.belief_origin_y, CELL_SIZE)
        print(f"[HabitatEnv] Ground Truth updated for height {key:.2f}m", flush=True)


    def _detect_stairs_from_navmesh(self, sample_points=4000, grid_size=1.0, height_bin=0.75, max_stairs=4):
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
        try:
            self.stairs_coords_list = []
        except Exception:
            self.stairs_coords_list = []
        try:
            self.discovered_stairs = set()
        except Exception:
            self.discovered_stairs = set()

        try:
            enable = int(os.environ.get("HABITAT_STAIRS_GEN_ENABLE", "1")) != 0
        except Exception:
            enable = True
        if not enable:
            return

        has_vlm = bool(getattr(self, "use_vlm_stairs", True)) and hasattr(self, "vlm") and self.vlm is not None
        has_vlm_backend = False
        if has_vlm:
            try:
                v = self.vlm
                has_vlm_backend = bool(getattr(v, "client", None)) or bool(getattr(v, "use_legacy_openai", False)) or bool(getattr(v, "http_fallback", False))
            except Exception:
                has_vlm_backend = False

        try:
            navmesh_enable_default = "0"
            navmesh_enable = int(os.environ.get("HABITAT_STAIRS_NAVMESH_ENABLE", navmesh_enable_default)) != 0
        except Exception:
            navmesh_enable = False

        if not navmesh_enable:
            return

        try:
            sample_points = int(os.environ.get("HABITAT_STAIRS_NAVMESH_SAMPLES", "5000"))
        except Exception:
            sample_points = 5000
        sample_points = int(max(500, min(sample_points, 30000)))
        try:
            grid_size = float(os.environ.get("HABITAT_STAIRS_NAVMESH_GRID_M", "1.0"))
        except Exception:
            grid_size = 1.0
        grid_size = float(max(0.5, min(grid_size, 4.0)))
        try:
            height_bin = float(os.environ.get("HABITAT_STAIRS_NAVMESH_HBIN_M", "0.75"))
        except Exception:
            height_bin = 0.75
        height_bin = float(max(0.25, min(height_bin, 2.0)))
        try:
            max_stairs = int(os.environ.get("HABITAT_STAIRS_MAX", "6"))
        except Exception:
            max_stairs = 6
        max_stairs = int(max(1, min(max_stairs, 20)))

        try:
            stairs = self._detect_stairs_from_navmesh(sample_points=sample_points, grid_size=grid_size, height_bin=height_bin, max_stairs=max_stairs)
        except Exception:
            stairs = []
        if stairs:
            pf = None
            try:
                pf = self.sim.pathfinder if (self.sim is not None and self.sim.pathfinder.is_loaded) else None
            except Exception:
                pf = None

            try:
                max_geo = float(os.environ.get("HABITAT_STAIRS_MAX_GEO_M", "60.0"))
            except Exception:
                max_geo = 60.0
            max_geo = float(max(5.0, min(max_geo, 500.0)))

            try:
                max_snap = float(os.environ.get("HABITAT_STAIRS_MAX_SNAP_M", "2.0"))
            except Exception:
                max_snap = 2.0
            max_snap = float(max(0.25, min(max_snap, 20.0)))

            height = None
            try:
                height = float(self.sim.get_agent(0).get_state().position[1])
            except Exception:
                height = None
            if height is None or (not np.isfinite(height)):
                try:
                    p = self.sim.pathfinder.get_random_navigable_point()
                    height = float(p[1])
                except Exception:
                    height = 0.0

            selected = []
            for x, z in stairs:
                x = float(x)
                z = float(z)
                if self.mapper is not None:
                    try:
                        mx = int((x - float(self.mapper.origin_x)) / float(self.mapper.cell_size))
                        my = int((z - float(self.mapper.origin_y)) / float(self.mapper.cell_size))
                        h, w = self.mapper.global_map.shape
                        if not (0 <= mx < w and 0 <= my < h):
                            continue
                    except Exception:
                        pass

                x2 = x
                z2 = z
                if pf is not None:
                    try:
                        sp = pf.snap_point(np.array([x, float(height), z], dtype=np.float32))
                    except Exception:
                        sp = None
                    if sp is None or (not np.all(np.isfinite(sp))):
                        continue
                    if float(np.hypot(float(sp[0]) - x, float(sp[2]) - z)) > float(max_snap):
                        continue
                    x2 = float(sp[0])
                    z2 = float(sp[2])

                if hasattr(self, "ground_truth") and self.ground_truth is not None and self.mapper is not None:
                    try:
                        mx2 = int((x2 - float(self.mapper.origin_x)) / float(self.mapper.cell_size))
                        my2 = int((z2 - float(self.mapper.origin_y)) / float(self.mapper.cell_size))
                        if 0 <= my2 < self.ground_truth.shape[0] and 0 <= mx2 < self.ground_truth.shape[1]:
                            if not (int(self.ground_truth[my2, mx2]) > 200):
                                continue
                    except Exception:
                        pass

                geo = None
                if hasattr(self, "get_shortest_path_with_distance"):
                    try:
                        _, geo = self.get_shortest_path_with_distance(self.robot_location, np.array([x2, z2], dtype=np.float32))
                    except Exception:
                        geo = None
                if geo is None or (not np.isfinite(float(geo))):
                    continue
                if float(geo) > float(max_geo):
                    continue

                selected.append((float(geo), [float(x2), float(z2)]))

            selected.sort(key=lambda t: t[0])
            self.stairs_coords_list = [p for _, p in selected[: int(max_stairs)]]

    def discover_stairs(self):
        """
        Check if robot is close to any stairs and mark them as discovered.
        """
        if not self.stairs_coords_list:
            return

        try:
            discovery_radius = float(os.environ.get("HABITAT_STAIRS_DISCOVERY_RADIUS_M", "5.0"))
        except Exception:
            discovery_radius = 5.0
        discovery_radius = float(max(1.0, min(discovery_radius, 20.0)))

        robot_pos = self.robot_location
        for idx, coords in enumerate(self.stairs_coords_list):
            dist = np.linalg.norm(robot_pos - np.array(coords))
            if dist < float(discovery_radius):
                if idx not in self.discovered_stairs:
                    print(f"[HabitatEnv] Discovered stairs {idx} at {coords}!", flush=True)
                    self.discovered_stairs.add(idx)
                    try:
                        save_snap = int(os.environ.get("HABITAT_SAVE_STAIRS_DISCOVERY", "1")) != 0
                    except Exception:
                        save_snap = True
                    if save_snap:
                        try:
                            if not os.path.exists(gifs_path):
                                os.makedirs(gifs_path)
                        except Exception:
                            pass
                        try:
                            snap_id = int(getattr(self, "_stairs_discovery_snap_id", 0)) + 1
                        except Exception:
                            snap_id = 1
                        self._stairs_discovery_snap_id = snap_id

                        step_tag = getattr(self, "_last_plot_step", None)
                        if step_tag is None:
                            step_tag = f"stairs_{idx}_{snap_id}"
                        else:
                            step_tag = f"{step_tag}_stairs_{idx}_{snap_id}"

                        old_plot = bool(getattr(self, "plot", False))
                        old_target = getattr(self, "current_target_stairs_index", None)
                        try:
                            self.plot = True
                            self.current_target_stairs_index = idx
                            self.plot_env(step_tag)
                        except Exception:
                            pass
                        try:
                            if hasattr(self, "rgb_image") and self.rgb_image is not None:
                                rgb = self.rgb_image
                                if getattr(rgb, "ndim", 0) == 3 and rgb.shape[-1] >= 3:
                                    rgb_path = f"{gifs_path}/{self.episode_index}_{step_tag}_rgb.png"
                                    plt.imsave(rgb_path, rgb[..., :3])
                                    if hasattr(self, "frame_files"):
                                        self.frame_files.append(rgb_path)
                        except Exception:
                            pass
                        self.current_target_stairs_index = old_target
                        self.plot = old_plot

    def _scan_stairs_in_view(self):
        use_vlm = bool(getattr(self, "use_vlm_stairs", True)) and hasattr(self, "vlm") and self.vlm is not None and hasattr(self.vlm, "detect_stairs_in_rgb")
        if not use_vlm:
            return
        try:
            enable = int(os.environ.get("HABITAT_STAIRS_VLM_VIEW_ENABLE", "1")) != 0
        except Exception:
            enable = True
        if not enable:
            return

        step_i = int(getattr(self, "_step_counter", 0))
        try:
            interval = int(os.environ.get("HABITAT_STAIRS_VLM_VIEW_INTERVAL", "20"))
        except Exception:
            interval = 20
        interval = int(max(1, min(400, interval)))
        if step_i - int(getattr(self, "_vlm_view_last_step", -10**9)) < interval:
            return
        self._vlm_view_last_step = step_i

        res = self.vlm.detect_stairs_in_rgb(getattr(self, "rgb_image", None))
        is_stairs = res.get("is_stairs", None) if isinstance(res, dict) else None
        conf = res.get("confidence", None) if isinstance(res, dict) else None
        try:
            conf_thr = float(os.environ.get("HABITAT_STAIRS_VLM_CONF", "0.55"))
        except Exception:
            conf_thr = 0.55

        try:
            save_all = int(os.environ.get("HABITAT_STAIRS_VLM_VIEW_SAVE_ALL", "0")) != 0
        except Exception:
            save_all = False

        should_save = bool(save_all)
        if is_stairs is True and (conf is None or float(conf) >= float(conf_thr)):
            should_save = True

        confirmed = False
        try:
            high_conf = float(os.environ.get("HABITAT_STAIRS_VLM_HIGH_CONF", "0.75"))
        except Exception:
            high_conf = 0.75
        high_conf = float(max(float(conf_thr), min(high_conf, 1.0)))

        try:
            hist = list(getattr(self, "_vlm_view_stairs_hist", []))
        except Exception:
            hist = []
        if is_stairs is True and (conf is None or float(conf) >= float(conf_thr)):
            hist.append((True, None if conf is None else float(conf)))
        else:
            hist.append((False, None if conf is None else float(conf)))
        if len(hist) > 3:
            hist = hist[-3:]
        self._vlm_view_stairs_hist = hist

        try:
            yes_cnt = int(sum(1 for a, _ in hist if bool(a)))
        except Exception:
            yes_cnt = 0
        if is_stairs is True and conf is not None and float(conf) >= float(high_conf):
            confirmed = True
        elif yes_cnt >= 2 and is_stairs is True:
            confirmed = True

        if confirmed:
            try:
                merge_r = float(os.environ.get("HABITAT_STAIRS_MERGE_RADIUS_M", "3.0"))
            except Exception:
                merge_r = 3.0
            merge_r = float(max(0.5, min(merge_r, 15.0)))

            base_xy = np.array(self.robot_location, dtype=np.float32).reshape(2)
            try:
                ahead_m = float(os.environ.get("HABITAT_STAIRS_VLM_AHEAD_M", "3.0"))
            except Exception:
                ahead_m = 3.0
            ahead_m = float(max(0.0, min(ahead_m, 10.0)))

            dir_xy = None
            target_xy = base_xy.copy()

            depth_xy = None
            depth_ok = False
            try:
                dimg = getattr(self, "depth_image", None)
                ss = getattr(self, "_last_depth_sensor_state", None)
                if dimg is not None and ss is not None:
                    d = np.asarray(dimg)
                    if d.ndim == 3 and int(d.shape[-1]) == 1:
                        d = d[..., 0]
                    if d.ndim == 2 and int(d.shape[0]) >= 4 and int(d.shape[1]) >= 4:
                        h, w = int(d.shape[0]), int(d.shape[1])
                        try:
                            roi_y0 = float(os.environ.get("HABITAT_STAIRS_VLM_DEPTH_ROI_Y0", "0.55"))
                        except Exception:
                            roi_y0 = 0.55
                        try:
                            roi_y1 = float(os.environ.get("HABITAT_STAIRS_VLM_DEPTH_ROI_Y1", "0.92"))
                        except Exception:
                            roi_y1 = 0.92
                        try:
                            roi_x0 = float(os.environ.get("HABITAT_STAIRS_VLM_DEPTH_ROI_X0", "0.25"))
                        except Exception:
                            roi_x0 = 0.25
                        try:
                            roi_x1 = float(os.environ.get("HABITAT_STAIRS_VLM_DEPTH_ROI_X1", "0.75"))
                        except Exception:
                            roi_x1 = 0.75
                        roi_y0 = float(max(0.0, min(roi_y0, 0.99)))
                        roi_y1 = float(max(0.01, min(roi_y1, 1.0)))
                        roi_x0 = float(max(0.0, min(roi_x0, 0.99)))
                        roi_x1 = float(max(0.01, min(roi_x1, 1.0)))
                        if roi_y1 > roi_y0 and roi_x1 > roi_x0:
                            y0 = int(max(0, min(h - 1, round(roi_y0 * (h - 1)))))
                            y1 = int(max(0, min(h - 1, round(roi_y1 * (h - 1)))))
                            x0 = int(max(0, min(w - 1, round(roi_x0 * (w - 1)))))
                            x1 = int(max(0, min(w - 1, round(roi_x1 * (w - 1)))))
                            if y1 > y0 and x1 > x0:
                                roi = d[y0 : y1 + 1, x0 : x1 + 1].astype(np.float32, copy=False)
                                try:
                                    max_depth = float(os.environ.get("HABITAT_DEPTH_MAX", "8.0"))
                                except Exception:
                                    max_depth = 8.0
                                max_depth = float(max(2.0, min(max_depth, 30.0)))
                                is_norm = False
                                try:
                                    p99 = float(np.nanpercentile(roi, 99))
                                    maxv = float(np.nanmax(roi))
                                    is_norm = np.isfinite(p99) and (0.80 <= p99 <= 1.05) and np.isfinite(maxv) and (maxv <= 1.05)
                                except Exception:
                                    is_norm = False
                                if is_norm:
                                    roi_m = roi * float(max_depth)
                                else:
                                    roi_m = roi
                                valid = np.isfinite(roi_m) & (roi_m > 0.05) & (roi_m < (float(max_depth) - 0.05))
                                if np.any(valid):
                                    try:
                                        edge_thr = float(os.environ.get("HABITAT_STAIRS_VLM_DEPTH_EDGE_M", "0.08"))
                                    except Exception:
                                        edge_thr = 0.08
                                    edge_thr = float(max(0.0, min(edge_thr, 1.0)))
                                    try:
                                        min_edges = int(os.environ.get("HABITAT_STAIRS_VLM_DEPTH_MIN_EDGES", "4"))
                                    except Exception:
                                        min_edges = 4
                                    min_edges = int(max(0, min(min_edges, 50)))
                                    try:
                                        min_span = float(os.environ.get("HABITAT_STAIRS_VLM_DEPTH_MIN_SPAN_M", "0.25"))
                                    except Exception:
                                        min_span = 0.25
                                    min_span = float(max(0.0, min(min_span, 5.0)))

                                    roi_masked = np.where(valid, roi_m, np.nan)
                                    try:
                                        row_med = np.nanmedian(roi_masked, axis=1).astype(np.float32, copy=False)
                                    except Exception:
                                        row_med = None
                                    if row_med is not None and row_med.ndim == 1:
                                        finite = np.isfinite(row_med)
                                        if int(np.count_nonzero(finite)) >= 6:
                                            rm = row_med[finite]
                                            try:
                                                med_k = int(os.environ.get("HABITAT_STAIRS_VLM_DEPTH_ROW_MED_K", "5"))
                                            except Exception:
                                                med_k = 5
                                            med_k = int(max(0, min(med_k, 31)))
                                            rm_s = rm
                                            if med_k >= 3 and (med_k % 2) == 1 and rm.size >= med_k:
                                                pad = med_k // 2
                                                padded = np.pad(rm.astype(np.float32, copy=False), (pad, pad), mode="edge")
                                                sm = np.empty_like(rm, dtype=np.float32)
                                                for ii in range(int(rm.size)):
                                                    sm[ii] = float(np.median(padded[ii : ii + med_k]))
                                                rm_s = sm

                                            span = float(np.nanpercentile(rm_s, 90) - np.nanpercentile(rm_s, 10))
                                            dif = np.diff(rm_s.astype(np.float32, copy=False))
                                            edge_idx = np.flatnonzero(np.abs(dif) >= float(edge_thr))
                                            if edge_idx.size > 0:
                                                keep = np.ones((edge_idx.size,), dtype=bool)
                                                if edge_idx.size > 1:
                                                    keep[1:] = (np.diff(edge_idx) > 1)
                                                edge_idx = edge_idx[keep]
                                            edges = int(edge_idx.size)
                                            try:
                                                span_ratio = float(os.environ.get("HABITAT_STAIRS_VLM_DEPTH_EDGE_SPAN_RATIO", "0.35"))
                                            except Exception:
                                                span_ratio = 0.35
                                            span_ratio = float(max(0.0, min(span_ratio, 1.0)))
                                            edge_span_ok = False
                                            if edges >= 2:
                                                edge_span = int(edge_idx.max() - edge_idx.min())
                                                edge_span_ok = edge_span >= int(np.ceil(float(span_ratio) * float(max(1, int(rm_s.size) - 1))))
                                            try:
                                                min_pix = int(os.environ.get("HABITAT_STAIRS_EDGE_MIN_PIX", "10"))
                                            except Exception:
                                                min_pix = 10
                                            try:
                                                max_pix = int(os.environ.get("HABITAT_STAIRS_EDGE_MAX_PIX", "60"))
                                            except Exception:
                                                max_pix = 60
                                            try:
                                                max_edges = int(os.environ.get("HABITAT_STAIRS_VLM_DEPTH_MAX_EDGES", "12"))
                                            except Exception:
                                                max_edges = 12
                                            min_pix = int(max(1, min(min_pix, 500)))
                                            max_pix = int(max(min_pix, min(max_pix, 1000)))
                                            max_edges = int(max(int(min_edges), min(max_edges, 200)))
                                            try:
                                                max_cv = float(os.environ.get("HABITAT_STAIRS_EDGE_SPACING_CV_MAX", "0.60"))
                                            except Exception:
                                                max_cv = 0.60
                                            max_cv = float(max(0.0, min(max_cv, 2.0)))
                                            spacing_ok = False
                                            slope_ok = False
                                            if edges >= 3:
                                                difi = np.diff(edge_idx.astype(np.float32, copy=False))
                                                if difi.size > 0:
                                                    sp_med = float(np.median(difi))
                                                    sp_std = float(np.std(difi))
                                                    cv = float(sp_std) / float(max(1e-6, sp_med))
                                                    spacing_ok = (sp_med >= float(min_pix)) and (sp_med <= float(max_pix)) and (cv <= float(max_cv))
                                            try:
                                                min_slope = float(os.environ.get("HABITAT_STAIRS_ROW_DEPTH_MIN_SLOPE", "0.002"))
                                            except Exception:
                                                min_slope = 0.002
                                            min_slope = float(max(0.0, min(min_slope, 0.2)))
                                            x = np.arange(rm_s.size, dtype=np.float32)
                                            xm = float(np.mean(x)); ym = float(np.mean(rm_s))
                                            num = float(np.sum((x - xm) * (rm_s - ym)))
                                            den = float(np.sum((x - xm) ** 2.0))
                                            slope = float(num) / float(max(1e-6, den))
                                            slope_ok = abs(float(slope)) >= float(min_slope)
                                            if edges >= int(min_edges) and edges <= int(max_edges) and span >= float(min_span) and (edge_span_ok or edges < 2) and spacing_ok and slope_ok:
                                                depth_ok = True
                                    try:
                                        q = float(os.environ.get("HABITAT_STAIRS_VLM_DEPTH_Q", "35.0"))
                                    except Exception:
                                        q = 35.0
                                    q = float(max(0.0, min(q, 100.0)))
                                    dv = float(np.nanpercentile(np.where(valid, roi_m, np.nan), q))
                                    if np.isfinite(dv) and dv > 0.10:
                                        uu = float(0.5 * (x0 + x1))
                                        vv = float(0.5 * (y0 + y1))
                                        try:
                                            hfov = float(os.environ.get("HABITAT_SENSOR_HFOV", "90"))
                                        except Exception:
                                            hfov = 90.0
                                        fx = (float(w) / 2.0) / float(np.tan(np.deg2rad(float(hfov) / 2.0)))
                                        vfov = 2.0 * float(np.arctan((float(h) / float(max(1.0, float(w)))) * np.tan(np.deg2rad(float(hfov) / 2.0))))
                                        fy = (float(h) / 2.0) / float(np.tan(float(vfov) / 2.0))
                                        cx = (float(w) - 1.0) / 2.0
                                        cy = (float(h) - 1.0) / 2.0
                                        x_cam = (uu - cx) * dv / float(fx)
                                        y_cam = -(vv - cy) * dv / float(fy)
                                        z_cam = -dv
                                        v_cam = np.array([x_cam, y_cam, z_cam], dtype=np.float32)
                                        r = getattr(ss, "rotation", None)
                                        t = getattr(ss, "position", None)
                                        if r is not None and t is not None:
                                            if hasattr(r, "x"):
                                                qx, qy, qz, qw = float(r.x), float(r.y), float(r.z), float(r.w)
                                            else:
                                                qx, qy, qz, qw = float(r[0]), float(r[1]), float(r[2]), float(r[3])
                                            nrm = float(np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw))
                                            if nrm > 1e-9:
                                                qx, qy, qz, qw = qx / nrm, qy / nrm, qz / nrm, qw / nrm
                                                tx, ty, tz = float(t[0]), float(t[1]), float(t[2])
                                                vx, vy, vz = float(v_cam[0]), float(v_cam[1]), float(v_cam[2])
                                                t2x = 2.0 * (qy * vz - qz * vy)
                                                t2y = 2.0 * (qz * vx - qx * vz)
                                                t2z = 2.0 * (qx * vy - qy * vx)
                                                rx = vx + qw * t2x + (qy * t2z - qz * t2y)
                                                ry = vy + qw * t2y + (qz * t2x - qx * t2z)
                                                rz = vz + qw * t2z + (qx * t2y - qy * t2x)
                                                wx = float(tx) + float(rx)
                                                wz = float(tz) + float(rz)
                                                if np.isfinite(wx) and np.isfinite(wz):
                                                    depth_xy = np.array([wx, wz], dtype=np.float32)
            except Exception:
                depth_xy = None
                depth_ok = False

            try:
                require_depth = int(os.environ.get("HABITAT_STAIRS_VLM_REQUIRE_DEPTH", "1")) != 0
            except Exception:
                require_depth = True
            if require_depth and (not depth_ok):
                confirmed = False

            if depth_xy is not None and self.mapper is not None and hasattr(self, "robot_belief") and self.robot_belief is not None:
                try:
                    mx0 = int(round((float(depth_xy[0]) - float(self.mapper.origin_x)) / float(self.mapper.cell_size)))
                    my0 = int(round((float(depth_xy[1]) - float(self.mapper.origin_y)) / float(self.mapper.cell_size)))
                    h_map, w_map = self.robot_belief.shape
                    try:
                        snap_r = int(os.environ.get("HABITAT_STAIRS_VLM_BELIEF_SNAP_R", "10"))
                    except Exception:
                        snap_r = 10
                    snap_r = int(max(0, min(snap_r, 80)))
                    best = None
                    if 0 <= mx0 < w_map and 0 <= my0 < h_map:
                        if int(self.robot_belief[my0, mx0]) == 255:
                            best = (mx0, my0)
                    if best is None and snap_r > 0:
                        y0 = int(max(0, my0 - snap_r))
                        y1 = int(min(h_map - 1, my0 + snap_r))
                        x0 = int(max(0, mx0 - snap_r))
                        x1 = int(min(w_map - 1, mx0 + snap_r))
                        patch = self.robot_belief[y0 : y1 + 1, x0 : x1 + 1]
                        free = np.argwhere(patch == 255)
                        if free.size > 0:
                            dy = free[:, 0].astype(np.float32) - float(my0 - y0)
                            dx = free[:, 1].astype(np.float32) - float(mx0 - x0)
                            d2 = dx * dx + dy * dy
                            j = int(np.argmin(d2))
                            best = (int(x0 + int(free[j, 1])), int(y0 + int(free[j, 0])))
                    if best is not None:
                        target_xy = np.array([float(best[0]) * float(self.mapper.cell_size) + float(self.mapper.origin_x),
                                              float(best[1]) * float(self.mapper.cell_size) + float(self.mapper.origin_y)], dtype=np.float32)
                except Exception:
                    pass

            try:
                if self.sim:
                    st = self.sim.get_agent(0).get_state()
                    q = getattr(st, "rotation", None)
                    if q is not None and hasattr(q, "w") and hasattr(q, "y"):
                        yaw = 2.0 * float(np.arctan2(float(q.y), float(q.w)))
                        dir_xy = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float32)
                        if depth_xy is None:
                            target_xy = base_xy + dir_xy * float(ahead_m)
                    elif len(getattr(self, "trajectory_x", [])) >= 2 and len(getattr(self, "trajectory_y", [])) >= 2:
                        dx = float(self.trajectory_x[-1]) - float(self.trajectory_x[-2])
                        dy = float(self.trajectory_y[-1]) - float(self.trajectory_y[-2])
                        n = float(np.hypot(dx, dy))
                        if n > 1e-3:
                            dir_xy = np.array([dx / n, dy / n], dtype=np.float32)
                            if depth_xy is None:
                                target_xy = base_xy + dir_xy * float(ahead_m)
            except Exception:
                target_xy = base_xy.copy()

            try:
                if self.sim and self.sim.pathfinder.is_loaded:
                    st = self.sim.get_agent(0).get_state()
                    y = float(st.position[1])
                    sp = self.sim.pathfinder.snap_point(np.array([float(target_xy[0]), y, float(target_xy[1])], dtype=np.float32))
                    if sp is not None and np.all(np.isfinite(sp)):
                        target_xy = np.array([float(sp[0]), float(sp[2])], dtype=np.float32)
            except Exception:
                pass
            if confirmed:
                best_i = None
                try:
                    for i0, c0 in enumerate(self.stairs_coords_list):
                        if np.linalg.norm(target_xy - np.array(c0, dtype=np.float32).reshape(2)) <= merge_r:
                            best_i = int(i0)
                            break
                except Exception:
                    best_i = None

                if best_i is None:
                    try:
                        self.stairs_coords_list.append([float(target_xy[0]), float(target_xy[1])])
                        best_i = int(len(self.stairs_coords_list) - 1)
                    except Exception:
                        best_i = None

                if best_i is not None:
                    try:
                        self.discovered_stairs.add(int(best_i))
                    except Exception:
                        pass

                    try:
                        save_snap = int(os.environ.get("HABITAT_SAVE_STAIRS_DISCOVERY", "1")) != 0
                    except Exception:
                        save_snap = True
                    if save_snap:
                        try:
                            if not os.path.exists(gifs_path):
                                os.makedirs(gifs_path)
                        except Exception:
                            pass
                        try:
                            snap_id = int(getattr(self, "_stairs_discovery_snap_id", 0)) + 1
                        except Exception:
                            snap_id = 1
                        self._stairs_discovery_snap_id = snap_id

                        step_tag = getattr(self, "_last_plot_step", None)
                        if step_tag is None:
                            step_tag = f"stairs_vlm_{best_i}_{snap_id}"
                        else:
                            step_tag = f"{step_tag}_stairs_vlm_{best_i}_{snap_id}"

                        old_plot = bool(getattr(self, "plot", False))
                        old_target = getattr(self, "current_target_stairs_index", None)
                        try:
                            self.plot = True
                            self.current_target_stairs_index = int(best_i)
                            self.plot_env(step_tag)
                        except Exception:
                            pass
                        try:
                            if hasattr(self, "rgb_image") and self.rgb_image is not None:
                                rgb0 = self.rgb_image
                                if getattr(rgb0, "ndim", 0) == 3 and rgb0.shape[-1] >= 3:
                                    rgb_path = f"{gifs_path}/{self.episode_index}_{step_tag}_rgb.png"
                                    plt.imsave(rgb_path, rgb0[..., :3])
                                    if hasattr(self, "frame_files"):
                                        self.frame_files.append(rgb_path)
                        except Exception:
                            pass
                        self.current_target_stairs_index = old_target
                        self.plot = old_plot

        if not should_save:
            return

        try:
            if not os.path.exists(gifs_path):
                os.makedirs(gifs_path)
        except Exception:
            return

        try:
            self._vlm_view_snap_id = int(getattr(self, "_vlm_view_snap_id", 0)) + 1
        except Exception:
            self._vlm_view_snap_id = 1

        tag = "UNK" if is_stairs is None else ("YES" if is_stairs else "NO")
        conf_s = "na" if conf is None else f"{float(conf):.2f}"
        ctag = "Y" if bool(confirmed) else "N"
        rgb = getattr(self, "rgb_image", None)
        if rgb is None:
            return

        save_path = f"{gifs_path}/{self.episode_index}_vlmview_stairs_step_{step_i}_{self._vlm_view_snap_id}_{tag}_{conf_s}.png"
        try:
            from PIL import Image, ImageDraw
            arr = np.asarray(rgb)[..., :3]
            if arr.dtype != np.uint8:
                amax = float(np.nanmax(arr)) if np.size(arr) else 1.0
                if np.isfinite(amax) and amax <= 1.05:
                    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
                else:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
            im = Image.fromarray(arr)
            draw = ImageDraw.Draw(im)
            draw.rectangle([(0, 0), (int(im.size[0]), 36)], fill=(0, 0, 0))
            draw.text((8, 8), f"vlm_view_stairs={tag} conf={conf_s} confirmed={ctag} step={step_i}", fill=(255, 255, 255))
            im.save(save_path)
        except Exception:
            try:
                plt.imsave(save_path, np.asarray(rgb)[..., :3])
            except Exception:
                return

        if hasattr(self, "frame_files"):
            try:
                self.frame_files.append(save_path)
            except Exception:
                pass

    def reset(self):
        # Reset episode-specific state variables
        self.frame_files = []
        self._habitat_plot_frames = 0
        self._plot_disabled_maxframes = False
        self.travel_dist = 0
        self.explored_rate = 0
        try:
            self.discovered_stairs = set()
        except Exception:
            self.discovered_stairs = set()
        try:
            self.stairs_coords_list = []
        except Exception:
            self.stairs_coords_list = []
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
            floor_y = float(agent_state.position[1])
            try:
                if self.sim.pathfinder.is_loaded:
                    snap = self.sim.pathfinder.snap_point(np.array(agent_state.position))
                    if np.isfinite(snap[1]):
                        floor_y = float(snap[1])
            except Exception:
                pass
            base_state = SimpleNamespace(position=np.array([float(agent_state.position[0]), floor_y, float(agent_state.position[2])], dtype=np.float32))
            self.robot_belief = self.mapper.update(observations['depth_sensor'], sensor_state, base_state)
            self.belief_info.update_map_info(self.robot_belief, self.mapper.origin_x, self.mapper.origin_y, vlm_rgb_map=self.mapper.get_vlm_rgb_map())
            self.update_robot_location_from_sim(agent_state)
            try:
                self.update_ground_truth_for_floor(height=float(floor_y))
            except Exception:
                pass
            
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

            try:
                self.depth_image = observations.get('depth_sensor', None)
            except Exception:
                try:
                    self.depth_image = observations['depth_sensor']
                except Exception:
                    self.depth_image = None

            try:
                gen_on_reset = int(os.environ.get("HABITAT_STAIRS_GEN_ON_RESET", "1")) != 0
            except Exception:
                gen_on_reset = True
            if gen_on_reset:
                try:
                    self._generate_stairs()
                except Exception:
                    pass
        
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

        try:
            gain_radius_m = float(os.environ.get("HABITAT_FRONTIER_GAIN_RADIUS_M", "4.0"))
        except Exception:
            gain_radius_m = 4.0
        gain_radius_m = float(max(0.5, min(gain_radius_m, 20.0)))
        try:
            r_px = int(round(gain_radius_m / float(self.mapper.cell_size)))
        except Exception:
            r_px = 10
        r_px = int(max(1, min(r_px, 200)))
        unknown_bin = (mask_unknown > 0).astype(np.uint8)
        
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

                gain = float(area)
                try:
                    y0 = int(max(0, cy - r_px))
                    y1 = int(min(int(unknown_bin.shape[0] - 1), cy + r_px))
                    x0 = int(max(0, cx - r_px))
                    x1 = int(min(int(unknown_bin.shape[1] - 1), cx + r_px))
                    patch = unknown_bin[y0:y1 + 1, x0:x1 + 1]
                    gain = float(np.count_nonzero(patch))
                except Exception:
                    gain = float(area)

                self.global_frontiers.append(((wx, wy), gain))
        
        # print(f"[HabitatEnv] Updated Frontiers: {len(self.global_frontiers)} found.", flush=True)

    def _move_and_update(self, target_pos_3d):
        current_state = self.sim.get_agent(0).get_state()
        new_state = habitat_sim.AgentState()
        target_pos_3d = np.array(target_pos_3d, dtype=np.float32).reshape(3)
        curr_pos = np.array(current_state.position, dtype=np.float32).reshape(3)
        try:
            pf = self.sim.pathfinder if (self.sim and self.sim.pathfinder.is_loaded) else None
        except Exception:
            pf = None

        if pf is not None:
            try:
                sp = pf.snap_point(curr_pos)
            except Exception:
                sp = None
            if sp is not None and np.all(np.isfinite(sp)):
                curr_pos = np.array(sp, dtype=np.float32).reshape(3)

            try:
                tp = pf.snap_point(target_pos_3d)
            except Exception:
                tp = None
            if tp is not None and np.all(np.isfinite(tp)):
                target_pos_3d = np.array(tp, dtype=np.float32).reshape(3)

            try:
                max_step = float(os.environ.get("HABITAT_TRY_STEP_MAX_M", "0.25"))
            except Exception:
                max_step = 0.25
            max_step = float(max(0.05, min(max_step, 2.0)))
            try:
                max_iters = int(os.environ.get("HABITAT_TRY_STEP_MAX_ITERS", "12"))
            except Exception:
                max_iters = 12
            max_iters = int(max(1, min(max_iters, 200)))
            try:
                min_prog = float(os.environ.get("HABITAT_TRY_STEP_MIN_PROGRESS_M", "0.01"))
            except Exception:
                min_prog = 0.01
            min_prog = float(max(0.0, min(min_prog, 0.5)))

            curr0 = np.array(curr_pos, dtype=np.float32).reshape(3)
            tgt = np.array(target_pos_3d, dtype=np.float32).reshape(3)
            cur = curr0.copy()
            for _ in range(int(max_iters)):
                dx = float(tgt[0]) - float(cur[0])
                dz = float(tgt[2]) - float(cur[2])
                d = float(np.hypot(dx, dz))
                if d <= float(max_step):
                    sub = tgt
                else:
                    a = float(max_step) / float(max(d, 1e-6))
                    sub = cur + (tgt - cur) * a
                try:
                    stepped = pf.try_step(cur, sub)
                except Exception:
                    stepped = None
                if stepped is None or (not np.all(np.isfinite(stepped))):
                    break
                stepped = np.array(stepped, dtype=np.float32).reshape(3)
                prog = float(np.hypot(float(stepped[0]) - float(cur[0]), float(stepped[2]) - float(cur[2])))
                cur = stepped
                if prog < float(min_prog):
                    break
                if d <= float(max_step):
                    break
            target_pos_3d = cur

        new_state.position = target_pos_3d
        import quaternion
        dx = float(target_pos_3d[0]) - float(curr_pos[0])
        dz = float(target_pos_3d[2]) - float(curr_pos[2])
        yaw = np.arctan2(dz, dx)
        new_state.rotation = quaternion.from_rotation_vector(np.array([0, yaw, 0]))
        self.sim.get_agent(0).set_state(new_state)
        
        observations = self.sim.get_sensor_observations()
        agent_state = self.sim.get_agent(0).get_state()
        sensor_state = agent_state.sensor_states['depth_sensor']
        try:
            self._last_depth_sensor_state = sensor_state
        except Exception:
            pass
        floor_y = float(agent_state.position[1])
        try:
            if self.sim.pathfinder.is_loaded:
                snap = self.sim.pathfinder.snap_point(np.array(agent_state.position))
                if np.isfinite(snap[1]):
                    floor_y = float(snap[1])
        except Exception:
            pass
        base_state = SimpleNamespace(position=np.array([float(agent_state.position[0]), floor_y, float(agent_state.position[2])], dtype=np.float32))
        self.robot_belief = self.mapper.update(observations['depth_sensor'], sensor_state, base_state)
        self.belief_info.update_map_info(self.robot_belief, self.mapper.origin_x, self.mapper.origin_y, vlm_rgb_map=self.mapper.get_vlm_rgb_map())
        self.update_robot_location_from_sim(agent_state)
        
        # Record trajectory at every micro-step for smooth visualization
        self.trajectory_x.append(self.robot_location[0])
        self.trajectory_y.append(self.robot_location[1])
        
        if 'color_sensor' in observations:
            self.rgb_image = observations['color_sensor'][..., :3]
        try:
            self.depth_image = observations.get('depth_sensor', None)
        except Exception:
            try:
                self.depth_image = observations['depth_sensor']
            except Exception:
                self.depth_image = None
            
        if self.plot:
            try:
                interval = int(os.environ.get("HABITAT_PLOT_INTERVAL", "25"))
            except Exception:
                interval = 25
            interval = int(max(1, min(interval, 1000)))
            try:
                max_frames = int(os.environ.get("HABITAT_PLOT_MAX_FRAMES", "300"))
            except Exception:
                max_frames = 300
            max_frames = int(max(0, min(max_frames, 20000)))

            try:
                if max_frames > 0 and int(getattr(self, "_habitat_plot_frames", 0)) >= max_frames:
                    return
            except Exception:
                pass

            step_i = int(getattr(self, "_step_counter", 0))
            last_saved = getattr(self, "_plot_last_step_saved", None)
            if last_saved is not None and int(last_saved) == int(step_i):
                return
            if interval > 1 and step_i > 0 and (step_i % interval) != 0:
                return

            try:
                self._plot_last_step_saved = int(step_i)
            except Exception:
                pass

            try:
                self.plot_env(step_i)
            except OSError as e:
                self.plot = False
                try:
                    print(f"[HabitatEnv] Plot disabled due to save error: {e}", flush=True)
                except Exception:
                    pass
            except Exception:
                pass

    def step(self, next_waypoint, force=False):
        """
        Moves the robot to next_waypoint using Dijkstra/A* pathfinding on the belief map.
        Simulates step-by-step movement to respect obstacles and update map continuously.
        If force=True, falls back to NavMesh pathfinding if Belief Map path fails.
        """
        if not self.sim: return 0
        try:
            self._step_counter = int(getattr(self, "_step_counter", 0)) + 1
        except Exception:
            self._step_counter = 1
        
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
                            print(f"[HabitatEnv] Belief A* failed; using NavMesh path ({len(nav_path)} pts) to {next_waypoint}.", flush=True)
                        else:
                            print(f"[HabitatEnv] NavMesh fallback also failed to {next_waypoint}.", flush=True)
                            return -0.1
                    else:
                        print("[HabitatEnv] NavMesh not available for fallback (PathFinder not loaded).", flush=True)
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
        self._scan_stairs_in_view()
        self.discover_stairs()

        # 4. Calculate Reward
        dist = np.linalg.norm(self.robot_location - old_position)
        self.travel_dist += dist
        self.trajectory_x.append(self.robot_location[0])
        self.trajectory_y.append(self.robot_location[1])
        self.evaluate_exploration_rate()
        
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

        pf = self.sim.pathfinder

        def _safe_snap_point(guess_3d, fallback_3d=None):
            if guess_3d is None:
                return None
            g0 = np.array(guess_3d, dtype=np.float32).reshape(3)
            try:
                fb = None if fallback_3d is None else np.array(fallback_3d, dtype=np.float32).reshape(3)
            except Exception:
                fb = None

            y_candidates = [
                float(g0[1]),
                float(height),
                float(g0[1] + 1.0),
                float(g0[1] - 1.0),
                0.0,
            ]
            for yy in y_candidates:
                gg = g0.copy()
                gg[1] = float(yy)
                try:
                    sp = pf.snap_point(gg)
                except Exception:
                    sp = None
                if sp is not None and np.all(np.isfinite(sp)):
                    return np.array(sp, dtype=np.float32).reshape(3)

            x0 = float(g0[0])
            z0 = float(g0[2])
            for r in (0.5, 1.0, 2.0, 4.0, 8.0, 12.0):
                for k in range(16):
                    ang = (2.0 * math.pi * float(k)) / 16.0
                    gg = np.array([x0 + float(r) * math.cos(ang), float(g0[1]), z0 + float(r) * math.sin(ang)], dtype=np.float32)
                    try:
                        sp = pf.snap_point(gg)
                    except Exception:
                        sp = None
                    if sp is not None and np.all(np.isfinite(sp)):
                        return np.array(sp, dtype=np.float32).reshape(3)

            if fb is not None and np.all(np.isfinite(fb)):
                try:
                    sp = pf.snap_point(fb)
                except Exception:
                    sp = None
                if sp is not None and np.all(np.isfinite(sp)):
                    return np.array(sp, dtype=np.float32).reshape(3)
            return None

        start_fallback = None
        try:
            start_fallback = np.array(agent_state.position, dtype=np.float32).reshape(3)
        except Exception:
            start_fallback = None

        start_3d = _safe_snap_point(start_guess, fallback_3d=start_fallback)
        if start_3d is None:
            print(f"[HabitatEnv] NavMesh snap_point returned NaN. start={start_guess}, end={end_guess}", flush=True)
            return None

        end_3d = _safe_snap_point(end_guess, fallback_3d=None)
        if end_3d is None:
            try:
                stepped = pf.try_step(start_3d, end_guess)
            except Exception:
                stepped = None
            if stepped is None or (not np.all(np.isfinite(stepped))):
                print(f"[HabitatEnv] NavMesh snap_point returned NaN. start={start_guess}, end={end_guess}", flush=True)
                return None
            return [np.array([float(stepped[0]), float(stepped[2])], dtype=np.float32)]
        
        path = habitat_sim.ShortestPath()
        path.requested_start = start_3d
        path.requested_end = end_3d
        
        found = self.sim.pathfinder.find_path(path)
        
        if not found:
            try:
                stepped = pf.try_step(start_3d, end_3d)
            except Exception:
                stepped = None
            if stepped is None or (not np.all(np.isfinite(stepped))):
                return None
            return [np.array([float(stepped[0]), float(stepped[2])], dtype=np.float32)]
            
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

        pf = self.sim.pathfinder

        def _safe_snap_point(guess_3d, fallback_3d=None):
            if guess_3d is None:
                return None
            g0 = np.array(guess_3d, dtype=np.float32).reshape(3)
            try:
                fb = None if fallback_3d is None else np.array(fallback_3d, dtype=np.float32).reshape(3)
            except Exception:
                fb = None

            y_candidates = [
                float(g0[1]),
                float(height),
                float(g0[1] + 1.0),
                float(g0[1] - 1.0),
                0.0,
            ]
            for yy in y_candidates:
                gg = g0.copy()
                gg[1] = float(yy)
                try:
                    sp = pf.snap_point(gg)
                except Exception:
                    sp = None
                if sp is not None and np.all(np.isfinite(sp)):
                    return np.array(sp, dtype=np.float32).reshape(3)

            x0 = float(g0[0])
            z0 = float(g0[2])
            for r in (0.5, 1.0, 2.0, 4.0, 8.0, 12.0):
                for k in range(16):
                    ang = (2.0 * math.pi * float(k)) / 16.0
                    gg = np.array([x0 + float(r) * math.cos(ang), float(g0[1]), z0 + float(r) * math.sin(ang)], dtype=np.float32)
                    try:
                        sp = pf.snap_point(gg)
                    except Exception:
                        sp = None
                    if sp is not None and np.all(np.isfinite(sp)):
                        return np.array(sp, dtype=np.float32).reshape(3)

            if fb is not None and np.all(np.isfinite(fb)):
                try:
                    sp = pf.snap_point(fb)
                except Exception:
                    sp = None
                if sp is not None and np.all(np.isfinite(sp)):
                    return np.array(sp, dtype=np.float32).reshape(3)
            return None

        start_fallback = None
        try:
            start_fallback = np.array(agent_state.position, dtype=np.float32).reshape(3)
        except Exception:
            start_fallback = None

        start_3d = _safe_snap_point(start_guess, fallback_3d=start_fallback)
        if start_3d is None:
            print(f"[HabitatEnv] NavMesh snap_point returned NaN. start={start_guess}, end={end_guess}", flush=True)
            return None, float('inf')

        end_3d = _safe_snap_point(end_guess, fallback_3d=None)
        if end_3d is None:
            try:
                stepped = pf.try_step(start_3d, end_guess)
            except Exception:
                stepped = None
            if stepped is None or (not np.all(np.isfinite(stepped))):
                print(f"[HabitatEnv] NavMesh snap_point returned NaN. start={start_guess}, end={end_guess}", flush=True)
                return None, float('inf')
            step2d = np.array([float(stepped[0]), float(stepped[2])], dtype=np.float32)
            d = float(np.linalg.norm(step2d - np.array([float(start_pos[0]), float(start_pos[1])], dtype=np.float32)))
            return [step2d], d
        
        sp = habitat_sim.ShortestPath()
        sp.requested_start = start_3d
        sp.requested_end = end_3d
        
        found = self.sim.pathfinder.find_path(sp)
        if not found:
            try:
                stepped = pf.try_step(start_3d, end_3d)
            except Exception:
                stepped = None
            if stepped is None or (not np.all(np.isfinite(stepped))):
                return None, float('inf')
            step2d = np.array([float(stepped[0]), float(stepped[2])], dtype=np.float32)
            d = float(np.linalg.norm(step2d - np.array([float(start_pos[0]), float(start_pos[1])], dtype=np.float32)))
            return [step2d], d
        
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
        try:
            belief = self.robot_belief.astype(np.uint8, copy=False)
            known = (belief > 200) | (belief < 50)
            if hasattr(self, "ground_truth") and self.ground_truth is not None:
                gt = self.ground_truth.astype(np.uint8, copy=False)
                if gt.shape[:2] == belief.shape[:2]:
                    gt_free = (gt > 200)
                    denom = int(np.count_nonzero(gt_free))
                    if denom > 0:
                        num = int(np.count_nonzero(known & gt_free))
                        self.explored_rate = float(num) / float(denom)
                        return
            denom = int(belief.size)
            if denom > 0:
                self.explored_rate = float(np.count_nonzero(known)) / float(denom)
            else:
                self.explored_rate = 0.0
        except Exception:
            self.explored_rate = 0.0

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
            floor_y = float(agent_state.position[1])
            try:
                if self.sim.pathfinder.is_loaded:
                    snap = self.sim.pathfinder.snap_point(np.array(agent_state.position))
                    if np.isfinite(snap[1]):
                        floor_y = float(snap[1])
            except Exception:
                pass
            base_state = SimpleNamespace(position=np.array([float(agent_state.position[0]), floor_y, float(agent_state.position[2])], dtype=np.float32))
            self.robot_belief = self.mapper.update(observations['depth_sensor'], sensor_state, base_state)
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
        self._last_plot_step = step
        if not self.plot:
            return

        try:
            max_frames = int(os.environ.get("HABITAT_PLOT_MAX_FRAMES", "300"))
        except Exception:
            max_frames = 300
        max_frames = int(max(0, min(max_frames, 20000)))
        if max_frames > 0:
            try:
                if int(getattr(self, "_habitat_plot_frames", 0)) >= max_frames:
                    if not bool(getattr(self, "_plot_disabled_maxframes", False)):
                        self._plot_disabled_maxframes = True
                        try:
                            print(f"[HabitatEnv] Plot reached HABITAT_PLOT_MAX_FRAMES={max_frames}, skipping further habitat plots.", flush=True)
                        except Exception:
                            pass
                    return
            except Exception:
                pass

        plt.switch_backend('agg')
        plt.close('all') # Ensure no previous figures interfere
        
        has_rgb = hasattr(self, 'rgb_image') and self.rgb_image is not None
        has_depth = hasattr(self, 'depth_image') and self.depth_image is not None
        has_gt_height = hasattr(self, 'ground_truth_height') and self.ground_truth_height is not None
        if (not has_rgb or not has_depth) and hasattr(self, "sim") and self.sim is not None:
            try:
                observations = self.sim.get_sensor_observations()
                if (not has_rgb) and ('color_sensor' in observations):
                    self.rgb_image = observations['color_sensor'][..., :3]
                    has_rgb = True
                if (not has_depth) and ('depth_sensor' in observations):
                    self.depth_image = observations['depth_sensor']
                    has_depth = True
            except Exception:
                pass
        try:
            plot_sensor_depth = int(os.environ.get("HABITAT_PLOT_SENSOR_DEPTH", "0")) != 0
        except Exception:
            plot_sensor_depth = False
        try:
            plot_gt_height = int(os.environ.get("HABITAT_PLOT_GT_HEIGHT", "1")) != 0
        except Exception:
            plot_gt_height = True

        show_depth = bool(has_depth and plot_sensor_depth)
        show_gt_height = bool(has_gt_height and plot_gt_height)

        ncols = 3 + int(bool(has_rgb)) + int(bool(show_gt_height)) + int(bool(show_depth))
        
        if ncols == 5:
            plt.figure(figsize=(30, 6))
        elif ncols == 4:
            plt.figure(figsize=(24, 6))
        else:
            plt.figure(figsize=(18, 6))
        plt.subplot(1, ncols, 1)

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

        x0 = 0
        y0 = 0
        x1 = int(map_disp.shape[1] - 1)
        y1 = int(map_disp.shape[0] - 1)
        try:
            known = mask_free | mask_occupied
            if np.any(known):
                ys, xs = np.nonzero(known)
                x0 = int(xs.min())
                x1 = int(xs.max())
                y0 = int(ys.min())
                y1 = int(ys.max())
        except Exception:
            pass

        try:
            margin_m = float(os.environ.get("HABITAT_PLOT_MARGIN_M", "6.0"))
        except Exception:
            margin_m = 6.0
        margin_m = float(max(0.0, min(margin_m, 50.0)))
        margin_px = 0
        if hasattr(self, "mapper") and self.mapper is not None:
            try:
                margin_px = int(round(margin_m / float(self.mapper.cell_size)))
            except Exception:
                margin_px = 0

        try:
            if hasattr(self, "mapper") and self.mapper is not None and len(self.trajectory_x) > 0:
                tx = (np.array(self.trajectory_x) - self.belief_origin_x) / self.mapper.cell_size
                ty = (np.array(self.trajectory_y) - self.belief_origin_y) / self.mapper.cell_size
                if tx.size > 0:
                    txi = tx[np.isfinite(tx)].astype(np.int64, copy=False)
                    tyi = ty[np.isfinite(ty)].astype(np.int64, copy=False)
                    if txi.size > 0 and tyi.size > 0:
                        x0 = int(min(x0, int(txi.min())))
                        x1 = int(max(x1, int(txi.max())))
                        y0 = int(min(y0, int(tyi.min())))
                        y1 = int(max(y1, int(tyi.max())))
        except Exception:
            pass

        h0, w0 = map_disp.shape[:2]
        x0 = int(max(0, x0 - margin_px))
        y0 = int(max(0, y0 - margin_px))
        x1 = int(min(w0 - 1, x1 + margin_px))
        y1 = int(min(h0 - 1, y1 + margin_px))

        map_disp_crop = map_disp[y0:y1 + 1, x0:x1 + 1]

        plt.imshow(map_disp_crop, origin='lower')
        plt.title(f'Explored Map (Step {step})')
        plt.axis('off')
        
        # Plot robot
        if hasattr(self, 'mapper'):
            rx = (self.robot_location[0] - self.belief_origin_x) / self.mapper.cell_size
            ry = (self.robot_location[1] - self.belief_origin_y) / self.mapper.cell_size
            plt.plot(rx - x0, ry - y0, 'mo', markersize=6, markeredgecolor='k', zorder=10, label='Robot')
        
        # Plot trajectory
        if len(self.trajectory_x) > 1:
            traj_x = (np.array(self.trajectory_x) - self.belief_origin_x) / self.mapper.cell_size
            traj_y = (np.array(self.trajectory_y) - self.belief_origin_y) / self.mapper.cell_size
            plt.plot(traj_x - x0, traj_y - y0, 'b-', linewidth=1.5, alpha=0.7, zorder=5, label='Trajectory')
        
        # Plot stairs
        if hasattr(self, "stairs_coords_list") and self.stairs_coords_list is not None:
            try:
                plot_all_stairs = int(os.environ.get("HABITAT_PLOT_ALL_STAIRS", "1")) != 0
            except Exception:
                plot_all_stairs = True
            for idx, coords in enumerate(self.stairs_coords_list):
                # Only plot if discovered or currently targeted
                is_discovered = (hasattr(self, 'discovered_stairs') and idx in self.discovered_stairs)
                is_target = (hasattr(self, "current_target_stairs_index") and 
                             self.current_target_stairs_index is not None and 
                             idx == self.current_target_stairs_index)
                if plot_all_stairs or is_discovered or is_target:
                    sx = (coords[0] - self.belief_origin_x) / self.mapper.cell_size
                    sy = (coords[1] - self.belief_origin_y) / self.mapper.cell_size

                    if is_target:
                        plt.plot(sx - x0, sy - y0, 'y*', markersize=12, markeredgecolor='k', zorder=8, label='Target Stair')
                    elif is_discovered:
                        plt.plot(sx - x0, sy - y0, 'r*', markersize=8, markeredgecolor='k', zorder=7, label='Discovered Stair')
                    else:
                        plt.plot(sx - x0, sy - y0, 'c+', markersize=7, markeredgewidth=1.0, zorder=6, label='Stair Candidate')

        gt_map_disp = None
        gt_map_disp_crop = None
        if hasattr(self, "ground_truth") and self.ground_truth is not None:
            try:
                gt_map_disp = np.zeros((*self.ground_truth.shape, 3), dtype=np.uint8)
                gt_mask_free = (self.ground_truth > 200)
                gt_mask_occupied = (self.ground_truth < 50)
                gt_map_disp[gt_mask_free] = [255, 255, 255]
                gt_map_disp[gt_mask_occupied] = [0, 0, 0]
                gt_map_disp[~gt_mask_free & ~gt_mask_occupied] = [127, 127, 127]
                if gt_map_disp.shape[0] == h0 and gt_map_disp.shape[1] == w0:
                    gt_map_disp_crop = gt_map_disp[y0:y1 + 1, x0:x1 + 1]
                else:
                    gt_map_disp_crop = gt_map_disp
            except Exception:
                gt_map_disp = None
                gt_map_disp_crop = None

        err_map_disp_crop = None
        metrics_text = ""
        try:
            if gt_map_disp is not None and gt_map_disp.shape[:2] == self.robot_belief.shape[:2]:
                belief = self.robot_belief.astype(np.uint8)
                gt = self.ground_truth.astype(np.uint8)
                belief_free = belief > 200
                belief_occ = belief < 50
                belief_unknown = (~belief_free) & (~belief_occ)
                gt_free = gt > 200
                gt_occ = gt < 50
                valid = (gt_free | gt_occ)
                known = (belief_free | belief_occ)
                denom = int(np.count_nonzero(known & valid))
                if denom > 0:
                    tp_free = int(np.count_nonzero(belief_free & gt_free))
                    fp_free = int(np.count_nonzero(belief_free & gt_occ))
                    fn_free = int(np.count_nonzero((~belief_free) & gt_free))
                    tn_free = int(np.count_nonzero((~belief_free) & gt_occ))

                    acc = float(tp_free + tn_free) / float(int(np.count_nonzero(valid)))
                    map_acc = float(np.count_nonzero(((belief_free & gt_free) | (belief_occ & gt_occ)) & (known & valid))) / float(denom)
                    false_free = float(fp_free) / float(denom)
                    false_occ = float(np.count_nonzero(belief_occ & gt_free)) / float(denom)
                    free_prec = float(tp_free) / float(max(1, tp_free + fp_free))
                    free_rec = float(tp_free) / float(max(1, tp_free + fn_free))
                    gt_free_total = int(np.count_nonzero(gt_free))
                    gt_free_cov = float(np.count_nonzero(known & gt_free)) / float(max(1, gt_free_total))
                    metrics_text = (
                        f" | KnownAcc: {map_acc:.1%} | FF: {false_free:.1%} | FO: {false_occ:.1%}"
                        f" | FreeP: {free_prec:.1%} | FreeR: {free_rec:.1%} | GTFreeCov: {gt_free_cov:.1%}"
                    )

                err = gt_map_disp.copy()
                err[belief_unknown] = [160, 160, 160]
                err[belief_free & gt_occ] = [255, 0, 0]
                err[belief_occ & gt_free] = [0, 90, 255]
                if err.shape[0] == h0 and err.shape[1] == w0:
                    err_map_disp_crop = err[y0:y1 + 1, x0:x1 + 1]
                else:
                    err_map_disp_crop = err
        except Exception:
            err_map_disp_crop = None
            metrics_text = ""

        plt.subplot(1, ncols, 2)
        if err_map_disp_crop is not None:
            plt.imshow(err_map_disp_crop, origin='lower')
            plt.title('GT + Errors (Red=FF, Blue=FO, Gray=Unknown)')
        else:
            plt.text(0.5, 0.5, 'No Error Map', ha='center', va='center')
        plt.axis('off')

        slot = 3
        if has_rgb:
            plt.subplot(1, ncols, slot)
            plt.imshow(self.rgb_image)
            plt.title('Agent View (RGB)')
            plt.axis('off')
            slot += 1

        if show_gt_height:
            try:
                hm = np.asarray(self.ground_truth_height).astype(np.float32, copy=False)
                if hm.shape[0] == h0 and hm.shape[1] == w0:
                    hm_crop = hm[y0:y1 + 1, x0:x1 + 1]
                else:
                    hm_crop = hm
                finite = np.isfinite(hm_crop)
                vmin = float(np.nanmin(hm_crop[finite])) if np.any(finite) else 0.0
                vmax = float(np.nanmax(hm_crop[finite])) if np.any(finite) else 1.0
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                    vmin, vmax = 0.0, 1.0
            except Exception:
                hm_crop = None
                vmin, vmax = 0.0, 1.0

            plt.subplot(1, ncols, slot)
            if hm_crop is not None:
                gt_free_crop = None
                try:
                    gt0 = np.asarray(self.ground_truth).astype(np.uint8, copy=False)
                    if gt0.shape[0] == h0 and gt0.shape[1] == w0:
                        gt_free_crop = (gt0[y0:y1 + 1, x0:x1 + 1] > 200)
                    else:
                        gt_free_crop = (gt0 > 200)
                    if gt_free_crop.shape != hm_crop.shape:
                        gt_free_crop = None
                except Exception:
                    gt_free_crop = None

                if gt_free_crop is not None:
                    hm_show = np.ma.array(hm_crop, mask=(~gt_free_crop) | (~np.isfinite(hm_crop)))
                    try:
                        import matplotlib.cm as cm
                        base_cmap = cm.get_cmap('viridis')
                        try:
                            cmap = base_cmap.copy()
                        except Exception:
                            try:
                                import copy as _copy
                                cmap = _copy.copy(base_cmap)
                            except Exception:
                                cmap = None
                        try:
                            if cmap is not None:
                                cmap.set_bad(color=(0.0, 0.0, 0.0))
                        except Exception:
                            pass
                    except Exception:
                        cmap = None
                    if cmap is None:
                        cmap = 'viridis'
                    plt.imshow(hm_show, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
                    plt.title('GT Height (NavMesh, Free Only)')
                else:
                    plt.imshow(hm_crop, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
                    plt.title('GT Height (NavMesh)')
            else:
                plt.text(0.5, 0.5, 'No GT Height', ha='center', va='center')
            plt.axis('off')
            slot += 1

        if show_depth:
            try:
                d = np.asarray(self.depth_image)
                if d.ndim == 3:
                    d = d[..., 0]
                d = d.astype(np.float32, copy=False)
                try:
                    max_depth = float(os.environ.get("HABITAT_DEPTH_MAX", str(SENSOR_RANGE)))
                except Exception:
                    max_depth = float(SENSOR_RANGE)
                max_depth = float(max(0.5, min(max_depth, 30.0)))

                finite = np.isfinite(d)
                dmax = float(np.nanmax(d[finite])) if np.any(finite) else 0.0
                is_norm = bool(dmax <= 1.05)
                d_m = (d * max_depth) if is_norm else d
                d_m = np.clip(d_m, 0.0, max_depth)
                d_norm = d_m / max_depth if max_depth > 1e-6 else d_m
            except Exception:
                d_norm = None

            plt.subplot(1, ncols, slot)
            if d_norm is not None:
                plt.imshow(d_norm, cmap='inferno', vmin=0.0, vmax=1.0)
                plt.title('View Depth (Sensor)')
            else:
                plt.text(0.5, 0.5, 'No Depth', ha='center', va='center')
            plt.axis('off')
            slot += 1

        plt.subplot(1, ncols, slot)

        if gt_map_disp_crop is not None:
            plt.imshow(gt_map_disp_crop, origin='lower')
            plt.title('Ground Truth Map (Real)')
            plt.axis('off')
            if hasattr(self, 'mapper'):
                rx = (self.robot_location[0] - self.belief_origin_x) / self.mapper.cell_size
                ry = (self.robot_location[1] - self.belief_origin_y) / self.mapper.cell_size
                plt.plot(rx - x0, ry - y0, 'mo', markersize=6, markeredgecolor='k', zorder=10)
        else:
            plt.text(0.5, 0.5, 'No Ground Truth', ha='center', va='center')
            plt.axis('off')

        plt.suptitle(f'Explored: {self.explored_rate:.1%} | Dist: {self.travel_dist:.1f}m | Step: {step}{metrics_text}', fontsize=14)
        plt.tight_layout()
        
        # Ensure gifs_path exists
        if not os.path.exists(gifs_path):
            os.makedirs(gifs_path)
            
        save_path = '{}/{}_{}_habitat.png'.format(gifs_path, self.episode_index, step)
        plt.savefig(save_path, dpi=100)
        plt.close()
        try:
            self.frame_files.append(save_path)
        except Exception:
            pass
        try:
            self._habitat_plot_frames = int(getattr(self, "_habitat_plot_frames", 0)) + 1
        except Exception:
            self._habitat_plot_frames = 1
