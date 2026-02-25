import numpy as np
import math

class SimpleMapper:
    """
    这是一个简化的映射器，负责将 Habitat 的 3D 观测转换为 2D 占据地图。
    """
    def __init__(self, cell_size, map_size_meters, origin_x=None, origin_y=None):
        self.cell_size = cell_size
        self.map_size_pixels = int(map_size_meters / cell_size)
        
        if origin_x is None:
            self.origin_x = - map_size_meters / 2.0
        else:
            self.origin_x = origin_x
            
        if origin_y is None:
            self.origin_y = - map_size_meters / 2.0
        else:
            self.origin_y = origin_y

        self.map_center = self.map_size_pixels // 2 # Deprecated but kept for compatibility if needed
        
        # 初始化一张空白地图 (127 代表未知)
        self.global_map = np.ones((self.map_size_pixels, self.map_size_pixels), dtype=np.uint8) * 127

    def reset(self):
        self.global_map.fill(127)
        return self.global_map

    def update(self, depth_obs, sensor_state, agent_base_state=None):
        """
        根据深度图和传感器状态更新地图 (3D Point Cloud Projection)
        depth_obs: Habitat 返回的深度图 (H, W)
        sensor_state: 深度相机的状态 (包含 position 和 rotation)
        agent_base_state: 机器人底座的状态 (可选，用于确定地面高度)
        """
        # 1. 获取传感器位姿
        sensor_pos = sensor_state.position # (x, y, z)
        sensor_rot = sensor_state.rotation # Quaternion
        
        # 确定地面高度
        floor_height = 0.0
        if agent_base_state is not None:
            floor_height = agent_base_state.position[1]
        else:
            # 启发式: 假设相机高度约为 1.5m
            floor_height = sensor_pos[1] - 1.5
            
        # 2. 深度图转点云 (Camera Coordinate)
        # 为了性能，可以进行降采样 (例如每 4 个像素取 1 个)
        downsample = 4
        depth_sub = depth_obs[::downsample, ::downsample]
        
        h, w = depth_sub.shape
        fov = 90
        # 重新计算内参 (考虑降采样)
        fx = (w / 2.0) / np.tan(np.deg2rad(fov / 2.0))
        fy = fx 
        cx = w / 2.0
        cy = h / 2.0
        
        # 生成网格坐标
        v_grid, u_grid = np.indices((h, w))
        
        # 过滤无效深度
        depth_scale = 1.0
        if np.max(depth_sub) <= 1.0 and np.max(depth_sub) > 0:
            depth_scale = 10.0
            
        z_cam = depth_sub * depth_scale
        valid_mask = (z_cam > 0.1) & (z_cam < 10.0) # 过滤太近或太远的点
        
        # 仅处理有效点
        z_valid = z_cam[valid_mask]
        u_valid = u_grid[valid_mask]
        v_valid = v_grid[valid_mask]
        
        # 反投影: P_cam = [x, y, z]
        # Habitat Camera: -Z Forward, +Y Up, +X Right
        x_cam = (u_valid - cx) * z_valid / fx
        y_cam = -(v_valid - cy) * z_valid / fy 
        z_cam_vec = -z_valid 
        
        # Stack into (N, 3)
        cam_points = np.stack([x_cam, y_cam, z_cam_vec], axis=1)
        
        # 3. 转换到世界坐标 (World Coordinate)
        # P_world = R * P_cam + T
        
        # 处理旋转
        try:
            from scipy.spatial.transform import Rotation as R
            # 兼容 Magnum Quaternion 对象
            if hasattr(sensor_rot, 'x'):
                quat = [sensor_rot.x, sensor_rot.y, sensor_rot.z, sensor_rot.w]
            else:
                quat = list(sensor_rot)
            
            r_mat = R.from_quat(quat)
            world_points = r_mat.apply(cam_points) + sensor_pos
            
        except ImportError:
            print("Error: scipy not installed. Cannot rotate points.")
            return self.global_map
            
        # 4. 投影到 2D 网格
        # 筛选高度 (Height Filtering) - Relative to Floor
        y_world = world_points[:, 1]
        y_local = y_world - floor_height
        
        # Ground Points (用于标记 Free): 地面附近 (-0.5 ~ 0.2m)
        ground_mask = (y_local < 0.2) & (y_local > -0.5)
        
        # Obstacle Points (用于标记 Occupied): 地面以上 (0.2m ~ 2.0m)
        obstacle_mask = (y_local >= 0.2) & (y_local < 2.0)
        
        # 获取 Map 索引
        def points_to_indices(points):
            px = ((points[:, 0] - self.origin_x) / self.cell_size).astype(int)
            py = ((points[:, 2] - self.origin_y) / self.cell_size).astype(int) # Z is Map Y
            return px, py
            
        h_map, w_map = self.global_map.shape
        
        # Update Free (Ground)
        if np.any(ground_mask):
            g_pts = world_points[ground_mask]
            gx, gy = points_to_indices(g_pts)
            # Boundary Check
            valid_g = (gx >= 0) & (gx < w_map) & (gy >= 0) & (gy < h_map)
            self.global_map[gy[valid_g], gx[valid_g]] = 255 # FREE
            
        # Update Occupied (Obstacles) - 覆盖 Free
        if np.any(obstacle_mask):
            o_pts = world_points[obstacle_mask]
            ox, oy = points_to_indices(o_pts)
            # Boundary Check
            valid_o = (ox >= 0) & (ox < w_map) & (oy >= 0) & (oy < h_map)
            self.global_map[oy[valid_o], ox[valid_o]] = 0 # OCCUPIED
            
        # 标记机器人自身位置为 Free (防止自身被识别为障碍)
        # Use sensor x/z as approximation if base not provided, but base is better
        robot_pos = agent_base_state.position if agent_base_state else sensor_pos
        robot_mx = int((robot_pos[0] - self.origin_x) / self.cell_size)
        robot_my = int((robot_pos[2] - self.origin_y) / self.cell_size)
        if 0 <= robot_mx < w_map and 0 <= robot_my < h_map:
             self.global_map[robot_my, robot_mx] = 255
        
        return self.global_map

    def _simulate_sensor_update(self, cx, cy, yaw, radius_meter, depth_obs=None):
        """简单的扇形更新，结合深度图"""
        radius_px = int(radius_meter / self.cell_size)
        h, w = self.global_map.shape
        
        # Process depth if available
        depth_slice = None
        depth_scale = 1.0
        fov_rad = math.radians(90)
        
        if depth_obs is not None:
            # Take the middle strip of the depth image
            h_img, w_img = depth_obs.shape[:2]
            mid_row = h_img // 2
            depth_slice = depth_obs[mid_row]
            if len(depth_slice.shape) > 1: 
                depth_slice = depth_slice.flatten()
            
            # Simple heuristic to detect normalization
            # Habitat default far plane is often 10.0 or 100.0. 
            # If all values are <= 1.0, it's likely normalized.
            if np.max(depth_slice) <= 1.0:
                depth_scale = 10.0 # Assume 10m max range
        
        for dy in range(-radius_px, radius_px + 1):
            for dx in range(-radius_px, radius_px + 1):
                dist_px = math.sqrt(dx**2 + dy**2)
                if dist_px > radius_px:
                    continue
                if dist_px == 0:
                    continue
                
                dist_m = dist_px * self.cell_size

                # 检查角度是否在 FOV 内
                angle = math.atan2(dy, dx) - yaw
                # 归一化角度
                while angle > math.pi: angle -= 2*math.pi
                while angle < -math.pi: angle += 2*math.pi
                
                if abs(angle) < fov_rad / 2: 
                    # Map angle to depth image column
                    # -45 deg -> col 0, +45 deg -> col W
                    # Note: Habitat camera: -x is left, +x is right? 
                    # Usually image left (col 0) corresponds to +angle (left) or -angle?
                    # Let's assume standard: col 0 is left (-FOV/2? or +FOV/2?)
                    # Usually col 0 is Left. In standard math, Left is Positive angle (CCW).
                    # But in image, Left is often negative x?
                    # Let's try linear mapping.
                    
                    # angle is in [-pi/4, pi/4]
                    # We map [-FOV/2, FOV/2] to [0, W]
                    # But need to be careful with direction. 
                    # Let's assume col 0 = Left = +FOV/2, col W = Right = -FOV/2 (Standard Camera)
                    # or col 0 = Left = -FOV/2?
                    # Visual check: if I turn left, objects move right.
                    
                    if depth_slice is not None:
                        # Simple mapping: 
                        # rel_angle from -FOV/2 to +FOV/2
                        # 0.5 + angle / FOV
                        pct = 0.5 - (angle / fov_rad) # Try flipping if needed
                        col = int(pct * w_img)
                        col = max(0, min(col, w_img - 1))
                        
                        obs_depth = depth_slice[col] * depth_scale
                        
                        # Correct for perspective (Z-depth vs Euclidean)
                        # obs_dist = obs_depth / math.cos(angle)
                        # But simpler: compare projected distance
                        # My dist_m is Euclidean.
                        # Projected dist is dist_m * cos(angle)
                        
                        proj_dist = dist_m * math.cos(angle)
                        
                        if proj_dist < obs_depth - 0.2: # Buffer
                            # Free space
                            px = cx + dx
                            py = cy + dy
                            if 0 <= px < w and 0 <= py < h:
                                self.global_map[py, px] = 255
                        elif proj_dist < obs_depth + 0.5:
                            # Obstacle
                            px = cx + dx
                            py = cy + dy
                            if 0 <= px < w and 0 <= py < h:
                                self.global_map[py, px] = 0
                    else:
                        # Fallback (Ghost Mode)
                        px = cx + dx
                        py = cy + dy
                        if 0 <= px < w and 0 <= py < h:
                            self.global_map[py, px] = 255 

    def quaternion_to_yaw(self, q):
        # 提取偏航角
        try:
            # 1. 获取四元数分量 (兼容 Habitat/Magnum 对象或数组)
            if hasattr(q, 'x'):
                qx, qy, qz, qw = q.x, q.y, q.z, q.w
            else:
                qx, qy, qz, qw = q[0], q[1], q[2], q[3]
            
            # 2. 手动计算旋转后的前向向量 (0, 0, -1)
            # 公式: v' = v + 2 * r x (r x v + w * v)
            # 简化: 只需计算旋转后的 z 和 x 分量
            # 原始向量 v = (0, 0, -1)
            
            # r x v = (y*-1 - z*0, z*0 - x*-1, x*0 - y*0) = (-y, x, 0)
            # r x v + w * v = (-y, x, -w)
            # r x (...) = (y*-w - z*x, z*-y - x*-w, x*x - y*-y)
            #           = (-yw - zx, -zy + xw, x^2 + y^2)
            # v' = (0, 0, -1) + 2 * (...)
            
            # 这种手动推导容易错，还是用标准公式吧:
            # Rotate vector (0, 0, -1) by quaternion
            # x' = 2(xz - wy) ... no
            
            # 使用 scipy 库最稳
            from scipy.spatial.transform import Rotation as R
            r = R.from_quat([qx, qy, qz, qw])
            v_rot = r.apply([0, 0, -1])
            
            # 3. 计算 Yaw (地图坐标系)
            # Map X <-> World X
            # Map Y <-> World Z
            return math.atan2(v_rot[2], v_rot[0])
            
        except Exception:
            # 如果出错 (如 scipy 未安装)，回退到 0
            return 0.0
