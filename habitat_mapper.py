import numpy as np
import math
import cv2

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
        self.free_count = np.zeros_like(self.global_map, dtype=np.float32)
        self.occ_count = np.zeros_like(self.global_map, dtype=np.float32)

    def reset(self):
        self.global_map.fill(127)
        self.free_count.fill(0.0)
        self.occ_count.fill(0.0)
        return self.global_map
    
    def get_state(self):
        return {
            'global_map': self.global_map.copy(),
            'free_count': self.free_count.copy(),
            'occ_count': self.occ_count.copy(),
            'origin_x': float(self.origin_x),
            'origin_y': float(self.origin_y),
            'cell_size': float(self.cell_size)
        }
    
    def set_state(self, state):
        gm = state.get('global_map', None)
        fc = state.get('free_count', None)
        oc = state.get('occ_count', None)
        if gm is not None and gm.shape == self.global_map.shape:
            self.global_map = gm.copy()
        if fc is not None and fc.shape == self.free_count.shape:
            self.free_count = fc.copy()
        if oc is not None and oc.shape == self.occ_count.shape:
            self.occ_count = oc.copy()
        self.origin_x = float(state.get('origin_x', self.origin_x))
        self.origin_y = float(state.get('origin_y', self.origin_y))

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
            
            # Remove heuristic: Assume depth is in meters (standard Habitat)
            # If normalized, this will result in a tiny map, but avoiding the "white behind wall" artifact
            # which happens when close to walls (max_depth < 1.0m) and heuristic incorrectly scales by 10x.
            depth_scale = 1.0
        
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
                    if depth_slice is not None:
                        pct = 0.5 - (angle / fov_rad) # Try flipping if needed
                        col = int(pct * w_img)
                        col = max(0, min(col, w_img - 1))
                        
                        obs_depth = depth_slice[col] * depth_scale
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
                        px = cx + dx
                        py = cy + dy
                        if 0 <= px < w and 0 <= py < h:
                            self.global_map[py, px] = 255

    def quaternion_to_yaw(self, q):
        # 提取偏航角 (纯 numpy 实现，无 scipy 依赖)
        try:
            # 1. 获取四元数分量
            if hasattr(q, 'x'):
                qx, qy, qz, qw = q.x, q.y, q.z, q.w
            else:
                qx, qy, qz, qw = q[0], q[1], q[2], q[3]
            
            # 2. 计算旋转后的前向向量 (0, 0, -1)
            # 旋转矩阵 R 的第三列是 Z 轴方向。前向是 -Z，所以取负的第三列。
            # R_02 = 2(xz + wy)
            # R_22 = 1 - 2(xx + yy)
            
            # vector_z (in world) = -R_22
            # vector_x (in world) = -R_02
            
            vx = -2 * (qx*qz + qw*qy)
            vz = -(1 - 2 * (qx*qx + qy*qy))
            
            # 3. 计算 Yaw
            return math.atan2(vz, vx)
            
        except Exception:
            return 0.0

    def _clear_footprint(self, agent_base_state):
        """强制清除机器人底座周围的障碍物 (0.3m 半径)"""
        if agent_base_state is None:
            return
            
        rx = agent_base_state.position[0]
        ry = agent_base_state.position[2]
        
        mx = int((rx - self.origin_x) / self.cell_size)
        my = int((ry - self.origin_y) / self.cell_size)
        
        radius_cells = 0 # 0.4m cell size -> 0 cell radius clears only current cell (0.4m box)
        
        h, w = self.global_map.shape
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                cy, cx = my + dy, mx + dx
                if 0 <= cx < w and 0 <= cy < h:
                    self.occ_count[cy, cx] = 0.0
                    self.free_count[cy, cx] = max(float(self.free_count[cy, cx]), 10.0)

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
        # 为了性能，可以进行降采样 (例如每 2 个像素取 1 个)
        # UPDATE: User complains about bad map quality. Using full resolution (1x) to catch thin obstacles.
        downsample = 1
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
        # REMOVED: Heuristic check for normalized depth (max <= 1.0).
        # This causes severe artifacts ("white behind wall") when the agent is close to a wall (< 1.0m),
        # as it incorrectly scales depth by 10x.
        # We assume standard Habitat behavior where depth is in meters.
        
        z_cam = depth_sub * depth_scale
        # Filter near plane noise (0.3m instead of 0.1m) to avoid camera-self-collisions
        valid_mask = (z_cam > 0.3) & (z_cam < 10.0) # 过滤太近或太远的点
        
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
        # 兼容：无 SciPy 时使用纯 numpy 的四元数旋转回退
        def quat_to_rot_matrix(qx, qy, qz, qw):
            norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
            if norm == 0:
                return np.eye(3, dtype=np.float32)
            qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
            xx, yy, zz = qx*qx, qy*qy, qz*qz
            xy, xz, yz = qx*qy, qx*qz, qy*qz
            wx, wy, wz = qw*qx, qw*qy, qw*qz
            return np.array([
                [1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)],
                [    2*(xy + wz),  1 - 2*(xx + zz),      2*(yz - wx)],
                [    2*(xz - wy),     2*(yz + wx),    1 - 2*(xx + yy)]
            ], dtype=np.float32)
        
        # 提取四元数分量（兼容对象或数组）
        if hasattr(sensor_rot, 'x'):
            qx, qy, qz, qw = sensor_rot.x, sensor_rot.y, sensor_rot.z, sensor_rot.w
        else:
            qx, qy, qz, qw = sensor_rot[0], sensor_rot[1], sensor_rot[2], sensor_rot[3]
        
        try:
            from scipy.spatial.transform import Rotation as R
            r_mat = R.from_quat([qx, qy, qz, qw]).as_matrix()
        except Exception:
            r_mat = quat_to_rot_matrix(qx, qy, qz, qw)
        
        world_points = (r_mat @ cam_points.T).T + np.array(sensor_pos)
            
        # 4. 投影到 2D 网格
        # 筛选高度 (Height Filtering) - Relative to Floor
        y_world = world_points[:, 1]
        y_local = y_world - floor_height
        
        # Ground Points (-0.5 ~ 0.2m)
        ground_mask = (y_local < 0.2) & (y_local > -0.5)
        
        # Obstacle Points (0.25m ~ 1.5m) - Reduced max height from 2.0m to 1.5m
        # This prevents ceiling lights or high wall features from cluttering the map.
        obstacle_mask = (y_local >= 0.25) & (y_local < 1.5)

        # DEBUG: Print stats if too few points
        # if np.sum(ground_mask) < 100 and np.sum(obstacle_mask) < 10:
        #    print(f"[Mapper] Low points! FloorH: {floor_height:.2f}, Y_local range: {y_local.min():.2f} ~ {y_local.max():.2f}", flush=True)
        
        # 获取 Map 索引
        def points_to_indices(points):
            px = ((points[:, 0] - self.origin_x) / self.cell_size).astype(int)
            py = ((points[:, 2] - self.origin_y) / self.cell_size).astype(int) # Z is Map Y
            return px, py
            
        h_map, w_map = self.global_map.shape
        
        # --- NEW: Visibility Polygon Clearing ---
        # 使用多边形填充来清除视野内的自由空间
        # 这比简单的点投影更有效，可以清除动态障碍物留下的残影
        
        # 1. 获取所有有效点的 X-Z 平面投影 (相对于传感器)
        # valid_points 已经在前面计算好了，是 Camera 坐标系下的点
        # Camera 坐标系: -Z 是前方, X 是右方, Y 是上方 (Habitat Convention usually -Z forward, Y up, X right)
        # 但我们前面的 rotation matrix 已经处理了坐标系转换
        
        # 让我们回顾一下坐标变换:
        # world_points = R @ cam_points + T
        # 这里的 world_points 是世界坐标系 (Y Up)
        # 我们需要转换到 Map 像素坐标系
        
        # --- SMART CLEARING: Only use rays that hit low objects/floor ---
        # If a ray hits a high wall (e.g. 1.5m), it passes OVER low obstacles (e.g. tables at 0.7m).
        # Using such rays to clear space would erase the tables.
        # UPDATE: 0.5m threshold is too strict. It prevents clearing when looking at walls (blind spot for floor).
        # We need to allow clearing even if we hit a wall, otherwise the map remains grey in front of the robot.
        # Raising to 2.0m to ensure we clear space up to the walls we see.
        clearing_mask = (y_local < 3.0)
        
        if np.sum(clearing_mask) > 0:
            clearing_points = world_points[clearing_mask]
            
            # 计算所有有效点的 Map 像素坐标
            valid_px, valid_py = points_to_indices(clearing_points)
            
            # 还需要计算相机位置的 Map 像素坐标
            cam_px = int((sensor_pos[0] - self.origin_x) / self.cell_size)
            cam_py = int((sensor_pos[2] - self.origin_y) / self.cell_size)
            
            # 构建多边形点集
            # 为了构建一个闭合的多边形，我们需要按角度对点进行排序
            # 计算相对于相机的角度
            rel_x = clearing_points[:, 0] - sensor_pos[0]
            rel_y = clearing_points[:, 2] - sensor_pos[2] # Z is Map Y
            angles = np.arctan2(rel_y, rel_x)
            
            # 对点按角度排序
            sort_idx = np.argsort(angles)
            sorted_px = valid_px[sort_idx]
            sorted_py = valid_py[sort_idx]
            
            # Calculate distances for depth jump check
            dists = np.sqrt(rel_x**2 + rel_y**2)
            sorted_dists = dists[sort_idx]
            
            # Calculate angles for angle jump check
            sorted_angles = angles[sort_idx]
            
            # 降采样多边形点以提高性能 (Reduced from 10 to 1 for maximum accuracy)
            # Increase slightly to 2 to smooth out noise and reduce speckles while maintaining detail
            poly_step = 2
            
            # We need to process in chunks based on depth discontinuities to avoid "veiling"
            # (clearing through walls when looking past corners)
            
            polygons_to_draw = []
            current_poly_pts = []
            
            cam_pt = np.array([[cam_px, cam_py]], dtype=np.int32)
            
            if len(sorted_px) > 0:
                # Start first polygon
                current_poly_pts.append([sorted_px[0], sorted_py[0]])
                last_dist = sorted_dists[0]
                last_angle = sorted_angles[0]
                
                for i in range(poly_step, len(sorted_px), poly_step):
                    curr_dist = sorted_dists[i]
                    curr_angle = sorted_angles[i]
                    
                    # Check depth jump threshold (e.g. 1.5m)
                    # If jump is too large, we shouldn't connect these points directly
                    dist_jump = abs(curr_dist - last_dist)
                    
                    # Check angle jump threshold (e.g. 10 degrees ~ 0.17 rad)
                    angle_diff = abs(curr_angle - last_angle)
                    if angle_diff > np.pi:
                        angle_diff = 2*np.pi - angle_diff
                    
                    if dist_jump > 1.5 or angle_diff > 0.2:
                        # Close current polygon
                        if len(current_poly_pts) > 1:
                            pts = np.array(current_poly_pts, dtype=np.int32)
                            # Polygon: Camera -> P1...Pn -> Camera
                            final_poly = np.vstack((cam_pt, pts, cam_pt))
                            polygons_to_draw.append(final_poly)
                        
                        # Start new polygon
                        current_poly_pts = []
                    
                    current_poly_pts.append([sorted_px[i], sorted_py[i]])
                    last_dist = curr_dist
                    last_angle = curr_angle
                
                # Add the last accumulated polygon
                if len(current_poly_pts) > 1:
                    pts = np.array(current_poly_pts, dtype=np.int32)
                    final_poly = np.vstack((cam_pt, pts, cam_pt))
                    polygons_to_draw.append(final_poly)
            
            # Draw all polygons
            if polygons_to_draw:
                try:
                    for i in range(len(polygons_to_draw)):
                        polygons_to_draw[i][:, 0] = np.clip(polygons_to_draw[i][:, 0], 0, w_map - 1)
                        polygons_to_draw[i][:, 1] = np.clip(polygons_to_draw[i][:, 1], 0, h_map - 1)

                    free_mask = np.zeros_like(self.global_map, dtype=np.uint8)
                    cv2.fillPoly(free_mask, polygons_to_draw, 255)
                    self.free_count[free_mask > 0] += 1.0
                    
                except Exception as e:
                    print(f"[Mapper] fillPoly error: {e}")

        
        # Update Free (Ground) - Optional now, but good for reinforcing details
        if np.any(ground_mask):
            g_pts = world_points[ground_mask]
            gx, gy = points_to_indices(g_pts)
            # Boundary Check
            valid_g = (gx >= 0) & (gx < w_map) & (gy >= 0) & (gy < h_map)
            self.free_count[gy[valid_g], gx[valid_g]] += 0.5
            
        # Update Occupied (Obstacles) - 覆盖 Free
        if np.any(obstacle_mask):
            o_pts = world_points[obstacle_mask]
            ox, oy = points_to_indices(o_pts)
            # Boundary Check
            valid_o = (ox >= 0) & (ox < w_map) & (oy >= 0) & (oy < h_map)
            
            self.occ_count[oy[valid_o], ox[valid_o]] += 2.5

            
        # 5. 强制清除机器人自身位置 (Footprint Clearing)
        self._clear_footprint(agent_base_state)
        
        self.free_count *= 0.99
        self.occ_count *= 0.99
        np.clip(self.free_count, 0.0, 50.0, out=self.free_count)
        np.clip(self.occ_count, 0.0, 50.0, out=self.occ_count)
        diff = self.free_count - self.occ_count
        self.global_map[(diff > 0.8)] = 255
        self.global_map[(diff < -0.8)] = 0
        mid_mask = (diff >= -0.8) & (diff <= 0.8)
        conf = self.free_count + self.occ_count
        self.global_map[mid_mask & (conf < 0.8)] = 127
        
        return self.global_map
