import os
import numpy as np
import math
import cv2
from parameter_clean import SENSOR_RANGE

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
        self.occ_low_count = np.zeros_like(self.global_map, dtype=np.float32)
        self.occ_mid_count = np.zeros_like(self.global_map, dtype=np.float32)
        self.occ_high_count = np.zeros_like(self.global_map, dtype=np.float32)
        self.floor_y_map = np.full_like(self.free_count, np.nan, dtype=np.float32)
        self.floor_y_conf = np.zeros_like(self.free_count, dtype=np.float32)
        self.log_odds = np.zeros_like(self.free_count, dtype=np.float32)
        self._prev_stamp_cell = None

    def reset(self):
        self.global_map.fill(127)
        self.free_count.fill(0.0)
        self.occ_count.fill(0.0)
        self.occ_low_count.fill(0.0)
        self.occ_mid_count.fill(0.0)
        self.occ_high_count.fill(0.0)
        self.floor_y_map.fill(np.nan)
        self.floor_y_conf.fill(0.0)
        self.log_odds.fill(0.0)
        self._floor_height = None
        self._prev_stamp_cell = None
        return self.global_map
    
    def get_state(self):
        fh = None
        try:
            fh = float(self._floor_height) if (hasattr(self, "_floor_height") and self._floor_height is not None and np.isfinite(self._floor_height)) else None
        except Exception:
            fh = None
        return {
            'global_map': self.global_map.copy(),
            'free_count': self.free_count.copy(),
            'occ_count': self.occ_count.copy(),
            'occ_low_count': self.occ_low_count.copy(),
            'occ_mid_count': self.occ_mid_count.copy(),
            'occ_high_count': self.occ_high_count.copy(),
            'floor_y_map': self.floor_y_map.copy(),
            'floor_y_conf': self.floor_y_conf.copy(),
            'log_odds': self.log_odds.copy(),
            'floor_height_est': fh,
            'origin_x': float(self.origin_x),
            'origin_y': float(self.origin_y),
            'cell_size': float(self.cell_size)
        }
    
    def set_state(self, state):
        gm = state.get('global_map', None)
        fc = state.get('free_count', None)
        oc = state.get('occ_count', None)
        ol = state.get('occ_low_count', None)
        om = state.get('occ_mid_count', None)
        oh = state.get('occ_high_count', None)
        fy = state.get('floor_y_map', None)
        fyc = state.get('floor_y_conf', None)
        lo = state.get('log_odds', None)
        if gm is not None and gm.shape == self.global_map.shape:
            self.global_map = gm.copy()
        if fc is not None and fc.shape == self.free_count.shape:
            self.free_count = fc.copy()
        if oc is not None and oc.shape == self.occ_count.shape:
            self.occ_count = oc.copy()
        if ol is not None and ol.shape == self.occ_low_count.shape:
            self.occ_low_count = ol.copy()
        if om is not None and om.shape == self.occ_mid_count.shape:
            self.occ_mid_count = om.copy()
        if oh is not None and oh.shape == self.occ_high_count.shape:
            self.occ_high_count = oh.copy()
        if fy is not None and fy.shape == self.floor_y_map.shape:
            self.floor_y_map = fy.copy()
        if fyc is not None and fyc.shape == self.floor_y_conf.shape:
            self.floor_y_conf = fyc.copy()
        if lo is not None and lo.shape == self.log_odds.shape:
            self.log_odds = lo.copy()
        self.origin_x = float(state.get('origin_x', self.origin_x))
        self.origin_y = float(state.get('origin_y', self.origin_y))
        fh = state.get('floor_height_est', None)
        if fh is not None:
            try:
                fh = float(fh)
            except Exception:
                fh = None
        if fh is not None and np.isfinite(fh):
            self._floor_height = float(fh)

    def get_vlm_rgb_map(self):
        base = self.global_map.astype(np.uint8)
        rgb = np.stack([base, base, base], axis=-1)
        counts = np.stack([self.occ_low_count, self.occ_mid_count, self.occ_high_count], axis=-1)
        idx = np.argmax(counts, axis=-1)
        maxc = np.max(counts, axis=-1)
        occ_mask = (base < 50) & (maxc > 1.0)
        if np.any(occ_mask):
            low = occ_mask & (idx == 0)
            mid = occ_mask & (idx == 1)
            high = occ_mask & (idx == 2)
            rgb[low] = np.array([220, 20, 60], dtype=np.uint8)
            rgb[mid] = np.array([255, 165, 0], dtype=np.uint8)
            rgb[high] = np.array([138, 43, 226], dtype=np.uint8)
        return rgb

    def _simulate_sensor_update(self, cx, cy, yaw, radius_meter, depth_obs=None):
        """简单的扇形更新，结合深度图"""
        h_map, w_map = self.global_map.shape

        try:
            fov_rad = math.radians(float(os.environ.get("HABITAT_SENSOR_HFOV", "90")))
        except Exception:
            fov_rad = math.radians(90)

        if depth_obs is None:
            return

        depth_obs = depth_obs.astype(np.float32, copy=False)
        h_img, w_img = depth_obs.shape[:2]
        mid_row = h_img // 2
        try:
            row_band_ratio = float(os.environ.get("HABITAT_2D_ROW_BAND_RATIO", "0.08"))
        except Exception:
            row_band_ratio = 0.08
        row_band_ratio = float(max(0.0, min(row_band_ratio, 0.49)))
        row_band = int(max(0, round(float(h_img) * row_band_ratio)))
        r0 = int(max(0, mid_row - row_band))
        r1 = int(min(h_img - 1, mid_row + row_band))
        band = depth_obs[r0:r1 + 1]
        if band.ndim != 2 or band.shape[1] != w_img:
            return

        max_depth = float(radius_meter)
        max_depth = float(max(2.0, min(max_depth, 30.0)))

        is_normalized_depth = False
        try:
            p99 = float(np.nanpercentile(band, 99))
            maxv = float(np.nanmax(band))
            near_one_ratio = float(np.mean((band >= 0.999) & np.isfinite(band)))
            is_normalized_depth = np.isfinite(p99) and (0.80 <= p99 <= 1.05) and np.isfinite(maxv) and (maxv <= 1.05) and (near_one_ratio >= 0.01)
        except Exception:
            is_normalized_depth = False

        try:
            stride = int(os.environ.get("HABITAT_2D_RAY_STRIDE", "1"))
        except Exception:
            stride = 1
        stride = int(max(1, min(16, stride)))

        try:
            ray_free_add = float(os.environ.get("HABITAT_2D_RAY_FREE_ADD", "0.30"))
        except Exception:
            ray_free_add = 0.20
        ray_free_add = float(max(0.0, min(ray_free_add, 5.0)))

        try:
            occ_add = float(os.environ.get("HABITAT_2D_OCC_ADD", "1.0"))
        except Exception:
            occ_add = 1.0
        occ_add = float(max(0.0, min(occ_add, 10.0)))

        try:
            hit_margin = float(os.environ.get("HABITAT_DEPTH_HIT_MARGIN", "0.05"))
        except Exception:
            hit_margin = 0.05
        hit_margin = float(max(0.0, min(hit_margin, 1.0)))

        try:
            nohit_clear_ratio = float(os.environ.get("HABITAT_2D_NOHIT_CLEAR_RATIO", "0.0"))
        except Exception:
            nohit_clear_ratio = 0.0
        nohit_clear_ratio = float(max(0.0, min(nohit_clear_ratio, 1.0)))

        try:
            min_hit_depth = float(os.environ.get("HABITAT_2D_MIN_HIT_DEPTH", "0.40"))
        except Exception:
            min_hit_depth = 0.40
        min_hit_depth = float(max(0.0, min(min_hit_depth, max_depth)))

        def bresenham(x0, y0, x1, y1):
            x0 = int(x0); y0 = int(y0); x1 = int(x1); y1 = int(y1)
            dx = abs(x1 - x0)
            dy = -abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx + dy
            x = x0
            y = y0
            out = []
            while True:
                out.append((x, y))
                if x == x1 and y == y1:
                    break
                e2 = 2 * err
                if e2 >= dy:
                    err += dy
                    x += sx
                if e2 <= dx:
                    err += dx
                    y += sy
            return out

        if w_img <= 1:
            return

        if is_normalized_depth:
            valid = np.isfinite(band) & (band > 1e-6) & (band < 0.999)
            depth_per_col = np.full((w_img,), 1.0, dtype=np.float32)
            if np.any(valid):
                try:
                    col_q = float(os.environ.get("HABITAT_2D_COL_QUANTILE", "35"))
                except Exception:
                    col_q = 35.0
                col_q = float(max(0.0, min(col_q, 100.0)))
                masked = band.copy()
                masked[~valid] = np.nan
                finite_any = np.any(np.isfinite(masked), axis=0)
                if np.any(finite_any):
                    depth_per_col[finite_any] = np.nanpercentile(masked[:, finite_any], col_q, axis=0).astype(np.float32, copy=False)
                depth_per_col[~np.isfinite(depth_per_col)] = 1.0
        else:
            valid = np.isfinite(band) & (band > 1e-6) & (band < (max_depth - hit_margin))
            depth_per_col = np.full((w_img,), float(max_depth), dtype=np.float32)
            if np.any(valid):
                try:
                    col_q = float(os.environ.get("HABITAT_2D_COL_QUANTILE", "35"))
                except Exception:
                    col_q = 35.0
                col_q = float(max(0.0, min(col_q, 100.0)))
                masked = band.copy()
                masked[~valid] = np.nan
                finite_any = np.any(np.isfinite(masked), axis=0)
                if np.any(finite_any):
                    depth_per_col[finite_any] = np.nanpercentile(masked[:, finite_any], col_q, axis=0).astype(np.float32, copy=False)
                depth_per_col[~np.isfinite(depth_per_col)] = float(max_depth)

        try:
            med_k = int(os.environ.get("HABITAT_2D_DEPTH_MEDIAN_K", "5"))
        except Exception:
            med_k = 5
        med_k = int(max(0, min(med_k, 21)))
        if med_k > 1:
            if (med_k % 2) == 0:
                med_k += 1
            pad = med_k // 2
            try:
                pad_val = 1.0 if is_normalized_depth else float(max_depth)
            except Exception:
                pad_val = 1.0
            padded = np.pad(depth_per_col.astype(np.float32, copy=False), (pad, pad), mode="constant", constant_values=float(pad_val))
            smoothed = np.empty_like(depth_per_col, dtype=np.float32)
            for i in range(int(w_img)):
                smoothed[i] = float(np.median(padded[i : i + med_k]))
            depth_per_col = smoothed

        try:
            min_cluster = int(os.environ.get("HABITAT_2D_HIT_MIN_CLUSTER", "2"))
        except Exception:
            min_cluster = 2
        min_cluster = int(max(1, min(min_cluster, 10)))
        if min_cluster > 1:
            if is_normalized_depth:
                hit_cols = np.isfinite(depth_per_col) & (depth_per_col < 0.999)
                nohit_val = 1.0
            else:
                hit_cols = np.isfinite(depth_per_col) & (depth_per_col < (float(max_depth) - float(hit_margin)))
                nohit_val = float(max_depth)
            i = 0
            while i < int(w_img):
                if not bool(hit_cols[i]):
                    i += 1
                    continue
                j = i + 1
                while j < int(w_img) and bool(hit_cols[j]):
                    j += 1
                if (j - i) < int(min_cluster):
                    depth_per_col[i:j] = float(nohit_val)
                i = j

        try:
            method = os.environ.get("HABITAT_2D_METHOD", "ray").strip().lower()
        except Exception:
            method = "ray"

        if method in {"circle", "disk", "radius"}:
            try:
                circle_radius_ratio = float(os.environ.get("HABITAT_2D_CIRCLE_RADIUS_RATIO", "0.85"))
            except Exception:
                circle_radius_ratio = 0.85
            circle_radius_ratio = float(max(0.1, min(circle_radius_ratio, 1.0)))

            try:
                circle_free_add = float(os.environ.get("HABITAT_2D_CIRCLE_FREE_ADD", "0.35"))
            except Exception:
                circle_free_add = 0.35
            circle_free_add = float(max(0.0, min(circle_free_add, 5.0)))

            try:
                clear_skip_strong_occ = int(os.environ.get("HABITAT_CLEAR_SKIP_STRONG_OCC", "1")) != 0
            except Exception:
                clear_skip_strong_occ = True
            try:
                strong_occ_margin = float(os.environ.get("HABITAT_STRONG_OCC_MARGIN", "4.0"))
            except Exception:
                strong_occ_margin = 4.0
            strong_occ_margin = float(max(0.0, min(strong_occ_margin, 50.0)))
            try:
                circle_mode = os.environ.get("HABITAT_2D_CIRCLE_MODE", "classic").strip().lower()
            except Exception:
                circle_mode = "classic"

            radius_px = int(max(1, round((max_depth * circle_radius_ratio) / float(self.cell_size))))
            x0 = int(max(0, cx - radius_px))
            x1 = int(min(w_map - 1, cx + radius_px))
            y0 = int(max(0, cy - radius_px))
            y1 = int(min(h_map - 1, cy + radius_px))

            xs = np.arange(x0, x1 + 1, dtype=np.int32)
            ys = np.arange(y0, y1 + 1, dtype=np.int32)
            xx, yy = np.meshgrid(xs, ys)
            dx = xx - int(cx)
            dy = yy - int(cy)
            disk = (dx.astype(np.int64) * dx.astype(np.int64) + dy.astype(np.int64) * dy.astype(np.int64)) <= int(radius_px) * int(radius_px)
            if np.any(disk):
                if clear_skip_strong_occ:
                    occ_strong = (self.occ_count[y0:y1 + 1, x0:x1 + 1] - self.free_count[y0:y1 + 1, x0:x1 + 1]) > strong_occ_margin
                    disk = disk & (~occ_strong)
                self.free_count[y0:y1 + 1, x0:x1 + 1][disk] += circle_free_add

            if circle_mode in {"hybrid", "endpoint", "edge"}:
                for col in range(0, int(w_img), stride):
                    dv = float(depth_per_col[col])
                    if not np.isfinite(dv) or dv <= 1e-6:
                        continue
                    hit = True
                    if is_normalized_depth:
                        if dv >= 0.999:
                            hit = False
                        else:
                            dist_m = float(dv) * max_depth
                    else:
                        if dv >= (max_depth - hit_margin):
                            hit = False
                        else:
                            dist_m = float(min(dv, max_depth))
                    if not hit:
                        continue
                    # 经典方圆时代的端点定位：按图像列线性映射到半圆，不使用 yaw 与 FOV
                    phi = (0.5 - (float(col) + 0.5) / float(w_img)) * math.pi  # [-pi/2, pi/2]
                    dir_x = float(math.cos(phi))
                    dir_y = float(math.sin(phi))
                    steps = int(max(1, round(dist_m / self.cell_size)))
                    ex = int(round(float(cx) + dir_x * float(steps)))
                    ey = int(round(float(cy) + dir_y * float(steps)))
                    ex = int(max(0, min(ex, w_map - 1)))
                    ey = int(max(0, min(ey, h_map - 1)))
                    self.occ_count[ey, ex] += occ_add
            return

        for col in range(0, int(w_img), stride):
            dv = float(depth_per_col[col])
            if not np.isfinite(dv) or dv <= 1e-6:
                continue

            hit = True
            if is_normalized_depth:
                if dv >= 0.999:
                    hit = False
                    dist_m = max_depth * nohit_clear_ratio
                else:
                    dist_m = float(dv) * max_depth
            else:
                if dv >= (max_depth - hit_margin):
                    hit = False
                    dist_m = max_depth * nohit_clear_ratio
                else:
                    dist_m = float(min(dv, max_depth))

            if not hit and nohit_clear_ratio <= 0.0:
                continue

            if hit and dist_m < min_hit_depth:
                continue

            theta = float(yaw) + (0.5 - (float(col) + 0.5) / float(w_img)) * float(fov_rad)
            dir_x = float(math.cos(theta))
            dir_y = float(math.sin(theta))

            steps = int(max(1, round(dist_m / self.cell_size)))
            ex = int(round(float(cx) + dir_x * float(steps)))
            ey = int(round(float(cy) + dir_y * float(steps)))
            ex = int(max(0, min(ex, w_map - 1)))
            ey = int(max(0, min(ey, h_map - 1)))

            line = bresenham(cx, cy, ex, ey)
            if len(line) <= 1:
                continue

            try:
                clear_skip_strong_occ = int(os.environ.get("HABITAT_CLEAR_SKIP_STRONG_OCC", "1")) != 0
            except Exception:
                clear_skip_strong_occ = True
            try:
                strong_occ_margin = float(os.environ.get("HABITAT_STRONG_OCC_MARGIN", "4.0"))
            except Exception:
                strong_occ_margin = 4.0
            strong_occ_margin = float(max(0.0, min(strong_occ_margin, 50.0)))

            blocked = False
            for lx, ly in line[:-1]:
                if 0 <= lx < w_map and 0 <= ly < h_map:
                    if clear_skip_strong_occ:
                        if float(self.occ_count[ly, lx]) - float(self.free_count[ly, lx]) > strong_occ_margin:
                            blocked = True
                            break
                    self.free_count[ly, lx] += ray_free_add

            if hit and (not blocked):
                self.occ_count[ey, ex] += occ_add

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
        
        radius_cells = 1
        
        h, w = self.global_map.shape
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                cy, cx = my + dy, mx + dx
                if 0 <= cx < w and 0 <= cy < h:
                    self.occ_count[cy, cx] = 0.0
                    try:
                        foot_free = float(os.environ.get("HABITAT_FOOTPRINT_FREE", "3.0"))
                    except Exception:
                        foot_free = 3.0
                    self.free_count[cy, cx] = max(float(self.free_count[cy, cx]), foot_free)

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
            fh = float(agent_base_state.position[1])
            if not np.isfinite(fh):
                fh = sensor_pos[1] - 1.5
            if not hasattr(self, "_floor_height"):
                self._floor_height = None
            if self._floor_height is None or not np.isfinite(self._floor_height):
                self._floor_height = fh
            else:
                self._floor_height = 0.9 * float(self._floor_height) + 0.1 * fh
            floor_height = float(self._floor_height)
        else:
            # 启发式: 假设相机高度约为 1.5m
            floor_height = sensor_pos[1] - 1.5

        try:
            depth_filter = os.environ.get("HABITAT_DEPTH_FILTER", "median3").strip().lower()
        except Exception:
            depth_filter = "median3"
        if depth_filter not in {"", "none", "off", "0", "false"}:
            try:
                k = 3
                if depth_filter.startswith("median"):
                    s = depth_filter.replace("median", "").strip()
                    if s:
                        k = int(float(s))
                k = int(max(1, min(k, 9)))
                if k % 2 == 0:
                    k += 1
                d = np.asarray(depth_obs, dtype=np.float32)
                if d.ndim == 3 and int(d.shape[-1]) == 1:
                    d = d[..., 0]
                if d.ndim == 2 and k > 1:
                    if not np.all(np.isfinite(d)):
                        d = d.copy()
                        d[~np.isfinite(d)] = 0.0
                    depth_obs = cv2.medianBlur(d, k)
            except Exception:
                pass

        mode = os.environ.get("HABITAT_MAPPER_MODE", "3d").strip().lower()
        if mode in {"2d", "fan", "slice"}:
            cx_map = int((sensor_pos[0] - self.origin_x) / self.cell_size)
            cy_map = int((sensor_pos[2] - self.origin_y) / self.cell_size)

            if hasattr(sensor_rot, 'x'):
                qx, qy, qz, qw = sensor_rot.x, sensor_rot.y, sensor_rot.z, sensor_rot.w
            else:
                qx, qy, qz, qw = sensor_rot[0], sensor_rot[1], sensor_rot[2], sensor_rot[3]

            norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
            if norm > 0:
                qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
            xx, yy, zz = qx*qx, qy*qy, qz*qz
            xy, xz, yz = qx*qy, qx*qz, qy*qz
            wx, wy, wz = qw*qx, qw*qy, qw*qz
            r_mat = np.array([
                [1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)],
                [    2*(xy + wz),  1 - 2*(xx + zz),      2*(yz - wx)],
                [    2*(xz - wy),     2*(yz + wx),    1 - 2*(xx + yy)]
            ], dtype=np.float32)
            fwd = r_mat @ np.array([0.0, 0.0, -1.0], dtype=np.float32)
            yaw = float(math.atan2(float(fwd[2]), float(fwd[0])))

            try:
                radius = float(os.environ.get("HABITAT_DEPTH_MAX", str(SENSOR_RANGE)))
            except Exception:
                radius = float(SENSOR_RANGE)
            radius = float(max(2.0, min(radius, 30.0)))

            self._simulate_sensor_update(cx_map, cy_map, yaw, radius, depth_obs)

            self._clear_footprint(agent_base_state)

            try:
                decay = float(os.environ.get("HABITAT_COUNT_DECAY", "0.999"))
            except Exception:
                decay = 0.999
            decay = float(max(0.95, min(decay, 1.0)))
            self.free_count *= decay
            self.occ_count *= decay
            self.occ_low_count *= decay
            self.occ_mid_count *= decay
            self.occ_high_count *= decay
            np.clip(self.free_count, 0.0, 50.0, out=self.free_count)
            np.clip(self.occ_count, 0.0, 50.0, out=self.occ_count)
            np.clip(self.occ_low_count, 0.0, 50.0, out=self.occ_low_count)
            np.clip(self.occ_mid_count, 0.0, 50.0, out=self.occ_mid_count)
            np.clip(self.occ_high_count, 0.0, 50.0, out=self.occ_high_count)

            diff = self.free_count - self.occ_count
            try:
                free_thresh = float(os.environ.get("HABITAT_MAP_FREE_THRESH", "1.0"))
            except Exception:
                free_thresh = 1.0
            try:
                occ_thresh = float(os.environ.get("HABITAT_MAP_OCC_THRESH", "2.0"))
            except Exception:
                occ_thresh = 2.0
            conf = self.free_count + self.occ_count
            try:
                conf_thresh = float(os.environ.get("HABITAT_MAP_CONF_THRESH_2D", os.environ.get("HABITAT_MAP_CONF_THRESH", "3.0")))
            except Exception:
                conf_thresh = 3.0
            conf_thresh = float(max(0.0, min(conf_thresh, 50.0)))

            self.global_map[(diff > free_thresh)] = 255
            self.global_map[(diff < -occ_thresh)] = 0
            mid_mask = (diff >= -occ_thresh) & (diff <= free_thresh)
            self.global_map[mid_mask | (conf < conf_thresh)] = 127

            try:
                free_erode_iters = int(os.environ.get("HABITAT_FREE_ERODE_ITERS_2D", "0"))
            except Exception:
                free_erode_iters = 0
            free_erode_iters = int(max(0, min(free_erode_iters, 3)))
            if free_erode_iters > 0:
                free_bin = (self.global_map == 255).astype(np.uint8)
                kernel = np.ones((3, 3), dtype=np.uint8)
                free_eroded = cv2.erode(free_bin, kernel, iterations=free_erode_iters)
                lost = (free_bin > 0) & (free_eroded == 0)
                if np.any(lost):
                    self.global_map[lost] = 127

            occ_bin = (self.global_map == 0).astype(np.uint8)
            if np.any(occ_bin):
                try:
                    cc_min_area = int(os.environ.get("HABITAT_OCC_CC_MIN_AREA_2D", "9"))
                except Exception:
                    cc_min_area = 9
                cc_min_area = int(max(0, min(cc_min_area, 10_000_000)))
                if cc_min_area > 1:
                    num, labels, stats, _ = cv2.connectedComponentsWithStats(occ_bin, connectivity=8)
                    if int(num) > 1:
                        areas = stats[1:, cv2.CC_STAT_AREA]
                        for i in range(1, int(num)):
                            if int(areas[i - 1]) < cc_min_area:
                                self.global_map[labels == i] = 127
                else:
                    kernel = np.ones((3, 3), dtype=np.uint8)
                    neigh = cv2.filter2D(occ_bin, -1, kernel, borderType=cv2.BORDER_CONSTANT)
                    remove = (occ_bin > 0) & (neigh < 2)
                    if np.any(remove):
                        self.global_map[remove] = 127

            return self.global_map
            
        # 2. 深度图转点云 (Camera Coordinate)
        try:
            downsample = int(os.environ.get("HABITAT_DEPTH_DOWNSAMPLE", "2"))
        except Exception:
            downsample = 2
        downsample = int(max(1, min(8, downsample)))
        depth_sub = depth_obs[::downsample, ::downsample].astype(np.float32, copy=False)
        
        h, w = depth_sub.shape
        try:
            hfov = float(os.environ.get("HABITAT_SENSOR_HFOV", "90"))
        except Exception:
            hfov = 90.0
        fx = (w / 2.0) / np.tan(np.deg2rad(hfov / 2.0))
        vfov = 2.0 * np.arctan((h / max(1.0, float(w))) * np.tan(np.deg2rad(hfov / 2.0)))
        fy = (h / 2.0) / np.tan(vfov / 2.0)
        cx = (w - 1) / 2.0
        cy = (h - 1) / 2.0
        
        # 生成网格坐标
        v_grid, u_grid = np.indices((h, w))
        
        try:
            max_depth = float(os.environ.get("HABITAT_DEPTH_MAX", str(SENSOR_RANGE)))
        except Exception:
            max_depth = float(SENSOR_RANGE)
        max_depth = float(max(2.0, min(max_depth, 30.0)))

        is_normalized_depth = False
        try:
            p99 = float(np.nanpercentile(depth_sub, 99))
            maxv = float(np.nanmax(depth_sub))
            near_one_ratio = float(np.mean((depth_sub >= 0.999) & np.isfinite(depth_sub)))
            is_normalized_depth = np.isfinite(p99) and (0.80 <= p99 <= 1.05) and np.isfinite(maxv) and (maxv <= 1.05) and (near_one_ratio >= 0.01)
        except Exception:
            is_normalized_depth = False

        if is_normalized_depth:
            z_norm = depth_sub
            z_cam = z_norm * max_depth
            valid_mask = np.isfinite(z_norm) & (z_norm > 1e-3) & (z_norm < 0.999)
            nohit_depth_mask = (np.isfinite(z_norm) & (z_norm >= 0.999)) | (~np.isfinite(z_norm)) | (z_norm <= 1e-3)
        else:
            z_cam = depth_sub
            valid_mask = np.isfinite(z_cam) & (z_cam > 0.3) & (z_cam <= max_depth)
            nohit_depth_mask = (np.isfinite(z_cam) & (z_cam <= 1e-3)) | (~np.isfinite(z_cam))
        try:
            nohit_zero_ratio = float(os.environ.get("HABITAT_NOHIT_ZERO_RATIO", "0.6"))
        except Exception:
            nohit_zero_ratio = 0.6
        nohit_zero_ratio = float(max(0.0, min(nohit_zero_ratio, 1.0)))
        try:
            min_valid_ratio = float(os.environ.get("HABITAT_MIN_VALID_RATIO", "0.02"))
        except Exception:
            min_valid_ratio = 0.02
        min_valid_ratio = float(max(0.0, min(min_valid_ratio, 1.0)))
        try:
            nohit_pixel_stride = int(os.environ.get("HABITAT_NOHIT_PIXEL_STRIDE", "8"))
        except Exception:
            nohit_pixel_stride = 8
        nohit_pixel_stride = int(max(1, min(32, nohit_pixel_stride)))
        try:
            nohit_row_band = float(os.environ.get("HABITAT_NOHIT_ROW_BAND", "0.12"))
        except Exception:
            nohit_row_band = 0.12
        nohit_row_band = float(max(0.0, min(nohit_row_band, 0.49)))
        
        # 仅处理有效点
        z_valid = z_cam[valid_mask]
        u_valid = u_grid[valid_mask]
        v_valid = v_grid[valid_mask]

        try:
            hit_margin = float(os.environ.get("HABITAT_DEPTH_HIT_MARGIN", "0.05"))
        except Exception:
            hit_margin = 0.05
        hit_margin = float(max(0.0, min(hit_margin, 1.0)))
        hit_mask = z_valid < (max_depth - hit_margin)
        if not is_normalized_depth:
            try:
                nohit_depth_mask = nohit_depth_mask | (np.isfinite(z_cam) & (z_cam >= (max_depth - hit_margin)))
            except Exception:
                pass
        
        try:
            obs_row_max_ratio = float(os.environ.get("HABITAT_OBS_ROW_MAX_RATIO", "0.75"))
        except Exception:
            obs_row_max_ratio = 0.75
        obs_row_max_ratio = float(max(0.1, min(obs_row_max_ratio, 1.0)))
        try:
            obs_row_min_ratio = float(os.environ.get("HABITAT_OBS_ROW_MIN_RATIO", "0.05"))
        except Exception:
            obs_row_min_ratio = 0.05
        obs_row_min_ratio = float(max(0.0, min(obs_row_min_ratio, float(obs_row_max_ratio) - 0.05)))
        vv = v_valid.astype(np.float32)
        obs_row_mask = (vv >= (float(h - 1) * float(obs_row_min_ratio))) & (vv <= (float(h - 1) * float(obs_row_max_ratio)))

        try:
            clear_row_max_ratio = float(os.environ.get("HABITAT_CLEAR_ROW_MAX_RATIO", "0.65"))
        except Exception:
            clear_row_max_ratio = 0.65
        clear_row_max_ratio = float(max(0.1, min(clear_row_max_ratio, 1.0)))
        clear_row_mask = v_valid.astype(np.float32) >= (float(h - 1) * (1.0 - clear_row_max_ratio))

        try:
            clear_max_depth = float(os.environ.get("HABITAT_CLEAR_MAX_DEPTH", str(min(float(max_depth), 8.0))))
        except Exception:
            clear_max_depth = float(min(float(max_depth), 8.0))
        clear_max_depth = float(max(0.5, min(clear_max_depth, float(max_depth))))
        try:
            clear_far_ratio = float(os.environ.get("HABITAT_CLEAR_FAR_RATIO", "0.92"))
        except Exception:
            clear_far_ratio = 0.92
        clear_far_ratio = float(max(0.1, min(clear_far_ratio, 1.0)))
        clear_depth_cap = float(min(float(clear_max_depth), float(max_depth) * float(clear_far_ratio)))

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
        try:
            floor_est_enable = int(os.environ.get("HABITAT_FLOOR_EST_ENABLE", "1")) != 0
        except Exception:
            floor_est_enable = True
        if floor_est_enable:
            try:
                floor_est_vmin = float(os.environ.get("HABITAT_FLOOR_EST_VMIN_RATIO", "0.78"))
            except Exception:
                floor_est_vmin = 0.78
            floor_est_vmin = float(max(0.0, min(floor_est_vmin, 0.99)))
            try:
                floor_est_zmax = float(os.environ.get("HABITAT_FLOOR_EST_ZMAX_RATIO", "0.65"))
            except Exception:
                floor_est_zmax = 0.65
            floor_est_zmax = float(max(0.1, min(floor_est_zmax, 1.0)))
            try:
                floor_est_beta = float(os.environ.get("HABITAT_FLOOR_EST_BETA", "0.3"))
            except Exception:
                floor_est_beta = 0.3
            floor_est_beta = float(max(0.0, min(floor_est_beta, 1.0)))
            try:
                floor_est_blend = float(os.environ.get("HABITAT_FLOOR_EST_BLEND", "0.5"))
            except Exception:
                floor_est_blend = 0.5
            floor_est_blend = float(max(0.0, min(floor_est_blend, 1.0)))
            try:
                floor_est_q = float(os.environ.get("HABITAT_FLOOR_EST_Q", "20"))
            except Exception:
                floor_est_q = 20.0
            floor_est_q = float(max(0.0, min(floor_est_q, 50.0)))
            try:
                floor_est_max_delta = float(os.environ.get("HABITAT_FLOOR_EST_MAX_DELTA", "1.0"))
            except Exception:
                floor_est_max_delta = 1.0
            floor_est_max_delta = float(max(0.1, min(floor_est_max_delta, 5.0)))
            try:
                cand_mask = (v_valid.astype(np.float32) >= float(h - 1) * floor_est_vmin) & hit_mask & (z_valid < (max_depth * floor_est_zmax))
            except Exception:
                cand_mask = None
            if cand_mask is not None and np.any(cand_mask):
                y_cand = world_points[cand_mask, 1]
                if getattr(y_cand, "size", 0) >= 50:
                    try:
                        y_est = float(np.nanpercentile(y_cand, floor_est_q))
                    except Exception:
                        try:
                            y_est = float(np.nanmedian(y_cand))
                        except Exception:
                            y_est = float("nan")
                    if np.isfinite(y_est) and (abs(float(y_est) - float(floor_height)) <= float(floor_est_max_delta)):
                        if not hasattr(self, "_floor_height"):
                            self._floor_height = None
                        if self._floor_height is None or not np.isfinite(self._floor_height):
                            self._floor_height = float(floor_height)
                        target = (1.0 - floor_est_blend) * float(floor_height) + floor_est_blend * float(y_est)
                        self._floor_height = (1.0 - floor_est_beta) * float(self._floor_height) + floor_est_beta * float(target)
                        floor_height = float(self._floor_height)

        y_world = world_points[:, 1]
        try:
            floor_map_enable = int(os.environ.get("HABITAT_FLOOR_MAP_ENABLE", "1")) != 0
        except Exception:
            floor_map_enable = True
        floor_map_conf_min = 0.0

        if floor_map_enable:
            try:
                floor_map_vmin = float(os.environ.get("HABITAT_FLOOR_MAP_VMIN_RATIO", "0.78"))
            except Exception:
                floor_map_vmin = 0.78
            floor_map_vmin = float(max(0.0, min(floor_map_vmin, 0.99)))
            try:
                floor_map_zmax = float(os.environ.get("HABITAT_FLOOR_MAP_ZMAX_RATIO", "0.8"))
            except Exception:
                floor_map_zmax = 0.8
            floor_map_zmax = float(max(0.1, min(floor_map_zmax, 1.0)))
            try:
                floor_map_beta = float(os.environ.get("HABITAT_FLOOR_MAP_BETA", "0.35"))
            except Exception:
                floor_map_beta = 0.35
            floor_map_beta = float(max(0.0, min(floor_map_beta, 1.0)))
            try:
                floor_map_rel_max = float(os.environ.get("HABITAT_FLOOR_MAP_REL_MAX", "0.9"))
            except Exception:
                floor_map_rel_max = 0.9
            floor_map_rel_max = float(max(0.1, min(floor_map_rel_max, 3.0)))
            try:
                floor_map_conf_inc = float(os.environ.get("HABITAT_FLOOR_MAP_CONF_INC", "2.0"))
            except Exception:
                floor_map_conf_inc = 2.0
            floor_map_conf_inc = float(max(0.0, min(floor_map_conf_inc, 10.0)))
            try:
                floor_map_conf_cap = float(os.environ.get("HABITAT_FLOOR_MAP_CONF_CAP", "50.0"))
            except Exception:
                floor_map_conf_cap = 50.0
            floor_map_conf_cap = float(max(1.0, min(floor_map_conf_cap, 200.0)))
            try:
                floor_map_conf_min = float(os.environ.get("HABITAT_FLOOR_MAP_CONF_MIN", "1.0"))
            except Exception:
                floor_map_conf_min = 1.0
            floor_map_conf_min = float(max(0.0, min(floor_map_conf_min, 200.0)))

            px_all = ((world_points[:, 0] - self.origin_x) / self.cell_size).astype(np.int32, copy=False)
            py_all = ((world_points[:, 2] - self.origin_y) / self.cell_size).astype(np.int32, copy=False)
            h_map, w_map = self.global_map.shape
            inb_all = (px_all >= 0) & (px_all < w_map) & (py_all >= 0) & (py_all < h_map)

            if np.any(inb_all):
                rel0 = y_world - float(floor_height)
                floor_cand = hit_mask & inb_all & (v_valid.astype(np.float32) >= (float(h - 1) * float(floor_map_vmin))) & (z_valid < (float(max_depth) * float(floor_map_zmax))) & (rel0 > -1.0) & (rel0 < float(floor_map_rel_max))
                if np.any(floor_cand):
                    lin = (py_all[floor_cand].astype(np.int64) * int(w_map) + px_all[floor_cand].astype(np.int64))
                    yv = y_world[floor_cand].astype(np.float32, copy=False)
                    counts = np.bincount(lin, minlength=int(h_map) * int(w_map)).astype(np.int32, copy=False)
                    if counts.size == int(h_map) * int(w_map) and np.any(counts > 0):
                        sums = np.bincount(lin, weights=yv, minlength=int(h_map) * int(w_map)).astype(np.float32, copy=False)
                        idx = np.flatnonzero(counts > 0)
                        mean_y = sums[idx] / np.maximum(1.0, counts[idx].astype(np.float32))
                        my = (idx // int(w_map)).astype(np.int32, copy=False)
                        mx = (idx % int(w_map)).astype(np.int32, copy=False)
                        old = self.floor_y_map[my, mx]
                        new = np.where(np.isfinite(old), (1.0 - float(floor_map_beta)) * old + float(floor_map_beta) * mean_y, mean_y)
                        self.floor_y_map[my, mx] = new.astype(np.float32, copy=False)
                        self.floor_y_conf[my, mx] = np.minimum(float(floor_map_conf_cap), self.floor_y_conf[my, mx] + float(floor_map_conf_inc))
            y_ref = np.full((y_world.shape[0],), float(floor_height), dtype=np.float32)
            if np.any(inb_all):
                fy = self.floor_y_map[py_all[inb_all], px_all[inb_all]]
                fyc = self.floor_y_conf[py_all[inb_all], px_all[inb_all]]
                ok = np.isfinite(fy) & (fyc >= float(floor_map_conf_min))
                y_ref[inb_all] = np.where(ok, fy, float(floor_height)).astype(np.float32, copy=False)
            y_local = y_world - y_ref
        else:
            y_local = y_world - float(floor_height)

        try:
            nohit_max_ylocal = float(os.environ.get("HABITAT_NOHIT_MAX_YLOCAL", "0.8"))
        except Exception:
            nohit_max_ylocal = 0.8
        nohit_max_ylocal = float(max(0.0, min(nohit_max_ylocal, 5.0)))
        nohit_mask = (~hit_mask) & (y_local < nohit_max_ylocal) & (y_local > -1.0)

        use_zero_nohit = False
        try:
            if depth_sub.size > 0:
                zr = float(np.count_nonzero(nohit_depth_mask)) / float(depth_sub.size)
                vr = float(np.count_nonzero(valid_mask)) / float(depth_sub.size)
                use_zero_nohit = (zr >= nohit_zero_ratio) and (vr >= min_valid_ratio)
        except Exception:
            use_zero_nohit = False
        
        # Ground / Obstacle masks are computed from projected points.
        # IMPORTANT: For no-hit rays (depth ~ max range), the "endpoint" is not a real surface and can create
        # a black ring if we treat it as an obstacle. Therefore occupancy / ground reinforcement only uses hit points.
        # Free space is handled separately via ray clearing.
        hit_points_mask = hit_mask

        try:
            ground_max_ylocal = float(os.environ.get("HABITAT_GROUND_MAX_YLOCAL", "0.3"))
        except Exception:
            ground_max_ylocal = 0.3
        ground_max_ylocal = float(max(0.05, min(ground_max_ylocal, 1.0)))

        try:
            obs_min_ylocal = float(os.environ.get("HABITAT_OBS_MIN_YLOCAL", str(max(0.60, ground_max_ylocal + 0.25))))
        except Exception:
            obs_min_ylocal = float(max(0.60, ground_max_ylocal + 0.25))
        obs_min_ylocal = float(max(0.1, min(obs_min_ylocal, 2.0)))

        try:
            obs_max_ylocal = float(os.environ.get("HABITAT_OBS_MAX_YLOCAL", "2.0"))
        except Exception:
            obs_max_ylocal = 2.0
        obs_max_ylocal = float(max(obs_min_ylocal + 0.1, min(obs_max_ylocal, 5.0)))
        try:
            obs_nav_max_ylocal = float(os.environ.get("HABITAT_OBS_NAV_MAX_YLOCAL", "1.3"))
        except Exception:
            obs_nav_max_ylocal = 1.3
        obs_nav_max_ylocal = float(max(float(obs_min_ylocal) + 0.1, min(float(obs_nav_max_ylocal), float(obs_max_ylocal))))

        try:
            obs_max_depth = float(os.environ.get("HABITAT_OBS_MAX_DEPTH", "6.0"))
        except Exception:
            obs_max_depth = 6.0
        obs_max_depth = float(max(0.5, min(obs_max_depth, max_depth)))
        try:
            obs_depth_falloff = float(os.environ.get("HABITAT_OBS_DEPTH_FALLOFF", str(obs_max_depth)))
        except Exception:
            obs_depth_falloff = float(obs_max_depth)
        obs_depth_falloff = float(max(0.0, min(obs_depth_falloff, max_depth)))
        try:
            obs_depth_min_scale = float(os.environ.get("HABITAT_OBS_DEPTH_MIN_SCALE", "0.35"))
        except Exception:
            obs_depth_min_scale = 0.35
        obs_depth_min_scale = float(max(0.0, min(obs_depth_min_scale, 1.0)))

        try:
            ground_max_depth = float(os.environ.get("HABITAT_GROUND_MAX_DEPTH", str(clear_depth_cap)))
        except Exception:
            ground_max_depth = float(clear_depth_cap)
        ground_max_depth = float(max(0.5, min(ground_max_depth, float(max_depth))))
        ground_mask = (y_local < ground_max_ylocal) & (y_local > -0.5) & hit_points_mask & (z_valid <= ground_max_depth)
        
        obstacle_low_mask = (y_local >= obs_min_ylocal) & (y_local < 0.7) & (y_local < obs_nav_max_ylocal) & hit_points_mask & obs_row_mask & (z_valid <= obs_max_depth)
        obstacle_mid_mask = (y_local >= 0.7) & (y_local < 1.5) & (y_local < obs_nav_max_ylocal) & hit_points_mask & obs_row_mask & (z_valid <= obs_max_depth)
        obstacle_high_mask = (y_local >= 1.5) & (y_local < 3.0) & (y_local < obs_nav_max_ylocal) & hit_points_mask & obs_row_mask & (z_valid <= obs_max_depth)
        obstacle_mask = (y_local >= obs_min_ylocal) & (y_local < obs_max_ylocal) & (y_local < obs_nav_max_ylocal) & hit_points_mask & obs_row_mask & (z_valid <= obs_max_depth)

        # DEBUG: Print stats if too few points
        # if np.sum(ground_mask) < 100 and np.sum(obstacle_mask) < 10:
        #    print(f"[Mapper] Low points! FloorH: {floor_height:.2f}, Y_local range: {y_local.min():.2f} ~ {y_local.max():.2f}", flush=True)
        
        # 获取 Map 索引
        def points_to_indices(points):
            px = ((points[:, 0] - self.origin_x) / self.cell_size).astype(int)
            py = ((points[:, 2] - self.origin_y) / self.cell_size).astype(int) # Z is Map Y
            return px, py
            
        h_map, w_map = self.global_map.shape
        
        try:
            clearing_max_ylocal = float(os.environ.get("HABITAT_CLEARING_MAX_YLOCAL", "2.0"))
        except Exception:
            clearing_max_ylocal = 2.0
        clearing_max_ylocal = float(max(0.5, min(clearing_max_ylocal, 5.0)))
        clearing_mask = (y_local < clearing_max_ylocal) & hit_mask & clear_row_mask & (z_valid <= clear_depth_cap)
        try:
            clear_floor_only = int(os.environ.get("HABITAT_CLEAR_FLOOR_ONLY_ENABLE", "1")) != 0
        except Exception:
            clear_floor_only = True
        if clear_floor_only:
            try:
                clear_floor_ymin = float(os.environ.get("HABITAT_CLEAR_FLOOR_YLOCAL_MIN", "-0.35"))
            except Exception:
                clear_floor_ymin = -0.35
            try:
                clear_floor_ymax = float(os.environ.get("HABITAT_CLEAR_FLOOR_YLOCAL_MAX", str(float(ground_max_ylocal) + 0.10)))
            except Exception:
                clear_floor_ymax = float(ground_max_ylocal) + 0.10
            clear_floor_ymin = float(max(-2.0, min(clear_floor_ymin, 2.0)))
            clear_floor_ymax = float(max(clear_floor_ymin + 0.05, min(clear_floor_ymax, 3.0)))
            clearing_mask = clearing_mask & (y_local >= clear_floor_ymin) & (y_local <= clear_floor_ymax)
        
        # Update Free (Ground) - Optional now, but good for reinforcing details
        if np.any(ground_mask):
            g_pts = world_points[ground_mask]
            gx, gy = points_to_indices(g_pts)
            # Boundary Check
            valid_g = (gx >= 0) & (gx < w_map) & (gy >= 0) & (gy < h_map)
            try:
                ground_add = float(os.environ.get("HABITAT_GROUND_FREE_ADD", "0.25"))
            except Exception:
                ground_add = 0.25
            self.free_count[gy[valid_g], gx[valid_g]] += ground_add
            try:
                logodds_enable = int(os.environ.get("HABITAT_LOGODDS_ENABLE", "0")) != 0
            except Exception:
                logodds_enable = False
            if logodds_enable and np.any(valid_g):
                try:
                    lo_free_ground = float(os.environ.get("HABITAT_LO_FREE_GROUND", "0.20"))
                except Exception:
                    lo_free_ground = 0.20
                lo_free_ground = float(max(0.0, min(lo_free_ground, 5.0)))
                self.log_odds[gy[valid_g], gx[valid_g]] -= lo_free_ground
            
        try:
            occ_add = float(os.environ.get("HABITAT_OCC_ADD", "0.7"))
        except Exception:
            occ_add = 0.7
        occ_add = float(max(0.0, min(occ_add, 10.0)))
        try:
            occ_add_low = float(os.environ.get("HABITAT_OCC_ADD_LOW", str(0.7 * float(occ_add))))
        except Exception:
            occ_add_low = 0.7 * float(occ_add)
        try:
            occ_add_mid = float(os.environ.get("HABITAT_OCC_ADD_MID", str(1.0 * float(occ_add))))
        except Exception:
            occ_add_mid = 1.0 * float(occ_add)
        try:
            occ_add_high = float(os.environ.get("HABITAT_OCC_ADD_HIGH", str(1.0 * float(occ_add))))
        except Exception:
            occ_add_high = 1.0 * float(occ_add)
        occ_add_low = float(max(0.0, min(occ_add_low, 10.0)))
        occ_add_mid = float(max(0.0, min(occ_add_mid, 10.0)))
        occ_add_high = float(max(0.0, min(occ_add_high, 10.0)))
        
        if np.any(obstacle_low_mask):
            o_pts = world_points[obstacle_low_mask]
            z_o = z_valid[obstacle_low_mask]
            ox, oy = points_to_indices(o_pts)
            valid_o = (ox >= 0) & (ox < w_map) & (oy >= 0) & (oy < h_map)
            oxv = ox[valid_o]
            oyv = oy[valid_o]
            if oxv.size > 0:
                if obs_depth_falloff > 1e-6:
                    wgt = (float(obs_depth_falloff) - z_o[valid_o].astype(np.float32, copy=False)) / float(obs_depth_falloff)
                    wgt = np.clip(wgt, float(obs_depth_min_scale), 1.0)
                else:
                    wgt = 1.0
                np.add.at(self.occ_count, (oyv, oxv), (float(occ_add_low) * wgt))
                np.add.at(self.occ_low_count, (oyv, oxv), 1.0)
            try:
                logodds_enable = int(os.environ.get("HABITAT_LOGODDS_ENABLE", "0")) != 0
            except Exception:
                logodds_enable = False
            if logodds_enable and oxv.size > 0:
                try:
                    lo_occ_low = float(os.environ.get("HABITAT_LO_OCC_LOW", str(0.65 * float(occ_add_low))))
                except Exception:
                    lo_occ_low = 0.65 * float(occ_add_low)
                lo_occ_low = float(max(0.0, min(lo_occ_low, 8.0)))
                np.add.at(self.log_odds, (oyv, oxv), (float(lo_occ_low) * wgt))

        if np.any(obstacle_mid_mask):
            o_pts = world_points[obstacle_mid_mask]
            z_o = z_valid[obstacle_mid_mask]
            ox, oy = points_to_indices(o_pts)
            valid_o = (ox >= 0) & (ox < w_map) & (oy >= 0) & (oy < h_map)
            oxv = ox[valid_o]
            oyv = oy[valid_o]
            if oxv.size > 0:
                if obs_depth_falloff > 1e-6:
                    wgt = (float(obs_depth_falloff) - z_o[valid_o].astype(np.float32, copy=False)) / float(obs_depth_falloff)
                    wgt = np.clip(wgt, float(obs_depth_min_scale), 1.0)
                else:
                    wgt = 1.0
                np.add.at(self.occ_count, (oyv, oxv), (float(occ_add_mid) * wgt))
                np.add.at(self.occ_mid_count, (oyv, oxv), 1.0)
            try:
                logodds_enable = int(os.environ.get("HABITAT_LOGODDS_ENABLE", "0")) != 0
            except Exception:
                logodds_enable = False
            if logodds_enable and oxv.size > 0:
                try:
                    lo_occ_mid = float(os.environ.get("HABITAT_LO_OCC_MID", str(0.65 * float(occ_add_mid))))
                except Exception:
                    lo_occ_mid = 0.65 * float(occ_add_mid)
                lo_occ_mid = float(max(0.0, min(lo_occ_mid, 8.0)))
                np.add.at(self.log_odds, (oyv, oxv), (float(lo_occ_mid) * wgt))

        if np.any(obstacle_high_mask):
            o_pts = world_points[obstacle_high_mask]
            z_o = z_valid[obstacle_high_mask]
            ox, oy = points_to_indices(o_pts)
            valid_o = (ox >= 0) & (ox < w_map) & (oy >= 0) & (oy < h_map)
            oxv = ox[valid_o]
            oyv = oy[valid_o]
            if oxv.size > 0:
                if obs_depth_falloff > 1e-6:
                    wgt = (float(obs_depth_falloff) - z_o[valid_o].astype(np.float32, copy=False)) / float(obs_depth_falloff)
                    wgt = np.clip(wgt, float(obs_depth_min_scale), 1.0)
                else:
                    wgt = 1.0
                np.add.at(self.occ_count, (oyv, oxv), (float(occ_add_high) * wgt))
                np.add.at(self.occ_high_count, (oyv, oxv), 1.0)
            try:
                logodds_enable = int(os.environ.get("HABITAT_LOGODDS_ENABLE", "0")) != 0
            except Exception:
                logodds_enable = False
            if logodds_enable and oxv.size > 0:
                try:
                    lo_occ_high = float(os.environ.get("HABITAT_LO_OCC_HIGH", str(0.65 * float(occ_add_high))))
                except Exception:
                    lo_occ_high = 0.65 * float(occ_add_high)
                lo_occ_high = float(max(0.0, min(lo_occ_high, 8.0)))
                np.add.at(self.log_odds, (oyv, oxv), (float(lo_occ_high) * wgt))

        if np.any(clearing_mask) or np.any(nohit_mask):
            cam_px = int((sensor_pos[0] - self.origin_x) / self.cell_size)
            cam_py = int((sensor_pos[2] - self.origin_y) / self.cell_size)

            try:
                ray_stride = int(os.environ.get("HABITAT_RAYCAST_STRIDE", "4"))
            except Exception:
                ray_stride = 4
            ray_stride = int(max(1, min(16, ray_stride)))

            try:
                ray_free_add = float(os.environ.get("HABITAT_RAY_FREE_ADD", "0.28"))
            except Exception:
                ray_free_add = 0.28

            try:
                nohit_enable = int(os.environ.get("HABITAT_RAY_CLEAR_NOHIT", "0")) != 0
            except Exception:
                nohit_enable = False
            try:
                nohit_stride = int(os.environ.get("HABITAT_NOHIT_STRIDE", "8"))
            except Exception:
                nohit_stride = 8
            nohit_stride = int(max(1, min(32, nohit_stride)))
            try:
                nohit_free_add = float(os.environ.get("HABITAT_NOHIT_FREE_ADD", "0.05"))
            except Exception:
                nohit_free_add = 0.05

            try:
                clear_skip_strong_occ = int(os.environ.get("HABITAT_CLEAR_SKIP_STRONG_OCC", "1")) != 0
            except Exception:
                clear_skip_strong_occ = True
            try:
                strong_occ_margin = float(os.environ.get("HABITAT_STRONG_OCC_MARGIN", "4.0"))
            except Exception:
                strong_occ_margin = 4.0
            strong_occ_margin = float(max(0.0, min(strong_occ_margin, 50.0)))

            try:
                occ_barrier_abs_min = float(os.environ.get("HABITAT_CLEAR_OCC_ABS_MIN", "2.0"))
            except Exception:
                occ_barrier_abs_min = 2.0
            occ_barrier_abs_min = float(max(0.0, min(occ_barrier_abs_min, 50.0)))
            try:
                occ_barrier_vert_min = float(os.environ.get("HABITAT_CLEAR_OCC_VERT_MIN", "1.0"))
            except Exception:
                occ_barrier_vert_min = 1.0
            occ_barrier_vert_min = float(max(0.0, min(occ_barrier_vert_min, 50.0)))

            def bresenham(x0, y0, x1, y1):
                x0 = int(x0); y0 = int(y0); x1 = int(x1); y1 = int(y1)
                dx = abs(x1 - x0)
                dy = -abs(y1 - y0)
                sx = 1 if x0 < x1 else -1
                sy = 1 if y0 < y1 else -1
                err = dx + dy
                x = x0
                y = y0
                out = []
                while True:
                    out.append((x, y))
                    if x == x1 and y == y1:
                        break
                    e2 = 2 * err
                    if e2 >= dy:
                        err += dy
                        x += sx
                    if e2 <= dx:
                        err += dx
                        y += sy
                return out

            def is_occ_barrier(lx, ly):
                if not (0 <= lx < w_map and 0 <= ly < h_map):
                    return False
                occ = float(self.occ_count[ly, lx])
                free = float(self.free_count[ly, lx])
                occ_vert = float(self.occ_mid_count[ly, lx]) + float(self.occ_high_count[ly, lx])
                occ_low = float(self.occ_low_count[ly, lx])
                try:
                    barrier_margin = float(os.environ.get("HABITAT_CLEAR_BARRIER_MARGIN", str(strong_occ_margin)))
                except Exception:
                    barrier_margin = float(strong_occ_margin)
                barrier_margin = float(max(0.0, min(barrier_margin, 50.0)))
                if (occ - free) <= barrier_margin:
                    return False
                if occ_vert >= occ_barrier_vert_min:
                    return True
                try:
                    occ_barrier_vert_ratio_min = float(os.environ.get("HABITAT_CLEAR_OCC_VERT_RATIO_MIN", "0.2"))
                except Exception:
                    occ_barrier_vert_ratio_min = 0.2
                occ_barrier_vert_ratio_min = float(max(0.0, min(occ_barrier_vert_ratio_min, 1.0)))
                occ_total = float(occ_low + occ_vert)
                occ_vert_ratio = float(occ_vert) / float(max(1e-6, occ_total))
                return (occ >= occ_barrier_abs_min) and (occ_vert_ratio >= occ_barrier_vert_ratio_min)

            mask_idx = np.flatnonzero(clearing_mask)
            if mask_idx.size > 0:
                mask_idx = mask_idx[::ray_stride]
                pts = world_points[mask_idx]
                px, py = points_to_indices(pts)
                inb = (px >= 0) & (px < w_map) & (py >= 0) & (py < h_map)
                if np.any(inb):
                    px = px[inb]
                    py = py[inb]
                    yloc = y_local[mask_idx][inb]
                    try:
                        ray_end_occ_add = float(os.environ.get("HABITAT_RAY_END_OCC_ADD", "0.15"))
                    except Exception:
                        ray_end_occ_add = 0.15
                    ray_end_occ_add = float(max(0.0, min(ray_end_occ_add, 10.0)))
                    try:
                        logodds_enable = int(os.environ.get("HABITAT_LOGODDS_ENABLE", "0")) != 0
                    except Exception:
                        logodds_enable = False
                    if logodds_enable:
                        try:
                            lo_free_ray = float(os.environ.get("HABITAT_LO_FREE_RAY", "0.16"))
                        except Exception:
                            lo_free_ray = 0.16
                        lo_free_ray = float(max(0.0, min(lo_free_ray, 5.0)))
                        try:
                            lo_occ_end = float(os.environ.get("HABITAT_LO_OCC_RAY_END", "0.25"))
                        except Exception:
                            lo_occ_end = 0.25
                        lo_occ_end = float(max(0.0, min(lo_occ_end, 8.0)))
                    for ex, ey, yy in zip(px.tolist(), py.tolist(), yloc.tolist()):
                        line = bresenham(cam_px, cam_py, ex, ey)
                        if len(line) > 1:
                            free_along = 0
                            try:
                                stop_before_end = int(os.environ.get("HABITAT_RAY_STOP_BEFORE_END_CELLS", "3"))
                            except Exception:
                                stop_before_end = 3
                            stop_before_end = int(max(0, min(stop_before_end, 128)))
                            steps = line[:-1]
                            if stop_before_end > 0 and len(steps) > stop_before_end:
                                steps = steps[:-stop_before_end]
                            for lx, ly in steps:
                                if 0 <= lx < w_map and 0 <= ly < h_map:
                                    if is_occ_barrier(lx, ly):
                                        break
                                    if clear_skip_strong_occ:
                                        if float(self.occ_count[ly, lx]) - float(self.free_count[ly, lx]) > strong_occ_margin:
                                            break
                                    self.free_count[ly, lx] += ray_free_add
                                    free_along += 1
                                    if logodds_enable:
                                        self.log_odds[ly, lx] -= lo_free_ray
                            end_add = float(ray_end_occ_add)
                            try:
                                near_band = float(os.environ.get("HABITAT_OBS_NEAR_BAND", "0.2"))
                            except Exception:
                                near_band = 0.2
                            near_band = float(max(0.0, min(near_band, 1.0)))
                            try:
                                occ_scale_near = float(os.environ.get("HABITAT_RAY_END_OCC_SCALE_NEAR", "0.6"))
                            except Exception:
                                occ_scale_near = 0.6
                            occ_scale_near = float(max(0.0, min(occ_scale_near, 1.0)))
                            if float(yy) < (float(obs_min_ylocal) + float(near_band)):
                                end_add *= occ_scale_near
                            try:
                                end_free_min = int(os.environ.get("HABITAT_RAY_END_OCC_FREE_MIN", "6"))
                            except Exception:
                                end_free_min = 6
                            end_free_min = int(max(0, min(end_free_min, 1000)))
                            try:
                                end_free_scale = float(os.environ.get("HABITAT_RAY_END_OCC_FREE_SCALE", "0.5"))
                            except Exception:
                                end_free_scale = 0.5
                            end_free_scale = float(max(0.0, min(end_free_scale, 1.0)))
                            if int(free_along) >= int(end_free_min):
                                end_add *= end_free_scale
                            if obs_min_ylocal <= float(yy) < obs_max_ylocal and end_add > 0.0:
                                self.occ_count[int(ey), int(ex)] += end_add
                                if logodds_enable:
                                    self.log_odds[int(ey), int(ex)] += (lo_occ_end * (end_add / max(1e-6, float(ray_end_occ_add))))

            if nohit_enable:
                mask2_idx = np.flatnonzero(nohit_mask)
                if mask2_idx.size > 0:
                    mask2_idx = mask2_idx[::nohit_stride]
                    pts2 = world_points[mask2_idx]
                    px2, py2 = points_to_indices(pts2)
                    inb2 = (px2 >= 0) & (px2 < w_map) & (py2 >= 0) & (py2 < h_map)
                    if np.any(inb2):
                        px2 = px2[inb2]
                        py2 = py2[inb2]
                        try:
                            logodds_enable = int(os.environ.get("HABITAT_LOGODDS_ENABLE", "0")) != 0
                        except Exception:
                            logodds_enable = False
                        if logodds_enable:
                            try:
                                lo_free_nohit = float(os.environ.get("HABITAT_LO_FREE_NOHIT", "0.05"))
                            except Exception:
                                lo_free_nohit = 0.05
                            lo_free_nohit = float(max(0.0, min(lo_free_nohit, 5.0)))
                        for ex, ey in zip(px2.tolist(), py2.tolist()):
                            line = bresenham(cam_px, cam_py, ex, ey)
                            if len(line) > 1:
                                try:
                                    nohit_max_cells = int(os.environ.get("HABITAT_NOHIT_CLEAR_MAX_CELLS", "6"))
                                except Exception:
                                    nohit_max_cells = 6
                                nohit_max_cells = int(max(0, min(nohit_max_cells, 4096)))
                                steps = line[:-1]
                                if int(nohit_max_cells) > 0:
                                    steps = steps[:int(nohit_max_cells)]
                                for lx, ly in steps:
                                    if 0 <= lx < w_map and 0 <= ly < h_map:
                                        if is_occ_barrier(lx, ly):
                                            break
                                        if clear_skip_strong_occ:
                                            if float(self.occ_count[ly, lx]) - float(self.free_count[ly, lx]) > strong_occ_margin:
                                                break
                                        self.free_count[ly, lx] += nohit_free_add
                                        if logodds_enable:
                                            self.log_odds[ly, lx] -= lo_free_nohit
                if use_zero_nohit:
                    row_band = int(max(1, round(float(h) * float(nohit_row_band))))
                    band_mask = (np.abs(v_grid.astype(np.int32) - int(round(float(cy)))) <= row_band)
                    pick = nohit_depth_mask & band_mask
                    uu = u_grid[pick].astype(np.int64, copy=False)
                    vv = v_grid[pick].astype(np.int64, copy=False)
                    if uu.size > 0:
                        uu = uu[::nohit_pixel_stride]
                        vv = vv[::nohit_pixel_stride]
                        z2 = np.full_like(uu, fill_value=max_depth, dtype=np.float32)
                        x2 = (uu.astype(np.float32) - float(cx)) * z2 / float(fx)
                        y2 = -(vv.astype(np.float32) - float(cy)) * z2 / float(fy)
                        z2v = -z2
                        cam2 = np.stack([x2, y2, z2v], axis=1)
                        w2 = (r_mat @ cam2.T).T + np.array(sensor_pos)
                        if floor_map_enable:
                            px_t = ((w2[:, 0] - self.origin_x) / self.cell_size).astype(np.int32, copy=False)
                            py_t = ((w2[:, 2] - self.origin_y) / self.cell_size).astype(np.int32, copy=False)
                            inb_t = (px_t >= 0) & (px_t < w_map) & (py_t >= 0) & (py_t < h_map)
                            yref2 = np.full((w2.shape[0],), float(floor_height), dtype=np.float32)
                            if np.any(inb_t):
                                fy2 = self.floor_y_map[py_t[inb_t], px_t[inb_t]]
                                fyc2 = self.floor_y_conf[py_t[inb_t], px_t[inb_t]]
                                ok2 = np.isfinite(fy2) & (fyc2 >= float(floor_map_conf_min))
                                yref2[inb_t] = np.where(ok2, fy2, float(floor_height)).astype(np.float32, copy=False)
                            y2l = w2[:, 1].astype(np.float32, copy=False) - yref2
                        else:
                            y2l = w2[:, 1] - float(floor_height)
                        m2 = (y2l < nohit_max_ylocal) & (y2l > -1.0)
                        if np.any(m2):
                            px3, py3 = points_to_indices(w2[m2])
                            inb3 = (px3 >= 0) & (px3 < w_map) & (py3 >= 0) & (py3 < h_map)
                            if np.any(inb3):
                                px3 = px3[inb3]
                                py3 = py3[inb3]
                                try:
                                    logodds_enable = int(os.environ.get("HABITAT_LOGODDS_ENABLE", "0")) != 0
                                except Exception:
                                    logodds_enable = False
                                if logodds_enable:
                                    try:
                                        lo_free_nohit = float(os.environ.get("HABITAT_LO_FREE_NOHIT", "0.05"))
                                    except Exception:
                                        lo_free_nohit = 0.05
                                    lo_free_nohit = float(max(0.0, min(lo_free_nohit, 5.0)))
                                for ex, ey in zip(px3.tolist(), py3.tolist()):
                                    line = bresenham(cam_px, cam_py, ex, ey)
                                    if len(line) > 1:
                                        try:
                                            nohit_max_cells = int(os.environ.get("HABITAT_NOHIT_CLEAR_MAX_CELLS", "6"))
                                        except Exception:
                                            nohit_max_cells = 6
                                        nohit_max_cells = int(max(0, min(nohit_max_cells, 4096)))
                                        steps = line[:-1]
                                        if int(nohit_max_cells) > 0:
                                            steps = steps[:int(nohit_max_cells)]
                                        for lx, ly in steps:
                                            if 0 <= lx < w_map and 0 <= ly < h_map:
                                                if is_occ_barrier(lx, ly):
                                                    break
                                                if clear_skip_strong_occ:
                                                    if float(self.occ_count[ly, lx]) - float(self.free_count[ly, lx]) > strong_occ_margin:
                                                        break
                                                self.free_count[ly, lx] += nohit_free_add
                                                if logodds_enable:
                                                    self.log_odds[ly, lx] -= lo_free_nohit

            
        # 5. 强制清除机器人自身位置 (Footprint Clearing)
        self._clear_footprint(agent_base_state)
        
        try:
            decay = float(os.environ.get("HABITAT_COUNT_DECAY", "0.999"))
        except Exception:
            decay = 0.999
        decay = float(max(0.95, min(decay, 1.0)))
        self.free_count *= decay
        self.occ_count *= decay
        self.occ_low_count *= decay
        self.occ_mid_count *= decay
        self.occ_high_count *= decay
        try:
            floor_conf_decay = float(os.environ.get("HABITAT_FLOOR_MAP_CONF_DECAY", "0.997"))
        except Exception:
            floor_conf_decay = 0.997
        floor_conf_decay = float(max(0.9, min(floor_conf_decay, 1.0)))
        self.floor_y_conf *= floor_conf_decay
        np.clip(self.free_count, 0.0, 50.0, out=self.free_count)
        np.clip(self.occ_count, 0.0, 50.0, out=self.occ_count)
        np.clip(self.occ_low_count, 0.0, 50.0, out=self.occ_low_count)
        np.clip(self.occ_mid_count, 0.0, 50.0, out=self.occ_mid_count)
        np.clip(self.occ_high_count, 0.0, 50.0, out=self.occ_high_count)
        np.clip(self.floor_y_conf, 0.0, 200.0, out=self.floor_y_conf)
        try:
            logodds_enable = int(os.environ.get("HABITAT_LOGODDS_ENABLE", "0")) != 0
        except Exception:
            logodds_enable = False
        if logodds_enable:
            try:
                lo_decay = float(os.environ.get("HABITAT_LO_DECAY", str(decay)))
            except Exception:
                lo_decay = float(decay)
            lo_decay = float(max(0.95, min(lo_decay, 1.0)))
            self.log_odds *= lo_decay
            try:
                lo_clip = float(os.environ.get("HABITAT_LO_CLIP", "6.0"))
            except Exception:
                lo_clip = 6.0
            lo_clip = float(max(0.5, min(lo_clip, 30.0)))
            np.clip(self.log_odds, -lo_clip, lo_clip, out=self.log_odds)
        try:
            floor_conf_drop = float(os.environ.get("HABITAT_FLOOR_MAP_CONF_DROP", "0.25"))
        except Exception:
            floor_conf_drop = 0.25
        floor_conf_drop = float(max(0.0, min(floor_conf_drop, 50.0)))
        if floor_conf_drop > 0.0:
            drop = self.floor_y_conf < float(floor_conf_drop)
            if np.any(drop):
                self.floor_y_map[drop] = np.nan

        try:
            stamp_enable = int(os.environ.get("HABITAT_ROBOT_STAMP_ENABLE", "1")) != 0
        except Exception:
            stamp_enable = True
        if stamp_enable and agent_base_state is not None:
            try:
                rx = float(agent_base_state.position[0])
                rz = float(agent_base_state.position[2])
                ry = float(agent_base_state.position[1]) if (len(agent_base_state.position) >= 2) else 0.0
                mx0 = int(round((rx - float(self.origin_x)) / float(self.cell_size)))
                my0 = int(round((rz - float(self.origin_y)) / float(self.cell_size)))
                h_map, w_map = self.global_map.shape
                if 0 <= mx0 < w_map and 0 <= my0 < h_map:
                    try:
                        r_cells = int(os.environ.get("HABITAT_ROBOT_STAMP_RADIUS_CELLS", "2"))
                    except Exception:
                        r_cells = 2
                    r_cells = int(max(0, min(r_cells, 20)))
                    try:
                        free_add = float(os.environ.get("HABITAT_ROBOT_STAMP_FREE_ADD", "4.0"))
                    except Exception:
                        free_add = 4.0
                    free_add = float(max(0.0, min(free_add, 50.0)))
                    try:
                        occ_veto = float(os.environ.get("HABITAT_ROBOT_STAMP_OCC_VETO", "6.0"))
                    except Exception:
                        occ_veto = 6.0
                    occ_veto = float(max(0.0, min(occ_veto, 50.0)))
                    try:
                        trail_enable = int(os.environ.get("HABITAT_ROBOT_STAMP_TRAIL_ENABLE", "1")) != 0
                    except Exception:
                        trail_enable = True
                    try:
                        trail_max_dist_m = float(os.environ.get("HABITAT_ROBOT_STAMP_TRAIL_MAX_DIST_M", "1.25"))
                    except Exception:
                        trail_max_dist_m = 1.25
                    trail_max_dist_m = float(max(0.0, min(trail_max_dist_m, 20.0)))
                    try:
                        trail_free_scale = float(os.environ.get("HABITAT_ROBOT_STAMP_TRAIL_FREE_SCALE", "0.8"))
                    except Exception:
                        trail_free_scale = 0.8
                    trail_free_scale = float(max(0.0, min(trail_free_scale, 10.0)))

                    def _bresenham(x0, y0, x1, y1):
                        x0 = int(x0); y0 = int(y0); x1 = int(x1); y1 = int(y1)
                        dx = abs(x1 - x0)
                        dy = -abs(y1 - y0)
                        sx = 1 if x0 < x1 else -1
                        sy = 1 if y0 < y1 else -1
                        err = dx + dy
                        x = x0
                        y = y0
                        out = []
                        while True:
                            out.append((x, y))
                            if x == x1 and y == y1:
                                break
                            e2 = 2 * err
                            if e2 >= dy:
                                err += dy
                                x += sx
                            if e2 <= dx:
                                err += dx
                                y += sy
                        return out

                    def _stamp_at(cx, cy, add_val):
                        if add_val <= 0.0:
                            return
                        pf = getattr(self, "pathfinder", None)
                        if pf is not None and getattr(pf, "is_loaded", False):
                            try:
                                wx = float(self.origin_x) + (float(cx) + 0.5) * float(self.cell_size)
                                wz = float(self.origin_y) + (float(cy) + 0.5) * float(self.cell_size)
                                if not bool(pf.is_navigable(np.array([wx, float(ry), wz], dtype=np.float32))):
                                    return
                            except Exception:
                                pass
                        y0 = int(max(0, cy - r_cells))
                        y1 = int(min(h_map - 1, cy + r_cells))
                        x0 = int(max(0, cx - r_cells))
                        x1 = int(min(w_map - 1, cx + r_cells))
                        if y1 < y0 or x1 < x0:
                            return
                        yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
                        dy = yy - int(cy)
                        dx = xx - int(cx)
                        disk = (dx * dx + dy * dy) <= int(r_cells * r_cells)
                        if not bool(np.any(disk)):
                            return
                        occ_patch = self.occ_count[y0 : y1 + 1, x0 : x1 + 1]
                        allow = disk & (occ_patch <= float(occ_veto))
                        if not bool(np.any(allow)):
                            return
                        free_patch = self.free_count[y0 : y1 + 1, x0 : x1 + 1]
                        free_patch[allow] = free_patch[allow] + float(add_val)

                    if trail_enable and getattr(self, "_prev_stamp_cell", None) is not None:
                        px, py = self._prev_stamp_cell
                        if 0 <= int(px) < w_map and 0 <= int(py) < h_map:
                            d_cells = float(np.hypot(float(mx0 - int(px)), float(my0 - int(py))))
                            if (d_cells * float(self.cell_size)) <= float(trail_max_dist_m):
                                add_val = float(free_add) * float(trail_free_scale)
                                for cx, cy in _bresenham(int(px), int(py), int(mx0), int(my0)):
                                    _stamp_at(int(cx), int(cy), add_val)

                    _stamp_at(int(mx0), int(my0), float(free_add))
                    self._prev_stamp_cell = (int(mx0), int(my0))
            except Exception:
                pass
        diff = self.free_count - self.occ_count
        try:
            free_thresh = float(os.environ.get("HABITAT_MAP_FREE_THRESH", "1.0"))
        except Exception:
            free_thresh = 1.0
        try:
            occ_thresh = float(os.environ.get("HABITAT_MAP_OCC_THRESH", "2.8"))
        except Exception:
            occ_thresh = 2.8
        occ_thresh = float(max(0.5, min(occ_thresh, 20.0)))

        conf = self.free_count + self.occ_count
        try:
            conf_thresh = float(os.environ.get("HABITAT_MAP_CONF_THRESH", "3.0"))
        except Exception:
            conf_thresh = 3.0
        conf_thresh = float(max(0.0, min(conf_thresh, 50.0)))

        try:
            occ_vert_enable = int(os.environ.get("HABITAT_OCC_VERT_ENABLE", "1")) != 0
        except Exception:
            occ_vert_enable = True
        if occ_vert_enable:
            try:
                occ_vert_min = float(os.environ.get("HABITAT_OCC_VERT_MIN", "4.0"))
            except Exception:
                occ_vert_min = 4.0
            occ_vert_min = float(max(0.0, min(occ_vert_min, 50.0)))
            try:
                occ_abs_min = float(os.environ.get("HABITAT_OCC_ABS_MIN", "18.0"))
            except Exception:
                occ_abs_min = 18.0
            occ_abs_min = float(max(0.0, min(occ_abs_min, 50.0)))
            try:
                occ_vert_ratio_min = float(os.environ.get("HABITAT_OCC_VERT_RATIO_MIN", "0.25"))
            except Exception:
                occ_vert_ratio_min = 0.25
            occ_vert_ratio_min = float(max(0.0, min(occ_vert_ratio_min, 1.0)))

            occ_vert = self.occ_mid_count + self.occ_high_count
            occ_total = self.occ_low_count + occ_vert
            occ_vert_ratio = occ_vert / np.maximum(1e-6, occ_total)
            occ_ok = (occ_vert >= occ_vert_min) | ((self.occ_count >= occ_abs_min) & (occ_vert_ratio >= occ_vert_ratio_min))
            occ_mask = (diff < -occ_thresh) & occ_ok
        else:
            occ_mask = (diff < -occ_thresh)

        try:
            occ_require_conf = int(os.environ.get("HABITAT_OCC_REQUIRE_CONF", "1")) != 0
        except Exception:
            occ_require_conf = True
        if occ_require_conf:
            try:
                occ_conf_ratio = float(os.environ.get("HABITAT_OCC_CONF_RATIO", "1.15"))
            except Exception:
                occ_conf_ratio = 1.15
            occ_conf_ratio = float(max(0.0, min(occ_conf_ratio, 2.0)))
            occ_mask = occ_mask & (conf >= (conf_thresh * occ_conf_ratio))

        try:
            free_require_conf = int(os.environ.get("HABITAT_FREE_REQUIRE_CONF", "1")) != 0
        except Exception:
            free_require_conf = True
        try:
            free_conf_ratio = float(os.environ.get("HABITAT_FREE_CONF_RATIO", "0.8"))
        except Exception:
            free_conf_ratio = 0.8
        free_conf_ratio = float(max(0.0, min(free_conf_ratio, 2.0)))

        try:
            free_occ_veto_abs = float(os.environ.get("HABITAT_FREE_OCC_VETO_ABS", "3.0"))
        except Exception:
            free_occ_veto_abs = 3.0
        free_occ_veto_abs = float(max(0.0, min(free_occ_veto_abs, 50.0)))
        try:
            free_occ_veto_vert = float(os.environ.get("HABITAT_FREE_OCC_VETO_VERT", "2.0"))
        except Exception:
            free_occ_veto_vert = 2.0
        free_occ_veto_vert = float(max(0.0, min(free_occ_veto_vert, 50.0)))

        occ_vert_for_free = self.occ_mid_count + self.occ_high_count
        try:
            free_strong_clear_margin = float(os.environ.get("HABITAT_FREE_STRONG_CLEAR_MARGIN", "1.0"))
        except Exception:
            free_strong_clear_margin = 1.0
        free_strong_clear_margin = float(max(0.0, min(free_strong_clear_margin, 50.0)))

        prev_map = self.global_map.copy()
        free_allow = (self.occ_count <= free_occ_veto_abs) | (diff >= (free_thresh + free_strong_clear_margin))
        free_mask = (diff > free_thresh) & free_allow & (occ_vert_for_free <= free_occ_veto_vert)
        if free_require_conf:
            free_mask = free_mask & (conf >= (conf_thresh * free_conf_ratio))
        self.global_map[free_mask] = 255
        self.global_map[occ_mask] = 0
        mid_mask = (diff >= -occ_thresh) & (diff <= free_thresh)
        try:
            weak_diff = float(max(float(free_thresh), float(occ_thresh)))
        except Exception:
            weak_diff = float(free_thresh)
        weak = np.abs(diff) <= float(weak_diff)
        unk_mask = (mid_mask | ((conf < conf_thresh) & weak)) & (~free_mask) & (~occ_mask) & (prev_map == 127)
        self.global_map[unk_mask] = 127

        try:
            occ_dilate_iters = int(os.environ.get("HABITAT_OCC_DILATE_ITERS", "1"))
        except Exception:
            occ_dilate_iters = 1
        occ_dilate_iters = int(max(0, min(occ_dilate_iters, 5)))
        if occ_dilate_iters > 0:
            try:
                occ_dilate_conf_ratio = float(os.environ.get("HABITAT_OCC_DILATE_CONF_RATIO", "1.3"))
            except Exception:
                occ_dilate_conf_ratio = 1.3
            occ_dilate_conf_ratio = float(max(0.0, min(occ_dilate_conf_ratio, 10.0)))
            try:
                occ_dilate_diff_margin = float(os.environ.get("HABITAT_OCC_DILATE_DIFF_MARGIN", "0.5"))
            except Exception:
                occ_dilate_diff_margin = 0.5
            occ_dilate_diff_margin = float(max(-10.0, min(occ_dilate_diff_margin, 10.0)))
            try:
                occ_bin = (self.global_map == 0).astype(np.uint8)
                if np.any(occ_bin):
                    kernel = np.ones((3, 3), dtype=np.uint8)
                    occ_d = cv2.dilate(occ_bin, kernel, iterations=int(occ_dilate_iters))
                    cand = (occ_d > 0) & (self.global_map != 0)
                    if np.any(cand):
                        gate_conf = conf < (conf_thresh * float(occ_dilate_conf_ratio))
                        gate_diff = diff < (float(free_thresh) + float(occ_dilate_diff_margin))
                        fill = cand & gate_conf & gate_diff
                        if np.any(fill):
                            self.global_map[fill] = 0
            except Exception:
                pass

        try:
            free_thin_clear_iters = int(os.environ.get("HABITAT_FREE_THIN_CLEAR_ITERS", "1"))
        except Exception:
            free_thin_clear_iters = 1
        free_thin_clear_iters = int(max(0, min(free_thin_clear_iters, 5)))
        if free_thin_clear_iters > 0:
            try:
                free_thin_conf_ratio = float(os.environ.get("HABITAT_FREE_THIN_CONF_RATIO", "1.4"))
            except Exception:
                free_thin_conf_ratio = 1.4
            free_thin_conf_ratio = float(max(0.0, min(free_thin_conf_ratio, 10.0)))
            try:
                free_thin_diff_margin = float(os.environ.get("HABITAT_FREE_THIN_DIFF_MARGIN", "0.5"))
            except Exception:
                free_thin_diff_margin = 0.5
            free_thin_diff_margin = float(max(-10.0, min(free_thin_diff_margin, 10.0)))
            try:
                free_bin = (self.global_map == 255).astype(np.uint8)
                if np.any(free_bin):
                    kernel = np.ones((3, 3), dtype=np.uint8)
                    free_er = cv2.erode(free_bin, kernel, iterations=int(free_thin_clear_iters))
                    free_op = cv2.dilate(free_er, kernel, iterations=int(free_thin_clear_iters))
                    thin = (free_bin > 0) & (free_op == 0) & (self.global_map != 0)
                    if np.any(thin):
                        gate_conf = conf < (conf_thresh * float(free_thin_conf_ratio))
                        gate_diff = diff < (float(free_thresh) + float(free_thin_diff_margin))
                        thin_apply = thin & gate_conf & gate_diff
                        if np.any(thin_apply):
                            self.global_map[thin_apply] = 127
            except Exception:
                pass

        try:
            speckle_enable = int(os.environ.get("HABITAT_OCC_SPECKLE_CLEAR_ENABLE", "1")) != 0
        except Exception:
            speckle_enable = True
        if speckle_enable:
            try:
                occ_pix = (self.global_map == 0)
                if np.any(occ_pix):
                    free_bin = (self.global_map == 255).astype(np.uint8)
                    free_sum = cv2.filter2D(free_bin, ddepth=-1, kernel=np.ones((3, 3), dtype=np.uint8), borderType=cv2.BORDER_CONSTANT)
                    occ_vert_for_speckle = self.occ_mid_count + self.occ_high_count
                    speckle = occ_pix & (free_sum >= 6) & (occ_vert_for_speckle < 1.0) & (conf < (conf_thresh * 1.2))
                    if np.any(speckle):
                        speckle_to_free = speckle & (diff > 0.5) & (conf >= (conf_thresh * 0.7))
                        self.global_map[speckle] = 127
                        self.global_map[speckle_to_free] = 255
            except Exception:
                pass

        try:
            occ_close_enable = int(os.environ.get("HABITAT_OCC_CLOSE_ENABLE", "0")) != 0
        except Exception:
            occ_close_enable = False
        if occ_close_enable:
            try:
                close_iters = int(os.environ.get("HABITAT_OCC_CLOSE_ITERS", "1"))
            except Exception:
                close_iters = 1
            close_iters = int(max(0, min(close_iters, 3)))
            try:
                conf_ratio = float(os.environ.get("HABITAT_OCC_CLOSE_CONF_RATIO", "0.7"))
            except Exception:
                conf_ratio = 0.7
            conf_ratio = float(max(0.0, min(conf_ratio, 1.0)))
            occ_seed = (occ_mask & (conf >= conf_thresh)).astype(np.uint8)
            if close_iters > 0 and np.any(occ_seed):
                kernel_close = np.ones((3, 3), dtype=np.uint8)
                occ_closed = cv2.morphologyEx(occ_seed, cv2.MORPH_CLOSE, kernel_close, iterations=close_iters)
                occ_apply = (occ_closed > 0) & (conf >= (conf_thresh * conf_ratio))
                self.global_map[occ_apply] = 0

        try:
            free_close_enable = int(os.environ.get("HABITAT_FREE_CLOSE_ENABLE", "0")) != 0
        except Exception:
            free_close_enable = False
        if free_close_enable:
            try:
                free_close_iters = int(os.environ.get("HABITAT_FREE_CLOSE_ITERS", "1"))
            except Exception:
                free_close_iters = 1
            free_close_iters = int(max(0, min(free_close_iters, 3)))
            try:
                free_conf_ratio = float(os.environ.get("HABITAT_FREE_CLOSE_CONF_RATIO", "0.7"))
            except Exception:
                free_conf_ratio = 0.7
            free_conf_ratio = float(max(0.0, min(free_conf_ratio, 1.0)))
            free_seed = ((diff > free_thresh) & (conf >= conf_thresh) & (self.occ_count <= free_occ_veto_abs) & (occ_vert_for_free <= free_occ_veto_vert) & (self.global_map != 0)).astype(np.uint8)
            if free_close_iters > 0 and np.any(free_seed):
                kernel_close = np.ones((3, 3), dtype=np.uint8)
                free_closed = cv2.morphologyEx(free_seed, cv2.MORPH_CLOSE, kernel_close, iterations=free_close_iters)
                free_apply = (free_closed > 0) & (conf >= (conf_thresh * free_conf_ratio)) & (diff > 0.0) & (self.global_map != 0)
                self.global_map[free_apply] = 255

        try:
            escape_enable = int(os.environ.get("HABITAT_ESCAPE_TRAP_ENABLE", "1")) != 0
        except Exception:
            escape_enable = True
        if escape_enable and agent_base_state is not None:
            try:
                rx = float(agent_base_state.position[0])
                rz = float(agent_base_state.position[2])
                mx0 = int((rx - self.origin_x) / self.cell_size)
                my0 = int((rz - self.origin_y) / self.cell_size)
                h_map, w_map = self.global_map.shape
                if 1 <= mx0 < (w_map - 1) and 1 <= my0 < (h_map - 1):
                    neigh = self.global_map[my0 - 1 : my0 + 2, mx0 - 1 : mx0 + 2]
                    occ_neigh = (neigh == 0)
                    occ_neigh[1, 1] = False
                    if bool(np.all(occ_neigh)):
                        try:
                            esc_conf_ratio = float(os.environ.get("HABITAT_ESCAPE_CONF_RATIO", "1.5"))
                        except Exception:
                            esc_conf_ratio = 1.5
                        esc_conf_ratio = float(max(0.5, min(esc_conf_ratio, 10.0)))
                        conf_lim = float(conf_thresh) * float(esc_conf_ratio)
                        try:
                            esc_release_k = int(os.environ.get("HABITAT_ESCAPE_RELEASE_K", "2"))
                        except Exception:
                            esc_release_k = 2
                        esc_release_k = int(max(1, min(esc_release_k, 6)))

                        cand = []
                        for dy in (-1, 0, 1):
                            for dx in (-1, 0, 1):
                                if dx == 0 and dy == 0:
                                    continue
                                x = mx0 + dx
                                y = my0 + dy
                                if self.global_map[y, x] != 0:
                                    continue
                                c = float(conf[y, x])
                                if not np.isfinite(c) or c > conf_lim:
                                    continue
                                d = float(self.occ_count[y, x]) - float(self.free_count[y, x])
                                cand.append((d, c, x, y))

                        if cand:
                            cand.sort(key=lambda t: (t[0], t[1]))
                            for _, _, x, y in cand[:esc_release_k]:
                                self.global_map[y, x] = 127
                                self.occ_count[y, x] *= 0.5
            except Exception:
                pass

        occ_bin = (self.global_map == 0).astype(np.uint8)
        if np.any(occ_bin):
            try:
                cc_min_area = int(os.environ.get("HABITAT_OCC_CC_MIN_AREA_3D", os.environ.get("HABITAT_OCC_CC_MIN_AREA_2D", "9")))
            except Exception:
                cc_min_area = 9
            cc_min_area = int(max(0, min(cc_min_area, 10_000_000)))
            if cc_min_area > 1:
                num, labels, stats, _ = cv2.connectedComponentsWithStats(occ_bin, connectivity=8)
                if int(num) > 1:
                    areas = stats[1:, cv2.CC_STAT_AREA]
                    for i in range(1, int(num)):
                        if int(areas[i - 1]) < cc_min_area:
                            self.global_map[labels == i] = 127
            else:
                kernel = np.ones((3, 3), dtype=np.uint8)
                neigh = cv2.filter2D(occ_bin, -1, kernel, borderType=cv2.BORDER_CONSTANT)
                remove = (occ_bin > 0) & (neigh < 2)
                if np.any(remove):
                    self.global_map[remove] = 127

        try:
            edge_seal = int(os.environ.get("HABITAT_MAP_EDGE_SEAL_CELLS", "2"))
        except Exception:
            edge_seal = 2
        edge_seal = int(max(0, min(edge_seal, 64)))
        if edge_seal > 0:
            try:
                h_map, w_map = self.global_map.shape
                m = self.global_map
                top = m[:edge_seal, :]
                top[top != 0] = 127
                bot = m[max(0, h_map - edge_seal) : h_map, :]
                bot[bot != 0] = 127
                left = m[:, :edge_seal]
                left[left != 0] = 127
                right = m[:, max(0, w_map - edge_seal) : w_map]
                right[right != 0] = 127
            except Exception:
                pass

        try:
            keep_conn = int(os.environ.get("HABITAT_FREE_KEEP_CONNECTED_ENABLE", "1")) != 0
        except Exception:
            keep_conn = True
        if keep_conn and agent_base_state is not None:
            try:
                rx = float(agent_base_state.position[0])
                rz = float(agent_base_state.position[2])
                mx0 = int(round((rx - float(self.origin_x)) / float(self.cell_size)))
                my0 = int(round((rz - float(self.origin_y)) / float(self.cell_size)))
                h_map, w_map = self.global_map.shape
                if 0 <= mx0 < w_map and 0 <= my0 < h_map:
                    free_bin = (self.global_map == 255).astype(np.uint8)
                    if int(free_bin[my0, mx0]) != 0:
                        try:
                            max_drop_area = int(os.environ.get("HABITAT_FREE_KEEP_CONNECTED_MAX_DROP_AREA", "600"))
                        except Exception:
                            max_drop_area = 600
                        max_drop_area = int(max(0, min(max_drop_area, int(h_map) * int(w_map))))
                        try:
                            drop_edge = int(os.environ.get("HABITAT_FREE_DROP_EDGE_COMPONENTS", "1")) != 0
                        except Exception:
                            drop_edge = True
                        try:
                            drop_fill = int(os.environ.get("HABITAT_FREE_DROP_FILL", "127"))
                        except Exception:
                            drop_fill = 127
                        drop_fill = 0 if int(drop_fill) <= 0 else 127

                        num, labels, stats, _ = cv2.connectedComponentsWithStats(free_bin, connectivity=8)
                        if int(num) > 1:
                            keep_label = int(labels[my0, mx0])
                            if keep_label > 0:
                                for lbl in range(1, int(num)):
                                    if int(lbl) == int(keep_label):
                                        continue
                                    area = int(stats[int(lbl), cv2.CC_STAT_AREA])
                                    left = int(stats[int(lbl), cv2.CC_STAT_LEFT])
                                    top = int(stats[int(lbl), cv2.CC_STAT_TOP])
                                    width = int(stats[int(lbl), cv2.CC_STAT_WIDTH])
                                    height = int(stats[int(lbl), cv2.CC_STAT_HEIGHT])
                                    touches_edge = (left <= 0) or (top <= 0) or ((left + width) >= int(w_map)) or ((top + height) >= int(h_map))
                                    if (bool(drop_edge) and bool(touches_edge)) or (int(area) <= int(max_drop_area)):
                                        self.global_map[labels == int(lbl)] = int(drop_fill)
            except Exception:
                pass
        
        return self.global_map
