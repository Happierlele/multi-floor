import os
import matplotlib.pyplot as plt
from skimage import io
from skimage.measure import block_reduce
from copy import deepcopy

from sensor import sensor_work
from utils import *


class Env:
    def __init__(self, episode_index, plot=False, floor_id=0):
        self.episode_index = episode_index
        self.plot = plot
        self.floor_id = floor_id # Explicit floor ID
        self.ground_truth, self.robot_cell, self.map_list, self.map_index, self.stairs_cells = self.import_ground_truth(episode_index)
        self.ground_truth_size = np.shape(self.ground_truth)  # cell
        self.cell_size = CELL_SIZE  # meter

        self.robot_location = np.array([0.0, 0.0])  # meter

        self.robot_belief = np.ones(self.ground_truth_size) * 127
        self.belief_origin_x = -np.round(self.robot_cell[0] * self.cell_size, 1)   # meter
        self.belief_origin_y = -np.round(self.robot_cell[1] * self.cell_size, 1)  # meter

        self.global_frontiers = set()

        self.sensor_range = SENSOR_RANGE  # meter
        self.travel_dist = 0  # meter
        self.explored_rate = 0

        self.robot_belief = sensor_work(self.robot_cell, self.sensor_range / self.cell_size, self.robot_belief,
                                        self.ground_truth)
        self.old_belief = deepcopy(self.robot_belief)

        self.belief_info = MapInfo(self.robot_belief, self.belief_origin_x, self.belief_origin_y, self.cell_size)

        self.ground_truth_info = MapInfo(self.ground_truth, self.belief_origin_x, self.belief_origin_y, self.cell_size)
        self.stairs_coords_list = []
        self.stairs_cell = None
        self.stairs_coords = None
        self.current_target_stairs_index = None
        self.discovered_stairs = set()
        if self.stairs_cells is not None:
            for cell in self.stairs_cells:
                coords = get_coords_from_cell_position(np.array(cell), self.ground_truth_info)
                self.stairs_coords_list.append(coords)
            if len(self.stairs_cells) > 0:
                self.stairs_cell = self.stairs_cells[0]
                self.stairs_coords = self.stairs_coords_list[0]

        if self.plot:
            self.frame_files = []
            self.trajectory_x = [self.robot_location[0]]
            self.trajectory_y = [self.robot_location[1]]

    def import_ground_truth(self, episode_index):
        map_dir = f'maps'
        map_list = os.listdir(map_dir)
        map_list.sort() # Ensure deterministic order
        if len(map_list) == 0:
            raise ValueError("No maps found in 'maps' directory")
        start_index = episode_index % np.size(map_list)
        map_index = start_index
        ground_truth_raw = None
        robot_cell = None
        stairs_cells = None
        while True:
            # Load image. If it's uint8, normalize to 0-1 float to match expected behavior
            print(f"Loading map index {map_index}: {map_list[map_index]}", flush=True)
            image = io.imread(map_dir + '/' + map_list[map_index], 1)
            if image.dtype == np.uint8:
                image = image / 255.0
            ground_truth_raw = (image * 255).astype(int)
            
            ground_truth_tmp = block_reduce(ground_truth_raw, (2, 2), np.min)
            robot_indices = np.nonzero(ground_truth_tmp == 208)
            if np.array(robot_indices).shape[1] > 0:
                break
            map_index = (map_index + 1) % np.size(map_list)
            if map_index == start_index:
                raise ValueError("No map contains a robot start cell with value 208")

        ground_truth = ground_truth_tmp

        robot_indices = np.nonzero(ground_truth == 208)
        robot_cell = np.array([np.array(robot_indices)[1, 0], np.array(robot_indices)[0, 0]])
        stairs_indices = np.nonzero(ground_truth == STAIRS_VALUE)
        if np.array(stairs_indices).shape[1] > 0:
            xs = np.array(stairs_indices)[1]
            ys = np.array(stairs_indices)[0]
            stairs_cells = np.stack((xs, ys), axis=-1).tolist()

        ground_truth = (ground_truth > 150) | ((ground_truth <= 80) & (ground_truth >= 50))
        ground_truth = ground_truth * 254 + 1

        return ground_truth, robot_cell, map_list, map_index, stairs_cells

    def update_robot_location(self, robot_location):
        self.robot_location = robot_location
        self.robot_cell = np.array([round((robot_location[0] - self.belief_origin_x) / self.cell_size),
                                    round((robot_location[1] - self.belief_origin_y) / self.cell_size)])
        if self.plot:
            self.trajectory_x.append(self.robot_location[0])
            self.trajectory_y.append(self.robot_location[1])

    def update_robot_belief(self):
        self.robot_belief = sensor_work(self.robot_cell, round(self.sensor_range / self.cell_size), self.robot_belief,
                                        self.ground_truth)

    def calculate_reward(self, dist):
        reward = 0
        reward -= dist / UPDATING_MAP_SIZE * 5
        
        global_frontiers = get_frontier_in_map(self.belief_info)
        if len(global_frontiers) == 0:
            delta_num = len(self.global_frontiers)
        else:
            observed_frontiers = self.global_frontiers - global_frontiers
            delta_num = len(observed_frontiers)

        reward += delta_num / (SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE)

        self.global_frontiers = global_frontiers
        self.old_belief = deepcopy(self.robot_belief)

        return reward

    def evaluate_exploration_rate(self):
        self.explored_rate = np.sum(self.robot_belief == 255) / np.sum(self.ground_truth == 255)

    def discover_stairs(self):
        if not hasattr(self, "stairs_coords_list") or self.stairs_coords_list is None:
            return
        
        if not hasattr(self, "discovered_stairs"):
            self.discovered_stairs = set()

        for idx, coords in enumerate(self.stairs_coords_list):
            if idx in self.discovered_stairs:
                continue
            dist = np.linalg.norm(self.robot_location - np.array(coords))
            # Use sensor range or a slightly larger visual range for discovery
            if dist <= self.sensor_range:
                self.discovered_stairs.add(idx)
                print(f"Stairs {idx} discovered at {coords}!")

    def step(self, next_waypoint):
        # 计算移动距离（欧几里得距离）
        dist = np.linalg.norm(self.robot_location - next_waypoint)
        # 更新机器人位置
        self.update_robot_location(next_waypoint)
        # 更新机器人的信念地图（模拟传感器扫描）
        self.update_robot_belief()

        # 累加总行驶距离
        self.travel_dist += dist
        # 计算当前探索率（已知区域占总区域的比例）
        self.evaluate_exploration_rate()

        # 计算这一步的奖励（Reward）
        # 奖励通常包括：+发现新区域，-移动消耗
        reward = self.calculate_reward(dist)

        return reward

    def at_stairs(self):
        if self.stairs_coords is None:
            return False
        return np.linalg.norm(self.robot_location - np.array(self.stairs_coords)) <= STAIRS_DIST_TOLERANCE

    def switch_to_next_map(self):
        if self.map_list is None or len(self.map_list) == 0:
            return
        self.map_index = (self.map_index + 1) % np.size(self.map_list)
        # Increment floor ID
        self.floor_id += 1
        
        self.ground_truth, self.robot_cell, _, _, self.stairs_cells = self.import_ground_truth(self.map_index)
        self.robot_location = np.array([0.0, 0.0])
        self.robot_belief = np.ones(np.shape(self.ground_truth)) * 127
        self.belief_origin_x = -np.round(self.robot_cell[0] * self.cell_size, 1)
        self.belief_origin_y = -np.round(self.robot_cell[1] * self.cell_size, 1)
        self.robot_belief = sensor_work(self.robot_cell, self.sensor_range / self.cell_size, self.robot_belief,
                                        self.ground_truth)
        self.old_belief = deepcopy(self.robot_belief)
        self.travel_dist = 0
        self.explored_rate = 0
        self.belief_info = MapInfo(self.robot_belief, self.belief_origin_x, self.belief_origin_y, self.cell_size)
        self.ground_truth_info = MapInfo(self.ground_truth, self.belief_origin_x, self.belief_origin_y, self.cell_size)
        self.global_frontiers = set()
        self.stairs_coords_list = []
        self.stairs_cell = None
        self.stairs_coords = None
        self.current_target_stairs_index = None
        self.discovered_stairs = set()
        if self.stairs_cells is not None:
            for cell in self.stairs_cells:
                coords = get_coords_from_cell_position(np.array(cell), self.ground_truth_info)
                self.stairs_coords_list.append(coords)
            if len(self.stairs_cells) > 0:
                self.stairs_cell = self.stairs_cells[0]
                self.stairs_coords = self.stairs_coords_list[0]

        if self.plot:
            self.trajectory_x = [self.robot_location[0]]
            self.trajectory_y = [self.robot_location[1]]

    def plot_env(self, step):

        plt.switch_backend('agg')
        plt.subplot(1, 3, 1)
        plt.imshow(self.robot_belief, cmap='gray')
        plt.axis('off')
        plt.plot((self.robot_location[0] - self.belief_origin_x) / self.cell_size,
                 (self.robot_location[1] - self.belief_origin_y) / self.cell_size, 'mo', markersize=4, zorder=5)
        plt.plot((np.array(self.trajectory_x) - self.belief_origin_x) / self.cell_size,
                 (np.array(self.trajectory_y) - self.belief_origin_y) / self.cell_size, 'b', linewidth=2, zorder=1)
        if hasattr(self, "stairs_coords_list") and self.stairs_coords_list is not None:
            for idx, coords in enumerate(self.stairs_coords_list):
                # Only plot if discovered or currently targeted
                is_discovered = (hasattr(self, 'discovered_stairs') and idx in self.discovered_stairs)
                is_target = (hasattr(self, "current_target_stairs_index") and 
                             self.current_target_stairs_index is not None and 
                             idx == self.current_target_stairs_index)
                
                if is_discovered or is_target:
                    x = (coords[0] - self.belief_origin_x) / self.cell_size
                    y = (coords[1] - self.belief_origin_y) / self.cell_size
                    plt.plot(x, y, 'r*', markersize=8, zorder=6)
                    if is_target:
                        plt.plot(x, y, 'ys', markersize=10, zorder=7)
        plt.suptitle('Explored ratio: {:.4g}  Travel distance: {:.4g}'.format(self.explored_rate, self.travel_dist))
        plt.tight_layout()
        # plt.show()
        plt.savefig('{}/{}_{}_samples.png'.format(gifs_path, self.episode_index, step), dpi=150)
        frame = '{}/{}_{}_samples.png'.format(gifs_path, self.episode_index, step)
        plt.close()
        self.frame_files.append(frame)

    def block_stairs_entrance(self, stairs_cell, radius_cells=2):
        if stairs_cell is None:
            return
        x0, y0 = stairs_cell
        h, w = self.robot_belief.shape
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                x = x0 + dx
                y = y0 + dy
                if 0 <= x < w and 0 <= y < h:
                    if self.robot_belief[y, x] == FREE:
                        self.robot_belief[y, x] = OCCUPIED
        self.belief_info.update_map_info(self.robot_belief, self.belief_origin_x, self.belief_origin_y)

    def reset_semantic_map(self):
        # Do NOT clear the map, just clear temporary obstacles if needed, or do nothing.
        # Clearing map causes "amnesia" and graph rebuild from scratch.
        # For now, let's just refresh sensor reading to clear dynamic obstacles?
        # Actually, if we are stuck, we might want to clear local area only?
        # But for this task, clearing everything is too aggressive.
        # Let's KEEP the map but maybe re-evaluate location.
        pass 
        # self.robot_belief = np.ones(self.ground_truth_size) * UNKNOWN
        # self.update_robot_location(self.robot_location)
        # self.robot_belief = sensor_work(self.robot_cell, self.sensor_range / self.cell_size, self.robot_belief, self.ground_truth)
        # self.old_belief = deepcopy(self.robot_belief)
        # self.belief_info.update_map_info(self.robot_belief, self.belief_origin_x, self.belief_origin_y)
        # self.global_frontiers = set()
        # self.explored_rate = 0
