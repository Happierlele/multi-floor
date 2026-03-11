import time

import numpy as np
import torch
import matplotlib.pyplot as plt
import copy
import quads

from utils import *
from parameter_clean import *
from node_manager import NodeManager


class Agent:
    def __init__(self, policy_net, device='cpu', plot=False):
        self.device = device
        self.policy_net = policy_net
        self.plot = plot

        # location and map
        self.location = None
        self.map_info = None

        # map related parameters
        self.cell_size = CELL_SIZE
        self.node_resolution = NODE_RESOLUTION 
        self.updating_map_size = UPDATING_MAP_SIZE

        # map and updating map
        self.map_info = None
        self.updating_map_info = None

        # frontiers
        self.frontier = set()

        # node managers
        self.node_manager = NodeManager(plot=self.plot)

        # graph
        self.node_coords, self.utility, self.guidepost = None, None, None
        self.current_index, self.adjacent_matrix, self.neighbor_indices = None, None, None
    def reset_for_new_map(self, preserve_memory=False):
        self.location = None
        self.map_info = None
        self.updating_map_info = None
        self.frontier = set()
        if not preserve_memory:
            self.node_manager = NodeManager(plot=self.plot)
        else:
            self.node_manager.reset_frontier()
        self.node_coords, self.utility, self.guidepost = None, None, None
        self.current_index, self.adjacent_matrix, self.neighbor_indices = None, None, None

    def update_map(self, map_info):
        # no need in training because of shallow copy
        self.map_info = map_info

    def update_updating_map(self, location):
        self.updating_map_info = self.get_updating_map(location)

    def update_location(self, location):
        self.location = location
        # Use rounded key to match NodeManager's insertion logic
        key = (round(float(location[0]), 1), round(float(location[1]), 1))
        node = self.node_manager.nodes_dict.find(key)
        
        # Fallback: if exact rounded key not found, try fuzzy search (though update_graph should have added it)
        if node is None:
             bbox_size = 3.0 # Increased to match NodeManager's 3.0m threshold (was 2.2)
             found_nodes = self.node_manager.nodes_dict.within_bb(quads.BoundingBox(
                 min_x=location[0]-bbox_size, min_y=location[1]-bbox_size,
                 max_x=location[0]+bbox_size, max_y=location[1]+bbox_size
             ))
             if found_nodes:
                 node = min(found_nodes, key=lambda n: np.linalg.norm(np.array(n.data.coords) - location))
                 # Verify distance is within reasonable "snap" range
                 if np.linalg.norm(np.array(node.data.coords) - location) > 2.5:
                     node = None # Too far, effectively off-grid but skipped by manager? Should not happen if logic aligns.
        
        if self.node_manager.nodes_dict.__len__() == 0:
            pass
        elif node is not None:
            node.data.set_visited()
        else:
            # Safe fallback if node really doesn't exist (prevent crash)
            pass

    def update_frontiers(self):
        self.frontier = get_frontier_in_map(self.updating_map_info)

    def update_graph(self, location, floor_id=0):
        self.node_manager.update_graph(location,
                                       self.frontier,
                                       self.updating_map_info,
                                       self.map_info,
                                       floor_id)
        
    def get_updating_map(self, location):
        # 局部更新地图包含了所有可能需要更新状态的节点
        # 计算局部地图的左下角坐标（原点）：当前位置减去更新尺寸的一半
        updating_map_origin_x = (location[
                                  0] - self.updating_map_size / 2)
        updating_map_origin_y = (location[
                                  1] - self.updating_map_size / 2)

        # 计算局部地图的右上角坐标（顶点）：原点加上更新尺寸
        updating_map_top_x = updating_map_origin_x + self.updating_map_size
        updating_map_top_y = updating_map_origin_y + self.updating_map_size

        # 获取全局地图的边界范围
        min_x = self.map_info.map_origin_x
        min_y = self.map_info.map_origin_y
        max_x = (self.map_info.map_origin_x + self.cell_size * (self.map_info.map.shape[1] - 1))
        max_y = (self.map_info.map_origin_y + self.cell_size * (self.map_info.map.shape[0] - 1))

        # 边界检查：如果局部地图超出了全局地图边界，则将其截断
        if updating_map_origin_x < min_x:
            updating_map_origin_x = min_x
        if updating_map_origin_y < min_y:
            updating_map_origin_y = min_y
        if updating_map_top_x > max_x:
            updating_map_top_x = max_x
        if updating_map_top_y > max_y:
            updating_map_top_y = max_y

        # 将物理坐标对齐到栅格地图的单元格边界（取整操作）
        updating_map_origin_x = (updating_map_origin_x // self.cell_size + 1) * self.cell_size
        updating_map_origin_y = (updating_map_origin_y // self.cell_size + 1) * self.cell_size
        updating_map_top_x = (updating_map_top_x // self.cell_size) * self.cell_size
        updating_map_top_y = (updating_map_top_y // self.cell_size) * self.cell_size

        # 四舍五入保留一位小数，避免浮点数精度问题
        updating_map_origin_x = np.round(updating_map_origin_x, 1)
        updating_map_origin_y = np.round(updating_map_origin_y, 1)
        updating_map_top_x = np.round(updating_map_top_x, 1)
        updating_map_top_y = np.round(updating_map_top_y, 1)

        # 将对齐后的物理坐标转换为全局地图中的数组索引
        updating_map_origin = np.array([updating_map_origin_x, updating_map_origin_y])
        updating_map_origin_in_global_map = get_cell_position_from_coords(updating_map_origin, self.map_info)

        updating_map_top = np.array([updating_map_top_x, updating_map_top_y])
        updating_map_top_in_global_map = get_cell_position_from_coords(updating_map_top, self.map_info)

        # 从全局地图中切片提取出局部地图数据
        # 注意：这里使用了 numpy 的切片操作，extracted_map 是一个新的数组视图
        updating_map = self.map_info.map[
                    updating_map_origin_in_global_map[1]:updating_map_top_in_global_map[1]+1,
                    updating_map_origin_in_global_map[0]:updating_map_top_in_global_map[0]+1]

        # 封装成 MapInfo 对象返回
        updating_map_info = MapInfo(updating_map, updating_map_origin_x, updating_map_origin_y, self.cell_size)

        return updating_map_info

    def update_planning_state(self, map_info, location, floor_id=0):
        # 1. 更新全局地图信息
        self.update_map(map_info)
        # 2. 更新机器人当前位置，并标记当前节点为已访问
        self.update_location(location)
        # 3. 更新局部更新地图（提取当前位置周围的局部地图区域）
        self.update_updating_map(self.location)
        # 4. 更新前沿点（Frontiers），即已知自由区域与未知区域的交界处
        self.update_frontiers()
        # 5. 更新拓扑图结构：
        #    - 添加新节点
        #    - 计算节点效用（Utility）
        #    - 更新邻居关系
        self.update_graph(self.location, floor_id)
        # Debug log
        if self.node_manager.nodes_dict.__len__() < 5:
             print(f"[Agent] Graph updated. Nodes: {[n.data.coords for n in self.node_manager.nodes_dict.__iter__()]}")
        # 6. 更新观测数据：
        #    - 提取所有节点的坐标、效用、引导标志
        #    - 构建邻接矩阵
        #    - 确定当前节点索引和邻居索引
        self.node_coords, self.utility, self.guidepost, self.adjacent_matrix, self.current_index, self.neighbor_indices = \
            self.update_observation()

    def update_observation(self):
        # 初始化列表，用于存储所有节点的坐标
        all_node_coords = []
        all_nodes_data = [] # Store actual node objects to avoid re-lookup
        # 遍历节点管理器中的所有节点，收集坐标
        for node in self.node_manager.nodes_dict.__iter__():
            all_node_coords.append(node.data.coords)
            all_nodes_data.append(node.data)
        # 将坐标列表转换为 numpy 数组，形状为 (N, 2)
        all_node_coords = np.array(all_node_coords).reshape(-1, 2)
        
        # 初始化列表，用于存储节点的属性
        utility = [] # 效用值
        guidepost = [] # 引导标志（访问状态）

        # 获取节点总数
        n_nodes = all_node_coords.shape[0]
        # 初始化邻接矩阵，全为1（表示初始化或无连接），之后连接的设为0
        # 注意：这里的逻辑是 0 表示有连接（距离为0或掩码），1 表示无连接
        adjacent_matrix = np.ones((n_nodes, n_nodes)).astype(int)
        
        # 将节点坐标转换为复数形式，利用复数的唯一性进行快速索引查找
        # x + yi 的形式
        node_coords_to_check = all_node_coords[:, 0] + all_node_coords[:, 1] * 1j
        
        # 遍历所有节点，提取属性并构建邻接矩阵
        for i, coords in enumerate(all_node_coords):
            # 直接使用已获取的节点对象，避免重复查找导致的精度问题
            node = all_nodes_data[i]
            
            # 收集效用值
            utility.append(node.utility / (1 + node.visit_count))
            # 收集访问状态
            guidepost.append(node.visited)
            # 遍历该节点的邻居集合
            for neighbor in node.neighbor_set:
                # 查找邻居节点在 all_node_coords 中的索引
                # 使用模糊匹配或 rounded key 匹配可能更稳健，但目前保持原有逻辑，
                # 假设 neighbor_set 中的坐标与 node.coords 来源一致
                neighbor_complex = neighbor[0] + neighbor[1] * 1j
                
                # 尝试精确匹配
                indices = np.argwhere(node_coords_to_check == neighbor_complex)
                
                if len(indices) == 0:
                    # 如果精确匹配失败（可能是浮点误差），尝试最小距离匹配
                    dists = np.abs(node_coords_to_check - neighbor_complex)
                    min_dist_idx = np.argmin(dists)
                    if dists[min_dist_idx] < 0.1: # 允许 10cm 的误差
                        index = min_dist_idx
                    else:
                        continue # 找不到对应的邻居节点，跳过
                else:
                    index = indices[0][0]

                # 在邻接矩阵中标记连接关系（设为0）
                adjacent_matrix[i, index] = 0

        # 将列表转换为 numpy 数组
        utility = np.array(utility)
        guidepost = np.array(guidepost)

        # 找到机器人当前所在位置对应的节点索引
        # FIX: Handle case where robot location is not in node_coords_to_check
        # This can happen if the robot is slightly off-grid or at initialization
        robot_loc_complex = self.location[0] + self.location[1] * 1j
        
        if n_nodes == 0:
            print("Warning: No nodes in graph! Creating dummy node at robot location.")
            all_node_coords = np.array([self.location]).reshape(-1, 2)
            utility = np.array([0.0])
            guidepost = np.array([0])
            adjacent_matrix = np.zeros((1, 1), dtype=int) 
            adjacent_matrix[0, 0] = 0 # Connected to self
            current_index = 0
            neighbor_indices = np.array([0])
            return all_node_coords, utility, guidepost, adjacent_matrix, current_index, neighbor_indices
            
        match_indices = np.argwhere(node_coords_to_check == robot_loc_complex)
        
        if len(match_indices) > 0:
            current_index = match_indices[0][0]
        else:
            # Fallback: Find the closest node
            # This is robust against floating point errors or slight drifts
            dists = np.abs(node_coords_to_check - robot_loc_complex)
            current_index = np.argmin(dists)
            # Optional: Warning if distance is too large
            if dists[current_index] > 4.1: # Relaxed threshold (NODE_RESOLUTION=4.0)
                print(f"Warning: Robot at {self.location} is far from nearest node at index {current_index} (dist={dists[current_index]:.2f})")

        # 找出当前节点的所有邻居节点的索引（在邻接矩阵中值为0的位置）
        neighbor_indices = np.argwhere(adjacent_matrix[current_index] == 0).reshape(-1)
        
        # 返回更新后的所有观测数据
        return all_node_coords, utility, guidepost, adjacent_matrix, current_index, neighbor_indices

    def select_next_waypoint(self, observation):
        # 提取观测数据中的 current_edge，这是当前节点连接的邻居节点的索引列表
        _, _, _, _, current_edge, _ = observation
        
        # 使用策略网络（policy_net）进行前向传播，计算动作的对数概率
        # torch.no_grad() 表示不进行梯度计算，因为这是推理过程
        with torch.no_grad():
            logp = self.policy_net(*observation)

        # --- 启发式修正逻辑开始 ---
        # 目的：防止机器人在角落或死胡同徘徊
        # 策略：对“访问次数过多”或“距离过近”的邻居节点降权
        
        # 1. 获取所有邻居节点的全局索引
        # current_edge shape: (1, num_neighbors, 2) -> (num_neighbors,)
        if current_edge.shape[1] == 0:
            print("[Agent] No neighbors in current_edge! Attempting to force-connect to nearest node...")
            # Fallback: Find nearest node in node_coords
            if self.node_coords is not None and len(self.node_coords) > 0:
                dists = np.linalg.norm(self.node_coords - self.location, axis=1)
                nearest_idx = np.argmin(dists)
                print(f"[Agent] Forcing connection to nearest node {self.node_coords[nearest_idx]} (dist={dists[nearest_idx]:.2f})")
                return self.node_coords[nearest_idx], torch.tensor(0).long().to(logp.device)
            return None # No valid neighbors

        neighbor_indices = current_edge[0, :, 0].cpu().numpy()
        
        # 2. 获取每个邻居节点的 visit_count 和 距离
        visit_counts = []
        dists = []
        
        current_pos = self.location
        
        for idx in neighbor_indices:
            # Check if idx is valid
            if idx >= len(self.node_coords):
                visit_counts.append(0)
                dists.append(0)
                continue

            coords = self.node_coords[idx]
            # 查找节点对象
            # 使用与 ground_truth_node_manager 相同的模糊查找逻辑
            node_obj = self.node_manager.nodes_dict.find((coords[0], coords[1]))
            if node_obj is None:
                 # Fallback: bbox search
                 bbox_size = 0.1
                 found_nodes = self.node_manager.nodes_dict.within_bb(quads.BoundingBox(
                    min_x=coords[0]-bbox_size, min_y=coords[1]-bbox_size,
                    max_x=coords[0]+bbox_size, max_y=coords[1]+bbox_size
                ))
                 if found_nodes: node_obj = found_nodes[0]
            
            if node_obj:
                node = node_obj.data
                # 获取访问次数 (如果没有该属性默认为0)
                count = getattr(node, "visit_count", 0)
                visit_counts.append(count)
            else:
                 visit_counts.append(0)

            # 计算距离 (欧氏距离)
            d = np.linalg.norm(coords - current_pos)
            dists.append(d)
            
        visit_counts = np.array(visit_counts)
        dists = np.array(dists)
        
        # 3. 计算启发式权重
        # 参数设定 (可调整)
        alpha = 1.0  # Increased from 0.5 to prefer farther nodes
        beta = 5.0   # Increased from 1.0 to heavily penalize re-visiting nodes (anti-stuck)
        
        # 归一化距离 (可选，防止距离数值过大)
        # 这里假设 NODE_RESOLUTION = 4.0，距离一般在 0~10 之间
        # dists_norm = dists / 4.0 
        
        # 公式：越远越好，访问越少越好
        heuristic_weights = (1.0 + alpha * dists) / (1.0 + beta * visit_counts)
        
        # 转换为 tensor 并归一化
        heuristic_weights = torch.FloatTensor(heuristic_weights).to(logp.device)
        # 防止全0
        if heuristic_weights.sum() == 0:
            heuristic_weights = torch.ones_like(heuristic_weights)
            
        # 4. 融合概率
        # logp.exp() 是网络给出的原始概率
        probs = logp.exp().squeeze(0) # shape: (num_neighbors,)
        
        # 确保 probs 和 weights 长度一致 (防止 mask 导致的长度不匹配)
        # 注意：policy_net 输出的 logp 长度应该等于 valid neighbors 的数量
        if probs.shape[0] == heuristic_weights.shape[0]:
            # 乘法融合
            combined_probs = probs * heuristic_weights
            # 重新归一化
            if combined_probs.sum() > 0:
                combined_probs = combined_probs / combined_probs.sum()
            else:
                combined_probs = probs # 回退到原始概率
        else:
            # 如果长度不一致 (极少情况，除非 observation mask 处理有误)，则不修正
            combined_probs = probs
            
        # --- 启发式修正逻辑结束 ---

        # 根据概率分布采样一个动作
        # logp.exp() 将对数概率转换为概率
        # torch.multinomial(..., 1) 根据概率分布进行随机采样，返回采样的索引
        # 这个索引对应于 current_edge 中的第几个邻居
        # action_index = torch.multinomial(logp.exp(), 1).long().squeeze(1) # 原代码
        
        # 使用修正后的概率采样
        action_index = torch.multinomial(combined_probs.unsqueeze(0), 1).long().squeeze(1)
        
        # 根据采样到的 action_index，从 current_edge 中获取实际的下一个节点在全局图中的索引
        # current_edge 存储的是邻居节点在所有节点列表中的索引
        next_node_index = current_edge[0, action_index.item(), 0].item()
        
        # 根据全局索引获取下一个目标点的坐标
        next_position = self.node_coords[next_node_index]

        # 返回下一个目标点的坐标和动作索引
        return next_position, action_index

    def get_observation(self):
        # 构建策略网络输入，与 GroundTruth 观测保持同形状
        node_coords = self.node_coords
        node_utility = self.utility.reshape(-1, 1)
        node_guidepost = self.guidepost.reshape(-1, 1)
        # 没有“已探索”标签，复用访问信息以保持维度
        node_guidepost2 = node_guidepost
        edge_mask = self.adjacent_matrix
        current_index = self.current_index
        neighbor_indices = self.neighbor_indices
        n_node = node_coords.shape[0]

        # 以当前节点为中心归一化坐标
        current_node_coords = node_coords[current_index]
        node_coords = np.concatenate(
            (node_coords[:, 0].reshape(-1, 1) - current_node_coords[0],
             node_coords[:, 1].reshape(-1, 1) - current_node_coords[1]),
            axis=-1
        ) / UPDATING_MAP_SIZE / 2

        # 归一化效用
        node_utility = node_utility / (SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE)

        # 拼接节点特征 (coords(2) + util(1) + expl(1) + visit(1)) => dim 5
        node_inputs = np.concatenate((node_coords, node_utility, node_guidepost, node_guidepost2), axis=1)
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)

        # 节点数不应超过填充上线
        assert node_coords.shape[0] <= NODE_PADDING_SIZE, print(node_coords.shape[0], NODE_PADDING_SIZE)

        # 节点输入零填充到 NODE_PADDING_SIZE
        padding2d = torch.nn.ZeroPad2d((0, 0, 0, NODE_PADDING_SIZE - n_node))
        node_inputs = padding2d(node_inputs)

        # 节点 padding mask: 前 n_node 为 0，后为 1
        node_padding_mask = torch.zeros((1, 1, n_node), dtype=torch.int16).to(self.device)
        node_padding = torch.ones((1, 1, NODE_PADDING_SIZE - n_node), dtype=torch.int16).to(self.device)
        node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)

        # 边掩码及其填充
        edge_mask = torch.tensor(edge_mask).unsqueeze(0).to(self.device)
        pad2d = torch.nn.ConstantPad2d((0, NODE_PADDING_SIZE - n_node, 0, NODE_PADDING_SIZE - n_node), 1)
        edge_mask = pad2d(edge_mask)

        # 当前边列表：确保包含 current_index 自环，便于 edge_padding_mask 标记
        if current_index not in neighbor_indices:
            neighbor_indices = np.concatenate(([current_index], neighbor_indices))
        current_in_edge = np.argwhere(neighbor_indices == current_index)[0][0]
        current_edge = torch.tensor(neighbor_indices).unsqueeze(0)
        k_size = current_edge.size()[-1]
        pad1d = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 0)
        current_edge = pad1d(current_edge)
        current_edge = current_edge.unsqueeze(-1)

        # 边填充掩码：标记当前节点位置
        edge_padding_mask = torch.zeros((1, 1, k_size), dtype=torch.int16).to(self.device)
        edge_padding_mask[0, 0, current_in_edge] = 1
        pad1d_mask = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 1)
        edge_padding_mask = pad1d_mask(edge_padding_mask)

        # 当前索引张量
        current_index_t = torch.tensor([current_index]).reshape(1, 1, 1).to(self.device)

        return [node_inputs, node_padding_mask, edge_mask, current_index_t, current_edge, edge_padding_mask]

    def plot_env(self):
        plt.subplot(2, 2, 1)
        plt.imshow(self.map_info.map, origin='lower')
        plt.plot(self.location[0] / self.cell_size - self.map_info.map_origin_x / self.cell_size,
                 self.location[1] / self.cell_size - self.map_info.map_origin_y / self.cell_size, 'rx')
        plt.title("Robot Planning")
        # plt.show() # Removed to prevent blocking and allow subplot composition

    def get_rgb_image(self):
        """
        Generate a synthetic RGB image from the robot's current local map view.
        Since we are in a 2D map environment without a camera, we visualize the
        top-down map as what the VLM 'sees'.
        """
        if self.updating_map_info is None or self.updating_map_info.map is None:
            # Return a blank black image if no map available
            return np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Get local map (grid values: 255=Free, 1=Occupied, 127=Unknown)
        local_map = self.updating_map_info.map
        
        # Ensure it is uint8
        img_gray = local_map.astype(np.uint8)
        
        # Stack to 3 channels to simulate RGB
        img_rgb = np.stack((img_gray,)*3, axis=-1)
        
        return img_rgb
