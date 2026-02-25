import torch
import numpy as np

from utils import *
from parameter_clean import *
import quads
import matplotlib.pyplot as plt


class GroundTruthNodeManager:
    def __init__(self, node_manager, ground_truth_map_info, device='cpu', plot=False):
        # 初始化四叉树，用于存储和快速查询节点
        self.nodes_dict = quads.QuadTree((0, 0), 1000, 1000)
        # 引用普通的 node_manager，可能用于获取信念状态下的节点信息
        self.node_manager = node_manager
        # 存储真实地图信息（Ground Truth Map） - 将其设为 None，移除对真值地图的依赖
        self.ground_truth_map_info = None
        # 存储真实地图中的节点坐标
        self.ground_truth_node_coords = None
        # 存储真实地图中的节点效用值
        self.ground_truth_node_utility = None
        # 存储节点的探索状态（是否被探索过）
        self.explored_sign = None
        # 设置计算设备（CPU 或 GPU）
        self.device = device
        # 是否绘图标志
        self.plot = plot

        # 初始化图结构，生成节点并建立连接
        # self.initialize_graph()
        print("DEBUG: GroundTruthNodeManager initialized in BELIEF-ONLY mode.", flush=True)

    def get_ground_truth_observation(self, robot_location):
        all_node_coords = []
        all_nodes = []
        for node in self.node_manager.nodes_dict.__iter__():
            all_node_coords.append(node.data.coords)
            all_nodes.append(node.data)

        if not all_node_coords:
            print("CRITICAL ERROR: No nodes in GroundTruth graph. Returning padded dummy data.", flush=True)
            dummy_dim = 5
            dummy_node_inputs = torch.zeros((1, NODE_PADDING_SIZE, dummy_dim)).to(self.device)
            dummy_node_padding_mask = torch.ones((1, 1, NODE_PADDING_SIZE), dtype=torch.int16).to(self.device)
            dummy_edge_mask = torch.ones((1, NODE_PADDING_SIZE, NODE_PADDING_SIZE)).to(self.device)
            dummy_current_index = torch.zeros((1, 1, 1)).to(self.device)
            dummy_current_edge = torch.zeros((1, K_SIZE, 1)).to(self.device)
            dummy_edge_padding_mask = torch.ones((1, 1, K_SIZE), dtype=torch.int16).to(self.device)
            return [dummy_node_inputs, dummy_node_padding_mask, dummy_edge_mask, dummy_current_index, dummy_current_edge, dummy_edge_padding_mask]

        all_node_coords = np.array(all_node_coords).reshape(-1, 2)
        utility = np.array([n.utility for n in all_nodes])
        explored_sign = np.array([n.visited for n in all_nodes])
        guidepost = np.array([n.visited for n in all_nodes])

        n_nodes = all_node_coords.shape[0]
        adjacent_matrix = np.ones((n_nodes, n_nodes)).astype(int)
        node_coords_to_check = all_node_coords[:, 0] + all_node_coords[:, 1] * 1j

        for i, node in enumerate(all_nodes):
            for neighbor in node.neighbor_set:
                n_complex = neighbor[0] + neighbor[1] * 1j
                matches = np.argwhere(node_coords_to_check == n_complex)
                if len(matches) > 0:
                    index = matches[0][0]
                    adjacent_matrix[i, index] = 0

        r_complex = robot_location[0] + robot_location[1] * 1j
        dists = np.abs(node_coords_to_check - r_complex)
        current_index = int(np.argmin(dists))

        neighbor_indices = []
        current_node = all_nodes[current_index]
        for neighbor in current_node.neighbor_set:
            n_complex = neighbor[0] + neighbor[1] * 1j
            n_dists = np.abs(node_coords_to_check - n_complex)
            if len(n_dists) > 0:
                n_idx = int(np.argmin(n_dists))
                if n_dists[n_idx] < 3.0:
                    neighbor_indices.append(n_idx)

        if current_index not in neighbor_indices:
            neighbor_indices.append(current_index)

        neighbor_indices = np.sort(np.array(neighbor_indices))

        # 保存当前的真实地图节点信息
        self.ground_truth_node_coords = all_node_coords
        self.ground_truth_node_utility = utility
        self.explored_sign = explored_sign

        # 准备构建神经网络的输入
        node_coords = all_node_coords
        node_utility = utility.reshape(-1, 1)
        node_guidepost = explored_sign.reshape(-1, 1)
        node_guidepost2 = guidepost.reshape(-1, 1)
        current_index = current_index
        edge_mask = adjacent_matrix
        current_edge = neighbor_indices
        n_node = node_coords.shape[0]

        # 归一化节点坐标：以当前节点为中心，并缩放到一定范围内
        current_node_coords = node_coords[current_index]
        node_coords = np.concatenate((node_coords[:, 0].reshape(-1, 1) - current_node_coords[0],
                                      node_coords[:, 1].reshape(-1, 1) - current_node_coords[1]),
                                      axis=-1) / UPDATING_MAP_SIZE / 2
        #node_coords = node_coords / UPDATING_MAP_SIZE / 3
        
        # 归一化效用值
        node_utility = node_utility / (SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE)
        
        # 拼接节点的所有特征（坐标、效用、探索标志、访问标志）
        node_inputs = np.concatenate((node_coords, node_utility, node_guidepost, node_guidepost2), axis=1)
        # 转换为 PyTorch 张量，并增加 batch 维度，移动到指定设备
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)

        # 断言节点数量不超过最大填充大小
        assert node_coords.shape[0] < NODE_PADDING_SIZE, print(node_coords.shape[0], NODE_PADDING_SIZE)
        
        # 对节点输入进行零填充，使其达到固定大小 NODE_PADDING_SIZE
        padding = torch.nn.ZeroPad2d((0, 0, 0, NODE_PADDING_SIZE - n_node))
        node_inputs = padding(node_inputs)

        # 创建节点填充掩码（mask），用于指示哪些是真实节点，哪些是填充的
        node_padding_mask = torch.zeros((1, 1, n_node), dtype=torch.int16).to(self.device)
        node_padding = torch.ones((1, 1, NODE_PADDING_SIZE - n_node), dtype=torch.int16).to(
            self.device)
        node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)

        # 处理边掩码（邻接矩阵），转为张量并移动到设备
        edge_mask = torch.tensor(edge_mask).unsqueeze(0).to(self.device)

        # 对边掩码进行填充
        padding = torch.nn.ConstantPad2d(
            (0, NODE_PADDING_SIZE - n_node, 0, NODE_PADDING_SIZE - n_node), 1)
        edge_mask = padding(edge_mask)

        # 找到当前节点在邻居列表中的索引（自环）
        current_in_edge = np.argwhere(current_edge == current_index)[0][0]
        # 处理当前边的索引列表
        current_edge = torch.tensor(current_edge).unsqueeze(0)
        k_size = current_edge.size()[-1]
        # 对当前边列表进行填充，使其达到固定大小 K_SIZE
        padding = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 0)
        current_edge = padding(current_edge)
        current_edge = current_edge.unsqueeze(-1)

        # 创建边填充掩码
        edge_padding_mask = torch.zeros((1, 1, k_size), dtype=torch.int16).to(self.device)
        # 标记当前节点位置
        edge_padding_mask[0, 0, current_in_edge] = 1
        # 填充边掩码
        padding = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 1)
        edge_padding_mask = padding(edge_padding_mask)

        # 将当前节点索引转为张量
        current_index = torch.tensor([current_index]).reshape(1, 1, 1).to(self.device)

        # 返回处理好的观测数据列表
        return [node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask]

    def add_node_to_dict(self, coords):
        # 将坐标元组作为键
        # Explicitly cast to float and round for quads library compatibility
        key = (round(float(coords[0]), 1), round(float(coords[1]), 1))
        # 创建新的节点对象
        node = Node(coords)
        # 插入到四叉树中
        self.nodes_dict.insert(point=key, data=node)
        return node

    def initialize_graph(self):
        print(f"DEBUG: Initializing graph for map {self.ground_truth_map_info.map_name if hasattr(self.ground_truth_map_info, 'map_name') else 'unknown'}...", flush=True)
        # 根据真实地图信息生成所有可能的节点坐标
        node_coords = self.get_ground_truth_node_coords(self.ground_truth_map_info)
        
        # Save the generated coords for plotting and debugging
        self.ground_truth_node_coords = node_coords
        
        print(f"DEBUG: Generated {len(node_coords)} potential nodes. Inserting into QuadTree...", flush=True)
        # 将这些节点添加到字典（四叉树）中
        for coords in node_coords:
            self.add_node_to_dict(coords)
        
        print("DEBUG: Nodes inserted. Building neighbor connections...", flush=True)
        # 遍历所有节点，建立邻居连接关系
        for i, node in enumerate(self.nodes_dict.__iter__()):
            if i % 1000 == 0:
                print(f"DEBUG: Processed neighbors for {i} nodes...", flush=True)
            node.data.get_neighbor_nodes(self.ground_truth_map_info, self.nodes_dict)
        
        print("DEBUG: Graph initialization complete.", flush=True)
        
    def update_graph(self):
        # 遍历信念地图中的节点，更新真实地图中对应节点的信息
        # Since we removed Ground Truth dependency, this function is now a no-op
        # or could be used to sync belief map stats if needed (but belief map updates itself).
        pass
        
        # Original logic removed:
        # for node in self.node_manager.nodes_dict.__iter__(): ...

    def get_ground_truth_node_coords(self, ground_truth_map_info):
        # 获取地图边界
        x_min = ground_truth_map_info.map_origin_x
        y_min = ground_truth_map_info.map_origin_y
        x_max = ground_truth_map_info.map_origin_x + (ground_truth_map_info.map.shape[1] - 1) * CELL_SIZE
        y_max = ground_truth_map_info.map_origin_y + (ground_truth_map_info.map.shape[0] - 1) * CELL_SIZE

        # 对齐到节点分辨率网格
        if x_min % NODE_RESOLUTION != 0:
            x_min = (x_min // NODE_RESOLUTION + 1) * NODE_RESOLUTION
        if x_max % NODE_RESOLUTION != 0:
            x_max = x_max // NODE_RESOLUTION * NODE_RESOLUTION
        if y_min % NODE_RESOLUTION != 0:
            y_min = (y_min // NODE_RESOLUTION + 1) * NODE_RESOLUTION
        if y_max % NODE_RESOLUTION != 0:
            y_max = y_max // NODE_RESOLUTION * NODE_RESOLUTION

        # 生成网格坐标点
        x_coords = np.arange(x_min, x_max + 0.1, NODE_RESOLUTION)
        y_coords = np.arange(y_min, y_max + 0.1, NODE_RESOLUTION)
        t1, t2 = np.meshgrid(x_coords, y_coords)
        # 展平并堆叠坐标
        nodes = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
        nodes = np.around(nodes, 1)

        indices = []
        # 将物理坐标转换为栅格地图的单元格坐标
        nodes_cells = get_cell_position_from_coords(nodes, ground_truth_map_info).reshape(-1, 2)
        # 筛选出位于自由区域（非障碍物）的节点
        for i, cell in enumerate(nodes_cells):
            assert 0 <= cell[1] < ground_truth_map_info.map.shape[0] and 0 <= cell[0] < ground_truth_map_info.map.shape[1]
            # Relaxed check: Allow almost-free cells (>= 250) to handle compression/noise
            # Standard FREE is 255, but sometimes values like 254 or 250 appear
            if ground_truth_map_info.map[cell[1], cell[0]] > 200:
                indices.append(i)
        
        # FIX: Ensure indices is an integer array for numpy indexing
        if len(indices) == 0:
            # Handle case where no valid nodes are found
            nodes = np.empty((0, 2))
        else:
            indices = np.array(indices, dtype=int)
            # 只保留有效的节点
            nodes = nodes[indices].reshape(-1, 2)

        return nodes

    def plot_ground_truth_env(self, robot_location):
        # 绘制真实环境及其节点状态
        # Since ground_truth_map_info is None, we cannot plot the GT map.
        # We can either remove this or adapt it to plot belief map if needed.
        # For now, we'll just return to avoid crashes.
        return
        
        # Original code commented out:
        # plt.subplot(2, 2, 3)
        # # plt.imshow(self.ground_truth_map_info.map, cmap='gray') # Removed duplicate call
        # plt.axis('off')
        # # 获取机器人和节点的栅格位置
        # robot = get_cell_position_from_coords(robot_location, self.ground_truth_map_info)
        # nodes = get_cell_position_from_coords(self.ground_truth_node_coords, self.ground_truth_map_info)
        # # 绘制地图背景
        # # Plot map once
        # plt.imshow(self.ground_truth_map_info.map, cmap='gray', vmin=0, vmax=255)
        # # 绘制节点，颜色表示探索状态
        # if nodes.size > 0:
        #     plt.scatter(nodes[:, 0], nodes[:, 1], c=self.explored_sign if self.explored_sign is not None else np.zeros((nodes.shape[0],)), zorder=2)
        # # 绘制机器人位置
        # plt.plot(robot[0], robot[1], 'mo', markersize=16, zorder=5)


class Node:
    def __init__(self, coords):
        # 节点坐标
        self.coords = coords
        # 节点效用值，初始化为一个负值
        self.utility = -(SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE)
        # 探索状态：0表示未探索，1表示已探索
        self.explored = 0
        # 访问状态：0表示未访问
        self.visited = 0

        # 邻居矩阵，用于局部邻居查找，-1表示未初始化或无连接
        self.neighbor_matrix = -np.ones((5, 5))
        # 邻居集合，存储邻居节点的坐标
        self.neighbor_set = set()
        # 将自身加入邻居集合
        self.neighbor_set.add((self.coords[0], self.coords[1]))

    def get_neighbor_nodes(self, ground_truth_map_info, nodes_dict):
        # 计算邻居矩阵的中心索引
        center_index = self.neighbor_matrix.shape[0] // 2
        # 遍历邻居矩阵的每一个位置
        for i in range(self.neighbor_matrix.shape[0]):
            for j in range(self.neighbor_matrix.shape[1]):
                # 如果已经处理过（不为-1），则跳过
                if self.neighbor_matrix[i, j] != -1:
                    continue
                else:
                    # 如果是中心点（自身），标记为1并跳过
                    if i == center_index and j == center_index:
                        self.neighbor_matrix[i, j] = 1
                        continue

                    # 计算邻居节点的物理坐标
                    neighbor_coords = np.around(np.array([self.coords[0] + (i - center_index) * NODE_RESOLUTION,
                                                self.coords[1] + (j - center_index) * NODE_RESOLUTION]), 1)
                    # 在节点字典中查找邻居节点
                    neighbor_key = (round(float(neighbor_coords[0]), 1), round(float(neighbor_coords[1]), 1))
                    neighbor_node = nodes_dict.find(neighbor_key)
                    if neighbor_node is None:
                        continue
                    else:
                        neighbor_node = neighbor_node.data
                    # 检查从当前节点到邻居节点之间是否有障碍物阻挡
                    # FIX: Convert world coords to grid indices before checking collision
                    start_cell = get_cell_position_from_coords(self.coords, ground_truth_map_info)
                    end_cell = get_cell_position_from_coords(neighbor_coords, ground_truth_map_info)
                    
                    collision = check_collision(start_cell[0], start_cell[1], end_cell[0], end_cell[1], ground_truth_map_info.map)
                    
                    # 计算在邻居节点的矩阵中，当前节点的相对位置索引
                    neighbor_matrix_x = center_index + (center_index - i)
                    neighbor_matrix_y = center_index + (center_index - j)
                        # 如果没有碰撞（路径通畅）
                    if not collision:
                            # 标记连接关系
                        self.neighbor_matrix[i, j] = 1
                        self.neighbor_set.add((neighbor_coords[0], neighbor_coords[1]))

                            # 双向连接：在邻居节点中也标记当前节点
                        neighbor_node.neighbor_matrix[neighbor_matrix_x, neighbor_matrix_y] = 1
                        neighbor_node.neighbor_set.add((self.coords[0], self.coords[1]))
