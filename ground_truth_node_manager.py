import torch

from utils import *
from parameter_clean import *
import quads
import matplotlib.pyplot as plt


class GroundTruthNodeManager:
    def __init__(self, node_manager, ground_truth_map_info, device='cpu', plot=False):
        # Calculate appropriate bounds for QuadTree based on map info
        if hasattr(ground_truth_map_info, 'map_origin_x') and hasattr(ground_truth_map_info, 'map') and ground_truth_map_info.map is not None:
            h, w = ground_truth_map_info.map.shape
            map_width = w * CELL_SIZE
            map_height = h * CELL_SIZE
            center_x = ground_truth_map_info.map_origin_x + map_width / 2.0
            center_y = ground_truth_map_info.map_origin_y + map_height / 2.0
            size_x = map_width * 2.0 # Add padding
            size_y = map_height * 2.0
            qt_size = max(size_x, size_y, 200.0) # Min size 200m
            print(f"DEBUG: Initializing QuadTree with center=({center_x:.1f}, {center_y:.1f}), size={qt_size:.1f}", flush=True)
            self.nodes_dict = quads.QuadTree((center_x, center_y), qt_size, qt_size)
        else:
            # Fallback
            self.nodes_dict = quads.QuadTree((0, 0), 2000, 2000)

        # 引用普通的 node_manager，可能用于获取信念状态下的节点信息
        self.node_manager = node_manager
        # 存储真实地图信息（Ground Truth Map）
        self.ground_truth_map_info = ground_truth_map_info
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
        self.initialize_graph()

    def get_ground_truth_observation(self, robot_location):
        # 更新图结构（例如同步信念地图中的效用值到真实地图节点）
        self.update_graph()

        # 获取信念地图中所有节点的坐标
        all_node_coords = []
        for node in self.node_manager.nodes_dict.__iter__():
            all_node_coords.append(node.data.coords)
        # 获取真实地图中尚未被探索的节点坐标
        for node in self.nodes_dict.__iter__():
            if node.data.explored == 0:
                all_node_coords.append(node.data.coords)
        # 将坐标列表转换为 numpy 数组，形状为 (N, 2)
        all_node_coords = np.array(all_node_coords).reshape(-1, 2)
        
        # 初始化列表用于存储节点属性
        utility = []
        explored_sign = []
        guidepost = []

        # 获取节点总数
        n_nodes = all_node_coords.shape[0]
        # 初始化邻接矩阵，全为 1（表示全连接或初始化状态），之后会修改
        adjacent_matrix = np.ones((n_nodes, n_nodes)).astype(int)
        # 将节点坐标转换为复数形式，便于后续快速查找索引
        node_coords_to_check = all_node_coords[:, 0] + all_node_coords[:, 1] * 1j
        
        # 遍历所有节点，提取属性并构建邻接关系
        # 预先过滤掉无效节点，避免循环中的空值检查导致 utility 等列表长度不一致
        valid_indices = []
        for i, coords in enumerate(all_node_coords):
            # Explicitly cast to float and round for quads library compatibility
            key = (round(float(coords[0]), 1), round(float(coords[1]), 1))
            node_obj = self.nodes_dict.find(key)
            
            # 如果精确查找失败，尝试模糊查找 (处理浮点数精度问题)
            if node_obj is None:
                # 在四叉树中搜索最近的邻居
                bbox_size = 0.5
                found_nodes = self.nodes_dict.within_bb(quads.BoundingBox(
                    min_x=key[0]-bbox_size, min_y=key[1]-bbox_size,
                    max_x=key[0]+bbox_size, max_y=key[1]+bbox_size
                ))
                if found_nodes:
                    # Find truly closest node
                    node_obj = min(found_nodes, key=lambda n: np.linalg.norm(np.array(n.data.coords) - coords))
            
            if node_obj is None:
                # 这是一个紧急修复：如果四叉树中找不到该坐标，说明同步出现了严重问题
                # 尝试再次放宽搜索范围
                bbox_size = 3.0
                found_nodes = self.nodes_dict.within_bb(quads.BoundingBox(
                    min_x=key[0]-bbox_size, min_y=key[1]-bbox_size,
                    max_x=key[0]+bbox_size, max_y=key[1]+bbox_size
                ))
                if found_nodes:
                    node_obj = min(found_nodes, key=lambda n: np.linalg.norm(np.array(n.data.coords) - coords))

            if node_obj is None:
                # 如果仍然找不到，只打印一次警告（避免刷屏），或者直接跳过
                # print(f"Warning: Node at {coords} not found in GroundTruth QuadTree. Skipping.", flush=True)
                continue
                
            node = node_obj.data
            all_node_coords[i] = np.array(node.coords)
            # 添加效用值
            utility.append(node.utility)
            # 添加探索状态
            explored_sign.append(node.explored)
            # 添加访问状态（guidepost）
            guidepost.append(node.visited)
            
            valid_indices.append(i) # 记录有效节点的原始索引

        # 如果没有有效节点，尝试添加机器人当前位置作为节点
        if not valid_indices:
             print("Warning: No valid nodes found. Attempting to add robot location as a node.", flush=True)
             robot_rounded = np.around(robot_location, 1)
             self.add_node_to_dict(robot_rounded)
             
             # Re-run node lookup for this new node
             key = (round(float(robot_rounded[0]), 1), round(float(robot_rounded[1]), 1))
             node_obj = self.nodes_dict.find(key)
             
             if node_obj:
                 # Override lists with single node
                 all_node_coords_list = [np.array(node_obj.data.coords)]
                 utility = [node_obj.data.utility]
                 explored_sign = [node_obj.data.explored]
                 guidepost = [node_obj.data.visited]
                 valid_indices = [0]
                 
                 # IMPORTANT: Connect to nearest existing nodes to prevent isolation
                 bbox_size = 10.0
                 nearby_nodes = self.nodes_dict.within_bb(quads.BoundingBox(
                    min_x=key[0]-bbox_size, min_y=key[1]-bbox_size,
                    max_x=key[0]+bbox_size, max_y=key[1]+bbox_size
                ))
                 
                 if nearby_nodes:
                     nearby_nodes.sort(key=lambda n: np.linalg.norm(np.array(n.data.coords) - robot_location))
                     for neighbor_obj in nearby_nodes[:5]: # Connect up to 5 neighbors
                         if neighbor_obj == node_obj: continue
                         n_data = neighbor_obj.data
                         all_node_coords_list.append(np.array(n_data.coords))
                         utility.append(n_data.utility)
                         explored_sign.append(n_data.explored)
                         guidepost.append(n_data.visited)
                         valid_indices.append(len(valid_indices))
                 
                 all_node_coords = np.array(all_node_coords_list)
                 print(f"Successfully added fallback node and connected to {len(all_node_coords)-1} neighbors.", flush=True)
             else:
                 print("CRITICAL ERROR: Failed to add fallback node!", flush=True)

        # 如果没有有效节点，直接返回空或报错，防止后续计算崩溃
        if not valid_indices:
             print("CRITICAL ERROR: No valid nodes found in GroundTruth QuadTree! Returning PADDED dummy data.", flush=True)
             
             # Create dummy data matching the shapes of normal output
             # 1. node_inputs: (1, NODE_PADDING_SIZE, 5) - Assuming 5 dims (coords(2)+util(1)+expl(1)+visit(1))
             dummy_dim = 5 
             dummy_node_inputs = torch.zeros((1, NODE_PADDING_SIZE, dummy_dim)).to(self.device)
             
             # 2. node_padding_mask: (1, 1, NODE_PADDING_SIZE)
             dummy_node_padding_mask = torch.ones((1, 1, NODE_PADDING_SIZE), dtype=torch.int16).to(self.device)
             
             # 3. edge_mask: (1, NODE_PADDING_SIZE, NODE_PADDING_SIZE)
             dummy_edge_mask = torch.ones((1, NODE_PADDING_SIZE, NODE_PADDING_SIZE)).to(self.device)
             
             # 4. current_index: (1, 1, 1)
             dummy_current_index = torch.zeros((1, 1, 1)).to(self.device)
             
             # 5. current_edge: (1, K_SIZE, 1)
             dummy_current_edge = torch.zeros((1, K_SIZE, 1)).to(self.device)
             
             # 6. edge_padding_mask: (1, 1, K_SIZE)
             dummy_edge_padding_mask = torch.ones((1, 1, K_SIZE), dtype=torch.int16).to(self.device)
             
             return [dummy_node_inputs, dummy_node_padding_mask, dummy_edge_mask, dummy_current_index, dummy_current_edge, dummy_edge_padding_mask]

        # 重建 all_node_coords 只包含有效节点
        all_node_coords = all_node_coords[valid_indices]
        n_nodes = len(valid_indices)
        
        # 重新计算 node_coords_to_check，只包含有效节点
        node_coords_to_check = all_node_coords[:, 0] + all_node_coords[:, 1] * 1j
        
        # 初始化邻接矩阵，大小为有效节点数
        adjacent_matrix = np.ones((n_nodes, n_nodes)).astype(int)

        # 重新遍历有效节点构建邻接关系
        # Use robust distance check instead of exact complex number matching
        for i, coords in enumerate(all_node_coords):
            # 1. Find the node object for the current coords
            key = (round(float(coords[0]), 1), round(float(coords[1]), 1))
            node_obj = self.nodes_dict.find(key)
            
            # Fuzzy fallback
            if node_obj is None:
                 found = self.nodes_dict.within_bb(quads.BoundingBox(
                    min_x=coords[0]-0.5, min_y=coords[1]-0.5,
                    max_x=coords[0]+0.5, max_y=coords[1]+0.5
                ))
                 if found:
                     node_obj = min(found, key=lambda n: np.linalg.norm(np.array(n.data.coords) - coords))
            
            if node_obj:
                node = node_obj.data
                # 2. Iterate through neighbors
                for neighbor in node.neighbor_set:
                    # neighbor is (x, y) tuple
                    n_arr = np.array(neighbor)
                    
                    # 3. Find index in all_node_coords using vectorized distance check
                    dists = np.linalg.norm(all_node_coords - n_arr, axis=1)
                    
                    # Find matches within small threshold (0.2m)
                    matches = np.where(dists < 0.2)[0]
                    
                    for index in matches:
                        adjacent_matrix[i, index] = 0

        # 将属性列表转换为 numpy 数组
        utility = np.array(utility)
        explored_sign = np.array(explored_sign)
        guidepost = np.array(guidepost)

        # 找到机器人当前所在位置对应的节点索引
        # Robust lookup: Find nearest node instead of exact match
        dists = np.abs(node_coords_to_check - (robot_location[0] + robot_location[1] * 1j))
        current_index = np.argmin(dists)
        
        # Optional: Warn if the snap distance is large
        # Relaxed threshold to 4.1m for NODE_RESOLUTION=4.0m
        if dists[current_index] > 4.1:
            print(f"Warning: Robot at {robot_location} is far from nearest node at index {current_index} (dist={dists[current_index]:.2f})", flush=True)

        
        # neighbor_indices = np.argwhere(adjacent_matrix[current_index] == 0).reshape(-1)
        # 下面的代码用于获取当前机器人的邻居节点索引
        neighbor_indices = []
        # 在信念地图的节点管理器中查找当前节点
        # Use robust find for belief node as well
        current_key = (round(float(robot_location[0]), 1), round(float(robot_location[1]), 1))
        current_node_in_belief_obj = self.node_manager.nodes_dict.find(current_key)
        
        # If exact find fails, try nearest in belief graph
        if current_node_in_belief_obj is None:
             bbox_size = 3.0
             found_nodes = self.node_manager.nodes_dict.within_bb(quads.BoundingBox(
                    min_x=robot_location[0]-bbox_size, min_y=robot_location[1]-bbox_size,
                    max_x=robot_location[0]+bbox_size, max_y=robot_location[1]+bbox_size
                ))
             if found_nodes:
                 # Find closest
                 current_node_in_belief_obj = min(found_nodes, key=lambda n: np.linalg.norm(np.array(n.data.coords) - robot_location))

        if current_node_in_belief_obj:
            current_node_in_belief = current_node_in_belief_obj.data
            # 遍历邻居节点
            for neighbor in current_node_in_belief.neighbor_set:
                # 找到邻居在当前观测列表中的索引 - Robust lookup
                n_complex = neighbor[0] + neighbor[1] * 1j
                n_dists = np.abs(node_coords_to_check - n_complex)
                if len(n_dists) > 0:
                    n_idx = np.argmin(n_dists)
                    # Only accept if close enough
                    # Relaxed threshold from 1.0 to 3.0 to handle NODE_RESOLUTION=4.0
                    if n_dists[n_idx] < 3.0:
                        neighbor_indices.append(n_idx)
        
        # Ensure current_index is always in neighbor_indices (self-loop)
        # This prevents IndexError when finding current_in_edge later
        if current_index not in neighbor_indices:
            neighbor_indices.append(current_index)

        # 对邻居索引进行排序
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
        for node in self.node_manager.nodes_dict.__iter__():
            coords = node.data.coords
            
            # Use rounded key for lookup, consistent with add_node_to_dict
            key = (round(float(coords[0]), 1), round(float(coords[1]), 1))
            ground_truth_node = self.nodes_dict.find(key)
            
            # If exact match fails, try fuzzy match (nearest neighbor)
            if ground_truth_node is None:
                # Use a reasonable search radius (e.g., half of NODE_RESOLUTION or slightly more)
                # Assuming NODE_RESOLUTION is 4.0, we search within 2.5m
                bbox_size = 2.5 
                found_nodes = self.nodes_dict.within_bb(quads.BoundingBox(
                    min_x=coords[0]-bbox_size, min_y=coords[1]-bbox_size,
                    max_x=coords[0]+bbox_size, max_y=coords[1]+bbox_size
                ))
                if found_nodes:
                    # Find closest node in the bounding box
                    ground_truth_node = min(found_nodes, key=lambda n: np.linalg.norm(np.array(n.data.coords) - coords))

            # Auto-repair: If node is still missing in GT map but exists in Belief Map, add it.
            if ground_truth_node is None:
                # print(f"Repairing: Adding missing belief node {coords} to Ground Truth Map", flush=True)
                self.add_node_to_dict(coords)
                # Retrieve the wrapper node object from the QuadTree to ensure we have the correct structure
                ground_truth_node = self.nodes_dict.find(key)
                
                if ground_truth_node:
                    # print(f"DEBUG: Successfully repaired missing node at {coords}", flush=True)
                    pass
                else:
                    print(f"DEBUG: Failed to repair missing node at {coords} even after adding!", flush=True)
                
                # Since it's new, we need to find its neighbors in the GT graph to maintain connectivity
                if ground_truth_node is not None and hasattr(ground_truth_node, 'data'):
                    ground_truth_node.data.get_neighbor_nodes(self.ground_truth_map_info, self.nodes_dict)
                    
                    # IMPORTANT: Update self.ground_truth_node_coords cache!
                    # The get_ground_truth_observation method relies on this list matching the QuadTree/valid_indices logic.
                    # If we add a node but don't update this list, it might be skipped or cause index mismatch later.
                    if self.ground_truth_node_coords is not None:
                        self.ground_truth_node_coords = np.vstack((self.ground_truth_node_coords, coords))

            # FIX: Handle case where ground_truth_node is None (e.g., node mismatch)
            if ground_truth_node is not None and hasattr(ground_truth_node, 'data'):
                # 同步效用值
                ground_truth_node.data.utility = node.data.utility
                # 标记为已探索
                ground_truth_node.data.explored = 1
                # 同步访问状态
                ground_truth_node.data.visited = node.data.visited
            else:
                # Optional: Log warning if mismatch occurs frequently
                pass

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

    def plot_ground_truth_env(self, robot_location, position_history=None, current_floor=None, stairs_coords=None):
        # 绘制真实环境及其节点状态
        plt.subplot(2, 2, 3)
        plt.cla() # Clear current axis
        plt.axis('off')
        plt.title(f"GT Graph: {len(self.ground_truth_node_coords)} Nodes", fontsize=8)
        
        # 绘制地图背景
        if self.ground_truth_map_info.map is not None:
            plt.imshow(self.ground_truth_map_info.map, cmap='gray', vmin=0, vmax=255, alpha=0.6, origin='upper')

        # 获取机器人和节点的栅格位置
        robot = get_cell_position_from_coords(robot_location, self.ground_truth_map_info)
        nodes_grid = get_cell_position_from_coords(self.ground_truth_node_coords, self.ground_truth_map_info)
        
        # 绘制边 (Edges) - Make them fainter to reduce clutter
        # 注意：如果节点非常多，画所有边可能会很乱。这里只画部分或全部。
        # 为了性能和清晰度，我们遍历 self.nodes_dict
        if hasattr(self, 'nodes_dict') and self.nodes_dict:
            for node in self.nodes_dict.__iter__():
                p1_world = node.data.coords
                p1_grid = get_cell_position_from_coords(p1_world, self.ground_truth_map_info)
                
                # 遍历邻居
                for neighbor_world in node.data.neighbor_set:
                    # neighbor_world is tuple (x, y)
                    p2_grid = get_cell_position_from_coords(neighbor_world, self.ground_truth_map_info)
                    
                    # 绘制线段 - Use very low alpha to show connectivity without overwhelming the view
                    plt.plot([p1_grid[0], p2_grid[0]], [p1_grid[1], p2_grid[1]], color='cyan', linewidth=0.5, alpha=0.15)

        # 绘制历史轨迹 (Trajectory) - Show actual path
        if position_history and len(position_history) > 1:
            traj_coords = []
            last_valid_p = None
            
            for p in position_history:
                # Handle tuple (x, y) or (x, y, floor_id)
                if isinstance(p, (tuple, list, np.ndarray)):
                    p_flat = np.array(p).flatten()
                    
                    # Filter by floor: if p has floor info and it doesn't match current_floor
                    if len(p_flat) >= 3 and current_floor is not None and int(p_flat[2]) != current_floor:
                         traj_coords.append([np.nan, np.nan])
                         last_valid_p = None
                         continue
                    
                    current_p = p_flat[:2]
                    
                    # Check for large jumps (Teleportation / Stuck Recovery)
                    if last_valid_p is not None:
                        dist = np.linalg.norm(current_p - last_valid_p)
                        if dist > 1.0: # If moved more than 1.0m in one step, it's a jump
                             traj_coords.append([np.nan, np.nan])
                    
                    traj_coords.append(current_p)
                    last_valid_p = current_p

            if len(traj_coords) > 1:
                traj_coords = np.array(traj_coords)
                # Handle NaNs correctly when converting to grid
                # get_cell_position_from_coords might fail with NaNs, so we need a robust way
                # Filter out NaNs for conversion, then re-insert, or just iterate segments.
                # Simpler: convert point by point or use masked array.
                
                # Let's use a simpler approach: 
                # Convert valid points to grid, keep NaNs as NaNs
                traj_grid_list = []
                for pt in traj_coords:
                    if np.isnan(pt[0]):
                        traj_grid_list.append([np.nan, np.nan])
                    else:
                        grid_pt = get_cell_position_from_coords(pt, self.ground_truth_map_info)
                        traj_grid_list.append(grid_pt)
                
                traj_grid = np.array(traj_grid_list)
                
                # Draw trajectory with distinct color
                # Use plot which handles NaNs by breaking the line
                plt.plot(traj_grid[:, 0], traj_grid[:, 1], color='blue', linewidth=1.5, alpha=0.8, label='Path')

        # 绘制节点
        if nodes_grid.size > 0:
            # 区分已探索和未探索
            # explored_sign: 1 = explored, 0 = unexplored
            # 确保 explored_sign 长度与 nodes_grid 一致
            if self.explored_sign is not None and len(self.explored_sign) == len(nodes_grid):
                colors = ['red' if e == 0 else 'green' for e in self.explored_sign]
            else:
                colors = 'blue' # Default if mismatch
                
            plt.scatter(nodes_grid[:, 0], nodes_grid[:, 1], c=colors, s=10, zorder=2, edgecolors='none', alpha=0.8)

        # 绘制楼梯位置 (Stairs) - Explicitly requested by user
        if stairs_coords is not None and len(stairs_coords) > 0:
            stairs_grid = get_cell_position_from_coords(np.array(stairs_coords), self.ground_truth_map_info)
            if stairs_grid.size > 0:
                # Yellow Star for stairs
                plt.scatter(stairs_grid[:, 0], stairs_grid[:, 1], c='yellow', marker='*', s=150, zorder=6, edgecolors='black', label='Stairs')
            
        # 绘制机器人位置
        if robot is not None and robot.size > 0:
             plt.plot(robot[0], robot[1], 'mo', markersize=8, markeredgecolor='white', markeredgewidth=1, zorder=5, label='Robot')


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
