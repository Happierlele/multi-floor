import time
import heapq
import numpy as np
from utils import *
from parameter_clean import *
import quads


class NodeManager:
    def __init__(self, plot=False):
        self.graphs = {} # floor_id -> QuadTree
        self.current_floor_id = 0
        self.graphs[0] = quads.QuadTree((0, 0), 1000, 1000)
        self.plot = plot
        self.frontier = None

    @property
    def nodes_dict(self):
        if self.current_floor_id not in self.graphs:
            self.graphs[self.current_floor_id] = quads.QuadTree((0, 0), 1000, 1000)
        return self.graphs[self.current_floor_id]

    def set_floor(self, floor_id):
        if floor_id != self.current_floor_id:
            self.current_floor_id = floor_id
            self.frontier = None # Reset frontier cache when switching floors

    def reset_frontier(self):
        self.frontier = None


    def check_node_exist_in_dict(self, coords):
        key = (round(float(coords[0]), 1), round(float(coords[1]), 1))
        exist = self.nodes_dict.find(key)
        return exist

    def add_node_to_dict(self, coords, frontiers, updating_map_info, floor_id=0):
        key = (round(float(coords[0]), 1), round(float(coords[1]), 1))
        node = Node(coords, frontiers, updating_map_info, floor_id)
        self.nodes_dict.insert(point=key, data=node)
        return node

    def remove_node_from_dict(self, node):
        floor_id = getattr(node, 'floor_id', self.current_floor_id)
        if floor_id not in self.graphs:
            return

        target_graph = self.graphs[floor_id]
        
        for neighbor_coords in node.neighbor_set:
            if neighbor_coords != (node.coords[0], node.coords[1]):
                neighbor_key = (round(float(neighbor_coords[0]), 1), round(float(neighbor_coords[1]), 1))
                neighbor_node = target_graph.find(neighbor_key)
                if neighbor_node:
                    # Note: neighbor_set stores tuples of coords.
                    # node.coords is numpy array, so we must convert to tuple for removal.
                    try:
                        neighbor_node.data.neighbor_set.remove((node.coords[0], node.coords[1]))
                    except (ValueError, KeyError):
                        pass # Already removed or mismatch format
                        
        key = (round(float(node.coords[0]), 1), round(float(node.coords[1]), 1))
        target_graph.remove(key)

    def update_graph(self, robot_location, frontiers, updating_map_info, map_info, floor_id=0):
        self.set_floor(floor_id)
        node_coords, _ = get_updating_node_coords(robot_location, updating_map_info)

        # Ensure robot is connected to the graph
        # If robot is far from all candidate nodes, add robot location as a node
        min_dist_to_candidates = float('inf')
        if len(node_coords) > 0:
            dists = np.linalg.norm(node_coords - robot_location, axis=1)
            min_dist_to_candidates = np.min(dists)
        
        # Tighten threshold to 1.0m to prevent excessive off-grid node creation while maintaining connectivity
        off_grid_node = None
        
        # Check for existing nodes near the robot (including previously added off-grid nodes)
        bbox_size = 1.5
        existing_nodes_nearby = self.nodes_dict.within_bb(quads.BoundingBox(
             min_x=robot_location[0]-bbox_size, min_y=robot_location[1]-bbox_size,
             max_x=robot_location[0]+bbox_size, max_y=robot_location[1]+bbox_size
        ))
        
        min_dist_to_existing = float('inf')
        nearest_existing_node = None
        if existing_nodes_nearby:
             nearest_existing_node = min(existing_nodes_nearby, key=lambda n: np.linalg.norm(n.data.coords - robot_location)).data
             min_dist_to_existing = np.linalg.norm(nearest_existing_node.coords - robot_location)
        
        # Consider both theoretical grid candidates and existing nodes
        min_dist_effective = min(min_dist_to_candidates, min_dist_to_existing)

        # Increase threshold to 2.0m (half of NODE_RESOLUTION 4.0m) to aggressively snap to existing nodes
        if min_dist_effective > 2.0: 
              print(f"[NodeManager] Adding off-grid node at robot location {robot_location} (min_dist={min_dist_effective:.2f})")
              if len(node_coords) > 0:
                  node_coords = np.vstack([node_coords, robot_location])
              else:
                  node_coords = np.array([robot_location])
        elif min_dist_to_existing <= 2.0 and nearest_existing_node is not None:
              # Snap to existing node (likely an off-grid node from previous step)
              # Ensure it is included in update list
              # Note: node_coords are numpy arrays, we append the existing node's coords
              if len(node_coords) > 0:
                  node_coords = np.vstack([node_coords, nearest_existing_node.coords])
              else:
                  node_coords = np.array([nearest_existing_node.coords])

        if self.frontier is None:
            new_frontier = frontiers

        else:
            new_frontier = frontiers - self.frontier
            new_out_range= []
            for frontier in new_frontier:
                if np.linalg.norm(robot_location - np.array(frontier).reshape(2)) > SENSOR_RANGE + FRONTIER_CELL_SIZE:
                    new_out_range.append(frontier)
            for frontier in new_out_range:
                new_frontier.remove(frontier)

        self.frontier = frontiers

        all_node_list = []
        global_frontiers = get_frontier_in_map(map_info)
        for coords in node_coords:
            node = self.check_node_exist_in_dict(coords)
            if node is None:
                node = self.add_node_to_dict(coords, frontiers, updating_map_info, floor_id)
                # Mark this node if it matches robot location (the off-grid node)
                if np.array_equal(coords, robot_location):
                    off_grid_node = node
            else:
                node = node.data
                # Update node floor ID just in case
                if not hasattr(node, 'floor_id'):
                    node.floor_id = floor_id
                    
                if node.utility == 0 or np.linalg.norm(node.coords - robot_location) > 2 * SENSOR_RANGE:
                    pass
                else:
                    node.update_node_observable_frontiers(new_frontier, global_frontiers, updating_map_info)
            all_node_list.append(node)

        for node in all_node_list:
            if node.need_update_neighbor and np.linalg.norm(node.coords - robot_location) < (
                    SENSOR_RANGE + NODE_RESOLUTION):
                node.update_neighbor_nodes(updating_map_info, self.nodes_dict)
        
        # [Emergency Fix] Force connection for isolated nodes at robot location
        # Whether we added a new node or snapped to an existing one, ensure it's connected.
        current_robot_node = off_grid_node
        if current_robot_node is None and all_node_list:
             # Find the nearest node that the robot is effectively "at"
             nearest_existing = min(all_node_list, key=lambda n: np.linalg.norm(n.coords - robot_location))
             if np.linalg.norm(nearest_existing.coords - robot_location) <= 1.5: # Slightly larger than snap threshold
                 current_robot_node = nearest_existing

        if current_robot_node and len(current_robot_node.neighbor_set) <= 1:
             print(f"[NodeManager] Emergency: Node {current_robot_node.coords} (at robot loc) has no neighbors! Forcing connection.")
             # Find nearest node in the whole graph
             search_radius = NODE_RESOLUTION * 3.0 # Search wider
             bbox = quads.BoundingBox(
                min_x=current_robot_node.coords[0] - search_radius, min_y=current_robot_node.coords[1] - search_radius,
                max_x=current_robot_node.coords[0] + search_radius, max_y=current_robot_node.coords[1] + search_radius
             )
             nearby = self.nodes_dict.within_bb(bbox)
             
             # Connect to top-3 nearest neighbors
             candidates = []
             for wrapper in nearby:
                 n = wrapper.data
                 if n == current_robot_node: continue
                 d = np.linalg.norm(n.coords - current_robot_node.coords)
                 candidates.append((d, n))
            
             candidates.sort(key=lambda x: x[0])
             
             for d, n in candidates[:3]:
                 print(f"[NodeManager] Force connecting {current_robot_node.coords} <-> {n.coords} (dist={d:.2f})")
                 current_robot_node.neighbor_set.add((n.coords[0], n.coords[1]))
                 n.neighbor_set.add((current_robot_node.coords[0], current_robot_node.coords[1]))

    def get_node(self, floor_id, coords):
        if floor_id not in self.graphs:
            return None
        
        target_graph = self.graphs[floor_id]
        key = (round(float(coords[0]), 1), round(float(coords[1]), 1))
        
        node_wrapper = target_graph.find(key)
        if node_wrapper:
            return node_wrapper.data
            
        # Fuzzy search
        bbox_size = 1.5
        found_nodes = target_graph.within_bb(quads.BoundingBox(
             min_x=coords[0]-bbox_size, min_y=coords[1]-bbox_size,
             max_x=coords[0]+bbox_size, max_y=coords[1]+bbox_size
        ))
        
        if found_nodes:
             nearest = min(found_nodes, key=lambda n: np.linalg.norm(np.array(n.data.coords) - coords))
             if np.linalg.norm(np.array(nearest.data.coords) - coords) < 2.0:
                 return nearest.data
                 
        return None

    def add_stair_connection(self, floor1, coords1, floor2, coords2):
        node1 = self.get_node(floor1, coords1)
        node2 = self.get_node(floor2, coords2)
        
        if node1 and node2:
            # Store tuple (floor_id, x, y)
            # Use raw coords or rounded? Use raw coords for storage, A* will handle keys.
            node1.stair_neighbors.add((floor2, node2.coords[0], node2.coords[1]))
            node2.stair_neighbors.add((floor1, node1.coords[0], node1.coords[1]))
            print(f"[NodeManager] Added Stair Connection: F{floor1} {node1.coords} <-> F{floor2} {node2.coords}")
        else:
            print(f"[NodeManager] Failed to add stair connection: Nodes not found. F{floor1}:{node1 is not None}, F{floor2}:{node2 is not None}")

    def Dijkstra(self, start, boundary=None):
        q = set()
        dist_dict = {}
        prev_dict = {}

        # Use rounded keys for graph traversal to match QuadTree keys
        for node in self.nodes_dict.__iter__():
            coords = node.data.coords
            key = (round(float(coords[0]), 1), round(float(coords[1]), 1))
            dist_dict[key] = 1e8
            prev_dict[key] = None
            q.add(key)

        start_key = (round(float(start[0]), 1), round(float(start[1]), 1))
        
        # Robust check: if start node was just added but not in iterator yet? 
        if start_key not in dist_dict:
             # Fuzzy search for start in existing nodes
             bbox_size = 3.0 # Sync with A* and Emergency Connection (was 1.2)
             found_nodes = self.nodes_dict.within_bb(quads.BoundingBox(
                 min_x=start[0]-bbox_size, min_y=start[1]-bbox_size,
                 max_x=start[0]+bbox_size, max_y=start[1]+bbox_size
             ))
             if found_nodes:
                 nearest = min(found_nodes, key=lambda n: np.linalg.norm(np.array(n.data.coords) - start))
                 # Validate distance
                 if np.linalg.norm(np.array(nearest.data.coords) - start) < 2.5:
                     start_key = (round(float(nearest.data.coords[0]), 1), round(float(nearest.data.coords[1]), 1))
             
             # Double check if the fuzzy match is in dist_dict
             if start_key not in dist_dict:
                 dist_dict[start_key] = 1e8
                 prev_dict[start_key] = None
                 q.add(start_key)

        dist_dict[start_key] = 0

        while len(q) > 0:
            u = None
            for coords in q:
                if u is None:
                    u = coords
                elif dist_dict[coords] < dist_dict[u]:
                    u = coords

            q.remove(u)

            # u is a rounded key, so find() works
            node_wrapper = self.nodes_dict.find(u)
            if node_wrapper is None:
                continue
            node = node_wrapper.data
            
            for neighbor_node_coords in node.neighbor_set:
                # Round neighbor coords to match keys
                v = (round(float(neighbor_node_coords[0]), 1), round(float(neighbor_node_coords[1]), 1))
                if v in q:
                    # Calculate cost using raw coords for better precision, or use rounded u/v
                    # Using u (rounded) and neighbor_node_coords (raw) is slightly inconsistent but fine for cost
                    # Let's use stored neighbor coords and node coords for distance
                    cost = np.linalg.norm(np.array(neighbor_node_coords) - np.array(node.coords))
                    cost = np.round(cost, 2)
                    alt = dist_dict[u] + cost
                    if alt < dist_dict[v]:
                        dist_dict[v] = alt
                        prev_dict[v] = u

        return dist_dict, prev_dict

    def get_Dijkstra_path_and_dist(self, dist_dict, prev_dict, end):
        end_key = (round(float(end[0]), 1), round(float(end[1]), 1))
        
        if end_key not in dist_dict:
            print(f"destination {end_key} is not in Dijkstra graph")
            return [], 1e8

        dist = dist_dict[end_key]

        path = [end_key]
        prev_node = prev_dict[end_key]
        while prev_node is not None:
            path.append(prev_node)
            temp = prev_node
            prev_node = prev_dict[temp]

        path.reverse()
        # path contains rounded keys.
        return path[1:], np.round(dist, 2)

    def h(self, coords_1, coords_2):
        # h = abs(coords_1[0] - coords_2[0]) + abs(coords_1[1] - coords_2[1])
        # h = ((coords_1[0] - coords_2[0]) ** 2 + (coords_1[1] - coords_2[1]) ** 2) ** (1 / 2)
        h = np.linalg.norm(np.array([coords_1[0] - coords_2[0], coords_1[1] - coords_2[1]]))
        # h = np.round(h, 2)
        return h

    def a_star(self, start, destination, max_dist=None, start_floor=None, dest_floor=None):
        # Default to current floor if not specified
        if start_floor is None: start_floor = self.current_floor_id
        if dest_floor is None: dest_floor = self.current_floor_id
        
        # Robust start node lookup
        start_key_2d = (round(float(start[0]), 1), round(float(start[1]), 1))
        start_key = (start_floor,) + start_key_2d
        
        start_node_data = self.get_node(start_floor, start)
        if start_node_data:
             start_key = (start_floor, round(float(start_node_data.coords[0]), 1), round(float(start_node_data.coords[1]), 1))
        else:
             print(f"A* Warning: Start position {start} on floor {start_floor} not found in graph")
             return [], 1e8

        # Robust destination node lookup
        dest_key_2d = (round(float(destination[0]), 1), round(float(destination[1]), 1))
        dest_key = (dest_floor,) + dest_key_2d
        
        dest_node_data = self.get_node(dest_floor, destination)
        if dest_node_data:
             dest_key = (dest_floor, round(float(dest_node_data.coords[0]), 1), round(float(dest_node_data.coords[1]), 1))
        else:
             print(f"A* Warning: End position {destination} on floor {dest_floor} not found in graph")
             return [], 1e8

        if start_key == dest_key:
            return [], 0

        open_list = {start_key}
        closed_list = set()
        g = {start_key: 0}
        parents = {start_key: start_key}

        open_heap = []
        heapq.heappush(open_heap, (0, start_key))

        while len(open_list) > 0:
            if not open_heap:
                break
            _, n = heapq.heappop(open_heap)
            
            if n in closed_list:
                continue
            if n not in open_list:
                continue
            
            # n is (floor, x, y)
            current_floor, nx, ny = n
            n_key_2d = (nx, ny)
            
            if current_floor not in self.graphs:
                open_list.remove(n)
                continue
                
            node_wrapper = self.graphs[current_floor].find(n_key_2d)
            if node_wrapper is None:
                open_list.remove(n)
                continue
            node = node_wrapper.data

            if max_dist is not None:
                if g[n] > max_dist:
                    return [], 1e8

            if n == dest_key:
                path = []
                length = g[n]
                while parents[n] != n:
                    path.append(n)
                    n = parents[n]
                path.reverse()
                # Return path of (floor, x, y)
                return path, np.round(length, 2)

            # Expand neighbors
            # 1. Same floor neighbors
            neighbor_list = list(node.neighbor_set) # [(x,y), ...]
            # 2. Stair neighbors
            stair_neighbors = list(node.stair_neighbors) # [(f, x, y), ...]

            all_neighbors = []
            # Format same floor: (current_floor, x, y)
            for nb in neighbor_list:
                all_neighbors.append((current_floor, nb[0], nb[1]))
            # Format stair: (f, x, y)
            for nb in stair_neighbors:
                all_neighbors.append(nb)

            for nb in all_neighbors:
                nb_floor, nb_x, nb_y = nb
                m = (nb_floor, round(float(nb_x), 1), round(float(nb_y), 1))
                
                # Calculate cost
                dist_2d = np.linalg.norm(np.array([nb_x, nb_y]) - np.array([node.coords[0], node.coords[1]]))
                cost = dist_2d
                if nb_floor != current_floor:
                    cost = max(dist_2d, 1.0) # Minimum cost for floor switch
                cost = np.round(cost, 2)
                
                if m not in open_list and m not in closed_list:
                    open_list.add(m)
                    parents[m] = n
                    g[m] = g[n] + cost
                    heapq.heappush(open_heap, (g[m], m))
                else:
                    if g.get(m, float('inf')) > g[n] + cost:
                        g[m] = g[n] + cost
                        parents[m] = n
                        heapq.heappush(open_heap, (g[m], m))

            open_list.remove(n)
            closed_list.add(n)

        print('Path does not exist!')
        return [], 1e8


class Node:
    def __init__(self, coords, frontiers, updating_map_info, floor_id=0):
        self.coords = coords
        self.floor_id = floor_id
        self.utility_range = UTILITY_RANGE
        self.utility = 0
        self.observable_frontiers = self.initialize_observable_frontiers(frontiers, updating_map_info)
        self.visited = 0
        self.visit_count = 0

        self.neighbor_matrix = -np.ones((5, 5))
        self.neighbor_set = set()
        self.neighbor_matrix[2, 2] = 1
        self.neighbor_set.add((self.coords[0], self.coords[1]))
        self.stair_neighbors = set() # Stores tuples: (floor_id, x, y)
        self.need_update_neighbor = True

    def initialize_observable_frontiers(self, frontiers, updating_map_info):
        if len(frontiers) == 0:
            self.utility = 0
            return set()
        else:
            observable_frontiers = set()
            frontiers = np.array(list(frontiers)).reshape(-1, 2)
            dist_list = np.linalg.norm(frontiers - self.coords, axis=-1)
            new_frontiers_in_range = frontiers[dist_list < self.utility_range]
            for point in new_frontiers_in_range:
                # Fix: Convert to grid coordinates for collision check
                start_cell = get_cell_position_from_coords(self.coords, updating_map_info)
                end_cell = get_cell_position_from_coords(point, updating_map_info)
                collision = check_collision(start_cell[0], start_cell[1], end_cell[0], end_cell[1], updating_map_info.map)
                if not collision:
                    observable_frontiers.add((point[0], point[1]))
            self.utility = len(observable_frontiers)
            if self.utility <= MIN_UTILITY:
                self.utility = 0
                observable_frontiers = set()
            return observable_frontiers

    def update_neighbor_nodes(self, updating_map_info, nodes_dict):
        # Robust Logic: Search for nearby nodes using QuadTree range search
        # instead of rigid grid assumptions. This handles off-grid nodes correctly.
        search_radius = NODE_RESOLUTION * 1.5 
        bbox = quads.BoundingBox(
            min_x=self.coords[0] - search_radius,
            min_y=self.coords[1] - search_radius,
            max_x=self.coords[0] + search_radius,
            max_y=self.coords[1] + search_radius
        )
        
        nearby_nodes = nodes_dict.within_bb(bbox)
        
        for neighbor_wrapper in nearby_nodes:
            neighbor_node = neighbor_wrapper.data
            neighbor_coords = neighbor_node.coords
            
            # Skip self
            if np.array_equal(neighbor_coords, self.coords):
                continue
            
            # Skip different floor
            if getattr(neighbor_node, 'floor_id', 0) != getattr(self, 'floor_id', 0):
                continue

            dist = np.linalg.norm(neighbor_coords - self.coords)
            if dist > search_radius: 
                continue
                
            # Check collision
            # Fix: Convert to grid coordinates for collision check
            start_cell = get_cell_position_from_coords(self.coords, updating_map_info)
            end_cell = get_cell_position_from_coords(neighbor_coords, updating_map_info)
            collision = check_collision(start_cell[0], start_cell[1], end_cell[0], end_cell[1], updating_map_info.map)
            if not collision:
                # Add to set (coordinates tuple)
                self.neighbor_set.add((neighbor_coords[0], neighbor_coords[1]))
                
                # check if aligned with grid
                diff = neighbor_coords - self.coords
                if abs(diff[0]) < 1e-3 or abs(diff[1]) < 1e-3: # axis aligned
                     pass

                # Bidirectional connection: Ensure neighbor also connects to this node
                neighbor_node.neighbor_set.add((self.coords[0], self.coords[1]))

        if self.utility == 0:
            self.need_update_neighbor = False

    def update_node_observable_frontiers(self, new_frontiers, global_frontiers, updating_map_info):
        # remove frontiers observed
        frontiers_observed = []
        for frontier in self.observable_frontiers:
            if frontier not in global_frontiers:
                frontiers_observed.append(frontier)
        for frontier in frontiers_observed:
            self.observable_frontiers.remove(frontier)

        # add new frontiers in the observable frontiers
        if len(new_frontiers) > 0:
            new_frontiers = np.array(list(new_frontiers)).reshape(-1, 2)
            dist_list = np.linalg.norm(new_frontiers - self.coords, axis=-1)
            new_frontiers_in_range = new_frontiers[dist_list < self.utility_range]
            for point in new_frontiers_in_range:
                # Fix: Convert to grid coordinates for collision check
                start_cell = get_cell_position_from_coords(self.coords, updating_map_info)
                end_cell = get_cell_position_from_coords(point, updating_map_info)
                collision = check_collision(start_cell[0], start_cell[1], end_cell[0], end_cell[1], updating_map_info.map)
                if not collision:
                    self.observable_frontiers.add((point[0], point[1]))

        self.utility = len(self.observable_frontiers)
        if self.utility <= MIN_UTILITY:
            self.utility = 0
            self.observable_frontiers = set()

    def set_visited(self):
        self.visited = 1
        self.visit_count += 1
        self.observable_frontiers = set()
