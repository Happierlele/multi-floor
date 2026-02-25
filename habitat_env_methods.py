import matplotlib.pyplot as plt
import numpy as np
import os
from parameter_clean import gifs_path

def plot_env(self, step):
    if not self.plot:
        return

    plt.switch_backend('agg')
    plt.figure(figsize=(12, 6)) # Larger figure for better visibility
    
    # Subplot 1: Explored Map (2D State Representation)
    plt.subplot(1, 2, 1)
    
    # Enhanced Map Visualization
    # Assuming robot_belief is 0-255 (0=Occupied, 255=Free, 127=Unknown)
    # We can create a colored map for better contrast
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
        
        # Plot orientation if available (requires robot yaw)
        # arrow_len = 5.0
        # plt.arrow(rx, ry, arrow_len*np.cos(yaw), arrow_len*np.sin(yaw), color='m', width=0.5)

        # Plot trajectory
        if len(self.trajectory_x) > 1:
            traj_x = (np.array(self.trajectory_x) - self.belief_origin_x) / self.mapper.cell_size
            traj_y = (np.array(self.trajectory_y) - self.belief_origin_y) / self.mapper.cell_size
            plt.plot(traj_x, traj_y, 'b-', linewidth=1.5, alpha=0.7, zorder=5, label='Trajectory')
    
    # Plot stairs
    if hasattr(self, "stairs_coords_list") and self.stairs_coords_list is not None:
        for idx, coords in enumerate(self.stairs_coords_list):
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

    # Plot Frontiers (Green dots)
    if hasattr(self, "global_frontiers") and self.global_frontiers:
        frontier_x = []
        frontier_y = []
        for item in self.global_frontiers:
            # Handle format ((x,y), area) or (x,y)
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], tuple):
                coords = item[0]
            else:
                coords = item
            
            fx = (coords[0] - self.belief_origin_x) / self.mapper.cell_size
            fy = (coords[1] - self.belief_origin_y) / self.mapper.cell_size
            frontier_x.append(fx)
            frontier_y.append(fy)
            
        if frontier_x:
            plt.scatter(frontier_x, frontier_y, c='lime', s=10, marker='.', zorder=4, label='Frontiers')

    # Legend for map elements
    # plt.legend(loc='upper right', fontsize='small')

    # Subplot 2: RGB View (Agent Perspective)
    if hasattr(self, 'rgb_image') and self.rgb_image is not None:
         plt.subplot(1, 2, 2)
         plt.imshow(self.rgb_image)
         plt.title('Agent View (RGB)')
         plt.axis('off')
    else:
         plt.subplot(1, 2, 2)
         plt.text(0.5, 0.5, 'No RGB Image', ha='center', va='center')
         plt.axis('off')

    # Global Title with Metrics
    plt.suptitle(f'Explored: {self.explored_rate:.1%} | Dist: {self.travel_dist:.1f}m | Step: {step}', fontsize=14)
    plt.tight_layout()
    
    # Ensure directory exists
    if not os.path.exists(gifs_path):
        os.makedirs(gifs_path)
        
    save_path = '{}/{}_{}_habitat.png'.format(gifs_path, self.episode_index, step)
    plt.savefig(save_path, dpi=100)
    plt.close()
    
    if hasattr(self, 'frame_files'):
        self.frame_files.append(save_path)

def reset_semantic_map(self):
    print("[HabitatEnv] Resetting semantic map (clearing mapper state)...")
    if hasattr(self, 'mapper'):
        self.robot_belief = self.mapper.reset()
    
    if hasattr(self, 'belief_info'):
        self.belief_info.update_map_info(self.robot_belief, self.belief_origin_x, self.belief_origin_y)
    
    self.global_frontiers = []
    self.explored_rate = 0
