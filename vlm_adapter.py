import matplotlib.pyplot as plt
import io
import numpy as np
import torch
from PIL import Image
from utils import get_cell_position_from_coords

class VLMAdapter:
    def __init__(self, model_name="gpt-4o", api_key=None, base_url=None):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.provider = self._infer_provider(model_name)
        self.client = None
        self.use_legacy_openai = False
        
        if self.provider in ("openai", "qwen"):
            import os
            key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("QWEN_API_KEY")
            base = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("QWEN_BASE_URL")
            
            try:
                from openai import OpenAI
                if not key and base:
                    key = "EMPTY"
                self.client = OpenAI(api_key=key, base_url=base) if (key or base) else None
            except ImportError:
                # Fallback for Python 3.6 / OpenAI < 1.0.0
                try:
                    import openai
                    self.use_legacy_openai = True
                    if key: openai.api_key = key
                    if base: openai.api_base = base
                    print("[VLMAdapter] Using legacy OpenAI API (v0.28)")
                except ImportError:
                    self.client = None
                    print("[VLMAdapter] OpenAI library not found")
        elif self.provider == "gemini":
            try:
                import google.generativeai as genai
                import os
                key = api_key or os.getenv("GOOGLE_API_KEY")
                genai.configure(api_key=key)
                self.client = genai.GenerativeModel(model_name)
            except Exception:
                self.client = None
        elif self.provider == "anthropic":
            try:
                import anthropic
                import os
                key = api_key or os.getenv("ANTHROPIC_API_KEY")
                self.client = anthropic.Anthropic(api_key=key) if key else None
            except Exception:
                self.client = None

    def get_vlm_action(self, agent, observation, stairs_coords=None, rgb_image=None, debug_save_path=None):
        """
        使用 VLM 的逻辑替换 agent.select_next_waypoint。
        rgb_image: Optional 3D first-person view from Habitat (numpy array)
        debug_save_path: Optional path to save the rendered visualization for debugging
        """
        candidate_indices = agent.neighbor_indices
        candidate_indices = np.array([idx for idx in candidate_indices if idx != agent.current_index])
        if candidate_indices.size == 0:
            candidate_indices = np.array(agent.neighbor_indices)

        scores = []
        valid_indices = []
        for idx in candidate_indices:
            coords = agent.node_coords[idx]
            node_wrapper = agent.node_manager.nodes_dict.find((coords[0], coords[1]))
            if node_wrapper is None:
                continue
            node = node_wrapper.data
            base_utility = float(node.utility)
            visit_count = float(getattr(node, "visit_count", 0))
            
            # --- 启发式修正 ---
            # 计算距离
            dist = np.linalg.norm(coords - agent.location)
            # 引入距离和访问次数的权重
            # score = base_utility / (1.0 + visit_count) # 原逻辑
            
            # 新逻辑: 综合 utility, 距离, 访问次数
            # 增加 distance 因子 (倾向于去更远的地方)
            # 增加 visit_count 惩罚 (倾向于去没去过的地方)
            alpha = 0.5
            beta = 1.0
            heuristic_weight = (1.0 + alpha * dist) / (1.0 + beta * visit_count)
            
            score = (base_utility * heuristic_weight) / (1.0 + visit_count)
            
            scores.append(score)
            valid_indices.append(idx)

        if len(valid_indices) > 0:
            scores = np.array(scores)
            valid_indices = np.array(valid_indices, dtype=int)
            if np.any(scores > 0):
                mask = scores > 0
                scores = scores[mask]
                valid_indices = valid_indices[mask]
            order = np.argsort(-scores)
            valid_indices = valid_indices[order]
            max_candidates = min(10, len(valid_indices))
            candidate_indices = valid_indices[:max_candidates]
        else:
            candidate_indices = np.array(agent.neighbor_indices)

        # 2. 渲染带有候选节点标记的地图
        image_bytes = self.render_map_with_candidates(agent, candidate_indices, stairs_coords, rgb_image=rgb_image)
        
        # Debug: Save the image if path is provided
        if debug_save_path:
            try:
                img = Image.open(io.BytesIO(image_bytes))
                img.save(debug_save_path)
                print(f"[VLMAdapter] Saved debug visualization to {debug_save_path}")
            except Exception as e:
                print(f"[VLMAdapter] Failed to save debug image: {e}")

        prompt = self.construct_hierarchical_prompt(len(candidate_indices), has_3d_image=(rgb_image is not None))
        
        decision_idx = 0
        try:
            print(f"[VLMAdapter] Calling VLM API with {len(candidate_indices)} candidates...")
            response_text = self.call_vlm_api(image_bytes, prompt)
            print(f"[VLMAdapter] Raw VLM Response: {response_text}")
            
            # Use the unified parsing method
            decision_idx = self.parse_response(response_text, len(candidate_indices))

        except Exception as e:
            print(f"VLM query failed: {e}. Defaulting to heuristic.")
            decision_idx = 0
            
        action_index = candidate_indices[decision_idx]
        next_location = agent.node_coords[action_index]
        
        return (
            next_location,
            torch.tensor([[action_index]]).long()
        )

    def render_map_with_candidates(self, agent, candidate_indices, stairs_coords=None, rgb_image=None):
        """
        生成当前信念地图的图像，包含机器人和编号的候选节点。
        如果提供了 rgb_image (3D 视图)，则将其与地图并排显示。
        """
        # 切换后端以避免显示窗口
        plt.switch_backend('agg')
        plt.close('all')
        
        # Adjust figure size if we have two images
        if rgb_image is not None:
            fig = plt.figure(figsize=(20, 10))
            ax_map = plt.subplot(1, 2, 1)
            ax_3d = plt.subplot(1, 2, 2)
        else:
            fig = plt.figure(figsize=(10, 10))
            ax_map = plt.gca()
        
        # --- 1. 绘制地图 (在 ax_map 上) ---
        # 255=自由区域, 1=障碍物, 127=未知区域
        ax_map.imshow(agent.map_info.map, cmap='gray', origin='upper')
        
        # 绘制未知区域边界 (Frontiers) - GREEN dots
        if hasattr(agent, 'frontier') and agent.frontier:
            frontier_coords = np.array(list(agent.frontier))
            if len(frontier_coords) > 0:
                frontier_cells = get_cell_position_from_coords(frontier_coords, agent.map_info).reshape(-1, 2)
                ax_map.scatter(frontier_cells[:, 0], frontier_cells[:, 1], c='green', s=10, marker='.', label='Frontiers', zorder=3)

        # 绘制机器人 - RED star
        robot_cell = get_cell_position_from_coords(agent.location, agent.map_info)
        ax_map.plot(robot_cell[0], robot_cell[1], marker='*', color='red', markersize=15, label='Robot', zorder=10)

        # 绘制楼梯 (如果存在) - YELLOW star
        if stairs_coords is not None and len(stairs_coords) > 0:
            stairs_coords_arr = np.array(stairs_coords)
            stairs_cells = get_cell_position_from_coords(stairs_coords_arr, agent.map_info).reshape(-1, 2)
            ax_map.scatter(stairs_cells[:, 0], stairs_cells[:, 1], c='yellow', marker='*', s=300, edgecolors='black', label='Stairs', zorder=11)
        
        # 绘制候选节点 - BLUE circles
        candidate_coords = agent.node_coords[candidate_indices]
        candidate_cells = get_cell_position_from_coords(candidate_coords, agent.map_info).reshape(-1, 2)
        
        ax_map.scatter(candidate_cells[:, 0], candidate_cells[:, 1], c='blue', s=100, edgecolors='white', zorder=5, label='Candidates')
        
        # 为它们编号
        for i, (x, y) in enumerate(candidate_cells):
            ax_map.text(x, y, str(i), color='white', fontsize=12, fontweight='bold', zorder=6, ha='center', va='center')
            
        ax_map.legend()
        ax_map.axis('off')
        ax_map.set_title("Top-Down Map with Candidates")

        # --- 2. 绘制 3D 图像 (在 ax_3d 上) ---
        if rgb_image is not None:
            ax_3d.imshow(rgb_image)
            ax_3d.axis('off')
            ax_3d.set_title("Front-View Camera (3D)")

        plt.tight_layout()
        
        # 保存为字节流
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        
        return buf.getvalue()

    def construct_hierarchical_prompt(self, num_candidates, has_3d_image=False):
        base_prompt = f"""
You are a robot explorer in a multi-floor building. Your goal is to explore as much area as possible.
I will show you a top-down map of your current surroundings.
- The RED star indicates your current location.
- The BLUE circles with numbers (0 to {num_candidates-1}) are reachable candidate waypoints.
- The GREEN dots are unexplored frontiers (areas you haven't seen yet).
- The YELLOW star (if present) indicates a known stair location leading to another floor.
"""
        if has_3d_image:
            base_prompt += """
I have also provided your Front-View Camera (3D) image.
- Use this 3D view to identify open doors, hallways, or obstacles that might not be clear on the map.
- If the 3D view shows a clear path or stairs that corresponds to a candidate waypoint, prefer that candidate.
"""
        
        base_prompt += f"""
Your task is to select the best candidate waypoint (0-{num_candidates-1}) to maximize exploration.
Consider the following strategy:
1. If there are many GREEN frontiers nearby, choose a waypoint that leads to them to explore this floor.
2. If this floor is mostly explored (few GREEN frontiers), and you see a YELLOW stair, try to move towards the stair to switch floors.
3. Avoid going back to fully explored areas unless necessary to reach new frontiers or stairs.

Please output ONLY the number of the selected candidate waypoint (e.g., "3").
"""
        return base_prompt

    def call_vlm_api(self, image_bytes, prompt):
        import base64
        # 强制修正模型名称，防止传参错误
        actual_model = self.model_name
        if actual_model in ["openai", "qwen", "gemini", "anthropic", "OpenAI"]:
            if self.provider == "openai": actual_model = "gpt-4o"
            elif self.provider == "qwen": actual_model = "qwen-vl-max"
        
        if self.provider in ("openai", "qwen"):
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            messages = [{"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                    ]}]
            
            if self.client:
                try:
                    resp = self.client.chat.completions.create(
                        model=actual_model,
                        messages=messages
                    )
                    return resp.choices[0].message.content
                except Exception as e:
                    print(f"[VLMAdapter] OpenAI error: {e}")
            elif self.use_legacy_openai:
                try:
                    import openai
                    resp = openai.ChatCompletion.create(
                        model=actual_model,
                        messages=messages
                    )
                    return resp['choices'][0]['message']['content']
                except Exception as e:
                    print(f"[VLMAdapter] Legacy OpenAI error: {e}")
        if self.provider == "gemini" and self.client:
            try:
                resp = self.client.generate_content([prompt, {"mime_type": "image/png", "data": image_bytes}])
                return getattr(resp, "text", str(resp))
            except Exception as e:
                print(f"[VLMAdapter] Gemini error: {e}")
        if self.provider == "anthropic" and self.client:
            try:
                b64 = base64.b64encode(image_bytes).decode("utf-8")
                resp = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=30,
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}}
                    ]}]
                )
                return resp.content[0].text if hasattr(resp, "content") else str(resp)
            except Exception as e:
                print(f"[VLMAdapter] Anthropic error: {e}")
        return self._mock_choice(prompt)

    def _infer_provider(self, model_name):
        m = model_name.lower()
        if m.startswith("gpt-") or "gpt-4o" in m:
            return "openai"
        if "qwen" in m:
            return "qwen"
        if m.startswith("gemini") or "gemini" in m:
            return "gemini"
        if m.startswith("claude") or "anthropic" in m:
            return "anthropic"
        return "openai"

    def _mock_choice(self, prompt):
        import random, re
        match = re.search(r'\(0 to (\d+)\)', prompt)
        if match:
            max_idx = int(match.group(1))
            return str(random.randint(0, max_idx))
        return "0"

    def parse_response(self, response_text, num_candidates):
        import re
        # 在响应中查找第一个整数
        match = re.search(r'\d+', response_text)
        if match:
            idx = int(match.group())
            if 0 <= idx < num_candidates:
                return idx
        
        print(f"[VLMAdapter] 警告: 无效响应 '{response_text}'。默认为 0。")
        return 0
