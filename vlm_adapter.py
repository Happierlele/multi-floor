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
        self.http_fallback = False
        self._http_api_key = None
        self._http_base_url = None
        self._last_single_candidate = None
        self._single_repeat_count = 0
        
        if self.provider in ("openai", "qwen"):
            import os
            key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("QWEN_API_KEY")
            base = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("QWEN_BASE_URL")

            def _sanitize_base_url(u):
                if u is None:
                    return None
                s = str(u).strip()
                if (s.startswith(("`", "'", "\"")) and s.endswith(("`", "'", "\"")) and len(s) >= 2):
                    s = s[1:-1].strip()
                s = s.strip().strip("`").strip()
                return s or None

            def _sanitize_key(k):
                if k is None:
                    return None
                s = str(k).strip()
                if (s.startswith(("`", "'", "\"", "“", "‘")) and s.endswith(("`", "'", "\"", "”", "’")) and len(s) >= 2):
                    s = s[1:-1].strip()
                s = s.strip().strip("`").strip()
                return s or None

            base = _sanitize_base_url(base)
            key = _sanitize_key(key)
            self.base_url = base
            self._http_api_key = key
            self._http_base_url = base
            
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
                    if key and base:
                        self.http_fallback = True
                        print("[VLMAdapter] OpenAI library not found. Using HTTP fallback.", flush=True)
                    else:
                        print("[VLMAdapter] OpenAI library not found", flush=True)
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

    def get_vlm_action(self, agent, observation, stairs_coords=None, rgb_image=None, debug_save_path=None, position_history=None, current_floor=None):
        """
        使用 VLM 的逻辑替换 agent.select_next_waypoint。
        rgb_image: Optional 3D first-person view from Habitat (numpy array)
        debug_save_path: Optional path to save the rendered visualization for debugging
        position_history: List of previous positions (tuples or lists) to avoid ping-pong loops
        current_floor: Optional floor ID to filter history by floor
        """
        candidate_indices = agent.neighbor_indices
        current_index = getattr(agent, "current_index", None)
        if current_index is not None:
            candidate_indices = np.array([idx for idx in candidate_indices if idx != current_index])
        else:
            candidate_indices = np.array(candidate_indices)
        if candidate_indices.size == 0:
            candidate_indices = np.array(agent.neighbor_indices)

        # --- Filter out recently visited nodes (Anti-Ping-Pong) ---
        if position_history and len(position_history) > 1 and candidate_indices.size > 0:
             # Get recent positions (last 10 steps, EXCLUDING current position to allow moving away)
             # Current position is typically the last element in position_history
             recent_positions = position_history[:-1][-10:]
             
             # Calculate "last seen step" for each candidate
             # We want to prefer candidates that were NOT seen, or seen longest ago.
             candidate_last_seen = {}
             candidate_visit_counts = {} # Also check global visit counts if possible

             # Try to access node manager for global visit counts
             node_manager = getattr(agent, "node_manager", None)
             
             for idx in candidate_indices:
                 coords = agent.node_coords[idx]
                 p1 = coords[:2] if len(coords) >= 2 else coords
                 
                 # Check global visit count
                 visit_count = 0
                 if node_manager:
                     key = (round(float(p1[0]), 1), round(float(p1[1]), 1))
                     node = node_manager.nodes_dict.find(key)
                     if node:
                         visit_count = node.data.visit_count
                 candidate_visit_counts[idx] = visit_count

                 last_seen = -1 # Never seen
                 # Check against history (newest is last in list)
                 for i, h_pos in enumerate(recent_positions):
                     p2 = np.array(h_pos)[:2]
                     
                     # Check floor if provided and available in history
                     match_floor = True
                     if current_floor is not None and len(h_pos) > 2:
                         if h_pos[2] != current_floor:
                             match_floor = False

                     if match_floor and np.linalg.norm(p1 - p2) < 1.0:
                         last_seen = i # Store index (0=oldest in window, len-1=newest)
                 
                 candidate_last_seen[idx] = last_seen
             
             # Filter 1: Keep only candidates that were NOT seen in recent history
             filtered_indices = [idx for idx in candidate_indices if candidate_last_seen[idx] == -1]
             
             # Filter 2: Among remaining candidates, prefer LOW visit counts
             if len(filtered_indices) > 0:
                 # Sort by visit count
                 filtered_indices.sort(key=lambda idx: candidate_visit_counts.get(idx, 0))
                 
                 # If we have multiple candidates with LOW visits (e.g. 0 or 1), keep only those
                 min_visits = candidate_visit_counts.get(filtered_indices[0], 0)
                 best_indices = [idx for idx in filtered_indices if candidate_visit_counts.get(idx, 0) <= min_visits + 1] # Allow small margin
                 
                 candidate_indices = np.array(best_indices)
                 print(f"[VLMAdapter] Filtered out recently visited & high-visit nodes. Remaining candidates: {len(candidate_indices)} (Min visits: {min_visits})")
             else:
                fallback_sorted = sorted(
                    list(candidate_indices),
                    key=lambda idx: (candidate_last_seen.get(idx, 10**9), candidate_visit_counts.get(idx, 0)),
                )
                chosen_idx = int(fallback_sorted[0]) if fallback_sorted else None
                if chosen_idx is None:
                    return (None, torch.tensor([[0]]).long())
                candidate_indices = np.array([chosen_idx], dtype=int)
                print(f"[VLMAdapter] All candidates were recently visited. Falling back to least-recent candidate: {chosen_idx}")
        # ---------------------------------------------------------

        if not hasattr(agent, "node_manager") or agent.node_manager is None or not hasattr(agent.node_manager, "nodes_dict"):
            candidate_indices = np.array(candidate_indices[: min(10, len(candidate_indices))], dtype=int)
            image_bytes = self.render_map_with_candidates(agent, candidate_indices, stairs_coords, rgb_image=rgb_image, current_floor=current_floor)
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
                decision_idx = self.parse_response(response_text, len(candidate_indices))
            except Exception as e:
                print(f"VLM query failed: {e}. Defaulting to heuristic.")
                decision_idx = 0

            action_index = int(candidate_indices[decision_idx])
            next_location = agent.node_coords[action_index]
            return (
                next_location,
                torch.tensor([[action_index]]).long()
            )

        if candidate_indices.size > 0:
            coords_arr = np.array([agent.node_coords[int(i)] for i in candidate_indices])
            mask = np.linalg.norm(coords_arr - agent.location, axis=1) > 0.1
            candidate_indices = candidate_indices[mask]
        if candidate_indices.size == 0:
            return (None, torch.tensor([[0]]).long())
        if candidate_indices.size == 1:
            single_idx = int(candidate_indices[0])
            coords = agent.node_coords[single_idx]
            if self._last_single_candidate is not None and np.linalg.norm(np.array(self._last_single_candidate) - np.array(coords)) < 0.5:
                self._single_repeat_count += 1
            else:
                self._single_repeat_count = 0
            self._last_single_candidate = coords
            if self._single_repeat_count >= 2:
                print(f"[VLMAdapter] Single candidate repeated. Returning NONE to trigger random jump fallback.")
                return (None, torch.tensor([[0]]).long())
            print(f"[VLMAdapter] Only 1 candidate available: {single_idx} at {coords}")
            return (coords, torch.tensor([[single_idx]]).long())

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
        image_bytes = self.render_map_with_candidates(agent, candidate_indices, stairs_coords, rgb_image=rgb_image, position_history=position_history, current_floor=current_floor)
        
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

    def render_map_with_candidates(self, agent, candidate_indices, stairs_coords=None, rgb_image=None, position_history=None, current_floor=0):
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
        map_img = getattr(agent.map_info, "vlm_rgb_map", None)
        if map_img is not None:
            ax_map.imshow(map_img, origin='upper')
        else:
            ax_map.imshow(agent.map_info.map, cmap='gray', origin='upper')
        
        # 绘制未知区域边界 (Frontiers) - GREEN dots
        if hasattr(agent, 'frontier') and agent.frontier:
            frontier_coords = np.array(list(agent.frontier))
            if len(frontier_coords) > 0:
                frontier_cells = get_cell_position_from_coords(frontier_coords, agent.map_info).reshape(-1, 2)
                ax_map.scatter(frontier_cells[:, 0], frontier_cells[:, 1], c='green', s=10, marker='.', label='Frontiers', zorder=3)
        
        # 绘制历史轨迹 (History Trajectory) - CYAN line
        # Use passed position_history if available, otherwise check agent
        hist = position_history if position_history is not None else getattr(agent, 'position_history', [])
        
        if hist:
            # Filter history by current floor if needed
            # Assuming history stores (x, y, floor) or just (x, y)
            hist_points = []
            # Use passed current_floor
            
            for p in hist:
                if len(p) >= 3 and p[2] != current_floor:
                    continue
                hist_points.append(p[:2])
                
            if len(hist_points) > 1:
                hist_arr = np.array(hist_points)
                hist_cells = get_cell_position_from_coords(hist_arr, agent.map_info).reshape(-1, 2)
                # Plot faint line
                ax_map.plot(hist_cells[:, 0], hist_cells[:, 1], c='cyan', linewidth=2, alpha=0.6, label='History', zorder=2)
                # Plot end point (current loc) is already done by Red Star

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
- The CYAN line traces your recent path (History). AVOID going back to areas covered by the cyan line unless necessary.
- The YELLOW star (if present) indicates a known stair location leading to another floor.
- If obstacle cells are color-coded (not grayscale): they encode obstacle height (a z-layer hint). Prefer candidates that move toward useful height transitions (e.g., near stairs/ramps) when this floor is already well explored.
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
            elif self.http_fallback:
                try:
                    import json
                    import urllib.request

                    base = (self._http_base_url or "").strip().rstrip("/")
                    if not base:
                        return None
                    url = base + "/chat/completions"
                    payload = {
                        "model": actual_model,
                        "messages": messages,
                    }
                    data = json.dumps(payload).encode("utf-8")
                    headers = {"Content-Type": "application/json"}
                    if self._http_api_key:
                        headers["Authorization"] = f"Bearer {self._http_api_key}"
                    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        body = resp.read().decode("utf-8", errors="replace")
                    obj = json.loads(body)
                    choices = obj.get("choices") or []
                    if choices and isinstance(choices, list):
                        msg = (choices[0] or {}).get("message") or {}
                        content = msg.get("content")
                        return content
                except Exception as e:
                    print(f"[VLMAdapter] HTTP fallback error: {e}")
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

    def detect_stairs_in_rgb(self, rgb_image):
        if rgb_image is None:
            return {"is_stairs": None, "confidence": None, "raw": None}
        has_backend = bool(self.client) or bool(self.use_legacy_openai) or bool(self.http_fallback) or (self.provider == "gemini" and self.client) or (self.provider == "anthropic" and self.client)
        if not has_backend:
            return {"is_stairs": None, "confidence": None, "raw": None}

        try:
            arr = np.asarray(rgb_image)
            if arr.ndim != 3 or arr.shape[-1] < 3:
                return {"is_stairs": None, "confidence": None, "raw": None}
            arr = arr[..., :3]
            if arr.dtype != np.uint8:
                amax = float(np.nanmax(arr)) if np.size(arr) else 1.0
                if np.isfinite(amax) and amax <= 1.05:
                    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
                else:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
            try:
                a = arr.astype(np.float32, copy=False)
                p2 = np.nanpercentile(a, 2, axis=(0, 1))
                p98 = np.nanpercentile(a, 98, axis=(0, 1))
                rng = np.maximum(1.0, (p98 - p2))
                a = (a - p2) * (255.0 / rng)
                a = np.clip(a, 0.0, 255.0)
                if float(np.nanmean(p98 - p2)) < 45.0:
                    gamma = 0.85
                    a = 255.0 * np.power(np.clip(a / 255.0, 0.0, 1.0), gamma)
                arr = a.astype(np.uint8)
            except Exception:
                pass

            h, w = int(arr.shape[0]), int(arr.shape[1])
            y0 = int(max(0, round(0.30 * h)))
            x0 = int(max(0, round(0.15 * w)))
            x1 = int(min(w, round(0.85 * w)))
            crop = arr[y0:h, x0:x1, :]
            img_full = Image.fromarray(arr)
            img_crop = Image.fromarray(crop) if crop.size else img_full
            try:
                img_crop = img_crop.resize((img_full.size[0], img_full.size[1]))
            except Exception:
                img_crop = img_full

            try:
                composite = Image.new("RGB", (img_full.size[0] * 2, img_full.size[1]))
                composite.paste(img_full, (0, 0))
                composite.paste(img_crop, (img_full.size[0], 0))
                img = composite
            except Exception:
                img = img_full
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            image_bytes = buf.getvalue()
        except Exception:
            return {"is_stairs": None, "confidence": None, "raw": None}

        prompt = (
            "你是室内机器人导航的视觉检验员。请判断图中是否清晰出现“楼梯/台阶/楼梯间”(staircase/steps)，可以让机器人上/下楼。\n"
            "注意：地毯纹理、栏杆、桌椅、门框、阴影、地砖线条、斜坡不算楼梯。\n"
            "请只输出一行 JSON（不要输出其它内容）：\n"
            "{\"stairs\": true/false, \"confidence\": 0.0-1.0, \"reason\": \"短语\"}\n"
        )
        raw = None
        try:
            raw = self.call_vlm_api(image_bytes, prompt)
        except Exception:
            raw = None

        if not raw:
            return {"is_stairs": None, "confidence": None, "raw": raw}

        text0 = str(raw).strip()
        text = text0.lower()
        is_stairs = None
        conf = None

        try:
            import json, re
            m = re.search(r"\{[\s\S]*\}", text0)
            if m:
                obj = json.loads(m.group(0))
                v = obj.get("stairs", obj.get("is_stairs", None))
                if isinstance(v, str):
                    v2 = v.strip().lower()
                    if v2 in ("true", "yes", "y", "1", "是", "有"):
                        v = True
                    elif v2 in ("false", "no", "n", "0", "否", "没有", "不是"):
                        v = False
                    else:
                        v = None
                if isinstance(v, bool):
                    is_stairs = bool(v)
                c = obj.get("confidence", None)
                if c is not None:
                    conf = float(c)
                    conf = float(max(0.0, min(1.0, conf)))
        except Exception:
            is_stairs = None
            conf = None

        if "stairs: yes" in text or text.startswith("yes") or "楼梯" in text and ("有" in text or "是" in text):
            is_stairs = True
        if "stairs: no" in text or text.startswith("no") or ("楼梯" in text and ("没有" in text or "否" in text or "不是" in text)):
            is_stairs = False

        if conf is None:
            try:
                import re
                m = re.search(r'confidence\s*[:=]\s*([01](?:\.\d+)?)', text)
                if m:
                    conf = float(m.group(1))
                    conf = float(max(0.0, min(1.0, conf)))
            except Exception:
                conf = None

        return {"is_stairs": is_stairs, "confidence": conf, "raw": raw}

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
