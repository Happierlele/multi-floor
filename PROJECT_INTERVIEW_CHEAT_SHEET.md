# Project Interview Cheat Sheet: Habitat-MP3D Hierarchical Exploration

## 1. Project Overview
**Goal**: Autonomous exploration and mapping of 3D indoor environments (Matterport3D) using the Habitat Simulator.
**Core Tech**: PyTorch, Habitat-Sim, Ray (Distributed), hierarchical graph-based planning.
**Key Mechanism**: The agent builds a **Topological Graph** (nodes = navigable areas) overlaid on a **Metric Belief Map** (grid occupancy) to plan long-term goals while navigating locally.

## 2. Architecture & Key Components

### A. The Worker (`worker_habitat.py`)
- **Role**: The "Driver" for a single environment instance.
- **Responsibility**: Initializes the simulator, manages the agent loop, handles sensor observations (RGB-D), and executes actions.
- **Critical Logic**: **Emergency Recovery**. When the agent gets stuck or path planning fails, this module attempts to recover.
    - *Old Logic*: Naively checked Euclidean distance to neighbors. Resulted in teleporting through walls.
    - *New Logic*: **Tiered Recovery System**.
        1. **Tier 1 (Safe)**: Check A* reachability on the belief map (`get_grid_path_check`).
        2. **Tier 2 (Risky)**: If no reachable nodes, try close Euclidean neighbors (< 3.0m).
        3. **Tier 3 (Ultimate)**: Random navigable point teleport (prevent infinite loops).

### B. Environment Wrapper (`habitat_env.py`)
- **Role**: Bridges the raw Simulator and the Agent.
- **Key Feature**: **Belief Map Construction**. Converts depth sensors into a 2D occupancy grid.
- **Recent Additions**:
    - `get_grid_path_check()`: A fast A* utility to verify if a straight-line connection is actually traversable in the current map belief.

### C. Graph Managers (`node_manager.py` & `ground_truth_node_manager.py`)
- **Role**: Maintains the "memory" of where the agent has been.
- **Data Structure**: **QuadTree** for efficient spatial queries (finding nearest node).
- **Key Challenges**:
    - **Floating Point Precision**: "Off-grid" nodes caused lookups to fail.
    - **Fix**: Implemented **Robust Nearest-Neighbor Lookup** (using `np.argmin` and relaxed thresholds) instead of exact coordinate matching.
    - **Mode**: `GroundTruthNodeManager` runs in **BELIEF-ONLY** mode (dummy data) if ground truth is disabled, requiring careful handling of empty returns to avoid crashes.

## 3. "War Stories" - Technical Challenges Solved

### Challenge 1: The "Ghosting" Agent
**Symptom**: Agent would get stuck near walls or try to walk through them during recovery.
**Root Cause**: Recovery logic used Euclidean distance. A node 0.5m away might be on the other side of a wall.
**Solution**: Implemented **Geodesic/A* Validation**. Before teleporting/recovering to a node, we now ask the Belief Map: "Is there a valid grid path to this node?"

### Challenge 2: The "Black Screen" of Death (Headless Rendering)
**Symptom**: Running on remote HPC/Headless servers resulted in all-black RGB observations.
**Root Cause**: Conflict between NVIDIA drivers, GLVND, and EGL contexts.
**Solution**:
- Explicitly set `EGL_PLATFORM=device`.
- Filter `CUDA_VISIBLE_DEVICES` within Ray workers (Ray masks GPUs, confusing EGL).
- Preload `libEGL.so` but *not* `libGL.so` to avoid legacy GLX conflicts.

### Challenge 3: Graph Stagnation
**Symptom**: "No valid neighbors to move to" / "Critical Error: No nodes in graph".
**Root Cause**: Strict connectivity rules. If an agent drifted slightly off a node center, it lost connectivity to the graph.
**Solution**:
- **Forced Connectivity**: If an agent is "off-grid", we force-link it to the top-3 nearest neighbors.
- **Robust Lookup**: Increased search radius for neighbors from 2.0m to 3.0m to handle sparse graphs.

## 4. Key Algorithms / Math
- **Voronoi / Topological Graph**: Used for high-level decision making (where to explore next).
- **A* (A-Star)**: Used for local path planning on the occupancy grid.
- **Ray**: Used for parallelizing rollouts. Actors (Workers) run environments; Learner updates the policy.

## 5. Deployment / Ops
- **Platform**: Linux / Windows (WSL).
- **Environment**: Conda (Python 3.9+ for Bullet Physics support).
- **HPC**: Slurm integration.
- **Dependencies**: `habitat-sim` (headless), `torch` (CUDA), `magnum`.
