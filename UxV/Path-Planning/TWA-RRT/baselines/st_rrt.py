# ST-RRT* benchmark implementation
# Grothe, F., Hartmann, V. N., Orthey, A., and Toussaint, M. (2022). ST-RRT*: Asymptotically-Optimal Bidirectional Motion Planning through Space-Time. ICRA.
# https://doi.org/10.1109/ICRA46639.2022.9811814
# Original-author code reuse and independent reimplementation status: unverified.
# Benchmark changes: static 2D circles, single tree instead of original bidirectional planner; constant-speed final evaluation; common JSON and metrics.
# Redistribution license: unresolved; root MIT scope excludes baselines.
# Publication preparation changes only names, references and file/output paths.

# Created: 2025-06-12
# Modified: 2026-02-01 (ST-RRT* paperish, TB-RRT runner-compatible)
# Modified: 2026-02-18 (JSON config support, unified metrics with TB-RRT/TWA-RRT*)
# Modified: 2026-02-28 (R4: Constant Velocity Patch - Enforced T = L/V_MAX)
# NOTE: The paper includes dynamic obstacles, bidirectional trees, and goal trees.
#       This comparison implementation uses static obstacles and a single tree.

import math
import logging
import time
import matplotlib.pyplot as plt
import os
import csv
import datetime
import random
import numpy as np
import json
import argparse
import sys
import copy

# ============ Parameter section (TB-RRT compatible layout) ============
SEED = 42
WORLD = (0, 220, 0, 160)                 # xmin, xmax, ymin, ymax
OBSTACLES = [(60,60,13),(120,80,14),(160,110,12)]
START = (30.0, 30.0, 0.0)
TRIALS = 5

BASE_SEED = SEED

# ST-RRT* goal region (unknown arrival time)
GOAL_CENTER = (180.0, 120.0)
GOAL_RADIUS = 5.0

# ST-RRT* paperish parameters
LAMBDA = 0.5
V_MAX = 30.0
TARGET_TIME = 7.5  # Target arrival time for comparison (TB-RRT goal)
T_MAX_INIT = 12.0
T_MAX_CAP = 12.0
RANGE_FACTOR = 1.5
BATCH_SIZE = 250
P_GOAL = 0.15
MAX_COND_RESAMPLES = 40

# RRT* neighbor radius base
NEIGHBOR_RADIUS_BASE = 35.0

random.seed(SEED)
np.random.seed(SEED)

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Path length definition for CSV: sum of spatial distances (sum ||dq||)

# ============ REFERENCE HEADER (TWA-RRT* Standard) ============
REFERENCE_HEADER = [
    'case_id', 'trial', 'algo', 'success', 'runtime_s', 'path_len', 'arrival_t',
    'spatial_err', 'time_err', 'total_err', 'time_window_violation', 'nodes', 'seed',
    'first_sol_time', 'first_path_len', 'first_total_err', 'smoothness',
    't_ref', 'L_proxy', 'Tmin', 'Tmax', 'alpha', 'beta', 'kappa',
    'p_tw', 'loiter_used', 'loiter_len', 'late_pruned', 'lb_pruned'
]

# ============ Time Constraint Parsing ============
def parse_time_constraint(tc_dict):
    """Parse a time_constraint dict from JSON config.

    Returns a normalised dict with keys:
        type: 'window' | 'target' | None
        t_lo, t_hi: float (for window)
        target_time: float (for target / window centre)
    Returns None when there is no time constraint.
    """
    if tc_dict is None:
        return None

    tc_type = tc_dict.get("type", "window")

    if tc_type == "window":
        t_lo = tc_dict["t_lo"]
        t_hi = tc_dict["t_hi"]
        target = tc_dict.get("target_time", (t_lo + t_hi) / 2.0)
        return {"type": "window", "t_lo": t_lo, "t_hi": t_hi, "target_time": target}

    if tc_type == "target":
        target = tc_dict["target_time"]
        eps = tc_dict.get("epsilon", None)
        if eps is not None:
            return {"type": "window", "t_lo": target - eps, "t_hi": target + eps, "target_time": target}
        return {"type": "target", "target_time": target, "t_lo": None, "t_hi": None}

    return None


# ============ Unified Metric Helpers ============
class Dynamics:
    """Robot Dynamics Constraints (for Feasibility Check)"""
    @staticmethod
    def feasible_time_range(L: float, v0: float, v_min: float, v_max: float, a_max: float):
        """
        Returns (t_min, t_max) to traverse length L.
        t_min: Fastest possible (Accel to v_max, cruise).
        t_max: Slowest possible (Decel to v_min, cruise).
        """
        # --- t_min (Fastest) ---
        t_acc = (v_max - v0) / a_max
        d_acc = (v_max**2 - v0**2) / (2 * a_max)
        
        if d_acc >= L:
            val = v0**2 + 2*a_max*L
            # val < 0 check? v0^2 >= 0, a_max > 0, L >= 0 -> val >= 0
            t_min = (-v0 + math.sqrt(val)) / a_max
        else:
            d_cruise = L - d_acc
            t_cruise = d_cruise / v_max
            t_min = t_acc + t_cruise
            
        # --- t_max (Slowest) ---
        t_dec = (v0 - v_min) / a_max
        d_dec = (v0**2 - v_min**2) / (2 * a_max)
        
        if d_dec >= L:
            val = v0**2 - 2*a_max*L
            if val < 0: val = 0 
            t_max = (v0 - math.sqrt(val)) / a_max
        else:
            d_cruise = L - d_dec
            t_cruise = d_cruise / v_min
            t_max = t_dec + t_cruise
            
        return t_min, t_max

    @staticmethod
    def time_window_violation(tmin, tmax, t_window):
        t1, t2 = t_window
        if tmax < t1: return t1 - tmax
        if tmin > t2: return tmin - t2
        return 0.0

    @staticmethod
    def get_arrival_t_best(tmin, tmax, t_ref, t_window):
        t1, t2 = t_window
        # Check feasibility (overlap)
        if not (tmax < t1 or tmin > t2):
            # Feasible: clamp t_ref to [tmin, tmax]
            return max(tmin, min(tmax, t_ref))
        else:
            # Infeasible
            if tmax < t1: return tmax # Too Early
            if tmin > t2: return tmin # Too Late
        return t_ref 

# Helper: Path Smoothness
def calculate_smoothness(path) -> float:
    """Calculate total turning angle (radians) of the path."""
    if not path or len(path) < 3:
        return 0.0
    total_angle = 0.0
    for i in range(1, len(path)-1):
        # path nodes have .x, .y
        n_prev = path[i-1]
        n_curr = path[i]
        n_next = path[i+1]
        
        dx1 = n_curr.x - n_prev.x
        dy1 = n_curr.y - n_prev.y
        dx2 = n_next.x - n_curr.x
        dy2 = n_next.y - n_curr.y
        
        angle1 = math.atan2(dy1, dx1)
        angle2 = math.atan2(dy2, dx2)
        diff = abs(angle2 - angle1)
        if diff > math.pi:
            diff = 2*math.pi - diff
        total_angle += diff
    return total_angle 

def compute_spatial_err(x, y, goal_center, goal_radius):
    """spatial_err = max(0, dist - goal_radius). 0 inside goal region."""
    dist = math.hypot(x - goal_center[0], y - goal_center[1])
    return max(0.0, dist - goal_radius)


def compute_approach_spatial_err(node, goal_center):
    """Approach-distance spatial_err (TB-RRT compatible).

    When a node lands inside the goal region, reporting 0 is not informative.
    Instead, report the distance from the *approach node* (parent, or first
    non-goal ancestor) to goal_center.  If the node is NOT inside the goal
    region, simply return the Euclidean distance to goal_center.
    """
    gx, gy = goal_center
    dist_self = math.hypot(node.x - gx, node.y - gy)
    if dist_self > 1e-3:
        return dist_self  # node did not land exactly on goal
    # Node is essentially at goal -> walk back to find approach node
    cur = node.parent
    while cur is not None:
        d = math.hypot(cur.x - gx, cur.y - gy)
        if d > 1e-3:
            return d
        cur = cur.parent
    return dist_self  # fallback (all ancestors at goal – unlikely)


def compute_time_err(arrival_t, tc):
    """Compute time_err given a parsed time_constraint dict *tc*.
    - window: 0 inside [t_lo, t_hi], else min distance to boundaries
    - target (no eps): |arrival - target|
    - None: 0.0
    """
    if tc is None:
        return 0.0
    # if tc["type"] == "window":
    #     target = tc.get("target_time", (tc["t_lo"] + tc["t_hi"]) / 2.0)
    #     return abs(arrival_t - target)
    # if tc["type"] == "target":
    #     return abs(arrival_t - tc["target_time"])
    return abs(arrival_t - tc["target_time"])

def compute_total_err(spatial_err, time_err):
    return math.sqrt(spatial_err ** 2 + time_err ** 2)

def compute_success(spatial_err, arrival_t, tc, eps_spatial=1e-6):
    if spatial_err is None or spatial_err > eps_spatial:
        return False
    if tc is None:
        return True
    if tc["type"] == "window":
        return tc["t_lo"] <= arrival_t <= tc["t_hi"]
    return abs(arrival_t - tc["target_time"]) < 1e-9

def best_sort_key(spatial_err, time_err, total_err, arrival_t, tc):
    """Sorting key for best-solution selection (lower is better).

    Primary: total_err
    Secondary (window): |arrival - window_center|
    Tertiary: arrival_t
    """
    if tc is not None and tc["type"] == "window":
        t_target = tc.get("target_time", (tc["t_lo"] + tc["t_hi"]) / 2.0)
        return (total_err, abs(arrival_t - t_target), arrival_t)
    return (total_err, arrival_t, 0.0)

def find_best_candidate_node(tree, goal_center, goal_radius, tc):
    """Scan entire tree for the node with minimum total_err.

    Used to report metrics when the planner fails to find a success solution.
    For failure cases, spatial_err = raw distance to goal center (TB-RRT compatible).
    Returns (node, spatial_err, time_err, total_err) or (None,...) if tree empty.
    """
    best_node = None
    best_key = None
    best_se = None
    best_te = None
    best_tot = None

    for node in tree:
        # Use region-distance for sorting (to find best candidate)
        se_region = compute_spatial_err(node.x, node.y, goal_center, goal_radius)
        te = compute_time_err(node.t, tc)
        tot_region = compute_total_err(se_region, te)
        key = best_sort_key(se_region, te, tot_region, node.t, tc)
        if best_key is None or key < best_key:
            best_key = key
            best_node = node
            # Report raw distance to goal center (TB-RRT compatible)
            best_se = math.hypot(node.x - goal_center[0], node.y - goal_center[1])
            best_te = te
            best_tot = compute_total_err(best_se, te)

    return best_node, best_se, best_te, best_tot


def compute_time_window_violation(arrival_t, tc):
    """Returns 0.0 if inside window, else the violation amount. For non-window, 0.0."""
    if tc is None or tc["type"] != "window":
        return 0.0
    t_lo, t_hi = tc["t_lo"], tc["t_hi"]
    if arrival_t < t_lo:
        return t_lo - arrival_t
    if arrival_t > t_hi:
        return arrival_t - t_hi
    return 0.0


# ============ JSON Case Loader ============
def load_config(path):
    """Load JSON config file and return the parsed dict."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_case(case: dict) -> dict:
    """
    Ensure the case dict has all necessary fields with defaults if missing.
    Also validates structure (Aligned with TB-RRT).
    """
    # 1. World
    if "world" not in case:
        case["world"] = {"xlim": [WORLD[0], WORLD[1]], "ylim": [WORLD[2], WORLD[3]]}
    
    # 2. Start/Goal
    if "start" not in case:
        case["start"] = {"x": START[0], "y": START[1]}
    if "goal" not in case:
        # Map goal_center to goal for unified format
        case["goal"] = {"x": GOAL_CENTER[0], "y": GOAL_CENTER[1]}
        
    # 3. Obstacles
    if "obstacles" not in case:
        # Convert legacy OBSTACLES list (x,y,r) to dict list
        obs_list = []
        for i, (ox, oy, r) in enumerate(OBSTACLES):
            obs_list.append({"type": "circle", "cx": ox, "cy": oy, "r": r, "id": f"O{i}"})
        case["obstacles"] = obs_list
    
    # Validate Obstacles (Ensuring dict format)
    for ob in case["obstacles"]:
        if not isinstance(ob, dict):
             # Try to fix? Or assume it's list [x,y,r]
             continue
        if ob.get("type") != "circle":
             # We only support circle for now
             pass
            
    # 4. Time
    if "time" not in case:
        # Use TARGET_TIME as default t_ref
        case["time"] = {"t_ref": TARGET_TIME, "t_window": [TARGET_TIME-0.5, TARGET_TIME+0.5]}
        
    # 5. Dynamics
    if "dynamics" not in case:
        # Use ST-RRT globals
        case["dynamics"] = {"v_min": 0.0, "v_max": V_MAX, "a_max": 2.0, "v0": 0.0}
        
    # 6. Random
    if "random" not in case:
        case["random"] = {"seed_base": SEED, "num_trials": TRIALS}
        
    # 7. Planner (Common)
    if "planner" not in case:
        case["planner"] = {
            "n_iters": 6000, 
            "goal_sample_rate": P_GOAL,
            "step_size": 3.0,
            "neighbor_radius": NEIGHBOR_RADIUS_BASE
        }
        
    # 8. Algorithms (ST-RRT specific)
    if "algorithms" not in case:
        case["algorithms"] = {}
    if "st_rrt" not in case["algorithms"]:
        case["algorithms"]["st_rrt"] = {
            "eps_state": 5.0,
            "lambda": LAMBDA
        }
    
    # 9. Outputs
    if "outputs" not in case:
        case["outputs"] = {"out_dir": "output"}
        
    return case


# ============ Node class ============
class Node:
    def __init__(self, x: float, y: float, t: float = 0.0, parent=None, cost: float = 0.0):
        self.x = x
        self.y = y
        self.t = t
        self.parent = parent
        self.cost = cost
        self.children = set()
        if parent:
            parent.children.add(self)

    def set_parent(self, new_parent):
        if self.parent:
            self.parent.children.discard(self)
        self.parent = new_parent
        if new_parent:
            new_parent.children.add(self)

    def __repr__(self):
        return f"Node(x={self.x:.2f}, y={self.y:.2f}, t={self.t:.2f}, cost={self.cost:.2f})"


def propagate_cost_to_leaves(start_node: Node, v_max: float = None):
    """Iteratively update time/cost for children after a parent's time improved."""
    if v_max is None:
        v_max = V_MAX
        
    stack = [start_node]
    while stack:
        curr = stack.pop()
        for child in list(curr.children):
            dist = dQ_xy(curr, child)
            min_dt = dist / v_max
            new_child_t = curr.t + min_dt

            if new_child_t < child.t - 1e-9:
                child.t = new_child_t
                child.cost = new_child_t
                stack.append(child)


# ============ Helper functions ============
def dQ_xy(a: Node, b: Node) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    return math.hypot(dx, dy)


def pseudo_distance(a: Node, b: Node, lam: float = LAMBDA, v_max: float = V_MAX) -> float:
    dt = b.t - a.t
    if dt <= 0.0:
        return float('inf')
    d_q = dQ_xy(a, b)
    v_required = d_q / dt
    if v_required > v_max:
        return float('inf')
    return lam * d_q + (1.0 - lam) * dt


def compute_neighbor_radius(n: int, base_radius: float = NEIGHBOR_RADIUS_BASE) -> float:
    if n < 2:
        return base_radius
    return base_radius * math.sqrt(math.log(n) / n)


def is_ancestor(node: Node, potential_child: Node) -> bool:
    current = potential_child
    while current:
        if current is node:
            return True
        current = current.parent
    return False


def line_circle_collision(p1, p2, center, radius):
    (x1,y1), (x2,y2) = p1, p2
    cx, cy = center
    dx, dy = x2-x1, y2-y1
    denom = dx*dx + dy*dy
    t = 0.0 if denom < 1e-12 else max(0.0, min(1.0, ((cx-x1)*dx + (cy-y1)*dy)/denom))
    px, py = x1 + t*dx, y1 + t*dy
    return math.hypot(px-cx, py-cy) <= radius


def collision_free_xy(a: Node, b: Node, world=None, obstacles=None) -> bool:
    w = world if world is not None else WORLD
    obs = obstacles if obstacles is not None else OBSTACLES
    if not (w[0] <= b.x <= w[1] and w[2] <= b.y <= w[3]):
        return False
    for (ox,oy,r) in obs:
        if line_circle_collision((a.x,a.y), (b.x,b.y), (ox,oy), r):
            return False
    return True


def reconstruct_path(end_node: Node) -> list:
    path, node = [], end_node
    while node:
        path.append(node)
        node = node.parent
    return list(reversed(path))


def path_length(path: list) -> float:
    length = 0.0
    for i in range(1, len(path)):
        length += dQ_xy(path[i-1], path[i])
    return length


def neighbors_pseudo(nodes: list, target: Node, radius: float,
                     lam: float = LAMBDA, v_max: float = V_MAX) -> list:
    out = []
    for n in nodes:
        if pseudo_distance(n, target, lam, v_max) <= radius:
            out.append(n)
    return out


def sample_goal_region(goal_center, goal_radius) -> Node:
    ang = random.uniform(0.0, 2.0 * math.pi)
    rad = math.sqrt(random.uniform(0.0, 1.0)) * goal_radius
    gx = goal_center[0] + rad * math.cos(ang)
    gy = goal_center[1] + rad * math.sin(ang)
    return Node(gx, gy, 0.0)


# Helper: Path Smoothness
def calculate_smoothness(path: list) -> float:
    """Calculate total turning angle (radians) of the path."""
    if not path or len(path) < 3:
        return 0.0
    total_angle = 0.0
    for i in range(1, len(path)-1):
        dx1 = path[i].x - path[i-1].x
        dy1 = path[i].y - path[i-1].y
        dx2 = path[i+1].x - path[i].x
        dy2 = path[i+1].y - path[i].y

        angle1 = math.atan2(dy1, dx1)
        angle2 = math.atan2(dy2, dx2)
        diff = abs(angle2 - angle1)
        # Normalize to [0, pi]
        if diff > math.pi:
            diff = 2*math.pi - diff
        total_angle += diff
    return total_angle


def st_rrt_paperish(start: Node,
                    goal_center, goal_radius,
                    target_time,
                    max_iterations: int,
                    step_size: float,
                    max_nodes: int = 10000,
                    time_constraint=None,
                    lam: float = None,
                    v_max: float = None,
                    neighbor_radius_base: float = None,
                    t_max_init: float = None,
                    t_max_cap: float = None,
                    range_factor: float = None,
                    batch_size: int = None,
                    p_goal: float = None,
                    max_cond_resamples: int = None,
                    world=None,
                    obstacles=None):
    """ST-RRT* paperish planner with unified metric helpers.

    Parameters use global defaults when None.
    """
    # Resolve defaults from globals
    if lam is None: lam = LAMBDA
    if v_max is None: v_max = V_MAX
    if neighbor_radius_base is None: neighbor_radius_base = NEIGHBOR_RADIUS_BASE
    if t_max_init is None: t_max_init = T_MAX_INIT
    if t_max_cap is None: t_max_cap = T_MAX_CAP
    if range_factor is None: range_factor = RANGE_FACTOR
    if batch_size is None: batch_size = BATCH_SIZE
    if p_goal is None: p_goal = P_GOAL
    if max_cond_resamples is None: max_cond_resamples = MAX_COND_RESAMPLES
    if world is None: world = WORLD
    if obstacles is None: obstacles = OBSTACLES

    # Parse time constraint for unified metrics
    tc = parse_time_constraint(time_constraint)

    start_time_algo = time.time()
    tree = [Node(start.x, start.y, start.t, parent=None, cost=start.t)]
    best_goal_node = None
    best_goal_key = None  # sort key for best solution selection
    best_spatial_err = None
    best_time_err = None
    best_total_err = None
    first_sol_time = None
    first_path_len = None
    first_spatial_err = None
    first_time_err = None
    first_total_err = None

    cost_history = []  # list of (time, cost)

    t_max = t_max_init
    batch_counter = 0
    goal_samples = []  # list of (x,y,t)

    for i in range(max_iterations):
        # Progressive time bound expansion
        if batch_counter >= batch_size and best_goal_node is None:
            t_max = min(t_max * range_factor, t_max_cap)
            batch_counter = 0

        # (1) Sample
        sample = None
        for _ in range(max_cond_resamples):
            if random.random() < p_goal:
                q_goal = sample_goal_region(goal_center, goal_radius)
                t_min_sample = start.t + dQ_xy(start, q_goal) / v_max
                if t_min_sample >= t_max:
                    continue
                t_goal = random.uniform(t_min_sample, t_max)
                g = Node(q_goal.x, q_goal.y, t_goal)
                goal_samples.append((g.x, g.y, g.t))
                sample = g
                break

            # Conditional sampling
            xmin, xmax_w, ymin, ymax_w = world
            qx = random.uniform(xmin, xmax_w)
            qy = random.uniform(ymin, ymax_w)
            q = Node(qx, qy, 0.0)
            t_min_sample = start.t + dQ_xy(start, q) / v_max

            if not goal_samples:
                continue

            t_ub = -float('inf')
            for (gx, gy, gt) in goal_samples:
                qg = Node(gx, gy, 0.0)
                t_ub = max(t_ub, gt - dQ_xy(q, qg) / v_max)

            t_ub = min(t_ub, t_max)
            if t_min_sample >= t_ub:
                continue

            t_s = random.uniform(t_min_sample, t_ub)
            sample = Node(qx, qy, t_s)
            break

        # Fallback sample if conditional fails
        if sample is None:
            xmin, xmax_w, ymin, ymax_w = world
            qx = random.uniform(xmin, xmax_w)
            qy = random.uniform(ymin, ymax_w)
            q = Node(qx, qy, 0.0)
            t_min_sample = start.t + dQ_xy(start, q) / v_max
            if t_min_sample < t_max:
                t_s = random.uniform(t_min_sample, t_max)
                sample = Node(qx, qy, t_s)

        if sample is None:
            batch_counter += 1
            continue

        # (2) Nearest
        nearest = min(tree, key=lambda node: pseudo_distance(node, sample, lam, v_max))
        if pseudo_distance(nearest, sample, lam, v_max) == float('inf'):
            # fallback to spatial nearest
            nearest = min(tree, key=lambda node: dQ_xy(node, sample))

        # (3) Steer (interpolated)
        if pseudo_distance(nearest, sample, lam, v_max) == float('inf'):
            batch_counter += 1
            continue

        d_xy = dQ_xy(nearest, sample)
        if d_xy > step_size:
            ratio = step_size / d_xy
            nx = nearest.x + (sample.x - nearest.x) * ratio
            ny = nearest.y + (sample.y - nearest.y) * ratio
            actual_dist = math.hypot(nx - nearest.x, ny - nearest.y)
            nt = nearest.t + (actual_dist / v_max)
            new_node = Node(nx, ny, nt, parent=nearest, cost=nt)
        else:
            actual_dist = math.hypot(sample.x - nearest.x, sample.y - nearest.y)
            nt = nearest.t + (actual_dist / v_max)
            new_node = Node(sample.x, sample.y, nt, parent=nearest, cost=nt)

        # (4) Collision check
        if not collision_free_xy(nearest, new_node, world, obstacles):
            batch_counter += 1
            continue

        # (5) Parent selection (RRT*)
        r_n = compute_neighbor_radius(len(tree), neighbor_radius_base)
        cand_neighbors = neighbors_pseudo(tree, new_node, r_n, lam, v_max)
        if not cand_neighbors:
            cand_neighbors = [nearest]

        best_parent = None
        best_d = float('inf')
        for p in cand_neighbors:
            if not collision_free_xy(p, new_node, world, obstacles):
                continue
            d = pseudo_distance(p, new_node, lam, v_max)
            if d < best_d:
                best_d, best_parent = d, p

        if best_parent is None:
            batch_counter += 1
            continue

        new_node.set_parent(best_parent)
        new_node.t = best_parent.t + (dQ_xy(best_parent, new_node) / v_max)
        new_node.cost = new_node.t
        tree.append(new_node)

        # (6) Rewiring (only if q.t can be reduced)
        for q in neighbors_pseudo(tree, new_node, r_n, lam, v_max):
            if q is new_node or q is new_node.parent:
                continue
            if is_ancestor(q, new_node):
                continue
            if not collision_free_xy(new_node, q, world, obstacles):
                continue
            if pseudo_distance(new_node, q, lam, v_max) == float('inf'):
                continue
            dt_min = dQ_xy(new_node, q) / v_max
            cand_t = new_node.t + dt_min
            if cand_t + 1e-9 < q.t:
                q.set_parent(new_node)
                q.t = cand_t
                q.cost = q.t
                propagate_cost_to_leaves(q, v_max)

        # (7) Goal check – spatial-only for internal tracking
        #     (Time constraint evaluation is done in run_experiments for CSV success)
        se = compute_spatial_err(new_node.x, new_node.y, goal_center, goal_radius)
        te = compute_time_err(new_node.t, tc)
        tot = compute_total_err(se, te)
        in_goal_region = (se < 1e-9)  # spatial-only check

        # Approach-distance spatial_err for reporting (TB-RRT compatible)
        report_se = compute_approach_spatial_err(new_node, goal_center)
        report_tot = compute_total_err(report_se, te)

        if in_goal_region:
            # Use total_err as sort key so nodes closer to time target are preferred
            cur_key = best_sort_key(report_se, te, report_tot, new_node.t, tc)
            if best_goal_key is None or cur_key < best_goal_key:
                if best_goal_node is None:
                    first_sol_time = time.time() - start_time_algo
                    temp_path = reconstruct_path(new_node)
                    first_path_len = path_length(temp_path)
                    first_spatial_err = report_se
                    first_time_err = te
                    first_total_err = report_tot
                    logging.info(f"First solution found at iteration {i}: time={first_sol_time:.4f}s, len={first_path_len:.3f}, err={first_total_err:.3f}")

                best_goal_key = cur_key
                best_goal_node = new_node
                best_spatial_err = report_se
                best_time_err = te
                best_total_err = report_tot
                t_max = min(t_max, new_node.t)
                logging.info(f"Goal reached at iteration {i}: t_arrival={new_node.t:.3f}, approach_dist={report_se:.3f}")

                # History (Time from start, Current Best Cost)
                cur_t = time.time() - start_time_algo
                cost_history.append((cur_t, new_node.t))

        if len(tree) >= max_nodes:
            logging.warning(f"Max nodes {max_nodes} reached at iteration {i}")
            break

        batch_counter += 1

    if best_goal_node:
        return (reconstruct_path(best_goal_node), tree,
                best_spatial_err, best_time_err, best_total_err,
                first_sol_time, first_path_len, first_spatial_err, first_time_err, first_total_err,
                cost_history)

    logging.warning(f"Goal not reached within {max_iterations} iterations")
    
    # Best Effort: Return the node with the lowest Total Error
    best_partial_node = None
    best_partial_dist = float('inf')
    partial_spatial_err = None
    partial_time_err = None
    partial_total_err = None

    for node in tree:
        se = compute_spatial_err(node.x, node.y, goal_center, goal_radius)
        te = compute_time_err(node.t, tc)
        tot = compute_total_err(se, te)
        
        if tot < best_partial_dist:
            best_partial_dist = tot
            best_partial_node = node
            partial_spatial_err = compute_approach_spatial_err(node, goal_center) # Use approach distance
            partial_time_err = te
            partial_total_err = compute_total_err(partial_spatial_err, partial_time_err)
            
    if best_partial_node:
        # User Request TB-RRT compatibility: If exact connection but failed, report parent's distance
        if partial_spatial_err < 1e-3 and best_partial_node.parent:
             best_partial_node = best_partial_node.parent
             partial_spatial_err = compute_approach_spatial_err(best_partial_node, goal_center)
             partial_time_err = compute_time_err(best_partial_node.t, tc)
             partial_total_err = compute_total_err(partial_spatial_err, partial_time_err)
             
        logging.info(f"Best partial path: closest node at distance {best_partial_dist:.3f}")
        logging.info(f"  Partial spatial error: {partial_spatial_err:.3f}, time error: {partial_time_err:.3f}")
        partial_path = reconstruct_path(best_partial_node)
        return (partial_path, tree,
                partial_spatial_err, partial_time_err, partial_total_err,
                None, None, None, None, None, [])

    return (None, tree, None, None, None,
            None, None, None, None, None, [])



def plot_results(tree, path, start_node, goal_center, goal_radius,
                 spatial_err, time_err, total_err, elapsed_time, trial_idx, save_dir,
                 world=None, obstacles=None):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    w = world if world is not None else WORLD
    obs = obstacles if obstacles is not None else OBSTACLES

    fig, ax = plt.subplots(figsize=(6.6, 5.0), dpi=120)

    if path is None:
        title = f"Trial {trial_idx} - ST-RRT* (FAILED)"
    else:
        title = f"Trial {trial_idx} - ST-RRT*"
    ax.set_title(title)

    ax.set_xlim(w[0], w[1])
    ax.set_ylim(w[2], w[3])
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linewidth=0.4, alpha=0.4)

    for node in tree:
        if node.parent:
            ax.plot([node.x, node.parent.x], [node.y, node.parent.y], linewidth=0.4, alpha=0.20, color='#888888', zorder=1)

    for i_obs, (ox, oy, r) in enumerate(obs):
        circ = plt.Circle((ox, oy), r, fill=False, linewidth=1.2, edgecolor='k')
        ax.add_patch(circ)
        ax.text(ox, oy, f"O{i_obs:02d}", fontsize=7, ha="center", va="center", alpha=0.7)

    if path and len(path) >= 2:
        px, py, pt = zip(*[(n.x, n.y, n.t) for n in path])
        if spatial_err is not None:
            path_color = '#0072B2'
        else:
            path_color = '#FF6B6B'
        ax.plot(px, py, linewidth=2.0, label="Path", color=path_color, zorder=4)

        t_values = []
        pts = []
        for ii in range(len(path)-1):
            seg = np.linspace(np.array([path[ii].x, path[ii].y, path[ii].t]), np.array([path[ii+1].x, path[ii+1].y, path[ii+1].t]), num=10)
            if ii > 0:
                seg = seg[1:]
            for j in range(seg.shape[0]):
                pts.append((seg[j,0], seg[j,1]))
                t_values.append(seg[j,2])
        pts = np.array(pts, dtype=float)
        t_values = np.array(t_values, dtype=float)
        sc = ax.scatter(pts[:,0], pts[:,1], c=t_values, s=10, cmap="viridis", alpha=0.7, zorder=5)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.75)
        cbar.set_label("time (s)")

    ax.scatter(start_node.x, start_node.y, marker='o', s=90, color='#0099FF', edgecolor='k', label='Start', zorder=10)
    ax.scatter(goal_center[0], goal_center[1], marker='*', s=180, color='#FF9900', edgecolor='k', label='Goal Center', zorder=11)

    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, f"trial_{trial_idx:03d}_2d.png"))
    plt.close(fig)


def run_case(case: dict, out_root: str = None, save_plots: bool = True):
    """
    Run a single experiment case for ST-RRT*.
    """
    global WORLD, OBSTACLES, GOAL_CENTER, GOAL_RADIUS, START, SEED, BASE_SEED, TRIALS
    global LAMBDA, V_MAX, TARGET_TIME, T_MAX_INIT, T_MAX_CAP, RANGE_FACTOR
    global BATCH_SIZE, P_GOAL, MAX_COND_RESAMPLES, NEIGHBOR_RADIUS_BASE

    # 1. Parameter Extraction
    # 1. Parameter Extraction
    eff = normalize_case(case) if case else None
    if eff is None:
        # Fallback to globals
        eff = {
            "case_id": "default",
            "world": {"xlim": [WORLD[0], WORLD[1]], "ylim": [WORLD[2], WORLD[3]]},
            "start": {"x": START[0], "y": START[1]},
            "goal": {"x": GOAL_CENTER[0], "y": GOAL_CENTER[1]},
            "obstacles": [],
            "time": {"t_ref": TARGET_TIME, "t_window": [TARGET_TIME-0.5, TARGET_TIME+0.5]},
            "dynamics": {"v_min": 0.0, "v_max": V_MAX, "a_max": 2.0, "v0": 0.0},
            "random": {"seed_base": SEED, "num_trials": TRIALS},
            "planner": {"step_size": 3.0, "neighbor_radius": 35.0, "n_iters": 6000},
            "algorithms": {"st_rrt": {"eps_state": 5.0, "dt_min": 0.1}},
            "outputs": {"out_dir": "output"}
        }
    
    c_id = eff["case_id"]
    
    # 2. Extract Parameters (Aligned with TB-RRT)
    w = eff["world"]
    s = eff["start"]
    g = eff["goal"]
    obs_list = eff["obstacles"]
    tm = eff["time"]
    dyn = eff["dynamics"]
    rnd = eff["random"]
    pln = eff["planner"]
    algo_params = eff["algorithms"].get("st_rrt", {})
    
    l_world = (w["xlim"][0], w["xlim"][1], w["ylim"][0], w["ylim"][1])
    l_start = (s["x"], s["y"], s.get("t", 0.0))
    l_goal_center = (g["x"], g["y"])
    # goal_radius in ST-RRT usually from epsilon or specific algo param
    l_goal_radius = algo_params.get("eps_state", 5.0) 
    
    l_seed = rnd.get("seed_base", SEED)
    l_trials = rnd.get("num_trials", TRIALS)
    
    # Dynamics
    l_v_min, l_v_max, l_a_max, l_v0 = dyn["v_min"], dyn["v_max"], dyn["a_max"], dyn["v0"]
    l_t_ref = tm["t_ref"]
    l_t_window = tuple(tm["t_window"])
    
    # Planner
    l_mi = pln.get("n_iters", 6000)
    l_ss = pln.get("step_size", 3.0)
    l_nrb = pln.get("neighbor_radius", 35.0)
    l_pg = pln.get("goal_sample_rate", 0.15)
    
    # ST-RRT specific Algos
    l_lam = algo_params.get("lambda", LAMBDA)
    l_eps_state = algo_params.get("eps_state", 5.0)
    l_tmi = algo_params.get("t_max_init", T_MAX_INIT)
    l_tmc = algo_params.get("t_max_cap", T_MAX_CAP)
    l_rf = algo_params.get("range_factor", RANGE_FACTOR)
    l_bs = algo_params.get("batch_size", BATCH_SIZE)
    l_mcr = algo_params.get("max_cond_resamples", MAX_COND_RESAMPLES)
    l_mn = algo_params.get("max_nodes", 10000)

    # Time Constraint parsing for st_rrt_paperish
    tc_raw = {"type": "window", "t_lo": l_t_window[0], "t_hi": l_t_window[1], "target_time": l_t_ref}
    t_ref_val = l_t_ref
    t_min_val, t_max_val = l_t_window

    # Set globals (legacy fallback)
    WORLD = l_world
    OBSTACLES = [(o["cx"], o["cy"], o["r"]) for o in obs_list]
    l_obstacles = OBSTACLES # for loop
    GOAL_CENTER = l_goal_center
    GOAL_RADIUS = l_goal_radius
    NEIGHBOR_RADIUS_BASE = l_nrb
    LAMBDA = l_lam
    V_MAX = l_v_max

    # 3. Output Directory
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Use out_root from CLI if provided, else from normalized config
    if out_root is None:
        out_root = eff.get("outputs", {}).get("out_dir", "output")
    
    if not os.path.isabs(out_root):
        # Place relative to script parent or root? TB-RRT usually uses script dir or relative to it.
        # Let's follow TB-RRT: relative to script_dir.
        out_root = os.path.join(script_dir, out_root)

    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_root, c_id, "st_rrt", f"run_{run_timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Save config
    config_used = eff
    with open(os.path.join(run_dir, "config_used.json"), 'w', encoding='utf-8') as cf:
        json.dump(config_used, cf, indent=2, ensure_ascii=False)

    csv_file = os.path.join(run_dir, "summary.csv")
    history_file = os.path.join(run_dir, "cost_history.csv")

    with open(history_file, mode='w', newline='') as hf:
        h_writer = csv.writer(hf)
        h_writer.writerow(['trial', 'time', 'cost'])

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=REFERENCE_HEADER, extrasaction='ignore')
        writer.writeheader()

        for trial in range(1, l_trials + 1):
            trial_seed = l_seed + trial - 1
            random.seed(trial_seed)
            np.random.seed(trial_seed)

            start_node = Node(*l_start)

            start_time = time.time()
            path, tree, spatial_err_algo, time_err_algo, total_err_algo, \
                first_sol_time, first_path_len_algo, \
                first_spatial_err_algo, first_time_err_algo, first_total_err_algo, \
                cost_history = st_rrt_paperish(
                    start_node, l_goal_center, l_goal_radius,
                    t_ref_val if t_ref_val != "" else TARGET_TIME,
                    max_iterations=l_mi,
                    step_size=l_ss,
                    max_nodes=l_mn,
                    time_constraint=tc_raw,
                    lam=l_lam,
                    v_max=l_v_max,
                    neighbor_radius_base=l_nrb,
                    t_max_init=l_tmi,
                    t_max_cap=l_tmc,
                    range_factor=l_rf,
                    batch_size=l_bs,
                    p_goal=l_pg,
                    max_cond_resamples=l_mcr,
                    world=l_world,
                    obstacles=l_obstacles,
                )
            elapsed_time = time.time() - start_time

            with open(history_file, mode='a', newline='') as hf:
                h_writer = csv.writer(hf)
                for (ht, hc) in cost_history:
                    h_writer.writerow([trial, f"{ht:.4f}", f"{hc:.4f}"])

            # --- Unified Metrics (Option A: Actual Arrival Time) ---
            tc = parse_time_constraint(tc_raw)

            res_path_len = None
            res_arrival_t = None
            res_spatial_err = None
            res_time_err = None
            res_total_err = None
            res_success = 0
            res_twv = None
            res_tmin = None
            res_tmax = None
            res_L_proxy = ""     # raw/approach distance for interpretation
            res_smoothness = ""

            # First Solution placeholders (kept compatible with header)
            f_sol_time = ""
            f_path_len = ""
            f_total_err = ""

            if path:
                # 1) Path metrics
                res_path_len = path_length(path)
                res_arrival_t = path[-1].t

                # 2) Spatial error (Approach-distance, TB-RRT compatible)
                last = path[-1]
                res_spatial_err = compute_approach_spatial_err(last, l_goal_center)

                # 4) Time error (Tracking): |arrival_t - t_ref|
                res_time_err = compute_time_err(res_arrival_t, tc)

                # 5) Time-window violation and success (unified)
                res_twv = compute_time_window_violation(res_arrival_t, tc)
                is_time_ok = (res_twv < 1e-6)
                is_spatial_ok = (res_spatial_err <= l_eps_state)
                res_success = 1 if (is_time_ok and is_spatial_ok) else 0

                # 6) Total error
                res_total_err = math.hypot(res_spatial_err, res_time_err)

                # 6) Tmin/Tmax: for constant-velocity ST-RRT, set both to L/v_max (informative & consistent)
                #    (If you prefer blanks, set res_tmin/res_tmax = None)
                res_tmin = res_path_len / l_v_max
                res_tmax = res_path_len / l_v_max

                # 7) Smoothness
                res_smoothness = calculate_smoothness(path)

            else:
                # Failure case: report best candidate in tree for consistent logging
                # (uses region-distance for ranking; reports raw center distance in best_se)
                best_node, best_se_raw, best_te, best_tot = find_best_candidate_node(tree, l_goal_center, l_goal_radius, tc)

                if best_node is not None:
                    partial_path = reconstruct_path(best_node)
                    res_path_len = path_length(partial_path)
                    res_arrival_t = best_node.t

                    # spatial_err unified (region-distance)
                    res_spatial_err = best_se_raw
                    res_time_err = compute_time_err(res_arrival_t, tc)
                    res_total_err = compute_total_err(res_spatial_err, res_time_err)
                    res_success = 0
                    res_twv = compute_time_window_violation(res_arrival_t, tc)

                    # proxy: raw center distance for interpretability
                    res_L_proxy = f"{best_se_raw:.3f}" if best_se_raw is not None else ""

                    res_tmin = res_path_len / l_v_max if res_path_len is not None else None
                    res_tmax = res_tmin
                    res_smoothness = calculate_smoothness(partial_path)
                else:
                    # complete failure: keep blanks
                    res_smoothness = ""

            # First solution fields (keep your existing behavior, but make totals consistent if possible)
            if first_sol_time:
                f_sol_time = f"{first_sol_time:.4f}"
                f_path_len = f"{first_path_len_algo:.3f}" if first_path_len_algo is not None else ""
                f_total_err = f"{first_total_err_algo:.3f}" if first_total_err_algo is not None else ""
            elif path and res_path_len is not None:
                f_sol_time = f"{elapsed_time:.4f}"
                f_path_len = f"{res_path_len:.3f}"
                f_total_err = f"{res_total_err:.3f}" if res_total_err is not None else ""
            
            row = {
                'case_id': c_id,
                'trial': trial,
                'algo': 'st_rrt_r4',
                'success': res_success,
                'runtime_s': f"{elapsed_time:.4f}",
                'path_len': f"{res_path_len:.3f}" if res_path_len is not None else "",
                'arrival_t': f"{res_arrival_t:.3f}" if res_arrival_t is not None else "",
                'spatial_err': f"{res_spatial_err:.3f}" if res_spatial_err is not None else "",
                'time_err': f"{res_time_err:.3f}" if res_time_err is not None else "",
                'total_err': f"{res_total_err:.3f}" if res_total_err is not None else "",
                'time_window_violation': f"{res_twv:.3f}" if res_twv is not None else "",
                'nodes': len(tree),
                'seed': trial_seed,
                'first_sol_time': f_sol_time,
                'first_path_len': f_path_len,
                'first_total_err': f_total_err,
                'smoothness': f"{res_smoothness:.3f}" if res_smoothness != "" else "",
                't_ref': f"{t_ref_val}" if t_ref_val != "" else "",
                'L_proxy': res_L_proxy,
                'Tmin': f"{res_tmin:.3f}" if res_tmin is not None else "",
                'Tmax': f"{res_tmax:.3f}" if res_tmax is not None else "",
                'alpha': "", 'beta': "", 'kappa': "",
                'p_tw': "", 'loiter_used': 0, 'loiter_len': 0,
                'late_pruned': 0, 'lb_pruned': 0,
            }
            writer.writerow(row)
            
            if save_plots:
                 plot_results(tree, path, start_node, l_goal_center, l_goal_radius,
                              res_spatial_err, res_time_err, res_total_err, elapsed_time, trial, run_dir,
                              world=l_world, obstacles=l_obstacles)
                              
    logging.info(f"Experiments completed for Case {c_id}")

def main():
    parser = argparse.ArgumentParser(description="Run ST-RRT* Experiments")
    parser.add_argument("--cases", type=str, help="Path to JSON cases file", default="benchmarks/benchmark_cases_2.json")
    parser.add_argument("--case_id", type=str, help="Filter by case_id")
    parser.add_argument("--out_dir", type=str, help="Output root directory")
    parser.add_argument("--no_plots", action="store_true", help="Disable plotting")
    args = parser.parse_args()

    # Load Cases
    try:
        # Use simple json load for now, assuming standard format
        with open(args.cases, 'r') as f:
            data = json.load(f)
            cases = data.get("cases", [])
    except Exception as e:
        print(f"Error loading cases from {args.cases}: {e}")
        # Try finding it in parent dir if not found
        parent_path = os.path.join(os.path.dirname(__file__), "..", args.cases)
        if os.path.exists(parent_path):
             with open(parent_path, 'r') as f:
                data = json.load(f)
                cases = data.get("cases", [])
        else:
             return

    # Run for each case that includes 'st_rrt' in algorithms
    for case in cases:
        if args.case_id and case["case_id"] != args.case_id:
            continue
        if "algorithms" in case and "st_rrt" in case["algorithms"]:
            print(f"Running Case: {case['case_id']}")
            run_case(case, out_root=args.out_dir, save_plots=not args.no_plots)

if __name__ == "__main__":
    main()
