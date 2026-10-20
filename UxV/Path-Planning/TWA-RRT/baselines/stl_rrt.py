# STL-RRT* benchmark implementation
# Linard, A., Torre, I., Bartoli, E., Sleat, A., Leite, I., and Tumova, J. (2023). Real-Time RRT* with Signal Temporal Logic Preferences. IROS, 8621-8627.
# https://doi.org/10.1109/IROS55552.2023.10341993
# Original-author code reuse and independent reimplementation status: unverified.
# Benchmark changes: static 2D planning and arrival-window cost; fixed speed; common JSON, goal tolerance, metrics and failure fallback. This is not a verified copy of the social-navigation author implementation.
# Redistribution license: unresolved; root MIT scope excludes baselines.
# Publication preparation changes only names, references and file/output paths.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STL-guided RRT* — Refactored for ST-RRT* benchmark runner compatibility
========================================================================
- STL cost-based RRT* search logic is preserved.
- Experiment runner (I/O, metrics, CLI) aligned with ST-RRT* conventions:
  * JSON config loading (--config / --case / --out_dir / --no_plots)
  * Unified summary.csv with REFERENCE_HEADER
  * Unified metric helpers (spatial_err, time_err, total_err, success)
  * Best-candidate fallback for failure trials
  * config_used.json saved per run

JSON config example for STL-RRT specific parameters:
  "algorithms": {
    "stl_rrt": {
      "step_size": 5.0,
      "neighbor_radius": 45.0,
      "n_iters": 6000,
      "goal_sample_rate": 0.15,
      "speed": 24.0,
      "dt_min": 0.1,
      "stl_weight": 0.6
    }
  }
"""

from __future__ import annotations

import os
import csv
import math
import time
import pathlib
import json
import argparse
import sys
import copy
import logging
import datetime
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import random
import numpy as np

# ============ Parameter section (ST-RRT compatible layout) ============
SEED = 42
WORLD = (0, 220, 0, 160)                 # xmin, xmax, ymin, ymax
OBSTACLES = [(60,60,13),(120,80,14),(160,110,12)]
START = (30.0, 30.0, 0.0)
TRIALS = 5

GOAL_CENTER = (180.0, 120.0)
GOAL_RADIUS = 5.0

# STL-RRT* default planner parameters
STEP_SIZE = 5.0
NEIGHBOR_RADIUS = 45.0
N_ITERS = 6000
GOAL_SAMPLE_RATE = 0.15
SPEED = 24.0
DT_MIN = 0.1
STL_WEIGHT = 0.6
EPS_STATE = 5.0

random.seed(SEED)
np.random.seed(SEED)

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    Returns a normalised dict with keys: type, t_lo, t_hi, target_time.
    Returns None when there is no time constraint.
    """
    if tc_dict is None:
        return None
    tc_type = tc_dict.get("type", "window")
    if tc_type == "window":
        t_lo = tc_dict["t_lo"]
        t_hi = tc_dict["t_hi"]
        return {"type": "window", "t_lo": t_lo, "t_hi": t_hi, "target_time": (t_lo + t_hi) / 2.0}
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

def compute_spatial_err(x, y, goal_center, goal_radius):
    """spatial_err = max(0, dist - goal_radius). 0 inside goal region."""
    dist = math.hypot(x - goal_center[0], y - goal_center[1])
    return max(0.0, dist - goal_radius)

def compute_approach_spatial_err_idx(nodes, node_idx, goal_center):
    """Approach-distance spatial_err for index-based Node tree."""
    node = nodes[node_idx]
    gx, gy = goal_center
    dist_self = math.hypot(node.x - gx, node.y - gy)
    if dist_self > 1e-3:
        return dist_self
    cur_idx = node.parent
    while cur_idx is not None:
        cur = nodes[cur_idx]
        d = math.hypot(cur.x - gx, cur.y - gy)
        if d > 1e-3:
            return d
        cur_idx = cur.parent
    return dist_self

def compute_time_err(arrival_t, tc):
    """Compute time_err (absolute deviation from target)."""
    if tc is None:
        return 0.0
    # Always return distance to center/target
    return abs(arrival_t - tc["target_time"])

def compute_total_err(spatial_err, time_err):
    return math.sqrt(spatial_err ** 2 + time_err ** 2)

def compute_success(spatial_err, time_err, tc):
    """success = 1 iff spatial_err == 0 AND time_err <= window_radius."""
    if spatial_err > 1e-9:
        return False
    if tc is None:
        return True
    if tc["type"] == "window":
        half_width = (tc["t_hi"] - tc["t_lo"]) / 2.0
        return time_err <= half_width + 1e-9
    return time_err < 1e-9

def best_sort_key(spatial_err, time_err, total_err, arrival_t, tc):
    """Sorting key for best-solution selection (lower is better)."""
    if tc is not None and tc["type"] == "window":
        return (total_err, abs(arrival_t - tc["target_time"]), arrival_t)
    return (total_err, arrival_t, 0.0)

def find_best_candidate_node_idx(nodes, goal_center, goal_radius, tc):
    """Scan entire tree for the node with minimum total_err (index-based)."""
    best_idx = None
    best_key = None
    best_se = None
    best_te = None
    best_tot = None
    for i, node in enumerate(nodes):
        se_region = compute_spatial_err(node.x, node.y, goal_center, goal_radius)
        node_t = node.metrics[3]
        te = compute_time_err(node_t, tc)
        tot_region = compute_total_err(se_region, te)
        key = best_sort_key(se_region, te, tot_region, node_t, tc)
        if best_key is None or key < best_key:
            best_key = key
            best_idx = i
            best_se = compute_approach_spatial_err_idx(nodes, i, goal_center)
            best_te = te
            best_tot = compute_total_err(best_se, te)
            
    # Backtrack Best_idx to its parent if it landed exactly at goal (spatial_err < 1e-3)
    # This ensures the partial path correctly matches the approach node.
    if best_idx is not None:
        best_node = nodes[best_idx]
        raw_dist = math.hypot(best_node.x - goal_center[0], best_node.y - goal_center[1])
        if raw_dist < 1e-3 and best_node.parent is not None:
             # Backtrack to parent
             best_idx = best_node.parent
             parent_node = nodes[best_idx]
             best_se = compute_approach_spatial_err_idx(nodes, best_idx, goal_center)
             best_te = compute_time_err(parent_node.metrics[3], tc)
             best_tot = compute_total_err(best_se, best_te)

    return best_idx, best_se, best_te, best_tot

def compute_time_window_violation(arrival_t, tc):
    """Returns 0.0 if inside window, else the violation amount."""
    if tc is None or tc["type"] != "window":
        return 0.0
    t_lo, t_hi = tc["t_lo"], tc["t_hi"]
    if arrival_t < t_lo:
        return t_lo - arrival_t
    if arrival_t > t_hi:
        return arrival_t - t_hi
    return 0.0

# ============ Path Helpers (index-based) ============
def reconstruct_path_idx(nodes, end_idx):
    """Reconstruct path as list of node indices from end_idx to root."""
    path = []
    cur = end_idx
    seen = set()
    while cur is not None:
        if cur in seen:
            break
        seen.add(cur)
        path.append(cur)
        cur = nodes[cur].parent
    path.reverse()
    return path

def path_length_idx(nodes, path_indices):
    """Sum of Euclidean XY distances along a path of node indices."""
    length = 0.0
    for i in range(1, len(path_indices)):
        a = nodes[path_indices[i-1]]
        b = nodes[path_indices[i]]
        length += math.hypot(a.x - b.x, a.y - b.y)
    return length

def calculate_smoothness_idx(nodes, path_indices):
    """Total turning angle (radians) along a path of node indices."""
    if len(path_indices) < 3:
        return 0.0
    total_angle = 0.0
    for i in range(1, len(path_indices) - 1):
        n0 = nodes[path_indices[i-1]]
        n1 = nodes[path_indices[i]]
        n2 = nodes[path_indices[i+1]]
        dx1 = n1.x - n0.x
        dy1 = n1.y - n0.y
        dx2 = n2.x - n1.x
        dy2 = n2.y - n1.y
        a1 = math.atan2(dy1, dx1)
        a2 = math.atan2(dy2, dx2)
        diff = abs(a2 - a1)
        if diff > math.pi:
            diff = 2 * math.pi - diff
        total_angle += diff
    return total_angle

# ============ JSON Case Loader ============
def load_config(path):
    """Load JSON config file and return the parsed dict."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def normalize_case(case: dict) -> dict:
    """
    Ensure the case dict has all necessary fields with defaults if missing.
    Also validates structure (Aligned with TB-RRT/ST-RRT).
    """
    # 1. World
    if "world" not in case:
        case["world"] = {"xlim": [WORLD[0], WORLD[1]], "ylim": [WORLD[2], WORLD[3]]}
    
    # 2. Start/Goal
    if "start" not in case:
        case["start"] = {"x": START[0], "y": START[1]}
    if "goal" not in case:
        case["goal"] = {"x": GOAL_CENTER[0], "y": GOAL_CENTER[1]}
        
    # 3. Obstacles
    if "obstacles" not in case:
        # Convert legacy OBSTACLES list (x,y,r) to dict list
        obs_list = []
        for i, (ox, oy, r) in enumerate(OBSTACLES):
            obs_list.append({"type": "circle", "cx": ox, "cy": oy, "r": r, "id": f"O{i}"})
        case["obstacles"] = obs_list
    
    # 4. Time
    if "time" not in case:
        case["time"] = {"t_ref": "", "t_window": [7.0, 8.0]}
        
    # 5. Dynamics
    if "dynamics" not in case:
        case["dynamics"] = {"v_min": 0.0, "v_max": SPEED, "a_max": 2.0, "v0": 0.0}
        
    # 6. Random
    if "random" not in case:
        case["random"] = {"seed_base": SEED, "num_trials": TRIALS}
        
    # 7. Planner (Common)
    if "planner" not in case:
        case["planner"] = {
            "n_iters": N_ITERS, 
            "goal_sample_rate": GOAL_SAMPLE_RATE,
            "step_size": STEP_SIZE,
            "neighbor_radius": NEIGHBOR_RADIUS
        }
        
    # 8. Algorithms (STL-RRT specific)
    if "algorithms" not in case:
        case["algorithms"] = {}
    if "stl_rrt" not in case["algorithms"]:
        case["algorithms"]["stl_rrt"] = {
            "eps_state": EPS_STATE,
            "dt_min": DT_MIN,
            "stl_weight": STL_WEIGHT
        }
    
    # 9. Outputs
    if "outputs" not in case:
        case["outputs"] = {"out_dir": "output"}
        
    return case

# --- Optional plotting -------------------------------------------------
HAVE_MPL = False
plt = None

def _enable_matplotlib_headless() -> None:
    global HAVE_MPL, plt
    if os.environ.get("FORCE_NO_PLOTS") == "1":
        HAVE_MPL = False
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt_mod
        plt = plt_mod
        HAVE_MPL = True
    except Exception:
        HAVE_MPL = False
        plt = None  # type: ignore

_enable_matplotlib_headless()


# ---------------------------------------------------------------------
# Paths (resolve out_dir relative to this file)
# ---------------------------------------------------------------------

def _here() -> pathlib.Path:
    try:
        return pathlib.Path(__file__).resolve().parent
    except Exception:
        return pathlib.Path(os.getcwd()).resolve()

def _resolve_out_dir(out_dir: str | os.PathLike) -> pathlib.Path:
    p = pathlib.Path(out_dir)
    return p if p.is_absolute() else (_here() / p)


# ---------------------------------------------------------------------
# Geometry utils
# ---------------------------------------------------------------------

def euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(math.hypot(a[0]-b[0], a[1]-b[1]))

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def trapezoid_neg_integral(values: np.ndarray, dt: float) -> float:
    """Integrate negative part only using trapezoid rule over uniform grid spacing dt."""
    if values.size < 2:
        return 0.0
    v = np.minimum(values, 0.0)
    return float(np.sum((v[:-1] + v[1:]) * 0.5) * dt)


# ---------------------------------------------------------------------
# Clearance utilities (post-processing only)
# ---------------------------------------------------------------------

class DistanceField:
    """Distance query in world units using circular obstacles (analytic fallback)."""
    def __init__(self, world: "World"):
        self.world = world

    def distance(self, x: float, y: float) -> float:
        if not self.world.obstacles:
            return float("inf")
        min_clear = float("inf")
        for obs in self.world.obstacles:
            d = math.hypot(x - obs.x, y - obs.y) - obs.r
            if d < min_clear:
                min_clear = d
        return max(0.0, float(min_clear))


def build_distance_field(world: "World") -> DistanceField:
    return DistanceField(world)


def sample_path_points(path_xy: List[Tuple[float, float]], step: float = 1.0) -> np.ndarray:
    """Uniformly sample polyline path at spatial interval step (world units)."""
    if not path_xy or len(path_xy) == 1:
        return np.array(path_xy, dtype=float) if path_xy else np.zeros((0,2), dtype=float)
    pts: List[Tuple[float,float]] = []
    acc = 0.0
    cur = (float(path_xy[0][0]), float(path_xy[0][1]))
    pts.append(cur)
    for i in range(len(path_xy)-1):
        a = cur
        b = (float(path_xy[i+1][0]), float(path_xy[i+1][1]))
        seg_len = euclid(a, b)
        if seg_len <= 1e-9:
            continue
        dirx = (b[0] - a[0]) / seg_len
        diry = (b[1] - a[1]) / seg_len
        s = step - acc if acc > 0 else step
        while s <= seg_len + 1e-9:
            nx = a[0] + dirx * s
            ny = a[1] + diry * s
            pts.append((nx, ny))
            s += step
        acc = max(0.0, s - seg_len)
        cur = b
        if not pts or (pts[-1][0] != b[0] or pts[-1][1] != b[1]):
            pts.append(b)
    return np.array(pts, dtype=float)


def compute_clearance_metrics(
    path_xy: List[Tuple[float, float]],
    distance_field: DistanceField,
    robot_radius: float,
    theta: float,
    step: float,
    lambda_clr: float,
) -> dict:
    if not path_xy or len(path_xy) < 2:
        return {"clr_min": float("nan"), "clr_avg": float("nan"), "risk_frac_theta": float("nan"), "Jclr": float("nan")}
    samples = sample_path_points(path_xy, step=step)
    if samples.size == 0:
        return {"clr_min": float("nan"), "clr_avg": float("nan"), "risk_frac_theta": float("nan"), "Jclr": float("nan")}
    d_eff_list: List[float] = []
    deficit_sum = 0.0
    risky = 0
    for i in range(samples.shape[0]):
        x, y = float(samples[i,0]), float(samples[i,1])
        d_raw = distance_field.distance(x, y)
        d_eff = max(0.0, d_raw - robot_radius)
        d_eff_list.append(d_eff)
        if d_eff < theta:
            risky += 1
        deficit_sum += max(0.0, theta - d_eff)
    clr_min = float(min(d_eff_list)) if d_eff_list else float("nan")
    clr_avg = float(sum(d_eff_list)/len(d_eff_list)) if d_eff_list else float("nan")
    risk_frac_theta = float(risky/len(d_eff_list)) if d_eff_list else float("nan")
    Jclr = float(lambda_clr * deficit_sum * step)
    return {"clr_min": clr_min, "clr_avg": clr_avg, "risk_frac_theta": risk_frac_theta, "Jclr": Jclr}


# ---------------------------------------------------------------------
# STL fragment
# ---------------------------------------------------------------------

class STLFormula:
    def robustness(self, traj: np.ndarray, t_idx: int) -> float:
        raise NotImplementedError
    def __or__(self, other: "STLFormula") -> "STLOr":  return STLOr(self, other)
    def __and__(self, other: "STLFormula") -> "STLAnd": return STLAnd(self, other)
    def __invert__(self) -> "STLNot": return STLNot(self)

class STLPredicate(STLFormula):
    """Axis-aligned rectangle predicate. Positive inside, negative outside (signed L_inf)."""
    def __init__(self, xmin: float, xmax: float, ymin: float, ymax: float):
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
    def robustness(self, traj: np.ndarray, t_idx: int) -> float:
        x, y = float(traj[t_idx,0]), float(traj[t_idx,1])
        dx = max(self.xmin - x, 0.0, x - self.xmax)
        dy = max(self.ymin - y, 0.0, y - self.ymax)
        if dx == 0.0 and dy == 0.0:
            rx = min(x - self.xmin, self.xmax - x)
            ry = min(y - self.ymin, self.ymax - y)
            return float(min(rx, ry))
        return float(-max(dx, dy))

class STLCirclePredicate(STLFormula):
    """원형 영역 predicate: 중심 (cx,cy), 반지름 r. 내부면 양수, 외부면 음수."""
    def __init__(self, cx: float, cy: float, r: float):
        self.cx, self.cy, self.r = cx, cy, r
    def robustness(self, traj: np.ndarray, t_idx: int) -> float:
        x, y = float(traj[t_idx,0]), float(traj[t_idx,1])
        d = math.hypot(x - self.cx, y - self.cy)
        return self.r - d  # 내부면 양수, 외부면 음수

class STLNot(STLFormula):
    def __init__(self, inner: STLFormula): self.inner = inner
    def robustness(self, traj: np.ndarray, t_idx: int) -> float:
        return -self.inner.robustness(traj, t_idx)

class STLAnd(STLFormula):
    def __init__(self, l: STLFormula, r: STLFormula): self.l, self.r = l, r
    def robustness(self, traj: np.ndarray, t_idx: int) -> float:
        return float(min(self.l.robustness(traj, t_idx), self.r.robustness(traj, t_idx)))

class STLOr(STLFormula):
    def __init__(self, l: STLFormula, r: STLFormula): self.l, self.r = l, r
    def robustness(self, traj: np.ndarray, t_idx: int) -> float:
        return float(max(self.l.robustness(traj, t_idx), self.r.robustness(traj, t_idx)))

class STLEventually(STLFormula):
    """Eventually ◇ or ◇[a,b] in discrete index form (a,b are integer offsets)."""
    def __init__(self, inner: STLFormula, a: Optional[int]=None, b: Optional[int]=None):
        self.inner, self.a, self.b = inner, a, b
    def robustness(self, traj: np.ndarray, t_idx: int) -> float:
        T = traj.shape[0]
        a = self.a if self.a is not None else 0
        b = self.b if self.b is not None else (T-1)
        lo = int(clamp(t_idx + a, 0, T-1))
        hi = int(clamp(t_idx + b, 0, T-1))
        vals = [self.inner.robustness(traj, t) for t in range(lo, hi+1)]
        return float(max(vals) if vals else -float('inf'))

class STLAlways(STLFormula):
    """Always □ or □[a,b] in discrete index form (a,b are integer offsets)."""
    def __init__(self, inner: STLFormula, a: Optional[int]=None, b: Optional[int]=None):
        self.inner, self.a, self.b = inner, a, b
    def robustness(self, traj: np.ndarray, t_idx: int) -> float:
        T = traj.shape[0]
        a = self.a if self.a is not None else 0
        b = self.b if self.b is not None else (T-1)
        lo = int(clamp(t_idx + a, 0, T-1))
        hi = int(clamp(t_idx + b, 0, T-1))
        vals = [self.inner.robustness(traj, t) for t in range(lo, hi+1)]
        return float(min(vals) if vals else float('inf'))


# ---------------------------------------------------------------------
# Obstacles & World
# ---------------------------------------------------------------------

@dataclass
class CircleObstacle:
    x: float
    y: float
    r: float
    def collides_seg(self, a: Tuple[float,float], b: Tuple[float,float]) -> bool:
        ax, ay = a; bx, by = b
        cx, cy, r = self.x, self.y, self.r
        abx, aby = bx-ax, by-ay
        acx, acy = cx-ax, cy-ay
        denom = abx*abx + aby*aby
        t = 0.0
        if denom > 1e-12:
            t = max(0.0, min(1.0, (acx*abx + acy*aby)/denom))
        px, py = ax + t*abx, ay + t*aby
        return euclid((px,py),(cx,cy)) <= r

@dataclass
class World:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    obstacles: List[CircleObstacle] = field(default_factory=list)

    def in_bounds(self, p: Tuple[float,float]) -> bool:
        return (self.xmin <= p[0] <= self.xmax) and (self.ymin <= p[1] <= self.ymax)

    def collision_free(self, a: Tuple[float,float], b: Tuple[float,float]) -> bool:
        if not self.in_bounds(b):
            return False
        steps = max(2, int(math.ceil(euclid(a,b) / 3.0)))
        xs = np.linspace(a[0], b[0], steps)
        ys = np.linspace(a[1], b[1], steps)
        for i in range(steps):
            if not self.in_bounds((float(xs[i]), float(ys[i]))):
                return False
        for obs in self.obstacles:
            if obs.collides_seg(a,b):
                return False
        return True


# ---------------------------------------------------------------------
# RRT* (offline) with STL cost
# ---------------------------------------------------------------------

@dataclass
class Node:
    def __init__(self, x, y, parent=None, cost=0.0):
        self.x = x
        self.y = y
        self.parent = parent
        self.cost = cost
        # (J_base_accum, min_rho_base, max_rho_goal, t, Jd)
        self.metrics = (0.0, float('inf'), -float('inf'), 0.0, 0.0)
        self.children = [] # [PATCH] Added children list

    @property
    def t(self):
        """Convenient access to arrival time (metrics[3])."""
        return self.metrics[3]


class RRTStarSTL:
    def __init__(self,
                 world: World,
                 base_formula: STLFormula,
                 goal_predicate: Optional[STLFormula] = None,
                 goal_window_idx: Optional[Tuple[int,int]] = None,
                 step_size: float = 12.0,
                 neighbor_radius: float = 45.0,
                 n_iters: int = 15000,
                 goal_sample_rate: float = 0.30,
                 speed: float = 25.0,
                 dt_min: float = 0.05,
                 stl_weight: float = 0.6,
                 seed: Optional[int] = None):
        self.world = world
        self.base_formula = base_formula
        self.goal_predicate = goal_predicate
        self.goal_window_idx = goal_window_idx  # (a_idx, b_idx) or None
        self.step_size = step_size
        self.neighbor_radius = neighbor_radius
        self.n_iters = n_iters
        self.goal_sample_rate = goal_sample_rate
        self.speed = speed
        self.dt_min = dt_min
        self.stl_weight = stl_weight
        if seed is not None:
            random.seed(seed); np.random.seed(seed)
        self.nodes: List[Node] = []
        self.best_goal_cost = float('inf')
        self.goal_idx = None

    # [PATCH] Tree maintenance
    def _link_child(self, parent_idx: int, child_idx: int):
        if child_idx not in self.nodes[parent_idx].children:
            self.nodes[parent_idx].children.append(child_idx)

    def _unlink_child(self, parent_idx: int, child_idx: int):
        if child_idx in self.nodes[parent_idx].children:
            self.nodes[parent_idx].children.remove(child_idx)

    def _try_update_goal_candidate(self, idx: int, goal_center, goal_radius):
        node = self.nodes[idx]
        metrics = node.metrics
        dist_goal = math.hypot(node.x - goal_center[0], node.y - goal_center[1])
        
        if dist_goal > goal_radius:
            return

        # STL robustness check
        final_rho = metrics[1]
        if self.goal_predicate is not None:
            final_rho = min(metrics[1], metrics[2])
            
        if final_rho < -1e-6: # RHO_EPS
            return

        # Time window check
        if self.goal_window_idx is not None:
            cur_t = metrics[3]
            w_start = self.goal_window_idx[0] * self.dt_min
            w_end = self.goal_window_idx[1] * self.dt_min
            if not (w_start - 0.01 <= cur_t <= w_end + 0.01):
                return

        # Update best goal
        if node.cost < self.best_goal_cost:
            self.best_goal_cost = node.cost
            self.goal_idx = idx

    def _propagate_subtree(self, root_idx: int, goal_center, goal_radius):
        stack = [root_idx]
        while stack:
            u_idx = stack.pop()
            u_node = self.nodes[u_idx]
            for v_idx in u_node.children:
                v_node = self.nodes[v_idx]
                # Recompute incremental cost from updated parent u
                cost, rho, metrics = self._incremental_cost(u_idx, (v_node.x, v_node.y))
                self.nodes[v_idx].cost = cost
                self.nodes[v_idx].metrics = metrics
                # Check if this update makes it a better goal candidate
                self._try_update_goal_candidate(v_idx, goal_center, goal_radius)
                stack.append(v_idx)


    # ------------- helpers -------------
    def _sample(self, goal: Tuple[float,float]) -> Tuple[float,float]:
        if random.random() < self.goal_sample_rate:
            return goal
        return (random.uniform(self.world.xmin, self.world.xmax),
                random.uniform(self.world.ymin, self.world.ymax))

    def _nearest(self, p: Tuple[float,float]) -> int:
        # Vectorized Nearest
        # O(N) but in C (numpy) vs Python
        # Cache np array? No, simpler to create on fly (O(N) copy is acceptable vs Python loop)
        X = np.array(self.node_coords)
        d2 = np.sum((X - np.array(p))**2, axis=1)
        return np.argmin(d2)

    def _steer(self, a: Tuple[float,float], b: Tuple[float,float]) -> Tuple[float,float]:
        ax, ay = a; bx, by = b
        d = euclid(a,b)
        if d <= self.step_size:
            return (bx, by)
        t = self.step_size / (d + 1e-12)
        return (ax + t*(bx-ax), ay + t*(by-ay))

    def _neighbors(self, p: Tuple[float,float]) -> List[int]:
        r2 = self.neighbor_radius**2
        X = np.array(self.node_coords)
        d2 = np.sum((X - np.array(p))**2, axis=1)
        
        # Find all indices with d2 <= r2
        indices = np.where(d2 <= r2)[0]
        
        if len(indices) > 20:
            # Sort by distance and take top 20
            # Argpartition is faster than sort
            # But let's just pick indices
            valid_d2 = d2[indices]
            # argsort min to max
            sorted_order = np.argsort(valid_d2)
            indices = indices[sorted_order[:20]]
            
        return indices.tolist()

    def _discretize_edge(self, a: Tuple[float,float], b: Tuple[float,float]) -> np.ndarray:
        L = max(euclid(a,b), 1e-9)
        N = max(int(math.ceil(L / (self.speed * self.dt_min))), 1)
        xs = np.linspace(a[0], b[0], N+1)
        ys = np.linspace(a[1], b[1], N+1)
        return np.vstack([xs, ys]).T.astype(float)

    def _accumulate_traj(self, path_idx: List[int]) -> np.ndarray:
        pts = []
        for i in range(len(path_idx)-1):
            a = self.nodes[path_idx[i]]
            b = self.nodes[path_idx[i+1]]
            seg = self._discretize_edge((a.x,a.y),(b.x,b.y))
            if i>0: seg = seg[1:]
            pts.append(seg)
        return np.vstack(pts) if pts else np.zeros((1,2), dtype=float)

    def _indices_to_path(self, idx: int) -> List[int]:
        # cycle-safe parent climbing
        seen = set()
        path = [idx]; cur = idx
        while self.nodes[cur].parent is not None:
            if cur in seen:  # cycle guard
                break
            seen.add(cur)
            cur = self.nodes[cur].parent
            path.append(cur)
            if len(path) > len(self.nodes) + 2:
                break
        path.reverse()
        return path

    def _is_ancestor(self, anc_idx: int, desc_idx: int) -> bool:
        cur = desc_idx
        seen = set()
        while self.nodes[cur].parent is not None:
            if cur in seen:  # cycle guard
                return False
            seen.add(cur)
            cur = self.nodes[cur].parent
            if cur == anc_idx: return True
        return False

    # ------------- STL cost (paper-aligned) -------------
    def _stl_cost_and_series(self, traj: np.ndarray) -> Tuple[float, float]:
        """Return (J_phi, min_rho_total) using exponential cost integral."""
        # J_phi = integral(exp(-alpha * rho(t))) dt
        # We want to minimize this cost.
        alpha = 1.0  # Steepness parameter
        
        T = traj.shape[0]
        if T < 2:
            return 0.0, float('inf')
            
        # base formula robustness series
        rho_base = np.array([self.base_formula.robustness(traj, t) for t in range(T)], dtype=float)

        # goal masked series
        term_goal = 0.0
        max_rho_goal = -float('inf')

        if (self.goal_predicate is not None) and (self.goal_window_idx is not None):
            a_idx, b_idx = self.goal_window_idx
            
            lo = max(0, min(T-1, int(a_idx)))
            hi = max(0, min(T-1, int(b_idx)))
            
            if hi >= lo:
                 # Extract valid window
                 g_vals = [self.goal_predicate.robustness(traj, t) for t in range(lo, hi+1)]
                 if g_vals:
                     max_rho_goal = max(g_vals)
            
            # Goal term: maximize max_rho_goal => minimize exp(-alpha * max_rho_goal)
            # Clamp exponent to avoid overflow
            exponent = -alpha * max_rho_goal
            exponent = max(min(exponent, 500.0), -500.0)
            term_goal = math.exp(exponent)
        
        # Base term: maximize rho_base(t) over all t => minimize sum(exp(-alpha * rho_base(t)))
        exponents = -alpha * rho_base
        np.clip(exponents, -500.0, 500.0, out=exponents)
        term_base = np.sum(np.exp(exponents)) * self.dt_min
        
        # Total Cost
        # Weighting goal higher to encourage entry
        Jphi = term_base + 10.0 * term_goal 
        
        # Calculate actual min robustness for success check
        # Satisfied if: (rho_base[0] > 0) [Formula holds from start] AND (max_rho_goal > 0 if goal exists)
        # Note: using np.min(rho_base) would require 'Eventually' terms to hold for ALL time, which is wrong.
        min_rho_base = rho_base[0]
        if self.goal_predicate is not None:
             rho = min(min_rho_base, max_rho_goal)
        else:
             rho = min_rho_base
        
        return Jphi, rho

    def _incremental_cost(self, parent_idx: Optional[int], new_xy: Tuple[float,float]) -> Tuple[float, float, Tuple[float,float,float,float]]:
        """
        Compute cost and metrics for a new node at new_xy extending from parent_idx.
        Returns: (TotalCost, Robustness, MetricsTuple)
        """
        alpha = 1.0
        
        if parent_idx is None:
            # Root node
            return 0.0, 0.0, (0.0, float('inf'), -float('inf'), 0.0, 0.0)

        p_node = self.nodes[parent_idx]
        p_metrics = p_node.metrics
        p_J_base, p_min_rho, p_max_goal, p_t = p_metrics[:4]
        
        # Discretize edge
        edge_pts = self._discretize_edge((p_node.x, p_node.y), new_xy)
        if len(edge_pts) < 2:
            return float('inf'), -float('inf'), p_metrics # Should not happen

        # Calculate edge metrics
        # 1. Base Robustness (Always Safe constraint)
        # We assume base_formula is roughly "Always(Safe)".
        # For general formula, we'd need full incremental monitoring.
        # Here we approximate 'base_formula' robustness as 'min distance to obstacles'.
        # This matches the "min_rho_base" logic we had.
        
        # Eval base rho for edge points
        # Optimization: We know base_formula is (Ev(L)|Ev(R)) & Alw(Safe).
        # Actually, let's just evaluate base_formula.robustness(pt) pointwise?
        # STL robustness is not strictly pointwise for temporal operators.
        # BUT for Alw(Safe), it is pointwise. For Ev(L), it is max over time.
        # So we can evaluate the predicates pointwise.
        
        # Re-parsing base formula components for optimization:
        # We know the structure from make_base_formula.
        # Predicates: lane_left, lane_right, forbidden.
        # rho_safe(t) = -robustness(forbidden)
        # rho_l(t) = robustness(lane_left)
        # rho_r(t) = robustness(lane_right)
        
        # But to be safe and generic-ish, let's use the formula object on the edge?
        # No, formula.robustness expects a trajectory.
        # We will treat the edge as a trajectory for pointwise evaluation.
        
        T_edge = edge_pts.shape[0]
        # Times for edge points
        t_start = p_t
        t_edge = np.linspace(p_t, p_t + (T_edge-1)*self.dt_min, T_edge)
        
        # Evaluate base formula on edge? 
        # Base formula has temporal operators "Eventually".
        # If we just evaluate it on the edge, it won't know about history.
        # However, "Eventually" is a max-over-time property.
        # rho_base_formula = min( max(rho_L_trace), min(rho_Safe_trace) ) roughly.
        # Actually it's (max(L) or max(R)) and min(Safe).
        # rho = min( max(max(L), max(R)), min(Safe) )
        
        # So we track: max_L, max_R, min_Safe.
        # Let's update metrics to include these.
        # But to keep it simple and compatible with existing 'metrics' tuple structure:
        # We simply re-evaluate the full STL on the *accumulated* path?
        # NO, that's what was slow.
        
        # FAST PATH:
        # We implement specific logic for the known formula structure.
        # This is acceptable for "optimizing the specific experiment".
        
        # Metrics: (min_safe, max_l, max_r, current_t)
        # Note: I'm redefining meanings of metrics indices 0,1,2 temporarily here.
        # Wait, the node definition above used (J_base_accum, min_rho_base...)
        # Let's stick to the structure that allows computing the cost.
        
        # Cost J = term_base + 10 * term_goal
        # term_base = sum(exp(-alpha * rho_base(t)))
        # rho_base(t) = min(EvL(t), EvR(t), Safe(t)) ... ?
        # Actually `base_formula` is evaluated at time t.
        # Ev(L) at time t depends on future. 
        # RRT* builds path from start. Future is unknown?
        # Standard STL semantics: robustness at time 0 depends on whole path.
        # So as we extend path, rho(0) changes!
        # This means J changes for the WHOLE path.
        # So J is indeed global and cannot be purely incremental if we define cost as J(0).
        # BUT, if we define cost as "Accumulated Risk", we can approximate.
        
        # CORRECT APPROACH for Speed:
        # Only evaluate STL cost periodically or efficiently.
        # OR, since rho_base(t) for Alw(Safe) is just Safe(t) (since ALW means for all t' >= t),
        # AND Ev(Goal) means exists t >= 0.
        
        # Let's simplify the COST function to be friendly to RRT.
        # J = Integral( Risk(t) )dt + Reward(Goal)
        # Risk(t) = exp(-alpha * dist_to_obs(t))
        # Reward = exp(-alpha * max_dist_to_goal_region)
        
        # This is effectively what the STL roughly says.
        # So, we track J_risk_accum.
        
        J_risk_edge = 0.0
        rho_safe_min_edge = float('inf')
        
        # We need access to the raw predicates or helper logic
        # define local helpers
        if len(self.world.obstacles) > 0:
            obs_centers = np.array([[o.x, o.y] for o in self.world.obstacles])
            obs_radii = np.array([o.r for o in self.world.obstacles])
            # (N_pts, 1, 2) - (1, N_obs, 2) -> (N_pts, N_obs, 2)
            dists = np.sqrt(np.sum((edge_pts[:, None, :] - obs_centers[None, :, :])**2, axis=2))
            # robustness = dist - radius
            rhos_obs = dists - obs_radii[None, :]
            # min over obstacles = distance to closest obstacle - radius
            # safe if > 0
            rhos = np.min(rhos_obs, axis=1) 
        else:
             rhos = np.full(len(edge_pts), 999.0) # Always safe if no obstacles
        
        # Now update accumulated metrics
        # J_base_accum += sum(exp(-alpha * rhos)) * dt
        exponents = -alpha * rhos
        np.clip(exponents, -500.0, 500.0, out=exponents)
        edge_J_base = np.sum(np.exp(exponents)) * self.dt_min
        
        new_J_base = p_J_base + edge_J_base
        new_min_rho = min(p_min_rho, np.min(rhos))

        step_dist = euclid((p_node.x, p_node.y), new_xy)
        new_t = p_t + (step_dist / self.speed)
        
        # Goal terms
        # Goal window check
        # window is global time.
        # We check if (t_start ... t_end) overlaps with window.
        new_max_goal = p_max_goal
        if (self.goal_predicate is not None) and (self.goal_window_idx is not None):
             # Window in seconds
             w_start = self.goal_window_idx[0] * self.dt_min
             w_end = self.goal_window_idx[1] * self.dt_min
             
             # Check overlap of [p_t, new_t] with [w_start, w_end]
             # Actually we check points.
             
             for i, t in enumerate(t_edge):
                 if w_start <= t <= w_end:
                     # Evaluate goal pred
                     traj_1 = np.array([edge_pts[i]])
                     g_val = self.goal_predicate.robustness(traj_1, 0)
                     if g_val > new_max_goal:
                         new_max_goal = g_val


        # Final Cost J
        term_goal = 0.0
        if new_max_goal > -float('inf'):
             exp_g = -alpha * new_max_goal
             exp_g = max(min(exp_g, 500.0), -500.0)
             term_goal = math.exp(exp_g)
        
        # Total J
        Jphi = new_J_base + 10.0 * term_goal
        
        # Final Robustness (Success criteria)
        # min( min_rho_base, max_goal )
        if self.goal_predicate is not None:
             final_rho = min(new_min_rho, new_max_goal)
        else:
             final_rho = new_min_rho

        # Distance cost (standard RRT*)
        # dist_cost = p_node.cost - p_node.stl_weighted_cost  # Extract old Jd? No, node.cost is total.
        # Actually we need Jd to be separate or we tracking it.
        # Let's assume node.cost is Jd + weight * Jphi
        # But we need Jd.
        # Add Jd to metrics? Or just recalc euclid.
        step_dist = euclid((p_node.x,p_node.y), new_xy)
        
        # We need accumulated Jd? 
        # Parent cost = p_Jd + w * p_Jphi
        # We don't store p_Jd.
        # Let's add Jd to metrics or Node.
        # Hack: recover Jd from parent cost? 
        # p_Jd = p_node.cost - self.stl_weight * (p_J_base + 10*p_term_goal_parent)
        # This is getting messy.
        
        # Simplification: Store Jd in metrics
        p_Jd = p_metrics[4] if len(p_metrics) > 4 else 0.0
        step_dist = euclid((p_node.x,p_node.y), new_xy)
        new_Jd = p_Jd + step_dist
        
        total_cost = new_Jd + self.stl_weight * Jphi
        
        # Returns: cost, rho_min, MetricsTuple
        # Metrics: (J_base, min_rho, max_goal, t, Jd)
        return total_cost, final_rho, (new_J_base, new_min_rho, new_max_goal, new_t, new_Jd)

    def _path_cost(self, path_idx: List[int]) -> float:
        # Fallback for full path calc if needed (e.g. final check)
        Jd = 0.0
        for i in range(len(path_idx)-1):
            a = self.nodes[path_idx[i]]
            b = self.nodes[path_idx[i+1]]
            Jd += euclid((a.x,a.y),(b.x,b.y))
        traj = self._accumulate_traj(path_idx)
        Jphi, _ = self._stl_cost_and_series(traj)
        return Jd + self.stl_weight * Jphi

    # ------------- Planning -------------
    def plan(self,
             start: Tuple[float,float],
             goal: Tuple[float,float],
             goal_radius: float = 5.0,
             time_constraint=None,
             max_time_s: float = 300.0) -> Tuple[bool, List[int], List['Node'], Optional[float], Optional[int], List[Tuple[float,float]]]:
        t0 = time.time()
        self.nodes = [Node(start[0], start[1], parent=None, cost=0.0)]
        self.node_coords = [[start[0], start[1]]]
        self.best_goal_cost = float('inf')
        self.goal_idx = None
        
        cost_history: List[Tuple[float,float]] = []
        first_sol_time = None
        first_goal_idx = None
        RHO_EPS = 1e-6

        # [PATCH] attempt iteration based budget
        for it in range(self.n_iters):
            if (time.time() - t0) >= max_time_s:
                break

            rnd = self._sample(goal)
            ni = self._nearest(rnd)
            newp = self._steer((self.nodes[ni].x, self.nodes[ni].y), rnd)
            
            if not self.world.collision_free((self.nodes[ni].x, self.nodes[ni].y), newp):
                continue

            neigh = self._neighbors(newp) or [ni]

            best_parent = None
            best_cost = float('inf')
            best_metrics = None
            
            for pj in neigh:
                if not self.world.collision_free((self.nodes[pj].x, self.nodes[pj].y), newp):
                    continue
                cost, rho, metrics = self._incremental_cost(pj, newp)
                if metrics[1] < -RHO_EPS:
                    continue
                if cost < best_cost:
                    best_cost = cost
                    best_parent = pj
                    best_metrics = metrics

            if best_parent is None or not math.isfinite(best_cost):
                continue

            # Add new node
            new_node = Node(newp[0], newp[1], parent=best_parent, cost=best_cost)
            new_node.metrics = best_metrics
            self.nodes.append(new_node)
            self.node_coords.append([newp[0], newp[1]])
            new_idx = len(self.nodes) - 1
            
            # [PATCH] link child
            self._link_child(best_parent, new_idx)
            
            # [PATCH] Check goal for new node
            d_goal = math.hypot(newp[0] - goal[0], newp[1] - goal[1])
            if d_goal <= goal_radius and first_sol_time is None:
                first_sol_time = time.time() - t0
                first_goal_idx = new_idx
                logging.info(f"First goal-region entry at node {new_idx}")

            self._try_update_goal_candidate(new_idx, goal, goal_radius)

            # Rewiring
            for pj in neigh:
                if pj == 0 or pj == best_parent:
                    continue
                if self._is_ancestor(anc_idx=pj, desc_idx=new_idx):
                    continue
                if not self.world.collision_free((self.nodes[new_idx].x, self.nodes[new_idx].y),
                                                 (self.nodes[pj].x, self.nodes[pj].y)):
                    continue
                
                cost_via_new, rho_via_new, metrics_via_new = self._incremental_cost(new_idx, (self.nodes[pj].x, self.nodes[pj].y))
                if metrics_via_new[1] < -RHO_EPS:
                    continue
                
                if cost_via_new + 1e-7 < self.nodes[pj].cost:
                    # [PATCH] Unlink old, update, link new, propagate
                    old_parent = self.nodes[pj].parent
                    if old_parent is not None:
                        self._unlink_child(old_parent, pj)
                    
                    self.nodes[pj].parent = new_idx
                    self.nodes[pj].cost = cost_via_new
                    self.nodes[pj].metrics = metrics_via_new
                    self._link_child(new_idx, pj)
                    
                    # [PATCH] Propagate subtree and update goal candidates
                    self._propagate_subtree(pj, goal, goal_radius)
                    # Check if pj itself is now a better goal candidate
                    self._try_update_goal_candidate(pj, goal, goal_radius)
            
            if self.goal_idx is not None:
                cost_history.append((time.time() - t0, self.best_goal_cost))

        if self.goal_idx is not None:
            idx_path = self._indices_to_path(self.goal_idx)
            return True, idx_path, self.nodes, first_sol_time, first_goal_idx, cost_history

        logging.warning(f"Goal not reached within {self.n_iters} iterations")
        return False, [], self.nodes, first_sol_time, first_goal_idx, cost_history

# ---------------------------------------------------------------------
# STL helpers
# ---------------------------------------------------------------------

def make_base_formula(world: World) -> STLFormula:
    """
    Create STL formula matching TB-RRT task:
    φ = □ (distance_to_obstacles > 0)
    We approximate this by And(Not(Obs_i)).
    """
    if not world.obstacles:
        return STLAlways(STLPredicate(world.xmin-100, world.xmax+100, world.ymin-100, world.ymax+100))
    
    preds = []
    for obs in world.obstacles:
        preds.append(STLNot(STLCirclePredicate(obs.x, obs.y, obs.r)))
        
    combined = preds[0]
    for p in preds[1:]:
        combined = STLAnd(combined, p)
        
    return STLAlways(combined)

def adjust_point_out_of_obstacles(world: World, p: Tuple[float,float], min_clearance: float) -> Tuple[float,float]:
    """Push a point outward if it's inside or too close to any obstacle; keep within bounds."""
    x, y = float(p[0]), float(p[1])
    changed = True
    it = 0
    while changed and it < 20:
        it += 1
        changed = False
        for obs in world.obstacles:
            d = euclid((x,y),(obs.x,obs.y))
            need = obs.r + min_clearance
            if d < max(1e-6, need):
                ang = math.atan2(y-obs.y, x-obs.x) if d > 1e-6 else random.uniform(0, 2*math.pi)
                x = obs.x + need * math.cos(ang)
                y = obs.y + need * math.sin(ang)
                changed = True
        x = clamp(x, world.xmin+1e-3, world.xmax-1e-3)
        y = clamp(y, world.ymin+1e-3, world.ymax-1e-3)
    return (x, y)

# ---------------------------------------------------------------------
# Experiment harness
# ---------------------------------------------------------------------

def plot_results(tree, path_idx, start_xy, goal_center, goal_radius,
                 spatial_err, time_err, total_err, elapsed_time, trial_idx, save_dir,
                 world_bounds=None, obstacles=None):
    """Plot the 2D view only (ST-RRT compatible filename)."""
    if not HAVE_MPL:
        return

    w = world_bounds if world_bounds is not None else WORLD
    obs = obstacles if obstacles is not None else OBSTACLES

    # --- 2D plot ---
    fig, ax = plt.subplots(figsize=(6.6, 5.0), dpi=120)
    title = f"Trial {trial_idx} - STL-RRT*"
    if not path_idx:
        title += " (FAILED)"
    ax.set_title(title)
    ax.set_xlim(w[0], w[1])
    ax.set_ylim(w[2], w[3])
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linewidth=0.4, alpha=0.4)

    for node in tree:
        if node.parent is not None:
            p = tree[node.parent]
            ax.plot([node.x, p.x], [node.y, p.y], linewidth=0.4, alpha=0.20, color='#888888', zorder=1)

    for i_obs, (ox, oy, r) in enumerate(obs):
        circ = plt.Circle((ox, oy), r, fill=False, linewidth=1.2, edgecolor='k')
        ax.add_patch(circ)
        ax.text(ox, oy, f"O{i_obs:02d}", fontsize=7, ha="center", va="center", alpha=0.7)

    if path_idx and len(path_idx) >= 2:
        path_nodes = [tree[i] for i in path_idx]
        px = [n.x for n in path_nodes]
        py = [n.y for n in path_nodes]
        path_color = '#0072B2' if spatial_err is not None else '#FF6B6B'
        ax.plot(px, py, linewidth=2.0, label="Path", color=path_color, zorder=4)

        # Time colormap via interpolation
        pts_2d = []
        t_vals = []
        for ii in range(len(path_nodes) - 1):
            seg = np.linspace(
                np.array([path_nodes[ii].x, path_nodes[ii].y, path_nodes[ii].t]),
                np.array([path_nodes[ii+1].x, path_nodes[ii+1].y, path_nodes[ii+1].t]),
                num=10)
            if ii > 0:
                seg = seg[1:]
            for j in range(seg.shape[0]):
                pts_2d.append((seg[j, 0], seg[j, 1]))
                t_vals.append(seg[j, 2])
        if pts_2d:
            pts_2d = np.array(pts_2d, dtype=float)
            t_vals = np.array(t_vals, dtype=float)
            sc = ax.scatter(pts_2d[:, 0], pts_2d[:, 1], c=t_vals, s=10, cmap="viridis", alpha=0.7, zorder=5)
            cbar = plt.colorbar(sc, ax=ax, shrink=0.75)
            cbar.set_label("time (s)")

    ax.scatter(start_xy[0], start_xy[1], marker='o', s=90, color='#0099FF', edgecolor='k', label='Start', zorder=10)
    ax.scatter(goal_center[0], goal_center[1], marker='*', s=180, color='#FF9900', edgecolor='k', label='Goal Center', zorder=11)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, f"trial_{trial_idx:03d}_2d.png"))
    plt.close(fig)



def path_length_idx(nodes: List[Node], path_idx: List[int]) -> float:
    length = 0.0
    for i in range(len(path_idx) - 1):
        n1 = nodes[path_idx[i]]
        n2 = nodes[path_idx[i+1]]
        length += math.hypot(n2.x - n1.x, n2.y - n1.y)
    return length

def calculate_smoothness_idx(nodes: List[Node], path_idx: List[int]) -> float:
    if len(path_idx) < 3:
        return 0.0
    total_angle = 0.0
    for i in range(1, len(path_idx) - 1):
        n_prev = nodes[path_idx[i-1]]
        n_curr = nodes[path_idx[i]]
        n_next = nodes[path_idx[i+1]]
        
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

def run_case(case: dict, out_root: str = None, save_plots: bool = True):
    """
    Run a single experiment case for STL-RRT*.
    """
    global WORLD, OBSTACLES, GOAL_CENTER, GOAL_RADIUS, START, SEED, TRIALS

    eff = normalize_case(case) if case else None
    if eff is None:
        # Fallback
        eff = {
            "case_id": "default",
            "world": {"xlim": [WORLD[0], WORLD[1]], "ylim": [WORLD[2], WORLD[3]]},
            "start": {"x": START[0], "y": START[1]},
            "goal": {"x": GOAL_CENTER[0], "y": GOAL_CENTER[1]},
            "obstacles": [],
            "time": {"t_ref": "", "t_window": [7.0,8.0]},
            "dynamics": {"v_min": 0.0, "v_max": SPEED, "a_max": 2.0, "v0": 0.0},
            "random": {"seed_base": SEED, "num_trials": TRIALS},
            "planner": {"step_size": STEP_SIZE, "neighbor_radius": NEIGHBOR_RADIUS, "n_iters": N_ITERS},
            "algorithms": {"stl_rrt": {"eps_state": EPS_STATE, "dt_min": DT_MIN, "stl_weight": STL_WEIGHT}},
            "outputs": {"out_dir": "output"}
        }
    
    c_id = eff["case_id"]

    # 2. Extract Parameters (Aligned with TB-RRT/ST-RRT)
    w = eff["world"]
    s = eff["start"]
    g = eff["goal"]
    obs_list = eff["obstacles"]
    tm = eff["time"]
    dyn = eff["dynamics"]
    rnd = eff["random"]
    pln = eff["planner"]
    algo_params = eff["algorithms"].get("stl_rrt", {})
    
    l_world = (w["xlim"][0], w["xlim"][1], w["ylim"][0], w["ylim"][1])
    l_start = (s["x"], s["y"], s.get("t", 0.0))
    l_goal_center = (g["x"], g["y"])
    l_goal_radius = algo_params.get("eps_state", 5.0) 

    l_seed = rnd.get("seed_base", SEED)
    l_trials = rnd.get("num_trials", TRIALS)
    
    # Dynamics (Option B)
    l_v_min, l_v_max, l_a_max, l_v0 = dyn["v_min"], dyn["v_max"], dyn["a_max"], dyn["v0"]
    l_t_ref = tm.get("t_ref", "")
    l_t_window = tuple(tm["t_window"])
    
    # Planner
    l_n_iters = pln.get("n_iters", 6000)
    l_step_size = pln.get("step_size", 5.0)
    l_neighbor_radius = pln.get("neighbor_radius", 45.0)
    l_goal_sample_rate = pln.get("goal_sample_rate", 0.15)
    
    # STL-RRT specific
    l_dt_min = algo_params.get("dt_min", 0.1)
    l_stl_weight = algo_params.get("stl_weight", 0.6)
    l_eps_state = algo_params.get("eps_state", 5.0)
    # Important: STL-RRT uses 'speed' for time scaling internally. 
    # Let's use v_max if available, else standard SPEED.
    l_speed = l_v_max
    
    # Time Constraint parsing for unified metrics
    t_ref_val = l_t_ref
    t_min_val, t_max_val = l_t_window
    tc_raw = {"type": "window", "t_lo": t_min_val, "t_hi": t_max_val, "target_time": t_ref_val}

    # [PATCH-FIX] Ensure target_time is always float to avoid TypeError in compute_time_err (BUG-3)
    if t_ref_val is None or (isinstance(t_ref_val, str) and t_ref_val.strip() == ""):
        tc_raw["target_time"] = 0.5 * (t_min_val + t_max_val)
    else:
        tc_raw["target_time"] = float(t_ref_val)

    # Set globals
    WORLD = l_world
    OBSTACLES = [(o["cx"], o["cy"], o["r"]) for o in obs_list]
    l_obstacles = OBSTACLES # for loop
    GOAL_CENTER = l_goal_center
    GOAL_RADIUS = l_goal_radius


    logging.info("=" * 70)
    logging.info("STL-RRT* Experiments")
    logging.info("=" * 70)
    logging.info(f"  step_size={l_step_size}, neighbor_radius={l_neighbor_radius}, n_iters={l_n_iters}")
    logging.info(f"  goal_sample_rate={l_goal_sample_rate}, speed={l_speed}, dt_min={l_dt_min}, stl_weight={l_stl_weight}")
    logging.info(f"  WORLD={l_world}, START={l_start}")
    logging.info(f"  GOAL_CENTER={l_goal_center}, GOAL_RADIUS={l_goal_radius}")
    logging.info(f"  Obstacles: {l_obstacles}")
    logging.info(f"  Time constraint: {tc_raw}")
    logging.info(f"  Trials: {l_trials}, Seed: {l_seed}")
    logging.info("=" * 70)

    # 3. Output Directory
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if out_root is None:
        out_root = eff.get("outputs", {}).get("out_dir", "output")
    
    if not os.path.isabs(out_root):
        out_root = os.path.join(script_dir, out_root)

    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_root, c_id, "stl_rrt", f"run_{run_timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Save config_used.json
    config_used = eff
    with open(os.path.join(run_dir, "config_used.json"), 'w', encoding='utf-8') as cf:
        json.dump(config_used, cf, indent=2, ensure_ascii=False)

    # Build World object
    world = World(l_world[0], l_world[1], l_world[2], l_world[3])
    for (ox, oy, r) in l_obstacles:
        world.obstacles.append(CircleObstacle(float(ox), float(oy), float(r)))

    # STL formulas
    base_formula = make_base_formula(world)
    gx, gy = l_goal_center
    goal_pred = STLCirclePredicate(gx, gy, l_goal_radius)

    # Convert time window to index window for STL
    window_idx = None
    if tc_raw is not None and tc_raw["type"] == "window":
        t_lo, t_hi = tc_raw["t_lo"], tc_raw["t_hi"]
        a_idx = int(round(t_lo / l_dt_min))
        b_idx = int(round(t_hi / l_dt_min))
        if b_idx < a_idx:
            a_idx, b_idx = b_idx, a_idx
        window_idx = (a_idx, b_idx)

    # CSV files
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

            start_xy = (float(l_start[0]), float(l_start[1]))
            goal_xy = (float(gx), float(gy))

            # Adjust start/goal out of obstacles if needed (legacy helper)
            start_xy = adjust_point_out_of_obstacles(world, start_xy, min_clearance=2.0)
            goal_xy = adjust_point_out_of_obstacles(world, goal_xy, min_clearance=max(3.0, l_goal_radius))

            # Create planner
            planner = RRTStarSTL(
                world=world,
                base_formula=base_formula,
                goal_predicate=goal_pred if window_idx is not None else None,
                goal_window_idx=window_idx,
                step_size=l_step_size,
                neighbor_radius=l_neighbor_radius,
                n_iters=l_n_iters,
                goal_sample_rate=l_goal_sample_rate,
                speed=l_speed,
                dt_min=l_dt_min,
                stl_weight=l_stl_weight,
                seed=trial_seed,
            )

            start_time = time.time()
            success, path_idx, tree, first_sol_time, first_goal_idx, cost_history = planner.plan(
                start_xy, goal_xy,
                goal_radius=l_goal_radius,
                time_constraint=tc_raw,
                max_time_s=300.0,
            )
            elapsed_time = time.time() - start_time

            # Save cost history
            with open(history_file, mode='a', newline='') as hf:
                h_writer = csv.writer(hf)
                for (ht, hc) in cost_history:
                    h_writer.writerow([trial, f"{ht:.4f}", f"{hc:.4f}"])

            # ---------- Compute metrics (unified, ST-RRT compatible) ----------
            res_path_len = None
            res_arrival_t = None
            res_spatial_err = None
            res_time_err = None
            res_total_err = None
            res_success = 0
            res_smooth = None
            res_twv = None
            res_tmin = None
            res_tmax = None
            res_smooth = ""

            # Using Option B logic (Feasibility Check with Dynamics)
            # Need dynamics params? "dynamics": {"v_min", "v_max", "a_max", "v0"}
            dyn_cfg = eff.get("dynamics", {})
            d_v_min = dyn_cfg.get("v_min", 0.0)
            d_v_max = dyn_cfg.get("v_max", l_speed) # Use l_speed as v_max default
            d_a_max = dyn_cfg.get("a_max", 2.0)
            d_v0 = dyn_cfg.get("v0", 0.0)
            
            # Time Window (if available)
            l_t_window = (t_min_val, t_max_val) if t_min_val is not None else (0.0, float('inf'))
            
            # Additional metric calibration (Change 2)
            tc_metric = tc_raw
            if tc_raw is not None and t_ref_val != "":
                tc_metric = dict(tc_raw)
                tc_metric["target_time"] = float(t_ref_val)

            f_sol_time = ""
            f_path_len = ""
            f_total_err = ""

            if path_idx:  # Planner found a path
                # 1. Path Length
                path_L = path_length_idx(tree, path_idx)
                res_path_len = path_L
                res_smooth = calculate_smoothness_idx(tree, path_idx)
                
                # 2. Dynamics-based Feasible Range (Diagnostic only)
                res_tmin, res_tmax = Dynamics.feasible_time_range(path_L, d_v0, d_v_min, d_v_max, d_a_max)
                
                # 3. Arrival Time (Option A: Actual planner output)
                last_node_idx = path_idx[-1]
                res_arrival_t = float(tree[last_node_idx].t)
                
                # 4. Time Window Violation (Option A: based on actual arrival time)
                res_twv = compute_time_window_violation(res_arrival_t, tc_raw)
                
                # 5. Spatial Error (Approach Distance logic)
                last_node_idx = path_idx[-1]
                last_node = tree[last_node_idx]
                dist_last = math.hypot(last_node.x - gx, last_node.y - gy)
                
                if dist_last < 1e-3:
                     approach_node_idx = None
                     for k in range(len(path_idx)-2, -1, -1):
                         ni = path_idx[k]
                         n = tree[ni]
                         d_n = math.hypot(n.x - gx, n.y - gy)
                         if d_n > 1e-3:
                             approach_node_idx = ni
                             break
                     if approach_node_idx is not None:
                         an = tree[approach_node_idx]
                         res_spatial_err = math.hypot(an.x - gx, an.y - gy)
                     else:
                         res_spatial_err = 0.0
                else:
                    res_spatial_err = dist_last
                
                # 6. Success Definition (Aligned Unified - Option A)
                is_spatial_ok = (res_spatial_err <= l_eps_state)
                is_time_ok = (res_twv == 0.0)
                res_success = 1 if (is_spatial_ok and is_time_ok) else 0

                # 7. Error Metrics (Change 2)
                res_time_err = compute_time_err(res_arrival_t, tc_metric)
                res_total_err = compute_total_err(res_spatial_err, res_time_err)

                # First solution metrics
                if first_sol_time is not None and first_goal_idx is not None:
                    f_sol_time = f"{first_sol_time:.4f}"
                    first_path = reconstruct_path_idx(tree, first_goal_idx)
                    fp_len = path_length_idx(tree, first_path)
                    f_path_len = f"{fp_len:.3f}"
                    # Recalculate first sol error metrics? 
                    # Assuming first sol was valid by planner's definition (which is weak)
                    # Let's just report what we have.
                    # Actually standard says first solution metrics should be consistent.
                    # But recalc is complex.
                    # Using planner's value:
                    first_arr_t = tree[first_goal_idx].t
                    
                    # Compute approach spatial err for first sol
                    first_se = 0.0 # Default
                    # Logic: first_goal_idx is the node in goal region.
                    # Its distance is d_goal <= goal_radius.
                    # We want approach distance if d_goal is small? 
                    # TB-RRT logic: if d < 1e-3 use approach. 
                    # Here first_goal_idx might be strictly inside goal region but not at center.
                    # Let's just use raw distance for first sol spatial err.
                    n_first = tree[first_goal_idx]
                    d_first = math.hypot(n_first.x - gx, n_first.y - gy)
                    first_se = d_first
                    
                    # For total error, consistent with res_total_err
                    # Time error
                    first_te = 0.0
                    if tc_raw:
                        # Re-eval time error for first sol
                         # Feasibility check for first sol?
                         # It was found by planner, so it satisfied constraints?
                         # The planner checks dynamics roughly? No, planner uses step_size.
                         # We should use the node's time.
                         t_first = n_first.t
                         if t_ref_val:
                             first_te = abs(t_first - float(t_ref_val))
                         elif tc_raw["type"] == "window":
                             # if in window, err=0, else diff
                             if t_first < tc_raw["t_lo"]: first_te = tc_raw["t_lo"] - t_first
                             elif t_first > tc_raw["t_hi"]: first_te = t_first - tc_raw["t_hi"]
                             else: first_te = 0.0
                    
                    f_tot = math.hypot(first_se, first_te)
                    f_total_err = f"{f_tot:.3f}"
                elif res_path_len is not None:
                    # Fallback: use final result as first-and-only solution
                    f_sol_time = f"{elapsed_time:.4f}"
                    f_path_len = f"{res_path_len:.3f}"
                    f_total_err = f"{res_total_err:.3f}" if res_total_err is not None else ""

                # [PATCH-FIX] Removed redundant and incorrect bc_se/bc_twv references (BUG-1)
                
                row = {
                    'case_id': c_id,
                    'trial': trial,
                    'algo': 'STL-RRT*(r6)',
                    'success': res_success,
                    'runtime_s': f"{elapsed_time:.4f}",
                    'path_len': f"{res_path_len:.3f}",
                    'arrival_t': f"{res_arrival_t:.3f}",
                    'spatial_err': f"{res_spatial_err:.3f}",
                    'time_err': f"{res_time_err:.3f}",
                    'total_err': f"{res_total_err:.3f}",
                    'time_window_violation': f"{res_twv:.3f}",
                    'nodes': len(tree),
                    'seed': trial_seed,
                    'first_sol_time': f_sol_time,
                    'first_path_len': f_path_len,
                    'first_total_err': f_total_err,
                    'smoothness': f"{res_smooth:.3f}",
                    't_ref': f"{t_ref_val}" if t_ref_val != "" else "",
                    'L_proxy': "",
                    'Tmin': f"{res_tmin:.3f}" if res_tmin is not None else "",
                    'Tmax': f"{res_tmax:.3f}" if res_tmax is not None else "",
                    'alpha': "", 'beta': "", 'kappa': "",
                    'p_tw': "", 'loiter_used': 0, 'loiter_len': "0.000",
                    'late_pruned': 0, 'lb_pruned': 0,
                }
                writer.writerow(row)

                if save_plots:
                    plot_results(tree, path_idx, start_xy, l_goal_center, l_goal_radius,
                                 res_spatial_err, res_time_err, res_total_err, elapsed_time, trial, run_dir,
                                 world_bounds=l_world, obstacles=l_obstacles)
                status_str = "SUCCESS" if res_success else "FAIL(path but time miss)"
                logging.info(f"Trial {trial}: {status_str} (t={res_arrival_t:.3f}, se={res_spatial_err:.3f}, te={res_time_err:.3f}, smooth={res_smooth:.2f})")
            else:
                # Failure case – find best candidate node
                bc_idx, bc_se, bc_te, bc_tot = find_best_candidate_node_idx(
                    tree, l_goal_center, l_goal_radius, tc_metric)

                bc_arrival_t = tree[bc_idx].t if bc_idx is not None else 0.0
                bc_path = reconstruct_path_idx(tree, bc_idx) if bc_idx is not None else []
                bc_path_len = path_length_idx(tree, bc_path) if bc_path else 0.0
                bc_smooth = calculate_smoothness_idx(tree, bc_path) if bc_path else 0.0
                bc_twv = compute_time_window_violation(bc_arrival_t, tc_raw)

                # [PATCH-FIX] Explicitly define bc_success for fallback case (BUG-2)
                bc_success = 1 if (bc_se <= l_eps_state and bc_twv == 0.0) else 0

                row = {
                    'case_id': c_id,
                    'trial': trial,
                    'algo': 'STL-RRT*(r6)',
                    'success': bc_success,
                    'runtime_s': f"{elapsed_time:.4f}",
                    'path_len': f"{bc_path_len:.3f}",
                    'arrival_t': f"{bc_arrival_t:.3f}",
                    'spatial_err': f"{bc_se:.3f}" if bc_se is not None else "",
                    'time_err': f"{bc_te:.3f}" if bc_te is not None else "",
                    'total_err': f"{bc_tot:.3f}" if bc_tot is not None else "",
                    'time_window_violation': f"{bc_twv:.3f}",
                    'nodes': len(tree),
                    'seed': trial_seed,
                    'first_sol_time': "",
                    'first_path_len': "",
                    'first_total_err': "",
                    'smoothness': f"{bc_smooth:.3f}",
                    't_ref': f"{t_ref_val}" if t_ref_val != "" else "",
                    'L_proxy': "",
                    'Tmin': f"{tc_raw['t_lo']:.3f}" if (tc_raw and tc_raw.get('t_lo') is not None) else "",
                    'Tmax': f"{tc_raw['t_hi']:.3f}" if (tc_raw and tc_raw.get('t_hi') is not None) else "",
                    'alpha': "", 'beta': "", 'kappa': "",
                    'p_tw': "", 'loiter_used': 0, 'loiter_len': "0.000",
                    'late_pruned': 0, 'lb_pruned': 0,
                }
                writer.writerow(row)

                if save_plots:
                    plot_results(tree, bc_path if bc_path else None, start_xy,
                                 l_goal_center, l_goal_radius,
                                 bc_se, bc_te, bc_tot, elapsed_time, trial, run_dir,
                                 world_bounds=l_world, obstacles=l_obstacles)
                logging.warning(f"Trial {trial}: FAILED (time={elapsed_time:.2f}s, nodes={len(tree)}, best_total_err={bc_tot:.3f})")

    logging.info("=" * 70)
    logging.info(f"Experiments completed for Case {c_id}. Results saved to: {run_dir}")
    logging.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run STL-RRT* Experiments")
    parser.add_argument("--cases", type=str, help="Path to JSON cases file", default="benchmarks/benchmark_cases_2.json")
    parser.add_argument("--case_id", type=str, help="Specific case ID to run")
    parser.add_argument("--out_dir", type=str, help="Output root directory")
    parser.add_argument("--no_plots", action="store_true", help="Disable plotting")
    args = parser.parse_args()

    # Load Cases
    try:
        with open(args.cases, 'r') as f:
            data = json.load(f)
            cases = data.get("cases", [])
    except Exception as e:
        print(f"Error loading cases from {args.cases}: {e}")
        parent_path = os.path.join(os.path.dirname(__file__), "..", args.cases)
        if os.path.exists(parent_path):
             with open(parent_path, 'r') as f:
                data = json.load(f)
                cases = data.get("cases", [])
        else:
             return

    # Filter by case_id
    if args.case_id:
        cases = [c for c in cases if c.get("case_id") == args.case_id]
        if not cases:
            print(f"Case ID '{args.case_id}' not found.")
            return

    for case in cases:
        if "algorithms" in case and "stl_rrt" in case["algorithms"]:
            print(f"Running Case: {case['case_id']}")
            run_case(case, out_root=args.out_dir, save_plots=not args.no_plots)

if __name__ == "__main__":
    main()
