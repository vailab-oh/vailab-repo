#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Taegeun Oh
"""
TWA-RRT*: constant-speed path planning with arrival-time windows
======================================================
R7: Constant Velocity Model for Spatial-only Time Matching

[Description]
이 버전은 "시간 제약을 동역학적 속도 조절(가감속)이 아닌, 순수하게 기하학적 경로 길이 조절(Loitering, 우회)로만 해결한다"는 
논문의 핵심 기여를 강조하기 위해 작성되었습니다. 모든 속도는 상수 속도(v_max)로 고정되며, 
알고리즘이 목표 시간에 일찍 도착할 경우 속도를 줄이는 대신 물리적인 선회(Loiter) 궤적을 강제로 생성하도록 유도합니다.

2. **주요 구성 요소**:
   - **Dynamics**: 상수 속도 모델을 적용하여 주어진 경로 길이(L)에 대해 도달 시간을 결정합니다 (t = L / v_max). 
     가속도 제약을 제거하고 속도를 고정함으로써 Loiter Patch의 역할을 강조합니다.
   - **EuclideanRRTStar**: KDTree를 활용한 최적화된 RRT* 탐색을 수행하며, Lazy Collision Checking을 통해 탐색 속도를 높입니다.
   - **LoiterPatch**: 목표 시간보다 너무 일찍 도착하는 경우(Too Early), 목표 지점 인근에서 선회(Loiter)하여 도착 시간을 지연시키는 기법입니다.

3. **플래닝 파이프라인**:
   - **Step 1 (Geometric Search)**: RRT*를 통해 장애물이 없는 최단 경로를 탐색합니다.
   - **Step 2 (Feasibility Check)**: 발견된 경로가 시간 창 $[t_1, t_2]$ 내에 동역학적으로 도달 가능한지 판별합니다.
   - **Step 3 (Refinement)**:
     - *Direct*: 시간 창 만족 시 그대로 경로 채택.
     - *Loiter*: 너무 빠른 경우(Too Early), Loiter 구간을 추가하여 시간 조건을 만족시킵니다.
     - *Pruned*: 너무 늦은 경우(Too Late), 해당 경로는 기각합니다.

4. **특징**:
   - Feasible-First 접근 방식을 취하여, 첫 번째 유효한 해를 빠르게 찾는 데 집중합니다.
   - 경로 보정(Loiter Patch)을 통해 기하학적 최단 경로가 시간 제약을 충족하지 못하는 문제를 해결합니다.
"""

import math
import random
import time
import logging
import datetime
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple, Optional, Any, Dict
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache
import argparse
import sys
import json
# KD-Tree removed for fair comparison

# ============================================================
# CONFIGURATION
# ============================================================
@dataclass
class Config:
    # World
    world_bounds: Tuple[float, float, float, float] = (0, 220, 0, 160) # xmin, xmax, ymin, ymax
    obstacles: List[Tuple[float, float, float]] = field(default_factory=lambda: [
        (60, 60, 13), (120, 80, 14), (160, 110, 12)
    ])
    start: Tuple[float, float, float] = (30.0, 30.0, 0.0)
    goal: Tuple[float, float, float] = (180.0, 120.0, 0.0)
    
    # Time Constraints
    t_window: Tuple[float, float] = (7.0, 8.0) # [t1, t2] Strict target
    t_ref: float = 7.5  # reference arrival time (for evaluation); default aligns with window center
    time_window_mode: str = "fixed"  # 'fixed' or 'proxy_ratio'
    proxy_ratio: Dict[str, float] = field(default_factory=lambda: {"alpha": 1.0, "beta": 1.0})
    
    # --- Platform Profile: Small Fixed-Wing UAV ---
    v_min: float = 15.0  # m/s
    v_max: float = 25.0  # m/s
    a_max: float = 3.0   # m/s^2 (longitudinal)
    v0: float = 20.0     # m/s (initial speed)
    
    # --- Planner (Euclidean RRT*) ---
    max_iters: int = 6000
    step_size: float = 15.0
    goal_bias: float = 0.20
    goal_xy_tol: float = 15.0
    rrt_rewire_radius: float = 50.0
    
    # --- Detour Patch (Deprecated) ---
    # patch_margin: float = 2.0
    # min_seg_len: float = 10.0
    # max_patch_iter: int = 15
    
    # --- Utils ---
    collision_step: float = 2.0 
    seed: int = 42
    num_trials: int = 5
    save_dir_name: str = "rrt_runs"
    
    # --- TWA-RRT* ---
    # Weights
    w_L: float = 1.0
    w_tw: float = 0.05
    w_tw_pred: float = 0.5  # Guidance heuristic weight
    
    # Pruning & Constraints
    delta_late: float = 10.0
    p_tw_type: str = "hinge" # "hinge" or "squared_hinge"
    
    # Loiter
    r_hold: float = 20.0
    n_hold_samples: int = 10
    loiter_max_attempts: int = 5
    
    # --- Dynamic Time Window Parameters ---
    time_window_mode: str = "proxy_ratio" # "fixed" or "proxy_ratio"
    proxy_method: str = "euclid_scaled"   # "euclid_scaled"
    alpha: float = 0.10
    beta: float = 0.10
    kappa: float = 1.3
    
    # --- Debug / Instrumentation ---
    dump_progress_trace: bool = True
    progress_trace_fail_only: bool = True
    trace_dir_name: str = "traces"
    progress_event_mode: str = "milestone"  # "milestone" or "accepted"
    
    # --- Shortcut Repair (Option-1) ---
    enable_repair: bool = True
    repair_vio_eps_s: float = 0.20
    repair_max_attempts: int = 25
    repair_time_budget_s: float = 0.02
    repair_min_hops: int = 2
    repair_coarse_step: float = 2.0
    repair_fine_step: float = 0.5

CONFIG = Config()
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def resolve_time_window(case_override: Optional[Dict], config: Config) -> Tuple[float, float, Dict]:
    """Calculates window [Tmin, Tmax] and target t_ref [P0 Fix]"""
    # 1. Resolve Parameters
    alpha, beta, kappa = config.alpha, config.beta, config.kappa
    tw_mode = config.time_window_mode
    if case_override:
        alpha = case_override.get('alpha', alpha)
        beta = case_override.get('beta', beta)
        kappa = case_override.get('kappa', kappa)
        tw_mode = case_override.get('time_window_mode', tw_mode)
        
    # 2. Proxy Reference
    d = math.hypot(config.goal[0]-config.start[0], config.goal[1]-config.start[1])
    L_proxy = kappa * d
    t_ref_proxy = L_proxy / config.v_max
    
    # 3. Compute Window (P0 Fix: Preserve config.t_ref in fixed mode)
    if tw_mode == "fixed":
        t_min, t_max = config.t_window
        t_ref = config.t_ref 
    else:
        t_min = max(0.0, (1.0 - alpha) * t_ref_proxy)
        t_max = (1.0 + beta) * t_ref_proxy
        t_ref = t_ref_proxy
    
    meta = {
        't_ref': t_ref, 'L_proxy': L_proxy, 'alpha': alpha, 'beta': beta,
        'kappa': kappa, 'Tmin': t_min, 'Tmax': t_max, 'time_window_mode': tw_mode
    }
    return t_min, t_max, meta

# ============================================================
# JSON CASE LOADER Helpers
# ============================================================
def load_case_file(path: str) -> List[Dict]:
    """Load and parse JSON case file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Case file not found: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if "cases" not in data:
        raise ValueError("Invalid format: 'cases' list is missing")
    
    # Pre-normalize or just return raw?
    # We will normalize each case later
    return [normalize_case(c) for c in data["cases"]]

def normalize_case(case: Dict) -> Dict:
    """Validate and fill defaults for a case."""
    c_id = case.get("case_id", "unknown")
    
    # 1. World
    if "world" not in case:
        raise ValueError(f"Case {c_id}: 'world' block missing")
    w = case["world"]
    if "xlim" not in w or "ylim" not in w:
         raise ValueError(f"Case {c_id}: world must have xlim, ylim")
    
    # 2. Start/Goal
    if "start" not in case or "goal" not in case:
        raise ValueError(f"Case {c_id}: 'start' or 'goal' missing")
    
    # 3. Obstacles
    if "obstacles" not in case:
        case["obstacles"] = []
    
    # Validate Obstacles
    for i, obs in enumerate(case["obstacles"]):
        if obs.get("type") != "circle":
            raise ValueError(f"Case {c_id}: Unsupported obstacle type '{obs.get('type')}' at index {i}. Only 'circle' is supported.")
        if not all(k in obs for k in ("cx", "cy", "r")):
            raise ValueError(f"Case {c_id}: Circle obstacle {i} must have cx, cy, r")
            
    # 4. Time
    if "time" not in case:
        # Defaults to something reasonable or raise? Config has defaults.
        case["time"] = {"t_ref": 7.5, "t_window": [7.0, 8.0]}
    
    # 5. Dynamics
    if "dynamics" not in case:
        case["dynamics"] = {"v_min": 15.0, "v_max": 25.0, "a_max": 3.0, "v0": 20.0}
        
    # 6. Random
    if "random" not in case:
        case["random"] = {"seed_base": 42, "num_trials": 1}
        
    # 7. Planner (Common)
    if "planner" not in case:
        case["planner"] = {}
        
    # 8. Algorithms (Specific)
    if "algorithms" not in case:
        case["algorithms"] = {}
        
    # 9. Outputs
    if "outputs" not in case:
        case["outputs"] = {"out_dir": "output"}
        
    return case

def apply_case_to_config(case: Dict, base_config: Config) -> Config:
    """Override Config fields with case parameters."""
    import copy
    cfg = copy.deepcopy(base_config)
    
    # 1. World
    w = case["world"]
    cfg.world_bounds = (w["xlim"][0], w["xlim"][1], w["ylim"][0], w["ylim"][1])
    
    # 2. Start/Goal
    s = case["start"]
    g = case["goal"]
    cfg.start = (s["x"], s["y"], s.get("theta", 0.0)) # z=0 implicit
    cfg.goal = (g["x"], g["y"], g.get("theta", 0.0))
    
    # 3. Obstacles
    # Convert dicts to list of tuples (cx, cy, r)
    obs_tuples = []
    for o in case["obstacles"]:
        obs_tuples.append((o["cx"], o["cy"], o["r"]))
    cfg.obstacles = obs_tuples
    
    # 4. Time
    tm = case["time"]
    cfg.t_window = (tm["t_window"][0], tm["t_window"][1])
    cfg.t_ref = float(tm.get("t_ref", (cfg.t_window[0] + cfg.t_window[1]) / 2.0))
    
    # 5. Dynamics
    dyn = case["dynamics"]
    cfg.v_min = dyn["v_min"]
    cfg.v_max = dyn["v_max"]
    cfg.a_max = dyn["a_max"]
    cfg.v0 = dyn["v0"]
    
    # 6. Random
    rnd = case["random"]
    cfg.seed = rnd["seed_base"]
    cfg.num_trials = rnd["num_trials"]
    
    # 7. Planner (Common)
    pln = case["planner"]
    if "n_iters" in pln: cfg.max_iters = pln["n_iters"]
    if "goal_sample_rate" in pln: cfg.goal_bias = pln["goal_sample_rate"]
    if "step_size" in pln: cfg.step_size = pln["step_size"]
    if "neighbor_radius" in pln: cfg.rrt_rewire_radius = pln["neighbor_radius"]
    
    # 8. Output
    out_dir_override = case.get("outputs", {}).get("out_dir")
    if out_dir_override:
        cfg.save_dir_name = out_dir_override

    # 9. Algorithm Specific (WISER Override)
    algo = case["algorithms"].get("wiser_rrt", {})
    if "eps_state" in algo: cfg.goal_xy_tol = algo["eps_state"]
    if "goal_xy_tol" in algo: cfg.goal_xy_tol = algo["goal_xy_tol"]
    
    if "w_L" in algo: cfg.w_L = algo["w_L"]
    if "w_tw" in algo: cfg.w_tw = algo["w_tw"]
    if "w_tw_pred" in algo: cfg.w_tw_pred = algo["w_tw_pred"]
    
    if "delta_late" in algo: cfg.delta_late = algo["delta_late"]
    if "p_tw_type" in algo: cfg.p_tw_type = algo["p_tw_type"]
    
    if "r_hold" in algo: cfg.r_hold = algo["r_hold"]
    if "loiter_max_attempts" in algo: cfg.loiter_max_attempts = algo["loiter_max_attempts"]
    if "collision_step" in algo: cfg.collision_step = algo["collision_step"]
    
    if "time_window_mode" in algo: cfg.time_window_mode = algo["time_window_mode"]
    if "alpha" in algo: cfg.alpha = algo["alpha"]
    if "beta" in algo: cfg.beta = algo["beta"]
    if "kappa" in algo: cfg.kappa = algo["kappa"]
    
    return cfg

# ============================================================


# ============================================================
# DYNAMICS & MATH HELPERS
# ============================================================
class Dynamics:
    @staticmethod
    def feasible_time_range(L: float, v0: float, v_min: float, v_max: float, a_max: float) -> Tuple[float, float]:
        """Constant Velocity Model for Spatial-only Time Matching"""
        t_exact = L / v_max
        return t_exact, t_exact

    @staticmethod
    def P_tw(t: float, tw: Tuple[float, float], p_type: str = "hinge") -> float:
        """Time-window penalty function"""
        t1, t2 = tw
        if t < t1: return (t1 - t)**2 if p_type == "squared_hinge" else (t1 - t)
        if t > t2: return (t - t2)**2 if p_type == "squared_hinge" else (t - t2)
        return 0.0

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

class LoiterPatch:
    @staticmethod
    def generate_loiter(start_state: Tuple[float, float, float], 
                         goal_center: Tuple[float, float],
                         r_hold: float, 
                         delta_L_needed: float, 
                         obs_manager: 'ObstacleManager',
                         v_avg: float) -> Optional[Tuple[List, float]]:
        """
        [FIXED] P1: Correct orbit center, collision check, and goal return segment.
        """
        sx, sy, stheta = start_state
        gx, gy = goal_center

        # 1. Determine Circle Center tangent to heading (Left turn)
        cx = sx + r_hold * math.cos(stheta + math.pi/2)
        cy = sy + r_hold * math.sin(stheta + math.pi/2)
        
        # 2. Collision check at actual orbit center (cx, cy) and orbit path
        if not obs_manager.is_circle_safe(cx, cy, r_hold):
             return None

        # 3. Generate Loiter Points
        pts = []
        pts.append((sx, sy, stheta))
        
        tot_ang = delta_L_needed / r_hold
        step_ang = 0.1 
        cur_ang = 0.0
        start_radial = stheta - math.pi/2
        
        while cur_ang < tot_ang:
            ang = min(step_ang, tot_ang - cur_ang)
            cur_ang += ang
            px = cx + r_hold * math.cos(start_radial + cur_ang)
            py = cy + r_hold * math.sin(start_radial + cur_ang)
            ptheta = stheta + cur_ang # Tangent heading
            pts.append((px, py, ptheta))
            
        # 4. Goal Return: Add straight segment from loiter exit to goal center
        # This ensures dist_to_goal ~ 0 for success criteria and better plotting.
        last_x, last_y, _ = pts[-1]
        dist_to_goal = math.hypot(gx - last_x, gy - last_y)
        if dist_to_goal > 0.05:
             # Check collision for return segment? [R5 addition]
             ret_path_inner = EuclideanMath.shortest_path((last_x, last_y), (gx, gy))
             if obs_manager.check_path_collision_lazy(ret_path_inner, (last_x, last_y), 2.0, 0.5):
                  # If return is blocked, this loiter is invalid
                  return None
                  
             n_ret = 5
             for i in range(1, n_ret + 1):
                 f = i / n_ret
                 pts.append((
                     last_x + f * (gx - last_x),
                     last_y + f * (gy - last_y),
                     math.atan2(gy - last_y, gx - last_x)
                 ))
        
        actual_L = delta_L_needed + dist_to_goal
        path_obj = EuclideanMath.Path(length=actual_L, points=pts)
        return [(path_obj, start_state)], actual_L

class EuclideanMath:
    @dataclass
    class Path:
        length: float
        points: List[Tuple[float, float, float]] # x, y, theta
        
    @staticmethod
    def shortest_path(p1, p2):
        """Return a straight-line (Euclidean) path between two states/points.

        Accepts p1/p2 as (x,y) or (x,y,theta). The returned Path stores points as (x,y,theta).
        """
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        dx = x2 - x1
        dy = y2 - y1
        dist = math.hypot(dx, dy)
        theta = math.atan2(dy, dx) if dist > 1e-12 else (float(p1[2]) if len(p1) >= 3 else 0.0)
        return EuclideanMath.Path(length=dist, points=[(x1, y1, theta), (x2, y2, theta)])

    @staticmethod
    def interpolate(dist, path, start_state):
        if path.length < 1e-6: return path.points[0]
        
        # --- Multi-point Path Support (Loiter) ---
        if len(path.points) > 2:
             # Find segment
             # Uniform spacing assumption or cumulative?
             # Loiter points are generated with specific spacing. 
             # We need cumulative dists.
             # Lazy approach: 
             # If points are many, we assume they are dense enough or we just pick closest?
             # Better: Calculate cumulative distances if not stored.
             # For this specific Loiter implementation, we returned a list of points.
             # Let's assume uniform distribution for Loiter Circle? 
             # Actually loiter patch returns many points.
             # Map dist to index.
             
             # Total length is path.length.
             t = dist / path.length
             if t < 0: t = 0
             if t > 1: t = 1
             
             idx_float = t * (len(path.points) - 1)
             idx = int(idx_float)
             if idx >= len(path.points) - 1: return path.points[-1]
             
             p1 = path.points[idx]
             p2 = path.points[idx+1]
             
             # Local t
             t_local = idx_float - idx
             x = p1[0] + t_local * (p2[0] - p1[0])
             y = p1[1] + t_local * (p2[1] - p1[1])
             
             # Heading interpolation might be needed for Dubins/Loiter? 
             # Loiter points have heading.
             h1 = p1[2]
             h2 = p2[2]
             # simple linear interp for heading (careful with wrap)
             dh = h2 - h1
             if dh > math.pi: dh -= 2*math.pi
             elif dh < -math.pi: dh += 2*math.pi
             h = h1 + t_local * dh
             return (x, y, h)

        t = dist / path.length
        if t < 0: t = 0
        if t > 1: t = 1
        p1 = path.points[0]
        p2 = path.points[1]
        x = p1[0] + t * (p2[0] - p1[0])
        y = p1[1] + t * (p2[1] - p1[1])
        return (x, y, p2[2]) # Use heading of segment
    
    @staticmethod
    def sample_path(path, start_state, step):
        pts = []
        d = 0.0
        while d < path.length:
            pts.append(EuclideanMath.interpolate(d, path, start_state))
            d += step
        pts.append(path.points[-1]) # Use last point
        return pts

class ObstacleManager:
    def __init__(self, config: Config):
        self.config = config
        self.obs_list = config.obstacles
        self.stats = {'checks': 0, 'coarse_pass': 0, 'fine_pass': 0}
        logging.info(f"ObstacleManager initialized with {len(self.obs_list)} obstacles")

    @lru_cache(maxsize=100000)
    def is_collision_raw(self, x, y):
        # Point collision
        if not (self.config.world_bounds[0] <= x <= self.config.world_bounds[1] and
                self.config.world_bounds[2] <= y <= self.config.world_bounds[3]):
            return True
        for ox, oy, r in self.obs_list:
              if (x - ox)**2 + (y - oy)**2 <= r*r:
                  return True
        return False

    def is_collision(self, x, y):
        return self.is_collision_raw(x, y)

    def is_circle_safe(self, cx, cy, radius, step=0.1):
        """Checks if a circle at (cx, cy) with radius is collision-free."""
        # Check center
        if self.is_collision_raw(cx, cy): return False
        # Sample points on the perimeter
        n_pts = int(2 * math.pi * radius / step)
        n_pts = max(12, min(100, n_pts))
        for i in range(n_pts):
            th = 2 * math.pi * i / n_pts
            px = cx + radius * math.cos(th)
            py = cy + radius * math.sin(th)
            if self.is_collision_raw(px, py): return False
        return True

    def check_path_collision(self, points, coarse_step=None):
        self.stats['checks'] += 1
        if coarse_step is not None and len(points) > 1:
            step_idx = max(1, int(coarse_step / self.config.collision_step))
            for i in range(0, len(points), step_idx):
                if self.is_collision_raw(points[i][0], points[i][1]): return True
            self.stats['coarse_pass'] += 1

        for p in points:
             if self.is_collision_raw(p[0], p[1]): return True
        self.stats['fine_pass'] += 1
        return False

    def check_path_collision_lazy(self, path_data, start_state, coarse_step, fine_step):
        self.stats['checks'] += 1
        # 1. Coarse
        d = 0.0
        while d < path_data.length:
             state = EuclideanMath.interpolate(d, path_data, start_state)
             if self.is_collision_raw(state[0], state[1]): return True
             d += coarse_step
        if self.is_collision_raw(*EuclideanMath.interpolate(path_data.length, path_data, start_state)[:2]): return True
        self.stats['coarse_pass'] += 1

        # 2. Fine
        d = 0.0
        while d < path_data.length:
             state = EuclideanMath.interpolate(d, path_data, start_state)
             if self.is_collision_raw(state[0], state[1]): return True
             d += fine_step
             
        self.stats['fine_pass'] += 1
        return False

# ============================================================
# TRAJECTORY NODE & RRT*
# ============================================================

class Node:
    def __init__(self, x, y, theta):
        self.state = (x, y, theta) # Theta is segment heading or 0
        self.parent: Optional[Node] = None
        
        # Costs
        self.cost = 0.0      # Euclidean Cost (Length) - Kept for compatibility
        self.cost_J = 0.0    # Weighted Cost J(pi) = w_L*L + w_tw*P_tw(t)
        self.L = 0.0         # Accumulated Length
        self.t = 0.0         # Accumulated Time = L / v_avg
        
        self.path_from_parent: Optional[EuclideanMath.Path] = None
        self.detailed_path: List[Tuple[float, float, float]] = None # Lazy
        self.children: List[Node] = []

class EuclideanRRTStar:
    def __init__(self, config: Config, obs_manager: ObstacleManager):
        self.config = config
        self.obs = obs_manager
        
        max_n = config.max_iters + 500
        self.node_coords = np.zeros((max_n, 2), dtype=np.float32)
        
        self.start_node = Node(*config.start)
        # WISER init
        self.start_node.L = 0.0
        self.start_node.t = 0.0
        self.start_node.cost_J = config.w_tw * Dynamics.P_tw(0.0, config.t_window, config.p_tw_type) # Usually large if t=0
        # Actually t=0 is EARLY. P_tw(0) = max(7-0, 0) = 7. 
        
        self.nodes = [self.start_node]
        self.node_coords[0] = [config.start[0], config.start[1]]
        self.count = 1
        
        # KDTree removed for fair comparison
        
        self.goal = config.goal
        self.first_sol = None
        self.first_total_err = None # WISER: Track first solution error
        
        # Best J tracking
        self.best_J = float('inf') 
        
        self.stats = {
            'dubins_calls': 0, 'n_collision': 0, 
            'rewire_count': 0, 'rewire_success': 0,
            'kdtree_rebuilds': 0,
            'iters_to_first_feasible': -1,
            'iters_best': 0,
            'refine_iters': 0,
            'near_nodes_sum': 0,
            'near_nodes_count': 0,
            'iters_total': 0,
            # WISER Stats
            'late_pruned': 0,
            'lb_pruned': 0,
            'loiter_used': 0,
            # Instrumentation counters
            'collision_reject': 0,
            'no_parent': 0,
            'goal_connect_fail': 0,
            'loiter_fail': 0,
            'goal_region_hits': 0,
            # Extended instrumentation
            'skip_point_collision': 0,
            'skip_edge_collision': 0,
            'accepted_nodes': 0,
            'goal_connect_attempt': 0,
            'goal_connect_success': 0,
            'goal_connect_fail_collision': 0,
            'goal_late_infeasible': 0,
            'goal_early_infeasible': 0,
            'goal_feasible_hits': 0,
            'best_d_goal': float('inf'),
            'best_d_goal_iter': 0,
            'best_d_goal_path_len': 0.0,
            'best_d_goal_tmin_to_goal': 0.0,
            # First-hit iteration tracking
            'iters_to_first_goal_hit': -1,
            'iters_to_first_goal_feasible': -1,
            'iters_to_first_goal_attempt': -1,
            # Best t_hat tracking
            'best_d_goal_t_hat': float('inf'),
            'best_t_hat': float('inf'),
            'best_t_hat_iter': -1,
            # Repair stats
            'repair_calls': 0,
            'repair_applied': 0,
            'repair_to_feasible': 0,
            'repair_attempts_used': 0,
            'repair_delta_L': 0.0,
            'repair_vio_before': 0.0,
            'repair_vio_after': 0.0
        }
        self.progress_trace: list = []

    def get_nearest_node(self, state) -> Optional[Node]:
        # R6: Linear Search Only for Fair Comparison
        best_node, min_dist_sq = None, float('inf')
        sx, sy = state[0], state[1]
        for n in self.nodes:
            d2 = (n.state[0] - sx)**2 + (n.state[1] - sy)**2
            if d2 < min_dist_sq:
                min_dist_sq, best_node = d2, n
        return best_node

    def create_goal_node(self, parent: Node, goal_state: Tuple[float, float, float]) -> Optional[Node]:
        """Creates a node exactly at goal state from parent."""
        dist = math.hypot(goal_state[0]-parent.state[0], goal_state[1]-parent.state[1])
        # Force connection (Dubins or Euclidean)
        path = EuclideanMath.shortest_path(parent.state, goal_state)
        
        # Check collision
        if self.obs.check_path_collision_lazy(path, parent.state, 2.0, 0.5):
            self.stats['goal_connect_fail_collision'] += 1
            return None
            
        new_node = Node(*goal_state)
        new_node.parent = parent
        new_node.path_from_parent = path
        new_node.children = []
        
        new_node.cost = parent.cost + path.length
        
        # L and J calc
        new_node.L = parent.L + path.length
        # Optimal T and Cost
        if hasattr(self, 'get_optimal_t_and_cost'):
             t_opt, new_J = self.get_optimal_t_and_cost(new_node.L)
             new_node.t = t_opt
             new_node.cost_J = new_J
             
        return new_node

    def get_near_nodes(self, center, radius, k_limit=None) -> List[Node]:
        # R6: Linear Search Only for Fair Comparison
        near_nodes = []
        r2, cx, cy = radius**2, center[0], center[1]
        for n in self.nodes:
            d2 = (n.state[0] - cx)**2 + (n.state[1] - cy)**2
            if d2 <= r2:
                near_nodes.append(n)
        
        if k_limit and len(near_nodes) > k_limit:
            near_nodes.sort(key=lambda n: (n.state[0]-cx)**2 + (n.state[1]-cy)**2)
            near_nodes = near_nodes[:k_limit]
        return near_nodes

    def get_heuristic(self, node_state):
        return math.hypot(node_state[0] - self.goal[0], node_state[1] - self.goal[1])

    def get_optimal_t_and_cost(self, L: float) -> Tuple[float, float]:
        """Calculates optimal time arrival t and associated cost J for a given path length L."""
        t_min, t_max = Dynamics.feasible_time_range(
             L, self.config.v_max, self.config.v_min, self.config.v_max, self.config.a_max
        )
        # Use dynamic window stored in self.t_window
        t1, t2 = self.t_window 
        
        # Optimal t minimizes J = w_L*L + w_tw*P_tw(t)
        # Since L is fixed here, we minimize P_tw(t).
        # P_tw is 0 inside [t1, t2].
        
        t_opt = t_min # Default
        
        # Intersection of [t_min, t_max] and [t1, t2]
        start = max(t_min, t1)
        end = min(t_max, t2)
        
        if start <= end:
            t_opt = start 
            p_val = 0.0
        else:
            # No overlap
            if t_min > t2:
                # Late. Best we can do is t_min.
                t_opt = t_min
            else:
                # Early. Best is t_max.
                t_opt = t_max
            p_val = Dynamics.P_tw(t_opt, self.t_window, self.config.p_tw_type)
            
        J = self.config.w_L * L + self.config.w_tw * p_val
        return t_opt, J

    def compute_F(self, node: Node, new_state: Tuple[float, float, float], path_len: float):
        # F(n) = J(n) + w_L * d_g(x) + w_tw_pred * P_tw(t_hat)
        
        new_L = node.L + path_len
        t_opt, J_current = self.get_optimal_t_and_cost(new_L)
        
        dist_to_goal = self.get_heuristic(new_state)
        
        # Heuristic Cost
        # t_hat: Estimated time at goal.
        # We assume max speed for heuristic to be optimistic (admissible)?
        # t_hat_min = t_opt + dist / v_max
        t_hat = t_opt + dist_to_goal / self.config.v_max
        
        p_val_heuristic = Dynamics.P_tw(t_hat, self.t_window, self.config.p_tw_type)
        
        f_score = J_current + self.config.w_L * dist_to_goal + self.config.w_tw_pred * p_val_heuristic
        
        return f_score, J_current, new_L, t_opt

    def choose_parent(self, new_state, near_nodes):
        best_parent = None
        best_path = None
        min_F = float('inf')
        
        best_J_cand = float('inf')
        best_L_cand = 0.0
        best_t_cand = 0.0
        
        for n in near_nodes:
            self.stats['dubins_calls'] += 1
            dist = math.hypot(new_state[0]-n.state[0], new_state[1]-n.state[1])
            
            f_val, j_val, l_val, t_val = self.compute_F(n, new_state, dist)
            
            # (A) Viability Prune (Late-Infeasible)
            dist_to_goal = self.get_heuristic(new_state)
            t_min_new, _ = Dynamics.feasible_time_range(l_val, self.config.v_max, self.config.v_min, self.config.v_max, self.config.a_max)
            t_hat_earliest = t_min_new + dist_to_goal / self.config.v_max

            if t_hat_earliest > self.t_window[1] + self.config.delta_late:
                self.stats['late_pruned'] += 1
                continue

            # (A) LB Prune
            j_lb = j_val + self.config.w_L * dist_to_goal
            if j_lb >= self.best_J:
                 self.stats['lb_pruned'] += 1 
                 continue
                
            if f_val < min_F:
                path = EuclideanMath.shortest_path(n.state, new_state)
                self.stats['n_collision'] += 1
                if not self.obs.check_path_collision_lazy(path, n.state, 2.0, 0.5):
                    min_F = f_val
                    best_parent = n
                    best_path = path
                    best_J_cand = j_val
                    best_L_cand = l_val
                    best_t_cand = t_val
                else:
                    self.stats['collision_reject'] += 1

        return best_parent, best_path, best_J_cand, best_L_cand, best_t_cand

    def propagate_cost_to_children(self, node: Node):
        stack = [node]
        while stack:
            curr = stack.pop()
            if hasattr(curr, 'children'):
                for child in curr.children:
                    if child.path_from_parent:
                        dist = child.path_from_parent.length
                    else:
                        dist = math.hypot(child.state[0]-curr.state[0], child.state[1]-curr.state[1])
                    
                    child.L = curr.L + dist
                    # Re-optimize t and J
                    t_opt, J_new = self.get_optimal_t_and_cost(child.L)
                    child.t = t_opt
                    child.cost_J = J_new
                    
                    child.cost = curr.cost + dist # Copatibility
                    
                    stack.append(child)

    def rewire(self, new_node: Node, near_nodes: List[Node]):
        for near_node in near_nodes:
            if near_node == new_node.parent: continue
            
            dist = math.hypot(new_node.state[0]-near_node.state[0], new_node.state[1]-near_node.state[1])
            
            new_L = new_node.L + dist
            # Optimistic check before collision
            # We want to see if coming from new_node effectively reduces cost_J of near_node
            t_opt, new_J = self.get_optimal_t_and_cost(new_L)
            
            if new_J >= near_node.cost_J:
                continue
            
            # Cycle Check
            curr = new_node
            is_ancestor = False
            while curr is not None:
                if curr == near_node:
                    is_ancestor = True; break
                curr = curr.parent
            if is_ancestor: continue

            self.stats['dubins_calls'] += 1
            path = EuclideanMath.shortest_path(new_node.state, near_node.state)
            
            self.stats['n_collision'] += 1
            if not self.obs.check_path_collision_lazy(path, new_node.state, 2.0, 0.5):
                self.stats['rewire_count'] += 1
                
                if near_node.parent and hasattr(near_node.parent, 'children') and near_node in near_node.parent.children:
                    near_node.parent.children.remove(near_node)

                near_node.parent = new_node
                near_node.path_from_parent = path
                near_node.detailed_path = None
                
                near_node.L = new_L
                near_node.t = t_opt
                near_node.cost_J = new_J
                near_node.cost = new_node.cost + path.length
                
                if not hasattr(new_node, 'children'): new_node.children = []
                new_node.children.append(near_node)
                
                self.propagate_cost_to_children(near_node)
                self.stats['rewire_success'] += 1

    # --- Shortcut Repair Helpers ---
    def backtrace_waypoints(self, node):
        """Backtrace parent chain from node to start, return ordered waypoint states."""
        chain = []
        curr = node
        while curr is not None:
            chain.append(curr.state)
            curr = curr.parent
        chain.reverse()
        return chain

    def segments_from_waypoints(self, wps):
        """Build segment list [(Path, start_state), ...] from ordered waypoints."""
        segs = []
        for k in range(len(wps) - 1):
            path = EuclideanMath.shortest_path(wps[k], wps[k+1])
            segs.append((path, wps[k]))
        return segs

    def shortcut_repair_waypoints(self, wps, max_attempts, time_budget_s):
        """Greedy shortcut repair on a waypoint list.
        Returns (wps_out, L_out, applied, attempts_used).
        """
        t0 = time.time()
        applied = False
        attempts_used = 0
        min_hops = self.config.repair_min_hops

        for _ in range(max_attempts):
            if time.time() - t0 > time_budget_s:
                break
            n = len(wps)
            if n < min_hops + 2:
                break
            attempts_used += 1
            i = random.randint(0, n - min_hops - 2)
            j = random.randint(i + min_hops + 1, n - 1)
            path = EuclideanMath.shortest_path(wps[i], wps[j])
            if not self.obs.check_path_collision_lazy(path, wps[i], self.config.repair_coarse_step, self.config.repair_fine_step):
                wps = wps[:i+1] + wps[j:]
                applied = True

        # Compute total length from consecutive waypoint distances
        L_out = sum(math.hypot(wps[k+1][0]-wps[k][0], wps[k+1][1]-wps[k][1]) for k in range(len(wps)-1))
        return wps, L_out, applied, attempts_used

    def search(self, t_window: Optional[Tuple[float, float]] = None) -> List[Node]:
        self.start_time = time.time()
        self.best_J = float('inf')
        
        # Dynamic Time Window Injection
        if t_window:
            self.t_window = t_window
        else:
            self.t_window = self.config.t_window
            
        t1, t2 = self.t_window
        v_est = self.config.v_max 
        
        # Stats initialization
        self.stats.update({
            'iters_to_first_feasible': -1, 'refine_iters': 0, 'iters_best': 0,
            'late_pruned': 0, 'lb_pruned': 0, 'loiter_used': 0,
            'collision_reject': 0, 'no_parent': 0, 'goal_connect_fail': 0,
            'loiter_fail': 0, 'goal_region_hits': 0,
            'skip_point_collision': 0, 'skip_edge_collision': 0, 'accepted_nodes': 0,
            'goal_connect_attempt': 0, 'goal_connect_success': 0, 'goal_connect_fail_collision': 0,
            'goal_late_infeasible': 0, 'goal_early_infeasible': 0, 'goal_feasible_hits': 0,
            'best_d_goal': float('inf'), 'best_d_goal_iter': 0,
            'best_d_goal_path_len': 0.0, 'best_d_goal_tmin_to_goal': 0.0,
            'iters_to_first_goal_hit': -1, 'iters_to_first_goal_feasible': -1,
            'iters_to_first_goal_attempt': -1,
            'best_d_goal_t_hat': float('inf'),
            'best_t_hat': float('inf'), 'best_t_hat_iter': -1,
            'repair_calls': 0, 'repair_applied': 0, 'repair_to_feasible': 0,
            'repair_attempts_used': 0, 'repair_delta_L': 0.0,
            'repair_vio_before': 0.0, 'repair_vio_after': 0.0
        })
        
        self.cost_history = [] 
        self.progress_trace = []
        self.first_sol = None
        self.first_sol_time = None
        self.first_path_len = None
        self.first_total_err = None
        
        for i in range(self.config.max_iters):
            self.stats['iters_total'] = i + 1
            
            if random.random() < self.config.goal_bias:
                rnd_state = self.config.goal
            else:
                rx = random.uniform(self.config.world_bounds[0], self.config.world_bounds[1])
                ry = random.uniform(self.config.world_bounds[2], self.config.world_bounds[3])
                rt = random.uniform(-math.pi, math.pi)
                rnd_state = (rx, ry, rt)
            
            nearest_node = self.get_nearest_node(rnd_state)
            if nearest_node is None: continue
            
            self.stats['dubins_calls'] += 1
            path_near = EuclideanMath.shortest_path(nearest_node.state, rnd_state)
            
            final_state = rnd_state
            if path_near.length > self.config.step_size:
                final_state = EuclideanMath.interpolate(self.config.step_size, path_near, nearest_node.state)
                path_near = EuclideanMath.shortest_path(nearest_node.state, final_state)
            
            # Late Prune (Check against best possible time)
            dist_to_new = path_near.length
            # Best possible time to reach new node: nearest.t + dist/v_max 
            # (Wait, nearest.t is already optimized for L. Can we fly faster? 
            # No, nearest.t is effectively L/v_something. Actually optimize_t uses L.
            # So optimal time to new node is based on (nearest.L + dist).)
            L_pred = nearest_node.L + dist_to_new
            t_min_pred, _ = Dynamics.feasible_time_range(L_pred, self.config.v_max, self.config.v_min, self.config.v_max, self.config.a_max)
            # earliest arrival (kinematically feasible)
            t_hat_earliest = t_min_pred + self.get_heuristic(final_state) / self.config.v_max
            
            if t_hat_earliest > t2 + self.config.delta_late:
                 self.stats['late_pruned'] += 1
                 continue
            
            if self.obs.is_collision_raw(final_state[0], final_state[1]):
                self.stats['collision_reject'] += 1
                self.stats['skip_point_collision'] += 1
                continue
            if self.obs.check_path_collision_lazy(path_near, nearest_node.state, 2.0, 0.5):
                self.stats['collision_reject'] += 1
                self.stats['skip_edge_collision'] += 1
                continue
            
            n_nodes = len(self.nodes)
            radius = min(self.config.rrt_rewire_radius, 50.0 * math.sqrt(math.log(n_nodes+1)/(n_nodes+1)))
            radius = max(radius, 5.0)
            near_nodes = self.get_near_nodes(final_state, radius, k_limit=50)
            
            self.stats['near_nodes_sum'] += len(near_nodes)
            self.stats['near_nodes_count'] += 1
            
            if nearest_node not in near_nodes: near_nodes.append(nearest_node)
                
            parent, d_path, j_cand, l_cand, t_cand = self.choose_parent(final_state, near_nodes)
            if not parent:
                self.stats['no_parent'] += 1
                continue
            
            new_node = Node(*final_state)
            new_node.parent = parent
            new_node.path_from_parent = d_path
            new_node.children = []
            
            new_node.cost = parent.cost + d_path.length
            new_node.L = l_cand
            new_node.t = t_cand
            new_node.cost_J = j_cand
            
            # We assume t_cand is already optimal from choose_parent->compute_F
            
            if not hasattr(parent, 'children'): parent.children = []
            parent.children.append(new_node)
            
            if self.count < len(self.node_coords):
                self.node_coords[self.count] = [new_node.state[0], new_node.state[1]]
                self.nodes.append(new_node)
                self.count += 1
                self.stats['accepted_nodes'] += 1
            else:
                 logging.warning("Node buffer full.")
                 break
            
            self.rewire(new_node, near_nodes)
            
            # --- Track best distance to goal & best t_hat ---
            d_to_goal = math.hypot(new_node.state[0] - self.goal[0], new_node.state[1] - self.goal[1])
            tmin_node, _ = Dynamics.feasible_time_range(new_node.L, self.config.v_max, self.config.v_min, self.config.v_max, self.config.a_max)
            t_hat = tmin_node + d_to_goal / self.config.v_max
            
            if d_to_goal < self.stats['best_d_goal']:
                self.stats['best_d_goal'] = d_to_goal
                self.stats['best_d_goal_iter'] = i + 1
                self.stats['best_d_goal_path_len'] = new_node.L
                L_est = new_node.L + d_to_goal
                t_min_g, _ = Dynamics.feasible_time_range(L_est, self.config.v_max, self.config.v_min, self.config.v_max, self.config.a_max)
                self.stats['best_d_goal_tmin_to_goal'] = t_min_g
                self.stats['best_d_goal_t_hat'] = t_hat
                if self.config.dump_progress_trace and self.config.progress_event_mode == 'milestone':
                    self.progress_trace.append({'iter': i+1, 'nodes': self.count, 'event': 'BEST_D_GOAL', 'd_goal': f"{d_to_goal:.3f}", 'L': f"{new_node.L:.1f}", 'tmin': f"{tmin_node:.3f}", 'tmax': '', 'vio': '', 't_hat': f"{t_hat:.3f}", 'best_J': f"{self.best_J:.4f}" if self.best_J < float('inf') else ''})
            
            if t_hat < self.stats['best_t_hat']:
                self.stats['best_t_hat'] = t_hat
                self.stats['best_t_hat_iter'] = i + 1
                if self.config.dump_progress_trace and self.config.progress_event_mode == 'milestone':
                    self.progress_trace.append({'iter': i+1, 'nodes': self.count, 'event': 'BEST_T_HAT', 'd_goal': f"{d_to_goal:.3f}", 'L': f"{new_node.L:.1f}", 'tmin': f"{tmin_node:.3f}", 'tmax': '', 'vio': '', 't_hat': f"{t_hat:.3f}", 'best_J': f"{self.best_J:.4f}" if self.best_J < float('inf') else ''})
            
            # --- Goal Check & Exact Connection ---
            if d_to_goal <= self.config.goal_xy_tol:
                # Arrived at Goal Region.
                self.stats['goal_region_hits'] += 1
                if self.stats['iters_to_first_goal_hit'] < 0:
                    self.stats['iters_to_first_goal_hit'] = i + 1
                # Try to connect exactly
                self.stats['goal_connect_attempt'] += 1
                if self.stats['iters_to_first_goal_attempt'] < 0:
                    self.stats['iters_to_first_goal_attempt'] = i + 1
                exact_goal_node = self.create_goal_node(new_node, self.goal)
                if exact_goal_node:
                    self.stats['goal_connect_success'] += 1
                    # Append it to the tree
                    if self.count < len(self.node_coords):
                        self.nodes.append(exact_goal_node)
                        self.node_coords[self.count] = [exact_goal_node.state[0], exact_goal_node.state[1]]
                        self.count += 1
                        
                        goal_node = exact_goal_node
                    else:
                        goal_node = new_node
                else:
                    self.stats['goal_connect_fail'] += 1
                    goal_node = new_node
                
                # Now check success on goal_node
                # 1. Direct Time Check
                t_min, t_max = Dynamics.feasible_time_range(goal_node.L, self.config.v_max, self.config.v_min, self.config.v_max, self.config.a_max)
                
                final_L = goal_node.L
                final_t = goal_node.t 
                loiter_seg = None
                
                # Check for Early Need
                if t_max < t1:
                    # Even at slowest speed, we are early. Need Loiter.
                    delta_t = t1 - t_max
                    delta_L = delta_t * self.config.v_max
                    
                    loiter_res = LoiterPatch.generate_loiter(
                        goal_node.state, self.goal[:2], self.config.r_hold, delta_L, self.obs, v_est
                    )
                    
                    if loiter_res:
                         patches, real_dL = loiter_res
                         loiter_seg = patches
                         final_L += real_dL
                         final_t = t_max + real_dL / v_est
                         self.stats['loiter_used'] = 1
                    else:
                         self.stats['loiter_fail'] += 1
                         continue
                
                # Time feasibility classification (using final_L after loiter)
                tmin_g, tmax_g = Dynamics.feasible_time_range(final_L, self.config.v_max, self.config.v_min, self.config.v_max, self.config.a_max)
                vio_g = Dynamics.time_window_violation(tmin_g, tmax_g, (t1, t2))
                if tmin_g > t2:
                    self.stats['goal_late_infeasible'] += 1
                elif tmax_g < t1:
                    self.stats['goal_early_infeasible'] += 1
                if vio_g == 0.0:
                    self.stats['goal_feasible_hits'] += 1
                    if self.stats['iters_to_first_goal_feasible'] < 0:
                        self.stats['iters_to_first_goal_feasible'] = i + 1
                    if self.stats['iters_to_first_feasible'] < 0:
                        self.stats['iters_to_first_feasible'] = i + 1
                
                # Progress trace event
                if self.config.dump_progress_trace and self.config.progress_event_mode == 'milestone':
                    self.progress_trace.append({'iter': i+1, 'nodes': self.count, 'event': 'GOAL_HIT', 'd_goal': f"{d_to_goal:.3f}", 'L': f"{final_L:.1f}", 'tmin': f"{tmin_g:.3f}", 'tmax': f"{tmax_g:.3f}", 'vio': f"{vio_g:.3f}", 't_hat': '', 'best_J': f"{self.best_J:.4f}" if self.best_J < float('inf') else ''})
                
                # --- Shortcut Repair (near-miss late only) ---
                if (self.config.enable_repair
                        and vio_g > 0
                        and vio_g <= self.config.repair_vio_eps_s
                        and tmin_g > t2
                        and loiter_seg is None):
                    self.stats['repair_calls'] += 1
                    wps = self.backtrace_waypoints(goal_node)
                    if len(wps) >= 3:
                        old_final_L = final_L
                        wps2, L2, rep_applied, rep_attempts = self.shortcut_repair_waypoints(
                            wps, self.config.repair_max_attempts, self.config.repair_time_budget_s)
                        if rep_applied and L2 < final_L - 1e-6:
                            final_L = L2
                            tmin2, tmax2 = Dynamics.feasible_time_range(final_L, self.config.v_max, self.config.v_min, self.config.v_max, self.config.a_max)
                            vio_after = Dynamics.time_window_violation(tmin2, tmax2, (t1, t2))
                            final_t = Dynamics.get_arrival_t_best(tmin2, tmax2, self.config.t_ref, (t1, t2))
                            repaired_segments = self.segments_from_waypoints(wps2)
                            goal_node.final_segments = repaired_segments
                            self.stats['repair_applied'] = 1
                            self.stats['repair_attempts_used'] = rep_attempts
                            self.stats['repair_delta_L'] = old_final_L - final_L
                            self.stats['repair_vio_before'] = vio_g
                            self.stats['repair_vio_after'] = vio_after
                            if vio_after == 0.0:
                                self.stats['repair_to_feasible'] = 1
                                # Update feasibility classification
                                self.stats['goal_feasible_hits'] += 1
                                if self.stats['iters_to_first_goal_feasible'] < 0:
                                    self.stats['iters_to_first_goal_feasible'] = i + 1
                                if self.stats['iters_to_first_feasible'] < 0:
                                    self.stats['iters_to_first_feasible'] = i + 1
                
                # Final J calc
                p_val = Dynamics.P_tw(final_t, self.t_window, self.config.p_tw_type)
                j_goal = self.config.w_L * final_L + self.config.w_tw * p_val
                
                if j_goal < self.best_J:
                    self.best_J = j_goal
                    
                    if self.first_sol is None:
                        elapsed = time.time() - self.start_time
                        self.first_sol = (elapsed, j_goal, goal_node)
                        self.first_sol_time = elapsed
                        self.first_path_len = final_L
                        
                        # time_err = 0.0 if (t1 <= final_t <= t2) else min(abs(final_t-t1), abs(final_t-t2))
                        time_err = abs(final_t - self.config.t_ref)
                        
                        # Spatial Error (Approach Distance):
                        # Walk up parent chain to find the first ancestor NOT at goal
                        gx_f, gy_f = self.goal[0], self.goal[1]
                        s_err_dist = math.hypot(goal_node.state[0]-gx_f, goal_node.state[1]-gy_f)
                        if s_err_dist < 1e-3:
                            approach = goal_node.parent
                            while approach and math.hypot(approach.state[0]-gx_f, approach.state[1]-gy_f) < 1e-3:
                                approach = approach.parent
                            spatial_err = math.hypot(approach.state[0]-gx_f, approach.state[1]-gy_f) if approach else 0.0
                        else:
                            spatial_err = s_err_dist
                             
                        self.first_total_err = math.hypot(spatial_err, time_err)
                        logging.info(f"First Sol (WISER): CostJ={j_goal:.2f}, L={final_L:.1f}, T={final_t:.2f}, SpErr={spatial_err:.3f}")

                    elapsed = time.time() - self.start_time
                    self.cost_history.append((elapsed, j_goal))
                    self.stats['iters_best'] = i + 1
                    
                    goal_node.loiter_data = loiter_seg
                    goal_node.final_L = final_L
                    goal_node.final_t = final_t
                    goal_node.final_J = j_goal
                    
            if (i+1) % 500 == 0:
                logging.info(f"Iter {i+1}, Nodes: {self.count}, BestJ: {self.best_J:.2f}, LatePruned: {self.stats['late_pruned']}")
        
        # Finalize accepted_nodes and acceptance_rate
        self.stats['accepted_nodes'] = int(self.count)
        iters_fin = max(1, int(self.stats.get('iters_total', 0)))
        self.stats['acceptance_rate'] = float(self.count) / float(iters_fin)
            
        return self.nodes

    def get_goal_candidates(self) -> List[Node]:
        """Return goal candidates.

        - Primary: nodes explicitly validated during search (have final_J / final_t / final_L).
        - Fallback: raw nodes inside goal_xy_tol that also satisfy the time window.
        """
        t1, t2 = self.t_window if hasattr(self, 't_window') else self.config.t_window

        validated = [n for n in self.nodes if hasattr(n, 'final_J')]
        if validated:
            return validated

        cands: List[Node] = []
        for n in self.nodes:
            d = math.hypot(n.state[0] - self.goal[0], n.state[1] - self.goal[1])
            if d <= self.config.goal_xy_tol:
                if n.t is None:
                    continue
                if n.t > t2 or n.t < t1:
                    continue
                p_val = Dynamics.P_tw(n.t, (t1, t2), self.config.p_tw_type)
                n.final_J = self.config.w_L * n.L + self.config.w_tw * p_val
                n.final_t = n.t
                n.final_L = n.L
                cands.append(n)
        return cands

    def get_best_path_node(self) -> Optional[Node]:
        cands = self.get_goal_candidates()
        if not cands:
            return None

        # Prefer exact-goal candidates if any exist (ensures path connects to the goal marker).
        eps = 1e-6
        exact = [n for n in cands if math.hypot(n.state[0] - self.goal[0], n.state[1] - self.goal[1]) <= eps]
        pool = exact if exact else cands

        best = None
        min_j = float('inf')
        for n in pool:
            val = n.final_J
            if val < min_j:
                min_j = val
                best = n
        return best

    def extract_path(self, node: Node) -> List[Any]:
        segments = []
        curr = node
        while curr.parent is not None:
            segments.append((curr.path_from_parent, curr.parent.state))
            curr = curr.parent
        ordered = segments[::-1]
        if hasattr(node, 'loiter_data') and node.loiter_data:
            ordered.extend(node.loiter_data)
        return ordered

# ============================================================
# TWA-RRT* PLANNER (Updated for WISER)
# ============================================================

class WiserRrtPlanner:
    def __init__(self, config: Config, rrt: EuclideanRRTStar):
        self.cfg = config
        self.rrt = rrt
        # self.obs = rrt.obs # Accessed via rrt

    def plan(self, t_window: Optional[Tuple[float, float]] = None):
        logging.info("Step 1: TWA-RRT* Search...")
        self.rrt.search(t_window=t_window)
        
        # Best Node
        best_node = self.rrt.get_best_path_node()
        is_partial = False
        
        if not best_node:
             logging.warning("No feasible solution found. Extracting best partial path.")
             is_partial = True
             best_partial = None
             min_partial_err = float('inf')
             tw = t_window if t_window is not None else (self.rrt.t_window if hasattr(self.rrt, 't_window') else self.cfg.t_window)
             t1, t2 = tw
             gx, gy = float(self.cfg.goal[0]), float(self.cfg.goal[1])

             for n in self.rrt.nodes:
                 d = math.hypot(n.state[0] - gx, n.state[1] - gy)
                 te = 0.0
                 n_t = n.t if n.t is not None else 0.0
                 if n_t < t1: te = t1 - n_t
                 elif n_t > t2: te = n_t - t2
                 tot = math.hypot(d, te)
                 if tot < min_partial_err:
                     min_partial_err = tot
                     best_partial = n
                     
             if best_partial:
                 pd = math.hypot(best_partial.state[0] - gx, best_partial.state[1] - gy)
                 if pd < 1e-3 and best_partial.parent:
                     approach = best_partial.parent
                     while approach and math.hypot(approach.state[0]-gx, approach.state[1]-gy) < 1e-3:
                         approach = approach.parent
                     if approach:
                         best_partial = approach
             
             best_node = best_partial
             if not best_node:
                 return None
        
        # Extract
        if hasattr(best_node, 'final_segments') and best_node.final_segments:
            final_segments = best_node.final_segments
        else:
            final_segments = self.rrt.extract_path(best_node)
        
        # Stats
        # L, t are stored in node or recomputed
        if hasattr(best_node, 'final_L'):
            L = best_node.final_L
            t_final = best_node.final_t
        else:
            L = best_node.L
            t_final = best_node.t
            
        v_final = L / t_final if t_final > 0 else 0
        
        # Mode
        # Mode
        mode = "WISER_DIRECT"
        loiter_len = 0.0
        if hasattr(best_node, 'loiter_data') and best_node.loiter_data:
            mode = "WISER_LOITER"
            # loiter_data is list of (path, state)
            loiter_len = sum(s[0].length for s in best_node.loiter_data)
        
        # --- Error terms (saved to CSV) ---
        tw = t_window if t_window is not None else (self.rrt.t_window if hasattr(self.rrt, 't_window') else self.cfg.t_window)
        t1, t2 = tw
        gx, gy = float(self.cfg.goal[0]), float(self.cfg.goal[1])
        end_x, end_y = gx, gy
        if final_segments:
            last_path = final_segments[-1][0]
            if hasattr(last_path, 'points') and last_path.points:
                end_x, end_y = float(last_path.points[-1][0]), float(last_path.points[-1][1])
        
        # Spatial Error (Approach Distance)
        # If best_node is at goal, walk up parent chain to find approach node.
        dist_to_goal = math.hypot(end_x - gx, end_y - gy)
        if dist_to_goal < 1e-3:
            # Walk up parent chain to find first ancestor NOT at goal
            approach = best_node.parent
            while approach and math.hypot(approach.state[0]-gx, approach.state[1]-gy) < 1e-3:
                approach = approach.parent
            if approach:
                spatial_err = math.hypot(approach.state[0]-gx, approach.state[1]-gy)
            else:
                spatial_err = 0.0
        else:
            spatial_err = dist_to_goal

        # Time Error (Deviation from Target Center) - PRESERVE ACTUAL
        t_target_actual = (t1 + t2) / 2.0
        time_err_actual = abs(t_final - t_target_actual)
        
        # Window Violation - ACTUAL
        time_window_violation_actual = 0.0
        if t_final < t1:
            time_window_violation_actual = t1 - t_final
        elif t_final > t2:
            time_window_violation_actual = t_final - t2

        p_tw_actual = Dynamics.P_tw(t_final, tw, self.cfg.p_tw_type)
        success_actual = (time_window_violation_actual == 0.0) and (dist_to_goal <= self.cfg.goal_xy_tol)

        # --- Option-A evaluation (Actual Planner Output) ---
        # arrival_t is the actual time from the planner (potentially loitered)
        arrival_t = t_final
        time_err = abs(arrival_t - self.cfg.t_ref)
        total_err = math.hypot(spatial_err, time_err)
        
        # Window Violation - ACTUAL
        time_window_violation = 0.0
        if arrival_t < t1:
            time_window_violation = t1 - arrival_t
        elif arrival_t > t2:
            time_window_violation = arrival_t - t2
            
        success_out = (time_window_violation == 0.0) and (dist_to_goal <= self.cfg.goal_xy_tol)

        if is_partial:
            success_out = False

        # Dynamics-based feasible range (Option-B logic)
        v_max_val, vmin, vmax, amax = self.cfg.v_max, self.cfg.v_min, self.cfg.v_max, self.cfg.a_max
        tmin, tmax = Dynamics.feasible_time_range(L, v_max_val, vmin, vmax, amax)

        return {
            "success": success_out, "mode": mode, 
            "t": arrival_t, "L": L, "v": v_final,
            "p_tw": p_tw_actual, "spatial_err": spatial_err, "time_err": time_err, 
            "total_err": total_err, "time_window_violation": time_window_violation,
            
            # Persisted actuals
            "t_actual": t_final,
            "time_err_actual": time_err_actual,
            "time_window_violation_actual": time_window_violation_actual, 
            "p_tw_actual": p_tw_actual,
            "success_actual": success_out,
            "tmin": tmin, "tmax": tmax,

            "segments": final_segments, "nodes": self.rrt.nodes, "stats": self.rrt.stats,
            "loiter_len": loiter_len
        }

# ============================================================
# EXPERIMENT RUNNER
# ============================================================


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

def calculate_smoothness(path_pts):
    if not path_pts or len(path_pts) < 3: return 0.0
    total_angle = 0.0
    for i in range(1, len(path_pts)-1):
        p0, p1, p2 = path_pts[i-1], path_pts[i], path_pts[i+1]
        dx1, dy1 = p1[0]-p0[0], p1[1]-p0[1]
        dx2, dy2 = p2[0]-p1[0], p2[1]-p0[1] # typo fix in p2[1]-p1[1]? Wait p2-p1. dx2 = p2-p1.
        dx2, dy2 = p2[0]-p1[0], p2[1]-p1[1]
        a1 = math.atan2(dy1, dx1)
        a2 = math.atan2(dy2, dx2)
        diff = abs(a2 - a1)
        if diff > math.pi: diff = 2*math.pi - diff
        total_angle += diff
    return total_angle

def plot_results(config: Config, rrt: EuclideanRRTStar, plan_result=None, run_id=0, save_dir=None):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches


    # Standardize Goal Node for plotting
    t_center = (config.t_window[0] + config.t_window[1]) / 2.0
    
    # Prepare Path Data
    path_points = [] # list of (x, y, t)
    path_success = False
    
    if plan_result and plan_result.get('segments'):
        path_success = plan_result.get('success', False)
        # Reconstruct path
        first_seg = plan_result['segments'][0]
        curr = first_seg[1] # start state
        
        t_total = plan_result.get('t', 0.0)
        L_total = plan_result.get('L', 0.0)
        v_avg = L_total / t_total if t_total > 0 else config.v0
        
        current_t = 0.0
        path_points.append((curr[0], curr[1], current_t))
        
        for d_data, s_state in plan_result['segments']:
            pts = EuclideanMath.sample_path(d_data, s_state, 0.3)
            # pts are (x,y,h)
            for p in pts:
                dist = math.hypot(p[0]-path_points[-1][0], p[1]-path_points[-1][1])
                dt = dist / v_avg
                current_t += dt
                path_points.append((p[0], p[1], current_t))

    # --- 1. 2D Plot ---
    fig, ax = plt.subplots(figsize=(6.6, 5.0), dpi=120)

    title = f"Trial {run_id} — TWA-RRT*"
    if not path_success: title += " (FAILED)"
    ax.set_title(title)

    ax.set_xlim(config.world_bounds[0], config.world_bounds[1])
    ax.set_ylim(config.world_bounds[2], config.world_bounds[3])
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linewidth=0.4, alpha=0.4)

    # Tree
    for node in rrt.nodes:
        if node.parent:
            p1 = node.parent.state
            p2 = node.state
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.4, alpha=0.20, color='#888888', zorder=1)

    # Obstacles
    for i, (ox, oy, r) in enumerate(config.obstacles):
        circ = patches.Circle((ox, oy), r, fill=False, linewidth=1.2, edgecolor='k')
        ax.add_patch(circ)
        ax.text(ox, oy, f"O{i:02d}", fontsize=7, ha="center", va="center", alpha=0.7)

    # Nodes
    xs = [n.state[0] for n in rrt.nodes]
    ys = [n.state[1] for n in rrt.nodes]
    ax.scatter(xs, ys, s=8, color='#4682b4', alpha=0.18, zorder=2)

    # Path
    if path_points:
        px = [p[0] for p in path_points]
        py = [p[1] for p in path_points]
        pt = [p[2] for p in path_points]
        
        color = '#0072B2' if path_success else '#FF6B6B'
        ax.plot(px, py, linewidth=2.0, label="Path", color=color, zorder=4)
        try:
           sc = ax.scatter(px, py, c=pt, s=15, cmap="viridis", alpha=0.85, zorder=5)
           cbar = plt.colorbar(sc, ax=ax, shrink=0.75)
           cbar.set_label("time (s)")
        except: pass

    # Start / Goal
    ax.scatter(config.start[0], config.start[1], marker='o', s=90, color='#0099FF', edgecolor='k', label='Start', zorder=10)
    ax.scatter(config.goal[0], config.goal[1], marker='*', s=180, color='#FF9900', edgecolor='k', label='Goal', zorder=11)

    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"trial_{run_id:03d}_2d.png"))
    plt.close(fig)

# ============================================================
# STANDARDIZATION CONSTANTS (Match TB/ST-RRT exactly + WISER extensions)
# ============================================================
REFERENCE_HEADER = [
    'case_id', 'trial', 'algo', 'success', 'runtime_s', 'path_len', 'arrival_t',
    'spatial_err', 'time_err', 'total_err', 'time_window_violation', 'nodes', 'seed', 
    'first_sol_time', 'first_path_len', 'first_total_err', 'smoothness',
    't_ref', 'L_proxy', 'Tmin', 'Tmax', 'alpha', 'beta', 'kappa',
    'p_tw', 'loiter_used', 'loiter_len', 'late_pruned', 'lb_pruned',
    'collision_reject', 'no_parent', 'goal_connect_fail', 'loiter_fail', 'goal_region_hits',
    # Extended instrumentation
    'accepted_nodes', 'acceptance_rate',
    'skip_point_collision', 'skip_edge_collision', 'skip_parent_none',
    'skip_late_prune', 'skip_lb_prune', 'skip_loiter_fail',
    'goal_connect_attempt', 'goal_connect_success', 'goal_connect_fail_collision',
    'goal_late_infeasible', 'goal_early_infeasible', 'goal_feasible_hits',
    'best_d_goal', 'best_d_goal_iter', 'best_d_goal_path_len', 'best_d_goal_tmin_to_goal',
    'best_d_goal_t_hat', 'best_t_hat', 'best_t_hat_iter',
    'iters_to_first_goal_hit', 'iters_to_first_goal_feasible', 'iters_to_first_goal_attempt',
    'best_J', 'best_J_iter', 'fail_reason',
    # Repair stats
    'repair_calls', 'repair_applied', 'repair_to_feasible',
    'repair_attempts_used', 'repair_delta_L', 'repair_vio_before', 'repair_vio_after',
    
    # Comparison Debug Fields
    't_actual', 'time_err_actual', 'time_window_violation_actual', 'p_tw_actual', 'success_actual',
    'tmin', 'tmax'
]
# Note: inserted 'case_id' at start

DIAG_HEADER = [
    'case_id', 'trial_id', 'seed', 'success', 'fail_reason',
    'iters_total', 'accepted_nodes', 'acceptance_rate', 'runtime_s',
    'skip_point_collision', 'skip_edge_collision', 'skip_parent_none',
    'skip_late_prune', 'skip_lb_prune', 'skip_loiter_fail',
    'goal_region_hits', 'goal_feasible_hits', 'goal_late_infeasible', 'goal_early_infeasible',
    'goal_connect_attempt', 'goal_connect_success', 'goal_connect_fail', 'goal_connect_fail_collision',
    'iters_to_first_goal_hit', 'iters_to_first_goal_feasible', 'iters_to_first_goal_attempt',
    'best_d_goal', 'best_d_goal_iter', 'best_d_goal_path_len', 'best_d_goal_tmin_to_goal',
    'best_d_goal_t_hat', 'best_t_hat', 'best_t_hat_iter',
    'best_J', 'best_J_iter',
    'repair_calls', 'repair_applied', 'repair_to_feasible',
    'repair_attempts_used', 'repair_delta_L', 'repair_vio_before', 'repair_vio_after',
    'path_len', 'arrival_t'
]

def to_reference_row(case_id, trial_idx, seed, algo_name, res, runtime_s, nodes_count, 
                     first_sol_time, first_path_len, first_total_err, smoothness):
    """Maps planner results to the reference CSV schema."""
    
    # Common default
    row = {
        'case_id': case_id,
        'trial': trial_idx,
        'algo': algo_name,
        'runtime_s': f"{runtime_s:.4f}",
        'nodes': nodes_count,
        'seed': seed,
        'first_sol_time': f"{first_sol_time:.4f}" if first_sol_time is not None else "",
        'first_path_len': f"{first_path_len:.3f}" if first_path_len is not None else "",
        'first_total_err': f"{first_total_err:.3f}" if first_total_err is not None else "",
        'smoothness': f"{smoothness:.3f}" if res else "",
    }

    if res:
        # Log basics even if success=False (DIAGNOSIS)
        l_len = res.get('loiter_len', 0.0)
        try:
            l_len = float(l_len)
        except (ValueError, TypeError):
            l_len = 0.0
            
        # Use computed success from plan()
        s_val = 1 if res.get('success', False) else 0
        
        row.update({
            'success': s_val,
            'path_len': f"{res['L']:.3f}",
            'arrival_t': f"{res['t']:.3f}",
            'spatial_err': f"{res.get('spatial_err', 0.0):.3f}",
            'time_err': f"{res.get('time_err', 0.0):.3f}",
            'total_err': f"{res.get('total_err', 0.0):.3f}",
            'time_window_violation': f"{res.get('time_window_violation', 0.0):.3f}",
            'late_pruned': res['stats'].get('late_pruned', 0),
            'lb_pruned': res['stats'].get('lb_pruned', 0),
            'loiter_used': res['stats'].get('loiter_used', 0),
            'loiter_len': f"{l_len:.3f}",
            'collision_reject': res['stats'].get('collision_reject', 0),
            'no_parent': res['stats'].get('no_parent', 0),
            'goal_connect_fail': res['stats'].get('goal_connect_fail', 0),
            'loiter_fail': res['stats'].get('loiter_fail', 0),
            'goal_region_hits': res['stats'].get('goal_region_hits', 0)
        })
    else:
        # No Solution Found (res is None)
        row.update({
            'success': 0,
            'path_len': "0.000",
            'arrival_t': "",
            'spatial_err': "",
            'time_err': "",
            'total_err': "",
            'time_window_violation': "",
            'collision_reject': 0,
            'no_parent': 0,
            'goal_connect_fail': 0,
            'loiter_fail': 0,
            'goal_region_hits': 0
        })
    return row

def run_case(case: Dict = None, out_root: str = None, save_plots: bool = True):
    # 1. Setup Configuration
    # (1) 먼저 기본값으로 "caseA"를 설정합니다.
    c_id = "caseA"
    cfg = CONFIG
    
    # (2) 만약 외부에서 'case' 딕셔너리(설정값 묶음)를 전달받았다면,
    #     케이스 ID를 업데이트하고, 해당 설정값으로 CONFIG를 덮어씁니다.
    if case:
        c_id = case.get("case_id", "unknown")
        # Override fields
        cfg = apply_case_to_config(case, CONFIG)
    
    # 2. Setup Output Directory
    # Priority: CLI > case['outputs']['out_dir'] > '../output' (default matching TB-RRT)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming script is in <root>/STL_RRT_Prop/, and we want <root>/output/
    default_root = os.path.join(script_dir, "output")
    
    case_out = case.get("outputs", {}).get("out_dir") if case else None
    
    final_out_root = default_root
    if out_root:
        final_out_root = out_root
    elif case_out:
        final_out_root = case_out
    elif cfg.save_dir_name and cfg.save_dir_name != "rrt_runs":
        # Handle legacy config override (if user changed default manually)
        final_out_root = cfg.save_dir_name
        
    # Ensure absolute path resolution if relative
    if not os.path.isabs(final_out_root):
        final_out_root = os.path.join(script_dir, final_out_root)
    
    # Structure: <out_root>/<case_id>/wiser_rrt/subfolder
    # subfolder: run_YYYYMMDD_HHMMSS
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(final_out_root, c_id, "wiser_rrt", f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    logging.info("="*70)
    logging.info(f"TWA-RRT* (R7) Experiments (Case: {c_id})")
    logging.info(f"Window: {cfg.t_window}, DeltaLate: {cfg.delta_late}, V_max: {cfg.v_max}")
    logging.info(f"Target Dir: {run_dir}")
    logging.info("="*70)

    csv_file = os.path.join(run_dir, "summary.csv")
    history_file = os.path.join(run_dir, "cost_history.csv")
    diag_file = os.path.join(run_dir, "trial_diagnostics.csv")
    
    with open(history_file, 'w', newline='') as hf:
        csv.writer(hf).writerow(['trial', 'time', 'cost'])
    
    with open(diag_file, 'w', newline='') as df:
        csv.DictWriter(df, fieldnames=DIAG_HEADER).writeheader()
        
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=REFERENCE_HEADER, extrasaction='ignore')
        writer.writeheader()
        
        num_trials = cfg.num_trials
        for i in range(num_trials):
            seed = cfg.seed + i
            # Important: Set seed explicitly
            random.seed(seed); np.random.seed(seed)
            
            # [FIXED] Dynamic Window Resolution - Pass actual case override
            algo_params = case.get("algorithms", {}).get("wiser_rrt", {}) if case else {}
            t_min, t_max, meta = resolve_time_window(algo_params, cfg)
            
            # [P0 Fix] Sync t_ref ONLY for dynamic window mode
            if meta.get('time_window_mode') != "fixed":
                cfg.t_ref = meta['t_ref']

            logging.info(f"--- Trial {i+1}/{num_trials} (Seed {seed}) ---")
            logging.info(f"Window: [{t_min:.2f}, {t_max:.2f}]s (Ref: {meta['t_ref']:.2f}s) | Mode: {meta['time_window_mode']}")
            
            obs_manager = ObstacleManager(cfg)
            rrt = EuclideanRRTStar(cfg, obs_manager)
            planner = WiserRrtPlanner(cfg, rrt)
            
            start_t = time.time()
            res = None
            try: 
                res = planner.plan(t_window=(t_min, t_max))
            except Exception as e:
                logging.error(f"Crash: {e}", exc_info=True); res = None
            runtime = time.time() - start_t
            
            # Metrics
            first_sol_time = None
            first_path_len = None
            first_total_err = None
            
            if hasattr(planner.rrt, 'first_sol_time'):
                first_sol_time = planner.rrt.first_sol_time
                first_path_len = planner.rrt.first_path_len
                first_total_err = planner.rrt.first_total_err
            
            smoothness = 0.0
            if res and 'segments' in res:
                 pts = []
                 for d, s in res['segments']:
                     pts.extend(EuclideanMath.sample_path(d, s, 1.0))
                 smoothness = calculate_smoothness(pts)
            
            # --- Unified Evaluation Logic (Pre-calculated in plan()) ---
            # Result 'res' already contains:
            # - success (Lenient)
            # - t (Best Retimed)
            # - time_window_violation (Lenient)
            # - p_tw (Lenient)
            # - *_actual (Strict/Original)
            
            # Construct Row
            row = to_reference_row(c_id, i+1, seed, "TWA-RRT* (R7)", res, runtime, len(planner.rrt.nodes),
                                   first_sol_time, first_path_len, first_total_err, smoothness)
            
            # Add WISER Meta
            row.update(meta)
            if res:
                 # Update P_tw with actual window used (using best_t now)
                 row['p_tw'] = f"{res.get('p_tw', 0.0):.4f}"
                 
                 # Add Actuals for Debugging/Comparison
                 row['t_actual'] = f"{res.get('t_actual', 0.0):.3f}"
                 row['time_err_actual'] = f"{res.get('time_err_actual', 0.0):.3f}"
                 row['time_window_violation_actual'] = f"{res.get('time_window_violation_actual', 0.0):.3f}"
                 row['p_tw_actual'] = f"{res.get('p_tw_actual', 0.0):.4f}"
                 row['success_actual'] = 1 if res.get('success_actual', False) else 0
                 row['tmin'] = f"{res.get('tmin', 0.0):.3f}"
                 row['tmax'] = f"{res.get('tmax', 0.0):.3f}"
            
            # --- Extended instrumentation (always from planner stats) ---
            st = planner.rrt.stats
            iters_total = st.get('iters_total', 0)
            accepted = st.get('accepted_nodes', 0)
            
            row['accepted_nodes'] = accepted
            row['acceptance_rate'] = f"{accepted / max(iters_total, 1):.4f}"
            row['skip_point_collision'] = st.get('skip_point_collision', 0)
            row['skip_edge_collision'] = st.get('skip_edge_collision', 0)
            row['skip_parent_none'] = st.get('no_parent', 0)
            row['skip_late_prune'] = st.get('late_pruned', 0)
            row['skip_lb_prune'] = st.get('lb_pruned', 0)
            row['skip_loiter_fail'] = st.get('loiter_fail', 0)
            row['goal_connect_attempt'] = st.get('goal_connect_attempt', 0)
            row['goal_connect_success'] = st.get('goal_connect_success', 0)
            row['goal_connect_fail'] = st.get('goal_connect_fail', 0)
            row['goal_connect_fail_collision'] = st.get('goal_connect_fail_collision', 0)
            row['goal_late_infeasible'] = st.get('goal_late_infeasible', 0)
            row['goal_early_infeasible'] = st.get('goal_early_infeasible', 0)
            row['goal_feasible_hits'] = st.get('goal_feasible_hits', 0)
            row['goal_region_hits'] = st.get('goal_region_hits', 0)
            row['collision_reject'] = st.get('collision_reject', 0)
            row['no_parent'] = st.get('no_parent', 0)
            row['late_pruned'] = st.get('late_pruned', 0)
            row['lb_pruned'] = st.get('lb_pruned', 0)
            row['loiter_fail'] = st.get('loiter_fail', 0)
            
            bd = st.get('best_d_goal', float('inf'))
            row['best_d_goal'] = f"{bd:.3f}" if bd < float('inf') else ""
            row['best_d_goal_iter'] = st.get('best_d_goal_iter', 0)
            row['best_d_goal_path_len'] = f"{st.get('best_d_goal_path_len', 0.0):.3f}"
            row['best_d_goal_tmin_to_goal'] = f"{st.get('best_d_goal_tmin_to_goal', 0.0):.3f}"
            bdt = st.get('best_d_goal_t_hat', float('inf'))
            row['best_d_goal_t_hat'] = f"{bdt:.3f}" if bdt < float('inf') else ""
            bt = st.get('best_t_hat', float('inf'))
            row['best_t_hat'] = f"{bt:.3f}" if bt < float('inf') else ""
            row['best_t_hat_iter'] = st.get('best_t_hat_iter', -1)
            row['iters_to_first_goal_hit'] = st.get('iters_to_first_goal_hit', -1)
            row['iters_to_first_goal_feasible'] = st.get('iters_to_first_goal_feasible', -1)
            row['iters_to_first_feasible'] = st.get('iters_to_first_feasible', -1)
            row['iters_to_first_goal_attempt'] = st.get('iters_to_first_goal_attempt', -1)
            
            row['best_J'] = f"{planner.rrt.best_J:.4f}" if planner.rrt.best_J < float('inf') else ""
            row['best_J_iter'] = st.get('iters_best', 0)
            
            # Repair stats
            row['repair_calls'] = st.get('repair_calls', 0)
            row['repair_applied'] = st.get('repair_applied', 0)
            row['repair_to_feasible'] = st.get('repair_to_feasible', 0)
            row['repair_attempts_used'] = st.get('repair_attempts_used', 0)
            rdl = st.get('repair_delta_L', 0.0)
            row['repair_delta_L'] = f"{rdl:.4f}" if rdl > 0 else "0"
            rvb = st.get('repair_vio_before', 0.0)
            row['repair_vio_before'] = f"{rvb:.4f}" if rvb > 0 else "0"
            rva = st.get('repair_vio_after', 0.0)
            row['repair_vio_after'] = f"{rva:.4f}" if rva > 0 else "0"
            
            # Fail reason classification
            is_success = res and res.get('success', False)
            if is_success:
                fail_reason = "SUCCESS"
            elif st.get('goal_region_hits', 0) == 0:
                fail_reason = "NO_GOAL_REACHED"
            elif st.get('goal_late_infeasible', 0) > 0:
                fail_reason = "GOAL_REACHED_BUT_TOO_LATE"
            elif st.get('goal_early_infeasible', 0) > 0:
                fail_reason = "GOAL_REACHED_BUT_TOO_EARLY"
            elif st.get('goal_connect_attempt', 0) > 0 and st.get('goal_connect_success', 0) == 0:
                fail_reason = "GOAL_CONNECT_FAIL_ONLY"
            else:
                fail_reason = "OTHER"
            row['fail_reason'] = fail_reason
            
            writer.writerow(row)
            f.flush()
            
            # Collision stats
            n_coll = st.get('n_collision', 0)
            
            diag_str = (f"CollRej: {st.get('collision_reject',0)} | "
                    f"NoParent: {st.get('no_parent',0)} | "
                    f"GoalHits: {st.get('goal_region_hits',0)} | "
                    f"GoalLate: {st.get('goal_late_infeasible',0)} | "
                    f"GoalFeas: {st.get('goal_feasible_hits',0)} | "
                    f"FirstGoalHit: {st.get('iters_to_first_goal_hit',-1)} | "
                    f"FirstFeas: {st.get('iters_to_first_feasible',-1)} | "
                    f"AccRate: {st.get('acceptance_rate',0.0):.3f} | "
                    f"Reason: {fail_reason}")
            if res:
                p_tw_val = row.get('p_tw', '')
                logging.info(f"Trial {i+1}: SUCCESS (t={res['t']:.3f}, P_tw={p_tw_val}) | Coll: {n_coll} | {diag_str}")
            else:
                logging.warning(f"Trial {i+1}: FAILED | Nodes: {len(planner.rrt.nodes)} | Coll: {n_coll} | {diag_str}")
            
            # --- Write trial_diagnostics.csv ---
            diag_row = {
                'case_id': c_id, 'trial_id': i+1, 'seed': seed,
                'success': 1 if is_success else 0, 'fail_reason': fail_reason,
                'iters_total': iters_total, 'accepted_nodes': accepted,
                'acceptance_rate': f"{st.get('acceptance_rate',0.0):.4f}",
                'runtime_s': f"{runtime:.4f}",
                'skip_point_collision': st.get('skip_point_collision', 0),
                'skip_edge_collision': st.get('skip_edge_collision', 0),
                'skip_parent_none': st.get('no_parent', 0),
                'skip_late_prune': st.get('late_pruned', 0),
                'skip_lb_prune': st.get('lb_pruned', 0),
                'skip_loiter_fail': st.get('loiter_fail', 0),
                'goal_region_hits': st.get('goal_region_hits', 0),
                'goal_feasible_hits': st.get('goal_feasible_hits', 0),
                'goal_late_infeasible': st.get('goal_late_infeasible', 0),
                'goal_early_infeasible': st.get('goal_early_infeasible', 0),
                'goal_connect_attempt': st.get('goal_connect_attempt', 0),
                'goal_connect_success': st.get('goal_connect_success', 0),
                'goal_connect_fail': st.get('goal_connect_fail', 0),
                'goal_connect_fail_collision': st.get('goal_connect_fail_collision', 0),
                'iters_to_first_goal_hit': st.get('iters_to_first_goal_hit', -1),
                'iters_to_first_goal_feasible': st.get('iters_to_first_goal_feasible', -1),
                'iters_to_first_goal_attempt': st.get('iters_to_first_goal_attempt', -1),
                'best_d_goal': f"{bd:.3f}" if bd < float('inf') else "",
                'best_d_goal_iter': st.get('best_d_goal_iter', 0),
                'best_d_goal_path_len': f"{st.get('best_d_goal_path_len', 0.0):.3f}",
                'best_d_goal_tmin_to_goal': f"{st.get('best_d_goal_tmin_to_goal', 0.0):.3f}",
                'best_d_goal_t_hat': f"{bdt:.3f}" if bdt < float('inf') else "",
                'best_t_hat': f"{bt:.3f}" if bt < float('inf') else "",
                'best_t_hat_iter': st.get('best_t_hat_iter', -1),
                'best_J': row['best_J'], 'best_J_iter': row['best_J_iter'],
                'repair_calls': st.get('repair_calls', 0),
                'repair_applied': st.get('repair_applied', 0),
                'repair_to_feasible': st.get('repair_to_feasible', 0),
                'repair_attempts_used': st.get('repair_attempts_used', 0),
                'repair_delta_L': row['repair_delta_L'],
                'repair_vio_before': row['repair_vio_before'],
                'repair_vio_after': row['repair_vio_after'],
                'path_len': row.get('path_len', ''), 'arrival_t': row.get('arrival_t', '')
            }
            with open(diag_file, 'a', newline='') as df:
                csv.DictWriter(df, fieldnames=DIAG_HEADER).writerow(diag_row)
            
            # --- Save progress trace CSV ---
            if cfg.dump_progress_trace:
                should_dump = True
                if cfg.progress_trace_fail_only:
                    should_dump = (not res) or (not res.get('success', False))
                if should_dump and hasattr(planner.rrt, 'progress_trace') and planner.rrt.progress_trace:
                    trace_dir = os.path.join(run_dir, cfg.trace_dir_name)
                    os.makedirs(trace_dir, exist_ok=True)
                    # Use shorter filename to avoid Windows MAX_PATH (260 char) limits
                    trace_file = os.path.join(trace_dir, f"prog_t{i+1}.csv")
                    trace_fields = ['iter', 'nodes', 'event', 'd_goal', 'L', 'tmin', 'tmax', 'vio', 't_hat', 'best_J']
                    with open(trace_file, 'w', newline='') as tf:
                        tw = csv.DictWriter(tf, fieldnames=trace_fields)
                        tw.writeheader()
                        tw.writerows(planner.rrt.progress_trace)

            # Save History
            with open(history_file, 'a', newline='') as hf:
                h_writer = csv.writer(hf)
                for (ht, hc) in planner.rrt.cost_history:
                    h_writer.writerow([i+1, f"{ht:.4f}", f"{hc:.4f}"])
            
            # Plot
            if save_plots:
                plot_results(cfg, planner.rrt, res, i+1, run_dir)
            
    logging.info(f"Experiments completed for Case {c_id}. Results saved to: {run_dir}")

def main():
    parser = argparse.ArgumentParser(description="Run TWA-RRT* Experiments")
    parser.add_argument("--cases", type=str, help="Path to benchmark_cases.json file", default="benchmarks/benchmark_cases_2.json")
    parser.add_argument("--case_id", type=str, help="Specific case ID to run (e.g., caseA)")
    parser.add_argument("--out_dir", type=str, help="Output root directory override")
    parser.add_argument("--no_plots", action="store_true", help="Disable plotting")
    
    args = parser.parse_args()
    
    # 1. Load Cases from JSON
    try:
        if os.path.exists(args.cases):
             cases = load_case_file(args.cases)
        else:
             # Try parent dir
             parent_path = os.path.join(os.path.dirname(__file__), "..", args.cases)
             if os.path.exists(parent_path):
                 cases = load_case_file(parent_path)
             else:
                 logging.error(f"Cases file not found: {args.cases}")
                 sys.exit(1)
    except Exception as e:
        logging.error(f"Failed to load cases: {e}")
        sys.exit(1)
        
    # 2. Filter by case_id if provided
    if args.case_id:
        filtered = [c for c in cases if c.get("case_id") == args.case_id]
        if not filtered:
            logging.warning(f"Case ID '{args.case_id}' not found in {args.cases}")
            sys.exit(0)
        cases_to_run = filtered
    else:
        # Only run if algorithm is present in case config? 
        # Standard behavior: Run all matching
        cases_to_run = cases
        
    logging.info(f"Loaded {len(cases)} cases. Running {len(cases_to_run)} cases matching filter...")
    
    # 3. Run Experiments
    for case in cases_to_run:
        # Check if wiser_rrt is in algorithms list
        if "algorithms" in case:
             if "wiser_rrt" in case["algorithms"] or "manual" in case["algorithms"]: # generic check
                 run_case(case, args.out_dir, not args.no_plots)
        elif "algo" in case: 
             # Single case legacy format?
             run_case(case, args.out_dir, not args.no_plots)

if __name__ == "__main__":
    main()


