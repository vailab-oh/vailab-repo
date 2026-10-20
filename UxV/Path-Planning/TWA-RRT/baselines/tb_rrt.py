# TB-RRT* benchmark implementation (paper label: TB-RRT)
# Sintov, A., and Shapiro, A. (2014). Time-based RRT Algorithm for Rendezvous Planning of Two Dynamic Systems. ICRA, 6745-6750.
# https://doi.org/10.1109/ICRA.2014.6907855
# Original-author code reuse and independent reimplementation status: unverified.
# Benchmark changes: star-style parent choice/rewiring; static 2D circles; fixed goal/window; common JSON and metrics. The cited paper is TB-RRT, not a verified TB-RRT* reference.
# Redistribution license: unresolved; root MIT scope excludes baselines.
# Publication preparation changes only names, references and file/output paths.

# Created: 2025-06-12
# Modified: 2026-01-31 (TB-RRT* 논문 8가지 보완 반영)
## TB-RRT* (Time-Biased RRT*) 알고리즘 완전 개선 버전
## 개선사항:
##   1) 파라미터 중복 제거 → NEIGHBOR_RADIUS_BASE 단일 정의
##   2) 동적 이웃 반경 함수 추가 → r_n = r_0 * sqrt(log(n)/n)
##   3) neighbors_gamma 함수 추가 → dist_gamma 기반 이웃 검색 (XY만 사용 대신)
##   4) is_ancestor 함수 추가 → rewire 순환참조 회피
##   5) path_length 수정 → euclidean → dist_gamma (일관성)
##   6) parent selection에 동적 반경 적용
##   7) rewiring에 dist_gamma + 순환참조 회피 적용
# Updated : 2026-02-18 (Option B 평가 기준 적용. 통일된 동역학 및 T_WINDOW)
# Refactored: 2026-02-18 (Support External JSON Experiment Cases)
##   8) [New] Option B 평가 기준 적용 (통일된 동역학 프로파일, XY 경로 길이 기반 시간 타당성 검사)
##   9) [Refactor] JSON 기반 실험 케이스 로딩 지원 (benchmark_cases.json)
#
# [JSON Schema Example]
# {
#   "meta": {"name": "benchmark_set_v1"},
#   "cases": [
#     {
#       "case_id": "caseA",
#       "world": {"xlim": [0, 220], "ylim": [0, 160]},
#       "start": {"x": 30.0, "y": 30.0},
#       "goal":  {"x": 180.0, "y": 120.0},
#       "obstacles": [
#         {"type": "circle", "cx": 60.0,  "cy": 60.0,  "r": 13.0},
#         ...
#       ],
#       "time": {"t_ref": 7.5, "t_window": [7.0, 8.0]},
#       "dynamics": {"v_min": 15.0, "v_max": 25.0, "a_max": 3.0, "v0": 20.0},
#       "random": {"seed_base": 42, "num_trials": 5},
#       "planner": {"n_iters": 6000, "goal_sample_rate": 0.15, "step_size": 3.0, "neighbor_radius": 35.0},
#       "outputs": {"out_dir": "outputs"},
#       "algorithms": { "tb_rrt": { "eps_state": 5.0, "dt_min": 0.1 } }
#     }
#   ]
# }

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

# ============ 파라미터 정의 (중복 제거) ============
SEED = 42
WORLD = (0, 220, 0, 160)                 # xmin, xmax, ymin, ymax
OBSTACLES = [(60,60,13),(120,80,14),(160,110,12)]  # ST-RRT와 일치
START = (30.0, 30.0, 0.0)
GOAL  = (180.0, 120.0, 7.5)
TRIALS = 5

# --- Platform Profile (Option B: 통일된 동역학) ---
V_MIN = 15.0
V_MAX = 25.0
A_MAX = 3.0
V0    = 20.0

# --- Time Constraints ---
T_REF = 7.5                              # 목표 도착 시간 (Reference Time)
T_WINDOW = (7.0, 8.0)                    # 목표 시간 윈도우 (Tolerance: ±0.5s)
                                         # 참고: 알고리즘 내부 종료 조건(eps_time=0.5)과 평가 기준(T_WINDOW)을 일치시킴
                                         # Option B 성공조건: [Tmin, Tmax] ∩ [7.0, 8.0] ≠ ∅

GAMMA = 1.0                              # 시간축 가중 (ST-RRT와 동일)
NEIGHBOR_RADIUS_BASE = 35.0              # 기본 이웃 반경 (동적 r_n 계산에 사용)

random.seed(SEED)
np.random.seed(SEED)

# Base seed for per-trial seeding (legacy name used in run_experiments)
BASE_SEED = SEED

# Goal bias (fraction of samples that are the goal). Can be overridden
# by setting the environment variable `GOAL_BIAS` before running.
GOAL_BIAS = float(os.environ.get('GOAL_BIAS', '0.15'))

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============ JSON Case Loader ============
def normalize_case(case: dict) -> dict:
    """
    Ensure the case dict has all necessary fields with defaults if missing.
    Also validates structure.
    """
    # 1. World
    if "world" not in case:
        case["world"] = {"xlim": [WORLD[0], WORLD[1]], "ylim": [WORLD[2], WORLD[3]]}
    
    # 2. Start/Goal
    if "start" not in case:
        case["start"] = {"x": START[0], "y": START[1]}
    if "goal" not in case:
        case["goal"] = {"x": GOAL[0], "y": GOAL[1]}
        
    # 3. Obstacles
    if "obstacles" not in case:
        # Convert legacy OBSTACLES list (x,y,r) to dict list
        obs_list = []
        for i, (ox, oy, r) in enumerate(OBSTACLES):
            obs_list.append({"type": "circle", "cx": ox, "cy": oy, "r": r, "id": f"O{i}"})
        case["obstacles"] = obs_list
    
    # Validate Obstacles
    for ob in case["obstacles"]:
        if ob.get("type") != "circle":
            raise ValueError(f"Unsupported obstacle type: {ob.get('type')}")
            
    # 4. Time
    if "time" not in case:
        case["time"] = {"t_ref": T_REF, "t_window": list(T_WINDOW)}
        
    # 5. Dynamics
    if "dynamics" not in case:
        case["dynamics"] = {"v_min": V_MIN, "v_max": V_MAX, "a_max": A_MAX, "v0": V0}
        
    # 6. Random
    if "random" not in case:
        case["random"] = {"seed_base": SEED, "num_trials": TRIALS}
        
    # 7. Planner (Common)
    if "planner" not in case:
        case["planner"] = {
            "n_iters": 6000, 
            "goal_sample_rate": GOAL_BIAS,
            "step_size": 3.0,
            "neighbor_radius": NEIGHBOR_RADIUS_BASE
        }
        
    # 8. Algorithms (TB-RRT specific)
    if "algorithms" not in case:
        case["algorithms"] = {}
    if "tb_rrt" not in case["algorithms"]:
        case["algorithms"]["tb_rrt"] = {
            "eps_state": 5.0,
            "eps_time": 0.5,
            "dt_min": 0.1 # delta_t
        }
    
    # 9. Outputs
    if "outputs" not in case:
        case["outputs"] = {"out_dir": "output"}
        
    return case

def load_case_file(path: str) -> list:
    """Load and normalize cases from a JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if "cases" not in data:
        raise ValueError("JSON file must contain a 'cases' list.")
        
    normalized_cases = []
    for c in data["cases"]:
        # Ensure case_id exists
        if "case_id" not in c:
            raise ValueError("Each case must have a 'case_id'.")
        normalized_cases.append(normalize_case(c))
        
    return normalized_cases

# ============ Dynamics Helper (Option B) ============
class Dynamics:
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
            if tmax < t1: return tmax # Too Early -> Slowest is tmax
            if tmin > t2: return tmin # Too Late -> Fastest is tmin
        return t_ref # Should not reach here

def strip_visual_goal_tail(path, goal_x, goal_y, goal_t, start_x, start_y, tol=1.0):
    """
    Remove a goal node appended ONLY for visualization (parent is None) from evaluation.
    Plotting can still use the original path.
    Updated: Use larger tol for XY to handle eps_state, and allow any T if it was a visual-only addition.
    In paper2014, the added goal node has parent=None (it's not in tree link).
    """
    if not path or len(path) < 2:
        return path
    last = path[-1]
    # In my paper2014 implementation, I added Node(goal_x, goal_y, q_goal.t) to the path list.
    # This node is NOT in the parent-child structure of the tree (it's a leaf without being 'set_parent').
    # However, wait, reconstruct_path returns nodes that have parents. 
    # Let's check tb_rrt_paper2014: path.append(Node(goal_x, goal_y, q_goal.t))
    # This new node has parent=None by default.
    
    is_at_goal_xy = (math.hypot(last.x - goal_x, last.y - goal_y) <= 1e-3)
    is_visual_only = (last.parent is None)
    is_not_start = (math.hypot(last.x - start_x, last.y - start_y) > 1e-3)
    
    if is_at_goal_xy and is_visual_only and is_not_start:
        return path[:-1]
    return path

# ============ Node 클래스 ============
class Node:
    """TB-RRT* 노드 (비용, 부모 포함)"""
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

def create_new_node(nearest: Node, random_node: Node, step_size: float, delta_t: float) -> Node:
    dx, dy = random_node.x - nearest.x, random_node.y - nearest.y
    dist_xy = math.hypot(dx, dy)
    
    # 충돌 방지용 최소 이동 거리 (선택사항, 0이면 이동 안함)
    if dist_xy < 1e-6:
        return None

    # 1. 공간 Steering: step_size 통제
    ratio = min(step_size, dist_xy) / dist_xy
    new_x = nearest.x + dx * ratio
    new_y = nearest.y + dy * ratio
    
    # 실제 이동한 거리
    actual_dist = min(step_size, dist_xy)

    # 2. 시간 Steering: 물리적 제약 (V_MAX) 및 delta_t 통제
    # 최소 필요 시간 (물리적 제약)
    min_dt_required = actual_dist / V_MAX
    max_dt_allowed = actual_dist / V_MIN
    
    dt_sample = random_node.t - nearest.t
    # dt_sample이 길더라도 max_dt_allowed(v_min 속도일 때의 시간)를 넘을 수 없도록 클램핑
    dt_target = max(min_dt_required, min(max_dt_allowed, dt_sample))
    
    # 최대 시간 스텝 제한
    final_dt = min(dt_target, delta_t)
    
    if final_dt < min_dt_required:
         final_dt = min_dt_required

    new_t = nearest.t + final_dt

    return Node(new_x, new_y, new_t, parent=nearest)


# ============ Paper 2014 (Algorithm 1) Helpers ============

def build_goal_set(goal_x, goal_y, t_window, delta_t):
    """
    논문 식: X_goal = { (goal_x, goal_y, t) | t in [t_lo, t_hi] }
    Δt_goal = delta_t 간격으로 샘플링.
    """
    t_lo, t_hi = t_window
    # arange behaves better with floating point eps
    times = np.arange(t_lo, t_hi + 1e-9, delta_t)
    return [Node(goal_x, goal_y, t) for t in times]

def select_best_control_step(nearest: Node, rand: Node, delta_t, step_size, v_min, v_max, obstacles, x_max, y_max, t_max):
    """
    논문(2014) Algorithm 1 핵심: 후보 제어 입력 u_i 샘플링 및 평가.
    """
    V_CAND = [v_min, 0.5 * (v_min + v_max), v_max]
    
    base_heading = math.atan2(rand.y - nearest.y, rand.x - nearest.x)
    OFFSETS = [0.0, -math.pi/12, math.pi/12, -math.pi/6, math.pi/6] 
    
    best_node = None
    min_score = float('inf')
    
    for v in V_CAND:
        for offset in OFFSETS:
            heading = base_heading + offset
            move_dist = min(step_size, v * delta_t)
            
            new_x = nearest.x + move_dist * math.cos(heading)
            new_y = nearest.y + move_dist * math.sin(heading)
            new_t = nearest.t + delta_t 
            
            if not (0 <= new_x <= x_max and 0 <= new_y <= y_max and 0 <= new_t <= t_max):
                continue
            
            # Temporary node for collision check
            temp_node = Node(new_x, new_y, new_t, parent=nearest)
            if not collision_free_xy(nearest, temp_node, obstacles):
                continue
                
            dx, dy, dt = new_x - rand.x, new_y - rand.y, new_t - rand.t
            # 논문: rho = sqrt(dx^2 + dy^2 + (v_max * dt)^2)
            score = math.sqrt(dx*dx + dy*dy + (v_max * dt)**2)
            
            if score < min_score:
                min_score, best_node = score, temp_node
                
    return best_node


def rho_star_reached(x_new, goal_set, eps_state, eps_time):
    """
    논문 식 (15): hyper-rectangle (tolerance vector e1) 기반 판정.
    단, 벤치마크 평가와 일치하도록 공간 오차는 Euclidean distance(L2)를 사용합니다.
    """
    # X_goal 내에서 x_new와 가장 가까운 샘플 찾기 (t 축 위주)
    # 2D 좌표는 모두 같으므로 시간 차이가 가장 적은 노드가 NN
    q_near = min(goal_set, key=lambda g: abs(g.t - x_new.t))
    
    dt = abs(x_new.t - q_near.t)
    spatial_err = math.hypot(x_new.x - q_near.x, x_new.y - q_near.y)
    time_err = dt
    
    # [Fix] eps_time 허용 오차가 window 밖(evaluator failure)으로 나가는 것을 방지하려면
    # x_new.t가 goal_set의 시간 범위 [t_min, t_max] 내에 있어야 함.
    t_min_set = goal_set[0].t
    t_max_set = goal_set[-1].t
    within_window = (t_min_set - 1e-6 <= x_new.t <= t_max_set + 1e-6)
    
    reached = (spatial_err <= eps_state and dt <= eps_time and within_window)
    return reached, q_near, spatial_err, time_err

def tb_rrt_paper2014(start_node: Node, goal_x, goal_y, t_window,
                     x_max, y_max, t_max,
                     delta_t, step_size, eps_state, eps_time,
                     obstacles, v_min, v_max,
                     max_iterations=6000, max_nodes=10000):
    """
    2014 Paper Algorithm 1: Greedy TB-RRT (No Globals)
    """
    start_time_algo = time.time()
    goal_set = build_goal_set(goal_x, goal_y, t_window, delta_t)
    tree = [Node(start_node.x, start_node.y, start_node.t, parent=None, cost=0.0)]
    
    best_solution = None
    best_total_err = float('inf')
    first_sol_metrics = None # (sol_time, path_len, s_err, t_err, total_err)
    
    best_partial_node = None
    best_partial_dist = float('inf')
    p_spatial_err = 0.0
    p_time_err = 0.0
    
    for i in range(max_iterations):
        if random.random() < GOAL_BIAS:
            g_sample = random.choice(goal_set)
            rand = Node(g_sample.x, g_sample.y, g_sample.t)
        else:
            rand = sample_random_state_time(x_max, y_max, t_max)
            
        # Step 4: Nearest neighbor (rho space metric)
        def paper_metric(n, r, vmax_val):
            dt = r.t - n.t
            if dt < -1e-6: return float('inf') # Causality
            dx, dy = r.x - n.x, r.y - n.y
            dist_sq = dx*dx + dy*dy
            # Reachability: dist / dt <= vmax
            if dist_sq > (vmax_val * dt + 1e-6)**2:
                return float('inf')
            return math.sqrt(dist_sq + (vmax_val * dt)**2)

        nearest = None
        min_dist = float('inf')
        for n in tree:
            d = paper_metric(n, rand, v_max)
            if d < min_dist:
                min_dist, nearest = d, n
        
        if nearest is None:
            # Fallback: time-weighted L1/L2 distance
            nearest = min(tree, key=lambda n: math.hypot(n.x-rand.x, n.y-rand.y) + v_max*abs(n.t-rand.t))
        
        # Step 5-6: Propagate (Best Candidate Selection)
        new_node = select_best_control_step(nearest, rand, delta_t, step_size, 
                                            v_min, v_max, obstacles, x_max, y_max, t_max)
        
        if new_node is None:
            continue
            
        new_node.cost = nearest.cost + math.hypot(new_node.x - nearest.x, new_node.y - nearest.y)
        tree.append(new_node)
        
        # Step 10: Goal check
        reached, q_goal, s_err, t_err = rho_star_reached(new_node, goal_set, eps_state, eps_time)
        if reached:
            path = reconstruct_path(new_node)
            path.append(Node(goal_x, goal_y, q_goal.t))
            
            sol_time = time.time() - start_time_algo
            total_err = math.sqrt(s_err**2 + t_err**2)
            path_len = path_length_xy(path[:-1])
            
            if not first_sol_metrics:
                logging.info(f"Paper 2014 TB-RRT (Anytime): First solution found at iter {i}")
                first_sol_metrics = (sol_time, path_len, s_err, t_err, total_err)
                
            if total_err < best_total_err:
                best_total_err = total_err
                best_solution = (path, s_err, t_err, total_err, sol_time, path_len)

        # Partial update
        tot = math.sqrt(s_err**2 + (v_max*t_err)**2)
        if tot < best_partial_dist:
            best_partial_dist, best_partial_node = tot, new_node
            p_spatial_err, p_time_err = s_err, t_err

        if len(tree) >= max_nodes:
            break

    if best_solution:
        path, s_err, t_err, total_err, sol_time, path_len = best_solution
        fs_time, f_plen, fs_err, ft_err, f_tot_err = first_sol_metrics
        return path, tree, s_err, t_err, total_err, sol_time, path_len, fs_err, ft_err, f_tot_err, []

    if best_partial_node:
        path = reconstruct_path(best_partial_node)
        return path, tree, p_spatial_err, p_time_err, math.sqrt(p_spatial_err**2 + p_time_err**2), None, None, None, None, None, []
    
    return None, tree, None, None, None, None, None, None, None, None, []


def is_kinematically_feasible(from_node: Node, to_xyzt) -> bool:
    """판별: 주어진 상태 전이가 물리적 제약(V_MAX) 및 인과율(시간 흐름)을 만족하는지 검사"""
    if isinstance(to_xyzt, Node):
        bx, by, bt = to_xyzt.x, to_xyzt.y, to_xyzt.t
    else:
        bx, by, bt = to_xyzt
        
    dt = bt - from_node.t
    if dt < 0:
        return False  # 과거로 이동 불가

    dist = math.hypot(bx - from_node.x, by - from_node.y)
    if dist > 1e-9 and dt < (dist / V_MAX) - 1e-6:
        return False  # 최고 속도 초과 불가
        
    return True

def propagate_time_shift_to_leaves(start_node: Node, time_shift: float):
    """부모 노드의 시간이 시간만큼 지연/앞당겨졌을 때, 자식 노드들에게 동일한 변위를 전파 (배열 기반 스택/반복문 사용)
       단순히 dt = dist/V_MAX 로 덮어씌우면 각 자식의 고유한 이동 속도 프로파일이 훼손되므로,
       부모의 시간 변화량(time_shift) 단위로 전체 서브트리의 시간을 밀거나 당깁니다.
    """
    stack = [(start_node, time_shift)]
    while stack:
        node, shift = stack.pop()
        for child in list(node.children):
            # 자식의 도착 시간 역시 shift 만큼 이동
            new_child_t = child.t + shift
            
            # 만약 시간을 당기다(shift < 0) 절대시간이 음수가 되는 경우 예외처리
            if new_child_t < 0: 
                new_child_t = 0.0
                
            child.t = new_child_t
            
            # 비용 재계산 (TB-RRT의 dist_gamma는 시간에 의존함)
            child.cost = node.cost + dist_gamma(node, child)
            
            stack.append((child, shift))

def is_within_bounds(node: Node, x_max: float, y_max: float, t_max: float) -> bool:
    return 0 <= node.x <= x_max and 0 <= node.y <= y_max and 0 <= node.t <= t_max

def reconstruct_path(end_node: Node) -> list:
    path, node = [], end_node
    while node:
        path.append(node)
        node = node.parent
    return list(reversed(path))

def path_length(path: list) -> float:
    """경로의 총 길이 (dist_gamma 기반 — 개선사항 5)"""
    length = 0.0
    for i in range(1, len(path)):
        length += dist_gamma(path[i-1], path[i])
    return length

def path_length_xy(path: list) -> float:
    """순수 XY 유클리드 길이 (Option B 평가용)"""
    length = 0.0
    for i in range(1, len(path)):
        dx = path[i].x - path[i-1].x
        dy = path[i].y - path[i-1].y
        length += math.hypot(dx, dy)
    return length

# ============ 거리 함수 ============
def dist_gamma(a: Node, b: Node) -> float:
    """시간 가중 거리: √(dx² + dy² + (γ*dt)²) — TB-RRT* 논문"""
    dx = a.x - b.x
    dy = a.y - b.y
    dt = a.t - b.t
    return math.sqrt(dx*dx + dy*dy + (GAMMA*dt)*(GAMMA*dt))

def step_cost(parent: Node, child_xyzt) -> float:
    """부모에서 자식까지의 단계 비용 (dist_gamma 기반)"""
    bx, by, bt = child_xyzt
    pseudo = Node(bx, by, bt)
    return dist_gamma(parent, pseudo)

# ============ 동적 이웃 반경 (RRT* 이론, 개선사항 2) ============
def compute_neighbor_radius(n: int, base_radius: float = NEIGHBOR_RADIUS_BASE) -> float:
    """
    RRT*의 이론적 반경 공식: r_n ∝ (log(n)/n)^(1/d)
    3차원 공간(x,y,t): r_n = r_0 * sqrt(log(n) / n)
    """
    if n < 2:
        return base_radius
    return base_radius * math.sqrt(math.log(n) / n)

# ============ 이웃 검색 (개선사항 3) ============
def neighbors_gamma(nodes: list, pt_xyzt, radius: float) -> list:
    """
    TB-RRT* 논문 준수: dist_gamma 기반 이웃 검색
    이전 코드의 neighbors_xy (XY만 고려)를 시간 가중거리로 개선
    """
    px, py, pt = pt_xyzt
    pseudo = Node(px, py, pt)
    out = []
    for n in nodes:
        d = dist_gamma(pseudo, n)
        if d <= radius:
            out.append(n)
    return out


def sample_random_state_time(x_max: float, y_max: float, t_max: float) -> Node:
    """상태-시간 공간에서 무작위 샘플 생성"""
    return Node(random.uniform(0, x_max), random.uniform(0, y_max), random.uniform(0, t_max))


def get_nearest_node(nodes: list, target: Node) -> Node:
    """dist_gamma 기반 최근접 노드 검색 (시간 가중 거리 사용)"""
    return min(nodes, key=lambda node: dist_gamma(node, target))

# ============ 순환참조 회피 (개선사항 4) ============
def is_ancestor(node: Node, potential_child: Node) -> bool:
    """node가 potential_child의 조상인지 판정 (rewire 시 순환 참조 방지)"""
    current = potential_child
    while current:
        if current is node:
            return True
        current = current.parent
    return False

def try_connect_to_goal(from_node: Node, goal_node: Node,
                        x_max: float, y_max: float, t_max: float,
                        step_size: float, delta_t: float,
                        eps_state: float, eps_time: float, 
                        obstacles: list, max_steps: int = 10) -> Node:
    """
    목표 도달 가능성 판정 (공간/시간적 타당성 모두 검사)
    성공 시 해당 궤적의 마지막 노드(goal에 도달한 노드) 반환, 실패 시 None
    """
    curr = from_node
    for _ in range(max_steps):
        # attempt a steering step toward the goal
        candidate = create_new_node(curr, goal_node, step_size, delta_t)
        if candidate is None:
            return None
        if not is_within_bounds(candidate, x_max, y_max, t_max):
            return None
        if not collision_free_xy(curr, candidate, obstacles):
            return None
        if not is_kinematically_feasible(curr, candidate):
             return None

        # check goal criterion (공간 수렴 확인)
        spatial_err = math.hypot(candidate.x - goal_node.x, candidate.y - goal_node.y)
        
        # 임시로 진행 (tree에 추가하지 않음, 호출부에서 판단)
        curr = candidate
        
        if spatial_err <= eps_state:  
             # 목표 도달 (공간적으로)
             return curr

    # max_steps만큼 진행했는데도 goal 미도달
    return None

def neighbors_xy(nodes: list, pt_xy, radius=NEIGHBOR_RADIUS_BASE):
    """XY 평면만 고려한 이웃 검색 (backward compatibility)"""
    px, py = pt_xy
    r2 = radius * radius
    out = []
    for n in nodes:
        if (n.x - px)**2 + (n.y - py)**2 <= r2:
            out.append(n)
    return out

def line_circle_collision(p1, p2, center, radius):
    # 선분 p1-p2와 원(center,radius)의 최근접점 거리 판정
    (x1,y1), (x2,y2) = p1, p2
    cx, cy = center
    dx, dy = x2-x1, y2-y1
    denom = dx*dx + dy*dy
    t = 0.0 if denom < 1e-12 else max(0.0, min(1.0, ((cx-x1)*dx + (cy-y1)*dy)/denom))
    px, py = x1 + t*dx, y1 + t*dy
    return math.hypot(px-cx, py-cy) <= radius

def collision_free_xy(a: Node, b: Node, obstacles=None) -> bool:
    # 월드 경계 + 장애물 체크 (TB-RRT에도 공정비교용으로 추가)
    if not (WORLD[0] <= b.x <= WORLD[1] and WORLD[2] <= b.y <= WORLD[3]):
        return False
    
    if obstacles:
        for obs in obstacles:
            # Check dict format or tuple format
            if isinstance(obs, dict):
                 ox, oy, r = obs['cx'], obs['cy'], obs['r']
            else:
                 ox, oy, r = obs
            
            if line_circle_collision((a.x,a.y), (b.x,b.y), (ox,oy), r):
                return False
    elif 'OBSTACLES' in globals() and OBSTACLES:
         # Fallback to global
         for (ox,oy,r) in OBSTACLES:
            if line_circle_collision((a.x,a.y), (b.x,b.y), (ox,oy), r):
                return False
                
    return True

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
        if diff > math.pi:
            diff = 2*math.pi - diff
        total_angle += diff
    return total_angle

def tb_rrt_fixed_goal(start: Node, goal: Node, 
                      x_max, y_max, t_max,
                      delta_t, step_size, eps_state, eps_time,
                      obstacles,
                      max_iterations=5000, max_nodes=10000,
                      t_window=None, v_min=15.0, v_max=25.0):
    """
    TB-RRT Wrapper (Refactored)
    - t_window가 주어지면 2014 Paper Algorithm 1 (Greedy) 수행
    - t_window가 None이면 기존 TB-RRT* (최적화) 수행
    """
    if t_window is not None:
        return tb_rrt_paper2014(start, goal.x, goal.y, t_window,
                                x_max, y_max, t_max,
                                delta_t, step_size, eps_state, eps_time,
                                obstacles, v_min, v_max,
                                max_iterations, max_nodes)

    """TB-RRT* (Time-Biased RRT*) 알고리즘
    - dist_gamma 기반 이웃 검색 (TB-RRT* 논문 준수)
    - 동적 반경 r_n = r_0 * sqrt(log(n)/n)
    - 순환참조 회피
    - 목표 도달 후 계속 최적화
    """
    start_time_algo = time.time()
    
    # 트리 초기화
    tree = [Node(start.x, start.y, start.t, parent=None, cost=0.0)]
    
    best_goal_node = None
    best_goal_cost = float('inf')
    best_spatial_err = None
    best_time_err = None
    best_total_err = None
    first_sol_time = None
    first_path_len = None
    first_spatial_err = None
    first_time_err = None
    first_total_err = None
    
    # History: list of (time, best_cost)
    cost_history = []

    for i in range(max_iterations):
        # 목표 바이어스 샘플 (configurable via GOAL_BIAS env var)
        if random.random() < GOAL_BIAS:
            rand = Node(goal.x, goal.y, goal.t)
        else:
            rand = sample_random_state_time(x_max, y_max, t_max)

        # 최근접
        nearest = get_nearest_node(tree, rand)

        # steer (XY는 step_size, T는 delta_t 제한)
        new_node = create_new_node(nearest, rand, step_size, delta_t)
        if new_node is None:
            continue

        # 경계/시간 범위
        if not is_within_bounds(new_node, x_max, y_max, t_max):
            continue

        # === RRT* Parent Selection (TB-RRT* 논문 핵심) ===
        # 동적 이웃 반경 계산
        r_n = compute_neighbor_radius(len(tree), NEIGHBOR_RADIUS_BASE)
        
        # 이웃 검색: dist_gamma 기반 (XY만 사용하는 neighbors_xy 대신)
        cand_xyzt = (new_node.x, new_node.y, new_node.t)
        cand_neighbors = neighbors_gamma(tree, cand_xyzt, r_n)
        if not cand_neighbors:
            cand_neighbors = [nearest]

        best_parent = None
        best_cost = float('inf')

        for p in cand_neighbors:
            # 충돌 검사 및 물리적 타당성 검사 (과거로 이동 불가, V_MAX 초과 불가)
            if not collision_free_xy(p, new_node, obstacles):
                continue
            if not is_kinematically_feasible(p, new_node):
                continue
                
            # 비용 계산
            c = p.cost + step_cost(p, cand_xyzt)
            if c < best_cost:
                best_cost, best_parent = c, p

        if best_parent is None:
            # nearest로만이라도 연결 시도
            if not collision_free_xy(nearest, new_node, obstacles) or not is_kinematically_feasible(nearest, new_node):
                continue
            best_parent = nearest
            best_cost = nearest.cost + step_cost(nearest, cand_xyzt)

        # 부모/비용 확정 (아직 트리에 추가하지 않음 — 한 반복당 하나의 노드만 추가)
        new_node.set_parent(best_parent)
        new_node.cost = best_cost

        # 이번 반복에 실제로 트리에 추가할 노드 결정 (new_node 또는 direct_goal)
        appended_node = None

        # 목표 도달 가능성 확인 (공간 AND 시간적 타당성 모두 검사)
        direct_goal_node = try_connect_to_goal(new_node, goal,
                                               x_max, y_max, t_max,
                                               step_size, delta_t,
                                               eps_state, eps_time,
                                               obstacles,
                                               max_steps=20)

        if direct_goal_node is not None:
             # try_connect_to_goal 반환 노드는 이미 모든 검사를 통과한 상태
             direct_goal_node.set_parent(new_node)
             direct_goal_node.cost = new_node.cost + step_cost(new_node, (direct_goal_node.x, direct_goal_node.y, direct_goal_node.t))
             appended_node = direct_goal_node

        # direct_goal이 선택되지 않으면 기본적으로 new_node를 추가
        if appended_node is None:
            appended_node = new_node

        # 트리에 실제로 단 한 번만 추가
        tree.append(appended_node)

        # 노드 임계값 로깅
        if len(tree) % 5000 == 0:
            logging.info(f"Iteration {i}: {len(tree)} nodes created")

        # 노드 상한 검사
        if len(tree) >= max_nodes:
            logging.warning(f"Max nodes {max_nodes} reached at iteration {i}")
            break

        # === RRT* Rewiring (추가된 노드 중심으로 수행) ===
        cand_xyzt_app = (appended_node.x, appended_node.y, appended_node.t)
        for q in neighbors_gamma(tree, cand_xyzt_app, r_n):
            # 자신 및 부모는 스킵
            if q is appended_node or q is appended_node.parent:
                continue
            if is_ancestor(q, appended_node):
                continue
            if not collision_free_xy(appended_node, q, obstacles):
                continue
            
            # 물리적 타당성 검증 (새로운 부모 appended_node -> 자식 q 방향)
            if not is_kinematically_feasible(appended_node, q):
                continue
                
            new_cost_via_app = appended_node.cost + step_cost(appended_node, (q.x, q.y, q.t))
            if new_cost_via_app + 1e-9 < q.cost:
                q.set_parent(appended_node)
                q.cost = new_cost_via_app
                
                # (중지) 새로운 부모가 생겼으므로, 도달 시간(t)도 갱신되어야 함
                # 간단히 dt = dist/V_MAX 로 재계산 하거나, 최단거리 기준 시간차를 구함.
                # 여기서는 논문의 일관성을 위해 "가장 빠른 도착 시간"으로 가정하여
                # dist / V_MAX 만큼의 시간 후 도착한다고 정의함 (가장 이상적)
                dist_to_q = math.hypot(q.x - appended_node.x, q.y - appended_node.y)
                ideal_new_t = appended_node.t + (dist_to_q / V_MAX)
                
                # 기존 도착시간과 새로운 도착시간의 차이 계산
                time_shift = ideal_new_t - q.t
                
                # q 노드의 시간 업데이트
                q.t = ideal_new_t
                
                # 자식들에게 단순 'cost'만 전파하는 것이 아니라,
                # 시간 변위(time_shift)를 전파하여 상대적인 이동 속성(Velocity profiles)을 유지함.
                propagate_time_shift_to_leaves(q, time_shift)

        # === 목표 도달 판정: appended_node 기준 (공간 AND 시간) ===
        spatial_err = math.hypot(appended_node.x - goal.x, appended_node.y - goal.y)
        time_err = abs(appended_node.t - goal.t)
        if spatial_err <= eps_state and time_err <= eps_time:
            # User Request: If exact connection (spatial_err ~ 0), report distance from parent as error.
            # This represents "approach distance".
            report_spatial_err = spatial_err
            if spatial_err < 1e-6 and appended_node.parent:
                 p = appended_node.parent
                 report_spatial_err = math.hypot(p.x - goal.x, p.y - goal.y)
            
            total_err = math.sqrt(report_spatial_err**2 + time_err**2)
            
            if appended_node.cost < best_goal_cost:
                if best_goal_node is None:
                    first_sol_time = time.time() - start_time_algo
                    # Reconstruct first path to measure length
                    temp_path = reconstruct_path(appended_node)
                    first_path_len = path_length_xy(temp_path) # Option B: XY Length
                    first_spatial_err = report_spatial_err
                    first_time_err = time_err
                    first_total_err = total_err # Raw error (re-calced in run_experiments)
                    logging.info(f"First solution found at iteration {i}: time={first_sol_time:.4f}s, len={first_path_len:.3f}, err={first_total_err:.3f}")
                best_goal_cost = appended_node.cost
                best_goal_node = appended_node
                best_spatial_err = report_spatial_err
                best_time_err = time_err
                best_total_err = total_err
                logging.info(f"Goal reached (spatial+time) at iteration {i}: cost={best_goal_cost:.3f}, t={appended_node.t:.3f}")
                logging.info(f"  Spatial error: {best_spatial_err:.3f}m, Time error: {best_time_err:.3f}s (both within tolerances)")
                
                # History
                cur_t = time.time() - start_time_algo
                cost_history.append((cur_t, best_goal_cost))
        
        # 성능 모니터링 (매 1000 iteration마다)
        if i % 1000 == 0 and best_goal_node:
            logging.debug(f"Iteration {i}: best_goal_cost={best_goal_cost:.3f}, nodes={len(tree)}")

    if best_goal_node:
        logging.info(f"Final best cost: {best_goal_cost:.3f} at iteration {i}")
        final_path = reconstruct_path(best_goal_node)
        
        # 마지막 노드에서 실제 Goal까지 직접 연결 (visualization용)
        if final_path:
            last_node = final_path[-1]
            spatial_dist = math.hypot(last_node.x - goal.x, last_node.y - goal.y)
            if spatial_dist > 1e-3:  # Goal이 노드와 다르면
                logging.info(f"Adding final segment to Goal: distance={spatial_dist:.3f}m")
                final_path.append(goal)  # 실제 Goal 노드 추가
        
        return final_path, tree, best_spatial_err, best_time_err, best_total_err, first_sol_time, first_path_len, first_spatial_err, first_time_err, first_total_err, cost_history
    
    # Goal 도달 실패 시: 목표에 가장 가까운 노드를 찾아 부분 경로 반환
    logging.warning(f"Goal not reached within {max_iterations} iterations")
    
    # best effort: 목표에 가장 가까운 노드 찾기
    best_partial_node = None
    best_partial_dist = float('inf')
    partial_spatial_err = None
    partial_time_err = None
    partial_total_err = None
    
    for node in tree:
        spatial_err = math.hypot(node.x - goal.x, node.y - goal.y)
        time_err = abs(node.t - goal.t)
        total_err = math.sqrt(spatial_err**2 + time_err**2)
        
        if total_err < best_partial_dist:
            best_partial_dist = total_err
            best_partial_node = node
            partial_spatial_err = spatial_err
            partial_time_err = time_err
            partial_total_err = total_err
    
    if best_partial_node:
        # User Request: If best partial node is AT goal (spatial=0) but failed time,
        # report the APPROACH distance (parent) as spatial error.
        # Also ensure Time and Total error are consistent with the reported node.
        if partial_spatial_err < 1e-3 and best_partial_node.parent:
            # Backtrack to parent (Approach Node)
            best_partial_node = best_partial_node.parent
            
            # Recalculate all errors for consistency
            partial_spatial_err = math.hypot(best_partial_node.x - goal.x, best_partial_node.y - goal.y)
            partial_time_err = abs(best_partial_node.t - goal.t)
            best_partial_dist = math.sqrt(partial_spatial_err**2 + partial_time_err**2) # Total
            partial_total_err = best_partial_dist

        logging.info(f"Best partial path: closest node at distance {best_partial_dist:.3f}")
        logging.info(f"  Partial spatial error: {partial_spatial_err:.3f}, time error: {partial_time_err:.3f}")
        partial_path = reconstruct_path(best_partial_node)
        return partial_path, tree, partial_spatial_err, partial_time_err, partial_total_err, None, None, None, None, None, []
    
    return None, tree, None, None, None, None, None, None, None, None, []

def plot_results(tree, path, start_node, goal_node, spatial_err, time_err, total_err, elapsed_time, trial_idx, save_dir, world_bounds, obstacles):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # STL-RRT* 스타일 2D 플롯
    fig, ax = plt.subplots(figsize=(6.6, 5.0), dpi=120)
    
    # fail case는 "FAILED" 표시
    if path is None:
        title = f"Trial {trial_idx} — TB-RRT (FAILED)"
    else:
        title = f"Trial {trial_idx} — TB-RRT"
    ax.set_title(title)

    ax.set_xlim(world_bounds[0], world_bounds[1])
    ax.set_ylim(world_bounds[2], world_bounds[3])

    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linewidth=0.4, alpha=0.4)

    # 트리 (엣지)
    for node in tree:
        if node.parent:
            ax.plot([node.x, node.parent.x], [node.y, node.parent.y], linewidth=0.4, alpha=0.20, color='#888888', zorder=1)

    # === 장애물 그리기 추가 ===
    if obstacles:
        for i, (ox, oy, r) in enumerate(obstacles):
            circ = plt.Circle((ox, oy), r, fill=False, linewidth=1.2, edgecolor='k')
            ax.add_patch(circ)
            ax.text(ox, oy, f"O{i:02d}", fontsize=7, ha="center", va="center", alpha=0.7)

    # 트리 (노드)
    xs = [n.x for n in tree]
    ys = [n.y for n in tree]
    ax.scatter(xs, ys, s=8, color='#4682b4', alpha=0.18, zorder=2)

    # 경로 및 시간 컬러맵 (path가 있을 때만)
    if path and len(path) >= 2:
        px, py, pt = zip(*[(n.x, n.y, n.t) for n in path])
        # fail case 부분 경로는 다른 색으로 표시
        if spatial_err is not None:
            path_color = '#0072B2'  # 성공: 파란색
        else:
            path_color = '#FF6B6B'  # 부분 경로: 빨간색 (실패)
        ax.plot(px, py, linewidth=2.0, label="Path", color=path_color, zorder=4)
        # 시간 컬러맵 (경로를 세분화)
        t_values = []
        pts = []
        for i in range(len(path)-1):
            seg = np.linspace(np.array([path[i].x, path[i].y, path[i].t]), np.array([path[i+1].x, path[i+1].y, path[i+1].t]), num=10)
            if i > 0:
                seg = seg[1:]
            for j in range(seg.shape[0]):
                pts.append((seg[j,0], seg[j,1]))
                t_values.append(seg[j,2])
        pts = np.array(pts, dtype=float)
        t_values = np.array(t_values, dtype=float)
        sc = ax.scatter(pts[:,0], pts[:,1], c=t_values, s=10, cmap="viridis", alpha=0.7, zorder=5)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.75)
        cbar.set_label("time (s)")

    # 시작/목표 마커 (STL-RRT* 스타일)
    ax.scatter(start_node.x, start_node.y, marker='o', s=90, color='#0099FF', edgecolor='k', label='Start', zorder=10)
    ax.scatter(goal_node.x, goal_node.y, marker='*', s=180, color='#FF9900', edgecolor='k', label='Goal', zorder=11)

    # 범례
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, f"trial_{trial_idx:03d}_2d.png"))
    plt.close(fig)


# ============ REFERENCE HEADER (TWA-RRT* Standard) ============
REFERENCE_HEADER = [
    'case_id', 'trial', 'algo', 'success', 'runtime_s', 'path_len', 'arrival_t',
    'spatial_err', 'time_err', 'total_err', 'time_window_violation', 'nodes', 'seed', 
    'first_sol_time', 'first_path_len', 'first_total_err', 'smoothness',
    't_ref', 'L_proxy', 'Tmin', 'Tmax', 'alpha', 'beta', 'kappa',
    'p_tw', 'loiter_used', 'loiter_len', 'late_pruned', 'lb_pruned'
]

def run_case(case: dict, out_root: str = None, save_plots: bool = True):
    """
    Run a single experiment case for TB-RRT.
    """
    global WORLD, OBSTACLES, GOAL_CENTER, GOAL_RADIUS, START, SEED, TRIALS
    global GAMMA, NEIGHBOR_RADIUS_BASE, V_MAX, V_MIN, A_MAX, V0, T_REF, T_WINDOW
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
            "time": {"t_ref": T_REF, "t_window": T_WINDOW},
            "dynamics": {"v_min": V_MIN, "v_max": V_MAX, "a_max": A_MAX, "v0": V0},
            "random": {"seed_base": SEED, "num_trials": TRIALS},
            "planner": {"step_size": STEP_SIZE, "neighbor_radius": 35.0, "n_iters": 6000},
            "algorithms": {"tb_rrt": {"eps_state": 5.0, "eps_time": 0.5, "dt_min": 0.1}},
            "outputs": {"out_dir": "output"}
        }
    case = eff

    # 1. Setup Output Directory
    c_id = case["case_id"]
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if out_root is None:
        out_root = case.get("outputs", {}).get("out_dir", "output")
    
    if not os.path.isabs(out_root):
        out_root = os.path.join(script_dir, out_root)
        
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_root, c_id, "tb_rrt", f"run_{run_timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # 2. Extract Parameters
    w = case["world"]
    s = case["start"]
    g = case["goal"]
    obs_list = case["obstacles"]
    tm = case["time"]
    dyn = case["dynamics"]
    rnd = case["random"]
    pln = case["planner"]
    algo_params = case["algorithms"].get("tb_rrt", {})
    l_world = w["xlim"] + w["ylim"]
    
    # Dynamics (Option B)
    l_v_min, l_v_max, l_a_max, l_v0 = dyn["v_min"], dyn["v_max"], dyn["a_max"], dyn["v0"]
    l_t_ref = tm["t_ref"]
    l_t_window = tuple(tm["t_window"])
    
    # Planner / Algo
    l_step_size = pln.get("step_size", 3.0)
    l_neighbor_radius = pln.get("neighbor_radius", 35.0)
    l_max_iters = pln.get("n_iters", 6000)

    # Global Injection (Fairness)
    global GOAL_BIAS, NEIGHBOR_RADIUS_BASE
    GOAL_BIAS = float(pln.get("goal_sample_rate", GOAL_BIAS))
    NEIGHBOR_RADIUS_BASE = float(pln.get("neighbor_radius", NEIGHBOR_RADIUS_BASE))
    
    l_eps_state = algo_params.get("eps_state", 5.0)
    l_eps_time = algo_params.get("eps_time", 0.5)
    l_dt = algo_params.get("dt_min", 0.1)
    
    # Globals Update (Temporary hack if needed by helpers, but we passed obstacles explicitly)
    # However, dist_gamma uses global GAMMA.
    # GAMMA is hardcoded? TB-RRT uses 1.0 usually
    # GAMMA is hardcoded? TB-RRT uses 1.0 usually
    GAMMA = 1.0 
    NEIGHBOR_RADIUS_BASE = l_neighbor_radius
    V_MAX, V_MIN, A_MAX, V0 = l_v_max, l_v_min, l_a_max, l_v0
    T_REF = l_t_ref
    T_WINDOW = l_t_window
    
    # Obstacles for collision check
    obstacles = [(o["cx"], o["cy"], o["r"]) for o in obs_list]
    
    # Files
    csv_file = os.path.join(run_dir, "summary.csv")
    history_file = os.path.join(run_dir, "cost_history.csv")
    
    with open(history_file, mode='w', newline='') as hf:
        h_writer = csv.writer(hf)
        h_writer.writerow(["trial", "time", "cost"])

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=REFERENCE_HEADER, extrasaction='ignore')
        writer.writeheader()
        
        num_trials = rnd["num_trials"]
        base_seed = rnd["seed_base"]
        
        for trial in range(1, num_trials + 1):
            trial_seed = base_seed + trial - 1
            random.seed(trial_seed)
            np.random.seed(trial_seed)
            
            start_node = Node(s["x"], s["y"], 0.0)
            goal_x, goal_y = g["x"], g["y"]
            goal_node = Node(goal_x, goal_y, l_t_ref)
            
            x_max, y_max = w["xlim"][1], w["ylim"][1]

            start_time = time.time()
            
            # t_max: derive from time window (not hardcoded)
            l_t_max_search = max(l_t_window[1], l_t_ref) * 1.5

            path, tree, spatial_err_algo, time_err_algo, total_err_algo, first_sol_time, first_path_len_algo, \
            first_spatial_err_algo, first_time_err_algo, first_total_err_algo, cost_history = tb_rrt_fixed_goal(
                start_node, goal_node,
                x_max=x_max, y_max=y_max, t_max=l_t_max_search,
                delta_t=l_dt, step_size=l_step_size,
                eps_state=l_eps_state, eps_time=l_eps_time,
                obstacles=obstacles,
                max_iterations=l_max_iters, max_nodes=10000,
                t_window=l_t_window,
                v_min=l_v_min, v_max=l_v_max # Pass explicit dynamics
            ) 
            elapsed_time = time.time() - start_time
            
            # Save history
            with open(history_file, mode='a', newline='') as hf:
                h_writer = csv.writer(hf)
                for (ht, hc) in cost_history:
                    h_writer.writerow([trial, f"{ht:.4f}", f"{hc:.4f}"])
            
            # --- Unified Metrics Calculation ---
            res_path_len = None
            res_arrival_t = None
            res_tmin = None
            res_tmax = None
            res_time_err = None
            res_total_err = None
            res_spatial_err = None
            res_time_window_violation = None
            res_success = 0
            
            if path:
                # STRIP VISUAL TAIL for evaluation
                eval_path = strip_visual_goal_tail(
                    path, goal_x, goal_y, l_t_ref,
                    start_node.x, start_node.y
                )

                # 1. Path Length & Dynamics
                L = path_length_xy(eval_path)
                res_path_len = L
                tmin, tmax = Dynamics.feasible_time_range(L, l_v0, l_v_min, l_v_max, l_a_max)
                res_tmin = tmin
                res_tmax = tmax
                
                # 2. Arrival Time (actual planner output, not dynamics-based)
                res_arrival_t = eval_path[-1].t
                # Time Window Violation (based on actual arrival time)
                tw_lo, tw_hi = l_t_window
                if res_arrival_t < tw_lo:
                    res_time_window_violation = tw_lo - res_arrival_t
                elif res_arrival_t > tw_hi:
                    res_time_window_violation = res_arrival_t - tw_hi
                else:
                    res_time_window_violation = 0.0
                
                # 3. Spatial Error (Approach Logic)
                last_node = eval_path[-1]
                dist_last_to_goal = math.hypot(last_node.x - goal_x, last_node.y - goal_y)
                
                if dist_last_to_goal < 1e-3:
                    # Trace back to find approach node
                    approach_node = None
                    for k in range(len(eval_path) - 2, -1, -1):
                        d = math.hypot(eval_path[k].x - goal_x, eval_path[k].y - goal_y)
                        if d > 1e-3:
                            approach_node = eval_path[k]
                            break
                    if approach_node:
                        res_spatial_err = math.hypot(approach_node.x - goal_x, approach_node.y - goal_y)
                    else:
                        res_spatial_err = 0.0 # Started at goal?
                else:
                    res_spatial_err = dist_last_to_goal

                # Re-check success based on EVALUATION criteria (Unified)
                # Success = Feasible Time AND Close Enough
                is_time_ok = (res_time_window_violation < 1e-6)
                is_spatial_ok = (res_spatial_err <= l_eps_state) 
                
                if is_time_ok and is_spatial_ok:
                    res_success = 1
                else:
                    res_success = 0
                    
                res_time_err = abs(res_arrival_t - l_t_ref)
                res_total_err = math.hypot(res_spatial_err, res_time_err)
                
                # Smoothness
                res_smoothness = calculate_smoothness(eval_path)
            else:
                 res_smoothness = 0.0


            # First Solution Metrics
            f_sol_time = ""
            f_path_len = ""
            f_total_err = ""
            if first_sol_time is not None:
                f_sol_time = f"{first_sol_time:.4f}"
                f_path_len = f"{first_path_len_algo:.3f}"
                # Recalculate first total err based on standardized metrics?
                # Simplify: just use what algo reported (it's close enough or we lack data to re-calc fully)
                f_total_err = f"{first_total_err_algo:.3f}"
            elif path and res_path_len is not None:
                # User requested fallback: if no first solution found (failed case),
                # report the best partial solution metrics as "first solution" logic
                f_sol_time = f"{elapsed_time:.4f}"
                f_path_len = f"{res_path_len:.3f}"
                f_total_err = f"{res_total_err:.3f}"
            
            print(f"Case {c_id} Trial {trial}: Success={res_success}, TotalErr={res_total_err}")

            row = {
                'case_id': c_id,
                'trial': trial,
                'algo': 'tb_rrt_r7',
                'success': res_success,
                'runtime_s': f"{elapsed_time:.4f}",
                'path_len': f"{res_path_len:.3f}" if res_path_len is not None else "",
                'arrival_t': f"{res_arrival_t:.3f}" if res_arrival_t is not None else "",
                'spatial_err': f"{res_spatial_err:.3f}" if res_spatial_err is not None else "",
                'time_err': f"{res_time_err:.3f}" if res_time_err is not None else "",
                'total_err': f"{res_total_err:.3f}" if res_total_err is not None else "",
                'time_window_violation': f"{res_time_window_violation:.3f}" if res_time_window_violation is not None else "",
                'nodes': len(tree),
                'seed': trial_seed,
                'first_sol_time': f_sol_time,
                'first_path_len': f_path_len, 
                'first_total_err': f_total_err, 
                'smoothness': f"{res_smoothness:.3f}" if res_smoothness != "" else "",
                't_ref': l_t_ref,
                'L_proxy': "", 
                'Tmin': f"{res_tmin:.3f}" if res_tmin is not None else "", 
                'Tmax': f"{res_tmax:.3f}" if res_tmax is not None else "",
                'alpha': "", 'beta': "", 'kappa': "",
                'p_tw': "",
                'loiter_used': 0, 'loiter_len': 0.0, 'late_pruned': 0, 'lb_pruned': 0
            }
            writer.writerow(row)
            
            if save_plots:
                 plot_results(tree, path, start_node, goal_node, res_spatial_err, res_time_err, res_total_err, elapsed_time, trial, run_dir, world_bounds=l_world, obstacles=obstacles)

def main():
    parser = argparse.ArgumentParser(description="Run TB-RRT Experiments")
    parser.add_argument("--cases", type=str, help="Path to JSON cases file", default="benchmarks/benchmark_cases_2.json")
    parser.add_argument("--case_id", type=str, help="Specific case ID to run")
    parser.add_argument("--out_dir", type=str, help="Output root directory")
    parser.add_argument("--no_plots", action="store_true", help="Disable plotting")
    args = parser.parse_args()
    
    # Load Cases
    try:
        cases = load_case_file(args.cases)
    except Exception as e:
        print(f"Error loading cases from {args.cases}: {e}")
        # Try finding it in parent dir if not found
        parent_path = os.path.join(os.path.dirname(__file__), "..", args.cases)
        if os.path.exists(parent_path):
             cases = load_case_file(parent_path)
        else:
             return

    # Filter by case_id
    if args.case_id:
        cases = [c for c in cases if c.get("case_id") == args.case_id]
        if not cases:
            print(f"Case ID '{args.case_id}' not found.")
            return

    # Run for each case that includes 'tb_rrt' in algorithms
    for case in cases:
        if "algorithms" in case and "tb_rrt" in case["algorithms"]:
            print(f"Running Case: {case['case_id']}")
            run_case(case, out_root=args.out_dir, save_plots=not args.no_plots)

if __name__ == "__main__":
    main()
