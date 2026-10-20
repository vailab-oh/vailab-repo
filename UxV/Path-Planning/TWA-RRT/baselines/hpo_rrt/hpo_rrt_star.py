# HPO-RRT* benchmark implementation
# Guo, Y., Liu, X., Jia, Q., Liu, X., and Zhang, W. (2023). HPO-RRT*: A Sampling-Based Algorithm for UAV Real-Time Path Planning in a Dynamic Environment. Complex & Intelligent Systems 9, 7133-7153.
# https://doi.org/10.1007/s40747-023-01115-2
# Local implementation notes describe a paper-based reimplementation; no original-author code repository is recorded. Independent authorship still requires confirmation.
# Benchmark changes: static 2D environment; common 6000-draw cap, speed, step, radius and evaluation; native geometric-goal termination retained. See docs/HPO_IMPLEMENTATION_NOTES.md.
# Redistribution license: unresolved; root MIT scope excludes baselines.
# Publication preparation changes only names, references and file/output paths.

"""HPO-RRT* R1: explicit timing events, unchanged native search/termination.

DOI: 10.1007/s40747-023-01115-2. See README.md for paper ambiguities,
finite-budget cycle mapping and the distinction between tree and flight time.
This module has no arrival-window inputs and no RRT* rewiring operation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
import time

import numpy as np


@dataclass(frozen=True)
class HPOConfig:
    # Table 2 values, except explicitly documented benchmark harmonization.
    k_att: float = 20.0
    k_rep: float = 30.0
    rho_0: float = 3.0
    k_bias: float = 5.0
    N_c: int = 10
    N_l: int = 15
    k_l: float = 1.0
    k_opt: float = 1.0
    epsilon: float = 0.1
    max_angle_deg: float = 60.0
    step_size: float = 3.0
    speed: float = 25.0
    goal_tolerance: float = 5.0
    gamma: float = 35.0
    neighbor_cap: float = 35.0
    max_iters: int = 6000
    # Named reproducibility interpretations; not reported paper constants.
    samples_per_cycle: int = 256
    execution_steps: int = 10
    parent_order: str = 'cost'
    obstacle_distance: str = 'surface'
    numerical_epsilon: float = 1e-10

    @property
    def delta_t(self):
        return self.step_size / self.speed

    def __post_init__(self):
        positive = ('rho_0', 'k_bias', 'k_opt', 'epsilon', 'step_size', 'speed',
                    'goal_tolerance', 'gamma', 'neighbor_cap', 'numerical_epsilon')
        for name in positive:
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive and finite')
        for name in ('k_att', 'k_rep', 'k_l'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be nonnegative and finite')
        for name in ('N_c', 'N_l', 'max_iters', 'samples_per_cycle', 'execution_steps'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not 1 <= self.execution_steps <= self.N_c < self.N_l:
            raise ValueError('Require 1 <= execution_steps <= N_c < N_l')
        if not 0 < self.max_angle_deg <= 180:
            raise ValueError('max_angle_deg must be in (0, 180]')
        if self.parent_order not in ('cost', 'timestamp_cost'):
            raise ValueError('Unknown parent_order')
        if self.obstacle_distance not in ('surface', 'center'):
            raise ValueError('Unknown obstacle_distance')


@dataclass(eq=False)
class Node:
    xy: np.ndarray
    parent: Node | None = None
    children: list = field(default_factory=list)
    depth: int = 0
    timestamp: float = 0.0
    cost: float = 0.0
    checked: bool = True
    valid: bool = True

    @property
    def state(self):
        return (float(self.xy[0]), float(self.xy[1]), 0.0)


@dataclass
class PlanResult:
    points: list
    reached_goal: bool
    reason: str
    iterations: int
    samples: int
    created_nodes: int
    cycles: int
    runtime_s: float


class HPORRTStar:
    def __init__(self, start, goal, environment, config=None):
        self.cfg = config or HPOConfig()
        self.env = environment
        self.goal = np.asarray(goal, dtype=float).copy()
        initial = np.asarray(start, dtype=float).copy()
        if initial.shape != (2,) or self.goal.shape != (2,):
            raise ValueError('The manuscript environment is spatially 2D')
        if not self.env.point_free(initial) or not self.env.point_free(self.goal):
            raise ValueError('Start and goal must be free states')
        self.root = Node(initial)
        self.nodes = [self.root]
        self.executed = [initial.copy()]
        self.incoming = None
        self.iterations = self.samples = self.cycles = 0
        self.created_nodes = 1
        self.stats = {'lazy_edges': 0, 'checked_edges': 0, 'invalidated_edges': 0,
                      'optimizer_sweeps': 0, 'optimizer_accepted_sweeps': 0,
                      'parent_candidates': 0, 'reroots': 0}
        self.cycle_history = []
        self._used = False
        self.first_geometric_path_s = None
        self.first_geometric_path_iteration = None
        self.optimizer_s = 0.0

    def apf_force(self, xy):
        """Algorithms 2; Eqs. (7)-(13), vector directions from (8),(10)."""
        xy = np.asarray(xy, dtype=float)
        attractive = self.cfg.k_att * (self.goal - xy)
        repulsive = np.zeros(2)
        for ox, oy, radius in self.env.predict(self.root.timestamp, self.cfg.N_l*self.cfg.delta_t):
            offset = xy - (ox, oy)
            center_dist = float(np.linalg.norm(offset))
            if center_dist <= self.cfg.numerical_epsilon:
                # Gradient undefined at a centre; deterministic finite direction.
                direction = np.array([1.0, 0.0])
            else:
                direction = offset / center_dist
            distance = center_dist - radius if self.cfg.obstacle_distance == 'surface' else center_dist
            distance = max(distance, self.cfg.numerical_epsilon)
            if distance <= self.cfg.rho_0:
                magnitude = self.cfg.k_rep * (1/distance - 1/self.cfg.rho_0) / distance**2
                repulsive += magnitude * direction
        return attractive, repulsive

    def bias_sample(self, xy):
        force = sum(self.apf_force(xy))
        norm = float(np.linalg.norm(force))
        biased = np.asarray(xy, dtype=float).copy()
        if norm > self.cfg.numerical_epsilon:
            biased += self.cfg.k_bias * force / norm  # Eq. (14)
        xmin, xmax, ymin, ymax = self.env.bounds
        return np.clip(biased, (xmin, ymin), (xmax, ymax))

    def sample_apf(self):
        """Algorithm 2: uniform rejection sampling in X_free, then APF bias.

        Each draw (including obstacle rejection) consumes one global iteration.
        Thus a narrow free region cannot silently obtain extra samples.
        """
        xmin, xmax, ymin, ymax = self.env.bounds
        while self.iterations < self.cfg.max_iters:
            self.iterations += 1
            self.samples += 1
            xy = np.array([random.uniform(xmin, xmax), random.uniform(ymin, ymax)])
            if self.env.point_free(xy):
                return self.bias_sample(xy)
        return None

    def angle_ok(self, points, incoming=None):
        """Eq. (4)-(6); climbing angle is zero in this 2D environment."""
        previous = incoming
        cosine_limit = math.cos(math.radians(self.cfg.max_angle_deg))
        for a, b in zip(points[:-1], points[1:]):
            vector = np.asarray(b) - a
            norm = float(np.linalg.norm(vector))
            if norm <= self.cfg.numerical_epsilon:
                continue
            if previous is not None:
                pn = float(np.linalg.norm(previous))
                if pn > self.cfg.numerical_epsilon:
                    if float(np.dot(previous, vector))/(pn*norm) < cosine_limit-1e-12:
                        return False
            previous = vector
        return True

    def _parent_angle_ok(self, parent, xy):
        incoming = parent.xy-parent.parent.xy if parent.parent is not None else self.incoming
        return self.angle_ok([parent.xy, xy], incoming)

    def extend(self, sample=None):
        """Algorithm 3: SampleAPF -> Nearest -> Steer -> Near -> parent.

        Rewiring is intentionally omitted according to HPO-RRT*. Adding a
        node never changes an existing node's parent, cost, depth or timestamp.
        """
        if sample is None:
            sample = self.sample_apf()
        if sample is None:
            return None
        eligible = [n for n in self.nodes if n.valid and n.depth < self.cfg.N_l]
        if not eligible:
            return None
        coordinates = np.asarray([n.xy for n in eligible])
        nearest = eligible[int(np.argmin(np.sum((coordinates-sample)**2, axis=1)))]
        vector = sample-nearest.xy
        norm = float(np.linalg.norm(vector))
        if norm <= self.cfg.numerical_epsilon:
            return None
        xy = nearest.xy + vector * min(1.0, self.cfg.step_size/norm)
        n = max(sum(node.valid for node in self.nodes), 2)
        radius = min(self.cfg.neighbor_cap, self.cfg.gamma*(math.log(n)/n)**0.5)
        distances = np.linalg.norm(coordinates-xy, axis=1)
        candidates = [(p, float(d)) for p, d in zip(eligible, distances)
                      if (d <= radius or p is nearest) and d > self.cfg.numerical_epsilon]
        # OrderParent is ambiguous: prose orders timestamps but calls the parent
        # optimal. Default fulfils requested RRT* minimum length parent selection;
        # timestamp_cost exposes the literal prose interpretation for sensitivity.
        key = (lambda item: (item[0].depth, item[0].cost+item[1])) if self.cfg.parent_order == 'timestamp_cost' else (lambda item: item[0].cost+item[1])
        candidates.sort(key=key)
        for parent, distance in candidates:
            self.stats['parent_candidates'] += 1
            if not self._parent_angle_ok(parent, xy):
                continue
            depth = parent.depth+1
            timestamp = self.root.timestamp+depth*self.cfg.delta_t
            checked = depth <= self.cfg.N_c
            if checked and not self.env.edge_free(parent.xy, xy, parent.timestamp, timestamp):
                continue
            node = Node(xy.copy(), parent, depth=depth, timestamp=timestamp,
                        cost=parent.cost+distance, checked=checked)
            parent.children.append(node)
            self.nodes.append(node)
            self.created_nodes += 1
            self.stats['checked_edges' if checked else 'lazy_edges'] += 1
            return node
        return None

    @staticmethod
    def backtrace(node):
        path = []
        while node is not None:
            path.append(node)
            node = node.parent
        return path[::-1]

    def heuristic_path_finding(self):
        """Algorithm 4, Eqs. (15)-(17): g + k_l*distance at exactly N_l."""
        frontier = [n for n in self.nodes if n.valid and n.depth == self.cfg.N_l]
        if not frontier:
            return []
        end = min(frontier, key=lambda n: n.cost+self.cfg.k_l*np.linalg.norm(n.xy-self.goal))
        return self.backtrace(end)

    def invalidate(self, node):
        pending = [node]
        while pending:
            current = pending.pop()
            current.valid = False
            pending.extend(current.children)

    def validate_branch(self, branch):
        for node in branch[1:]:
            if not node.valid:
                return False
            if not node.checked:
                if not self.env.edge_free(node.parent.xy, node.xy, node.parent.timestamp, node.timestamp):
                    self.invalidate(node)
                    self.stats['invalidated_edges'] += 1
                    return False
                node.checked = True
        return True

    @staticmethod
    def path_cost(points):
        return sum(float(np.linalg.norm(np.asarray(b)-a)) for a,b in zip(points[:-1], points[1:]))

    def low_cost_optimizer(self, points, incoming=None):
        started = time.perf_counter()
        try:
            return self._low_cost_optimizer(points, incoming)
        finally:
            self.optimizer_s += time.perf_counter() - started

    def _low_cost_optimizer(self, points, incoming=None):
        """Algorithm 5, Eqs. (18)-(19); synchronous midpoint offset sweeps.

        A failed sweep is rolled back in full (lines 11-14). Improvement below
        epsilon also returns the PREVIOUS path (lines 18-21). No shortcut/spline.
        Node positions and timestamps in the tree are never modified here.
        """
        current = np.asarray(points, dtype=float).copy()
        if len(current) < 3:
            return current
        while True:
            self.stats['optimizer_sweeps'] += 1
            candidate = current.copy()
            for i in range(1, len(current)-1):
                offset = (current[i-1]+current[i+1])/2-current[i]
                norm = float(np.linalg.norm(offset))
                if norm <= self.cfg.numerical_epsilon:
                    continue
                candidate[i] = current[i]+self.cfg.k_opt*offset/norm
                if (not self.env.edge_free(current[i-1], candidate[i]) or
                    not self.env.edge_free(candidate[i], current[i+1]) or
                    not self.angle_ok([current[i-1], candidate[i], current[i+1]])):
                    return current
            # Synchronous neighbours both move: validate assembled edges too.
            if not self.path_valid(candidate, incoming):
                return current
            if self.path_cost(current)-self.path_cost(candidate) < self.cfg.epsilon:
                return current
            current = candidate
            self.stats['optimizer_accepted_sweeps'] += 1

    def path_valid(self, points, incoming=None):
        return (self.angle_ok(points, incoming) and
                all(self.env.edge_free(a,b) for a,b in zip(points[:-1],points[1:])))

    def reroot(self, selected, optimized_root):
        """Algorithm 1 perception/update: retain reachable descendants only.

        Optimized execution is off-tree. Clone the surviving subtree with the
        new root coordinate; descendant adjacency is retained, never rewired.
        Update costs at this execution boundary and check the new N_c region.
        """
        root = Node(np.asarray(optimized_root).copy(), timestamp=selected.timestamp)
        new_nodes = [root]
        pending = [(selected, root)]
        while pending:
            old, new = pending.pop()
            for child in old.children:
                if not child.valid:
                    continue
                depth = new.depth+1
                if depth > self.cfg.N_l:
                    continue
                incoming = new.xy-new.parent.xy if new.parent is not None else self.incoming
                if not self.angle_ok([new.xy, child.xy], incoming):
                    continue
                checked = depth <= self.cfg.N_c
                timestamp = root.timestamp+depth*self.cfg.delta_t
                if checked and not self.env.edge_free(new.xy, child.xy, new.timestamp, timestamp):
                    self.stats['invalidated_edges'] += 1
                    continue
                node = Node(child.xy.copy(), new, depth=depth, timestamp=timestamp,
                            cost=new.cost+float(np.linalg.norm(child.xy-new.xy)), checked=checked)
                new.children.append(node)
                new_nodes.append(node)
                pending.append((child,node))
        self.root, self.nodes = root, new_nodes
        self.stats['reroots'] += 1

    def plan(self):
        """Algorithm 1 under one shared, finite manuscript iteration budget."""
        if self._used:
            raise RuntimeError('Create a fresh planner for each trial')
        self._used = True
        started = time.perf_counter()
        reached = bool(np.linalg.norm(self.root.xy-self.goal) <= self.cfg.goal_tolerance)
        if reached:
            self.first_geometric_path_s = 0.0
            self.first_geometric_path_iteration = 0
        reason = 'goal_region' if reached else 'iteration_budget'
        while self.iterations < self.cfg.max_iters and not reached:
            self.cycles += 1
            self.env.predict(self.root.timestamp, self.cfg.N_l*self.cfg.delta_t)
            cycle_end = min(self.cfg.max_iters, self.iterations+self.cfg.samples_per_cycle)
            goal_branch = []
            while self.iterations < cycle_end:
                new = self.extend()
                if new is not None and np.linalg.norm(new.xy-self.goal) <= self.cfg.goal_tolerance:
                    branch = self.backtrace(new)
                    if self.validate_branch(branch):
                        if self.first_geometric_path_s is None:
                            self.first_geometric_path_s = time.perf_counter()-started
                            self.first_geometric_path_iteration = self.iterations
                        goal_branch = branch
                        break
            branch = goal_branch or self.heuristic_path_finding()
            self.cycle_history.append({'cycle':self.cycles,'iteration':self.iterations,
                                       'root_time':self.root.timestamp,'nodes':len(self.nodes),
                                       'frontier_found':bool(branch)})
            if not branch:
                continue  # Keep growing; do not invent a shallower subgoal.
            raw = [n.xy for n in branch]
            optimized = self.low_cost_optimizer(raw, self.incoming)
            if goal_branch:
                if not self.path_valid(optimized, self.incoming):
                    continue
                self.executed.extend(p.copy() for p in optimized[1:])
                reached, reason = True, 'goal_region'
                break
            count = min(self.cfg.execution_steps, len(branch)-1)
            # Only execute a checked prefix; far lazy collisions are not flights.
            if not self.validate_branch(branch[:count+1]):
                continue
            prefix = optimized[:count+1]
            if not self.path_valid(prefix, self.incoming):
                continue
            self.executed.extend(p.copy() for p in prefix[1:])
            self.incoming = prefix[-1]-prefix[-2]
            self.reroot(branch[count], prefix[-1])
            reached = bool(np.linalg.norm(self.root.xy-self.goal) <= self.cfg.goal_tolerance)
            if reached:
                reason = 'goal_region'
        # On exhaustion, report only executed (validated) trajectory, not lazy tree.
        if not self.path_valid(self.executed):
            raise AssertionError('Executed trajectory failed final collision/angle audit')
        return PlanResult([p.tolist() for p in self.executed], reached, reason,
                          self.iterations, self.samples, self.created_nodes, self.cycles,
                          time.perf_counter()-started)
