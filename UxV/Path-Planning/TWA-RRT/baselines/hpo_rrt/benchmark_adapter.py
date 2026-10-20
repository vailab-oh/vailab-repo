"""Read-only bridge to the manuscript's existing environment and evaluation.

No existing planner search is invoked. The evaluation facade supplies a finished
path to WiserRrtPlanner.plan(), whose search call is a no-op here.
"""
from __future__ import annotations

import copy
import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def load_module(name, relative):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    # Do not create bytecode next to the existing research scripts.
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


manuscript = load_module('hpo_manuscript', 'twa_rrt.py')
spatial = load_module('hpo_spatial', 'baselines/st_rrt.py')


def load_cases(path):
    return manuscript.load_case_file(str(path))


def configure(case):
    cfg = manuscript.apply_case_to_config(copy.deepcopy(case), manuscript.CONFIG)
    lo, hi, meta = manuscript.resolve_time_window(
        case.get('algorithms', {}).get('wiser_rrt', {}), cfg)
    cfg.t_window = (lo, hi)
    if meta['time_window_mode'] != 'fixed':
        cfg.t_ref = meta['t_ref']
    return cfg, meta


class StaticEnvironment:
    """The unchanged 2D circular-obstacle world; prediction is a static query."""
    def __init__(self, cfg):
        self.bounds = tuple(cfg.world_bounds)
        self.obstacles = tuple(cfg.obstacles)
        self._points = manuscript.ObstacleManager(cfg)

    def predict(self, timestamp, horizon):
        return self.obstacles

    def point_free(self, xy, timestamp=0.0):
        return all(math.isfinite(float(v)) for v in xy) and not self._points.is_collision(*xy)

    def edge_free(self, a, b, t0=0.0, t1=0.0):
        # Analytic checker already used by ST-RRT: same circles, zero clearance.
        if not self.point_free(a, t0) or not self.point_free(b, t1):
            return False
        return spatial.collision_free_xy(
            SimpleNamespace(x=a[0], y=a[1]), SimpleNamespace(x=b[0], y=b[1]),
            world=self.bounds, obstacles=self.obstacles)


class FinishedPath:
    """Minimal evaluation-only interface; never instantiate an existing RRT*."""
    def __init__(self, points, cfg):
        self.nodes = []
        self.stats = {}
        self.t_window = cfg.t_window
        self.segments = []
        length = 0.0
        parent = None
        for i, point in enumerate(points):
            state = (float(point[0]), float(point[1]), 0.0)
            if parent is not None:
                segment = manuscript.EuclideanMath.shortest_path(parent.state, state)
                self.segments.append((segment, parent.state))
                length += segment.length
            arrival = manuscript.Dynamics.feasible_time_range(
                length, cfg.v0, cfg.v_min, cfg.v_max, cfg.a_max)[0]
            node = SimpleNamespace(state=state, parent=parent, L=length, t=arrival)
            self.nodes.append(node)
            parent = node

    def search(self, t_window=None):
        pass  # Evaluation only: no search, repair, retiming, or loitering.

    def get_best_path_node(self):
        return self.nodes[-1] if self.nodes else None

    def extract_path(self, node):
        return self.segments


def evaluate(points, cfg):
    facade = FinishedPath(points, cfg)
    result = manuscript.WiserRrtPlanner(cfg, facade).plan(t_window=cfg.t_window)
    # Root-only output has no segments; the legacy evaluator assumes a goal
    # endpoint in that case. Supply a zero-length segment to expose actual xy.
    if len(points) == 1:
        state = facade.nodes[0].state
        facade.segments = [(manuscript.EuclideanMath.shortest_path(state, state), state)]
        result = manuscript.WiserRrtPlanner(cfg, facade).plan(t_window=cfg.t_window)
    return result


def smoothness(result):
    pts = []
    for data, start in result['segments']:
        pts.extend(manuscript.EuclideanMath.sample_path(data, start, 1.0))
    return manuscript.calculate_smoothness(pts)


def plot(cfg, nodes, result, trial, folder):
    # Reuse the plotting function verbatim; change only the algorithm title.
    original = matplotlib.axes.Axes.set_title

    def title(ax, label, *args, **kwargs):
        return original(ax, label.replace('TWA-RRT*', 'HPO-RRT*'), *args, **kwargs)

    with patch.object(matplotlib.axes.Axes, 'set_title', title):
        manuscript.plot_results(cfg, SimpleNamespace(nodes=nodes), result,
                                run_id=trial, save_dir=str(folder))
    plt.close('all')
