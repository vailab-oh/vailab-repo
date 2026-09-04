"""Corner-guided adaptive RRT* path planning.

The module contains no plotting or file-output side effects. Coordinates use map
units throughout; occupancy grids use conventional ``grid[y, x]`` indexing.
"""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable, Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray


@dataclass(eq=False)
class Node:
    """A point in the search tree."""

    x: float
    y: float
    parent: Node | None = None
    cost: float = 0.0


@dataclass(frozen=True)
class ObstacleCircle:
    """A circular obstacle described by its center and radius."""

    x: float
    y: float
    radius: float

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError("circle radius must be positive")

    def contains(self, point: Node) -> bool:
        return math.hypot(point.x - self.x, point.y - self.y) <= self.radius


@dataclass(frozen=True)
class ObstacleRectangle:
    """An axis-aligned rectangular obstacle."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("rectangle width and height must be positive")

    def contains(self, point: Node) -> bool:
        return self.x <= point.x <= self.x + self.width and self.y <= point.y <= (
            self.y + self.height
        )


Obstacle: TypeAlias = ObstacleCircle | ObstacleRectangle
FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int32]
BoolArray: TypeAlias = NDArray[np.bool_]


@dataclass(frozen=True)
class PlannerConfig:
    """Configuration values for :func:`plan`."""

    x_max: float = 10.0
    y_max: float = 10.0
    max_iterations: int = 1500
    goal_radius: float = 0.5
    min_step: float = 0.5
    max_step: float = 0.7
    search_radius: float = 1.4
    density_radius: float = 2.0
    max_corner_density: int = 8
    uniform_sample_rate: float = 0.2
    grid_resolution: float = 0.2
    collision_resolution: float = 0.05

    def __post_init__(self) -> None:
        positive_values = {
            "x_max": self.x_max,
            "y_max": self.y_max,
            "goal_radius": self.goal_radius,
            "min_step": self.min_step,
            "max_step": self.max_step,
            "search_radius": self.search_radius,
            "density_radius": self.density_radius,
            "grid_resolution": self.grid_resolution,
            "collision_resolution": self.collision_resolution,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.max_corner_density <= 0:
            raise ValueError("max_corner_density must be positive")
        if self.min_step > self.max_step:
            raise ValueError("min_step cannot be greater than max_step")
        if not 0.0 <= self.uniform_sample_rate <= 1.0:
            raise ValueError("uniform_sample_rate must be between 0 and 1")


@dataclass(frozen=True)
class PlanningResult:
    """Output from a planning run."""

    nodes: tuple[Node, ...]
    target_goal: Node
    goal_node: Node | None
    elapsed_time_ms: float
    first_path_time_ms: float | None
    first_path_node_count: int | None
    first_path_cost: float | None
    iterations: int
    seed: int | None

    @property
    def success(self) -> bool:
        return self.goal_node is not None

    @property
    def path(self) -> tuple[Node, ...]:
        return extract_path(self.goal_node)

    @property
    def path_length(self) -> float | None:
        return path_length(self.goal_node) if self.goal_node is not None else None


class _CornerIndex:
    """Small NumPy-based spatial index used to avoid a mandatory SciPy dependency."""

    def __init__(self, corners: Sequence[tuple[float, float]]) -> None:
        self._points = np.asarray(corners, dtype=float).reshape((-1, 2))

    def count_within(self, position: tuple[float, float], radius: float) -> int:
        if self._points.size == 0:
            return 0
        delta = self._points - np.asarray(position, dtype=float)
        return int(np.count_nonzero(np.einsum("ij,ij->i", delta, delta) <= radius**2))


def euclidean_distance(node1: Node, node2: Node) -> float:
    """Return the Euclidean distance between two nodes."""

    return math.hypot(node1.x - node2.x, node1.y - node2.y)


def is_point_in_obstacle(point: Node, obstacle: Obstacle) -> bool:
    """Return whether *point* is on or inside *obstacle*."""

    return obstacle.contains(point)


def _is_in_bounds(point: Node, x_max: float, y_max: float) -> bool:
    return 0.0 <= point.x <= x_max and 0.0 <= point.y <= y_max


def is_collision_free(
    node1: Node,
    node2: Node,
    obstacles: Sequence[Obstacle],
    x_max: float,
    y_max: float,
    resolution: float = 0.05,
) -> bool:
    """Check a segment, including both endpoints, against bounds and obstacles."""

    if resolution <= 0:
        raise ValueError("resolution must be positive")

    distance = euclidean_distance(node1, node2)
    sample_count = max(1, math.ceil(distance / resolution))
    for index in range(sample_count + 1):
        ratio = index / sample_count
        point = Node(
            node1.x + (node2.x - node1.x) * ratio,
            node1.y + (node2.y - node1.y) * ratio,
        )
        if not _is_in_bounds(point, x_max, y_max):
            return False
        if any(is_point_in_obstacle(point, obstacle) for obstacle in obstacles):
            return False
    return True


def create_alias_table(probabilities: Iterable[float] | FloatArray) -> tuple[FloatArray, IntArray]:
    """Build Walker alias tables without modifying the caller's input array."""

    source = probabilities if isinstance(probabilities, np.ndarray) else list(probabilities)
    values = np.asarray(source, dtype=float)
    values = values.ravel().copy()
    if values.size == 0:
        raise ValueError("probabilities cannot be empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("probabilities must be finite")
    if np.any(values < 0):
        raise ValueError("probabilities cannot be negative")

    total = float(values.sum())
    if total <= 0:
        raise ValueError("probabilities must have a positive sum")

    count = values.size
    scaled = values * count / total
    probability_table = np.zeros(count, dtype=float)
    alias_table = np.zeros(count, dtype=np.int32)
    small = [index for index, value in enumerate(scaled) if value < 1.0]
    large = [index for index, value in enumerate(scaled) if value >= 1.0]

    while small and large:
        small_index = small.pop()
        large_index = large.pop()
        probability_table[small_index] = scaled[small_index]
        alias_table[small_index] = large_index
        scaled[large_index] -= 1.0 - scaled[small_index]
        (small if scaled[large_index] < 1.0 else large).append(large_index)

    for index in small + large:
        probability_table[index] = 1.0
        alias_table[index] = index

    return probability_table, alias_table


def _sample_alias(
    probability_table: FloatArray,
    alias_table: IntArray,
    x_coordinates: FloatArray,
    y_coordinates: FloatArray,
    rng: random.Random,
) -> Node:
    index = rng.randrange(probability_table.size)
    if rng.random() >= probability_table[index]:
        index = int(alias_table[index])
    return Node(float(x_coordinates[index]), float(y_coordinates[index]))


def build_occupancy_grid(
    obstacles: Sequence[Obstacle], config: PlannerConfig
) -> BoolArray:
    """Rasterize obstacles into a conventional ``grid[y, x]`` occupancy array."""

    width = math.ceil(config.x_max / config.grid_resolution)
    height = math.ceil(config.y_max / config.grid_resolution)
    x_coordinates = np.minimum(
        (np.arange(width, dtype=float) + 0.5) * config.grid_resolution,
        config.x_max,
    )
    y_coordinates = np.minimum(
        (np.arange(height, dtype=float) + 0.5) * config.grid_resolution,
        config.y_max,
    )
    x_grid, y_grid = np.meshgrid(x_coordinates, y_coordinates)
    occupied = np.zeros((height, width), dtype=bool)

    for obstacle in obstacles:
        if isinstance(obstacle, ObstacleRectangle):
            occupied |= (
                (x_grid >= obstacle.x)
                & (x_grid <= obstacle.x + obstacle.width)
                & (y_grid >= obstacle.y)
                & (y_grid <= obstacle.y + obstacle.height)
            )
        elif isinstance(obstacle, ObstacleCircle):
            occupied |= (x_grid - obstacle.x) ** 2 + (y_grid - obstacle.y) ** 2 <= (
                obstacle.radius**2
            )
        else:
            raise TypeError(f"unsupported obstacle type: {type(obstacle).__name__}")

    return occupied


def extract_sampling_corners(
    occupied: BoolArray, config: PlannerConfig, threshold: int = 3
) -> tuple[tuple[float, float], ...]:
    """Find free grid cells that characterize obstacle and map-boundary corners."""

    if occupied.ndim != 2:
        raise ValueError("occupied grid must be two-dimensional")
    if threshold < 0 or threshold > 8:
        raise ValueError("threshold must be between 0 and 8")

    padded = np.pad(occupied, pad_width=1, mode="constant", constant_values=True)
    corners: list[tuple[float, float]] = []
    height, width = occupied.shape
    for y_index in range(height):
        for x_index in range(width):
            if occupied[y_index, x_index]:
                continue
            y_pad = y_index + 1
            x_pad = x_index + 1
            neighbors = padded[y_pad - 1 : y_pad + 2, x_pad - 1 : x_pad + 2]
            occupied_neighbors = int(np.count_nonzero(neighbors))
            if occupied_neighbors > threshold or occupied_neighbors == 1:
                corners.append(
                    (
                        min((x_index + 0.5) * config.grid_resolution, config.x_max),
                        min((y_index + 0.5) * config.grid_resolution, config.y_max),
                    )
                )
    return tuple(corners)


def create_line(start: tuple[int, int], end: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    """Return integer cells on a line using Bresenham's algorithm."""

    x1, y1 = start
    x2, y2 = end
    points: list[tuple[int, int]] = []
    delta_x = abs(x2 - x1)
    delta_y = abs(y2 - y1)
    step_x = 1 if x1 < x2 else -1
    step_y = 1 if y1 < y2 else -1
    error = delta_x - delta_y

    while True:
        points.append((x1, y1))
        if x1 == x2 and y1 == y2:
            return tuple(points)
        doubled_error = 2 * error
        if doubled_error > -delta_y:
            error -= delta_y
            x1 += step_x
        if doubled_error < delta_x:
            error += delta_x
            y1 += step_y


def _coordinate_to_cell(
    x: float, y: float, occupied: BoolArray, config: PlannerConfig
) -> tuple[int, int]:
    height, width = occupied.shape
    x_index = min(max(int(x / config.grid_resolution), 0), width - 1)
    y_index = min(max(int(y / config.grid_resolution), 0), height - 1)
    return x_index, y_index


def _line_is_blocked(line: Sequence[tuple[int, int]], occupied: BoolArray) -> bool:
    height, width = occupied.shape
    return any(
        x < 0 or y < 0 or x >= width or y >= height or occupied[y, x] for x, y in line
    )


def build_visibility_pdf(
    occupied: BoolArray,
    corners: Sequence[tuple[float, float]],
    start: Node,
    goal: Node,
    config: PlannerConfig,
) -> FloatArray:
    """Build a sampling distribution from collision-free corner visibility lines."""

    cells = [_coordinate_to_cell(start.x, start.y, occupied, config)]
    cells.extend(_coordinate_to_cell(x, y, occupied, config) for x, y in corners)
    cells.append(_coordinate_to_cell(goal.x, goal.y, occupied, config))

    line_density = np.zeros(occupied.shape, dtype=float)
    for first_index, first in enumerate(cells):
        for second in cells[first_index + 1 :]:
            line = create_line(first, second)
            if not _line_is_blocked(line, occupied):
                for x_index, y_index in line:
                    line_density[y_index, x_index] += 1.0

    total = float(line_density.sum())
    if total <= 0:
        free = ~occupied
        free_count = int(np.count_nonzero(free))
        if free_count == 0:
            raise ValueError("the occupancy grid has no free cells")
        line_density[free] = 1.0 / free_count
        return line_density
    return line_density / total


def _prepare_alias_sampler(
    pdf: FloatArray, config: PlannerConfig
) -> tuple[FloatArray, IntArray, FloatArray, FloatArray]:
    probability_table, alias_table = create_alias_table(pdf)
    height, width = pdf.shape
    x_coordinates = np.minimum(
        (np.tile(np.arange(width), height) + 0.5) * config.grid_resolution,
        config.x_max,
    ).astype(float)
    y_coordinates = np.minimum(
        (np.repeat(np.arange(height), width) + 0.5) * config.grid_resolution,
        config.y_max,
    ).astype(float)
    return probability_table, alias_table, x_coordinates, y_coordinates


def _nearest_node(nodes: Sequence[Node], target: Node) -> Node:
    return min(nodes, key=lambda node: euclidean_distance(node, target))


def _nearby_nodes(nodes: Sequence[Node], target: Node, radius: float) -> list[Node]:
    return [node for node in nodes if euclidean_distance(node, target) <= radius]


def _adaptive_step_size(
    density: int, max_density: int, min_step: float, max_step: float
) -> float:
    midpoint = max_density / 2.0
    steepness = 0.5
    return (max_step - min_step) / (1.0 + math.exp((density - midpoint) / steepness)) + (
        min_step
    )


def _steer(nearest: Node, target: Node, step_size: float) -> Node:
    distance = euclidean_distance(nearest, target)
    if distance == 0:
        return Node(nearest.x, nearest.y, parent=nearest, cost=nearest.cost)
    travel = min(step_size, distance)
    ratio = travel / distance
    return Node(
        nearest.x + (target.x - nearest.x) * ratio,
        nearest.y + (target.y - nearest.y) * ratio,
    )


def _is_ancestor(candidate: Node, node: Node) -> bool:
    current: Node | None = node
    seen: set[Node] = set()
    while current is not None and current not in seen:
        if current is candidate:
            return True
        seen.add(current)
        current = current.parent
    return False


def _reparent(
    node: Node,
    new_parent: Node,
    new_cost: float,
    children: dict[Node, set[Node]],
) -> None:
    old_parent = node.parent
    if old_parent is not None:
        children[old_parent].discard(node)
    cost_delta = new_cost - node.cost
    node.parent = new_parent
    node.cost = new_cost
    children[new_parent].add(node)

    queue: deque[Node] = deque(children[node])
    visited: set[Node] = {node}
    while queue:
        child = queue.popleft()
        if child in visited:
            raise RuntimeError("cycle detected while propagating rewired costs")
        visited.add(child)
        child.cost += cost_delta
        queue.extend(children[child])


def extract_path(goal_node: Node | None) -> tuple[Node, ...]:
    """Return a start-to-goal path and reject malformed cyclic parent chains."""

    if goal_node is None:
        return ()
    reversed_path: list[Node] = []
    seen: set[Node] = set()
    current: Node | None = goal_node
    while current is not None:
        if current in seen:
            raise RuntimeError("cycle detected in path parent chain")
        seen.add(current)
        reversed_path.append(current)
        current = current.parent
    reversed_path.reverse()
    return tuple(reversed_path)


def path_length(goal_node: Node | None) -> float:
    """Calculate geometric length from a goal node's parent chain."""

    path_nodes = extract_path(goal_node)
    return sum(
        euclidean_distance(path_nodes[index - 1], path_nodes[index])
        for index in range(1, len(path_nodes))
    )


def _validate_endpoint(
    name: str,
    node: Node,
    obstacles: Sequence[Obstacle],
    config: PlannerConfig,
) -> None:
    if not _is_in_bounds(node, config.x_max, config.y_max):
        raise ValueError(f"{name} node is outside the configured map")
    if any(is_point_in_obstacle(node, obstacle) for obstacle in obstacles):
        raise ValueError(f"{name} node is inside an obstacle")


def plan(
    start: Node,
    goal: Node,
    obstacles: Sequence[Obstacle] = (),
    *,
    config: PlannerConfig | None = None,
    seed: int | None = None,
) -> PlanningResult:
    """Plan a path with corner-guided adaptive RRT*.

    The caller's start and goal objects are not mutated. The returned result includes
    the complete search tree, the best goal connection, timing metadata, and a
    start-to-goal path convenience property.
    """

    planner_config = config or PlannerConfig()
    _validate_endpoint("start", start, obstacles, planner_config)
    _validate_endpoint("goal", goal, obstacles, planner_config)

    start_node = Node(float(start.x), float(start.y))
    target_goal = Node(float(goal.x), float(goal.y))
    occupied = build_occupancy_grid(obstacles, planner_config)
    corners = extract_sampling_corners(occupied, planner_config)
    visibility_pdf = build_visibility_pdf(
        occupied, corners, start_node, target_goal, planner_config
    )
    probability_table, alias_table, x_coordinates, y_coordinates = _prepare_alias_sampler(
        visibility_pdf, planner_config
    )
    corner_index = _CornerIndex(corners)
    rng = random.Random(seed)

    nodes: list[Node] = [start_node]
    children: dict[Node, set[Node]] = defaultdict(set)
    best_goal: Node | None = None
    first_path_time_ms: float | None = None
    first_path_node_count: int | None = None
    first_path_cost: float | None = None
    started_at = time.perf_counter()

    for _iteration in range(1, planner_config.max_iterations + 1):
        if rng.random() < planner_config.uniform_sample_rate:
            random_node = Node(
                rng.uniform(0.0, planner_config.x_max),
                rng.uniform(0.0, planner_config.y_max),
            )
        else:
            random_node = _sample_alias(
                probability_table, alias_table, x_coordinates, y_coordinates, rng
            )

        nearest = _nearest_node(nodes, random_node)
        density = corner_index.count_within(
            (nearest.x, nearest.y), planner_config.density_radius
        )
        step_size = _adaptive_step_size(
            density,
            planner_config.max_corner_density,
            planner_config.min_step,
            planner_config.max_step,
        )
        new_node = _steer(nearest, random_node, step_size)
        if euclidean_distance(nearest, new_node) <= 1e-12:
            continue
        if not is_collision_free(
            nearest,
            new_node,
            obstacles,
            planner_config.x_max,
            planner_config.y_max,
            planner_config.collision_resolution,
        ):
            continue

        nearby = _nearby_nodes(nodes, new_node, planner_config.search_radius)
        if nearest not in nearby:
            nearby.append(nearest)
        valid_parents = [
            candidate
            for candidate in nearby
            if is_collision_free(
                candidate,
                new_node,
                obstacles,
                planner_config.x_max,
                planner_config.y_max,
                planner_config.collision_resolution,
            )
        ]
        if not valid_parents:
            continue

        parent = min(
            valid_parents,
            key=lambda candidate: candidate.cost + euclidean_distance(candidate, new_node),
        )
        new_node.parent = parent
        new_node.cost = parent.cost + euclidean_distance(parent, new_node)
        nodes.append(new_node)
        children[parent].add(new_node)

        for nearby_node in nearby:
            if nearby_node is start_node or nearby_node is parent:
                continue
            if _is_ancestor(nearby_node, new_node):
                continue
            candidate_cost = new_node.cost + euclidean_distance(new_node, nearby_node)
            if candidate_cost + 1e-12 >= nearby_node.cost:
                continue
            if is_collision_free(
                new_node,
                nearby_node,
                obstacles,
                planner_config.x_max,
                planner_config.y_max,
                planner_config.collision_resolution,
            ):
                _reparent(nearby_node, new_node, candidate_cost, children)

        distance_to_goal = euclidean_distance(new_node, target_goal)
        if distance_to_goal <= planner_config.goal_radius and is_collision_free(
            new_node,
            target_goal,
            obstacles,
            planner_config.x_max,
            planner_config.y_max,
            planner_config.collision_resolution,
        ):
            candidate_cost = new_node.cost + distance_to_goal
            if best_goal is None or candidate_cost + 1e-12 < best_goal.cost:
                if best_goal is not None and best_goal.parent is not None:
                    children[best_goal.parent].discard(best_goal)
                best_goal = Node(target_goal.x, target_goal.y, new_node, candidate_cost)
                children[new_node].add(best_goal)
                if first_path_time_ms is None:
                    first_path_time_ms = (time.perf_counter() - started_at) * 1000.0
                    first_path_node_count = len(nodes)
                    first_path_cost = candidate_cost

    elapsed_time_ms = (time.perf_counter() - started_at) * 1000.0
    return PlanningResult(
        nodes=tuple(nodes),
        target_goal=target_goal,
        goal_node=best_goal,
        elapsed_time_ms=elapsed_time_ms,
        first_path_time_ms=first_path_time_ms,
        first_path_node_count=first_path_node_count,
        first_path_cost=first_path_cost,
        iterations=planner_config.max_iterations,
        seed=seed,
    )
