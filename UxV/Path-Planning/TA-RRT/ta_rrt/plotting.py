"""Optional Matplotlib visualization for TA-RRT* results."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .planner import Obstacle, ObstacleCircle, ObstacleRectangle, PlannerConfig, PlanningResult


def plot_result(
    result: PlanningResult,
    obstacles: Sequence[Obstacle],
    config: PlannerConfig,
    *,
    show: bool = True,
    save_path: str | Path | None = None,
):
    """Plot the search tree and final path.

    Matplotlib is imported lazily so the core planner remains usable without the
    optional plotting dependency. The function returns ``(figure, axes)``.
    """

    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle
    except ImportError as error:
        raise RuntimeError(
            'Plotting requires Matplotlib. Install it with: pip install ".[plot]"'
        ) from error

    figure, axes = plt.subplots(figsize=(6, 6))
    axes.set_xlim(0.0, config.x_max)
    axes.set_ylim(0.0, config.y_max)
    axes.set_aspect("equal", adjustable="box")
    axes.grid(True, alpha=0.3)

    for obstacle in obstacles:
        if isinstance(obstacle, ObstacleCircle):
            patch = Circle((obstacle.x, obstacle.y), obstacle.radius, color="0.4", alpha=0.6)
        elif isinstance(obstacle, ObstacleRectangle):
            patch = Rectangle(
                (obstacle.x, obstacle.y),
                obstacle.width,
                obstacle.height,
                color="0.4",
                alpha=0.6,
            )
        else:
            raise TypeError(f"unsupported obstacle type: {type(obstacle).__name__}")
        axes.add_patch(patch)

    for node in result.nodes:
        if node.parent is not None:
            axes.plot(
                [node.parent.x, node.x],
                [node.parent.y, node.y],
                color="0.15",
                linewidth=0.35,
                alpha=0.45,
            )

    if result.path:
        axes.plot(
            [node.x for node in result.path],
            [node.y for node in result.path],
            color="tab:purple",
            linewidth=2.5,
            label="Path",
        )

    start = result.nodes[0]
    axes.scatter(start.x, start.y, c="tab:blue", s=55, label="Start", zorder=5)
    axes.scatter(
        result.target_goal.x,
        result.target_goal.y,
        c="tab:green",
        s=55,
        label="Goal",
        zorder=5,
    )

    axes.set_title("TA-RRT* path planning")
    axes.legend(loc="best")

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return figure, axes
