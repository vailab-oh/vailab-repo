"""Minimal example of using TA-RRT* as a library."""

from ta_rrt import Node, ObstacleRectangle, PlannerConfig, plan


def main() -> None:
    start = Node(1.0, 1.0)
    goal = Node(9.0, 9.0)
    obstacles = [
        ObstacleRectangle(3.0, 0.0, 1.0, 6.5),
        ObstacleRectangle(6.0, 3.5, 1.0, 6.5),
    ]
    config = PlannerConfig(max_iterations=1200)
    result = plan(start, goal, obstacles, config=config, seed=7)

    if not result.success:
        raise SystemExit("No path found within the iteration limit")
    print(f"Found a {result.path_length:.3f}-unit path using {len(result.nodes)} nodes")


if __name__ == "__main__":
    main()
