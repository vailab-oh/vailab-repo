"""Command-line interface for a deterministic TA-RRT* demonstration."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

from .planner import Node, ObstacleRectangle, PlannerConfig, PlanningResult, plan


def demo_environment() -> tuple[Node, Node, list[ObstacleRectangle]]:
    """Return a small environment suitable for a quick smoke test."""

    start = Node(1.0, 1.0)
    goal = Node(9.0, 9.0)
    obstacles = [
        ObstacleRectangle(3.0, 0.0, 1.0, 6.5),
        ObstacleRectangle(6.0, 3.5, 1.0, 6.5),
    ]
    return start, goal, obstacles


def _write_result(path: str | Path, result: PlanningResult) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    with output_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "seed",
                "iterations",
                "success",
                "elapsed_time_ms",
                "nodes",
                "path_length",
                "first_path_time_ms",
                "first_path_nodes",
                "first_path_cost",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "seed": result.seed,
                "iterations": result.iterations,
                "success": int(result.success),
                "elapsed_time_ms": result.elapsed_time_ms,
                "nodes": len(result.nodes),
                "path_length": result.path_length,
                "first_path_time_ms": result.first_path_time_ms,
                "first_path_nodes": result.first_path_node_count,
                "first_path_cost": result.first_path_cost,
            }
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7, help="random seed (default: 7)")
    parser.add_argument(
        "--iterations", type=int, default=1200, help="maximum iterations (default: 1200)"
    )
    parser.add_argument("--plot", action="store_true", help="open an interactive plot")
    parser.add_argument("--save-figure", metavar="PATH", help="save the plot to PATH")
    parser.add_argument("--output-csv", metavar="PATH", help="append summary data to PATH")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    start, goal, obstacles = demo_environment()
    config = PlannerConfig(max_iterations=args.iterations)
    result = plan(start, goal, obstacles, config=config, seed=args.seed)

    print(f"success: {result.success}")
    print(f"iterations: {result.iterations}")
    print(f"nodes: {len(result.nodes)}")
    print(f"elapsed time: {result.elapsed_time_ms:.1f} ms")
    if result.path_length is not None:
        print(f"path length: {result.path_length:.3f} map units")

    if args.output_csv:
        _write_result(args.output_csv, result)
        print(f"result appended to: {args.output_csv}")

    if args.plot or args.save_figure:
        from .plotting import plot_result

        plot_result(
            result,
            obstacles,
            config,
            show=args.plot,
            save_path=args.save_figure,
        )
        if args.save_figure:
            print(f"figure saved to: {args.save_figure}")

    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
