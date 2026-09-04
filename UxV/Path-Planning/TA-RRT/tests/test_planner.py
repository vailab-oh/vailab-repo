import unittest

import numpy as np

from ta_rrt import (
    Node,
    ObstacleCircle,
    ObstacleRectangle,
    PlannerConfig,
    build_occupancy_grid,
    create_alias_table,
    is_collision_free,
    plan,
)


class AliasTableTests(unittest.TestCase):
    def test_input_is_not_modified(self) -> None:
        probabilities = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float)
        original = probabilities.copy()

        probability_table, alias_table = create_alias_table(probabilities)

        np.testing.assert_array_equal(probabilities, original)
        self.assertEqual(probability_table.shape, (4,))
        self.assertEqual(alias_table.shape, (4,))

    def test_invalid_probabilities_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            create_alias_table([0.0, 0.0])
        with self.assertRaises(ValueError):
            create_alias_table([0.5, -0.5])


class GeometryTests(unittest.TestCase):
    def test_collision_check_includes_endpoints(self) -> None:
        obstacle = ObstacleRectangle(1.0, 1.0, 1.0, 1.0)
        self.assertFalse(is_collision_free(Node(1.0, 1.0), Node(0.0, 0.0), [obstacle], 3, 3))
        self.assertTrue(is_collision_free(Node(0.0, 0.0), Node(0.5, 0.5), [obstacle], 3, 3))

    def test_circle_and_rectangular_map_rasterization(self) -> None:
        config = PlannerConfig(x_max=8.0, y_max=4.0, grid_resolution=0.25)
        grid = build_occupancy_grid([ObstacleCircle(4.0, 2.0, 0.75)], config)
        self.assertEqual(grid.shape, (16, 32))
        self.assertGreater(int(np.count_nonzero(grid)), 0)


class PlannerTests(unittest.TestCase):
    def test_seeded_plan_is_repeatable_and_collision_free(self) -> None:
        config = PlannerConfig(
            x_max=6.0,
            y_max=6.0,
            max_iterations=450,
            grid_resolution=0.25,
        )
        obstacles = [ObstacleRectangle(2.5, 0.0, 0.8, 4.5)]
        start = Node(0.5, 0.5)
        goal = Node(5.5, 5.5)

        first = plan(start, goal, obstacles, config=config, seed=11)
        second = plan(start, goal, obstacles, config=config, seed=11)

        self.assertTrue(first.success)
        self.assertEqual(
            [(node.x, node.y) for node in first.path],
            [(node.x, node.y) for node in second.path],
        )
        self.assertIsNone(start.parent)
        self.assertIsNone(goal.parent)
        for node1, node2 in zip(first.path, first.path[1:], strict=False):
            self.assertTrue(
                is_collision_free(
                    node1,
                    node2,
                    obstacles,
                    config.x_max,
                    config.y_max,
                    config.collision_resolution,
                )
            )

    def test_invalid_endpoints_are_rejected(self) -> None:
        obstacle = ObstacleRectangle(1.0, 1.0, 1.0, 1.0)
        with self.assertRaisesRegex(ValueError, "start node is inside"):
            plan(Node(1.5, 1.5), Node(4.0, 4.0), [obstacle], seed=1)


if __name__ == "__main__":
    unittest.main()
