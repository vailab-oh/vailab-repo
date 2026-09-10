import unittest

import torch

from evaluation.common_metrics import (
    DepthMetricAccumulator,
    DetectionMetricAccumulator,
    seg_metrics_from_confmat,
    update_seg_confmat,
)


class CommonMetricTests(unittest.TestCase):
    def test_segmentation_ignores_255_and_uses_global_confusion_matrix(self):
        confusion = torch.zeros((2, 2), dtype=torch.long)
        prediction = torch.tensor([[0, 1], [1, 0]])
        target = torch.tensor([[0, 1], [0, 255]])
        update_seg_confmat(confusion, prediction, target, 2, ignore_index=255)
        metrics = seg_metrics_from_confmat(confusion)
        self.assertAlmostEqual(metrics["seg_pixel_acc"], 2.0 / 3.0, places=6)
        self.assertAlmostEqual(metrics["seg_miou"], 0.5, places=6)
        self.assertAlmostEqual(metrics["seg_mean_dice"], 2.0 / 3.0, places=6)

    def test_depth_is_mean_of_batch_metrics(self):
        accumulator = DepthMetricAccumulator()
        accumulator.update(
            torch.tensor([1.0, 2.0]),
            torch.tensor([1.0, 1.0]),
            torch.tensor([True, True]),
        )
        accumulator.update(
            torch.tensor([4.0]),
            torch.tensor([2.0]),
            torch.tensor([True]),
        )
        metrics = accumulator.compute()
        self.assertAlmostEqual(metrics["depth_abs_rel"], 0.75, places=6)

    def test_detection_perfect_predictions_have_unit_ap(self):
        accumulator = DetectionMetricAccumulator(num_classes=2)
        annotations = [
            torch.tensor(
                [[0.0, 0.0, 10.0, 10.0, 0.0], [20.0, 20.0, 40.0, 40.0, 1.0]]
            )
        ]
        detections = [
            {
                "rois": [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 40.0, 40.0]],
                "scores": [0.9, 0.8],
                "class_ids": [0, 1],
            }
        ]
        accumulator.update(detections, annotations)
        metrics = accumulator.compute()
        self.assertAlmostEqual(metrics["od_map50"], 1.0, places=6)
        self.assertAlmostEqual(metrics["od_map50_95"], 1.0, places=6)
        self.assertAlmostEqual(metrics["od_precision50"], 1.0, places=6)
        self.assertAlmostEqual(metrics["od_recall50"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
