"""Metric definitions shared by MTL training and post-training SOTA evaluation.

The implementations in this module intentionally preserve the metric protocol that
was used by ``train.py``. In particular, depth metrics are computed over all valid
pixels in a batch and then averaged across batches.
"""

import numpy as np
import torch
from torchvision.ops import box_iou


OD_IOU_THRESHOLDS = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)


def to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().mean().item())
    return float(value)


def update_seg_confmat(confmat, pred, target, num_classes, ignore_index=255):
    valid = target != int(ignore_index)
    if not valid.any():
        return confmat
    pred = pred[valid].clamp(0, int(num_classes) - 1)
    target = target[valid].clamp(0, int(num_classes) - 1)
    bins = target * int(num_classes) + pred
    confmat += torch.bincount(
        bins, minlength=int(num_classes) ** 2
    ).reshape(int(num_classes), int(num_classes))
    return confmat


def seg_metrics_from_confmat(confmat):
    confmat = confmat.float()
    tp = confmat.diag()
    row_sum = confmat.sum(dim=1)
    col_sum = confmat.sum(dim=0)
    union = row_sum + col_sum - tp
    iou = tp / union.clamp(min=1.0)
    acc = tp / row_sum.clamp(min=1.0)
    dice = (2.0 * tp) / (row_sum + col_sum).clamp(min=1.0)
    present = row_sum > 0
    return {
        "seg_pixel_acc": to_float(tp.sum() / confmat.sum().clamp(min=1.0)),
        "seg_mean_acc": to_float(acc[present].mean()) if present.any() else 0.0,
        "seg_miou": to_float(iou[present].mean()) if present.any() else 0.0,
        "seg_mean_dice": to_float(dice[present].mean()) if present.any() else 0.0,
        "seg_class_iou": [to_float(value) for value in iou],
        "seg_class_acc": [to_float(value) for value in acc],
        "seg_class_dice": [to_float(value) for value in dice],
    }


def depth_metric_values(pred, gt, mask):
    pred = pred[mask.bool()].clamp(min=1e-6)
    gt = gt[mask.bool()].clamp(min=1e-6)
    if pred.numel() == 0:
        return None
    diff = pred - gt
    abs_diff = diff.abs()
    ratio = torch.maximum(pred / gt, gt / pred)
    return {
        "abs_rel": to_float((abs_diff / gt).mean()),
        "sq_rel": to_float((diff.pow(2) / gt).mean()),
        "rmse": to_float(torch.sqrt(diff.pow(2).mean())),
        "rmse_log": to_float(
            torch.sqrt((torch.log(pred) - torch.log(gt)).pow(2).mean())
        ),
        "mae": to_float(abs_diff.mean()),
        "delta1": to_float((ratio < 1.25).float().mean()),
        "delta2": to_float((ratio < 1.25 ** 2).float().mean()),
        "delta3": to_float((ratio < 1.25 ** 3).float().mean()),
    }


class DepthMetricAccumulator:
    """Reproduce MTL's mean-of-batch depth metric aggregation."""

    metric_names = (
        "abs_rel",
        "sq_rel",
        "rmse",
        "rmse_log",
        "mae",
        "delta1",
        "delta2",
        "delta3",
    )

    def __init__(self):
        self.sums = {name: 0.0 for name in self.metric_names}
        self.batch_count = 0

    def update(self, pred, gt, mask):
        values = depth_metric_values(pred, gt, mask)
        if values is None:
            return
        for name, value in values.items():
            self.sums[name] += float(value)
        self.batch_count += 1

    def compute(self):
        if self.batch_count == 0:
            return {"depth_{}".format(name): 0.0 for name in self.metric_names}
        return {
            "depth_{}".format(name): value / self.batch_count
            for name, value in self.sums.items()
        }


class DetectionMetricAccumulator:
    """Dataset-level class-aware AP implementation used by MTL training."""

    def __init__(self, num_classes, iou_thresholds=None):
        self.num_classes = int(num_classes)
        self.iou_thresholds = tuple(iou_thresholds or OD_IOU_THRESHOLDS)
        self.gt_counts = [0 for _ in range(self.num_classes)]
        self.records = {
            threshold: [[] for _ in range(self.num_classes)]
            for threshold in self.iou_thresholds
        }

    def update(self, detections, annotations):
        for detection, ground_truth in zip(detections, annotations):
            ground_truth = torch.as_tensor(ground_truth)
            if ground_truth.ndim != 2 or ground_truth.shape[1] != 5:
                raise ValueError("Detection annotations must have shape [N, 5].")
            valid = ground_truth[ground_truth[:, 4] >= 0].detach().cpu()
            gt_boxes = valid[:, :4].float()
            gt_labels = valid[:, 4].long()
            pred_boxes = torch.as_tensor(detection["rois"], dtype=torch.float32).reshape(-1, 4)
            pred_scores = torch.as_tensor(detection["scores"], dtype=torch.float32).reshape(-1)
            pred_labels = torch.as_tensor(detection["class_ids"], dtype=torch.long).reshape(-1)

            if not (len(pred_boxes) == len(pred_scores) == len(pred_labels)):
                raise ValueError("Detection boxes, scores, and labels must have equal length.")

            for class_id in range(self.num_classes):
                self.gt_counts[class_id] += int((gt_labels == class_id).sum().item())

            for threshold in self.iou_thresholds:
                for class_id in range(self.num_classes):
                    pred_mask = pred_labels == class_id
                    class_pred_boxes = pred_boxes[pred_mask]
                    class_scores = pred_scores[pred_mask]
                    if class_scores.numel() == 0:
                        continue
                    order = torch.argsort(class_scores, descending=True)
                    class_pred_boxes = class_pred_boxes[order]
                    class_scores = class_scores[order]

                    class_gt_boxes = gt_boxes[gt_labels == class_id]
                    matched = torch.zeros((class_gt_boxes.shape[0],), dtype=torch.bool)
                    if class_gt_boxes.numel() > 0:
                        ious = box_iou(class_pred_boxes, class_gt_boxes)
                    else:
                        ious = torch.zeros((class_pred_boxes.shape[0], 0))

                    for pred_index, score in enumerate(class_scores):
                        is_true_positive = 0
                        if class_gt_boxes.shape[0] > 0:
                            max_iou, max_gt = ious[pred_index].max(dim=0)
                            if max_iou >= threshold and not matched[max_gt]:
                                matched[max_gt] = True
                                is_true_positive = 1
                        self.records[threshold][class_id].append(
                            (float(score), is_true_positive)
                        )

    @staticmethod
    def _ap_from_records(records, gt_count):
        if gt_count == 0:
            return None
        if not records:
            return 0.0
        records = sorted(records, key=lambda item: item[0], reverse=True)
        true_positives = np.array([item[1] for item in records], dtype=np.float32)
        false_positives = 1.0 - true_positives
        true_positives = np.cumsum(true_positives)
        false_positives = np.cumsum(false_positives)
        recall = true_positives / max(gt_count, 1)
        precision = true_positives / np.maximum(
            true_positives + false_positives, 1e-12
        )
        recall_envelope = np.concatenate(([0.0], recall, [1.0]))
        precision_envelope = np.concatenate(([0.0], precision, [0.0]))
        for index in range(precision_envelope.size - 1, 0, -1):
            precision_envelope[index - 1] = max(
                precision_envelope[index - 1], precision_envelope[index]
            )
        changed = np.where(recall_envelope[1:] != recall_envelope[:-1])[0]
        return float(
            np.sum(
                (recall_envelope[changed + 1] - recall_envelope[changed])
                * precision_envelope[changed + 1]
            )
        )

    def compute(self):
        per_threshold_map = {}
        per_class_ap50 = []
        for threshold in self.iou_thresholds:
            average_precisions = []
            for class_id in range(self.num_classes):
                average_precision = self._ap_from_records(
                    self.records[threshold][class_id], self.gt_counts[class_id]
                )
                if average_precision is not None:
                    average_precisions.append(average_precision)
                if abs(threshold - 0.5) < 1e-9:
                    per_class_ap50.append(
                        0.0 if average_precision is None else average_precision
                    )
            per_threshold_map[threshold] = (
                float(np.mean(average_precisions)) if average_precisions else 0.0
            )

        threshold50 = min(self.iou_thresholds, key=lambda value: abs(value - 0.5))
        records50 = [
            record
            for class_records in self.records[threshold50]
            for record in class_records
        ]
        true_positives = sum(record[1] for record in records50)
        false_positives = len(records50) - true_positives
        false_negatives = sum(self.gt_counts) - true_positives
        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        return {
            "od_map50": per_threshold_map.get(threshold50, 0.0),
            "od_map50_95": float(np.mean(list(per_threshold_map.values())))
            if per_threshold_map
            else 0.0,
            "od_precision50": float(precision),
            "od_recall50": float(recall),
            "od_f1_50": float(f1),
            "od_ap50_per_class": per_class_ap50,
        }
