from copy import deepcopy
from collections.abc import Mapping
import math

import numpy as np
import torch
from torch import nn
from torchvision.ops import batched_nms

from .loss import calc_iou
from .model import models as OriginalTriVNet


_ENCODER_ALIASES = {
    "resnet": "Resnet",
    "resnet50": "Resnet",
    "resnet-50": "Resnet",
}
_DECODER_ALIASES = {
    "fpn": "FPN",
}


def _normalized_legacy_config(cfg):
    if not isinstance(cfg, Mapping):
        raise TypeError("cfg must be a mapping.")

    normalized = deepcopy(dict(cfg))
    model_cfg = normalized.setdefault("model", {})
    if not isinstance(model_cfg, Mapping):
        raise TypeError("cfg['model'] must be a mapping.")
    model_cfg = dict(model_cfg)
    normalized["model"] = model_cfg

    encoder = model_cfg.get("encoder", model_cfg.get("backbone", "Resnet"))
    encoder_key = str(encoder).strip().lower().replace("_", "-")
    if encoder_key not in _ENCODER_ALIASES:
        raise ValueError(
            "Original TriV-Net uses ResNet-50; " f"got encoder {encoder!r}."
        )
    model_cfg["encoder"] = _ENCODER_ALIASES[encoder_key]

    decoder = model_cfg.get("decoder", model_cfg.get("neck", "FPN"))
    decoder_key = str(decoder).strip().lower().replace("_", "-")
    if decoder_key not in _DECODER_ALIASES:
        raise ValueError("Original TriV-Net uses FPN; " f"got decoder {decoder!r}.")
    model_cfg["decoder"] = _DECODER_ALIASES[decoder_key]

    model_conf = normalized.setdefault("model_conf", {})
    if not isinstance(model_conf, Mapping):
        raise TypeError("cfg['model_conf'] must be a mapping.")
    model_conf = dict(model_conf)
    for task_name in ("Seg", "Depth", "od"):
        model_conf.setdefault(task_name, False)
    normalized["model_conf"] = model_conf

    if model_conf["Seg"] and "seg_num_classes" in normalized:
        seg_num_classes = int(normalized["seg_num_classes"])
        if seg_num_classes <= 0:
            raise ValueError("seg_num_classes must be positive.")
        normalized["seg_num_classes"] = seg_num_classes
    if model_conf["od"]:
        od_num_classes = int(
            normalized.get("od_num_classes", normalized.get("class_id", 5))
        )
        if od_num_classes <= 0:
            raise ValueError("od_num_classes must be positive.")
        normalized["od_num_classes"] = od_num_classes

    return normalized


def build_original_trivnet(cfg):
    return OriginalTriVNet(_normalized_legacy_config(cfg))


def _unwrap_model(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def _model_anchors(model, images, dtype):
    model = _unwrap_model(model)
    if not hasattr(model, "anchors"):
        raise ValueError("Original TriV-Net was built with object detection disabled.")
    anchors = model.anchors(images, dtype=dtype)
    return anchors.to(device=images.device, dtype=dtype)


def _validate_od_tensors(images, classification, regression, annotations=None):
    if images.ndim != 4:
        raise ValueError("images must have shape [B, C, H, W].")
    if classification.ndim != 3:
        raise ValueError("classification must have shape [B, N, C].")
    if regression.ndim != 3 or regression.shape[-1] != 4:
        raise ValueError("regression must have shape [B, N, 4].")
    if classification.shape[:2] != regression.shape[:2]:
        raise ValueError("classification and regression must use the same anchors.")
    if classification.shape[0] != images.shape[0]:
        raise ValueError("OD outputs and images must have the same batch size.")
    if classification.device != regression.device or images.device != regression.device:
        raise ValueError(
            "images, classification, and regression must be on one device."
        )
    if not classification.is_floating_point() or not regression.is_floating_point():
        raise TypeError("classification and regression must be floating-point tensors.")
    if annotations is not None:
        if annotations.ndim != 3 or annotations.shape[-1] != 5:
            raise ValueError("annotations must have shape [B, M, 5].")
        if annotations.shape[0] != images.shape[0]:
            raise ValueError("annotations and images must have the same batch size.")


def _device_safe_focal_loss(classifications, regressions, anchors, annotations):

    alpha = 0.25
    gamma = 2.0
    anchor = anchors[0]
    anchor_widths = anchor[:, 2] - anchor[:, 0]
    anchor_heights = anchor[:, 3] - anchor[:, 1]
    anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
    anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights

    classification_losses = []
    regression_losses = []
    iou_scores = []

    for batch_index in range(classifications.shape[0]):
        classification = classifications[batch_index].clamp(1e-4, 1.0 - 1e-4)
        regression = regressions[batch_index]
        bbox_annotation = annotations[batch_index]
        bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

        if bbox_annotation.shape[0] == 0:
            focal_weight = (1.0 - alpha) * classification.pow(gamma)
            classification_losses.append(
                (focal_weight * (-torch.log1p(-classification))).sum()
            )
            regression_losses.append(regression.sum() * 0.0)
            iou_scores.append(classification.sum() * 0.0)
            continue

        iou = calc_iou(anchor, bbox_annotation[:, :4])
        iou_max, iou_argmax = torch.max(iou, dim=1)
        positive = iou_max >= 0.5
        iou_scores.append(
            iou_max[positive].mean() if positive.any() else iou_max.sum() * 0.0
        )

        targets = torch.full_like(classification, -1.0)
        targets[iou_max < 0.4] = 0.0
        assigned_annotations = bbox_annotation[iou_argmax]
        targets[positive] = 0.0
        if positive.any():
            assigned_classes = assigned_annotations[positive, 4].long()
            if (
                (assigned_classes < 0) | (assigned_classes >= classification.shape[1])
            ).any():
                raise ValueError(
                    "Detection annotation contains an out-of-range class id."
                )
            targets[positive, assigned_classes] = 1.0

        alpha_factor = torch.full_like(targets, alpha)
        alpha_factor = torch.where(targets == 1.0, alpha_factor, 1.0 - alpha_factor)
        focal_weight = torch.where(targets == 1.0, 1.0 - classification, classification)
        focal_weight = alpha_factor * focal_weight.pow(gamma)
        bce = -(
            targets * torch.log(classification)
            + (1.0 - targets) * torch.log1p(-classification)
        )
        classification_loss = torch.where(
            targets != -1.0, focal_weight * bce, torch.zeros_like(bce)
        )
        num_positive = positive.sum().to(classification.dtype).clamp(min=1.0)
        classification_losses.append(classification_loss.sum() / num_positive)

        if positive.any():
            assigned_positive = assigned_annotations[positive]
            anchor_widths_positive = anchor_widths[positive]
            anchor_heights_positive = anchor_heights[positive]
            anchor_ctr_x_positive = anchor_ctr_x[positive]
            anchor_ctr_y_positive = anchor_ctr_y[positive]

            gt_widths = (assigned_positive[:, 2] - assigned_positive[:, 0]).clamp(min=1)
            gt_heights = (assigned_positive[:, 3] - assigned_positive[:, 1]).clamp(
                min=1
            )
            gt_ctr_x = assigned_positive[:, 0] + 0.5 * gt_widths
            gt_ctr_y = assigned_positive[:, 1] + 0.5 * gt_heights
            regression_targets = torch.stack(
                (
                    (gt_ctr_x - anchor_ctr_x_positive) / anchor_widths_positive,
                    (gt_ctr_y - anchor_ctr_y_positive) / anchor_heights_positive,
                    torch.log(gt_widths / anchor_widths_positive),
                    torch.log(gt_heights / anchor_heights_positive),
                ),
                dim=1,
            )
            regression_targets = regression_targets / regression.new_tensor(
                (0.1, 0.1, 0.2, 0.2)
            )
            regression_diff = torch.abs(regression_targets - regression[positive])
            regression_loss = torch.where(
                regression_diff <= (1.0 / 9.0),
                0.5 * 9.0 * regression_diff.pow(2),
                regression_diff - (0.5 / 9.0),
            )
            regression_losses.append(regression_loss.mean())
        else:
            regression_losses.append(regression.sum() * 0.0)

    if not classification_losses:
        zero = classifications.sum() * 0.0
        return zero, zero, zero
    return (
        torch.stack(classification_losses).mean(),
        torch.stack(regression_losses).mean(),
        torch.stack(iou_scores).mean(),
    )


class _DeviceSafeFocalLoss(nn.Module):

    def forward(self, classifications, regressions, anchors, annotations):
        return _device_safe_focal_loss(
            classifications, regressions, anchors, annotations
        )


class OriginalTriVNetODLoss(nn.Module):

    def __init__(self, classification_weight=1.0, regression_weight=1.0):
        super().__init__()
        self.classification_weight = float(classification_weight)
        self.regression_weight = float(regression_weight)
        self.focal_loss = _DeviceSafeFocalLoss()

    def forward(self, model, images, classification, regression, annotations):
        annotations = torch.as_tensor(annotations, device=classification.device)
        _validate_od_tensors(images, classification, regression, annotations)
        annotations = annotations.to(dtype=regression.dtype)

        valid_annotations = annotations[..., 4] != -1
        swapped_coordinates = annotations[..., [1, 0, 3, 2]]
        legacy_annotations = annotations.clone()
        legacy_annotations[..., :4] = torch.where(
            valid_annotations.unsqueeze(-1),
            swapped_coordinates,
            annotations[..., :4],
        )
        anchors = _model_anchors(model, images, regression.dtype)
        if anchors.ndim != 3 or anchors.shape[-1] != 4:
            raise ValueError("anchors must have shape [1, N, 4].")
        if anchors.shape[1] != regression.shape[1]:
            raise ValueError(
                "Anchor count does not match legacy OD output: "
                f"{anchors.shape[1]} anchors vs {regression.shape[1]} predictions."
            )

        cls_loss, reg_loss, mean_iou = self.focal_loss(
            classification, regression, anchors, legacy_annotations
        )

        loss = self.classification_weight * cls_loss + self.regression_weight * reg_loss
        return loss.reshape(()), mean_iou.reshape(())


def _empty_detection():
    return {
        "rois": np.empty((0, 4), dtype=np.float32),
        "scores": np.empty((0,), dtype=np.float32),
        "class_ids": np.empty((0,), dtype=np.int64),
    }


@torch.no_grad()
def decode_original_detections(
    model,
    images,
    classification,
    regression,
    score_thresh=0.001,
    nms_thresh=0.5,
    topk=100,
):

    _validate_od_tensors(images, classification, regression)
    score_thresh = float(score_thresh)
    nms_thresh = float(nms_thresh)
    topk = int(topk)
    if not math.isfinite(score_thresh):
        raise ValueError("score_thresh must be finite.")
    if not math.isfinite(nms_thresh) or not 0.0 <= nms_thresh <= 1.0:
        raise ValueError("nms_thresh must be between 0 and 1.")
    if topk <= 0:
        raise ValueError("topk must be positive.")

    legacy_model = _unwrap_model(model)
    anchors = _model_anchors(legacy_model, images, regression.dtype)
    if anchors.shape[1] != regression.shape[1]:
        raise ValueError(
            "Anchor count does not match legacy OD output: "
            f"{anchors.shape[1]} anchors vs {regression.shape[1]} predictions."
        )
    if not hasattr(legacy_model, "regressBoxes") or not hasattr(
        legacy_model, "clipBoxes"
    ):
        raise ValueError("Original TriV-Net OD decode modules are unavailable.")

    regression_std = regression.new_tensor((0.1, 0.1, 0.2, 0.2))
    boxes = legacy_model.regressBoxes(anchors, regression * regression_std)
    boxes = legacy_model.clipBoxes(boxes, images)
    anchor_scores, anchor_classes = classification.max(dim=2)
    results = []

    pre_nms_topk = max(topk, topk * 10)
    for batch_index in range(images.shape[0]):
        boxes_per_image = boxes[batch_index]
        scores_per_image = anchor_scores[batch_index]
        classes_per_image = anchor_classes[batch_index]
        valid = (
            torch.isfinite(boxes_per_image).all(dim=1)
            & torch.isfinite(scores_per_image)
            & (scores_per_image >= score_thresh)
            & (boxes_per_image[:, 2] > boxes_per_image[:, 0])
            & (boxes_per_image[:, 3] > boxes_per_image[:, 1])
        )
        candidate_indices = torch.where(valid)[0]
        if candidate_indices.numel() == 0:
            results.append(_empty_detection())
            continue

        if candidate_indices.numel() > pre_nms_topk:
            candidate_scores = scores_per_image[candidate_indices]
            _, order = torch.topk(candidate_scores, k=pre_nms_topk, sorted=False)
            candidate_indices = candidate_indices[order]

        candidate_boxes = boxes_per_image[candidate_indices]
        candidate_scores = scores_per_image[candidate_indices]
        candidate_classes = classes_per_image[candidate_indices]
        keep = batched_nms(
            candidate_boxes, candidate_scores, candidate_classes, nms_thresh
        )
        keep = keep[:topk]

        results.append(
            {
                "rois": candidate_boxes[keep].detach().float().cpu().numpy(),
                "scores": candidate_scores[keep].detach().float().cpu().numpy(),
                "class_ids": candidate_classes[keep].detach().long().cpu().numpy(),
            }
        )

    return results


__all__ = [
    "build_original_trivnet",
    "OriginalTriVNetODLoss",
    "decode_original_detections",
]
