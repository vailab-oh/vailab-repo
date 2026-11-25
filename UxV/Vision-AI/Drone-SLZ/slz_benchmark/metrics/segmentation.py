from __future__ import annotations

from typing import Dict, Optional

import torch


@torch.no_grad()
def update_confusion_matrix(
    cm: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 2,
    ignore_index: Optional[int] = 255,
) -> torch.Tensor:
    """
    Update confusion matrix in-place.

    Convention:
        cm[row=true, col=pred]

    SLZ labels:
        0   = Unsafe
        1   = Safe
        255 = Ignore  -> excluded from the confusion matrix (and therefore from every
              downstream metric).

    Pixels are counted only where ``0 <= target < num_classes`` and, when
    ``ignore_index`` is set, ``target != ignore_index``. The out-of-range check alone
    already drops 255 for binary SLZ; ``ignore_index`` makes the exclusion explicit and
    also covers a non-default ignore value.
    """
    pred = pred.reshape(-1).long()
    target = target.reshape(-1).long()

    valid = (target >= 0) & (target < num_classes)
    if ignore_index is not None:
        valid &= target != ignore_index
    pred = pred[valid]
    target = target[valid]

    idx = target * num_classes + pred
    bins = torch.bincount(
        idx,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)

    cm += bins.to(cm.device)
    return cm


def binary_segmentation_metrics(cm: torch.Tensor) -> Dict[str, float]:
    """
    Metrics for the SLZ binary convention:
        0 = unsafe
        1 = safe
    """
    if tuple(cm.shape) != (2, 2):
        raise ValueError(f"Expected 2x2 confusion matrix, got {tuple(cm.shape)}")

    cm = cm.double()

    # class 0 = unsafe
    tp0 = cm[0, 0]
    fn0 = cm[0, 1]
    fp0 = cm[1, 0]

    # class 1 = safe
    tp1 = cm[1, 1]
    fn1 = cm[1, 0]
    fp1 = cm[0, 1]

    eps = 1e-12

    iou_unsafe = tp0 / (tp0 + fp0 + fn0 + eps)
    iou_safe = tp1 / (tp1 + fp1 + fn1 + eps)
    miou = 0.5 * (iou_unsafe + iou_safe)

    precision_safe = tp1 / (tp1 + fp1 + eps)
    recall_safe = tp1 / (tp1 + fn1 + eps)
    f1_safe = (
        2 * precision_safe * recall_safe
        / (precision_safe + recall_safe + eps)
    )

    pixel_acc = (tp0 + tp1) / (cm.sum() + eps)

    return {
        "pixel_acc": float(pixel_acc.item()),
        "iou_unsafe": float(iou_unsafe.item()),
        "iou_safe": float(iou_safe.item()),
        "miou": float(miou.item()),
        "precision_safe": float(precision_safe.item()),
        "recall_safe": float(recall_safe.item()),
        "f1_safe": float(f1_safe.item()),
    }
