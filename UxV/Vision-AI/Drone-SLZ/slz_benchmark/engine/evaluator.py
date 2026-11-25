"""The one SLZ evaluation path, shared by train-time validation and ``evaluate.py``.

Metrics flow only through ``update_confusion_matrix`` / ``binary_segmentation_metrics``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from tqdm.auto import tqdm

from ..adapters.base import SegAdapter
from ..metrics import binary_segmentation_metrics, update_confusion_matrix
from ..losses import SLZ_IGNORE_INDEX

METRIC_KEYS = (
    "iou_unsafe",
    "iou_safe",
    "miou",
    "precision_safe",
    "recall_safe",
    "f1_safe",
)


@torch.inference_mode()
def evaluate(
    adapter: SegAdapter,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    amp: bool = True,
    class_weights: Optional[torch.Tensor] = None,
    max_steps: Optional[int] = None,
    desc: str = "Eval",
) -> Tuple[dict, torch.Tensor]:
    """Return ``(metrics, confusion_matrix)``.

    ``metrics`` always has the six SLZ metric keys (:data:`METRIC_KEYS`), plus
    ``pixel_acc``, ``loss`` and ``loss_kind``. ``255`` (Ignore) is excluded from the
    confusion matrix and therefore from every metric.

    The reported ``loss`` is each model's **own objective** evaluated on this split
    (via ``adapter.compute_train_loss`` — no grad, since ``evaluate`` runs under
    ``inference_mode``). ``loss_kind`` tags what it is. It is a per-model diagnostic:
    **never compare ``loss`` across models** — checkpoint selection and model
    comparison use ``miou`` / the SLZ metrics.
    """
    model.eval()
    amp_enabled = bool(amp) and device.type == "cuda"

    cm = torch.zeros((2, 2), dtype=torch.int64, device=device)
    running_loss = 0.0
    seen = 0
    breakdown_sums: dict[str, float] = {}

    pbar = tqdm(loader, desc=desc, dynamic_ncols=True, leave=True)
    for step, batch in enumerate(pbar):
        if max_steps is not None and step >= max_steps:
            break

        target = batch["slz"].to(device, non_blocking=True)
        bsz = target.shape[0]

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            loss, logits = adapter.compute_train_loss(
                model, batch, device, class_weights=class_weights
            )
            if logits is None:
                logits = adapter.forward(
                    model,
                    batch["image"].to(device, non_blocking=True),
                    tuple(target.shape[-2:]),
                )

        if torch.isfinite(loss):
            running_loss += float(loss.item()) * bsz
            seen += bsz
            for k, v in adapter.loss_breakdown(model).items():
                breakdown_sums[k] = breakdown_sums.get(k, 0.0) + v * bsz

        pred = logits.argmax(dim=1)
        update_confusion_matrix(cm, pred, target, ignore_index=SLZ_IGNORE_INDEX)
        pbar.set_postfix(loss=f"{float(loss.item()):.4f}")

    metrics = binary_segmentation_metrics(cm)
    metrics["loss"] = running_loss / max(seen, 1)
    metrics["loss_kind"] = adapter.loss_kind
    if breakdown_sums:
        metrics["loss_breakdown"] = {
            k: v / max(seen, 1) for k, v in breakdown_sums.items()
        }
    return metrics, cm.cpu()


def format_metrics(prefix: str, m: dict) -> str:
    """One-line metric summary."""
    kind = m.get("loss_kind")
    loss_tag = f" [{kind}]" if kind else ""
    return (
        f"{prefix} "
        f"loss={m.get('loss', float('nan')):.4f}{loss_tag} | "
        f"mIoU={m['miou']:.4f} | "
        f"IoU_safe={m['iou_safe']:.4f} | "
        f"IoU_unsafe={m['iou_unsafe']:.4f} | "
        f"P_safe={m['precision_safe']:.4f} | "
        f"R_safe={m['recall_safe']:.4f} | "
        f"F1_safe={m['f1_safe']:.4f} | "
        f"Acc={m['pixel_acc']:.4f}"
    )
