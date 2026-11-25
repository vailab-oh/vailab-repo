"""SLZ segmentation loss.

One loss for every model. SLZ labels:

    0   = Unsafe
    1   = Safe
    255 = Ignore  -> excluded from the loss via ``ignore_index``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

SLZ_IGNORE_INDEX = 255


def slz_class_weights(
    safe_weight: float = 1.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Per-class CE weights ``[unsafe=1.0, safe=safe_weight]``."""
    return torch.tensor(
        [1.0, float(safe_weight)],
        dtype=torch.float32,
        device=device,
    )


def slz_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    ignore_index: int = SLZ_IGNORE_INDEX,
) -> torch.Tensor:
    """Cross-entropy for SLZ logits ``[B, 2, H, W]`` vs target ``[B, H, W]`` in
    ``{0, 1, 255}``: weighted ``F.cross_entropy`` with ``ignore_index``, so ``255``
    pixels contribute no loss.
    """
    if logits.shape[-2:] != target.shape[-2:]:
        raise ValueError(
            f"logits spatial {tuple(logits.shape[-2:])} != target spatial "
            f"{tuple(target.shape[-2:])}; the adapter must resize logits to the target."
        )
    return F.cross_entropy(
        logits,
        target.long(),
        weight=class_weights,
        ignore_index=ignore_index,
    )
