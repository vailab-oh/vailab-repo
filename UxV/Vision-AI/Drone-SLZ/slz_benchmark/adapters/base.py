"""Model adapter contract.

Every segmentation model is used through an adapter so the dataset, loss, metric,
trainer and evaluator stay model-agnostic. The contract, deliberately small (no deep
class tree):

  - ``build_model(device, **cfg)``               -> ``nn.Module``
  - ``forward(model, images, out_size)``         -> semantic scores/logits ``[B, 2, H, W]``
    (channel 0 = Unsafe, 1 = Safe), already resized to ``out_size = (H, W)``; consumed by
    the common evaluator's ``argmax`` + confusion matrix
  - ``compute_train_loss(model, batch, device, *, class_weights)`` -> ``(loss, logits)``;
    the model's **training objective** (default = pixel-wise CE with ``ignore_index=255``;
    an adapter may override this with its own objective). ``logits`` (or
    ``None``) is only used for the running train-time confusion matrix.
  - ``loss_kind`` -> str tag identifying what ``compute_train_loss`` returns, so the
    reported "loss" is never cross-compared between models with different objectives.
  - ``save(model, out_dir)`` / ``load(ckpt_dir, device)``  -> checkpoint round-trip
  - ``input_normalization()``                    -> ``(mean, std)`` the backbone expects

Nothing downstream of the adapter ever sees a model-specific output shape, key,
checkpoint format, or training objective.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..losses import SLZ_IGNORE_INDEX, slz_cross_entropy

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SegAdapter(ABC):
    #: model family name (recorded in checkpoint metadata); the ``--model`` keys are in
    #: ``slz_benchmark.adapters.MODELS``
    name: str = ""
    #: SLZ is binary (0 = Unsafe, 1 = Safe); 255 = Ignore is handled by the loss/metric
    num_classes: int = 2
    #: concrete identifier the adapter builds by default (HF id, torchvision entry, …).
    #: Used to resolve a non-null ``model_name`` for the experiment config / metadata.
    default_model_name: str = ""
    #: tag for the value returned by ``compute_train_loss`` (recorded in metrics/config).
    #: Dense models: pixel-wise CE with ignore. Never compare "loss" across kinds.
    loss_kind: str = "pixel_ce_ignore255"

    def resolve_model_name(self, model_name: Optional[str]) -> str:
        """CLI / config / default -> the concrete model name actually used.

        ``None`` is only an intermediate state; the returned value is always a
        non-empty string and is what gets recorded in ``config.json`` and checkpoint
        metadata.
        """
        return model_name or self.default_model_name or self.name

    @abstractmethod
    def build_model(self, device: torch.device, **cfg) -> nn.Module:
        ...

    @abstractmethod
    def forward(
        self,
        model: nn.Module,
        images: torch.Tensor,
        out_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Return logits ``[B, num_classes, out_size[0], out_size[1]]``."""

    @abstractmethod
    def save(self, model: nn.Module, out_dir: str | Path) -> None:
        ...

    @abstractmethod
    def load(self, ckpt_dir: str | Path, device: torch.device) -> nn.Module:
        ...

    def input_normalization(self) -> Tuple[Sequence[float], Sequence[float]]:
        return IMAGENET_MEAN, IMAGENET_STD

    # -- training objective ---------------------------------------------------
    def compute_train_loss(
        self,
        model: nn.Module,
        batch: dict,
        device: torch.device,
        *,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """The model's training objective on one batch.

        Default (dense semantic models): ``forward`` -> semantic logits ``[B, 2, H, W]``
        -> ``slz_cross_entropy(ignore_index=255)``.

        Returns ``(loss, logits)``. ``logits`` (``[B, 2, H, W]`` or ``None``) is used
        only for the running confusion matrix; the common evaluator always re-derives
        the semantic prediction it scores from ``forward`` / this method.
        """
        images = batch["image"].to(device, non_blocking=True)
        target = batch["slz"].to(device, non_blocking=True)
        logits = self.forward(model, images, tuple(target.shape[-2:]))
        loss = slz_cross_entropy(
            logits.float(),
            target,
            class_weights=class_weights,
            ignore_index=SLZ_IGNORE_INDEX,
        )
        return loss, logits

    # -- optional debugging extension ------------------------------------------
    def loss_breakdown(self, model: nn.Module) -> dict:
        """Optional per-term breakdown of the loss ``compute_train_loss`` just
        returned, for debugging one model. Default: none. The trainer adds a
        ``loss_breakdown`` key to the epoch metrics only when it is non-empty.
        """
        return {}

    # -- shared helper ----------------------------------------------------------
    @staticmethod
    def _resize_logits(logits: torch.Tensor, out_size: Tuple[int, int]) -> torch.Tensor:
        """Bilinear upsample logits to ``out_size`` (``align_corners=False``)."""
        if tuple(logits.shape[-2:]) == tuple(out_size):
            return logits
        return torch.nn.functional.interpolate(
            logits,
            size=tuple(out_size),
            mode="bilinear",
            align_corners=False,
        )
