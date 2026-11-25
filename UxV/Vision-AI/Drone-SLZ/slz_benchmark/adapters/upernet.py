"""HuggingFace UPerNet-Swin adapter (Swin-T / Swin-B / Swin-L).

Encoder–decoder semantic segmentation: Swin backbone + UPerNet (PPM + FPN) head.
All UPerNet-specific handling is here:

  - build from ``openmmlab/upernet-swin-{tiny,base,large}`` (Swin backbone pretrained
    on ImageNet, UPerNet head trained on ADE20K), reinit the classifier to 2 classes
  - **drop the auxiliary FCN head** (``auxiliary_head = None`` +
    ``config.use_auxiliary_head = False``) — the common trainer has no aux-loss support
    and must not gain one; only the main semantic output is used
  - forward returns ``model(pixel_values=...).logits`` — already upsampled to the input
    resolution by UPerNet, so ``_resize_logits`` is a no-op at (640, 768)
  - loss is the common ``slz_cross_entropy(ignore_index=255)`` applied by the trainer
  - checkpoint = HF ``save_pretrained`` directory (like SegFormer)
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

from .base import SegAdapter

DEFAULT_MODEL_NAME = "openmmlab/upernet-swin-tiny"


class UperNetSwinAdapter(SegAdapter):
    name = "upernet_swin"
    default_model_name = DEFAULT_MODEL_NAME

    def _disable_aux(self, model: nn.Module) -> nn.Module:
        model.auxiliary_head = None
        model.config.use_auxiliary_head = False
        return model

    def build_model(
        self,
        device: torch.device,
        model_name: str | None = None,
        **_ignored,
    ) -> nn.Module:
        from transformers import UperNetForSemanticSegmentation

        model_name = model_name or self.default_model_name
        model = UperNetForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=self.num_classes,
            ignore_mismatched_sizes=True,
        )
        return self._disable_aux(model).to(device)

    def forward(
        self,
        model: nn.Module,
        images: torch.Tensor,
        out_size: Tuple[int, int],
    ) -> torch.Tensor:
        logits = model(pixel_values=images).logits  # [B, 2, H_in, W_in]
        return self._resize_logits(logits, out_size)  # [B, 2, H, W]

    def save(self, model: nn.Module, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir)

    def load(self, ckpt_dir: str | Path, device: torch.device) -> nn.Module:
        from transformers import UperNetForSemanticSegmentation

        model = UperNetForSemanticSegmentation.from_pretrained(str(ckpt_dir))
        return self._disable_aux(model).to(device)
