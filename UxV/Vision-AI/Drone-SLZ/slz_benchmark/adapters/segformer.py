"""HuggingFace SegFormer adapter (B0 / B3 / B5).

  - ``SegformerForSemanticSegmentation.from_pretrained(model_name, num_labels=2,
    id2label/label2id, ignore_mismatched_sizes=True)``
  - forward via ``model(pixel_values=...).logits`` (``H/4 x W/4``), then bilinear
    upsample to the target size
  - checkpoint via HF ``save_pretrained`` / ``from_pretrained``
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

from .base import SegAdapter

DEFAULT_MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"

_ID2LABEL = {0: "unsafe", 1: "safe"}
_LABEL2ID = {"unsafe": 0, "safe": 1}


class SegFormerAdapter(SegAdapter):
    name = "segformer"
    default_model_name = DEFAULT_MODEL_NAME

    def build_model(
        self,
        device: torch.device,
        model_name: str | None = None,
        **_ignored,
    ) -> nn.Module:
        from transformers import SegformerForSemanticSegmentation

        model_name = model_name or self.default_model_name
        model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=self.num_classes,
            id2label=_ID2LABEL,
            label2id=_LABEL2ID,
            ignore_mismatched_sizes=True,
        )
        return model.to(device)

    def forward(
        self,
        model: nn.Module,
        images: torch.Tensor,
        out_size: Tuple[int, int],
    ) -> torch.Tensor:
        logits = model(pixel_values=images).logits  # [B, 2, H/4, W/4]
        return self._resize_logits(logits, out_size)  # [B, 2, H, W]

    def save(self, model: nn.Module, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir)

    def load(self, ckpt_dir: str | Path, device: torch.device) -> nn.Module:
        from transformers import SegformerForSemanticSegmentation

        model = SegformerForSemanticSegmentation.from_pretrained(str(ckpt_dir))
        return model.to(device)
