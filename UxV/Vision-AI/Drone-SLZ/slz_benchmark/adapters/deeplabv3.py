"""torchvision DeepLabV3 adapter (ResNet-50 / ResNet-101 backbones).

All DeepLabV3-specific handling is here:
  - build from the torchvision ``DEFAULT`` pretrained weights for the chosen backbone
    (COCO/VOC pretrained: ASPP + decoder + backbone)
  - replace only the main classifier's final 1x1 conv with a 2-class conv
  - **remove the aux branch entirely** (``aux_classifier = None``) after loading the
    pretrained weights, so no aux tensor is ever forwarded (the common trainer has no
    aux-loss support and must not gain one)
  - ``forward`` returns ``model(images)["out"]`` -> ``[B, 2, H, W]`` (already at input
    resolution; ``_resize_logits`` is a no-op at (640, 768))
  - checkpoint = plain ``state_dict`` in ``<dir>/model.pt`` + ``meta.json`` recording
    which backbone it is; ``load`` reads that back and rebuilds the matching arch with
    ``weights=None, weights_backbone=None`` (no downloads), then ``strict`` loads
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

from .base import SegAdapter

_WEIGHTS_ID = "COCO_WITH_VOC_LABELS_V1"

#: torchvision constructor name -> (builder, pretrained-weights enum). Looked up by
#: `model_name` (falls back to resnet50, the original/default, for `None`/unknown).
#: Extend here, not by branching build_model/load, if another backbone is ever added.
def _backbone_table():
    from torchvision.models.segmentation import (
        DeepLabV3_ResNet50_Weights,
        DeepLabV3_ResNet101_Weights,
        deeplabv3_resnet50,
        deeplabv3_resnet101,
    )

    return {
        "deeplabv3_resnet50": (deeplabv3_resnet50, DeepLabV3_ResNet50_Weights.DEFAULT),
        "deeplabv3_resnet101": (deeplabv3_resnet101, DeepLabV3_ResNet101_Weights.DEFAULT),
    }


class DeepLabV3Adapter(SegAdapter):
    name = "deeplab"
    default_model_name = "deeplabv3_resnet50"

    # -- architecture ------------------------------------------------------------
    def _build_arch(self, *, pretrained: bool, backbone: str) -> nn.Module:
        table = _backbone_table()
        if backbone not in table:
            raise ValueError(
                f"unknown DeepLabV3 backbone {backbone!r}; choices: {sorted(table)}"
            )
        builder, weights = table[backbone]
        if pretrained:
            model = builder(weights=weights)
        else:
            # No downloads: the state_dict fully defines the model on load().
            model = builder(weights=None, weights_backbone=None)

        in_ch = model.classifier[-1].in_channels
        model.classifier[-1] = nn.Conv2d(in_ch, self.num_classes, kernel_size=1)
        model.aux_classifier = None
        return model

    def build_model(
        self, device: torch.device, model_name: str | None = None, **_ignored
    ) -> nn.Module:
        backbone = model_name or self.default_model_name
        model = self._build_arch(pretrained=True, backbone=backbone)
        self._last_backbone = backbone
        return model.to(device)

    def forward(
        self,
        model: nn.Module,
        images: torch.Tensor,
        out_size: Tuple[int, int],
    ) -> torch.Tensor:
        logits = model(images)["out"]  # [B, 2, H_in, W_in]
        return self._resize_logits(logits, out_size)  # [B, 2, H, W]

    # -- checkpoint -------------------------------------------------------------
    def save(self, model: nn.Module, out_dir: str | Path) -> None:
        backbone = getattr(self, "_last_backbone", self.default_model_name)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_dir / "model.pt")
        (out_dir / "meta.json").write_text(
            json.dumps(
                {
                    "adapter": self.name,
                    "torchvision_model": backbone,
                    "pretrained_weights": _WEIGHTS_ID,
                    "num_classes": self.num_classes,
                    "aux_classifier": None,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def load(self, ckpt_dir: str | Path, device: torch.device) -> nn.Module:
        ckpt = Path(ckpt_dir)
        path = ckpt if ckpt.is_file() else ckpt / "model.pt"
        if not path.exists():
            raise FileNotFoundError(f"DeepLabV3 checkpoint not found: {path}")
        meta_path = ckpt / "meta.json" if ckpt.is_dir() else ckpt.parent / "meta.json"
        backbone = self.default_model_name  # pre-backbone-table checkpoints: resnet50
        if meta_path.exists():
            backbone = json.loads(meta_path.read_text()).get("torchvision_model", backbone)
        model = self._build_arch(pretrained=False, backbone=backbone)
        state = torch.load(path, map_location=device)
        model.load_state_dict(state, strict=True)
        return model.to(device)
