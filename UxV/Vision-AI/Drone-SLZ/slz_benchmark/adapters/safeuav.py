"""Official SafeUAV-Net adapter: Small (``HVO_small.pkl``) and Large (``HVO_large.pkl``).

The official architectures (ported in ``_safeuav_net.py``) and pretrained checkpoints
(3-class Horizontal / Vertical / Other) are reused. Only the final classifier layer is
replaced to go from 3-class HVO to binary SLZ; every other pretrained tensor is
transplanted unchanged (``_safeuav_checkpoint.py``). The head is a plain dense per-pixel
classifier, so the default loss (``slz_cross_entropy(ignore_index=255)``) with the same
per-fold Safe class weight as the other families applies.

Preprocessing: the official inference scripts scale each frame to ~[0, 1] (min-max), not
ImageNet mean / std, so ``input_normalization`` is a pass-through (mean 0, std 1) of the
loader's [0, 1] image tensor.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ._safeuav_checkpoint import (
    ARCHITECTURE_BY_CHECKPOINT_NAME,
    EXPECTED_PARAM_COUNT,
    EXPECTED_PARAM_COUNT_LARGE,
    EXPECTED_TENSOR_COUNT,
    EXPECTED_TENSOR_COUNT_LARGE,
    _build_reference_model,
    build_binary_state_dict,
    load_pretrained_hvo3_state_dict,
)
from .base import SegAdapter

_REPO_ROOT = Path(__file__).resolve().parents[2]
#: Official checkpoint paths, relative to the repo root (download: README). The
#: ``safeuav_net_small`` / ``safeuav_net_large`` model keys select one of them.
DEFAULT_CHECKPOINT = "external/safeuav/checkpoints/HVO_small.pkl"
LARGE_CHECKPOINT = "external/safeuav/checkpoints/HVO_large.pkl"
_D_IN = 3
_EXPECTED_BY_ARCH = {
    "tiny_sum": (EXPECTED_TENSOR_COUNT, EXPECTED_PARAM_COUNT),
    "dilated_conv": (EXPECTED_TENSOR_COUNT_LARGE, EXPECTED_PARAM_COUNT_LARGE),
}


class SafeUAVAdapter(SegAdapter):
    name = "safeuav_net"
    default_model_name = DEFAULT_CHECKPOINT
    # Dense per-pixel classifier, same objective family as SegFormer/DeepLabV3/UPerNet.
    loss_kind = "pixel_ce_ignore255"

    def __init__(self) -> None:
        super().__init__()
        #: populated by build_model(); {"missing": [...], "unexpected": [...]} from the
        #: strict=False binary-head transplant, for smoke-test verification.
        self.last_load_report: Optional[dict] = None

    # -- checkpoint path resolution -------------------------------------------------
    @staticmethod
    def _resolve_checkpoint_path(model_name: Optional[str]) -> Path:
        raw = model_name or DEFAULT_CHECKPOINT
        p = Path(raw)
        if p.exists():
            return p
        p2 = _REPO_ROOT / raw
        if p2.exists():
            return p2
        raise FileNotFoundError(
            f"SafeUAV checkpoint not found at {p} or {p2}. Download the official "
            f"SafeUAV pretrained snapshots (HVO_small.pkl / HVO_large.pkl, see README "
            f"'SafeUAV weights'), put them in {_REPO_ROOT / 'external/safeuav/checkpoints'}, "
            f"or pass --model-name <path to the .pkl>."
        )

    # -- architecture / pretrained load ----------------------------------------------
    def build_model(
        self,
        device: torch.device,
        model_name: str | None = None,
        **_ignored,
    ) -> nn.Module:
        ckpt_path = self._resolve_checkpoint_path(model_name)
        arch_key, num_filters = ARCHITECTURE_BY_CHECKPOINT_NAME.get(
            ckpt_path.name, ("tiny_sum", 16)
        )
        hvo3_state = load_pretrained_hvo3_state_dict(
            ckpt_path, architecture=arch_key, num_filters=num_filters, d_in=_D_IN
        )
        binary_state = build_binary_state_dict(hvo3_state)

        model = _build_reference_model(arch_key, num_filters, d_in=_D_IN, d_out=self.num_classes)
        missing, unexpected = model.load_state_dict(binary_state, strict=False)
        expected_tensor_count, expected_param_count = _EXPECTED_BY_ARCH[arch_key]
        self.last_load_report = {
            "checkpoint": str(ckpt_path),
            "architecture": arch_key,
            "num_filters": num_filters,
            "missing": sorted(missing),
            "unexpected": sorted(unexpected),
            "expected_tensor_count": expected_tensor_count,
            "expected_param_count": expected_param_count,
        }
        # Both architectures name their output layer `finalConv` (see _safeuav_net.py),
        # so this check is unchanged regardless of which one was just built.
        expected_missing = {"finalConv.weight", "finalConv.bias"}
        if set(missing) != expected_missing or unexpected:
            raise RuntimeError(
                "SafeUAV binary-head transplant did not affect exactly "
                f"{expected_missing}: missing={missing}, unexpected={unexpected}"
            )
        self._last_checkpoint_path = str(ckpt_path)
        self._last_architecture = arch_key
        self._last_num_filters = num_filters
        return model.to(device)

    # -- inference --------------------------------------------------------------
    def forward(
        self,
        model: nn.Module,
        images: torch.Tensor,
        out_size: Tuple[int, int],
    ) -> torch.Tensor:
        # The ported architecture keeps the official code's own NHWC in/out convention
        # (see _safeuav_net.py); bridge to/from our common NCHW contract at the boundary
        # only, without touching the ported forward() itself.
        x = images.permute(0, 2, 3, 1).contiguous()  # [B,3,H,W] -> [B,H,W,3]
        logits = model(x)  # [B,H,W,2]
        logits = logits.permute(0, 3, 1, 2).contiguous()  # -> [B,2,H,W]
        return self._resize_logits(logits, out_size)

    # -- preprocessing ------------------------------------------------------------
    def input_normalization(self) -> Tuple[Sequence[float], Sequence[float]]:
        # No confirmed ImageNet-normalization convention for this pretrained checkpoint
        # (see module docstring) -- pass through the loader's [0,1]-scaled image as-is.
        return (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)

    # -- checkpoint (fine-tuned) ---------------------------------------------------
    _ARCH_DESCRIPTIONS = {
        "tiny_sum": "ModelUNetTinySum (ported from official SafeUAV, HVO_small.pkl)",
        "dilated_conv": "ModelUNetDilatedConv (ported from neural_wrappers v0.31, HVO_large.pkl)",
    }

    def save(self, model: nn.Module, out_dir: str | Path) -> None:
        # Falls back to the Small defaults if build_model() was never called on this
        # adapter instance (shouldn't happen in the normal train/save flow, but keeps
        # this method independently correct rather than assuming call order).
        arch_key = getattr(self, "_last_architecture", "tiny_sum")
        num_filters = getattr(self, "_last_num_filters", 16)
        pretrained_source = getattr(self, "_last_checkpoint_path", DEFAULT_CHECKPOINT)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_dir / "model.pt")
        (out_dir / "meta.json").write_text(
            json.dumps(
                {
                    "adapter": self.name,
                    "architecture": self._ARCH_DESCRIPTIONS[arch_key],
                    "architecture_key": arch_key,
                    "num_filters": num_filters,
                    "num_classes": self.num_classes,
                    "pretrained_source": pretrained_source,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def load(self, ckpt_dir: str | Path, device: torch.device) -> nn.Module:
        ckpt = Path(ckpt_dir)
        path = ckpt if ckpt.is_file() else ckpt / "model.pt"
        if not path.exists():
            raise FileNotFoundError(f"SafeUAV checkpoint not found: {path}")
        meta_path = ckpt / "meta.json" if ckpt.is_dir() else ckpt.parent / "meta.json"
        arch_key, num_filters = "tiny_sum", 16  # pre-Large-support checkpoints
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            arch_key = meta.get("architecture_key", arch_key)
            num_filters = meta.get("num_filters", num_filters)
        model = _build_reference_model(arch_key, num_filters, d_in=_D_IN, d_out=self.num_classes)
        state = torch.load(path, map_location=device)
        model.load_state_dict(state, strict=True)
        return model.to(device)
