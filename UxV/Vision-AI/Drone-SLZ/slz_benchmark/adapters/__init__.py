"""Model registry: one ``--model`` key per model, one adapter per model family.

    from slz_benchmark.adapters import get_adapter
    adapter = get_adapter("upernet_swin_l")

A new model is added as an adapter module that subclasses
:class:`~slz_benchmark.adapters.base.SegAdapter` and a row in ``MODELS``, never by
branching the trainer, dataset, metric or loss.
"""

from __future__ import annotations

from typing import Dict, Tuple, Type

from .base import SegAdapter
from .deeplabv3 import DeepLabV3Adapter
from .safeuav import SafeUAVAdapter
from .segformer import SegFormerAdapter
from .upernet import UperNetSwinAdapter

#: ``--model`` key -> (adapter, backbone it builds by default). ``--model-name``
#: overrides the backbone. The table in MODELS.md lists the same rows.
MODELS: Dict[str, Tuple[Type[SegAdapter], str]] = {
    "segformer_b0": (SegFormerAdapter, "nvidia/segformer-b0-finetuned-ade-512-512"),
    "segformer_b3": (SegFormerAdapter, "nvidia/segformer-b3-finetuned-ade-512-512"),
    "segformer_b5": (SegFormerAdapter, "nvidia/segformer-b5-finetuned-ade-640-640"),
    "deeplab_r50": (DeepLabV3Adapter, "deeplabv3_resnet50"),
    "deeplab_r101": (DeepLabV3Adapter, "deeplabv3_resnet101"),
    "upernet_swin_t": (UperNetSwinAdapter, "openmmlab/upernet-swin-tiny"),
    "upernet_swin_b": (UperNetSwinAdapter, "openmmlab/upernet-swin-base"),
    "upernet_swin_l": (UperNetSwinAdapter, "openmmlab/upernet-swin-large"),
    "safeuav_net_small": (SafeUAVAdapter, "external/safeuav/checkpoints/HVO_small.pkl"),
    "safeuav_net_large": (SafeUAVAdapter, "external/safeuav/checkpoints/HVO_large.pkl"),
}


def available() -> list[str]:
    return list(MODELS)


def get_adapter(model: str) -> SegAdapter:
    """The adapter of ``model``, with that key's backbone as its default."""
    try:
        cls, model_name = MODELS[model]
    except KeyError:
        raise ValueError(f"Unknown model {model!r}; available: {available()}") from None
    adapter = cls()
    adapter.default_model_name = model_name
    return adapter


__all__ = [
    "MODELS",
    "SegAdapter",
    "SegFormerAdapter",
    "DeepLabV3Adapter",
    "UperNetSwinAdapter",
    "SafeUAVAdapter",
    "get_adapter",
    "available",
]
