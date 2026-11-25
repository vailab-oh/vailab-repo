"""Dataset / DataLoader construction shared by the train and eval entry points."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from .adapters.base import SegAdapter
from .config import RunConfig
from .datasets.uavscenes import build_dataset
from .datasets.uavscenes.transforms import NormalizeImage


def make_loader(
    cfg: RunConfig,
    split: str,
    *,
    adapter: SegAdapter | None = None,
    shuffle: bool | None = None,
    drop_last: bool = False,
    generator: torch.Generator | None = None,
):
    """Build the ``(dataset, DataLoader)`` for one split, with the shared transforms
    and the adapter's input normalization.

    ``generator`` (optional): an explicit RNG for the training-shuffle order, so the
    batch order does not depend on how much global RNG other code has consumed.
    No ``worker_init_fn``: PyTorch already seeds each worker.
    """
    is_train = split == "train"
    if shuffle is None:
        shuffle = is_train

    ds = build_dataset(
        root=cfg.root,
        split=split,
        scheme=cfg.scheme,
        fold=cfg.fold,
        manifest=cfg.manifest,
        tasks=["slz"],
        image_size=cfg.image_size,
        train=is_train,
        hflip_prob=cfg.hflip_prob if is_train else 0.0,
        jitter_prob=cfg.jitter_prob if is_train else 0.0,
        blur_prob=cfg.blur_prob if is_train else 0.0,
        normalize=True,
        load_pose=False,
        strict=True,
        slz_encoding=cfg.slz_encoding,
    )

    # Honor the adapter's normalization if it differs from the ImageNet default.
    if adapter is not None and ds.transform is not None:
        mean, std = adapter.input_normalization()
        for i, t in enumerate(ds.transform.transforms):
            if isinstance(t, NormalizeImage):
                ds.transform.transforms[i] = NormalizeImage(mean=mean, std=std)

    # DDP: only the train split is sharded. val/test stay whole on every rank; the
    # Trainer only iterates them on rank 0. `cfg.batch_size` is already the per-process
    # batch here (train.py divides the global batch by world_size).
    sampler = None
    if is_train and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        sampler = DistributedSampler(ds, shuffle=shuffle, seed=cfg.seed)
        shuffle = False  # DataLoader forbids shuffle=True together with a sampler

    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=(cfg.num_workers > 0),
        generator=generator,
    )
    return ds, loader


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
