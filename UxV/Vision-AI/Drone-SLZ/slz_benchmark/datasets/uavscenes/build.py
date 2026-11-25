from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from torch.utils.data import DataLoader

from .dataset import UAVScenesDataset
from .splits import manifest_path as _manifest_path
from .transforms import (
    ColorJitter,
    Compose,
    NormalizeImage,
    RandomGaussianBlur,
    RandomHorizontalFlip,
    Resize,
)


def resolve_manifest(
    root: str | Path,
    split: str,
    scheme: str = "fixed",
    fold: Optional[int] = None,
) -> Path:
    """Thin delegate to :func:`slz_benchmark.datasets.uavscenes.splits.manifest_path`
    (single source of truth for the scheme -> path mapping)."""
    return _manifest_path(root, split, scheme=scheme, fold=fold)


def build_transform(
    train: bool,
    image_size: Optional[tuple[int, int]] = None,
    hflip_prob: float = 0.5,
    jitter_prob: float = 0.0,
    blur_prob: float = 0.0,
    normalize: bool = True,
):
    ts = []

    if image_size is not None:
        ts.append(Resize(image_size))

    if train and hflip_prob > 0:
        ts.append(RandomHorizontalFlip(hflip_prob))

    # Photometric-only (image-only, no mask/geometry sync) — see transforms.py.
    if train and jitter_prob > 0:
        ts.append(ColorJitter(p=jitter_prob))
    if train and blur_prob > 0:
        ts.append(RandomGaussianBlur(p=blur_prob))

    if normalize:
        ts.append(NormalizeImage())

    return Compose(ts) if ts else None


def build_dataset(
    root: str | Path,
    split: str,
    tasks: Sequence[str],
    scheme: str = "fixed",
    fold: Optional[int] = None,
    image_size: Optional[tuple[int, int]] = None,
    train: Optional[bool] = None,
    hflip_prob: float = 0.0,
    jitter_prob: float = 0.0,
    blur_prob: float = 0.0,
    normalize: bool = True,
    load_pose: bool = False,
    strict: bool = True,
    slz_encoding: str = "class_0_1_255",
    manifest: Optional[str | Path] = None,
):
    if train is None:
        train = split == "train"

    # An explicit manifest path (evaluation only) bypasses scheme/fold.
    if manifest is not None:
        manifest = Path(manifest)
        if not manifest.is_absolute():
            manifest = Path(root) / manifest
        if not manifest.exists():
            raise FileNotFoundError(manifest)
    else:
        manifest = resolve_manifest(root, split, scheme=scheme, fold=fold)
    transform = build_transform(
        train=train,
        image_size=image_size,
        hflip_prob=hflip_prob,
        jitter_prob=jitter_prob,
        blur_prob=blur_prob,
        normalize=normalize,
    )

    return UAVScenesDataset(
        root=root,
        manifest=manifest,
        tasks=tasks,
        transform=transform,
        load_pose=load_pose,
        align_to_geometry=True,
        strict=strict,
        slz_encoding=slz_encoding,
    )


def build_dataloader(
    dataset,
    batch_size: int = 4,
    shuffle: Optional[bool] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
):
    if shuffle is None:
        # Use manifest path rather than dataset internals beyond this.
        shuffle = "train.csv" in str(dataset.manifest)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )
