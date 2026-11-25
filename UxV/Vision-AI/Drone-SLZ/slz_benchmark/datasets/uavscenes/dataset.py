from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


SUPPORTED_TASKS = {
    "semantic",
    "slz",
    "depth",
    "normal",
    "slope",
    "roughness",
}

TASK_TO_COLUMN = {
    "semantic": "semantic_path",
    "slz": "slz_path",
    "depth": "depth_path",
    "normal": "normal_path",
    "slope": "slope_path",
    "roughness": "roughness_path",
}


def _resolve(root: Path, value) -> Optional[Path]:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    p = Path(s)
    return p if p.is_absolute() else root / p


def _load_rgb(path: Path) -> torch.Tensor:
    # JPG -> float32 [3,H,W], range [0,1]
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)


def _load_label_png(path: Path, name: str) -> torch.Tensor:
    # Semantic/SLZ are expected to be single-channel ID/mask PNGs.
    with Image.open(path) as im:
        arr = np.asarray(im).copy()
    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be a single-channel PNG, got shape={arr.shape}: {path}"
        )
    return torch.from_numpy(arr.astype(np.int64, copy=False))


SLZ_IGNORE_INDEX = 255

SLZ_ENCODINGS = ("class_0_1_255", "binary_0_255", "binary_0_1")


def _load_slz(path: Path, encoding: str = "class_0_1_255") -> torch.Tensor:
    """Load an SLZ target PNG as int64 ``[H, W]``.

    SLZ labels: ``0 = Unsafe``, ``1 = Safe``, ``255 = Ignore``.

    ``encoding``:
      - ``class_0_1_255`` (default, this repo's generator): values must be a subset of
        ``{0, 1, 255}`` and are returned unchanged (``255`` kept as the ignore label).
      - ``binary_0_255`` (masks without an ignore label): values must be a subset of
        ``{0, 255}``; ``255`` maps to ``1`` (Safe). No ignore label.
      - ``binary_0_1``: values must be a subset of ``{0, 1}``; returned unchanged.
    """
    x = _load_label_png(path, "slz")
    unique = torch.unique(x)

    if encoding == "class_0_1_255":
        allowed = (unique == 0) | (unique == 1) | (unique == 255)
        if not torch.all(allowed):
            raise ValueError(
                f"SLZ encoding 'class_0_1_255' expects values in {{0,1,255}}, "
                f"found {unique.tolist()[:20]} in {path}"
            )
        return x.long()

    if encoding == "binary_0_255":
        allowed = (unique == 0) | (unique == 255)
        if not torch.all(allowed):
            raise ValueError(
                f"SLZ encoding 'binary_0_255' expects values in {{0,255}}, "
                f"found {unique.tolist()[:20]} in {path}"
            )
        return (x > 0).long()

    if encoding == "binary_0_1":
        allowed = (unique == 0) | (unique == 1)
        if not torch.all(allowed):
            raise ValueError(
                f"SLZ encoding 'binary_0_1' expects values in {{0,1}}, "
                f"found {unique.tolist()[:20]} in {path}"
            )
        return x.long()

    raise ValueError(
        f"Unknown slz_encoding={encoding!r}; expected one of {SLZ_ENCODINGS}"
    )


def _load_npy_float(path: Path) -> torch.Tensor:
    # Storage may be float16 or float32. Training interface is ALWAYS float32.
    arr = np.load(path, allow_pickle=False)
    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim == 2:
        # [H,W] -> [1,H,W]
        arr = arr[None, ...]
    elif arr.ndim == 3:
        # Generator normal is normally [H,W,3].
        if arr.shape[-1] == 3 and arr.shape[0] != 3:
            arr = np.transpose(arr, (2, 0, 1))
        elif arr.shape[0] in (1, 3):
            pass
        else:
            raise ValueError(f"Unsupported npy shape {arr.shape}: {path}")
    else:
        raise ValueError(f"Unsupported npy ndim={arr.ndim}: {path}")

    return torch.from_numpy(np.ascontiguousarray(arr))


def _resize_image(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(x.shape[-2:]) == tuple(size):
        return x
    return F.interpolate(
        x.unsqueeze(0),
        size=size,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(0)


def _resize_label(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(x.shape[-2:]) == tuple(size):
        return x
    dtype = x.dtype
    y = F.interpolate(
        x[None, None].float(),
        size=size,
        mode="nearest",
    )[0, 0]
    return y.to(dtype=dtype)


class UAVScenesDataset(Dataset):
    """
    Canonical UAVScenes dataset interface.

    Expected prepared manifest columns:
        family, scene, frame,
        rgb_path,
        depth_path,
        normal_path,
        semantic_path,
        slope_path,
        roughness_path,
        slz_path,
        pose_path

    Canonical returned tensors:
        image       float32 [3,H,W], normalized only if transform does it
        semantic    int64   [H,W]
        slz         int64   [H,W], 0=unsafe, 1=safe
        depth       float32 [1,H,W], meters
        normal      float32 [3,H,W], unit vector where valid
        slope       float32 [1,H,W], degrees
        roughness   float32 [1,H,W], meters

        depth_valid      bool [1,H,W]
        normal_valid     bool [1,H,W]
        geometry_valid   bool [1,H,W]

    Important:
    - Stored float16 arrays are cast to float32 at load time.
    - Float16 normals are renormalized after loading.
    - The current prepared generator can symlink original-resolution RGB/semantic
      while geometry targets are generated at a smaller target size. Therefore this
      dataset automatically aligns RGB/semantic to the geometry reference size.
    """

    def __init__(
        self,
        root: str | Path,
        manifest: str | Path,
        tasks: Sequence[str] = ("slz",),
        transform=None,
        load_pose: bool = False,
        align_to_geometry: bool = True,
        strict: bool = True,
        normal_eps: float = 1e-6,
        slz_encoding: str = "class_0_1_255",
    ):
        self.root = Path(root).expanduser().resolve()
        m = Path(manifest)
        self.manifest = m if m.is_absolute() else self.root / m
        self.tasks = tuple(tasks)
        self.transform = transform
        self.load_pose = bool(load_pose)
        self.align_to_geometry = bool(align_to_geometry)
        self.strict = bool(strict)
        self.normal_eps = float(normal_eps)
        if slz_encoding not in SLZ_ENCODINGS:
            raise ValueError(
                f"slz_encoding={slz_encoding!r} not in {SLZ_ENCODINGS}"
            )
        self.slz_encoding = slz_encoding

        unknown = set(self.tasks) - SUPPORTED_TASKS
        if unknown:
            raise ValueError(
                f"Unsupported tasks: {sorted(unknown)}. "
                f"Supported={sorted(SUPPORTED_TASKS)}"
            )

        if not self.manifest.exists():
            raise FileNotFoundError(self.manifest)

        self.df = pd.read_csv(self.manifest)
        if "rgb_path" not in self.df.columns:
            raise ValueError(f"'rgb_path' missing from manifest: {self.manifest}")

        for task in self.tasks:
            col = TASK_TO_COLUMN[task]
            if col not in self.df.columns and self.strict:
                raise ValueError(
                    f"Task '{task}' requires manifest column '{col}', "
                    f"but it is missing: {self.manifest}"
                )

    def __len__(self) -> int:
        return len(self.df)

    def _path(self, row, col: str, required: bool = True) -> Optional[Path]:
        if col not in row.index:
            if required:
                raise KeyError(f"Manifest column missing: {col}")
            return None
        p = _resolve(self.root, row[col])
        if p is None or not p.exists():
            if required:
                raise FileNotFoundError(
                    f"Missing '{col}' for scene={row.get('scene')} "
                    f"frame={row.get('frame')}: {p}"
                )
            return None
        return p

    def _load_pose(self, row) -> dict:
        p = self._path(row, "pose_path", required=self.strict)
        if p is None:
            return {}
        info = json.loads(p.read_text(encoding="utf-8"))
        K = torch.tensor(info["P3x3"], dtype=torch.float32)
        T = torch.tensor(info["T4x4"], dtype=torch.float32)

        return {
            "K": K,
            "T_cam_to_map": T,
            "_pose_width": int(info["Width"]),
            "_pose_height": int(info["Height"]),
        }

    @staticmethod
    def _reference_hw(sample: dict) -> tuple[int, int]:
        # Prefer metric geometry as canonical alignment reference.
        for key in ("depth", "normal", "slope", "roughness", "slz", "semantic", "image"):
            if key in sample:
                return int(sample[key].shape[-2]), int(sample[key].shape[-1])
        raise RuntimeError("No spatial tensor was loaded.")

    def _align(self, sample: dict) -> dict:
        h, w = self._reference_hw(sample)
        size = (h, w)

        if "image" in sample:
            sample["image"] = _resize_image(sample["image"], size)

        if "semantic" in sample:
            sample["semantic"] = _resize_label(sample["semantic"], size)

        if "slz" in sample:
            sample["slz"] = _resize_label(sample["slz"], size)

        # Geometry .npy targets should normally already be aligned.
        for key in ("depth", "normal", "slope", "roughness"):
            if key in sample and tuple(sample[key].shape[-2:]) != size:
                raise ValueError(
                    f"{key} is not aligned with geometry reference size {size}: "
                    f"got {tuple(sample[key].shape[-2:])}. "
                    "Do not silently resample pseudo-geometry at dataset-load stage."
                )

        # P3x3 in pose JSON is for the original camera image.
        if "K" in sample:
            src_w = float(sample.pop("_pose_width"))
            src_h = float(sample.pop("_pose_height"))
            sx = w / src_w
            sy = h / src_h

            K = sample["K"].clone()
            K[0, 0] *= sx
            K[0, 2] *= sx
            K[1, 1] *= sy
            K[1, 2] *= sy
            sample["K"] = K

        return sample

    def _build_valid_masks(self, sample: dict) -> None:
        if "depth" in sample:
            d = sample["depth"]
            sample["depth_valid"] = torch.isfinite(d) & (d > 0)

        if "normal" in sample:
            n = sample["normal"]
            finite = torch.isfinite(n).all(dim=0, keepdim=True)
            norm = torch.linalg.vector_norm(
                torch.nan_to_num(n, nan=0.0, posinf=0.0, neginf=0.0),
                dim=0,
                keepdim=True,
            )
            valid = finite & (norm > self.normal_eps)
            sample["normal_valid"] = valid

            # Source normal can be float16 on disk. Renormalize in float32.
            n = torch.nan_to_num(n, nan=0.0, posinf=0.0, neginf=0.0)
            n = torch.where(
                valid.expand_as(n),
                n / norm.clamp_min(self.normal_eps),
                torch.zeros_like(n),
            )
            sample["normal"] = n

        if "depth_valid" in sample and "normal_valid" in sample:
            sample["geometry_valid"] = (
                sample["depth_valid"] & sample["normal_valid"]
            )
        elif "depth_valid" in sample:
            sample["geometry_valid"] = sample["depth_valid"].clone()
        elif "normal_valid" in sample:
            sample["geometry_valid"] = sample["normal_valid"].clone()

    def __getitem__(self, index: int) -> dict:
        row = self.df.iloc[index]

        image_path = self._path(row, "rgb_path", required=True)
        sample = {
            "image": _load_rgb(image_path),
            "scene": str(row.get("scene", "")),
            "family": str(row.get("family", "")),
            "frame_id": str(row.get("frame", "")),
            "index": int(index),
        }

        for task in self.tasks:
            col = TASK_TO_COLUMN[task]
            p = self._path(row, col, required=self.strict)
            if p is None:
                continue

            if task == "semantic":
                sample["semantic"] = _load_label_png(p, "semantic").long()
            elif task == "slz":
                sample["slz"] = _load_slz(p, self.slz_encoding)
            else:
                sample[task] = _load_npy_float(p).float()

        if self.load_pose:
            sample.update(self._load_pose(row))

        if self.align_to_geometry:
            sample = self._align(sample)

        self._build_valid_masks(sample)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
