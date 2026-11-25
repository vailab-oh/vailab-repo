"""Raw scene discovery, frame selection and source-path resolution."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import numpy as np


def scene_from_raw_dir(p: Path, interval: str) -> str:
    prefix = f"{interval}_"
    return p.name[len(prefix):] if p.name.startswith(prefix) else p.name


def find_scene_dirs(raw: Path, interval: str) -> List[Path]:
    """Find scene folders in either canonical UAVScenes layout or a flat extracted layout.

    Canonical examples:
      raw/interval1/CAM_LIDAR/interval1_AMtown02/...
      raw/interval5/CAM_LIDAR/interval5_CAM_LIDAR/interval5_AMtown02/...

    Flat extracted example (Windows/home test):
      raw/interval5_CAM_LIDAR/interval5_AMtown02/...
    """
    candidates = [
        raw / interval / "CAM_LIDAR",
        raw / f"{interval}_CAM_LIDAR",
    ]

    for base in candidates:
        if not base.exists():
            continue

        direct = sorted([
            p for p in base.iterdir()
            if p.is_dir()
            and p.name.startswith(f"{interval}_")
            and p.name != f"{interval}_CAM_LIDAR"
        ])
        if direct:
            return direct

        nested = base / f"{interval}_CAM_LIDAR"
        if nested.exists():
            scenes = sorted([
                p for p in nested.iterdir()
                if p.is_dir() and p.name.startswith(f"{interval}_")
            ])
            if scenes:
                return scenes

    return []


def cam_label_dir(raw: Path, interval: str, raw_scene: str, kind: str) -> Path:
    canonical = raw / interval / "CAM_LABEL" / f"{interval}_CAM_label" / raw_scene / f"{interval}_CAM_label_{kind}"
    if canonical.exists():
        return canonical
    return raw / f"{interval}_CAM_label" / raw_scene / f"{interval}_CAM_label_{kind}"


def scene_map_group(scene: str) -> str:
    """AMtown03 -> AMtown, HKisland_GNSS03 -> HKisland_GNSS.

    Regex ``0[1-9]|[1-9][0-9]$``, unlike ``render.mesh.mesh_group_from_scene`` and
    ``slz.label.scene_map_group`` (``0[1-3]$``). The three are kept as they are: the
    labels were generated with them.
    """
    m = re.match(r"^(.*?)(0[1-9]|[1-9][0-9])$", str(scene))
    return m.group(1) if m else str(scene)


def sample_frame_files(files: List[Path], n: int, mode: str, seed: int) -> List[Path]:
    files = sorted(files)
    if n is None or n <= 0 or len(files) <= n:
        return files

    if mode == "first":
        return files[:n]
    if mode == "random":
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(files), size=n, replace=False))
        return [files[int(i)] for i in idx]
    if mode == "uniform":
        # n <= len(files), so spacing is >= 1 and rounded indices stay unique.
        idx = np.rint(np.linspace(0, len(files) - 1, n)).astype(np.int64)
        return [files[int(i)] for i in idx]
    raise ValueError(mode)
