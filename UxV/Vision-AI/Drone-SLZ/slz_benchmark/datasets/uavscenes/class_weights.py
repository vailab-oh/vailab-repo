"""Per-fold SLZ class statistics and the inverse-frequency class weight.

Every model is trained with cross-entropy weighted by

    w_unsafe = 1.0
    w_safe   = N_unsafe / N_safe

where ``N_unsafe`` / ``N_safe`` count **only the labeled training pixels** (value 0 /
value 1) over the frames in that fold's ``train.csv``. Ignore (255) pixels are not
counted; validation and test pixels are never used. No free parameter: it is fixed by
the split.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

SLZ_IGNORE_INDEX = 255


def _resolve(root: Path, value: str) -> Path:
    p = Path(str(value).strip())
    return p if p.is_absolute() else root / p


def count_slz_pixels(root: str | Path, manifest_path: str | Path) -> dict:
    """Count SLZ label pixels over every frame listed in ``manifest_path``.

    Returns ``{n_unsafe, n_safe, n_ignore, n_other, n_frames}`` — exact integer counts
    over the raw label PNGs (no resize, no transform), so the result is a pure function
    of the manifest + on-disk labels.
    """
    root = Path(root)
    manifest_path = Path(manifest_path)
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    n_unsafe = n_safe = n_ignore = n_other = 0
    for row in rows:
        slz_path = _resolve(root, row["slz_path"])
        with Image.open(slz_path) as im:
            arr = np.asarray(im)
        counts = np.bincount(arr.reshape(-1), minlength=256)
        n_unsafe += int(counts[0])
        n_safe += int(counts[1])
        n_ignore += int(counts[SLZ_IGNORE_INDEX])
        n_other += int(arr.size - counts[0] - counts[1] - counts[SLZ_IGNORE_INDEX])

    return {
        "n_unsafe": n_unsafe,
        "n_safe": n_safe,
        "n_ignore": n_ignore,
        "n_other": n_other,
        "n_frames": len(rows),
    }


def fold_class_weight(root: str | Path, train_manifest: str | Path) -> dict:
    """`count_slz_pixels` + the inverse-frequency ``w_safe``."""
    c = count_slz_pixels(root, train_manifest)
    if c["n_other"]:
        raise ValueError(
            f"{train_manifest}: found {c['n_other']} label pixels outside {{0,1,255}}"
        )
    if c["n_safe"] == 0:
        raise ValueError(f"{train_manifest}: no Safe (value 1) pixels — cannot weight")
    w_safe = c["n_unsafe"] / c["n_safe"]
    labeled = c["n_unsafe"] + c["n_safe"]
    return {
        **c,
        "safe_fraction_labeled": c["n_safe"] / labeled,
        "w_unsafe": 1.0,
        "w_safe": w_safe,
    }
