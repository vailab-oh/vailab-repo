"""Semantic hazard veto: pixels of the listed hazard classes are Unsafe."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from ..config import DEFAULT_HAZARD_IDS

__all__ = ["DEFAULT_HAZARD_IDS", "semantic_safe"]


def semantic_safe(path, shape, hazard_ids):
    """Return semantic veto mask: False only for explicit hazard IDs.

    Unknown/background/unlisted classes pass this semantic stage. If the
    semantic label image itself is missing, semantics performs no veto and the
    geometry + footprint criteria remain responsible for the decision.
    """
    H, W = shape
    if path is None or not path.exists():
        return np.ones((H,W), bool)
    lab = np.array(Image.open(path))
    if lab.ndim == 3:
        lab = lab[...,0]
    if lab.shape != (H,W):
        lab = cv2.resize(lab.astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST)
    bad = np.isin(lab, np.asarray(hazard_ids, dtype=lab.dtype))
    return ~bad
