"""Generator configuration and constants (label values, hazard IDs, family splits).

Pure data: imports nothing from the other ``generation`` modules. The defaults are the
paper settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Tuple

# -- SLZ target encoding (unchanged) ----------------------------------------------
#   0   = Unsafe
#   1   = Safe
#   255 = Ignore / unknown (insufficient geometry support)
UNSAFE_LABEL = 0
SAFE_LABEL = 1
IGNORE_LABEL = 255

# -- semantic hazard blacklist (unchanged) -----------------------------------------
# Only explicitly hazardous semantic classes veto an otherwise geometry-safe SLZ
# candidate. Unknown/background/unlisted classes are NOT removed by semantics;
# slope/roughness/footprint still decide them.
DEFAULT_HAZARD_IDS = [
    4,   # river
    5,   # pool
    11,  # traffic_barrier
    15,  # solar_board
    16,  # umbrella
    17,  # transparent_roof
    20,  # sedan
    24,  # truck
]

# -- physical scene families / default scenes / splits (unchanged) ----------------
PHYSICAL_FAMILIES = ["AMtown", "AMvalley", "HKairport", "HKisland"]

DEFAULT_SCENES = [
    "AMtown03",
    "AMvalley03",
    "HKairport03",
    "HKairport_GNSS_Evening",
    "HKisland03",
    "HKisland_GNSS_Evening",
]

FIXED_FAMILY_SPLIT = {
    "train": ["AMtown", "AMvalley"],
    "val": ["HKairport"],
    "test": ["HKisland"],
}

FOUR_FOLD_FAMILY_SPLITS = {
    "fold_1": {
        "train": ["HKairport", "HKisland"],
        "val": ["AMvalley"],
        "test": ["AMtown"],
    },
    "fold_2": {
        "train": ["AMtown", "HKisland"],
        "val": ["HKairport"],
        "test": ["AMvalley"],
    },
    "fold_3": {
        "train": ["AMtown", "AMvalley"],
        "val": ["HKisland"],
        "test": ["HKairport"],
    },
    "fold_4": {
        "train": ["AMvalley", "HKairport"],
        "val": ["AMtown"],
        "test": ["HKisland"],
    },
}


@dataclass(frozen=True)
class GenerationConfig:
    """Every setting the generator stages need. ``run.py`` fills each field from the
    command line; the defaults are the paper settings, the same as the CLI defaults.
    """

    # -- paths (resolved by the CLI before constructing this object) --------------
    root: Path
    raw_dir: Path
    mesh_root: Path
    dataset_name: str
    intervals: Tuple[str, ...] = ("interval5",)

    # -- scene / frame selection ---------------------------------------------------
    scenes: Tuple[str, ...] = tuple(DEFAULT_SCENES)
    frames_per_scene: Optional[int] = None          # None -> ALL frames per scene
    sample_seed: int = 42

    # -- output size  (H, W) = (640, 768);  W x H = 768 x 640 ------------------------
    height: int = 640
    width: int = 768

    # -- PCA geometry ----------------------------------------------------------------
    pca_radius_m: float = 0.50
    min_support: float = 0.50
    min_points: int = 9
    max_metric_radius_px: int = 512
    row_chunk: int = 128
    up_axis: str = "z"

    # -- SLZ thresholds --------------------------------------------------------------
    slope_threshold_deg: float = 6.0
    roughness_threshold_m: float = 0.10
    hazard_ids: Tuple[int, ...] = tuple(DEFAULT_HAZARD_IDS)

    # -- footprint -------------------------------------------------------------------
    with_footprint: bool = True
    uav_radius_m: float = 0.35
    safety_buffer_m: float = 0.15
    footprint_border_as_unsafe: bool = False

    # -- dataset assembly -------------------------------------------------------------
    asset_mode: str = "symlink"          # symlink | hardlink | copy
    overwrite: bool = False
    keep_work: bool = False

    @property
    def image_size_hw(self) -> Tuple[int, int]:
        return (self.height, self.width)
