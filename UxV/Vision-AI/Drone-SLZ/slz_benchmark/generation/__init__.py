"""SLZ pseudo-label generator for UAVScenes (metric landable region).

    run.py            command line and the order of the stages (no algorithm code)
    config.py         GenerationConfig and constants
    render/           raw frames + terra mesh -> RGB-aligned depth / normal
                      (raw_scenes, mesh, renderer, depth_normal)
    geometry/         intrinsics, backprojection, local PCA -> slope / roughness
                      (projection, pca)
    slz/              semantic veto, safety buffer + UAV footprint, labelling policy
                      (semantic, footprint, label)
    export/           prepared-dataset layout, family splits, dataset_config.yaml
                      (layout, manifest)
    visualization.py  SLZ colorization and diagnostic figures

Importing this package pulls in numpy / pandas / cv2 (and, lazily, scipy / numba); the
training and evaluation code never imports it. Labels: 0 = Unsafe, 1 = Safe, 255 = Ignore.
"""

from __future__ import annotations

from .config import (
    DEFAULT_HAZARD_IDS,
    IGNORE_LABEL,
    SAFE_LABEL,
    UNSAFE_LABEL,
    GenerationConfig,
)
from .geometry.pca import pca_geometry
from .geometry.projection import backproject, scaled_K
from .slz.footprint import (
    metric_clearance_from_blocked,
    projected_radii_xy,
    reconstruct_landable_region,
)
from .slz.label import build_slz_label, run_slz_generation
from .slz.semantic import semantic_safe
from .render.depth_normal import run_preprocessing
from .render.renderer import render_mesh, vertex_normals
from .export.layout import organize_prepared_dataset
from .export.manifest import generate_splits, write_dataset_metadata

__all__ = [
    "GenerationConfig",
    "DEFAULT_HAZARD_IDS",
    "UNSAFE_LABEL",
    "SAFE_LABEL",
    "IGNORE_LABEL",
    "pca_geometry",
    "backproject",
    "scaled_K",
    "semantic_safe",
    "projected_radii_xy",
    "metric_clearance_from_blocked",
    "reconstruct_landable_region",
    "render_mesh",
    "vertex_normals",
    "run_preprocessing",
    "run_slz_generation",
    "build_slz_label",
    "organize_prepared_dataset",
    "generate_splits",
    "write_dataset_metadata",
]
