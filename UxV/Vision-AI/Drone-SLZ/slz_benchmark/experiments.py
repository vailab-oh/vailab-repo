"""Experiment settings of the paper.

Models, folds, seeds, the training recipe, the footprint-radius study and the output
layout. ``tools/run_experiments.py`` and the summary scripts read everything from here,
so no run is tuned on its own.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent

# -- models ----------------------------------------------------------------------------
# ``--model`` key of train.py -> (family label, size label). One row each in MODELS.md;
# the backbone each key builds is set in ``slz_benchmark.adapters.MODELS``.
MODELS = {
    "segformer_b0": ("SegFormer", "B0"),
    "segformer_b3": ("SegFormer", "B3"),
    "segformer_b5": ("SegFormer", "B5"),
    "deeplab_r50": ("DeepLabV3", "ResNet-50"),
    "deeplab_r101": ("DeepLabV3", "ResNet-101"),
    "upernet_swin_t": ("UPerNet", "Swin-T"),
    "upernet_swin_b": ("UPerNet", "Swin-B"),
    "upernet_swin_l": ("UPerNet", "Swin-L"),
    "safeuav_net_small": ("SafeUAV", "Small"),
    "safeuav_net_large": ("SafeUAV", "Large"),
}
#: The base size of each family: the main comparison and the radius study use these;
#: every other size is compared against the base size of its own family.
BASE_MODELS = ("segformer_b0", "deeplab_r50", "upernet_swin_t", "safeuav_net_small")


def family(model: str) -> str:
    return MODELS[model][0]


def size(model: str) -> str:
    return MODELS[model][1]


def label(model: str) -> str:
    """``"UPerNet Swin-L"``"""
    return f"{family(model)} {size(model)}"


def is_base(model: str) -> bool:
    return model in BASE_MODELS


def base_of(model: str) -> str:
    """The base-size key of ``model``'s family."""
    return next(b for b in BASE_MODELS if family(b) == family(model))


def queue_order() -> list[str]:
    """Training order: the four base sizes first, then the larger sizes by family."""
    return [*BASE_MODELS, *(m for m in MODELS if m not in BASE_MODELS)]


# -- splits ----------------------------------------------------------------------------
# Leave-one-sequence-out: fold N holds out FAMILIES[N-1]. Train/val are separated by a
# 30-frame temporal guard and a 5 m spatial guard (see README, "Install the splits").
SPLIT_SCHEME = "loso"
FAMILIES = ("AMtown", "AMvalley", "HKairport", "HKisland")
FOLDS = (1, 2, 3, 4)
SEEDS = (42, 1234, 2025)

# -- training recipe (identical for every model) ------------------------------------------
# GLOBAL_BATCH_SIZE is summed over all DDP ranks; train.py divides it by WORLD_SIZE, so the
# same value works with torchrun --nproc_per_node=1 (one 80 GB GPU: 8 per GPU) or =2 (two
# 24 GB GPUs: 4 per GPU).
GLOBAL_BATCH_SIZE = 8
NPROC_PER_NODE = 2
LR = 1e-4
WEIGHT_DECAY = 0.01
EPOCHS = 50
# image-only photometric augmentation, on top of RandomHorizontalFlip(p=0.5)
JITTER_PROB = 0.5
BLUR_PROB = 0.3
WARMUP_RATIO = 0.05
POLY_POWER = 0.9
GRAD_CLIP = 1.0

# -- UAV footprint-radius study ------------------------------------------------------------
# tools/make_radius_variant.py writes each label set next to the base dataset, as
# ``<base root>_<tag>``, with ONLY the UAV radius changed (PCA radius 0.50 m, safety buffer
# 0.15 m, thresholds and semantic policy unchanged). The base sizes are trained on it with
# the same recipe and one seed, and compared against the base r = 0.35 m runs.
# Entries: (tag, uav_radius_m, pca_radius_m), in queue order.
RADIUS_BASE_M = 0.35
RADIUS_BASE_PCA_M = 0.50
RADIUS_VARIANTS = (("r050", 0.50, 0.50), ("r020", 0.20, 0.50))
RADIUS_SEED = 42


def radius_variant_root(base_root, tag: str) -> Path:
    """``<prepared>/uavscenes_slz`` -> ``<prepared>/uavscenes_slz_<tag>``."""
    base_root = Path(base_root)
    return base_root.parent / f"{base_root.name}_{tag}"


# -- output layout -------------------------------------------------------------------------
#   outputs/experiments_<YYYYMMDD>/
#     main/<model>/fold<N>/seed<S>/            every model x fold x seed
#     radius/<tag>/<model>/fold<N>/seed<S>/    footprint-radius study
#     logs/  results/  results_radius/
OUTPUTS = REPO / "outputs"
EXPERIMENTS_PREFIX = "experiments_"


def new_experiment_dir() -> Path:
    return OUTPUTS / f"{EXPERIMENTS_PREFIX}{date.today():%Y%m%d}"


def latest_experiment_dir() -> Optional[Path]:
    dirs = sorted(p for p in OUTPUTS.glob(f"{EXPERIMENTS_PREFIX}*") if p.is_dir())
    return dirs[-1] if dirs else None


def run_dir(exp_dir: Path, model: str, fold: int, seed: int) -> Path:
    return Path(exp_dir) / "main" / model / f"fold{fold}" / f"seed{seed}"


def radius_run_dir(exp_dir: Path, tag: str, model: str, fold: int, seed: int) -> Path:
    return Path(exp_dir) / "radius" / tag / model / f"fold{fold}" / f"seed{seed}"
