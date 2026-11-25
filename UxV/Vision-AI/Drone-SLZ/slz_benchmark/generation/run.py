"""SLZ label generator: command line and the order of the stages.

resolve paths -> ``GenerationConfig`` -> ``run_preprocessing`` (mesh depth / normals) ->
``run_slz_generation`` (PCA, semantic veto, footprint) -> ``organize_prepared_dataset`` ->
``generate_splits`` -> ``write_dataset_metadata``. No algorithm code lives here.
Normally run through ``prepare_data.py --stage labels``.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from ._util import mkdir
from .config import DEFAULT_HAZARD_IDS, DEFAULT_SCENES, GenerationConfig
from .export.manifest import generate_splits, validate_scene_families, write_dataset_metadata
from .export.layout import organize_prepared_dataset
from .render.depth_normal import run_preprocessing
from .slz.label import run_slz_generation


DEFAULT_DATASET_NAME = "uavscenes_slz"


def ensure_top_layout(root: Path):
    for p in [root / "raw", root / "prepared"]:
        mkdir(p)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Create a clean UAVScenes prepared dataset with family-aware "
            "fixed + 4-fold scene splits and 0/1/255 SLZ targets."
        ),
    )

    ap.add_argument(
        "--root",
        type=Path,
        required=True,
        help="UAVScenes root: the downloaded archives are unpacked under <root>/raw/, "
             "the prepared dataset is written to <root>/prepared/<dataset-name>/.",
    )
    ap.add_argument("--raw-dir", type=Path, default=None)
    ap.add_argument("--mesh-root", type=Path, default=None)
    ap.add_argument("--intervals", nargs="+", default=["interval5"])

    ap.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help=(
            f"Folder under prepared/. If omitted: '{DEFAULT_DATASET_NAME}' for all frames, "
            "or 'sample_uniformN' when --frames-per-scene N is used."
        ),
    )
    ap.add_argument(
        "--scenes",
        nargs="+",
        default=DEFAULT_SCENES,
        help="Explicit UAVScenes sequence names.",
    )
    ap.add_argument(
        "--frames-per-scene",
        type=int,
        default=None,
        help=(
            "Uniformly sample N frames from EACH selected scene. "
            "Omit this argument to generate ALL frames."
        ),
    )
    ap.add_argument("--sample-seed", type=int, default=42)

    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=640)

    ap.add_argument("--pca-radius-m", type=float, default=0.50)
    ap.add_argument("--slope-threshold-deg", type=float, default=6.0)
    ap.add_argument("--roughness-threshold-m", type=float, default=0.10)
    ap.add_argument(
        "--hazard-ids",
        type=int,
        nargs="+",
        default=DEFAULT_HAZARD_IDS,
        help=(
            "Semantic hazard blacklist. Only these IDs are excluded by semantics; "
            "unknown/background/unlisted IDs pass. Default: 4 5 11 15 16 17 20 24"
        ),
    )
    ap.add_argument("--min-support", type=float, default=0.50)
    ap.add_argument("--min-points", type=int, default=9)
    ap.add_argument("--max-metric-radius-px", type=int, default=512)
    ap.add_argument("--row-chunk", type=int, default=128)
    ap.add_argument(
        "--up-axis",
        choices=["x", "-x", "y", "-y", "z", "-z"],
        default="z",
    )

    # Paper defaults: footprint filter ON, UAV radius 0.35 m, safety buffer 0.15 m.
    ap.add_argument("--with-footprint", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--uav-radius-m", type=float, default=0.35)
    ap.add_argument("--safety-buffer-m", type=float, default=0.15)
    ap.add_argument(
        "--footprint-border-as-unsafe",
        action="store_true",
        help=(
            "Treat the image border as unsafe during the buffer / footprint checks "
            "(default: the camera edge is not a physical hazard)."
        ),
    )

    ap.add_argument(
        "--asset-mode",
        choices=["symlink", "hardlink", "copy"],
        default="symlink",
        help="How RGB / semantic files are placed in prepared dataset.",
    )
    ap.add_argument(
        "--overwrite", action="store_true",
        help="Delete/recompute current _work instead of automatic resume/cache reuse."
    )
    ap.add_argument(
        "--keep-work", action="store_true",
        help="Keep _work after a successful build. Interrupted runs always leave _work for resume."
    )
    ap.add_argument(
        "--splits-only",
        action="store_true",
        help="Rebuild only split manifests from an existing prepared dataset.",
    )

    args = ap.parse_args(argv)

    if args.frames_per_scene is not None and args.frames_per_scene < 1:
        ap.error("--frames-per-scene must be >= 1, or omit it for ALL frames")

    return args


def _resolve_cli_paths(args):
    """Default raw / mesh / dataset-name paths."""
    args.root = args.root.resolve()

    if args.raw_dir is None:
        args.raw_dir = args.root / "raw"
    if args.mesh_root is None:
        args.mesh_root = args.root / "raw"

    if args.dataset_name is None:
        args.dataset_name = (
            DEFAULT_DATASET_NAME
            if args.frames_per_scene is None
            else f"sample_uniform{args.frames_per_scene}"
        )
    return args


def _build_config(args) -> GenerationConfig:
    """CLI ``args`` (after ``_resolve_cli_paths``) -> the ``GenerationConfig`` every
    stage takes. One-to-one field mapping."""
    return GenerationConfig(
        root=args.root,
        raw_dir=args.raw_dir,
        mesh_root=args.mesh_root,
        dataset_name=args.dataset_name,
        intervals=tuple(args.intervals),
        scenes=tuple(args.scenes),
        frames_per_scene=args.frames_per_scene,
        sample_seed=args.sample_seed,
        height=args.height,
        width=args.width,
        pca_radius_m=args.pca_radius_m,
        min_support=args.min_support,
        min_points=args.min_points,
        max_metric_radius_px=args.max_metric_radius_px,
        row_chunk=args.row_chunk,
        up_axis=args.up_axis,
        slope_threshold_deg=args.slope_threshold_deg,
        roughness_threshold_m=args.roughness_threshold_m,
        hazard_ids=tuple(args.hazard_ids),
        with_footprint=args.with_footprint,
        uav_radius_m=args.uav_radius_m,
        safety_buffer_m=args.safety_buffer_m,
        footprint_border_as_unsafe=args.footprint_border_as_unsafe,
        asset_mode=args.asset_mode,
        overwrite=args.overwrite,
        keep_work=args.keep_work,
    )


def main(argv=None) -> Path:
    """Run the generator; returns the prepared dataset directory."""
    args = parse_args(argv)
    args = _resolve_cli_paths(args)

    ensure_top_layout(args.root)

    validate_scene_families(args.scenes)

    cfg = _build_config(args)

    prepared_root = cfg.root / "prepared" / cfg.dataset_name
    mkdir(prepared_root / "manifests")
    mkdir(prepared_root / "scenes")

    if args.splits_only:
        all_csv = prepared_root / "manifests" / "all.csv"
        if not all_csv.exists():
            raise FileNotFoundError(all_csv)

        df = pd.read_csv(all_csv)
        fmap, fixed, folds = generate_splits(df, prepared_root)
        write_dataset_metadata(cfg, prepared_root, df, fmap, fixed, folds)
        print(f"\n[DONE] split manifests rebuilt: {prepared_root / 'manifests'}")
        return prepared_root

    work_root = prepared_root / "_work"

    if work_root.exists() and cfg.overwrite:
        shutil.rmtree(work_root)
    elif work_root.exists():
        print(f"[RESUME] Reusing existing work cache: {work_root}")
    mkdir(work_root)

    print("\n" + "=" * 80)
    print("[1/4] RAW -> RGB-aligned mesh depth / normal")
    print("=" * 80)
    print("Scenes:")
    for s in cfg.scenes:
        print(f"  - {s}")
    if cfg.frames_per_scene is None:
        print("Sampling: ALL frames in each selected scene")
    else:
        print(f"Sampling: uniform {cfg.frames_per_scene} frames / scene")
    print("=" * 80)
    run_preprocessing(cfg, work_root)

    print("\n" + "=" * 80)
    print("[2/4] Local PCA -> slope / roughness / SLZ target (0 Unsafe / 1 Safe / 255 Ignore)")
    print("=" * 80)
    run_name = run_slz_generation(cfg, work_root)

    df = organize_prepared_dataset(cfg, work_root, prepared_root, run_name)

    if len(df) == 0:
        raise RuntimeError("No complete frames were generated.")

    fmap, fixed, folds = generate_splits(df, prepared_root)
    write_dataset_metadata(cfg, prepared_root, df, fmap, fixed, folds)

    if not cfg.keep_work and work_root.exists():
        shutil.rmtree(work_root)

    print("\n" + "=" * 80)
    print("[DONE]")
    print(f"Dataset : {prepared_root}")
    print(f"Frames  : {len(df)}")
    print(f"Manifest: {prepared_root / 'manifests' / 'all.csv'}")
    print("=" * 80)
    return prepared_root


if __name__ == "__main__":
    main()
