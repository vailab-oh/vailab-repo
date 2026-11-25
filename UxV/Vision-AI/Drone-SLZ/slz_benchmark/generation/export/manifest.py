"""Family-aware fixed / four-fold splits, manifest CSVs and ``dataset_config.yaml``."""

from __future__ import annotations

import json
from pathlib import Path

from .._util import mkdir
from ..config import FIXED_FAMILY_SPLIT, FOUR_FOLD_FAMILY_SPLITS, PHYSICAL_FAMILIES


def physical_family(scene: str) -> str:
    s = str(scene)
    for fam in PHYSICAL_FAMILIES:
        if s.startswith(fam):
            return fam
    return s


def family_scene_map(scenes):
    out = {}
    for scene in scenes:
        out.setdefault(physical_family(scene), []).append(str(scene))
    for fam in out:
        out[fam] = sorted(out[fam])
    return out


def validate_scene_families(scenes):
    fmap = family_scene_map(scenes)
    unknown = sorted(set(fmap) - set(PHYSICAL_FAMILIES))
    if unknown:
        raise ValueError(
            "Built-in split logic supports these physical families only: "
            + ", ".join(PHYSICAL_FAMILIES)
            + "\nUnknown family: "
            + ", ".join(unknown)
        )

    missing = [fam for fam in PHYSICAL_FAMILIES if fam not in fmap]
    if missing:
        raise ValueError(
            "Fixed/4-fold splits need at least one sequence from every physical family. "
            "Missing: " + ", ".join(missing)
        )
    return fmap


def expand_family_split(fmap, family_spec):
    return {
        split: [
            scene
            for fam in family_spec[split]
            for scene in fmap.get(fam, [])
        ]
        for split in ["train", "val", "test"]
    }


def write_split_csvs(df, base: Path, spec: dict):
    mkdir(base)
    info = {}

    for split in ["train", "val", "test"]:
        scenes = spec[split]
        sdf = df[df["scene"].isin(scenes)].copy()
        sdf.to_csv(base / f"{split}.csv", index=False)
        info[split] = {
            "scenes": scenes,
            "families": sorted({physical_family(s) for s in scenes}),
            "frames": int(len(sdf)),
        }

    (base / "split.json").write_text(
        json.dumps(info, indent=2),
        encoding="utf-8",
    )


def generate_splits(df, prepared_root: Path):
    present_scenes = sorted(df["scene"].astype(str).unique().tolist())
    fmap = validate_scene_families(present_scenes)

    fixed = expand_family_split(fmap, FIXED_FAMILY_SPLIT)
    manifests = prepared_root / "manifests"
    write_split_csvs(df, manifests / "fixed", fixed)

    folds = {}
    for fold_name, family_spec in FOUR_FOLD_FAMILY_SPLITS.items():
        spec = expand_family_split(fmap, family_spec)
        folds[fold_name] = spec
        write_split_csvs(
            df,
            manifests / "four_fold" / fold_name,
            spec,
        )

    print("\nPhysical scene families:")
    for fam in PHYSICAL_FAMILIES:
        print(f"  {fam:10s}: {', '.join(fmap[fam])}")

    print("\nFixed split:")
    for split in ["train", "val", "test"]:
        print(f"  {split:5s}: {', '.join(fixed[split])}")

    print("\n4-fold:")
    for fold_name, spec in folds.items():
        print(
            f"  {fold_name}: "
            f"train={'+'.join(spec['train'])} | "
            f"val={'+'.join(spec['val'])} | "
            f"test={'+'.join(spec['test'])}"
        )

    return fmap, fixed, folds


def write_dataset_metadata(args, prepared_root: Path, df, fmap, fixed, folds):
    """Write ``prepared/<variant>/dataset_config.yaml``.

    ``args`` is the ``GenerationConfig`` (only attribute access is used).
    """
    lines = [
        "dataset: UAVScenes",
        f"variant: {args.dataset_name}",
        "selected_scenes:",
    ]
    lines.extend([f"  - {s}" for s in sorted(df["scene"].astype(str).unique())])

    lines.append("scene_families:")
    for fam in PHYSICAL_FAMILIES:
        lines.append(f"  {fam}:")
        for scene in fmap[fam]:
            lines.append(f"    - {scene}")

    frame_value = (
        args.frames_per_scene
        if args.frames_per_scene is not None
        else "all"
    )
    sampling = (
        "uniform"
        if args.frames_per_scene is not None
        else "all_frames"
    )

    lines += [
        f"frames_per_scene: {frame_value}",
        f"sampling: {sampling}",
        "image_size:",
        f"  width: {args.width}",
        f"  height: {args.height}",
        "frame_naming:",
        "  canonical_id: rgb_stem",
        "  extensions_may_differ: true",
        "  pair_validation: true",
        "  stale_output_cleanup: true",
        "pseudo_geometry:",
        "  depth_source: photogrammetric_mesh_zbuffer",
        "  normal_source: rendered_mesh_vertex_normal",
        "local_pca:",
        f"  radius_m: {args.pca_radius_m}",
        f"  min_support: {args.min_support}",
        f"  min_points: {args.min_points}",
        f"  world_up_axis: {args.up_axis}",
        "slz_target:",
        "  encoding:",
        "    unsafe: 0",
        "    safe: 1",
        "    ignore: 255",
        "  algorithm_version: v11_metric_landable_region",
        "  invalid_geometry_policy: ignore",
        "  landing_feasibility_policy: staged_metric_region",
        "  stage_1: metric_safety_buffer_around_confirmed_unsafe",
        "  stage_2: temporary_uav_feasible_centers",
        "  stage_3: reconstruct_landable_region_from_feasible_centers",
        "  projection: mesh_camera_z_depth_with_separate_fx_fy",
        "slz_visualization:",
        "  training_target: false",
        "  directory: slz_viz",
        "  colors_rgb:",
        "    unsafe: [255, 0, 0]",
        "    safe: [0, 255, 0]",
        "    ignore: [128, 128, 128]",
        "slz_thresholds:",
        f"  slope_deg: {args.slope_threshold_deg}",
        f"  roughness_m: {args.roughness_threshold_m}",
        "  semantic_policy: hazard_blacklist",
        f"  semantic_hazard_ids: [{', '.join(str(x) for x in args.hazard_ids)}]",
        f"  footprint_filter: {str(args.with_footprint).lower()}",
        "  footprint_order: safety_buffer_then_uav_fit_then_region_reconstruction",
        f"  safety_buffer_m: {args.safety_buffer_m}",
        f"  uav_radius_m: {args.uav_radius_m}",
        f"  footprint_border_as_unsafe: {str(args.footprint_border_as_unsafe).lower()}",
        "fixed_split:",
    ]

    for split in ["train", "val", "test"]:
        lines.append(f"  {split}: [{', '.join(fixed[split])}]")

    lines += [
        "four_fold_split: family_aware_scene_disjoint",
        f"total_frames: {len(df)}",
    ]

    (prepared_root / "dataset_config.yaml").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

