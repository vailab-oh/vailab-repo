"""SLZ stage: PCA geometry + semantic veto -> raw SLZ -> safety buffer -> UAV feasible
centers -> landable region -> 0/1/255 target, for every frame ``run_preprocessing``
produced.

``build_slz_label`` holds the labelling policy for one frame; ``run_slz_generation`` runs
it over all frames and writes the per-frame arrays, CSVs and diagnostics.
"""

from __future__ import annotations

import json
import re
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .._util import load_array, mkdir, resolve_path
from ..config import GenerationConfig, IGNORE_LABEL, SAFE_LABEL, UNSAFE_LABEL
from ..geometry.pca import pca_geometry, slope_from_world_normal
from ..geometry.projection import (
    normal_angle_difference,
    normal_cam_to_world,
    orient_world_normals_up,
    scaled_K,
)
from ..visualization import color_scalar, load_rgb, normal_rgb, overlay, save_grid
from .footprint import metric_clearance_from_blocked, projected_radii_xy, reconstruct_landable_region
from .semantic import semantic_safe


def sample_df(df, n, mode, seed):
    if n is None or n >= len(df):
        return df.copy()
    if mode == "first":
        return df.head(n).copy()
    if mode == "random":
        return df.sample(n=n,random_state=seed).sort_index().copy()
    idx = np.linspace(0,len(df)-1,n).round().astype(int)
    return df.iloc[np.unique(idx)].copy()


def scene_map_group(scene: str) -> str:
    """Strip a trailing 01/02/03 run suffix to get the map group name.

    Kept separate from ``render.raw_scenes.scene_map_group`` (suffix ``01``-``99``) and
    ``render.mesh.mesh_group_from_scene`` (keeps ``*_Evening``): the three differ in edge
    cases and the labels were generated with each as it is.
    """
    s = str(scene)
    m = re.match(r"^(.*?)(0[1-3])$", s)
    return m.group(1) if m else s


def sample_scene03_uniform(
    df: pd.DataFrame,
    frames_per_scene: int = 200,
    mode: str = "uniform",
    seed: int = 42,
    scene_suffix: str = "03",
):
    """
    1) Keep only scene names ending with `scene_suffix` (default: 03).
    2) For each 03 scene independently, select up to `frames_per_scene` frames.
    3) Uniform mode samples from the beginning to the end of each sequence.

    Example:
        AMtown03          -> 200 frames
        AMvalley03        -> 200 frames
        HKisland_GNSS03   -> 200 frames
        ...
    """
    if "scene" not in df.columns:
        raise KeyError("manifest must contain a 'scene' column")

    scene_s = df["scene"].astype(str)
    selected = df[scene_s.str.endswith(scene_suffix)].copy()

    if len(selected) == 0:
        available = sorted(scene_s.dropna().unique().tolist())
        preview = "\n  ".join(available[:50])
        raise RuntimeError(
            f"No scene ending with '{scene_suffix}' was found.\n"
            f"Available scenes (first 50):\n  {preview}"
        )

    groups = []
    report = []

    for scene_name, sdf in selected.groupby("scene", sort=True):
        # Uniform sampling should follow flight/frame order rather than manifest accident.
        if "cam_timestamp" in sdf.columns:
            order = pd.to_numeric(sdf["cam_timestamp"], errors="coerce")
            if order.notna().any():
                sdf = sdf.assign(_sample_order=order).sort_values("_sample_order")
            elif "rgb_name" in sdf.columns:
                sdf = sdf.sort_values("rgb_name")
        elif "rgb_name" in sdf.columns:
            sdf = sdf.sort_values("rgb_name")

        take_n = min(int(frames_per_scene), len(sdf))
        sampled = sample_df(sdf, take_n, mode, seed)
        sampled = sampled.drop(columns=["_sample_order"], errors="ignore")
        sampled = sampled.copy()
        sampled["map_group"] = scene_map_group(scene_name)

        groups.append(sampled)
        report.append({
            "map_group": scene_map_group(scene_name),
            "scene": str(scene_name),
            "available_frames": int(len(sdf)),
            "selected_frames": int(len(sampled)),
            "sample_mode": mode,
        })

    out = pd.concat(groups, axis=0, ignore_index=False)
    report_df = pd.DataFrame(report)
    return out, report_df


def build_slz_label(
    slope, rough, valid, sem, st, rt, D, K,
    *,
    with_footprint: bool,
    uav_radius_m: float,
    safety_buffer_m: float,
    footprint_border_as_unsafe: bool,
    max_metric_radius_px: int,
):
    """Geometry result + semantic veto -> raw SLZ -> Safety Buffer -> UAV feasible
    centers -> landable-region reconstruction -> final 0/1/255 SLZ target.

    Returns the final label plus every intermediate mask the CSV rows and
    visualisations use. The asserts check that Safe / Unsafe / Ignore partition the image.
    """
    # Three-state target:
    #   confirmed unsafe = valid geometry that violates a
    #   physical/semantic landing criterion
    #   safe             = valid geometry satisfying all criteria
    #   ignore           = geometry cannot be evaluated reliably
    semantic_hazard = ~sem
    slope_ok = slope <= st
    rough_ok = rough <= rt

    invalid = ~valid
    confirmed_unsafe = valid & (
        semantic_hazard | ~slope_ok | ~rough_ok
    )
    raw_safe = valid & ~semantic_hazard & slope_ok & rough_ok

    # Landable-region logic:
    #   1) Raw SLZ = valid surface satisfying semantic/slope/roughness.
    #   2) Apply ONLY Safety Buffer around confirmed unsafe evidence.
    #   3) Find temporary UAV feasible centers whose full depth-aware
    #      footprint lies inside the verified buffered-safe region.
    #      Invalid/unknown pixels are not labeled as obstacles, but a
    #      footprint cannot be verified through them.
    #   4) Expand feasible centers back by the UAV footprint and union
    #      them.  This reconstructed union is the Final SLZ *region*.
    buffered_safe = raw_safe.copy()
    feasible_centers = raw_safe.copy()
    final_safe = raw_safe.copy()

    buffer_r=np.nan
    uav_fit_r=np.nan
    buffer_rx=np.zeros_like(D,np.float32)
    buffer_ry=np.zeros_like(D,np.float32)
    uav_rx=np.zeros_like(D,np.float32)
    uav_ry=np.zeros_like(D,np.float32)

    if with_footprint:
        if uav_radius_m <= 0:
            raise ValueError("--with-footprint requires --uav-radius-m > 0")
        if safety_buffer_m < 0:
            raise ValueError("--safety-buffer-m must be >= 0")

        # Stage 1 — metric Safety Buffer around confirmed unsafe.
        # The projected x/y radii use fx/fy separately and the
        # mesh-rendered camera-Z depth at each candidate pixel.
        buffer_r=safety_buffer_m
        buffer_rx,buffer_ry=projected_radii_xy(D,K,buffer_r)
        clear_from_confirmed_unsafe_buffer=metric_clearance_from_blocked(
            confirmed_unsafe,D,K,buffer_r,
            border_as_blocked=footprint_border_as_unsafe,
            max_radius_px=max_metric_radius_px,
        )
        buffered_safe = (
            raw_safe & clear_from_confirmed_unsafe_buffer
        )

        # Stage 2 — temporary UAV feasible-center map.
        # The footprint must fit completely inside verified
        # buffered_safe. Therefore both known non-landable and
        # PCA-invalid/unknown pixels prevent a center from being
        # declared feasible, without re-labeling invalid pixels as
        # physical obstacles.
        uav_fit_r=uav_radius_m
        uav_rx,uav_ry=projected_radii_xy(D,K,uav_fit_r)
        blocked_for_uav_fit = ~buffered_safe
        clear_for_uav_fit=metric_clearance_from_blocked(
            blocked_for_uav_fit,D,K,uav_fit_r,
            border_as_blocked=footprint_border_as_unsafe,
            max_radius_px=max_metric_radius_px,
        )
        feasible_centers = (
            buffered_safe & clear_for_uav_fit
        )

        # Stage 3 — Final SLZ remains a landable *region*.
        # Do NOT store only the eroded center map. Expand each
        # feasible center back by its depth-aware UAV footprint.
        final_safe = reconstruct_landable_region(
            feasible_centers,
            buffered_safe,
            D,
            K,
            uav_fit_r,
            max_radius_px=max_metric_radius_px,
        )

    # Three-state target:
    #   Safe   = final reconstructed landable region
    #   Unsafe = valid geometry that is not in the final region
    #   Ignore = PCA geometry unavailable/unreliable
    unsafe_after_footprint = valid & ~final_safe
    ignore_after_footprint = invalid.copy()

    final_label=np.full(D.shape,IGNORE_LABEL,dtype=np.uint8)
    final_label[unsafe_after_footprint]=UNSAFE_LABEL
    final_label[final_safe]=SAFE_LABEL

    if np.any(final_safe & unsafe_after_footprint):
        raise RuntimeError("Safe and Unsafe masks overlap")
    if np.any(final_safe & ignore_after_footprint):
        raise RuntimeError("Safe and Ignore masks overlap")
    if np.any(unsafe_after_footprint & ignore_after_footprint):
        raise RuntimeError("Unsafe and Ignore masks overlap")
    if not np.all(
        final_safe | unsafe_after_footprint | ignore_after_footprint
    ):
        raise RuntimeError("SLZ target partition is incomplete")

    return {
        "semantic_hazard": semantic_hazard,
        "slope_ok": slope_ok,
        "rough_ok": rough_ok,
        "invalid": invalid,
        "confirmed_unsafe": confirmed_unsafe,
        "raw_safe": raw_safe,
        "buffered_safe": buffered_safe,
        "feasible_centers": feasible_centers,
        "final_safe": final_safe,
        "buffer_r": buffer_r,
        "uav_fit_r": uav_fit_r,
        "buffer_rx": buffer_rx,
        "buffer_ry": buffer_ry,
        "uav_rx": uav_rx,
        "uav_ry": uav_ry,
        "unsafe_after_footprint": unsafe_after_footprint,
        "ignore_after_footprint": ignore_after_footprint,
        "final_label": final_label,
    }


def run_slz_generation(
    cfg: GenerationConfig,
    work_root: Path,
    run_name: str = "_dataset_build",
) -> str:
    """PCA geometry -> semantic veto -> SLZ policy, for every frame
    ``run_preprocessing`` produced. Writes
    ``work_root/diagnostics/final_pca_slz_runs/<run_name>/radius_<r>m/...`` (slope,
    roughness, valid, masks/<cfg>, diagnostics, CSVs) and returns ``run_name``.

    ``args`` is a local ``Namespace`` built from ``cfg``. ``radii`` /
    ``slope_thresholds`` / ``roughness_thresholds`` are one-element lists: the loops
    support a sweep, but the label generator always runs exactly one setting.
    """
    pca_take_n = cfg.frames_per_scene if cfg.frames_per_scene is not None else 2147483647
    args = Namespace(
        out_dir=work_root,
        scene_filter=list(cfg.scenes),
        scene_suffix="",
        frames_per_scene=pca_take_n,
        sample_mode="uniform",
        sample_seed=cfg.sample_seed,
        radii=[cfg.pca_radius_m],
        slope_thresholds=[cfg.slope_threshold_deg],
        roughness_thresholds=[cfg.roughness_threshold_m],
        run_name=run_name,
        min_support=cfg.min_support,
        min_points=cfg.min_points,
        max_metric_radius_px=cfg.max_metric_radius_px,
        row_chunk=cfg.row_chunk,
        up_axis=cfg.up_axis,
        hazard_ids=list(cfg.hazard_ids),
        overwrite=cfg.overwrite,
        with_footprint=cfg.with_footprint,
        uav_radius_m=cfg.uav_radius_m,
        safety_buffer_m=cfg.safety_buffer_m,
        footprint_border_as_unsafe=cfg.footprint_border_as_unsafe,
        save_viz=False,
        viz_slope_th=10.0,
        viz_roughness_th=0.10,
    )

    manifest=args.out_dir/"manifests"/"all.csv"
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    df=pd.read_csv(manifest)

    # Optional explicit pre-filter, then keep only run/scene suffix 03.
    if args.scene_filter:
        df=df[df["scene"].astype(str).isin(args.scene_filter)].copy()

    df, sample_report = sample_scene03_uniform(
        df,
        frames_per_scene=args.frames_per_scene,
        mode=args.sample_mode,
        seed=args.sample_seed,
        scene_suffix=args.scene_suffix,
    )

    if len(df)==0:
        raise RuntimeError("No frames selected.")

    root=args.out_dir/"diagnostics"/"final_pca_slz_runs"/args.run_name
    mkdir(root)

    keep_cols = [
        c for c in ["map_group","scene","interval","rgb_name","cam_timestamp"]
        if c in df.columns
    ]
    df[keep_cols].to_csv(root/"selected_frames.csv", index=False)
    sample_report.to_csv(root/"sampling_report.csv", index=False)

    print("\n=== 03 scene uniform sampling ===", flush=True)
    for _, rr0 in sample_report.iterrows():
        print(
            f"{rr0['scene']}: {int(rr0['selected_frames'])} / "
            f"{int(rr0['available_frames'])} frames",
            flush=True,
        )
    print(f"TOTAL: {len(df)} frames\n", flush=True)

    info_cache={}
    rows=[]

    for fi,(_,row) in enumerate(df.iterrows(),1):
        frame=str(row.get("cam_timestamp","")).strip()
        dp=resolve_path(args.out_dir,row.get("depth_mesh_path"))
        npth=resolve_path(args.out_dir,row.get("normal_vertex_path"))
        sp=resolve_path(args.out_dir,row.get("sampleinfo_json_path"))
        if dp is None or not dp.exists() or sp is None or not sp.exists():
            print(f"[SKIP] {frame}: missing depth/sampleinfo")
            continue

        D=np.asarray(load_array(dp,"depth_mesh"),np.float32)

        raw_normal_cam = None
        if npth is not None and npth.exists():
            raw_normal_cam = np.asarray(load_array(npth, "normal_vertex"), np.float32)
        k=str(sp)
        if k not in info_cache:
            infos=json.load(open(sp,"r",encoding="utf-8"))
            info_cache[k]={x["OriginalImageName"]:x for x in infos if "OriginalImageName" in x}
        rgb_name=str(row.get("rgb_name",""))
        info=info_cache[k].get(rgb_name)
        if info is None:
            print(f"[SKIP] {frame}: no pose for {rgb_name}")
            continue

        sem=semantic_safe(resolve_path(args.out_dir,row.get("cam_label_id_path")),D.shape,args.hazard_ids)
        rgb=load_rgb(resolve_path(args.out_dir,row.get("rgb_path")),D.shape) if args.save_viz else None

        for r in args.radii:
            rr=root/f"radius_{r:.3f}m"
            for sub in [
                "slope",
                "roughness",
                "valid",
                "support",
                "masks",
                "comparison",
                "pca_normal_cam",
                "pca_normal_world",
                "raw_normal_world",
            ]:
                mkdir(rr/sub)

            slope_path = rr/"slope"/f"{frame}.npy"
            rough_path = rr/"roughness"/f"{frame}.npy"
            valid_path = rr/"valid"/f"{frame}.png"
            support_path = rr/"support"/f"{frame}.npy"
            pca_cam_path = rr/"pca_normal_cam"/f"{frame}.npy"
            pca_world_path = rr/"pca_normal_world"/f"{frame}.npy"

            # Automatic resume/cache for expensive local PCA geometry. Semantic
            # policy and final masks are intentionally recomputed below every run,
            # so switching whitelist/blacklist policy cannot reuse a stale SLZ mask.
            geometry_cache_ready = (
                not args.overwrite
                and slope_path.exists()
                and rough_path.exists()
                and valid_path.exists()
                and pca_cam_path.exists()
                and pca_world_path.exists()
                and (not args.save_viz or support_path.exists())
            )

            if geometry_cache_ready:
                print(f"[{fi}/{len(df)}] {frame} | PCA radius={r:.3f}m [CACHE]",flush=True)
                slope=np.asarray(np.load(slope_path),np.float32)
                rough=np.asarray(np.load(rough_path),np.float32)
                valid=np.array(Image.open(valid_path)) > 0
                pca_normal_cam=np.asarray(np.load(pca_cam_path),np.float32)
                pca_normal_world=np.asarray(np.load(pca_world_path),np.float32)
                K=scaled_K(info,D.shape)
                support=(
                    np.asarray(np.load(support_path),np.float32)
                    if support_path.exists()
                    else np.zeros_like(D,np.float32)
                )
            else:
                print(f"[{fi}/{len(df)}] {frame} | PCA radius={r:.3f}m",flush=True)
                slope,rough,valid,K,support,pca_normal_cam=pca_geometry(
                    D,info,r,args.min_support,args.min_points,
                    args.max_metric_radius_px,args.row_chunk,args.up_axis
                )

                # PCA normal: camera coordinates and world coordinates.
                pca_normal_world = normal_cam_to_world(pca_normal_cam, info)
                pca_normal_world = orient_world_normals_up(
                    pca_normal_world, valid, args.up_axis
                )

                np.save(slope_path, slope)
                np.save(rough_path, rough)
                np.save(support_path, support.astype(np.float32))
                np.save(pca_cam_path, pca_normal_cam.astype(np.float32))
                np.save(pca_world_path, pca_normal_world.astype(np.float32))
                Image.fromarray(valid.astype(np.uint8)*255).save(valid_path)

            # Existing rendered mesh normal for direct comparison.
            raw_normal_world = None
            raw_normal_valid = None
            raw_slope = None
            normal_diff = None

            if raw_normal_cam is not None and raw_normal_cam.shape[:2] == D.shape:
                raw_len = np.linalg.norm(raw_normal_cam, axis=2)
                raw_normal_valid = (D > 0) & np.isfinite(raw_len) & (raw_len > 1e-8)
                raw_normal_world = normal_cam_to_world(raw_normal_cam, info)
                raw_normal_world = orient_world_normals_up(
                    raw_normal_world, raw_normal_valid, args.up_axis
                )
                np.save(
                    rr/"raw_normal_world"/f"{frame}.npy",
                    raw_normal_world.astype(np.float32),
                )

                common_normal_valid = valid & raw_normal_valid
                raw_slope = slope_from_world_normal(
                    raw_normal_world, raw_normal_valid, args.up_axis
                )
                normal_diff = normal_angle_difference(
                    raw_normal_world, pca_normal_world, common_normal_valid
                )

            viz_saved=False
            for st in args.slope_thresholds:
                for rt in args.roughness_thresholds:
                    label = build_slz_label(
                        slope, rough, valid, sem, st, rt, D, K,
                        with_footprint=args.with_footprint,
                        uav_radius_m=args.uav_radius_m,
                        safety_buffer_m=args.safety_buffer_m,
                        footprint_border_as_unsafe=args.footprint_border_as_unsafe,
                        max_metric_radius_px=args.max_metric_radius_px,
                    )
                    semantic_hazard = label["semantic_hazard"]
                    slope_ok = label["slope_ok"]
                    rough_ok = label["rough_ok"]
                    invalid = label["invalid"]
                    confirmed_unsafe = label["confirmed_unsafe"]
                    raw_safe = label["raw_safe"]
                    buffered_safe = label["buffered_safe"]
                    feasible_centers = label["feasible_centers"]
                    final_safe = label["final_safe"]
                    buffer_r = label["buffer_r"]
                    uav_fit_r = label["uav_fit_r"]
                    buffer_rx = label["buffer_rx"]
                    buffer_ry = label["buffer_ry"]
                    uav_rx = label["uav_rx"]
                    uav_ry = label["uav_ry"]
                    unsafe_after_footprint = label["unsafe_after_footprint"]
                    ignore_after_footprint = label["ignore_after_footprint"]
                    final_label = label["final_label"]

                    cfg_tag=f"slope_{st:g}deg_rough_{rt:.3f}m"
                    mkdir(rr/"masks"/cfg_tag)
                    Image.fromarray(final_label).save(rr/"masks"/cfg_tag/f"{frame}.png")

                    rows.append({
                        "map_group":str(row.get("map_group", scene_map_group(str(row.get("scene",""))))),
                        "scene":str(row.get("scene","")),
                        "interval":str(row.get("interval","")),
                        "frame":frame,
                        "radius_m":r,
                        "slope_threshold_deg":st,
                        "roughness_threshold_m":rt,
                        "hazard_ids":",".join(str(x) for x in args.hazard_ids),
                        "pca_valid_ratio":float(np.mean(valid)),
                        "pca_invalid_ratio":float(np.mean(invalid)),
                        "raw_safe_ratio":float(np.mean(raw_safe)),
                        "buffered_safe_ratio":float(np.mean(buffered_safe)),
                        "feasible_center_ratio":float(np.mean(feasible_centers)),
                        "final_safe_ratio":float(np.mean(final_label == SAFE_LABEL)),
                        "final_unsafe_ratio":float(np.mean(final_label == UNSAFE_LABEL)),
                        "final_ignore_ratio":float(np.mean(final_label == IGNORE_LABEL)),
                        "semantic_hazard_ratio":float(np.mean(valid & semantic_hazard)),
                        "slope_fail_ratio":float(np.mean(valid & ~slope_ok)),
                        "roughness_fail_ratio":float(np.mean(valid & ~rough_ok)),
                        "both_fail_ratio":float(np.mean(valid & ~slope_ok & ~rough_ok)),
                        "safety_buffer_removed_ratio":float(np.mean(raw_safe & ~buffered_safe)),
                        "uav_footprint_opening_removed_ratio":float(np.mean(buffered_safe & ~final_safe)),
                        "buffer_rx_median_px":float(np.median(buffer_rx[D>0])) if np.any(D>0) else np.nan,
                        "buffer_ry_median_px":float(np.median(buffer_ry[D>0])) if np.any(D>0) else np.nan,
                        "uav_rx_median_px":float(np.median(uav_rx[D>0])) if np.any(D>0) else np.nan,
                        "uav_ry_median_px":float(np.median(uav_ry[D>0])) if np.any(D>0) else np.nan,
                        "slope_p50_deg":float(np.percentile(slope[valid],50)) if np.any(valid) else np.nan,
                        "slope_p95_deg":float(np.percentile(slope[valid],95)) if np.any(valid) else np.nan,
                        "roughness_p50_m":float(np.percentile(rough[valid],50)) if np.any(valid) else np.nan,
                        "roughness_p95_m":float(np.percentile(rough[valid],95)) if np.any(valid) else np.nan,
                        "safety_buffer_radius_m":buffer_r,
                        "uav_footprint_radius_m":uav_fit_r,
                        "unknown_clearance_radius_m":uav_fit_r,
                        "footprint_border_as_unsafe":bool(args.footprint_border_as_unsafe),
                        "raw_vs_pca_normal_angle_mean_deg":(
                            float(np.mean(normal_diff[valid & raw_normal_valid]))
                            if normal_diff is not None and np.any(valid & raw_normal_valid)
                            else np.nan
                        ),
                        "raw_vs_pca_normal_angle_p95_deg":(
                            float(np.percentile(normal_diff[valid & raw_normal_valid],95))
                            if normal_diff is not None and np.any(valid & raw_normal_valid)
                            else np.nan
                        ),
                    })

                    if args.save_viz and not viz_saved:
                        panels = [
                            ("RGB", rgb),
                        ]

                        if raw_normal_world is not None:
                            panels += [
                                (
                                    "Raw mesh normal (world)",
                                    normal_rgb(raw_normal_world, raw_normal_valid),
                                ),
                                (
                                    "Local PCA normal (world)",
                                    normal_rgb(pca_normal_world, valid),
                                ),
                                (
                                    "Raw normal slope",
                                    color_scalar(raw_slope, raw_normal_valid, 30),
                                ),
                                (
                                    f"PCA normal slope r={r:.2f}m",
                                    color_scalar(slope, valid, 30),
                                ),
                                (
                                    "Raw vs PCA normal angle diff",
                                    color_scalar(
                                        normal_diff,
                                        valid & raw_normal_valid,
                                        45,
                                    ),
                                ),
                            ]
                        else:
                            panels += [
                                (
                                    "Local PCA normal (world)",
                                    normal_rgb(pca_normal_world, valid),
                                ),
                                (
                                    f"PCA normal slope r={r:.2f}m",
                                    color_scalar(slope, valid, 30),
                                ),
                            ]

                        panels += [
                            (
                                "PCA Surface Roughness",
                                color_scalar(rough, valid, .20),
                            ),
                            (
                                "Metric-neighborhood support",
                                color_scalar(support, D > 0, 1.0),
                            ),
                            (
                                "PCA valid",
                                np.repeat(
                                    (valid.astype(np.uint8) * 255)[..., None],
                                    3,
                                    axis=2,
                                ),
                            ),
                            (
                                "PCA invalid / Ignore source",
                                overlay(
                                    rgb,
                                    invalid,
                                    (255, 0, 255),
                                    .60,
                                ),
                            ),
                            (
                                "Semantic hazard",
                                overlay(
                                    rgb,
                                    valid & semantic_hazard,
                                    (0, 0, 255),
                                    .60,
                                ),
                            ),
                            (
                                f"Slope rejected > {st:g} deg",
                                overlay(
                                    rgb,
                                    valid & ~slope_ok,
                                    (0, 0, 255),
                                    .60,
                                ),
                            ),
                            (
                                f"Roughness rejected > {rt:.2f} m",
                                overlay(
                                    rgb,
                                    valid & ~rough_ok,
                                    (0, 165, 255),
                                    .60,
                                ),
                            ),
                            (
                                "Raw Safe before buffer / footprint",
                                overlay(rgb, raw_safe),
                            ),
                            (
                                (
                                    f"Buffered Safe (Safety Buffer={buffer_r:.2f}m)"
                                    if args.with_footprint
                                    else "Buffered Safe"
                                ),
                                overlay(rgb, buffered_safe),
                            ),
                            (
                                (
                                    f"Temporary UAV Feasible Centers (R={uav_fit_r:.2f}m)"
                                    if args.with_footprint
                                    else "Temporary UAV Feasible Centers"
                                ),
                                overlay(rgb, feasible_centers),
                            ),
                            (
                                (
                                    f"Final SLZ Region (reconstructed R={uav_fit_r:.2f}m)"
                                    if args.with_footprint
                                    else "Final SLZ Region"
                                ),
                                overlay(rgb, final_safe),
                            ),
                            (
                                "Final Ignore",
                                overlay(
                                    rgb,
                                    final_label == IGNORE_LABEL,
                                    (255, 0, 255),
                                    .60,
                                ),
                            ),
                        ]

                        if args.with_footprint:
                            panels += [
                                (
                                    "Removed by Safety Buffer",
                                    overlay(
                                        rgb,
                                        raw_safe & ~buffered_safe,
                                        (0, 0, 255),
                                        .65,
                                    ),
                                ),
                                (
                                    "Removed by UAV Size/Shape Feasibility",
                                    overlay(
                                        rgb,
                                        buffered_safe & (final_label != SAFE_LABEL),
                                        (0, 165, 255),
                                        .65,
                                    ),
                                ),
                            ]

                        save_grid(
                            rr/"comparison"/f"{frame}_pca.jpg",
                            [
                                ("RGB", rgb),
                                (
                                    f"PCA Slope r={r:.2f}m",
                                    color_scalar(slope, valid, 20),
                                ),
                                (
                                    "PCA Roughness",
                                    color_scalar(rough, valid, .20),
                                ),
                                (
                                    f"Final Safe slope<={st:g} / rough<={rt:.2f}m",
                                    overlay(rgb, final_safe),
                                ),
                                (
                                    "Final Unsafe",
                                    overlay(
                                        rgb,
                                        final_label == UNSAFE_LABEL,
                                        (0, 0, 255),
                                        .6,
                                    ),
                                ),
                                (
                                    "Final Ignore",
                                    overlay(
                                        rgb,
                                        final_label == IGNORE_LABEL,
                                        (255, 0, 255),
                                        .6,
                                    ),
                                ),
                            ],
                            cols=3,
                        )

                        save_grid(
                            rr/"comparison"/f"{frame}_FULL_s{st:g}_r{rt:.3f}.jpg",
                            panels,
                            cols=3,
                        )

                        viz_saved=True

    res=pd.DataFrame(rows)
    res.to_csv(root/"per_frame_results.csv",index=False)
    if len(res):
        metric_aggs = dict(
            frames=("frame","nunique"),
            mean_pca_valid_ratio=("pca_valid_ratio","mean"),
            mean_pca_invalid_ratio=("pca_invalid_ratio","mean"),
            mean_raw_safe_ratio=("raw_safe_ratio","mean"),
            mean_buffered_safe_ratio=("buffered_safe_ratio","mean"),
            mean_safety_buffer_removed_ratio=("safety_buffer_removed_ratio","mean"),
            mean_feasible_center_ratio=("feasible_center_ratio","mean"),
            mean_uav_footprint_opening_removed_ratio=("uav_footprint_opening_removed_ratio","mean"),
            mean_final_safe_ratio=("final_safe_ratio","mean"),
            mean_final_unsafe_ratio=("final_unsafe_ratio","mean"),
            mean_final_ignore_ratio=("final_ignore_ratio","mean"),
            mean_semantic_hazard_ratio=("semantic_hazard_ratio","mean"),
            mean_slope_fail_ratio=("slope_fail_ratio","mean"),
            mean_roughness_fail_ratio=("roughness_fail_ratio","mean"),
            mean_both_fail_ratio=("both_fail_ratio","mean"),
            mean_slope_p50_deg=("slope_p50_deg","mean"),
            mean_slope_p95_deg=("slope_p95_deg","mean"),
            mean_roughness_p50_m=("roughness_p50_m","mean"),
            mean_roughness_p95_m=("roughness_p95_m","mean"),
            mean_raw_vs_pca_normal_angle_deg=("raw_vs_pca_normal_angle_mean_deg","mean"),
            mean_raw_vs_pca_normal_angle_p95_deg=("raw_vs_pca_normal_angle_p95_deg","mean"),
        )

        agg=(res.groupby(["radius_m","slope_threshold_deg","roughness_threshold_m"])
             .agg(**metric_aggs).reset_index())
        agg.to_csv(root/"aggregate_results.csv",index=False)

        by_map=(res.groupby([
                    "map_group", "scene", "radius_m",
                    "slope_threshold_deg", "roughness_threshold_m"
                ])
                .agg(**metric_aggs).reset_index())
        by_map.to_csv(root/"aggregate_results_by_map.csv",index=False)

    print("\nFINAL PCA SLZ 03-sample finished")
    print(f"Saved: {root}")
    print(f"Selected: {len(df)} frames total")
    print(f"Scene suffix: {args.scene_suffix}")
    print(f"Frames per scene: {args.frames_per_scene}")
    print(f"Radii: {args.radii}")
    print(f"Slope thresholds: {args.slope_thresholds}")
    print(f"Roughness thresholds: {args.roughness_thresholds}")
    print(f"Semantic hazard IDs: {args.hazard_ids}")
    print("SLZ encoding: 0=Unsafe, 1=Safe, 255=Ignore")
    print("Invalid PCA geometry -> Ignore, not obstacle")
    print("Footprint policy: metric Safety Buffer -> UAV feasible centers -> footprint reconstruction")
    print("Projection: mesh camera-Z depth with fx/fy handled separately")
    print("PCA normal -> Terrain Slope")
    print("sqrt(lambda_min) -> Surface Roughness")

    return run_name
