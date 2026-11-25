"""Render stage: raw frames + terra mesh -> RGB-aligned depth / normal maps.

``run_preprocessing(cfg, work_root)`` discovers the scenes, renders (and caches) the mesh
depth and normals for every frame, and writes the frame manifest and a report. ``args``
is a local ``Namespace`` built from the ``GenerationConfig``.
"""

from __future__ import annotations

import json
import os
import shutil
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image

from .._util import log, mkdir
from ..config import GenerationConfig
from .mesh import find_meshes, load_terra_ply
from .renderer import render_mesh
from .raw_scenes import (
    cam_label_dir,
    find_scene_dirs,
    sample_frame_files,
    scene_from_raw_dir,
    scene_map_group,
)

MANIFEST_COLUMNS = [
    "map_group", "scene", "interval",
    "rgb_name", "cam_timestamp", "rgb_path",
    "cam_label_id_path", "cam_label_color_path",
    "depth_mesh_path", "normal_vertex_path", "normal_face_path",
    "sampleinfo_json_path", "rtk_csv_path", "split",
]


def rel(p: Optional[Path], base: Path) -> str:
    if p is None:
        return ""
    try:
        return str(p.relative_to(base))
    except Exception:
        return str(p)


def manifest_path(src: Path, dst: Path, processed_root: Path, link_mode: str) -> str:
    """
    If link_mode is none, keep the raw source path in manifest.
    Otherwise keep the processed path.
    """
    return rel(src.resolve() if link_mode == "none" else dst, processed_root)


def array_output_path(path: Path, fmt: str) -> Path:
    if fmt == "npz":
        return path.with_suffix(".npz")
    return path.with_suffix(".npy")


def save_array(path: Path, arr: np.ndarray, dtype, fmt: str, key: str = "data") -> Path:
    """
    Save dense pseudo GT array.

    npy:
      Faster dataloading, memory-map possible, but larger.

    npz:
      Smaller because compressed, but slower dataloading and no simple mmap.
    """
    out = array_output_path(path, fmt)
    mkdir(out.parent)
    arr = arr.astype(dtype)
    if fmt == "npz":
        np.savez_compressed(out, **{key: arr})
    elif fmt == "npy":
        np.save(out, arr)
    else:
        raise ValueError(f"Unknown array format: {fmt}")
    return out


def link_or_copy(src: Path, dst: Path, mode: str, overwrite: bool = False):
    """
    mode:
      - symlink/copy/hardlink: materialize raw file under processed tree
      - none: do not materialize raw file; manifest points to raw path directly
    """
    if mode == "none":
        return
    if not src or not src.exists():
        return
    mkdir(dst.parent)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        raise ValueError(mode)


def make_pca_input_dirs(scene_interval_root: Path) -> Dict[str, Path]:
    d = {
        "rgb": scene_interval_root / "rgb",
        "pose": scene_interval_root / "pose",
        "cam_label_color": scene_interval_root / "gt" / "cam_label_color",
        "cam_label_id": scene_interval_root / "gt" / "cam_label_id",
        "depth_mesh": scene_interval_root / "pseudo" / "depth_mesh",
        "normal_vertex": scene_interval_root / "pseudo" / "normal_vertex",
        "normal_face": scene_interval_root / "pseudo" / "normal_face",
    }
    for x in d.values():
        mkdir(x)
    return d


def run_preprocessing(cfg: GenerationConfig, work_root: Path) -> None:
    """RAW -> RGB-aligned mesh depth/normal, for every frame of every scene in
    ``cfg.scenes``. Writes ``work_root/manifests/all.csv`` (+ reports) and the per-scene
    ``pseudo/{depth_mesh,normal_vertex}/<cam_ts>.npy`` arrays.

    ``args`` is a local ``Namespace`` built from ``cfg`` plus fixed settings
    (``sample_mode="uniform"``, ``array_format="npy"``, ``link_mode="none"``, no previews).
    """
    args = Namespace(
        root=cfg.root,
        raw_dir=cfg.raw_dir,
        mesh_root=cfg.mesh_root,
        out_dir=work_root,
        intervals=list(cfg.intervals),
        scene_suffix="",
        scene_filter=list(cfg.scenes),
        frames_per_scene=cfg.frames_per_scene,
        sample_mode="uniform",
        sample_seed=cfg.sample_seed,
        target_size=[cfg.height, cfg.width],
        array_format="npy",
        link_mode="none",
        overwrite=cfg.overwrite,
        save_normal_face=False,
        save_depth_preview=False,
    )

    root = args.root
    raw = args.raw_dir or (root / "raw")
    out = args.out_dir or (root / "processed" / "08.28_pca_03_200")
    mesh_root = args.mesh_root or raw

    for p in [out, out / "manifests", out / "stats", out / "configs", out / "cache" / "ply"]:
        mkdir(p)

    cfg_dump = {}
    for k, v in vars(args).items():
        if isinstance(v, Path):
            cfg_dump[k] = str(v)
        elif isinstance(v, (list, tuple)):
            cfg_dump[k] = [str(x) if isinstance(x, Path) else x for x in v]
        else:
            cfg_dump[k] = v
    (out / "configs" / "prepare_pca_inputs_03_uniform.json").write_text(
        json.dumps(cfg_dump, indent=2), encoding="utf-8"
    )

    all_scene_dirs = []
    available = []
    for interval in args.intervals:
        dirs = find_scene_dirs(raw, interval)
        for d in dirs:
            scene = scene_from_raw_dir(d, interval)
            available.append((interval, scene, d.name))

        # Explicit scene list, if provided, takes precedence over suffix auto-filter.
        if args.scene_filter:
            allowed = set(args.scene_filter)
            dirs = [d for d in dirs if scene_from_raw_dir(d, interval) in allowed]
        else:
            dirs = [d for d in dirs if scene_from_raw_dir(d, interval).endswith(args.scene_suffix)]

        all_scene_dirs.extend((interval, d) for d in dirs)

    if not all_scene_dirs:
        print("[ERROR] No scene folders selected.", flush=True)
        print(f"[ERROR] scene suffix = {args.scene_suffix!r}", flush=True)
        print("[AVAILABLE SCENES]", flush=True)
        for interval, scene, raw_name in available[:200]:
            print(f"  {interval}: {raw_name} -> {scene}", flush=True)
        raise SystemExit(1)

    print("\n=== Selected scenes ===", flush=True)
    for interval, d in all_scene_dirs:
        print(f"  {scene_from_raw_dir(d, interval)} / {interval}", flush=True)

    rows = []
    missing = []
    sampling_report = []

    for interval, raw_scene_dir in all_scene_dirs:
        raw_scene = raw_scene_dir.name
        scene = scene_from_raw_dir(raw_scene_dir, interval)
        group = scene_map_group(scene)
        log(f"\n[SCENE] {scene} / {interval}")

        scene_out = out / "scenes" / scene / interval
        dirs = make_pca_input_dirs(scene_out)

        rgb_dir = raw_scene_dir / f"{interval}_CAM"
        sampleinfo = raw_scene_dir / "sampleinfos_interpolated.json"
        rtk = raw_scene_dir / "rtk_positions_raw.csv"

        if not rgb_dir.exists():
            missing.append({"scene": scene, "interval": interval, "missing": f"rgb_dir:{rgb_dir}"})
            continue
        if not sampleinfo.exists():
            missing.append({"scene": scene, "interval": interval, "missing": "sampleinfos_interpolated.json"})
            continue

        link_or_copy(sampleinfo, dirs["pose"] / "sampleinfos_interpolated.json", args.link_mode, args.overwrite)
        if rtk.exists():
            link_or_copy(rtk, dirs["pose"] / "rtk_positions_raw.csv", args.link_mode, args.overwrite)

        infos = json.loads(sampleinfo.read_text(encoding="utf-8"))
        info_map = {x["OriginalImageName"]: x for x in infos if "OriginalImageName" in x}

        cam_color_dir = cam_label_dir(raw, interval, raw_scene, "color")
        cam_id_dir = cam_label_dir(raw, interval, raw_scene, "id")

        mesh_paths = find_meshes(mesh_root, interval, scene)
        if not mesh_paths:
            missing.append({"scene": scene, "interval": interval, "missing": "mesh"})
            log(f"[WARN] no mesh found for {scene}")
            continue

        mesh_assets = []
        for mesh in mesh_paths:
            try:
                V_i, F_i = load_terra_ply(mesh, out / "cache" / "ply")
                mesh_assets.append((mesh, V_i, F_i))
                log(f"[MESH] {mesh}")
            except Exception as e:
                missing.append({"scene": scene, "interval": interval, "missing": f"mesh_load_failed:{mesh}:{e}"})
        if not mesh_assets:
            continue
        if len(mesh_assets) > 1:
            log(f"[MESH] using {len(mesh_assets)} blocks for {scene}")

        all_rgb = sorted(rgb_dir.glob("*.jpg"))
        selected_rgb = sample_frame_files(
            all_rgb,
            n=args.frames_per_scene,
            mode=args.sample_mode,
            seed=args.sample_seed,
        )
        sampling_report.append({
            "map_group": group,
            "scene": scene,
            "interval": interval,
            "available_frames": len(all_rgb),
            "selected_frames": len(selected_rgb),
            "sample_mode": args.sample_mode,
            "scene_suffix": args.scene_suffix,
        })
        log(f"[SAMPLE] {scene}: {len(selected_rgb)} / {len(all_rgb)} ({args.sample_mode})")

        for idx, rgb in enumerate(selected_rgb, 1):
            cam_ts = rgb.stem
            info = info_map.get(rgb.name)
            if info is None:
                missing.append({"scene": scene, "interval": interval, "rgb_name": rgb.name, "missing": "sampleinfo_pose"})
                continue

            row = {k: "" for k in MANIFEST_COLUMNS}
            row.update({
                "map_group": group,
                "scene": scene,
                "interval": interval,
                "rgb_name": rgb.name,
                "cam_timestamp": cam_ts,
                "split": "",
                "sampleinfo_json_path": manifest_path(
                    sampleinfo, dirs["pose"] / "sampleinfos_interpolated.json", out, args.link_mode
                ),
                "rtk_csv_path": manifest_path(
                    rtk, dirs["pose"] / "rtk_positions_raw.csv", out, args.link_mode
                ) if rtk.exists() else "",
            })

            # RGB: by default link_mode=none, so no duplicate RGB storage.
            rgb_dst = dirs["rgb"] / rgb.name
            link_or_copy(rgb, rgb_dst, args.link_mode, args.overwrite)
            row["rgb_path"] = manifest_path(rgb, rgb_dst, out, args.link_mode)

            # Camera semantic GT.
            cam_id = cam_id_dir / f"{cam_ts}.png"
            cam_color = cam_color_dir / f"{cam_ts}.png"
            cam_id_dst = dirs["cam_label_id"] / f"{cam_ts}.png"
            cam_color_dst = dirs["cam_label_color"] / f"{cam_ts}.png"

            if cam_id.exists():
                link_or_copy(cam_id, cam_id_dst, args.link_mode, args.overwrite)
                row["cam_label_id_path"] = manifest_path(cam_id, cam_id_dst, out, args.link_mode)
            else:
                missing.append({"scene": scene, "interval": interval, "rgb_name": rgb.name, "missing": "cam_label_id"})

            if cam_color.exists():
                link_or_copy(cam_color, cam_color_dst, args.link_mode, args.overwrite)
                row["cam_label_color_path"] = manifest_path(cam_color, cam_color_dst, out, args.link_mode)

            # Output size: CLI order is H W.
            with Image.open(rgb) as im:
                W0, H0 = im.size
            H, W = tuple(args.target_size) if args.target_size else (H0, W0)

            # Automatic resume/cache: rendered geometry is expensive but independent
            # of semantic SLZ policy. Reuse complete frame outputs unless --overwrite.
            pD_expected = array_output_path(
                dirs["depth_mesh"] / f"{cam_ts}.npy", args.array_format
            )
            pNV_expected = array_output_path(
                dirs["normal_vertex"] / f"{cam_ts}.npy", args.array_format
            )
            pNF_expected = array_output_path(
                dirs["normal_face"] / f"{cam_ts}.npy", args.array_format
            )
            prep_cache_ready = (
                pD_expected.exists()
                and pNV_expected.exists()
                and (not args.save_normal_face or pNF_expected.exists())
            )

            if prep_cache_ready and not args.overwrite:
                row["depth_mesh_path"] = rel(pD_expected, out)
                row["normal_vertex_path"] = rel(pNV_expected, out)
                if args.save_normal_face:
                    row["normal_face_path"] = rel(pNF_expected, out)
                log(f"  [CACHE preprocess] {rgb.name}")
            else:
                try:
                    # Z-buffer composite all Terra blocks into this RGB camera view.
                    D = np.zeros((H, W), dtype=np.float32)
                    NF = np.zeros((H, W, 3), dtype=np.float32)
                    NV = np.zeros((H, W, 3), dtype=np.float32)

                    for _mesh_path, V_i, F_i in mesh_assets:
                        D_i, NF_i, NV_i = render_mesh(V_i, F_i, info, (H, W))
                        valid_i = D_i > 0
                        replace = valid_i & ((D == 0) | (D_i < D))
                        D[replace] = D_i[replace]
                        NF[replace] = NF_i[replace]
                        NV[replace] = NV_i[replace]

                    pD = save_array(
                        dirs["depth_mesh"] / f"{cam_ts}.npy",
                        D, np.float32, args.array_format, key="depth_mesh"
                    )
                    pNV = save_array(
                        dirs["normal_vertex"] / f"{cam_ts}.npy",
                        NV, np.float16, args.array_format, key="normal_vertex"
                    )
                    row["depth_mesh_path"] = rel(pD, out)
                    row["normal_vertex_path"] = rel(pNV, out)

                    if args.save_normal_face:
                        pNF = save_array(
                            dirs["normal_face"] / f"{cam_ts}.npy",
                            NF, np.float16, args.array_format, key="normal_face"
                        )
                        row["normal_face_path"] = rel(pNF, out)

                    if args.save_depth_preview:
                        valid = D > 0
                        vis = np.zeros_like(D, dtype=np.uint8)
                        if np.any(valid):
                            lo, hi = np.percentile(D[valid], [2, 98])
                            if hi > lo:
                                vis[valid] = np.clip((D[valid] - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
                        prev_dir = scene_out / "viz" / "depth_mesh"
                        mkdir(prev_dir)
                        Image.fromarray(vis).save(prev_dir / f"{cam_ts}.png")

                except Exception as e:
                    missing.append({
                        "scene": scene, "interval": interval, "rgb_name": rgb.name,
                        "missing": f"mesh_render_failed:{type(e).__name__}:{e}"
                    })
                    log(f"[ERROR] {scene} {rgb.name}: {e}")
                    continue

            rows.append(row)
            if idx == 1 or idx % 10 == 0 or idx == len(selected_rgb):
                log(f"  [{idx}/{len(selected_rgb)}] {rgb.name}")

    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    df.to_csv(out / "manifests" / "all.csv", index=False)
    if len(df):
        for interval in sorted(df["interval"].dropna().unique()):
            df[df["interval"] == interval].to_csv(
                out / "manifests" / f"all_{interval}.csv", index=False
            )

    pd.DataFrame(sampling_report).to_csv(out / "manifests" / "sampling_report.csv", index=False)
    pd.DataFrame(missing).to_csv(out / "manifests" / "missing_files.csv", index=False)

    selected_cols = [c for c in ["map_group", "scene", "interval", "rgb_name", "cam_timestamp"] if c in df.columns]
    if len(df):
        df[selected_cols].to_csv(out / "manifests" / "selected_frames.csv", index=False)

    summary = {
        "num_rows": int(len(df)),
        "num_missing_records": int(len(missing)),
        "scenes": sorted(df["scene"].dropna().unique().tolist()) if len(df) else [],
        "intervals": sorted(df["interval"].dropna().unique().tolist()) if len(df) else [],
        "purpose": "PCA SLZ preprocessing inputs only",
        "contains_ransac": False,
        "contains_slope_or_roughness": False,
        "generated": ["depth_mesh", "normal_vertex"],
        "linked_or_referenced": ["rgb", "cam_label_id", "cam_label_color", "sampleinfos_interpolated", "rtk_csv"],
        "target_size_hw": list(args.target_size) if args.target_size else None,
        "sampling": f"{args.sample_mode}, {args.frames_per_scene} frames per selected scene",
    }
    (out / "stats" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log("\n=== Sampling summary ===")
    if sampling_report:
        for r in sampling_report:
            log(f"  {r['scene']}: {r['selected_frames']} / {r['available_frames']}")
    log(f"TOTAL processed rows: {len(df)}")
    log(f"Manifest: {out / 'manifests' / 'all.csv'}")
    log(f"[DONE] {out}")
