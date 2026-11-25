"""Prepared-dataset layout: directory structure, frame IDs (RGB file stem), copy / link of
the generated arrays, pose export, and RGB / label pair validation.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pandas as pd

from .._util import mkdir, path_exists, resolve_path
from ..config import GenerationConfig
from ..visualization import save_slz_visualization
from .manifest import physical_family


def safe_link_or_copy(src: Path, dst: Path, mode: str, overwrite: bool):
    src = src.resolve()
    mkdir(dst.parent)

    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()

    if mode == "symlink":
        os.symlink(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(mode)


def copy_generated_file(src: Path, dst: Path, overwrite: bool):
    """Sync generated work output into prepared/ while preserving _work cache.

    Prepared outputs are refreshed from the current work result even without
    --overwrite. This prevents stale SLZ masks when semantic/footprint settings
    change while expensive geometry is reused from cache.
    """
    if not src.exists():
        return False
    mkdir(dst.parent)
    if dst.exists() or dst.is_symlink():
        # Avoid a redundant copy when copy2 metadata already matches exactly.
        try:
            if (
                dst.is_file()
                and not dst.is_symlink()
                and dst.stat().st_size == src.stat().st_size
                and dst.stat().st_mtime_ns == src.stat().st_mtime_ns
            ):
                return True
        except OSError:
            pass
        dst.unlink()
    shutil.copy2(src, dst)
    return True


def create_scene_dirs(scene_root: Path):
    names = [
        "rgb", "depth", "normal", "semantic",
        "slope", "roughness", "slz", "slz_viz", "pose",
    ]
    out = {name: scene_root / name for name in names}
    for p in out.values():
        mkdir(p)
    return out


def cleanup_stale_scene_outputs(prepared_root: Path, scene_df: pd.DataFrame):
    """Remove files not referenced by the newly written scene manifest.

    This prevents stale RGB/SLZ/depth files from older sample sizes or generator
    versions from making directory contents look mis-paired. Only the generated
    modality directories under prepared/<variant>/scenes/<scene>/ are touched.
    """
    if len(scene_df) == 0:
        return

    scene = str(scene_df.iloc[0]["scene"])
    scene_root = prepared_root / "scenes" / scene
    column_by_dir = {
        "rgb": "rgb_path",
        "depth": "depth_path",
        "normal": "normal_path",
        "semantic": "semantic_path",
        "slope": "slope_path",
        "roughness": "roughness_path",
        "slz": "slz_path",
        "slz_viz": "slz_viz_path",
        "pose": "pose_path",
    }

    for dirname, column in column_by_dir.items():
        folder = scene_root / dirname
        if not folder.exists():
            continue

        expected = {
            Path(str(v)).name
            for v in scene_df[column].tolist()
            if str(v).strip() and str(v).lower() != "nan"
        }
        removed = 0
        for p in folder.iterdir():
            if p.is_dir() and not p.is_symlink():
                continue
            if p.name not in expected:
                p.unlink()
                removed += 1
        if removed:
            print(f"[CLEAN] {scene}/{dirname}: removed {removed} stale file(s)")


def validate_prepared_pairs(prepared_root: Path, df: pd.DataFrame):
    """Fail loudly when prepared modalities do not share the RGB stem frame ID."""
    if len(df) == 0:
        raise RuntimeError("Cannot validate an empty prepared manifest.")

    key_dups = df.duplicated(subset=["scene", "frame"], keep=False)
    if key_dups.any():
        bad = df.loc[key_dups, ["scene", "frame", "rgb_path"]].head(20)
        raise RuntimeError(
            "Duplicate (scene, frame) IDs found after RGB-stem normalization:\n"
            + bad.to_string(index=False)
        )

    required_cols = [
        "rgb_path", "depth_path", "normal_path", "slope_path",
        "roughness_path", "slz_path", "slz_viz_path",
    ]
    optional_cols = ["semantic_path", "pose_path"]
    errors = []

    for _, row in df.iterrows():
        scene = str(row["scene"])
        frame = str(row["frame"])

        for col in required_cols + optional_cols:
            value = str(row.get(col, "")).strip()
            if not value or value.lower() == "nan":
                if col in required_cols:
                    errors.append(f"{scene}/{frame}: empty required path {col}")
                continue

            p = prepared_root / value
            if not path_exists(p):
                errors.append(f"{scene}/{frame}: missing {col}: {p}")
                continue
            if p.stem != frame:
                errors.append(
                    f"{scene}/{frame}: stem mismatch {col}: {p.name}"
                )

        if len(errors) >= 50:
            break

    # Also ensure no unreferenced files remain in required modality directories.
    for scene, sdf in df.groupby("scene", sort=True):
        expected_stems = set(sdf["frame"].astype(str))
        for dirname in ["rgb", "depth", "normal", "slope", "roughness", "slz", "slz_viz"]:
            folder = prepared_root / "scenes" / str(scene) / dirname
            if not folder.exists():
                errors.append(f"{scene}: missing modality directory {dirname}: {folder}")
                continue
            actual_stems = {
                p.stem for p in folder.iterdir()
                if not (p.is_dir() and not p.is_symlink())
            }
            if actual_stems != expected_stems:
                missing = sorted(expected_stems - actual_stems)[:10]
                extra = sorted(actual_stems - expected_stems)[:10]
                errors.append(
                    f"{scene}/{dirname}: manifest-directory mismatch "
                    f"missing={missing} extra={extra}"
                )

    if errors:
        raise RuntimeError(
            "Prepared dataset pair validation failed:\n  - "
            + "\n  - ".join(errors[:50])
        )

    print(
        f"[PAIR CHECK] OK: {len(df)} frames; RGB/depth/normal/slope/roughness/SLZ stems match."
    )


def load_pose_info(sampleinfo_path: Path, rgb_name: str, cache: dict):
    key = str(sampleinfo_path)
    if key not in cache:
        infos = json.loads(sampleinfo_path.read_text(encoding="utf-8"))
        cache[key] = {
            str(x.get("OriginalImageName")): x
            for x in infos
            if x.get("OriginalImageName") is not None
        }
    return cache[key].get(rgb_name)


def write_pose_json(dst: Path, info: dict):
    payload = {
        "OriginalImageName": info.get("OriginalImageName"),
        "Width": info.get("Width"),
        "Height": info.get("Height"),
        "P3x3": info.get("P3x3"),
        "T4x4": info.get("T4x4"),
    }
    dst.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def organize_prepared_dataset(
    cfg: GenerationConfig, work_root: Path, prepared_root: Path, run_name: str
):
    """The prepared-dataset builder: RAW-stem-keyed work outputs -> RGB-stem-keyed
    ``prepared/<variant>/scenes/<scene>/{rgb,depth,normal,semantic,slope,roughness,
    slz,slz_viz,pose}/<frame>.*`` + ``manifests/all.csv`` + per-scene CSVs.

    ``args`` is the ``GenerationConfig``.
    """
    args = cfg
    manifest = work_root / "manifests" / "all.csv"
    if not manifest.exists():
        raise FileNotFoundError(manifest)

    df = pd.read_csv(manifest)
    df = df[df["scene"].astype(str).isin(args.scenes)].copy()

    pca_root = (
        work_root
        / "diagnostics"
        / "final_pca_slz_runs"
        / run_name
        / f"radius_{args.pca_radius_m:.3f}m"
    )
    slz_cfg = (
        f"slope_{args.slope_threshold_deg:g}deg_"
        f"rough_{args.roughness_threshold_m:.3f}m"
    )

    rows = []
    pose_cache = {}

    print("\n" + "=" * 80)
    print("[3/4] Organize clean prepared dataset")
    print("Canonical frame ID: RGB basename/stem (extension-independent)")
    print("=" * 80)

    for _, row in df.iterrows():
        scene = str(row["scene"])
        source_frame = str(row["cam_timestamp"])
        rgb_name = str(row["rgb_name"])
        frame = Path(rgb_name).stem

        if not frame or frame in {".", ".."}:
            print(f"[WARN] {scene} {source_frame}: invalid RGB name {rgb_name!r}")
            continue

        d = create_scene_dirs(prepared_root / "scenes" / scene)

        rgb_src = resolve_path(work_root, row.get("rgb_path"))
        sem_src = resolve_path(work_root, row.get("cam_label_id_path"))
        depth_src = resolve_path(work_root, row.get("depth_mesh_path"))
        normal_src = resolve_path(work_root, row.get("normal_vertex_path"))
        sampleinfo_src = resolve_path(work_root, row.get("sampleinfo_json_path"))

        # _work/PCA still uses the historical cam_timestamp as its internal key.
        # Only prepared/ is normalized to the RGB stem.
        slope_src = pca_root / "slope" / f"{source_frame}.npy"
        rough_src = pca_root / "roughness" / f"{source_frame}.npy"
        slz_src = pca_root / "masks" / slz_cfg / f"{source_frame}.png"

        rgb_suffix = Path(rgb_name).suffix or (rgb_src.suffix if rgb_src is not None else ".jpg")
        rgb_dst = d["rgb"] / f"{frame}{rgb_suffix}"
        sem_dst = d["semantic"] / f"{frame}.png"
        depth_dst = d["depth"] / f"{frame}.npy"
        normal_dst = d["normal"] / f"{frame}.npy"
        slope_dst = d["slope"] / f"{frame}.npy"
        rough_dst = d["roughness"] / f"{frame}.npy"
        slz_dst = d["slz"] / f"{frame}.png"
        slz_viz_dst = d["slz_viz"] / f"{frame}.png"
        pose_dst = d["pose"] / f"{frame}.json"

        required = {
            "rgb": rgb_src,
            "depth": depth_src,
            "normal": normal_src,
            "slope": slope_src,
            "roughness": rough_src,
            "slz": slz_src,
        }
        missing = [
            name for name, p in required.items()
            if p is None or not Path(p).exists()
        ]
        if missing:
            print(f"[WARN] {scene} {source_frame}: missing {', '.join(missing)}")
            continue

        safe_link_or_copy(rgb_src, rgb_dst, args.asset_mode, args.overwrite)

        if sem_src is not None and sem_src.exists():
            safe_link_or_copy(sem_src, sem_dst, args.asset_mode, args.overwrite)

        copy_generated_file(depth_src, depth_dst, args.overwrite)
        copy_generated_file(normal_src, normal_dst, args.overwrite)
        copy_generated_file(slope_src, slope_dst, args.overwrite)
        copy_generated_file(rough_src, rough_dst, args.overwrite)
        copy_generated_file(slz_src, slz_dst, args.overwrite)
        # Human-readable visualization only. Never use this RGB image as GT.
        save_slz_visualization(slz_dst, slz_viz_dst)

        if sampleinfo_src is not None and sampleinfo_src.exists():
            info = load_pose_info(sampleinfo_src, rgb_name, pose_cache)
            if info is not None:
                write_pose_json(pose_dst, info)

        rows.append({
            "family": physical_family(scene),
            "scene": scene,
            "frame": frame,
            "cam_timestamp": source_frame,
            "rgb_name": rgb_name,
            "rgb_path": str(rgb_dst.relative_to(prepared_root)),
            "depth_path": str(depth_dst.relative_to(prepared_root)),
            "normal_path": str(normal_dst.relative_to(prepared_root)),
            "semantic_path": (
                str(sem_dst.relative_to(prepared_root))
                if path_exists(sem_dst)
                else ""
            ),
            "slope_path": str(slope_dst.relative_to(prepared_root)),
            "roughness_path": str(rough_dst.relative_to(prepared_root)),
            "slz_path": str(slz_dst.relative_to(prepared_root)),
            "slz_viz_path": str(slz_viz_dst.relative_to(prepared_root)),
            "pose_path": (
                str(pose_dst.relative_to(prepared_root))
                if pose_dst.exists()
                else ""
            ),
        })

    out_df = pd.DataFrame(rows)
    if len(out_df):
        out_df = out_df.sort_values(["family", "scene", "frame"]).reset_index(drop=True)

        # First clean old outputs, then validate exact RGB-stem pairing.
        for _, scene_df in out_df.groupby("scene", sort=True):
            cleanup_stale_scene_outputs(prepared_root, scene_df)
        validate_prepared_pairs(prepared_root, out_df)

    manifests = prepared_root / "manifests"
    mkdir(manifests)
    out_df.to_csv(manifests / "all.csv", index=False)

    # Remove stale per-scene manifest CSVs from previous scene selections.
    current_scene_csvs = {f"{scene}.csv" for scene in out_df["scene"].unique()} if len(out_df) else set()
    for p in manifests.glob("*.csv"):
        if p.name == "all.csv":
            continue
        if p.name not in current_scene_csvs:
            p.unlink()

    if len(out_df):
        for scene in sorted(out_df["scene"].unique()):
            out_df[out_df["scene"] == scene].to_csv(
                manifests / f"{scene}.csv",
                index=False,
            )

    return out_df
