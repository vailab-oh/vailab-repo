#!/usr/bin/env python3
"""Derive a prepared dataset whose SLZ labels use another UAV footprint radius (and,
optionally, another local-PCA radius).

The label = local PCA (slope / roughness / valid) -> semantic veto -> metric safety
buffer -> UAV feasible centers -> landable-region reconstruction. Everything this needs is
stored in the prepared dataset: mesh depth (``depth``, float32, copied unchanged from the
generator's work dir), the pose json (``P3x3`` + original ``Width``/``Height`` + ``T4x4``,
i.e. the generator's sampleinfo entry), and the semantic label file the generator read. So
a variant is built by calling the generator's OWN functions with only the radii changed:

  * footprint only (``--pca-radius-m`` = source's): stored ``slope`` / ``roughness`` are
    reused and ``valid = stored slz != 255`` (the generator's Ignore region is exactly
    ``~valid``); only ``build_slz_label`` is re-run with the new ``uav_radius_m``;
  * PCA radius changed too: ``pca_geometry(D, info, r_pca, ...)`` is re-run on the stored
    depth -> new slope / roughness / valid (so the Ignore region can change), written to the
    variant, then ``build_slz_label``.

Thresholds, hazard IDs, safety buffer, border policy and the PCA support parameters come
from the source's ``dataset_config.yaml``; ``max_metric_radius_px`` (512) and ``row_chunk``
(128) are the generator CLI defaults.

    # The paper's variants (footprint only) -> <PREPARED>_r050, <PREPARED>_r020
    python tools/make_radius_variant.py --src-root <PREPARED> --uav-radius-m 0.50
    python tools/make_radius_variant.py --src-root <PREPARED> --uav-radius-m 0.20
    # Check 1: recompute the labels at the source radii from the stored slope/roughness.
    python tools/make_radius_variant.py --src-root <PREPARED> --check
    # Check 2: the same, but re-running PCA from the stored depth (slope/roughness must match too).
    python tools/make_radius_variant.py --src-root <PREPARED> --check --recompute-pca
    # PCA radius changed as well -> <PREPARED>_r020_pca035
    python tools/make_radius_variant.py --src-root <PREPARED> --uav-radius-m 0.20 --pca-radius-m 0.35

Output layout: per-frame directories that are unchanged (``rgb``, ``depth``, ``normal`` =
rendered mesh normal, ``semantic``, ``pose``, and ``slope``/``roughness`` when PCA is
reused) are *relative* symlinks to the source scene's directory (so the variant also works
after an rsync next to the source on another host); ``slz``/``slz_viz`` (+ ``slope``/
``roughness`` when PCA is re-run) are written fresh; ``manifests/`` is copied as-is (paths
are root-relative, so frame lists and splits are identical) EXCEPT every
``class_weights.json``, which depends on the labels: install the variant's shipped weights
with ``prepare_data.py --stage splits --variant r050|r020``, or recompute them with
``tools/compute_class_weights.py --root <variant>``.
``label_stats.csv`` holds per-frame pixel counts for the new and the source label, and
``radius_variant_info.json`` the parameters + a hash over all new label arrays.
"""

from __future__ import annotations

import os

# one BLAS thread per worker process (set before numpy is imported)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ALWAYS_LINKED = ("rgb", "depth", "normal", "semantic", "pose")
PCA_DIRS = ("slope", "roughness")
MAX_METRIC_RADIUS_PX = 512  # generator CLI default (--max-metric-radius-px)
ROW_CHUNK = 128             # generator CLI default (--row-chunk)

_P: dict = {}  # per-worker parameters (set by _init)


def _load_policy(src_root: Path) -> dict:
    cfg = yaml.safe_load((src_root / "dataset_config.yaml").read_text())
    t = cfg["slz_thresholds"]
    pca = cfg["local_pca"]
    if not t.get("footprint_filter", False):
        raise SystemExit("source dataset was generated without the footprint filter")
    if cfg["slz_target"]["algorithm_version"] != "v11_metric_landable_region":
        raise SystemExit(f"unsupported algorithm_version {cfg['slz_target']['algorithm_version']}")
    return {
        "slope_deg": float(t["slope_deg"]),
        "roughness_m": float(t["roughness_m"]),
        "hazard_ids": tuple(int(x) for x in t["semantic_hazard_ids"]),
        "safety_buffer_m": float(t["safety_buffer_m"]),
        "uav_radius_m": float(t["uav_radius_m"]),
        "footprint_border_as_unsafe": bool(t["footprint_border_as_unsafe"]),
        "max_metric_radius_px": MAX_METRIC_RADIUS_PX,
        "pca_radius_m": float(pca["radius_m"]),
        "min_support": float(pca["min_support"]),
        "min_points": int(pca["min_points"]),
        "up_axis": str(pca["world_up_axis"]),
        "row_chunk": ROW_CHUNK,
    }


def _init(params: dict) -> None:
    import cv2

    cv2.setNumThreads(1)
    _P.update(params)


def _label_one(row: dict) -> dict:
    from slz_benchmark.generation.geometry.pca import pca_geometry
    from slz_benchmark.generation.geometry.projection import scaled_K
    from slz_benchmark.generation.slz.label import build_slz_label
    from slz_benchmark.generation.slz.semantic import semantic_safe
    from slz_benchmark.generation.visualization import colorize_slz_label

    src = Path(_P["src_root"])
    D = np.asarray(np.load(src / row["depth_path"]), np.float32)
    old = np.array(Image.open(src / row["slz_path"]))
    info = json.loads((src / row["pose_path"]).read_text())
    sem = semantic_safe(src / row["semantic_path"], D.shape, _P["hazard_ids"])
    geom_diff = 0
    if _P["recompute_pca"]:
        slope, rough, valid, K, _, _ = pca_geometry(
            D, info, _P["pca_r"], _P["min_support"], _P["min_points"],
            _P["max_metric_radius_px"], _P["row_chunk"], _P["up_axis"])
        if _P["dst_root"] is None:  # check mode: geometry must equal the stored arrays
            s0 = np.load(src / row["slope_path"])
            r0 = np.load(src / row["roughness_path"])
            geom_diff = int((slope != s0).sum() + (rough != r0).sum() + (valid != (old != 255)).sum())
    else:
        slope = np.asarray(np.load(src / row["slope_path"]), np.float32)
        rough = np.asarray(np.load(src / row["roughness_path"]), np.float32)
        valid = old != 255
        K = scaled_K(info, D.shape)

    new = build_slz_label(
        slope, rough, valid, sem, _P["slope_deg"], _P["roughness_m"], D, K,
        with_footprint=True,
        uav_radius_m=_P["uav_r"],
        safety_buffer_m=_P["safety_buffer_m"],
        footprint_border_as_unsafe=_P["footprint_border_as_unsafe"],
        max_metric_radius_px=_P["max_metric_radius_px"],
    )["final_label"]

    if _P["dst_root"] is not None:
        dst = Path(_P["dst_root"])
        Image.fromarray(new).save(dst / row["slz_path"])
        Image.fromarray(colorize_slz_label(new), mode="RGB").save(dst / row["slz_viz_path"])
        if _P["recompute_pca"]:
            np.save(dst / row["slope_path"], slope)
            np.save(dst / row["roughness_path"], rough)

    return {
        "family": row["family"], "scene": row["scene"], "frame": row["frame"],
        "n_safe": int((new == 1).sum()), "n_unsafe": int((new == 0).sum()),
        "n_ignore": int((new == 255).sum()),
        "src_n_safe": int((old == 1).sum()), "src_n_unsafe": int((old == 0).sum()),
        "src_n_ignore": int((old == 255).sum()),
        "n_changed": int((new != old).sum()),
        "n_safe_to_unsafe": int(((old == 1) & (new == 0)).sum()),
        "n_unsafe_to_safe": int(((old == 0) & (new == 1)).sum()),
        "n_geometry_diff": geom_diff,
        "sha256": hashlib.sha256(new.tobytes()).hexdigest(),
    }


def _prepare_dst(src: Path, dst: Path, df: pd.DataFrame, policy: dict,
                 uav_r: float, pca_r: float, recompute_pca: bool) -> None:
    linked = ALWAYS_LINKED + (() if recompute_pca else PCA_DIRS)
    fresh = ("slz", "slz_viz") + (PCA_DIRS if recompute_pca else ())
    for scene in sorted(df["scene"].unique()):
        s_src, s_dst = src / "scenes" / scene, dst / "scenes" / scene
        s_dst.mkdir(parents=True)
        for sub in linked:
            if (s_src / sub).exists():
                os.symlink(os.path.relpath(s_src / sub, s_dst), s_dst / sub)
        for sub in fresh:
            (s_dst / sub).mkdir()
    # manifests: copied unchanged, minus the label-dependent class weights
    shutil.copytree(src / "manifests", dst / "manifests",
                    ignore=shutil.ignore_patterns("class_weights.json"))
    cfg = yaml.safe_load((src / "dataset_config.yaml").read_text())
    cfg["variant"] = dst.name
    cfg["slz_thresholds"]["uav_radius_m"] = uav_r
    cfg["local_pca"]["radius_m"] = pca_r
    changed = {"uav_radius_m": {"source": policy["uav_radius_m"], "variant": uav_r}}
    if recompute_pca:
        changed["pca_radius_m"] = {"source": policy["pca_radius_m"], "variant": pca_r}
    cfg["derived_from"] = {
        "source_variant": src.name,
        "tool": "tools/make_radius_variant.py",
        "changed": changed,
        "regenerated": list(fresh),
        "symlinked_to_source": list(linked),
        "note": ("local PCA re-run (generator pca_geometry) on the source's stored mesh depth; "
                 if recompute_pca else
                 "source's stored slope/roughness reused, valid = source slz != 255; ")
                + "labels from the generator's build_slz_label",
    }
    (dst / "dataset_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


def _tag(x: float) -> str:
    return f"{round(x * 100):03d}"


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--src-root", type=Path, required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--uav-radius-m", type=float, help="build a variant with this UAV radius")
    g.add_argument("--check", action="store_true",
                   help="recompute at the source's own radii; fail unless 0 pixels change")
    ap.add_argument("--pca-radius-m", type=float, default=None,
                    help="local-PCA radius of the variant (default: the source's; a different "
                         "value re-runs PCA from the stored depth)")
    ap.add_argument("--recompute-pca", action="store_true",
                    help="--check only: re-run PCA at the source radius instead of reusing "
                         "the stored slope/roughness (also checks those arrays bit-exactly)")
    ap.add_argument("--dst-root", type=Path, default=None,
                    help="default: <src-root>_r<UAV r, cm> (+ _pca<PCA r, cm> if PCA changes), "
                         "e.g. _r020 or _r020_pca035")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--save-json", type=Path, default=None, help="--check: write the report here")
    args = ap.parse_args()

    src = args.src_root.resolve()
    policy = _load_policy(src)
    df = pd.read_csv(src / "manifests" / "all.csv", dtype={"frame": str})
    rows = df.to_dict("records")

    if args.check:
        if args.pca_radius_m is not None:
            ap.error("--check always uses the source's radii; drop --pca-radius-m")
        uav_r, pca_r, dst = policy["uav_radius_m"], policy["pca_radius_m"], None
        recompute = args.recompute_pca
    else:
        if args.recompute_pca:
            ap.error("--recompute-pca is for --check; a variant re-runs PCA iff --pca-radius-m differs")
        uav_r = float(args.uav_radius_m)
        pca_r = policy["pca_radius_m"] if args.pca_radius_m is None else float(args.pca_radius_m)
        if uav_r <= 0 or pca_r <= 0:
            ap.error("radii must be > 0")
        recompute = not np.isclose(pca_r, policy["pca_radius_m"])
        if np.isclose(uav_r, policy["uav_radius_m"]) and not recompute:
            ap.error("both radii equal the source's; use --check instead")
        name = f"{src.name}_r{_tag(uav_r)}" + (f"_pca{_tag(pca_r)}" if recompute else "")
        dst = (args.dst_root or src.parent / name).resolve()
        if dst.exists():
            ap.error(f"{dst} already exists; remove it first (it is derived, safe to rebuild)")
        _prepare_dst(src, dst, df, policy, uav_r, pca_r, recompute)

    params = {**policy, "src_root": str(src), "dst_root": str(dst) if dst else None,
              "uav_r": uav_r, "pca_r": pca_r, "recompute_pca": recompute}
    print(f"source : {src}\nUAV r  : {uav_r} m (source {policy['uav_radius_m']} m)\n"
          f"PCA r  : {pca_r} m (source {policy['pca_radius_m']} m) -> "
          f"{'re-run from stored depth' if recompute else 'stored slope/roughness reused'}\n"
          f"fixed  : buffer {policy['safety_buffer_m']} m, slope {policy['slope_deg']} deg, "
          f"roughness {policy['roughness_m']} m\nframes : {len(rows)}  workers: {args.workers}\n"
          f"output : {dst if dst else '(check only, nothing written)'}")

    with Pool(args.workers, initializer=_init, initargs=(params,)) as pool:
        stats = list(tqdm(pool.imap(_label_one, rows, chunksize=8), total=len(rows),
                          desc="relabel", dynamic_ncols=True))
    st = pd.DataFrame(stats)
    combined = hashlib.sha256("".join(st["sha256"]).encode()).hexdigest()

    n_changed = int(st["n_changed"].sum())
    n_geom = int(st["n_geometry_diff"].sum())
    n_px = int(len(st) * np.prod(np.array(Image.open(src / rows[0]["slz_path"])).shape))
    lab = lambda s, u: float(s.sum() / (s.sum() + u.sum()))
    report = {
        "source": str(src), "uav_radius_m": uav_r, "source_uav_radius_m": policy["uav_radius_m"],
        "pca_radius_m": pca_r, "source_pca_radius_m": policy["pca_radius_m"],
        "pca_recomputed": bool(recompute),
        "policy": {k: (list(v) if isinstance(v, tuple) else v) for k, v in policy.items()},
        "n_frames": len(st), "n_pixels_changed_vs_source": n_changed,
        "frames_changed_vs_source": int((st["n_changed"] > 0).sum()),
        "safe_fraction_labeled": lab(st["n_safe"], st["n_unsafe"]),
        "source_safe_fraction_labeled": lab(st["src_n_safe"], st["src_n_unsafe"]),
        "ignore_fraction": float(st["n_ignore"].sum() / n_px),
        "source_ignore_fraction": float(st["src_n_ignore"].sum() / n_px),
        "label_arrays_sha256": combined,
    }
    if args.check and recompute:
        report["n_geometry_values_differing"] = n_geom
    print(json.dumps({k: v for k, v in report.items() if k != "policy"}, indent=2))

    if args.check:
        if args.save_json:
            args.save_json.parent.mkdir(parents=True, exist_ok=True)
            args.save_json.write_text(json.dumps(report, indent=2))
        if n_changed or n_geom:
            bad = st[(st["n_changed"] > 0) | (st["n_geometry_diff"] > 0)]
            print(bad[["scene", "frame", "n_changed", "n_geometry_diff"]].head(20).to_string(index=False))
            raise SystemExit(f"CHECK FAILED: {n_changed} / {n_px} label pixels and {n_geom} "
                             "geometry values differ from the stored dataset")
        print(f"CHECK PASSED: all {len(st)} frames reproduce the stored labels exactly"
              + (" (PCA re-run from depth; slope/roughness/valid bit-identical)" if recompute else ""))
        return

    st.to_csv(dst / "label_stats.csv", index=False)
    (dst / "radius_variant_info.json").write_text(json.dumps(report, indent=2))
    print(f"[DONE] {dst}")


if __name__ == "__main__":
    main()
