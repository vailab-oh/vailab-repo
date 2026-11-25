#!/usr/bin/env python3
"""Sanity checks of a prepared dataset.

Checks
------
1. A few individual samples:
   - dtype
   - tensor shape
   - min/max
   - spatial alignment

2. Full split statistics:
   - frame count
   - scene/family distribution
   - SLZ safe/unsafe class balance
   - depth valid ratio + valid depth range/mean
   - normal valid ratio
   - normal unit-length sanity
   - geometry_valid ratio
   - slope/roughness finite-value sanity
   - semantic class histogram
   - spatial mismatch count

3. Per-scene statistics:
   - frames
   - SLZ Safe %
   - Depth valid %
   - Normal valid %
   - Geometry valid %
   - mean Depth
   - mean Slope
   - mean Roughness

Examples
--------
Generator family split (fixed train/val/test), all tasks:
    python tools/verify_dataset.py --root <PREPARED> --split all

One fold of the paper's split, SLZ only, saved as JSON:
    python tools/verify_dataset.py --root <PREPARED> --scheme loso --fold 1 \
      --split all --tasks slz --save-json stats_loso_fold1.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slz_benchmark.datasets.uavscenes import build_dataset


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def describe(v):
    if torch.is_tensor(v):
        s = f"dtype={v.dtype}, shape={tuple(v.shape)}"
        if v.numel() and v.dtype != torch.bool:
            if torch.is_floating_point(v):
                finite = torch.isfinite(v)
                if finite.any():
                    vv = v[finite]
                    s += f", min={vv.min().item():.4g}, max={vv.max().item():.4g}"
            else:
                s += f", min={v.min().item()}, max={v.max().item()}"
        return s
    return repr(v)


def human_int(x: int) -> str:
    return f"{int(x):,}"


def safe_pct(num, den):
    if den == 0:
        return float("nan")
    return 100.0 * float(num) / float(den)


def fmt(x, digits=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.{digits}f}"


# ---------------------------------------------------------------------
# Generic accumulator
# ---------------------------------------------------------------------

class BasicStats:
    """
    Accumulates statistics for either:
      - the entire split
      - one scene
      - one family
    """

    def __init__(self, normal_tol: float):
        self.normal_tol = float(normal_tol)

        self.frames = 0
        self.spatial_mismatch = 0

        # SLZ (0 = Unsafe, 1 = Safe, 255 = Ignore)
        self.slz_total = 0
        self.slz_safe = 0
        self.slz_unsafe = 0
        self.slz_ignore = 0
        self.slz_invalid_value = 0

        # Semantic
        self.semantic_hist = Counter()

        # Depth
        self.depth_total = 0
        self.depth_valid = 0
        self.depth_sum = 0.0
        self.depth_min = math.inf
        self.depth_max = -math.inf

        # Normal
        self.normal_total = 0
        self.normal_valid = 0
        self.normal_norm_sum = 0.0
        self.normal_norm_error_sum = 0.0
        self.normal_norm_error_max = 0.0
        self.normal_within_tol = 0

        # Geometry
        self.geometry_total = 0
        self.geometry_valid = 0

        # Slope
        self.slope_total = 0
        self.slope_finite = 0
        self.slope_sum = 0.0
        self.slope_min = math.inf
        self.slope_max = -math.inf

        # Roughness
        self.rough_total = 0
        self.rough_finite = 0
        self.rough_sum = 0.0
        self.rough_min = math.inf
        self.rough_max = -math.inf

    def update(self, sample: dict):
        self.frames += 1

        image_hw = tuple(sample["image"].shape[-2:])
        for key in ("semantic", "slz", "depth", "normal", "slope", "roughness"):
            if key in sample and tuple(sample[key].shape[-2:]) != image_hw:
                self.spatial_mismatch += 1

        # SLZ
        if "slz" in sample:
            x = sample["slz"]
            self.slz_total += x.numel()
            self.slz_safe += int((x == 1).sum().item())
            self.slz_unsafe += int((x == 0).sum().item())
            self.slz_ignore += int((x == 255).sum().item())
            self.slz_invalid_value += int(
                ((x != 0) & (x != 1) & (x != 255)).sum().item()
            )

        # Semantic
        if "semantic" in sample:
            x = sample["semantic"].reshape(-1)
            values, counts = torch.unique(x, return_counts=True)
            for v, c in zip(values.tolist(), counts.tolist()):
                self.semantic_hist[int(v)] += int(c)

        # Depth
        if "depth" in sample:
            d = sample["depth"]
            valid = sample.get("depth_valid", torch.isfinite(d) & (d > 0))
            self.depth_total += valid.numel()
            nvalid = int(valid.sum().item())
            self.depth_valid += nvalid

            if nvalid:
                vv = d[valid]
                self.depth_sum += float(vv.double().sum().item())
                self.depth_min = min(self.depth_min, float(vv.min().item()))
                self.depth_max = max(self.depth_max, float(vv.max().item()))

        # Normal
        if "normal" in sample:
            n = sample["normal"]
            valid = sample.get("normal_valid")
            if valid is None:
                finite = torch.isfinite(n).all(dim=0, keepdim=True)
                norms = torch.linalg.vector_norm(torch.nan_to_num(n), dim=0, keepdim=True)
                valid = finite & (norms > 1e-6)

            norms = torch.linalg.vector_norm(n, dim=0, keepdim=True)
            self.normal_total += valid.numel()
            nvalid = int(valid.sum().item())
            self.normal_valid += nvalid

            if nvalid:
                vv = norms[valid].double()
                err = torch.abs(vv - 1.0)
                self.normal_norm_sum += float(vv.sum().item())
                self.normal_norm_error_sum += float(err.sum().item())
                self.normal_norm_error_max = max(
                    self.normal_norm_error_max,
                    float(err.max().item())
                )
                self.normal_within_tol += int((err <= self.normal_tol).sum().item())

        # Geometry valid
        if "geometry_valid" in sample:
            g = sample["geometry_valid"]
            self.geometry_total += g.numel()
            self.geometry_valid += int(g.sum().item())

        # Slope
        if "slope" in sample:
            x = sample["slope"]
            finite = torch.isfinite(x)
            self.slope_total += x.numel()
            nf = int(finite.sum().item())
            self.slope_finite += nf
            if nf:
                vv = x[finite]
                self.slope_sum += float(vv.double().sum().item())
                self.slope_min = min(self.slope_min, float(vv.min().item()))
                self.slope_max = max(self.slope_max, float(vv.max().item()))

        # Roughness
        if "roughness" in sample:
            x = sample["roughness"]
            finite = torch.isfinite(x)
            self.rough_total += x.numel()
            nf = int(finite.sum().item())
            self.rough_finite += nf
            if nf:
                vv = x[finite]
                self.rough_sum += float(vv.double().sum().item())
                self.rough_min = min(self.rough_min, float(vv.min().item()))
                self.rough_max = max(self.rough_max, float(vv.max().item()))

    def as_dict(self):
        out = {
            "frames": self.frames,
            "spatial_mismatch_count": self.spatial_mismatch,
        }

        if self.slz_total:
            labeled = max(self.slz_safe + self.slz_unsafe, 1)
            out["slz"] = {
                "safe_pixels": self.slz_safe,
                "unsafe_pixels": self.slz_unsafe,
                "ignore_pixels": self.slz_ignore,
                # ratios over labeled (0/1) pixels only — 255 is excluded, matching
                # the loss/metric convention
                "safe_ratio": self.slz_safe / labeled,
                "unsafe_ratio": self.slz_unsafe / labeled,
                "ignore_ratio": self.slz_ignore / self.slz_total,
                "invalid_value_pixels": self.slz_invalid_value,
            }

        if self.depth_total:
            out["depth"] = {
                "valid_ratio": self.depth_valid / self.depth_total,
                "valid_pixels": self.depth_valid,
                "total_pixels": self.depth_total,
                "valid_mean_m": self.depth_sum / self.depth_valid if self.depth_valid else None,
                "valid_min_m": self.depth_min if self.depth_valid else None,
                "valid_max_m": self.depth_max if self.depth_valid else None,
            }

        if self.normal_total:
            out["normal"] = {
                "valid_ratio": self.normal_valid / self.normal_total,
                "valid_pixels": self.normal_valid,
                "total_pixels": self.normal_total,
                "mean_norm": self.normal_norm_sum / self.normal_valid if self.normal_valid else None,
                "mean_abs_norm_error": (
                    self.normal_norm_error_sum / self.normal_valid
                    if self.normal_valid else None
                ),
                "max_abs_norm_error": self.normal_norm_error_max if self.normal_valid else None,
                f"within_{self.normal_tol:g}_ratio": (
                    self.normal_within_tol / self.normal_valid
                    if self.normal_valid else None
                ),
            }

        if self.geometry_total:
            out["geometry_valid"] = {
                "valid_ratio": self.geometry_valid / self.geometry_total,
                "valid_pixels": self.geometry_valid,
                "total_pixels": self.geometry_total,
            }

        if self.slope_total:
            out["slope"] = {
                "finite_ratio": self.slope_finite / self.slope_total,
                "finite_mean_deg": (
                    self.slope_sum / self.slope_finite if self.slope_finite else None
                ),
                "finite_min_deg": self.slope_min if self.slope_finite else None,
                "finite_max_deg": self.slope_max if self.slope_finite else None,
            }

        if self.rough_total:
            out["roughness"] = {
                "finite_ratio": self.rough_finite / self.rough_total,
                "finite_mean_m": (
                    self.rough_sum / self.rough_finite if self.rough_finite else None
                ),
                "finite_min_m": self.rough_min if self.rough_finite else None,
                "finite_max_m": self.rough_max if self.rough_finite else None,
            }

        if self.semantic_hist:
            total = sum(self.semantic_hist.values())
            out["semantic"] = {
                "class_histogram": dict(sorted(self.semantic_hist.items())),
                "class_ratio": {
                    int(k): v / total for k, v in sorted(self.semantic_hist.items())
                },
            }

        return out


# ---------------------------------------------------------------------
# Split accumulator
# ---------------------------------------------------------------------

class SplitStats:
    def __init__(self, normal_tol: float):
        self.normal_tol = normal_tol
        self.overall = BasicStats(normal_tol)
        self.scene_counts = Counter()
        self.family_counts = Counter()
        self.scene_stats = defaultdict(lambda: BasicStats(normal_tol))
        self.family_stats = defaultdict(lambda: BasicStats(normal_tol))

    def update(self, sample: dict):
        scene = str(sample.get("scene", ""))
        family = str(sample.get("family", ""))

        self.scene_counts[scene] += 1
        self.family_counts[family] += 1

        self.overall.update(sample)
        self.scene_stats[scene].update(sample)
        self.family_stats[family].update(sample)

    def as_dict(self):
        out = self.overall.as_dict()
        out["scenes"] = dict(self.scene_counts)
        out["families"] = dict(self.family_counts)
        out["per_scene"] = {
            scene: stats.as_dict()
            for scene, stats in sorted(self.scene_stats.items())
        }
        out["per_family"] = {
            family: stats.as_dict()
            for family, stats in sorted(self.family_stats.items())
        }
        return out


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def print_sample_check(ds, num: int):
    print(f"Dataset length: {len(ds)}")
    print(f"Manifest: {ds.manifest}")

    for i in range(min(num, len(ds))):
        x = ds[i]
        print(
            f"\n[{i}] "
            f"scene={x.get('scene')} "
            f"family={x.get('family')} "
            f"frame={x.get('frame_id')}"
        )

        for k, v in x.items():
            if k in {"scene", "family", "frame_id", "index"}:
                continue
            print(f"  {k:16s}: {describe(v)}")

        hw = tuple(x["image"].shape[-2:])
        for key in ("semantic", "slz", "depth", "normal", "slope", "roughness"):
            if key in x and tuple(x[key].shape[-2:]) != hw:
                raise RuntimeError(
                    f"Spatial mismatch: image={hw}, "
                    f"{key}={tuple(x[key].shape[-2:])}"
                )

    print("\n[OK] checked samples are spatially aligned.")


def print_scene_table(stats: dict):
    per_scene = stats.get("per_scene", {})
    if not per_scene:
        return

    print("\nPer-scene summary:")
    header = (
        f"{'Scene':<31}"
        f"{'Frames':>8}"
        f"{'Safe %':>10}"
        f"{'DepthV %':>11}"
        f"{'NormalV %':>12}"
        f"{'GeoV %':>10}"
        f"{'Depth m':>11}"
        f"{'Slope deg':>12}"
        f"{'Rough m':>11}"
    )
    print(header)
    print("-" * len(header))

    for scene, s in per_scene.items():
        safe = 100 * s["slz"]["safe_ratio"] if "slz" in s else float("nan")
        depth_v = 100 * s["depth"]["valid_ratio"] if "depth" in s else float("nan")
        normal_v = 100 * s["normal"]["valid_ratio"] if "normal" in s else float("nan")
        geo_v = (
            100 * s["geometry_valid"]["valid_ratio"]
            if "geometry_valid" in s else float("nan")
        )
        depth_mean = s["depth"]["valid_mean_m"] if "depth" in s else None
        slope_mean = s["slope"]["finite_mean_deg"] if "slope" in s else None
        rough_mean = s["roughness"]["finite_mean_m"] if "roughness" in s else None

        print(
            f"{scene:<31}"
            f"{s['frames']:>8,}"
            f"{safe:>10.3f}"
            f"{depth_v:>11.3f}"
            f"{normal_v:>12.3f}"
            f"{geo_v:>10.3f}"
            f"{fmt(depth_mean, 2):>11}"
            f"{fmt(slope_mean, 2):>12}"
            f"{fmt(rough_mean, 4):>11}"
        )


def print_family_table(stats: dict):
    per_family = stats.get("per_family", {})
    if len(per_family) <= 1:
        return

    print("\nPer-family summary:")
    header = (
        f"{'Family':<18}"
        f"{'Frames':>8}"
        f"{'Safe %':>10}"
        f"{'DepthV %':>11}"
        f"{'NormalV %':>12}"
        f"{'GeoV %':>10}"
        f"{'Slope deg':>12}"
        f"{'Rough m':>11}"
    )
    print(header)
    print("-" * len(header))

    for family, s in per_family.items():
        safe = 100 * s["slz"]["safe_ratio"] if "slz" in s else float("nan")
        depth_v = 100 * s["depth"]["valid_ratio"] if "depth" in s else float("nan")
        normal_v = 100 * s["normal"]["valid_ratio"] if "normal" in s else float("nan")
        geo_v = (
            100 * s["geometry_valid"]["valid_ratio"]
            if "geometry_valid" in s else float("nan")
        )
        slope_mean = s["slope"]["finite_mean_deg"] if "slope" in s else None
        rough_mean = s["roughness"]["finite_mean_m"] if "roughness" in s else None

        print(
            f"{family:<18}"
            f"{s['frames']:>8,}"
            f"{safe:>10.3f}"
            f"{depth_v:>11.3f}"
            f"{normal_v:>12.3f}"
            f"{geo_v:>10.3f}"
            f"{fmt(slope_mean, 2):>12}"
            f"{fmt(rough_mean, 4):>11}"
        )


def print_full_stats(split: str, stats: dict, normal_tol: float):
    print("\n" + "=" * 90)
    print(f"FULL SPLIT STATISTICS — {split.upper()}")
    print("=" * 90)

    print(f"Frames             : {human_int(stats['frames'])}")
    print(f"Spatial mismatches : {human_int(stats['spatial_mismatch_count'])}")

    print("\nScenes:")
    for scene, n in sorted(stats["scenes"].items()):
        print(f"  {scene:32s} {human_int(n)}")

    print("\nFamilies:")
    for family, n in sorted(stats["families"].items()):
        print(f"  {family:32s} {human_int(n)}")

    if "slz" in stats:
        s = stats["slz"]
        print("\nSLZ class balance:")
        print(
            f"  Unsafe (0) : {human_int(s['unsafe_pixels'])} "
            f"({100*s['unsafe_ratio']:.3f}%)"
        )
        print(
            f"  Safe   (1) : {human_int(s['safe_pixels'])} "
            f"({100*s['safe_ratio']:.3f}% of labeled)"
        )
        print(
            f"  Ignore (255): {human_int(s.get('ignore_pixels', 0))} "
            f"({100*s.get('ignore_ratio', 0.0):.3f}% of all)"
        )
        print(f"  Invalid    : {human_int(s['invalid_value_pixels'])}")

    if "depth" in stats:
        d = stats["depth"]
        print("\nDepth:")
        print(f"  Valid ratio : {100*d['valid_ratio']:.3f}%")
        if d["valid_mean_m"] is not None:
            print(f"  Mean        : {d['valid_mean_m']:.4f} m")
            print(f"  Min         : {d['valid_min_m']:.4f} m")
            print(f"  Max         : {d['valid_max_m']:.4f} m")

    if "normal" in stats:
        n = stats["normal"]
        key = f"within_{normal_tol:g}_ratio"
        print("\nNormal:")
        print(f"  Valid ratio         : {100*n['valid_ratio']:.3f}%")
        if n["mean_norm"] is not None:
            print(f"  Mean ||n||          : {n['mean_norm']:.8f}")
            print(f"  Mean | ||n|| - 1 |  : {n['mean_abs_norm_error']:.8e}")
            print(f"  Max  | ||n|| - 1 |  : {n['max_abs_norm_error']:.8e}")
            print(f"  |norm-1| <= {normal_tol:g} : {100*n[key]:.5f}%")

    if "geometry_valid" in stats:
        g = stats["geometry_valid"]
        print("\nGeometry valid (depth_valid & normal_valid):")
        print(f"  Valid ratio : {100*g['valid_ratio']:.3f}%")

    if "slope" in stats:
        s = stats["slope"]
        print("\nSlope:")
        print(f"  Finite ratio : {100*s['finite_ratio']:.3f}%")
        if s["finite_mean_deg"] is not None:
            print(f"  Mean         : {s['finite_mean_deg']:.4f} deg")
            print(f"  Min          : {s['finite_min_deg']:.4f} deg")
            print(f"  Max          : {s['finite_max_deg']:.4f} deg")

    if "roughness" in stats:
        r = stats["roughness"]
        print("\nRoughness:")
        print(f"  Finite ratio : {100*r['finite_ratio']:.3f}%")
        if r["finite_mean_m"] is not None:
            print(f"  Mean         : {r['finite_mean_m']:.6f} m")
            print(f"  Min          : {r['finite_min_m']:.6f} m")
            print(f"  Max          : {r['finite_max_m']:.6f} m")

    if "semantic" in stats:
        print("\nSemantic class distribution:")
        total = sum(stats["semantic"]["class_histogram"].values())
        for cls, count in stats["semantic"]["class_histogram"].items():
            ratio = 100.0 * count / max(total, 1)
            print(
                f"  class {int(cls):2d}: "
                f"{human_int(count):>14s} ({ratio:7.3f}%)"
            )

    print_scene_table(stats)
    print_family_table(stats)

    print("\nInterpretation:")
    print("  - geometry_valid is ONLY depth_valid & normal_valid.")
    print("  - It is NOT a PCA-fit-valid mask.")
    print("  - slope=90 deg can be a real steep surface OR a generator sentinel.")
    print("  - Use the per-scene table to check whether GNSS_Evening changes")
    print("    SLZ balance / geometry statistics relative to the regular sequence.")
    print("  - If slope/roughness become MTL supervision, save an explicit")
    print("    PCA/slope/roughness valid mask in the generator.")


def scan_dataset(ds, normal_tol: float, progress_every: int) -> dict:
    acc = SplitStats(normal_tol=normal_tol)

    with torch.no_grad():
        for i in range(len(ds)):
            sample = ds[i]
            acc.update(sample)

            if progress_every > 0 and (
                (i + 1) % progress_every == 0 or i + 1 == len(ds)
            ):
                print(
                    f"\r  scanning: {i+1:,}/{len(ds):,}",
                    end="",
                    flush=True,
                )

    if len(ds):
        print()

    return acc.as_dict()


def make_dataset(args, split: str):
    return build_dataset(
        root=args.root,
        split=split,
        tasks=args.tasks,
        scheme=args.scheme,
        fold=args.fold,
        normalize=False,
        load_pose=args.load_pose,
        strict=True,
    )


def compact_summary(all_stats: Dict[str, dict]):
    if len(all_stats) <= 1:
        return

    print("\n" + "=" * 90)
    print("TRAIN / VAL / TEST SUMMARY")
    print("=" * 90)

    header = (
        f"{'Split':<8}"
        f"{'Frames':>10}"
        f"{'Safe %':>12}"
        f"{'Depth valid %':>16}"
        f"{'Normal valid %':>17}"
        f"{'Geo valid %':>14}"
    )
    print(header)
    print("-" * len(header))

    for split in ("train", "val", "test"):
        if split not in all_stats:
            continue

        s = all_stats[split]
        safe = 100 * s["slz"]["safe_ratio"] if "slz" in s else float("nan")
        depth = 100 * s["depth"]["valid_ratio"] if "depth" in s else float("nan")
        normal = 100 * s["normal"]["valid_ratio"] if "normal" in s else float("nan")
        geo = (
            100 * s["geometry_valid"]["valid_ratio"]
            if "geometry_valid" in s else float("nan")
        )

        print(
            f"{split:<8}"
            f"{s['frames']:>10,}"
            f"{safe:>12.3f}"
            f"{depth:>16.3f}"
            f"{normal:>17.3f}"
            f"{geo:>14.3f}"
        )


def global_scene_summary(all_stats: Dict[str, dict]):
    rows = []
    for split, stats in all_stats.items():
        for scene, s in stats.get("per_scene", {}).items():
            rows.append((split, scene, s))

    if not rows:
        return

    print("\n" + "=" * 100)
    print("GLOBAL PER-SCENE SUMMARY")
    print("=" * 100)

    header = (
        f"{'Split':<8}"
        f"{'Scene':<31}"
        f"{'Frames':>8}"
        f"{'Safe %':>10}"
        f"{'DepthV %':>11}"
        f"{'GeoV %':>10}"
        f"{'Depth m':>11}"
        f"{'Slope':>10}"
        f"{'Rough':>10}"
    )
    print(header)
    print("-" * len(header))

    for split, scene, s in rows:
        safe = 100 * s["slz"]["safe_ratio"] if "slz" in s else float("nan")
        depth_v = 100 * s["depth"]["valid_ratio"] if "depth" in s else float("nan")
        geo_v = (
            100 * s["geometry_valid"]["valid_ratio"]
            if "geometry_valid" in s else float("nan")
        )
        depth_mean = s["depth"]["valid_mean_m"] if "depth" in s else None
        slope_mean = s["slope"]["finite_mean_deg"] if "slope" in s else None
        rough_mean = s["roughness"]["finite_mean_m"] if "roughness" in s else None

        print(
            f"{split:<8}"
            f"{scene:<31}"
            f"{s['frames']:>8,}"
            f"{safe:>10.3f}"
            f"{depth_v:>11.3f}"
            f"{geo_v:>10.3f}"
            f"{fmt(depth_mean, 2):>11}"
            f"{fmt(slope_mean, 2):>10}"
            f"{fmt(rough_mean, 4):>10}"
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="train",
    )
    ap.add_argument(
        "--scheme",
        choices=["fixed", "four_fold", "loso", "loso_temporal"],
        default="fixed",
    )
    ap.add_argument("--fold", type=int, default=None)

    ap.add_argument(
        "--tasks",
        nargs="+",
        default=[
            "semantic",
            "depth",
            "normal",
            "slope",
            "roughness",
            "slz",
        ],
    )

    ap.add_argument(
        "--num",
        type=int,
        default=3,
        help="Number of detailed samples to print for each selected split.",
    )
    ap.add_argument("--load-pose", action="store_true")

    ap.add_argument(
        "--no-full-stats",
        action="store_true",
        help="Only inspect the first --num samples; skip complete split scan.",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Progress print interval during full scan. 0 disables progress.",
    )
    ap.add_argument(
        "--normal-tol",
        type=float,
        default=1e-3,
        help="Tolerance used for checking | ||normal|| - 1 |.",
    )
    ap.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional path for full statistics JSON.",
    )

    args = ap.parse_args()

    if args.scheme == "four_fold" and args.fold is None:
        ap.error("--fold is required when --scheme four_fold")

    if args.fold is not None and args.fold not in {1, 2, 3, 4}:
        ap.error("--fold must be one of 1, 2, 3, 4")

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    all_stats = {}

    for split in splits:
        print("\n" + "#" * 90)
        print(f"# SPLIT: {split.upper()}")
        print("#" * 90)

        ds = make_dataset(args, split)

        if args.num > 0:
            print_sample_check(ds, args.num)

        if not args.no_full_stats:
            print(f"\nScanning complete '{split}' split...")
            stats = scan_dataset(
                ds,
                normal_tol=args.normal_tol,
                progress_every=args.progress_every,
            )
            all_stats[split] = stats
            print_full_stats(split, stats, args.normal_tol)

    if all_stats:
        compact_summary(all_stats)
        global_scene_summary(all_stats)

        if args.save_json is not None:
            out = args.save_json.expanduser()
            if not out.is_absolute():
                out = Path.cwd() / out
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(all_stats, indent=2), encoding="utf-8")
            print(f"\nSaved statistics JSON: {out}")


if __name__ == "__main__":
    main()
