#!/usr/bin/env python3
"""Add the 5 m spatial guard: ``manifests/loso_temporal`` -> ``manifests/loso``.

The 30-frame temporal guard assumes that frames far apart in time are far apart in space.
That does not hold for HKairport03: the UAV revisits places near the validation block
(e.g. two frames 88 s apart were taken 0.81 m apart). For each fold, this drops every
train frame within ``--d-guard`` metres (default 5.0) of any val frame of the same family,
using the camera positions in the pose JSONs. Val and test are copied unchanged.

Writes ``manifests/loso/fold{1..4}/{train,val,test}.csv``, ``removed_train_frames.csv``
per fold (family, scene, frame, distance to the nearest val frame) and ``split_info.json``
(guard radius, counts, sha256 of every file). The input manifests are never modified.

    python tools/apply_spatial_guard.py --root <PREPARED> --d-guard 5.0
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

SOURCE_SCHEME = "loso_temporal"
OUTPUT_SCHEME = "loso"


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        return fieldnames, list(reader)


def write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_position(root: Path, row: dict) -> np.ndarray:
    info = json.loads((root / row["pose_path"]).read_text())
    T = np.array(info["T4x4"], dtype=np.float64)
    return T[:3, 3]


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--d-guard", type=float, default=5.0,
                     help="spatial guard radius in metres")
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    args = ap.parse_args()

    root = Path(args.root)
    src_dir = root / "manifests" / SOURCE_SCHEME
    out_dir = root / "manifests" / OUTPUT_SCHEME

    manifest = {
        "scheme": OUTPUT_SCHEME,
        "source_scheme": SOURCE_SCHEME,
        "derivation": "spatial guard on top of the unmodified source scheme's train.csv; "
                       "val.csv/test.csv copied through unchanged",
        "d_guard_m": args.d_guard,
        "folds": {},
    }

    pose_cache: dict = {}

    def pos(row):
        key = (row["scene"], row["frame"])
        if key not in pose_cache:
            pose_cache[key] = load_position(root, row)
        return pose_cache[key]

    for fold in args.folds:
        fold_name = f"fold{fold}"
        src_fold_dir = src_dir / fold_name
        fields_train, train_rows = read_csv(src_fold_dir / "train.csv")
        fields_val, val_rows = read_csv(src_fold_dir / "val.csv")
        fields_test, test_rows = read_csv(src_fold_dir / "test.csv")

        by_family_train = defaultdict(list)
        by_family_val = defaultdict(list)
        for r in train_rows:
            by_family_train[r["family"]].append(r)
        for r in val_rows:
            by_family_val[r["family"]].append(r)

        removed_rows = []
        removed_keys = set()
        for family, fam_train_rows in by_family_train.items():
            fam_val_rows = by_family_val.get(family, [])
            if not fam_val_rows:
                continue
            train_pos = np.array([pos(r) for r in fam_train_rows])
            val_pos = np.array([pos(r) for r in fam_val_rows])
            tree = cKDTree(train_pos)
            neighbor_lists = tree.query_ball_point(val_pos, r=args.d_guard)
            for vi, idxs in enumerate(neighbor_lists):
                if not idxs:
                    continue
                val_row = fam_val_rows[vi]
                for ti in idxs:
                    tr = fam_train_rows[ti]
                    key = (tr["scene"], tr["frame"])
                    if key in removed_keys:
                        continue
                    removed_keys.add(key)
                    d = float(np.linalg.norm(pos(tr) - pos(val_row)))
                    removed_rows.append({
                        "family": family, "train_scene": tr["scene"], "train_frame": tr["frame"],
                        "triggering_val_scene": val_row["scene"], "triggering_val_frame": val_row["frame"],
                        "distance_m": d,
                    })

        kept_train_rows = [r for r in train_rows if (r["scene"], r["frame"]) not in removed_keys]

        out_fold_dir = out_dir / fold_name
        write_csv(out_fold_dir / "train.csv", fields_train, kept_train_rows)
        write_csv(out_fold_dir / "val.csv", fields_val, val_rows)
        write_csv(out_fold_dir / "test.csv", fields_test, test_rows)

        manifest["folds"][fold_name] = {
            "train_original": len(train_rows),
            "train_removed": len(removed_rows),
            "train_retained": len(kept_train_rows),
            "val_frames": len(val_rows),
            "test_frames": len(test_rows),
            "removed_by_family": {
                fam: sum(1 for r in removed_rows if r["family"] == fam)
                for fam in sorted(by_family_train)
            },
            "sha256": {
                "train": sha256_file(out_fold_dir / "train.csv"),
                "val": sha256_file(out_fold_dir / "val.csv"),
                "test": sha256_file(out_fold_dir / "test.csv"),
            },
        }
        # Full audit trail: exactly which frames were dropped and why.
        removed_path = out_fold_dir / "removed_train_frames.csv"
        write_csv(
            removed_path,
            ["family", "train_scene", "train_frame", "triggering_val_scene",
             "triggering_val_frame", "distance_m"],
            removed_rows,
        )
        print(f"[{fold_name}] train {len(train_rows)} -> {len(kept_train_rows)} "
              f"({len(removed_rows)} removed); val {len(val_rows)}; test {len(test_rows)}")

    combined = hashlib.sha256()
    for fold_name in sorted(manifest["folds"]):
        for split in ("train", "val", "test"):
            combined.update(manifest["folds"][fold_name]["sha256"][split].encode())
    manifest["manifest_sha256"] = combined.hexdigest()

    (out_dir / "split_info.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote {out_dir}/split_info.json")
    print(f"manifest_sha256: {manifest['manifest_sha256']}")


if __name__ == "__main__":
    main()
