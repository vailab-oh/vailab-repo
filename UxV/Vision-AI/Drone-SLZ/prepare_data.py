#!/usr/bin/env python3
"""Prepare the dataset: generate the SLZ labels and install the paper's splits.

    python prepare_data.py --root <UAVSCENES>                    # --stage all (default)
    python prepare_data.py --root <UAVSCENES> --stage labels     # labels only
    python prepare_data.py --root <UAVSCENES> --stage splits     # splits only
    python prepare_data.py --root <UAVSCENES> --stage splits --variant r050   # radius study

``<UAVSCENES>`` holds the downloaded archives under ``raw/`` (see README). The dataset is
written to ``<UAVSCENES>/prepared/uavscenes_slz/``.

labels  runs the label generator (``slz_benchmark.generation``) with the paper settings:
        interval5, PCA radius 0.50 m, slope 6 deg, roughness 0.10 m, semantic hazard veto,
        UAV radius 0.35 m + safety buffer 0.15 m. Generator options not listed in
        ``--help`` are passed through (``python -m slz_benchmark.generation.run --help``).
splits  copies ``splits/loso/`` (the exact frame lists and per-fold class weights of the
        paper) to ``<prepared>/manifests/loso/`` and checks that every frame it lists
        exists. A different existing file is only replaced with ``--force``.
        ``--variant r050|r020`` installs into ``<prepared>_<variant>`` (made by
        tools/make_radius_variant.py) with that label set's class weights.
"""

from __future__ import annotations

import argparse
import csv
import filecmp
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

SPLITS = REPO / "splits" / "loso"
SCHEME = "loso"
DEFAULT_DATASET_NAME = "uavscenes_slz"
#: columns whose files training and evaluation read
CHECK_COLUMNS = ("rgb_path", "slz_path")


def generate_labels(root: Path, dataset_name: str, extra: list[str]) -> Path:
    from slz_benchmark.generation.run import main as generate  # pulls in cv2 / numba

    return generate(["--root", str(root), "--dataset-name", dataset_name, *extra])


def _copy(src: Path, dst: Path, force: bool) -> str:
    if dst.exists():
        if filecmp.cmp(src, dst, shallow=False):
            return "same"
        if not force:
            raise SystemExit(f"{dst} exists and differs from {src}; pass --force to overwrite")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return "copied"


def _missing_files(prepared: Path, csv_path: Path) -> list[str]:
    missing = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for col in CHECK_COLUMNS:
                rel = row.get(col)
                if rel and not (prepared / rel).exists():
                    missing.append(rel)
    return missing


def install_splits(prepared: Path, variant: str | None, force: bool) -> None:
    if not (prepared / "scenes").is_dir():
        sys.exit(f"{prepared} does not look like a prepared dataset (no scenes/ dir)")
    dst = prepared / "manifests" / SCHEME

    counts = {"copied": 0, "same": 0}
    missing: dict[str, list[str]] = {}
    for csv_path in sorted(SPLITS.glob("fold*/*.csv")):
        rel = csv_path.relative_to(SPLITS)
        counts[_copy(csv_path, dst / rel, force)] += 1
        if csv_path.name in ("train.csv", "val.csv", "test.csv"):
            miss = _missing_files(prepared, csv_path)
            if miss:
                missing[str(rel)] = miss
    counts[_copy(SPLITS / "split_info.json", dst / "split_info.json", force)] += 1
    cw = SPLITS / ("class_weights.json" if variant is None else f"class_weights_{variant}.json")
    counts[_copy(cw, dst / "class_weights.json", force)] += 1

    print(f"{dst}: {counts['copied']} copied, {counts['same']} already identical "
          f"(class weights: {cw.name})")
    if missing:
        n = sum(len(v) for v in missing.values())
        for rel, miss in missing.items():
            print(f"  {rel}: {len(miss)} missing file(s), e.g. {miss[0]}")
        sys.exit(f"{n} referenced file(s) are missing under {prepared} — "
                 "were the labels generated with the paper settings?")
    print("all referenced rgb / slz files exist")


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__, allow_abbrev=False)
    ap.add_argument("--root", type=Path, required=True,
                    help="UAVScenes root (downloaded archives under <root>/raw/)")
    ap.add_argument("--stage", choices=["all", "labels", "splits"], default="all")
    ap.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME,
                    help="dataset folder under <root>/prepared/")
    ap.add_argument("--variant", choices=["r050", "r020"], default=None,
                    help="splits stage: install into <prepared>_<variant> with its class weights")
    ap.add_argument("--force", action="store_true",
                    help="splits stage: overwrite differing manifest files")
    args, extra = ap.parse_known_args()

    root = args.root.resolve()
    if extra and args.stage == "splits":
        ap.error(f"unrecognized arguments for --stage splits: {' '.join(extra)}")
    if args.variant and args.stage != "splits":
        ap.error("--variant is only used with --stage splits")

    prepared = root / "prepared" / args.dataset_name
    if args.stage in ("all", "labels"):
        prepared = generate_labels(root, args.dataset_name, extra)
    if args.stage in ("all", "splits"):
        if args.variant:
            prepared = prepared.parent / f"{prepared.name}_{args.variant}"
        install_splits(prepared, args.variant, args.force)
    print(f"\nprepared dataset: {prepared}")


if __name__ == "__main__":
    main()
