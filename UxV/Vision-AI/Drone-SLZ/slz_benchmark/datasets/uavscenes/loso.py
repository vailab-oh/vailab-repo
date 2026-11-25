"""Leave-one-sequence-out (LOSO) manifests with the temporal guard
(``tools/make_loso_splits.py`` is the command line).

Reads the family-aware manifests the label generator writes::

    <root>/manifests/<source_scheme>/{train,val,test}.csv

merges them, repartitions by ``family``:

  - outer split: one sequence (scene family) = test, the rest = train pool
  - inner val: a deterministic contiguous block carved from each TRAIN scene
    (the held-out family is never used for early stopping)

and writes::

    <root>/manifests/<output_scheme>/fold{1..4}/{train,val,test}.csv
    <root>/manifests/<output_scheme>/split_info.json

``GNSS_Evening`` sequences are left out unless ``include_evening=True``. The 5 m spatial
guard is added afterwards by ``tools/apply_spatial_guard.py``.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

#: Temporal guard: this many frames are removed from *train* on each side of every
#: inner-val block.
#: Cadence is a uniform ~0.5 s (2 Hz), so 30 frames == 15 s of flight.
DEFAULT_GUARD = 30

DEFAULT_FAMILY_ORDER = [
    "AMtown",
    "AMvalley",
    "HKairport",
    "HKisland",
]


def read_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not fieldnames:
        raise RuntimeError(f"No CSV header: {path}")

    for required in ("family", "scene"):
        if required not in fieldnames:
            raise KeyError(
                f"Required column '{required}' missing from {path}. "
                f"columns={fieldnames}"
            )

    return fieldnames, rows


def write_csv(path: Path, fieldnames: Sequence[str], rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def deduplicate_rows(rows: Iterable[dict]):
    out = []
    seen = set()

    for row in rows:
        key = (
            row.get("scene", ""),
            row.get("frame", row.get("frame_id", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)

    return out


def is_evening_scene(scene: str) -> bool:
    return "evening" in scene.lower()


def family_sort_key(family: str):
    if family in DEFAULT_FAMILY_ORDER:
        return (DEFAULT_FAMILY_ORDER.index(family), family)
    return (len(DEFAULT_FAMILY_ORDER), family)


def sort_rows(rows):
    def key(row):
        return (
            family_sort_key(row.get("family", "")),
            row.get("scene", ""),
            row.get("frame", row.get("frame_id", "")),
        )
    return sorted(rows, key=key)


def contiguous_val_split(
    rows,
    val_ratio: float,
    val_position: str = "middle",
    guard: int = 0,
):
    """
    Split each TRAIN scene independently.

    A single contiguous block is used for inner validation so adjacent frames
    are not randomly scattered across train/val.

    ``guard`` (frames): a temporal guard gap removed from *train* on each side of the
    inner-val block, so the train frames adjacent to the block are not near-duplicate
    views of its endpoints. The val block keeps its size; only
    train shrinks. Scenes too small for the nominal guard
    (``n <= 2 * guard + n_val``) fall back to ``guard_eff = max(0, (n - n_val) // 4)``.

    Returns ``(train_rows, val_rows, guard_report)`` where ``guard_report`` maps
    ``scene -> {n, n_val, guard_nominal, guard_effective, dropped_head, dropped_tail}``.
    """
    if not 0.0 < val_ratio < 0.5:
        raise ValueError(
            f"val_ratio must be between 0 and 0.5, got {val_ratio}"
        )
    if guard < 0:
        raise ValueError(f"guard must be >= 0, got {guard}")

    by_scene = defaultdict(list)
    for row in sort_rows(rows):
        by_scene[row["scene"]].append(row)

    train_rows = []
    val_rows = []
    guard_report: dict = {}

    for scene, scene_rows in sorted(by_scene.items()):
        n = len(scene_rows)

        if n <= 1:
            train_rows.extend(scene_rows)
            guard_report[scene] = {
                "n": n, "n_val": 0, "guard_nominal": guard,
                "guard_effective": 0, "dropped_head": 0, "dropped_tail": 0,
            }
            continue

        n_val = max(1, round(n * val_ratio))
        n_val = min(n_val, n - 1)

        if val_position == "middle":
            start = max((n - n_val) // 2, 0)
        elif val_position == "tail":
            start = n - n_val
        else:
            raise ValueError(f"Unknown val_position={val_position}")

        end = start + n_val

        # Deterministic index-based guard (cadence is uniform ~0.5 s).
        guard_eff = guard
        if guard > 0 and n <= 2 * guard + n_val:
            guard_eff = max(0, (n - n_val) // 4)

        head_cut = max(start - guard_eff, 0)
        tail_cut = min(end + guard_eff, n)

        val_rows.extend(scene_rows[start:end])
        train_rows.extend(scene_rows[:head_cut])
        train_rows.extend(scene_rows[tail_cut:])

        guard_report[scene] = {
            "n": n,
            "n_val": n_val,
            "guard_nominal": guard,
            "guard_effective": guard_eff,
            "dropped_head": start - head_cut,
            "dropped_tail": tail_cut - end,
        }

    return sort_rows(train_rows), sort_rows(val_rows), guard_report


def summarize(rows):
    family_counts = Counter(r["family"] for r in rows)
    scene_counts = Counter(r["scene"] for r in rows)

    return {
        "frames": len(rows),
        "families": dict(sorted(family_counts.items())),
        "scenes": dict(sorted(scene_counts.items())),
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def build_loso_manifests(
    root: str | Path,
    source_scheme: str = "fixed",
    output_scheme: str = "loso_temporal",
    val_ratio: float = 0.15,
    val_position: str = "middle",
    include_evening: bool = False,
    guard: int = DEFAULT_GUARD,
    verbose: bool = True,
) -> dict:
    root = Path(root)
    source_dir = root / "manifests" / source_scheme
    output_dir = root / "manifests" / output_scheme

    all_rows = []
    fieldnames = None

    for split in ("train", "val", "test"):
        csv_path = source_dir / f"{split}.csv"
        fields, rows = read_csv(csv_path)

        if fieldnames is None:
            fieldnames = fields
        elif fields != fieldnames:
            raise RuntimeError(
                f"CSV columns differ across source manifests.\n"
                f"{csv_path}: {fields}\n"
                f"expected: {fieldnames}"
            )

        all_rows.extend(rows)

    all_rows = deduplicate_rows(all_rows)

    if not include_evening:
        all_rows = [
            r for r in all_rows
            if not is_evening_scene(r["scene"])
        ]

    all_rows = sort_rows(all_rows)

    families = sorted(
        {r["family"] for r in all_rows},
        key=family_sort_key,
    )

    if len(families) < 2:
        raise RuntimeError(
            f"Need at least 2 families for LOSO, found: {families}"
        )

    if verbose:
        print("=" * 80)
        print("LOSO manifest generation")
        print("=" * 80)
        print("Root            :", root)
        print("Source scheme   :", source_scheme)
        print("Output scheme   :", output_scheme)
        print("Include Evening :", include_evening)
        print("Val ratio       :", val_ratio)
        print("Val position    :", val_position)
        print("Guard gap       :", guard, "frames each side")
        print("Families        :", families)
        print("Total frames    :", len(all_rows))
        print()

    metadata = {
        "scheme": "leave-one-sequence-out",
        "source_scheme": source_scheme,
        "include_evening": include_evening,
        "inner_val_ratio": val_ratio,
        "inner_val_position": val_position,
        "guard": guard,
        "families": families,
        "folds": {},
    }

    for fold_idx, test_family in enumerate(families, start=1):
        test_rows = [r for r in all_rows if r["family"] == test_family]
        train_pool = [r for r in all_rows if r["family"] != test_family]

        train_rows, val_rows, guard_report = contiguous_val_split(
            train_pool,
            val_ratio=val_ratio,
            val_position=val_position,
            guard=guard,
        )

        fold_dir = output_dir / f"fold{fold_idx}"

        write_csv(fold_dir / "train.csv", fieldnames, train_rows)
        write_csv(fold_dir / "val.csv", fieldnames, val_rows)
        write_csv(fold_dir / "test.csv", fieldnames, test_rows)

        metadata["folds"][f"fold{fold_idx}"] = {
            "test_family": test_family,
            "train": summarize(train_rows),
            "val": summarize(val_rows),
            "test": summarize(test_rows),
            "guard": guard_report,
            "sha256": {
                "train": _sha256(fold_dir / "train.csv"),
                "val": _sha256(fold_dir / "val.csv"),
                "test": _sha256(fold_dir / "test.csv"),
            },
        }

        if verbose:
            print(f"[fold{fold_idx}] held-out test family = {test_family}")
            print(
                f"  train: {len(train_rows):5d} frames | "
                f"families={sorted({r['family'] for r in train_rows})}"
            )
            print(
                f"  val  : {len(val_rows):5d} frames | "
                f"families={sorted({r['family'] for r in val_rows})}"
            )
            print(
                f"  test : {len(test_rows):5d} frames | "
                f"families={sorted({r['family'] for r in test_rows})}"
            )
            print()

    # One combined hash over every fold csv (order-stable) — a single value to quote.
    combined = hashlib.sha256()
    for fold_key in sorted(metadata["folds"]):
        for split in ("train", "val", "test"):
            combined.update(metadata["folds"][fold_key]["sha256"][split].encode())
    metadata["manifest_sha256"] = combined.hexdigest()

    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "split_info.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if verbose:
        print("Saved:")
        print(" ", output_dir)
        print(" ", meta_path)
        print("manifest_sha256 :", metadata["manifest_sha256"])

    return metadata
