#!/usr/bin/env python3
"""Rebuild the leave-one-sequence-out manifests with the temporal guard only.

First step of rebuilding the paper's split from scratch (normally you install the
shipped split with ``prepare_data.py --stage splits`` instead):

    python tools/make_loso_splits.py --root <PREPARED>                      # -> manifests/loso_temporal
    python tools/apply_spatial_guard.py --root <PREPARED> --d-guard 5.0     # -> manifests/loso
    python tools/compute_class_weights.py --root <PREPARED>

Writes ``<root>/manifests/loso_temporal/fold{1..4}/{train,val,test}.csv`` and
``split_info.json``. The GNSS_Evening sequences are left out unless ``--include-evening``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slz_benchmark.datasets.uavscenes.loso import DEFAULT_GUARD, build_loso_manifests


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--source-scheme", default="fixed")
    ap.add_argument("--output-scheme", default="loso_temporal")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument(
        "--val-position",
        choices=["middle", "tail"],
        default="middle",
    )
    ap.add_argument(
        "--include-evening",
        action="store_true",
        help="Include GNSS_Evening sequences. Default = core non-Evening only.",
    )
    ap.add_argument(
        "--guard",
        type=int,
        default=DEFAULT_GUARD,
        help="frames removed from train on each side of every val block",
    )
    args = ap.parse_args()

    build_loso_manifests(
        root=args.root,
        source_scheme=args.source_scheme,
        output_scheme=args.output_scheme,
        val_ratio=args.val_ratio,
        val_position=args.val_position,
        include_evening=args.include_evening,
        guard=args.guard,
    )


if __name__ == "__main__":
    main()
