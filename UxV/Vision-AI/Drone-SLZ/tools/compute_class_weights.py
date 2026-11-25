#!/usr/bin/env python3
"""Compute the per-fold inverse-frequency Safe class weight.

For each fold, ``w_safe = N_unsafe / N_safe`` over the **labeled** pixels of that fold's
``train.csv`` (255 excluded; val / test never used). Deterministic: the same manifests and
labels give byte-identical output.

    python tools/compute_class_weights.py --root <PREPARED>          # --scheme loso

Writes ``<root>/manifests/<scheme>/class_weights.json``; train.py reads it when no
``--safe-weight`` is given. The shipped weights are in ``splits/loso/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slz_benchmark.datasets.uavscenes.class_weights import fold_class_weight
from slz_benchmark.datasets.uavscenes.splits import VALID_FOLDS, manifest_path


def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--scheme", default="loso")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <root>/manifests/<scheme>/class_weights.json")
    args = ap.parse_args()

    out = args.out or (args.root / "manifests" / args.scheme / "class_weights.json")

    payload = {
        "policy": "inverse_frequency_labeled_train_pixels",
        "formula": "w_unsafe = 1.0 ; w_safe = N_unsafe / N_safe",
        "ignore_index_excluded": 255,
        "note": "applied as the CE weight of the Safe class for every model family",
        "scheme": args.scheme,
        "folds": {},
    }

    for n in VALID_FOLDS:
        train_csv = manifest_path(args.root, "train", scheme=args.scheme, fold=n)
        if not train_csv.exists():
            raise FileNotFoundError(train_csv)
        info = fold_class_weight(args.root, train_csv)
        payload["folds"][f"fold{n}"] = {
            "train_manifest": str(train_csv.relative_to(args.root)),
            **info,
        }
        print(f"fold{n}: n_frames={info['n_frames']:5d}  "
              f"N_unsafe={info['n_unsafe']:>14d}  N_safe={info['n_safe']:>14d}  "
              f"N_ignore={info['n_ignore']:>12d}  "
              f"safe%={info['safe_fraction_labeled']*100:5.2f}  "
              f"w_safe={info['w_safe']:.4f}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nSaved:", out)


if __name__ == "__main__":
    main()
