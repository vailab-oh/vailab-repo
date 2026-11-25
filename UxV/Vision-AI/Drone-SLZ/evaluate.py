#!/usr/bin/env python3
"""Score one checkpoint on one split.

    python evaluate.py --root <PREPARED> --model segformer_b0 \
        --checkpoint outputs/segformer_b0/fold1/seed42/best_model \
        --fold 1 --split test --save-json test_metrics.json

Same evaluator, transforms and metric as training. ``--model`` must be the key the
checkpoint was trained with. Image size is (H, W) = (640, 768).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from slz_benchmark.adapters import available, get_adapter
from slz_benchmark.config import CANONICAL_HEIGHT, CANONICAL_WIDTH, RunConfig
from slz_benchmark.data import make_loader, resolve_device
from slz_benchmark.datasets.uavscenes.splits import SCHEMES, validate_fold
from slz_benchmark.engine import evaluate, format_metrics
from slz_benchmark.losses import slz_class_weights


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--model", required=True, choices=available(),
                    help="model key the checkpoint was trained with (MODELS.md)")
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--scheme", choices=list(SCHEMES), default="loso")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--manifest", type=Path, default=None,
                    help="explicit manifest csv; bypasses --scheme/--fold. "
                         "Evaluation only — never train/tune on it.")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--slz-encoding",
                    choices=["class_0_1_255", "binary_0_255", "binary_0_1"],
                    default="class_0_1_255")
    ap.add_argument("--height", type=int, default=CANONICAL_HEIGHT)
    ap.add_argument("--width", type=int, default=CANONICAL_WIDTH)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--safe-weight", type=float, default=1.0)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--save-json", type=Path, default=None)
    args = ap.parse_args()

    if args.manifest is not None:
        if args.fold is not None:
            ap.error("--manifest and --fold are mutually exclusive")
        if not args.manifest.exists():
            ap.error(f"--manifest not found: {args.manifest}")
    else:
        try:
            validate_fold(args.scheme, args.fold)
        except ValueError as e:
            ap.error(str(e))

    cfg = RunConfig(
        root=args.root,
        scheme=args.scheme,
        fold=args.fold,
        manifest=str(args.manifest) if args.manifest is not None else None,
        slz_encoding=args.slz_encoding,
        height=args.height,
        width=args.width,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model=args.model,
        safe_weight=args.safe_weight,
        amp=not args.no_amp,
        deterministic=args.deterministic,
        max_steps=args.max_steps,
    )

    if args.deterministic:
        from slz_benchmark.engine import apply_determinism
        apply_determinism(True)

    if args.split == "test" or args.manifest is not None:
        print("=" * 72)
        print("  TEST SET / held-out manifest — final scoring only.")
        print("  No hyperparameter, seed, or checkpoint may be chosen from these numbers.")
        print("=" * 72)

    device = resolve_device()
    adapter = get_adapter(cfg.model)
    cfg.model_name = adapter.resolve_model_name(cfg.model_name)
    model = adapter.load(args.checkpoint, device)

    ds, loader = make_loader(cfg, args.split, adapter=adapter, shuffle=False)
    print(f"Split: {args.split} | frames: {len(ds)} | manifest: {ds.manifest}")

    metrics, cm = evaluate(
        adapter, model, loader, device,
        amp=(not args.no_amp),
        class_weights=slz_class_weights(cfg.safe_weight, device),
        max_steps=args.max_steps,
        desc=f"Eval {args.split}",
    )

    print("\nConfusion matrix [true, pred]:")
    print(cm.numpy())
    print(format_metrics(f"{args.split.capitalize():5s}:", metrics))
    for key, value in metrics.items():
        if isinstance(value, str):
            print(f"  {key:16s}: {value}")
        else:
            print(f"  {key:16s}: {value:.6f}")

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(
            json.dumps(
                {
                    "split": args.split,
                    "scheme": None if args.manifest is not None else args.scheme,
                    "fold": args.fold,
                    "manifest": str(args.manifest) if args.manifest is not None else None,
                    "model": cfg.model,
                    "model_name": cfg.model_name,
                    "frames": len(ds),
                    "checkpoint": str(args.checkpoint),
                    "image_size_hw": list(cfg.image_size),
                    "confusion_matrix": cm.tolist(),
                    "metrics": metrics,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print("Saved:", args.save_json)


if __name__ == "__main__":
    main()
