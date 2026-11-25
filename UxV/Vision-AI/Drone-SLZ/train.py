#!/usr/bin/env python3
"""Train one model on one fold.

    torchrun --standalone --nproc_per_node=1 train.py \
        --config configs/segformer_b0.yaml --root <PREPARED> \
        --output outputs/segformer_b0/fold1/seed42 --fold 1 --seed 42

``--model`` is any key of ``slz_benchmark.adapters.MODELS`` (see MODELS.md). The
checkpoint with the best val mIoU is saved to ``<output>/best_model/``. Image size is
(H, W) = (640, 768).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent))

from slz_benchmark.adapters import available, get_adapter
from slz_benchmark.config import CANONICAL_HEIGHT, CANONICAL_WIDTH, RunConfig
from slz_benchmark.data import make_loader, resolve_device
from slz_benchmark.datasets.uavscenes.splits import SCHEMES, canonical_scheme, validate_fold
from slz_benchmark.engine import Trainer


def ddp_env() -> tuple[int, int, int]:
    """``(rank, local_rank, world_size)`` from the env vars ``torchrun`` sets.

    Plain ``python train.py`` (no torchrun) leaves them unset and runs as a single
    process, the same as ``torchrun --nproc_per_node=1``. There is no ``--ddp`` flag:
    ``--nproc_per_node=N`` with N > 1 is what turns DDP on.
    """
    return (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("LOCAL_RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
    )


def apply_auto_class_weight(cfg: RunConfig, explicit_safe_weight) -> None:
    """With a LOSO scheme and no explicit ``--safe-weight``: set ``cfg.safe_weight`` to
    this fold's value in ``manifests/<scheme>/class_weights.json`` (the per-fold
    inverse-frequency Safe weight). Records ``safe_weight_source`` either way."""
    if explicit_safe_weight is not None:
        cfg.safe_weight_source = "manual"
        return
    try:
        canon = canonical_scheme(cfg.scheme)
        if not canon.startswith("loso") or cfg.fold is None:
            return
    except ValueError:
        return
    cw_path = Path(cfg.root) / "manifests" / canon / "class_weights.json"
    if not cw_path.exists():
        print(f"[class-weight] {cw_path} not found — keeping safe_weight={cfg.safe_weight}")
        return
    folds = json.loads(cw_path.read_text())["folds"]
    key = f"fold{int(cfg.fold)}"
    cfg.safe_weight = float(folds[key]["w_safe"])
    cfg.safe_weight_source = f"auto_inverse_freq_{key}"
    print(f"[class-weight] {key}: safe_weight <- {cfg.safe_weight:.4f} "
          f"({cfg.safe_weight_source})")


def build_config(args) -> RunConfig:
    cfg = RunConfig.from_yaml(args.config) if args.config else RunConfig()
    overrides = {
        "root": args.root,
        "output": args.output,
        "model": args.model,
        "model_name": args.model_name,
        "scheme": args.scheme,
        "fold": args.fold,
        "slz_encoding": args.slz_encoding,
        "height": args.height,
        "width": args.width,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "poly_power": args.poly_power,
        "safe_weight": args.safe_weight,
        "hflip_prob": args.hflip_prob,
        "jitter_prob": args.jitter_prob,
        "blur_prob": args.blur_prob,
        "grad_clip": args.grad_clip,
        "seed": args.seed,
        "amp": not args.no_amp,
        "deterministic": args.deterministic or None,
        "max_steps": args.max_steps,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)

    # The test loader is only built with --eval-split test. The paper runs pass
    # --eval-split none and score the test set once, afterwards, with evaluate.py.
    if args.eval_split == "test":
        cfg.skip_test = False
    elif args.eval_split in ("none", "val") or args.skip_test:
        cfg.skip_test = True
    return cfg


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--config", type=Path, default=None, help="optional YAML RunConfig")
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    # default None: a CLI default would override the config's `model:` key
    ap.add_argument("--model", default=None, choices=available(),
                    help="model key (MODELS.md); default: the config's, else segformer_b0")
    ap.add_argument("--model-name", default=None,
                    help="override the backbone the model key builds (HF id, torchvision "
                         "name or SafeUAV checkpoint path)")
    ap.add_argument("--scheme", choices=list(SCHEMES), default=None)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--slz-encoding",
                    choices=["class_0_1_255", "binary_0_255", "binary_0_1"], default=None)
    ap.add_argument("--height", type=int, default=None,
                    help=f"default: config value, else {CANONICAL_HEIGHT}")
    ap.add_argument("--width", type=int, default=None,
                    help=f"default: config value, else {CANONICAL_WIDTH}")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--warmup-ratio", type=float, default=None)
    ap.add_argument("--poly-power", type=float, default=None)
    ap.add_argument("--safe-weight", type=float, default=None)
    ap.add_argument("--hflip-prob", type=float, default=None)
    ap.add_argument("--jitter-prob", type=float, default=None,
                    help="probability of RGB-only ColorJitter (image-only, default 0.0)")
    ap.add_argument("--blur-prob", type=float, default=None,
                    help="probability of RGB-only Gaussian blur (image-only, default 0.0)")
    ap.add_argument("--grad-clip", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--deterministic", action="store_true",
                    help="torch.use_deterministic_algorithms(warn_only) + cudnn.deterministic")
    ap.add_argument("--skip-test", action="store_true",
                    help="same as --eval-split none")
    ap.add_argument("--eval-split", choices=["val", "test", "none"], default=None,
                    help="none|val: never build the test loader. "
                         "test: also score fold<N>/test.csv after training.")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="cap batches per epoch (smoke runs only)")
    args = ap.parse_args()

    if args.root is None or args.output is None:
        ap.error("--root and --output are required (unless fully set in --config)")

    cfg = build_config(args)
    try:
        validate_fold(cfg.scheme, cfg.fold)
    except ValueError as e:
        ap.error(str(e))

    adapter = get_adapter(cfg.model)
    # Store the concrete backbone back, so config.json records what was built.
    cfg.model_name = adapter.resolve_model_name(cfg.model_name)

    apply_auto_class_weight(cfg, args.safe_weight)

    # -- DDP ---------------------------------------------------------------------------
    rank, local_rank, world_size = ddp_env()
    ddp_enabled = world_size > 1
    is_main = rank == 0
    if ddp_enabled:
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = resolve_device()

    # --batch-size is the GLOBAL batch, summed over all DDP ranks. It is divided into
    # the per-process batch here, once, before any DataLoader is built.
    global_batch_size = cfg.batch_size
    if ddp_enabled:
        if global_batch_size % world_size != 0:
            ap.error(
                f"--batch-size {global_batch_size} (global) is not divisible by "
                f"world_size {world_size} (--nproc_per_node)"
            )
        cfg.batch_size = global_batch_size // world_size

    if is_main:
        print("Device:", device, "| model:", cfg.model, f"({cfg.model_name})",
              "| scheme:", cfg.scheme, "| image (H,W):", cfg.image_size,
              "| safe_weight:", f"{cfg.safe_weight:.4f} ({cfg.safe_weight_source})")
        if ddp_enabled:
            print(f"DDP: world_size={world_size} | global_batch={global_batch_size} "
                  f"| per_gpu_batch={cfg.batch_size}")

    model = adapter.build_model(device, model_name=cfg.model_name)

    # Explicit RNG for the training shuffle order, independent of the global RNG.
    # Same seed on every rank: DistributedSampler derives each rank's shard from
    # (seed, epoch).
    train_gen = torch.Generator().manual_seed(cfg.seed)
    train_ds, train_loader = make_loader(cfg, "train", adapter=adapter, generator=train_gen)
    val_ds, val_loader = make_loader(cfg, "val", adapter=adapter)
    test_loader = None
    if not cfg.skip_test:
        _, test_loader = make_loader(cfg, "test", adapter=adapter)

    if is_main:
        print(f"Train: {len(train_ds)}  {train_ds.manifest}")
        print(f"Val  : {len(val_ds)}  {val_ds.manifest}")

    Trainer(
        adapter, model, train_loader, val_loader, device, cfg, test_loader=test_loader,
        rank=rank, world_size=world_size, global_batch_size=global_batch_size,
    ).fit()

    if ddp_enabled:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
