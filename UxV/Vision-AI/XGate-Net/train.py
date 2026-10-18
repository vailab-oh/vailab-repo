import argparse
import csv
import hashlib
import json
import logging
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_model.loss import silog_loss
from train_model.data import CLASS_COLORS, Tartan, validate_split
from train_model.model import models


ROOT = Path(__file__).resolve().parent
SOURCES = [
    "train.py", "train_model/model.py", "train_model/data.py",
    "train_model/encoder.py", "train_model/decoder.py",
    "train_model/loss.py", "train_model/utils.py",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Train the full XGate-Net model for segmentation and depth on a train/val split.')
    parser.add_argument("--config", type=Path, default=ROOT / "train_config.yaml")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data",
                        help="Existing split root: train/{rgb,label,depth}, val/{rgb,label,depth}")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Training_result" / "xgate-net")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=None, help="Optional PyTorch CPU thread limit")
    parser.add_argument("--resume", action="store_true",
                        help="Resume last_checkpoint.pth; skip already completed matching runs")
    parser.add_argument("--dry-run", action="store_true", help="Print settings without loading data or creating outputs")
    parser.add_argument("--check-data", action="store_true", help="Validate frame pairs/split without training or writing outputs")
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size) < 1 or args.workers < 0:
        parser.error("epochs and batch-size must be positive; workers must be nonnegative")
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("lr must be positive and weight-decay nonnegative, both finite")
    if args.threads is not None and args.threads < 1:
        parser.error("threads must be positive")
    return args


def seed_worker(_):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rgb_targets(rgb):
    rgb = (rgb * 255).round().to(torch.int16)
    distance = torch.full_like(rgb[:, 0], 766)
    indices = torch.full_like(rgb[:, 0], 255, dtype=torch.long)
    for index, color in CLASS_COLORS.items():
        palette = rgb.new_tensor(color).view(1, 3, 1, 1)
        current = (rgb - palette).abs().sum(dim=1, dtype=torch.int16)
        better = current < distance
        indices[better] = index
        distance = torch.minimum(distance, current)
    exact = indices.masked_fill(distance != 0, 255)
    tolerant = indices.masked_fill(distance > 15, 255)
    return exact, tolerant


class EpochMetrics:
    def __init__(self, classes, device):
        self.confusion = torch.zeros(classes, classes, dtype=torch.int64, device=device)
        self.losses = np.zeros(3, dtype=np.float64)
        self.batches = 0
        self.depth_sums = np.zeros(7, dtype=np.float64)
        self.depth_images = 0

    @torch.no_grad()
    def update(self, losses, seg, labels, depth, gt, mask):
        self.losses += [float(x.detach()) for x in losses]
        self.batches += 1
        classes = self.confusion.shape[0]
        valid = labels != 255
        predicted = seg.argmax(1)
        self.confusion += torch.bincount(
            classes * labels[valid] + predicted[valid], minlength=classes ** 2
        ).reshape(classes, classes)
        for pred_i, gt_i, valid_i in zip(depth, gt, mask):
            valid_i = valid_i.bool() & (gt_i > 0)
            if not valid_i.any():
                continue
            p, g = pred_i[valid_i].clamp_min(1e-6), gt_i[valid_i].clamp_min(1e-6)
            ratio = torch.maximum(p / g, g / p)
            values = torch.stack([
                ((p - g).abs() / g).mean(), (p - g).square().mean().sqrt(),
                (ratio < 1.25).float().mean(), (ratio < 1.25 ** 2).float().mean(),
                (ratio < 1.25 ** 3).float().mean(),
                (p.log() - g.log()).square().mean().sqrt(),
                (p.log10() - g.log10()).abs().mean(),
            ])
            self.depth_sums += values.cpu().numpy()
            self.depth_images += 1

    def result(self):
        hist = self.confusion.cpu().numpy().astype(np.float64)
        if not self.batches or not hist.sum() or not self.depth_images:
            raise ValueError("Epoch has no usable segmentation/depth samples")
        tp = np.diag(hist)
        union = hist.sum(0) + hist.sum(1) - tp
        iou = tp / np.maximum(union, 1)
        dice = 2 * tp / np.maximum(hist.sum(0) + hist.sum(1), 1)
        result = dict(zip(("loss", "seg_loss", "depth_loss"), self.losses / self.batches))
        result.update(miou=float(iou.mean()), dice=float(dice.mean()),
                      pixel_acc=float(tp.sum() / hist.sum()))
        result.update(zip(("abs_rel", "rmse", "delta1", "delta2", "delta3", "rmse_log", "log10"),
                          self.depth_sums / self.depth_images))
        return {k: float(v) for k, v in result.items()}


def run_epoch(model, loader, device, optimizer, depth_loss, epoch, epochs):
    training = optimizer is not None
    model.train(training)
    accumulator = EpochMetrics(15, device)
    progress = tqdm(loader, desc=f"{'Train' if training else 'Val'} {epoch}/{epochs}",
                    dynamic_ncols=True, disable=not sys.stderr.isatty())
    with torch.set_grad_enabled(training):
        for image, seg_gt, depth_gt, mask in progress:
            image, seg_gt, depth_gt, mask = [x.to(device, non_blocking=True) for x in (image, seg_gt, depth_gt, mask)]
            if training:
                optimizer.zero_grad(set_to_none=True)
            seg, depth, _, _ = model(image)
            exact, labels = rgb_targets(seg_gt)
            seg_loss = (F.cross_entropy(seg, exact, ignore_index=255)
                        if (exact != 255).any() else seg.sum() * 0)
            dep_loss = depth_loss(depth, depth_gt, mask)
            loss = seg_loss + (0.3 if training else 1.0) * dep_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss in epoch {epoch}; inspect data and run.log")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
            accumulator.update((loss, seg_loss, dep_loss), seg.detach(), labels,
                               depth.detach(), depth_gt, mask)
            progress.set_postfix(loss=f"{loss.item():.4f}")
    return accumulator.result()


def atomic_save(value, path):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def write_metrics(path, rows):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_summary(output, checkpoint):
    settings = checkpoint["settings"]
    epoch, best_epoch = checkpoint["epoch"], checkpoint["best_epoch"]
    rows = checkpoint["metrics"]
    summary = {
        "completed_epochs": epoch,
        "requested_epochs": settings["training"]["epochs"],
        "completed": epoch == settings["training"]["epochs"],
        "total_parameters": checkpoint["total_parameters"],
        "best_epoch": best_epoch, "best_val_loss": checkpoint["best_val_loss"],
        "last_epoch_metrics": rows[-1], "best_epoch_metrics": rows[best_epoch - 1],
        "elapsed_seconds": checkpoint["elapsed_seconds"],
    }
    temporary = output / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output / "summary.json")


def rng_state(generator):
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader": generator.get_state(),
    }


def restore_rng(state, generator):
    random.setstate(state["python"])
    name, values, *tail = state["numpy"]
    np.random.set_state((name, np.asarray(values, dtype=np.uint32), *tail))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader"])


def main(argv=None):
    args = parse_args(argv)
    with args.config.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError("config must be a YAML mapping")
    if any(int(cfg[k]) < 32 or int(cfg[k]) % 32 for k in ("height", "width")):
        raise ValueError("height and width must be positive multiples of 32 for the existing FPN")
    if cfg["model_conf"].get("num_seg_classes") != 15:
        raise ValueError("This VKITTI2 runner expects the existing 15-class palette")
    if not cfg["model_conf"].get("Seg") or not cfg["model_conf"].get("Depth") or cfg["model_conf"].get("od"):
        raise ValueError("config must enable Seg and Depth and disable od")
    if cfg["model_conf"].get("max_depth", 80.0) != 80.0:
        raise ValueError("The depth transform expects max_depth=80")
    output = args.output_dir.resolve()
    settings = {
        "model_module": "train_model.model",
        "config": cfg, "data_root": str(args.data_root.resolve()),
        "training": {key: getattr(args, key) for key in (
            "epochs", "batch_size", "workers", "seed", "lr", "weight_decay", "device", "threads")},
        "preprocessing": {"depth": "raw / 65535 * 80 (legacy Tartan scale)",
                          "depth_mask": "0 < scaled_depth <= 80", "seg_resize": "bilinear (legacy Tartan)",
                          "rgb": "RGB / 255, ImageNet mean/std", "crop": None, "tta": False},
        "loss": {"train": "CE + 0.3 * SiLog", "val": "CE + SiLog", "silog_variance_focus": 0.85},
        "metrics": {"seg": "dataset confusion, macro over all 15 classes, RGB tolerance 15",
                    "depth": "per-image average using legacy-scaled depth, not verified physical meters"},
        "checkpoint_policy": "last_model.pth: final epoch; best_loss_model.pth: minimum val loss",
    }
    if args.dry_run:
        print(f"XGate-Net: epochs={args.epochs}, output={output}")
        return

    train_set = Tartan(True, cfg, args.data_root)
    val_set = Tartan(False, cfg, args.data_root)
    validate_split(train_set, val_set)
    settings["split"] = {
        "train_samples": len(train_set), "val_samples": len(val_set),
        "train_manifest_sha256": train_set.manifest_sha256, "val_manifest_sha256": val_set.manifest_sha256,
    }
    if args.check_data:
        print(f"Frame pairs OK, train={len(train_set)}, val={len(val_set)}")
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use the training environment or --device cpu for diagnostics")
    if args.threads:
        torch.set_num_threads(args.threads)
    settings["source_sha256"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}
    checkpoint_path = output / "last_checkpoint.pth"
    resume = None
    if output.exists():
        if not args.resume:
            raise FileExistsError(f"Output already exists: {output}. Use --resume or a new --output-dir")
        if checkpoint_path.is_file():
            resume = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if resume["settings"] != settings:
                raise ValueError("Resume refused: settings, source code or data manifest changed. Use a new output root.")
            if resume["epoch"] >= args.epochs:
                write_metrics(output / "metrics.csv", resume["metrics"])
                write_summary(output, resume)
                print(f"SKIP: completed {resume['epoch']} epochs in {output}")
                return
        elif (output / "config.yaml").is_file():
            previous = yaml.safe_load((output / "config.yaml").read_text(encoding="utf-8"))
            if previous != settings:
                raise ValueError("Restart refused: saved settings differ. Use a new output root.")
            print("RESTART: no completed epoch saved yet")
        elif any(output.iterdir()):
            raise FileExistsError(f"Cannot resume unrelated/incomplete output without config.yaml: {output}")
    else:
        output.mkdir(parents=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True,
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(output / "run.log", encoding="utf-8")])
    logging.info("XGate-Net: train=%d val=%d", len(train_set), len(val_set))
    with (output / "config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(settings, stream, allow_unicode=True, sort_keys=False)
    train_set.save_manifest(output / "train_manifest.csv")
    val_set.save_manifest(output / "val_manifest.csv")
    if resume is None:
        for name in SOURCES:
            target = output / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)

    seed_everything(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=device.type == "cuda", worker_init_fn=seed_worker, generator=generator)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            pin_memory=device.type == "cuda", worker_init_fn=seed_worker)
    net = models(cfg).to(device)
    parameter_count = sum(p.numel() for p in net.parameters())
    model = net
    if device.type == "cuda" and device.index is None and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(net)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, int(0.25 * args.epochs)), gamma=0.4)
    depth_loss = silog_loss(0.85).to(device)
    rows, best_loss, best_epoch, start_epoch, elapsed = [], float("inf"), 0, 1, 0.0
    if resume is not None:
        net.load_state_dict(resume["model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        restore_rng(resume["rng"], generator)
        rows, best_loss, best_epoch = resume["metrics"], resume["best_val_loss"], resume["best_epoch"]
        start_epoch, elapsed = resume["epoch"] + 1, resume["elapsed_seconds"]
        logging.info("Resuming at epoch %d", start_epoch)
    logging.info("Parameters: %d; device: %s; train loss: CE + 0.3*SiLog; val loss: CE + SiLog", parameter_count, device)
    started = time.monotonic()
    for epoch in range(start_epoch, args.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        train_metrics = run_epoch(model, train_loader, device, optimizer, depth_loss, epoch, args.epochs)
        val_metrics = run_epoch(model, val_loader, device, None, depth_loss, epoch, args.epochs)
        rows.append({"epoch": epoch, "lr": lr, **{f"train_{k}": v for k, v in train_metrics.items()},
                     **{f"val_{k}": v for k, v in val_metrics.items()}})
        improved = val_metrics["loss"] < best_loss
        if improved:
            best_loss, best_epoch = val_metrics["loss"], epoch
        scheduler.step()
        state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        if improved:
            atomic_save(state, output / "best_loss_model.pth")
        atomic_save(state, output / "last_model.pth")
        checkpoint = {
            "epoch": epoch, "settings": settings, "model": state,
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "rng": rng_state(generator), "metrics": rows, "best_val_loss": best_loss, "best_epoch": best_epoch,
            "elapsed_seconds": elapsed + time.monotonic() - started,
            "total_parameters": parameter_count,
        }
        atomic_save(checkpoint, checkpoint_path)
        write_metrics(output / "metrics.csv", rows)
        write_summary(output, checkpoint)
        logging.info("Epoch %d/%d: train_loss=%.5f val_loss=%.5f mIoU=%.5f Dice=%.5f PA=%.5f AbsRel=%.5f best_epoch=%d",
                     epoch, args.epochs, train_metrics["loss"], val_metrics["loss"], val_metrics["miou"],
                     val_metrics["dice"], val_metrics["pixel_acc"], val_metrics["abs_rel"], best_epoch)
    logging.info("Completed: %s", output)


if __name__ == "__main__":
    main()
