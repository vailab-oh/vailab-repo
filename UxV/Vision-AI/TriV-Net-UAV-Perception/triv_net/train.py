#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation.common_metrics import (
    DetectionMetricAccumulator,
    depth_metric_values,
    seg_metrics_from_confmat,
    update_seg_confmat,
)
from train_model.data import Tartan
from train_model.loss import silog_loss
from train_model.loss_balancing import LossBalancer, validate_fixed_weights
from train_model.original_trivnet_training import (
    OriginalTriVNetODLoss,
    build_original_trivnet,
    decode_original_detections,
)


def to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().mean().item())
    return float(value)


def save_yaml(path, data):
    with Path(path).open("w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, sort_keys=False, allow_unicode=True)


def set_global_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def collater(data):
    imgs, imgs_ori, seg_gt, depth_gt, depth_mask, annots_batch = zip(*data)
    imgs = torch.stack(imgs)
    imgs_ori = torch.stack(imgs_ori)
    seg_gt = torch.stack(seg_gt) if seg_gt[0] is not None else None
    if depth_gt[0] is not None:
        depth_gt = torch.stack(depth_gt)
        depth_mask = torch.stack(depth_mask)
    else:
        depth_gt = None
        depth_mask = None

    max_num_annots = max((annots.shape[0] for annots in annots_batch), default=0)
    padded_annots = torch.zeros(
        (len(annots_batch), max_num_annots, 5), dtype=torch.float32
    )
    padded_annots[..., 4] = -1
    for batch_index, annots in enumerate(annots_batch):
        if annots.shape[0] > 0:
            padded_annots[batch_index, : annots.shape[0]] = annots
    return imgs, imgs_ori, seg_gt, depth_gt, depth_mask, padded_annots


def make_save_dir(cfg, run_name):
    root = Path(cfg.get("output_root", "Training_result"))
    formatted = datetime.now().strftime("%y-%m-%d-%H")
    base_name = f"{formatted}_{run_name}" if run_name else formatted
    save_dir = root / base_name
    suffix = 1
    while save_dir.exists():
        save_dir = root / f"{base_name}-{suffix}"
        suffix += 1
    save_dir.mkdir(parents=True)
    return save_dir


def validate_config(cfg):
    model_cfg = cfg.get("model", {})
    family = str(model_cfg.get("architecture_family", "")).lower().replace("-", "_")
    if family != "original_trivnet":
        raise ValueError("model.architecture_family must be original_trivnet.")
    if Path(cfg.get("dataset_root", "")).name.lower() != "oldtown":
        raise ValueError(
            "This entrypoint is restricted to the current OldTown dataset."
        )
    if int(cfg.get("seg_num_classes", 0)) != 12:
        raise ValueError("OldTown original TriV-Net requires seg_num_classes=12.")
    if int(cfg.get("od_num_classes", 0)) != 5:
        raise ValueError("OldTown original TriV-Net requires od_num_classes=5.")
    encoder = str(model_cfg.get("encoder", "")).lower().replace("_", "-")
    if encoder not in ("resnet", "resnet50", "resnet-50"):
        raise ValueError("Original OldTown runs require the ResNet-50 encoder.")
    if str(model_cfg.get("decoder", "")).lower().replace("_", "-") != "fpn":
        raise ValueError("Original OldTown runs require the legacy FPN decoder.")
    if model_cfg.get("pretrained") is not True:
        raise ValueError("Original OldTown runs require ImageNet pretrained=true.")
    if int(cfg.get("seg_ignore_index", -1)) != 255:
        raise ValueError("OldTown segmentation requires seg_ignore_index=255.")
    height = int(cfg.get("height", 0))
    width = int(cfg.get("width", 0))
    if height <= 0 or width <= 0 or height % 128 or width % 128:
        raise ValueError("Legacy P3-P7 anchors require positive H/W divisible by 128.")
    if int(cfg.get("epochs", 0)) <= 0 or int(cfg.get("batch_size", 0)) <= 0:
        raise ValueError("epochs and batch_size must be positive.")
    conf = cfg.get("model_conf", {})
    if not any(bool(conf.get(name, False)) for name in ("Seg", "Depth", "od")):
        raise ValueError("At least one task must be enabled in model_conf.")
    validate_fixed_weights(cfg)
    forbidden = {"resume", "init_checkpoint", "checkpoint_path", "weights"}

    def nested_keys(value):
        if isinstance(value, dict):
            for key, child in value.items():
                yield str(key)
                yield from nested_keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from nested_keys(child)

    if forbidden.intersection(nested_keys(cfg)):
        raise ValueError(
            "Original TriV-Net runs must start from ImageNet initialization."
        )


def build_od_loss(cfg):
    return OriginalTriVNetODLoss(
        classification_weight=float(
            cfg.get("original_od_classification_loss_weight", 1.0)
        ),
        regression_weight=float(cfg.get("original_od_regression_loss_weight", 1.0)),
    )


def validation_checkpoint_score(val_stats, cfg):
    checkpoint_cfg = cfg.get("checkpoint", {})
    metric = str(checkpoint_cfg.get("metric", "val_loss")).lower()
    if metric == "val_loss":
        score = -float(val_stats["loss"])
    elif metric in ("seg_miou", "val_seg_miou"):
        score = float(val_stats["seg_miou"])
    elif metric in ("depth_abs_rel", "val_depth_abs_rel"):
        score = -float(val_stats["depth_abs_rel"])
    elif metric in ("od_map50_95", "val_od_map50_95"):
        score = float(val_stats["od_map50_95"])
    else:
        raise ValueError(f"Unsupported checkpoint.metric: {metric}")
    if not np.isfinite(score):
        raise FloatingPointError(f"Non-finite checkpoint score for {metric}: {score}")
    return score, metric


def empty_task_metrics(stats):
    stats.update(
        {
            "seg_pixel_acc": 0.0,
            "seg_mean_acc": 0.0,
            "seg_miou": 0.0,
            "seg_mean_dice": 0.0,
            "seg_class_iou": [],
            "seg_class_acc": [],
            "seg_class_dice": [],
        }
    )


def run_epoch(
    model,
    loader,
    cfg,
    device,
    depth_loss_fn,
    od_loss_fn,
    loss_balancer,
    optimizer=None,
    epoch=0,
):
    is_train = optimizer is not None
    model.train(is_train)
    conf = cfg["model_conf"]
    seg_ignore_index = int(cfg.get("seg_ignore_index", 255))
    seg_num_classes = int(cfg["seg_num_classes"])
    od_num_classes = int(cfg["od_num_classes"])
    seg_confmat = torch.zeros(
        (seg_num_classes, seg_num_classes), dtype=torch.long, device=device
    )
    depth_metric_sums = {
        name: 0.0
        for name in (
            "abs_rel",
            "sq_rel",
            "rmse",
            "rmse_log",
            "mae",
            "delta1",
            "delta2",
            "delta3",
        )
    }
    depth_metric_batches = 0
    od_metric_acc = (
        DetectionMetricAccumulator(od_num_classes)
        if conf.get("od", False)
        and not is_train
        and bool(cfg.get("eval_od_metrics", True))
        else None
    )
    totals = {
        name: 0.0
        for name in (
            "loss",
            "seg_loss",
            "depth_loss",
            "od_loss",
            "od_iou",
            "weighted_seg_loss",
            "weighted_depth_loss",
            "weighted_od_loss",
            "loss_weight_seg",
            "loss_weight_depth",
            "loss_weight_od",
        )
    }
    processed_batches = 0
    clipped_batches = 0
    clip_threshold = float(cfg.get("gradient_clip_norm", 5.0))
    phase = "train" if is_train else "val"
    progress = tqdm(
        enumerate(loader),
        total=len(loader),
        desc=f"{phase} epoch {epoch + 1}",
        dynamic_ncols=True,
        leave=False,
        disable=bool(cfg.get("disable_tqdm", False)) or not sys.stderr.isatty(),
    )

    for batch_index, batch in progress:
        imgs, _, seg_gt, depth_gt, depth_mask, annotations = batch
        imgs = imgs.to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        seg, depth, classification, regression = model(imgs)
        zero = imgs.sum() * 0.0
        loss_seg = zero
        loss_depth = zero
        loss_od = zero
        od_iou = torch.zeros((), device=device)

        if conf.get("Seg", False):
            seg_gt = seg_gt.to(device, non_blocking=True).long()
            loss_seg = F.cross_entropy(seg, seg_gt, ignore_index=seg_ignore_index)
            update_seg_confmat(
                seg_confmat,
                seg.argmax(dim=1),
                seg_gt,
                seg_num_classes,
                seg_ignore_index,
            )

        if conf.get("Depth", False):
            depth_gt = depth_gt.to(device, non_blocking=True).float()
            depth_mask = depth_mask.to(device, non_blocking=True)
            loss_depth = depth_loss_fn(depth, depth_gt, depth_mask)
            depth_metrics = depth_metric_values(depth, depth_gt, depth_mask)
            if depth_metrics is not None:
                for key, value in depth_metrics.items():
                    depth_metric_sums[key] += value
                depth_metric_batches += 1

        if conf.get("od", False):
            annotations = annotations.to(device, non_blocking=True)
            loss_od, od_iou = od_loss_fn(
                model,
                imgs,
                classification,
                regression,
                annotations,
            )
            if od_metric_acc is not None:
                detections = decode_original_detections(
                    model,
                    imgs,
                    classification,
                    regression,
                    score_thresh=float(cfg.get("eval_od_score_thresh", 0.001)),
                    nms_thresh=float(cfg.get("eval_od_nms_thresh", 0.5)),
                    topk=int(cfg.get("eval_od_topk", 100)),
                )
                od_metric_acc.update(detections, annotations)

        task_losses = {"seg": loss_seg, "depth": loss_depth, "od": loss_od}
        for task in loss_balancer.tasks:
            if not torch.isfinite(task_losses[task]).all():
                raise FloatingPointError(
                    f"Non-finite {task} loss at {phase} epoch {epoch + 1}, "
                    f"batch {batch_index + 1}."
                )
        loss, loss_weights = loss_balancer.combine(task_losses)
        if not torch.isfinite(loss).all():
            raise FloatingPointError(
                f"Non-finite total loss at {phase} epoch {epoch + 1}, "
                f"batch {batch_index + 1}."
            )
        if is_train:
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), clip_threshold
            )
            if to_float(gradient_norm) > clip_threshold:
                clipped_batches += 1
            optimizer.step()

        processed_batches += 1
        totals["loss"] += to_float(loss)
        totals["seg_loss"] += to_float(loss_seg)
        totals["depth_loss"] += to_float(loss_depth)
        totals["od_loss"] += to_float(loss_od)
        totals["od_iou"] += to_float(od_iou)
        for task in ("seg", "depth", "od"):
            weight = float(loss_weights.get(task, 0.0))
            totals[f"loss_weight_{task}"] += weight
            totals[f"weighted_{task}_loss"] += weight * to_float(task_losses[task])
        progress.set_postfix(
            refresh=False,
            loss=f"{to_float(loss):.3f}",
            seg=f"{to_float(loss_seg):.3f}",
            depth=f"{to_float(loss_depth):.3f}",
            od=f"{to_float(loss_od):.3f}",
        )

    divisor = max(1, processed_batches)
    stats = {key: value / divisor for key, value in totals.items()}
    stats["gradient_clip_rate"] = float(clipped_batches) / divisor if is_train else 0.0
    if conf.get("Seg", False):
        stats.update(seg_metrics_from_confmat(seg_confmat))
    else:
        empty_task_metrics(stats)
    if conf.get("Depth", False) and depth_metric_batches > 0:
        stats.update(
            {
                f"depth_{key}": value / depth_metric_batches
                for key, value in depth_metric_sums.items()
            }
        )
    else:
        stats.update({f"depth_{key}": 0.0 for key in depth_metric_sums})
    if od_metric_acc is not None:
        stats.update(od_metric_acc.compute())
    else:
        stats.update(
            {
                "od_map50": 0.0,
                "od_map50_95": 0.0,
                "od_precision50": 0.0,
                "od_recall50": 0.0,
                "od_f1_50": 0.0,
                "od_ap50_per_class": [],
            }
        )
    return stats


def metric_row(epoch, lr, checkpoint_metric, checkpoint_score, train_stats, val_stats):
    row = {
        "epoch": epoch,
        "lr": lr,
        "checkpoint_metric": checkpoint_metric,
        "checkpoint_score": checkpoint_score,
        "train_loss": train_stats["loss"],
        "val_loss": val_stats["loss"],
    }
    for split, stats in (("train", train_stats), ("val", val_stats)):
        for name in (
            "seg_pixel_acc",
            "seg_mean_acc",
            "seg_miou",
            "seg_mean_dice",
            "depth_abs_rel",
            "depth_sq_rel",
            "depth_rmse",
            "depth_rmse_log",
            "depth_mae",
            "depth_delta1",
            "depth_delta2",
            "depth_delta3",
            "od_iou",
            "od_map50",
            "od_map50_95",
            "od_precision50",
            "od_recall50",
            "od_f1_50",
        ):
            row[f"{split}_{name}"] = stats[name]
    row["val_seg_class_iou"] = val_stats["seg_class_iou"]
    row["val_seg_class_acc"] = val_stats["seg_class_acc"]
    row["val_seg_class_dice"] = val_stats["seg_class_dice"]
    row["val_od_ap50_per_class"] = val_stats["od_ap50_per_class"]
    for name in ("loss_weight_seg", "loss_weight_depth", "loss_weight_od"):
        row[name] = train_stats[name]
    for name in ("gradient_clip_rate",):
        row[f"train_{name}"] = train_stats[name]
    return row


def loss_row(epoch, train_stats, val_stats):
    row = {"epoch": epoch}
    for split, stats in (("train", train_stats), ("val", val_stats)):
        for name in (
            "loss",
            "seg_loss",
            "depth_loss",
            "od_loss",
            "weighted_seg_loss",
            "weighted_depth_loss",
            "weighted_od_loss",
        ):
            row[f"{split}_{name}"] = stats[name]
    for task in ("seg", "depth", "od"):
        row[f"loss_weight_{task}"] = train_stats[f"loss_weight_{task}"]
    return row


def epoch_summary(epoch, epochs, cfg, train_stats, val_stats):
    parts = [
        f"Epoch {epoch}/{epochs}",
        f"train_loss={train_stats['loss']:.4f}",
        f"val_loss={val_stats['loss']:.4f}",
    ]
    conf = cfg["model_conf"]
    if conf.get("od", False):
        parts.extend(
            [
                f"AP50={val_stats['od_map50']:.4f}",
                f"AP50-95={val_stats['od_map50_95']:.4f}",
            ]
        )
    if conf.get("Seg", False):
        parts.append(f"mIoU={val_stats['seg_miou']:.4f}")
    if conf.get("Depth", False):
        parts.append(f"AbsRel={val_stats['depth_abs_rel']:.4f}")
    return " | ".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Train TriV-Net on the OldTown dataset."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress and informational output; errors remain visible.",
    )
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    validate_config(cfg)

    if args.quiet:
        cfg["disable_tqdm"] = True
    logging.basicConfig(
        stream=sys.stdout,
        format="%(asctime)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.WARNING if args.quiet else logging.INFO,
    )
    logger = logging.getLogger(__name__)
    seed = int(cfg.get("seed", 20260806))
    deterministic = bool(cfg.get("deterministic", True))
    set_global_seed(seed, deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = make_save_dir(cfg, args.run_name or Path(args.config).stem)
    save_yaml(save_dir / "config.yaml", cfg)

    model = build_original_trivnet(cfg).to(device)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 1e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-7)),
    )
    epochs = int(cfg.get("epochs", 100))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(1, int(float(cfg.get("lr_step_ratio", 0.25)) * epochs)),
        gamma=float(cfg.get("lr_gamma", 0.4)),
    )

    train_set = Tartan(True, cfg)
    val_set = Tartan(False, cfg)
    train_generator = torch.Generator().manual_seed(seed)
    loader_args = {
        "batch_size": int(cfg.get("batch_size", 4)),
        "collate_fn": collater,
        "num_workers": int(cfg.get("workers", 4)),
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        generator=train_generator,
        **loader_args,
    )
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    depth_loss_fn = silog_loss(0.85).to(device)
    od_loss_fn = build_od_loss(cfg).to(device)

    loss_balancer = LossBalancer(cfg, device)
    set_global_seed(seed, deterministic)

    logger.info(
        "save_dir=%s device=%s params=%s train=%s val=%s tasks=%s",
        save_dir,
        device,
        f"{total_params:,}",
        len(train_set),
        len(val_set),
        cfg["model_conf"],
    )
    start_time = time.time()
    best_val_loss = float("inf")
    best_checkpoint_score = float("-inf")
    best_checkpoint_epoch = None
    best_val_stats = None
    checkpoint_metric = None
    metrics_history = []
    losses_history = []

    for epoch in range(epochs):
        epoch_lr = to_float(optimizer.param_groups[0]["lr"])
        train_stats = run_epoch(
            model,
            train_loader,
            cfg,
            device,
            depth_loss_fn,
            od_loss_fn,
            loss_balancer,
            optimizer=optimizer,
            epoch=epoch,
        )
        with torch.no_grad():
            val_stats = run_epoch(
                model,
                val_loader,
                cfg,
                device,
                depth_loss_fn,
                od_loss_fn,
                loss_balancer,
                epoch=epoch,
            )
        scheduler.step()
        checkpoint_score, checkpoint_metric = validation_checkpoint_score(
            val_stats, cfg
        )
        logger.info(epoch_summary(epoch + 1, epochs, cfg, train_stats, val_stats))
        metrics_history.append(
            metric_row(
                epoch + 1,
                epoch_lr,
                checkpoint_metric,
                checkpoint_score,
                train_stats,
                val_stats,
            )
        )
        losses_history.append(loss_row(epoch + 1, train_stats, val_stats))
        torch.save(model.state_dict(), save_dir / "last.pth")
        best_val_loss = min(best_val_loss, val_stats["loss"])
        if best_checkpoint_epoch is None or checkpoint_score > best_checkpoint_score:
            best_checkpoint_score = checkpoint_score
            best_checkpoint_epoch = epoch + 1
            best_val_stats = dict(val_stats)
            torch.save(model.state_dict(), save_dir / "best.pth")
        save_yaml(save_dir / "metrics.yaml", metrics_history)
        save_yaml(save_dir / "losses.yaml", losses_history)

    save_yaml(
        save_dir / "model_info.yaml",
        {
            "architecture_family": "original_trivnet",
            "total_parameters": int(total_params),
            "training_time_sec": time.time() - start_time,
            "model": cfg.get("model", {}),
            "model_conf": cfg.get("model_conf", {}),
            "seg_num_classes": int(cfg["seg_num_classes"]),
            "od_num_classes": int(cfg["od_num_classes"]),
            "best_val_loss": best_val_loss,
            "checkpoint_metric": checkpoint_metric,
            "best_checkpoint_score": best_checkpoint_score,
            "best_checkpoint_epoch": best_checkpoint_epoch,
            "best_val_stats": best_val_stats,
            "seed": seed,
            "deterministic": deterministic,
            "loss_balance": loss_balancer.summary(),
        },
    )
    logger.info("done save_dir=%s", save_dir)


if __name__ == "__main__":
    main()
