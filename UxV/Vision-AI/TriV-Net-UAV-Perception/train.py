import argparse
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
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
from train_model.anchor_free_od import AnchorFreeODLoss, decode_anchor_free
from train_model.data import Tartan
from train_model.loss import silog_loss
from train_model.loss_balancing import LossBalancer
from train_model.mtl_models import models


def to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().mean().item())
    return float(value)


def save_yaml(path, data):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


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
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def colorize_segmentation(mask, num_classes):
    mask = mask.astype(np.int64)
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for cls in range(num_classes):
        palette[cls] = ((37 * cls) % 255, (17 * cls + 80) % 255, (97 * cls + 30) % 255)
    mask = np.clip(mask, 0, num_classes - 1)
    return palette[mask]


def colorize_depth(depth, max_depth=80.0):
    depth = np.clip(depth, 0, max_depth)
    depth = (depth / max_depth * 255).astype(np.uint8)
    return cv2.applyColorMap(depth, cv2.COLORMAP_PLASMA)


@torch.no_grad()
def save_epoch_outputs(model, datasets, epoch, output_dir, device, cfg):
    if not bool(cfg.get("save_epoch_outputs", False)):
        return []

    model.eval()
    epoch_dir = Path(output_dir) / "epoch_outputs" / f"epoch_{epoch + 1:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    seg_num_classes = int(cfg.get("seg_num_classes", 12))
    max_per_split = int(cfg.get("save_epoch_outputs_max_per_split", 4))
    saved_paths = []

    for split_name, dataset in datasets:
        by_scene = {}
        for idx, image_path in enumerate(dataset.image_front):
            scene_name = Path(image_path).parents[1].name
            by_scene.setdefault(scene_name, idx)
        for scene_name, idx in list(by_scene.items())[:max_per_split]:
            image, image_ori, _, _, _, _ = dataset[idx]
            batch = image.unsqueeze(0).to(device)
            seg, depth, od_outputs, _ = model(batch)
            rgb = (image_ori.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            od_vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            if cfg["model_conf"].get("od", False):
                detections = decode_anchor_free(
                    od_outputs,
                    image_size=(int(cfg["height"]), int(cfg["width"])),
                    score_thresh=float(cfg.get("vis_score_thresh", 0.25)),
                    nms_thresh=0.5,
                )[0]
                for box, cls_id, score in zip(detections["rois"][:50], detections["class_ids"][:50], detections["scores"][:50]):
                    x1, y1, x2, y2 = [int(v) for v in box]
                    cv2.rectangle(od_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(od_vis, f"{int(cls_id)}:{float(score):.2f}", (x1, max(0, y1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

            panels = [od_vis]
            if cfg["model_conf"].get("Seg", False):
                panels.append(colorize_segmentation(seg.argmax(dim=1)[0].detach().cpu().numpy(), seg_num_classes))
            if cfg["model_conf"].get("Depth", False):
                panels.append(colorize_depth(depth[0, 0].detach().cpu().numpy()))
            output = np.concatenate(panels, axis=1)
            save_path = epoch_dir / f"{split_name}_{scene_name}.png"
            cv2.imwrite(str(save_path), output)
            saved_paths.append(str(save_path))
    return saved_paths


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

    max_num_annots = max(annots.shape[0] for annots in annots_batch)
    padded_annots = torch.zeros((len(annots_batch), max_num_annots, 5), dtype=torch.float32)
    padded_annots[..., 4] = -1
    for b, annots in enumerate(annots_batch):
        if annots.shape[0] > 0:
            padded_annots[b, :annots.shape[0], :] = annots
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


@torch.no_grad()
def calibrate_initial_losses(model, loader, cfg, device, loss_silog, od_loss_fn, max_batches):
    was_training = model.training
    buffer_state = {
        name: buffer.detach().clone()
        for name, buffer in model.named_buffers()
    }
    model.train()
    conf = cfg["model_conf"]
    totals = {"seg": 0.0, "depth": 0.0, "od": 0.0}
    counts = {"seg": 0, "depth": 0, "od": 0}

    for batch_index, (imgs, _, seg_gt, depth_gt, depth_mask, annotations) in enumerate(loader):
        if batch_index >= max_batches:
            break
        imgs = imgs.to(device, non_blocking=True)
        seg, depth, od_outputs, _ = model(imgs)

        if conf.get("Seg", False):
            loss = F.cross_entropy(
                seg,
                seg_gt.to(device, non_blocking=True).long(),
                ignore_index=int(cfg.get("seg_ignore_index", 255)),
            )
            totals["seg"] += to_float(loss)
            counts["seg"] += 1
        if conf.get("Depth", False):
            loss = loss_silog(
                depth,
                depth_gt.to(device, non_blocking=True).float(),
                depth_mask.to(device, non_blocking=True),
            )
            totals["depth"] += to_float(loss)
            counts["depth"] += 1
        if conf.get("od", False):
            loss, _ = od_loss_fn(
                od_outputs,
                annotations.to(device, non_blocking=True),
                image_size=(int(cfg["height"]), int(cfg["width"])),
            )
            totals["od"] += to_float(loss)
            counts["od"] += 1

    for name, buffer in model.named_buffers():
        buffer.copy_(buffer_state[name])
    model.train(was_training)
    references = {
        task: totals[task] / counts[task]
        for task in totals
        if counts[task] > 0
    }
    if not references or any(not np.isfinite(value) or value <= 0 for value in references.values()):
        raise RuntimeError("Initial-loss calibration produced invalid values: {}".format(references))
    return references


def gradnorm_shared_parameters(model, cfg):
    scope = str(cfg.get("loss_balance", {}).get("gradnorm_shared_scope", "neck_lateral5"))
    if scope == "neck_lateral5":
        return tuple(model.neck.lateral5.parameters())
    if scope == "neck":
        return tuple(model.neck.parameters())
    if scope == "backbone":
        return tuple(model.backbone.parameters())
    raise ValueError("Unsupported gradnorm_shared_scope: {}".format(scope))


def validation_checkpoint_score(val_stats, cfg):
    checkpoint_cfg = cfg.get("checkpoint", {})
    metric = str(checkpoint_cfg.get("metric", "val_loss")).lower()
    if metric == "val_loss":
        return -float(val_stats["loss"]), metric
    if metric != "stl_retention_geomean":
        raise ValueError("Unsupported checkpoint.metric: {}".format(metric))

    references = checkpoint_cfg.get("stl_reference", {})
    required = ("seg_miou", "depth_abs_rel", "od_map50_95")
    if any(float(references.get(key, 0.0)) <= 0 for key in required):
        raise ValueError("checkpoint.stl_reference requires positive {} values.".format(required))
    if val_stats["seg_miou"] <= 0 or val_stats["depth_abs_rel"] <= 0 or val_stats["od_map50_95"] <= 0:
        return float("-inf"), metric

    retention = (
        (float(val_stats["seg_miou"]) / float(references["seg_miou"]))
        * (float(references["depth_abs_rel"]) / float(val_stats["depth_abs_rel"]))
        * (float(val_stats["od_map50_95"]) / float(references["od_map50_95"]))
    )
    return float(retention ** (1.0 / 3.0)), metric


def run_epoch(model, loader, cfg, device, loss_silog, od_loss_fn, loss_balancer,
              optimizer=None, logger=None, epoch=0, shared_parameters=None):
    is_train = optimizer is not None
    model.train(is_train)
    conf = cfg["model_conf"]
    seg_ignore_index = int(cfg.get("seg_ignore_index", 255))
    seg_num_classes = int(cfg.get("seg_num_classes", 12))
    od_num_classes = int(cfg.get("od_num_classes", 5))
    seg_confmat = torch.zeros((seg_num_classes, seg_num_classes), dtype=torch.long, device=device)
    depth_metric_sums = {
        "abs_rel": 0.0,
        "sq_rel": 0.0,
        "rmse": 0.0,
        "rmse_log": 0.0,
        "mae": 0.0,
        "delta1": 0.0,
        "delta2": 0.0,
        "delta3": 0.0,
    }
    depth_metric_batches = 0
    od_metric_acc = DetectionMetricAccumulator(od_num_classes) if (conf.get("od", False) and not is_train and bool(cfg.get("eval_od_metrics", True))) else None
    totals = {
        "loss": 0.0,
        "seg_loss": 0.0,
        "depth_loss": 0.0,
        "od_loss": 0.0,
        "od_iou": 0.0,
        "weighted_seg_loss": 0.0,
        "weighted_depth_loss": 0.0,
        "weighted_od_loss": 0.0,
        "loss_weight_seg": 0.0,
        "loss_weight_depth": 0.0,
        "loss_weight_od": 0.0,
        "gradnorm_loss": 0.0,
        "gradnorm_raw_seg": 0.0,
        "gradnorm_raw_depth": 0.0,
        "gradnorm_raw_od": 0.0,
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
        disable=bool(cfg.get("disable_tqdm", False)),
    )
    for batchi, (imgs, _, seg_gt, depth_gt, depth_mask, annotations) in progress:
        imgs = imgs.to(device)
        if is_train:
            optimizer.zero_grad()

        seg, depth, od_outputs, _ = model(imgs)
        zero = imgs.sum() * 0.0
        loss_seg = zero
        loss_depth = zero
        loss_od = zero
        seg_iou = 0.0
        od_iou = torch.zeros((), device=device)

        if conf.get("Seg", False):
            seg_gt = seg_gt.to(device).long()
            loss_seg = F.cross_entropy(seg, seg_gt, ignore_index=seg_ignore_index)
            valid = seg_gt != seg_ignore_index
            if valid.any():
                seg_iou = (seg.argmax(dim=1)[valid] == seg_gt[valid]).float().mean().item()
                update_seg_confmat(seg_confmat, seg.argmax(dim=1), seg_gt, seg_num_classes, seg_ignore_index)

        if conf.get("Depth", False):
            depth_gt = depth_gt.to(device).float()
            depth_mask = depth_mask.to(device)
            loss_depth = loss_silog(depth, depth_gt, depth_mask)
            depth_metrics = depth_metric_values(depth, depth_gt, depth_mask)
            if depth_metrics is not None:
                for key, value in depth_metrics.items():
                    depth_metric_sums[key] += value
                depth_metric_batches += 1

        if conf.get("od", False):
            annotations = annotations.to(device)
            loss_od, od_iou = od_loss_fn(od_outputs, annotations, image_size=(int(cfg["height"]), int(cfg["width"])))
            if od_metric_acc is not None:
                detections = decode_anchor_free(
                    od_outputs,
                    image_size=(int(cfg["height"]), int(cfg["width"])),
                    score_thresh=float(cfg.get("eval_od_score_thresh", 0.001)),
                    nms_thresh=float(cfg.get("eval_od_nms_thresh", 0.5)),
                    topk=int(cfg.get("eval_od_topk", 100)),
                )
                od_metric_acc.update(detections, annotations)

        task_losses = {"seg": loss_seg, "depth": loss_depth, "od": loss_od}
        loss, loss_weights = loss_balancer.combine(task_losses)

        gradnorm_stats = None
        if is_train and loss_balancer.mode == "gradnorm":
            if shared_parameters is None:
                raise ValueError("GradNorm training requires shared_parameters.")
            gradnorm_stats = loss_balancer.gradnorm_step(task_losses, shared_parameters)

        if is_train:
            loss.backward()
            total_gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_threshold)
            if to_float(total_gradient_norm) > clip_threshold:
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
            totals["loss_weight_{}".format(task)] += weight
            totals["weighted_{}_loss".format(task)] += weight * to_float(task_losses[task])
        if gradnorm_stats is not None:
            totals["gradnorm_loss"] += gradnorm_stats["gradnorm_loss"]
            for task, value in gradnorm_stats["raw_gradient_norms"].items():
                totals["gradnorm_raw_{}".format(task)] += value

        if logger and is_train and (batchi + 1) % int(cfg.get("log_interval", 20)) == 0:
            logger.info(
                f"TRAIN[{epoch + 1}] [{batchi + 1}/{len(loader)}] "
                f"loss {to_float(loss):.4f} seg {to_float(loss_seg):.4f} "
                f"depth {to_float(loss_depth):.4f} od {to_float(loss_od):.4f} "
                f"weights {loss_weights}"
            )
        progress.set_postfix(
            loss=f"{to_float(loss):.3f}",
            seg=f"{to_float(loss_seg):.3f}",
            depth=f"{to_float(loss_depth):.3f}",
            od=f"{to_float(loss_od):.3f}",
        )

    divisor = max(1, processed_batches)
    stats = {k: v / divisor for k, v in totals.items()}
    stats["gradient_clip_rate"] = float(clipped_batches) / divisor if is_train else 0.0
    if conf.get("Seg", False):
        stats.update(seg_metrics_from_confmat(seg_confmat))
    else:
        stats.update({
            "seg_pixel_acc": 0.0,
            "seg_mean_acc": 0.0,
            "seg_miou": 0.0,
            "seg_mean_dice": 0.0,
            "seg_class_iou": [],
            "seg_class_acc": [],
            "seg_class_dice": [],
        })
    if conf.get("Depth", False) and depth_metric_batches > 0:
        stats.update({f"depth_{k}": v / depth_metric_batches for k, v in depth_metric_sums.items()})
    else:
        stats.update({f"depth_{k}": 0.0 for k in depth_metric_sums})
    if od_metric_acc is not None:
        stats.update(od_metric_acc.compute())
    else:
        stats.update({
            "od_map50": 0.0,
            "od_map50_95": 0.0,
            "od_precision50": 0.0,
            "od_recall50": 0.0,
            "od_f1_50": 0.0,
            "od_ap50_per_class": [],
        })
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="train_config.yaml")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    conf = cfg.get("model_conf", {})
    if not any([conf.get("Seg", False), conf.get("Depth", False), conf.get("od", False)]):
        raise ValueError("model_conf에서 최소 1개 태스크는 true여야 합니다.")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(stream=sys.stdout, format="%(asctime)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    logger = logging.getLogger(__name__)

    seed = int(cfg.get("seed", 20260806))
    deterministic = bool(cfg.get("deterministic", True))
    set_global_seed(seed, deterministic=deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = make_save_dir(cfg, args.run_name or Path(args.config).stem)
    save_yaml(save_dir / "config.yaml", cfg)

    model = models(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1e-4)), weight_decay=float(cfg.get("weight_decay", 1e-7)))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(1, int(float(cfg.get("lr_step_ratio", 0.25)) * int(cfg.get("epochs", 10)))),
        gamma=float(cfg.get("lr_gamma", 0.4)),
    )

    train_set = Tartan(True, cfg)
    val_set = Tartan(False, cfg)
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=int(cfg.get("batch_size", 4)), collate_fn=collater, shuffle=True,
                              num_workers=int(cfg.get("workers", 4)), pin_memory=True,
                              worker_init_fn=seed_worker, generator=train_generator)
    val_loader = DataLoader(val_set, batch_size=int(cfg.get("batch_size", 4)), collate_fn=collater, shuffle=False,
                            num_workers=int(cfg.get("workers", 4)), pin_memory=True,
                            worker_init_fn=seed_worker)

    loss_silog = silog_loss(0.85).to(device)
    od_loss_fn = AnchorFreeODLoss(
        num_classes=int(cfg.get("od_num_classes", 5)),
        box_weight=float(cfg.get("od_box_loss_weight", 5.0)),
    ).to(device)

    balance_cfg = cfg.get("loss_balance", {})
    balance_mode = str(balance_cfg.get("mode", "static")).lower()
    initial_losses = None
    if balance_mode in ("initial_loss", "gradnorm"):
        calibration_generator = torch.Generator()
        calibration_generator.manual_seed(seed + 1)
        calibration_loader = DataLoader(
            train_set,
            batch_size=int(cfg.get("batch_size", 4)),
            collate_fn=collater,
            shuffle=True,
            num_workers=int(cfg.get("workers", 4)),
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=calibration_generator,
        )
        calibration_batches = int(balance_cfg.get("calibration_batches", 100))
        initial_losses = calibrate_initial_losses(
            model,
            calibration_loader,
            cfg,
            device,
            loss_silog,
            od_loss_fn,
            calibration_batches,
        )
        logger.info(f"initial-loss calibration batches={calibration_batches} values={initial_losses}")

    loss_balancer = LossBalancer(cfg, device, initial_losses=initial_losses)
    shared_parameters = gradnorm_shared_parameters(model, cfg) if balance_mode == "gradnorm" else None
    save_yaml(save_dir / "loss_balance.yaml", loss_balancer.summary())

    # Calibration must not change the stochastic sequence used by actual training.
    set_global_seed(seed, deterministic=deterministic)

    best_val_loss = float("inf")
    best_checkpoint_score = float("-inf")
    best_checkpoint_epoch = None
    checkpoint_metric = None
    metrics_history = []
    losses_history = []
    epochs = int(cfg.get("epochs", 10))
    logger.info(f"save_dir={save_dir}")
    logger.info(
        f"device={device} params={total_params:,} train={len(train_set)} val={len(val_set)} "
        f"seed={seed} deterministic={deterministic} loss_balance={loss_balancer.summary()}"
    )

    for epoch in range(epochs):
        train_stats = run_epoch(
            model, train_loader, cfg, device, loss_silog, od_loss_fn, loss_balancer,
            optimizer, logger, epoch, shared_parameters,
        )
        with torch.no_grad():
            val_stats = run_epoch(
                model, val_loader, cfg, device, loss_silog, od_loss_fn, loss_balancer,
            )

        scheduler.step()
        checkpoint_score, checkpoint_metric = validation_checkpoint_score(val_stats, cfg)
        logger.info(
            f"VAL[{epoch + 1}] loss {val_stats['loss']:.4f} seg {val_stats['seg_loss']:.4f} "
            f"depth {val_stats['depth_loss']:.4f} od {val_stats['od_loss']:.4f} "
            f"seg_miou {val_stats['seg_miou']:.4f} depth_abs_rel {val_stats['depth_abs_rel']:.4f} "
            f"od_map50 {val_stats['od_map50']:.4f} od_map50_95 {val_stats['od_map50_95']:.4f} "
            f"checkpoint_score {checkpoint_score:.6f} weights {loss_balancer.current_weight_dict()}"
        )

        metrics_history.append({
            "epoch": epoch + 1,
            "lr": to_float(optimizer.param_groups[0]["lr"]),
            "checkpoint_metric": checkpoint_metric,
            "checkpoint_score": checkpoint_score,
            "train_loss": train_stats["loss"],
            "val_loss": val_stats["loss"],
            "train_seg_pixel_acc": train_stats["seg_pixel_acc"],
            "val_seg_pixel_acc": val_stats["seg_pixel_acc"],
            "train_seg_mean_acc": train_stats["seg_mean_acc"],
            "val_seg_mean_acc": val_stats["seg_mean_acc"],
            "train_seg_miou": train_stats["seg_miou"],
            "val_seg_miou": val_stats["seg_miou"],
            "train_seg_mean_dice": train_stats["seg_mean_dice"],
            "val_seg_mean_dice": val_stats["seg_mean_dice"],
            "val_seg_class_iou": val_stats["seg_class_iou"],
            "val_seg_class_acc": val_stats["seg_class_acc"],
            "val_seg_class_dice": val_stats["seg_class_dice"],
            "train_depth_abs_rel": train_stats["depth_abs_rel"],
            "val_depth_abs_rel": val_stats["depth_abs_rel"],
            "train_depth_sq_rel": train_stats["depth_sq_rel"],
            "val_depth_sq_rel": val_stats["depth_sq_rel"],
            "train_depth_rmse": train_stats["depth_rmse"],
            "val_depth_rmse": val_stats["depth_rmse"],
            "train_depth_rmse_log": train_stats["depth_rmse_log"],
            "val_depth_rmse_log": val_stats["depth_rmse_log"],
            "train_depth_mae": train_stats["depth_mae"],
            "val_depth_mae": val_stats["depth_mae"],
            "train_depth_delta1": train_stats["depth_delta1"],
            "val_depth_delta1": val_stats["depth_delta1"],
            "train_depth_delta2": train_stats["depth_delta2"],
            "val_depth_delta2": val_stats["depth_delta2"],
            "train_depth_delta3": train_stats["depth_delta3"],
            "val_depth_delta3": val_stats["depth_delta3"],
            "train_od_iou": train_stats["od_iou"],
            "val_od_iou": val_stats["od_iou"],
            "val_od_map50": val_stats["od_map50"],
            "val_od_map50_95": val_stats["od_map50_95"],
            "val_od_precision50": val_stats["od_precision50"],
            "val_od_recall50": val_stats["od_recall50"],
            "val_od_f1_50": val_stats["od_f1_50"],
            "val_od_ap50_per_class": val_stats["od_ap50_per_class"],
            "loss_weight_seg": train_stats["loss_weight_seg"],
            "loss_weight_depth": train_stats["loss_weight_depth"],
            "loss_weight_od": train_stats["loss_weight_od"],
            "train_gradient_clip_rate": train_stats["gradient_clip_rate"],
            "train_gradnorm_loss": train_stats["gradnorm_loss"],
            "train_gradnorm_raw_seg": train_stats["gradnorm_raw_seg"],
            "train_gradnorm_raw_depth": train_stats["gradnorm_raw_depth"],
            "train_gradnorm_raw_od": train_stats["gradnorm_raw_od"],
        })
        losses_history.append({
            "epoch": epoch + 1,
            "train_loss": train_stats["loss"],
            "val_loss": val_stats["loss"],
            "train_seg_loss": train_stats["seg_loss"],
            "val_seg_loss": val_stats["seg_loss"],
            "train_depth_loss": train_stats["depth_loss"],
            "val_depth_loss": val_stats["depth_loss"],
            "train_od_loss": train_stats["od_loss"],
            "val_od_loss": val_stats["od_loss"],
            "train_weighted_seg_loss": train_stats["weighted_seg_loss"],
            "val_weighted_seg_loss": val_stats["weighted_seg_loss"],
            "train_weighted_depth_loss": train_stats["weighted_depth_loss"],
            "val_weighted_depth_loss": val_stats["weighted_depth_loss"],
            "train_weighted_od_loss": train_stats["weighted_od_loss"],
            "val_weighted_od_loss": val_stats["weighted_od_loss"],
            "loss_weight_seg": train_stats["loss_weight_seg"],
            "loss_weight_depth": train_stats["loss_weight_depth"],
            "loss_weight_od": train_stats["loss_weight_od"],
        })

        torch.save(model.state_dict(), save_dir / "last.pth")
        best_val_loss = min(best_val_loss, val_stats["loss"])
        if best_checkpoint_epoch is None or checkpoint_score > best_checkpoint_score:
            best_checkpoint_score = checkpoint_score
            best_checkpoint_epoch = epoch + 1
            torch.save(model.state_dict(), save_dir / "best.pth")
        save_yaml(save_dir / "metrics.yaml", metrics_history)
        save_yaml(save_dir / "losses.yaml", losses_history)
        save_yaml(save_dir / "loss_balance.yaml", loss_balancer.summary())

        saved_outputs = save_epoch_outputs(model, [("train", train_set), ("val", val_set)], epoch, save_dir, device, cfg)
        if saved_outputs:
            logger.info(f"saved {len(saved_outputs)} epoch output images")

    model.eval()
    inference_time = None
    with torch.no_grad():
        for imgs, _, _, _, _, _ in val_loader:
            dummy_input = imgs[:1].to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.time()
            for _ in range(20):
                model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_time = (time.time() - start_time) / 20
            break

    save_yaml(save_dir / "model_info.yaml", {
        "total_parameters": int(total_params),
        "inference_time_per_image_sec": None if inference_time is None else round(inference_time, 6),
        "model": cfg.get("model", {}),
        "model_conf": cfg.get("model_conf", {}),
        "best_val_loss": best_val_loss,
        "checkpoint_metric": checkpoint_metric,
        "best_checkpoint_score": best_checkpoint_score,
        "best_checkpoint_epoch": best_checkpoint_epoch,
        "seed": seed,
        "deterministic": deterministic,
        "loss_balance": loss_balancer.summary(),
    })
    logger.info(f"done save_dir={save_dir}")


if __name__ == "__main__":
    main()
