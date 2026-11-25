"""Model-agnostic SLZ trainer.

Model-specific work goes through the adapter; evaluation goes through
:func:`slz_benchmark.engine.evaluator.evaluate`. The checkpoint with the best val mIoU is
written via ``adapter.save`` to ``<output>/best_model/``, next to ``config.json``,
``history.json``, ``best_metrics.json`` and (unless ``skip_test``) ``test_metrics.json``.
"""

from __future__ import annotations

import json
import platform
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from ..adapters.base import SegAdapter
from ..config import RunConfig
from ..datasets.uavscenes.splits import canonical_scheme, manifest_path
from ..metrics import binary_segmentation_metrics, update_confusion_matrix
from ..losses import SLZ_IGNORE_INDEX, slz_class_weights
from .. import run_info
from .evaluator import evaluate, format_metrics


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_determinism(enabled: bool) -> None:
    """Optional reproducible kernels. ``warn_only`` so a non-deterministic op degrades
    to a warning instead of crashing a run."""
    if not enabled:
        return
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _split_provenance(cfg: RunConfig) -> dict:
    """sha256 of the fold's train/val/test manifests + the guard gap (loso)."""
    out: dict = {"manifests": {}, "guard": None}
    try:
        paths = {
            s: manifest_path(cfg.root, s, scheme=cfg.scheme, fold=cfg.fold)
            for s in ("train", "val", "test")
        }
        out["manifests"] = {
            s: {"path": str(p), **run_info.sha256_files([p])}
            for s, p in paths.items()
        }
    except Exception:
        pass
    try:
        scheme = canonical_scheme(cfg.scheme)
        if scheme.startswith("loso"):
            info = json.loads(
                (Path(cfg.root) / "manifests" / scheme / "split_info.json").read_text()
            )
            out["guard"] = info.get("guard")
            out["d_guard_m"] = info.get("d_guard_m")
            out["split_info_manifest_sha256"] = info.get("manifest_sha256")
    except Exception:
        pass
    return out


def poly_scheduler(
    optimizer,
    total_steps: int,
    warmup_ratio: float = 0.05,
    power: float = 0.9,
) -> LambdaLR:
    """Linear warmup, then ``(1 - p) ** power``."""
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return max(step / max(warmup_steps, 1), 1e-8)
        remain = max(total_steps - step, 0)
        denom = max(total_steps - warmup_steps, 1)
        return (remain / denom) ** power

    return LambdaLR(optimizer, lr_lambda)


class Trainer:
    def __init__(
        self,
        adapter: SegAdapter,
        model: torch.nn.Module,
        train_loader,
        val_loader,
        device: torch.device,
        cfg: RunConfig,
        test_loader=None,
        rank: int = 0,
        world_size: int = 1,
        global_batch_size: Optional[int] = None,
    ):
        self.adapter = adapter
        # ``raw_model`` is always the plain nn.Module — everything adapter-specific
        # (save/load, loss_breakdown's ``_last_loss_dict``) lives there, never on the
        # DDP wrapper. ``self.model`` is what the train/eval forward/backward calls go
        # through (DDP-wrapped when world_size > 1, else the same object as raw_model).
        self.raw_model = model
        self.rank = rank
        self.world_size = world_size
        self.is_main = rank == 0
        self.ddp_enabled = world_size > 1
        self.global_batch_size = global_batch_size or cfg.batch_size
        if self.ddp_enabled:
            self.model = DistributedDataParallel(model, device_ids=[device.index])
        else:
            self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.cfg = cfg

        self.amp_enabled = bool(cfg.amp) and device.type == "cuda"
        self.class_weights = slz_class_weights(cfg.safe_weight, device)

        self.optimizer = AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        # With a DistributedSampler, len(train_loader) is already the per-rank batch
        # count (the sampler splits the dataset before the DataLoader ever sees it),
        # so the warmup/poly schedule's total_steps needs no DDP-specific adjustment —
        # every rank computes the identical total_steps from its identical per-rank
        # batch count.
        steps_per_epoch = self._planned_steps(len(train_loader))
        self.scheduler = poly_scheduler(
            self.optimizer,
            total_steps=cfg.epochs * steps_per_epoch,
            warmup_ratio=cfg.warmup_ratio,
            power=cfg.poly_power,
        )
        self.scaler = torch.amp.GradScaler(device.type, enabled=self.amp_enabled)

        self.output = Path(cfg.output)
        self.best_dir = self.output / "best_model"

    def _planned_steps(self, n_batches: int) -> int:
        if self.cfg.max_steps is not None:
            return min(self.cfg.max_steps, n_batches)
        return n_batches

    # -- one training epoch ---------------------------------------------------
    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        sampler = getattr(self.train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            # DistributedSampler reseeds its per-epoch shuffle from (seed, epoch); every
            # rank must call this with the same epoch number or their shards desync.
            sampler.set_epoch(epoch)
        cm = torch.zeros((2, 2), dtype=torch.int64, device=self.device)
        running_loss = 0.0
        seen = 0
        breakdown_sums: dict[str, float] = {}

        pbar = tqdm(
            self.train_loader,
            desc=f"Train {epoch}/{self.cfg.epochs}",
            dynamic_ncols=True,
            leave=True,
            disable=not self.is_main,
        )
        for step, batch in enumerate(pbar):
            if self.cfg.max_steps is not None and step >= self.cfg.max_steps:
                break

            target = batch["slz"].to(self.device, non_blocking=True)
            bsz = target.shape[0]

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=self.device.type, enabled=self.amp_enabled
            ):
                # The adapter owns the training objective (default = pixel CE with
                # ignore_index=255; a mask-classification model overrides it). The
                # trainer has no model-specific branch.
                loss, logits = self.adapter.compute_train_loss(
                    self.model, batch, self.device, class_weights=self.class_weights
                )

            self.scaler.scale(loss).backward()
            if self.cfg.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip
                )
            # scaler.step() silently SKIPS the wrapped optimizer.step() whenever it finds
            # inf/NaN gradients (routine on the first few iterations, before the AMP loss
            # scale has calibrated down from its default init_scale=65536 — see
            # torch/amp/grad_scaler.py's _maybe_opt_step). Advancing the LR scheduler
            # unconditionally after a skipped step both (a) desyncs the warmup/poly
            # schedule from the actual count of optimizer updates (it would run ahead by
            # one step per skip) and (b) trips PyTorch's own
            # "lr_scheduler.step() before optimizer.step()" UserWarning the first time this
            # happens. get_scale() before/after scaler.update() is the standard way to
            # detect a skip: update() halves the scale (backoff_factor) iff the step was
            # skipped, otherwise the scale stays the same or grows — it never decreases on
            # a real step. amp-disabled runs always report scale 1.0 -> 1.0, so this is a
            # no-op there (scheduler.step() still fires every iteration).
            scale_before_step = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            optimizer_step_happened = self.scaler.get_scale() >= scale_before_step
            if optimizer_step_happened:
                self.scheduler.step()

            running_loss += float(loss.item()) * bsz
            seen += bsz
            # optional per-term loss of the adapter (empty for every model here); read
            # from raw_model, since self.model may be the DDP wrapper
            for k, v in self.adapter.loss_breakdown(self.raw_model).items():
                breakdown_sums[k] = breakdown_sums.get(k, 0.0) + v * bsz
            if logits is not None:
                with torch.no_grad():
                    update_confusion_matrix(
                        cm, logits.argmax(1), target, ignore_index=SLZ_IGNORE_INDEX
                    )
            pbar.set_postfix(
                loss=f"{float(loss.item()):.4f}",
                lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
            )

        if self.ddp_enabled:
            # Each rank trained on its own shard, so cm / running_loss / seen are
            # partial: sum them so the reported train metrics cover the whole epoch.
            # Diagnostic only; the checkpoint is selected on val.
            dist.all_reduce(cm, op=dist.ReduceOp.SUM)
            loss_seen = torch.tensor(
                [running_loss, float(seen)], device=self.device, dtype=torch.float64
            )
            dist.all_reduce(loss_seen, op=dist.ReduceOp.SUM)
            running_loss, seen = loss_seen[0].item(), loss_seen[1].item()
            # loss_breakdown is not all-reduced: under DDP it is rank 0's shard only.

        metrics = binary_segmentation_metrics(cm)
        metrics["loss"] = running_loss / max(seen, 1)
        metrics["loss_kind"] = self.adapter.loss_kind
        if breakdown_sums:
            metrics["loss_breakdown"] = {
                k: v / max(seen, 1) for k, v in breakdown_sums.items()
            }
        return metrics

    # -- full run -----------------------------------------------------------
    def fit(self) -> dict:
        cfg = self.cfg
        # Wall-clock timing: rank 0 records per-epoch train/val seconds into
        # history.json and the whole run into timing.json. Timing only -- nothing here
        # changes what is trained or evaluated. train_s includes data loading (the real
        # cost of an epoch); the per-step .item() already synchronizes CUDA.
        fit_t0 = time.perf_counter()
        started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        # Same seed on every rank on purpose: DDP's constructor broadcasts rank 0's
        # parameters to every other rank right after wrapping, so any pre-wrap
        # divergence (e.g. a mismatched-size head reinitializing differently per rank)
        # self-heals — this call doesn't need a rank offset.
        seed_everything(cfg.seed)
        apply_determinism(cfg.deterministic)

        # Only rank 0 touches the filesystem or the outer test loader/eval — the other
        # ranks' job is purely to compute gradients DDP averages in. A barrier after
        # each rank-0-only block keeps every rank in step before the next epoch's
        # collective (backward all-reduce, or the train-metrics all_reduce above).
        if self.is_main:
            self.output.mkdir(parents=True, exist_ok=True)
            config_payload = cfg.to_dict()
            config_payload.update(
                device=str(self.device),
                amp_enabled=self.amp_enabled,
                train_batches=len(self.train_loader),
                val_batches=len(self.val_loader),
                # -- DDP -------------------------------------------------------------
                ddp_enabled=self.ddp_enabled,
                world_size=self.world_size,
                global_batch_size=self.global_batch_size,
                per_process_batch_size=cfg.batch_size,
                # -- what was run: model, loss, class weights, splits, environment ---
                model_adapter=self.adapter.name,
                model_name=cfg.model_name,
                loss_kind=self.adapter.loss_kind,
                class_weights=[float(w) for w in self.class_weights.tolist()],
                cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
                split_provenance=_split_provenance(cfg),
                provenance=run_info.collect(),
            )
            (self.output / "config.json").write_text(
                json.dumps(config_payload, indent=2), encoding="utf-8"
            )
        if self.ddp_enabled:
            dist.barrier()

        history = []
        best_miou = -1.0
        best_epoch = -1

        for epoch in range(1, cfg.epochs + 1):
            if self.is_main:
                print("\n" + "=" * 100)
                print(f"Epoch {epoch}/{cfg.epochs}")
                print("=" * 100)

            ep_t0 = time.perf_counter()
            train_metrics = self._train_epoch(epoch)
            ep_t1 = time.perf_counter()

            if self.is_main:
                # raw_model, not self.model: evaluate() runs no backward, so it does
                # not go through the DDP wrapper. Rank 0 alone scores the whole
                # (un-sharded) val loader.
                val_metrics, _ = evaluate(
                    self.adapter,
                    self.raw_model,
                    self.val_loader,
                    self.device,
                    amp=self.amp_enabled,
                    class_weights=self.class_weights,
                    max_steps=cfg.max_steps,
                    desc=f"Val   {epoch}/{cfg.epochs}",
                )

                ep_t2 = time.perf_counter()
                print(format_metrics("Train:", train_metrics))
                print(format_metrics("Val  :", val_metrics))
                print(f"[TIME] epoch {epoch}: train {ep_t1 - ep_t0:.1f}s, val {ep_t2 - ep_t1:.1f}s")

                history.append(
                    {
                        "epoch": epoch,
                        "lr": self.optimizer.param_groups[0]["lr"],
                        "train": train_metrics,
                        "val": val_metrics,
                        "time": {"train_s": round(ep_t1 - ep_t0, 3),
                                 "val_s": round(ep_t2 - ep_t1, 3),
                                 "epoch_s": round(ep_t2 - ep_t0, 3)},
                    }
                )
                (self.output / "history.json").write_text(
                    json.dumps(history, indent=2), encoding="utf-8"
                )

                if val_metrics["miou"] > best_miou:
                    best_miou = val_metrics["miou"]
                    best_epoch = epoch
                    self.adapter.save(self.raw_model, self.best_dir)
                    (self.output / "best_metrics.json").write_text(
                        json.dumps(
                            {
                                "best_epoch": best_epoch,
                                "best_val_miou": best_miou,
                                "val_metrics": val_metrics,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    print(f"[BEST] epoch={best_epoch}, val mIoU={best_miou:.4f}")

            if self.ddp_enabled:
                dist.barrier()

        result = {"best_epoch": best_epoch, "best_val_miou": best_miou}
        train_loop_s = time.perf_counter() - fit_t0
        test_s = None

        if self.is_main and not cfg.skip_test and self.test_loader is not None:
            test_t0 = time.perf_counter()
            best_model = self.adapter.load(self.best_dir, self.device)
            test_metrics, _ = evaluate(
                self.adapter,
                best_model,
                self.test_loader,
                self.device,
                amp=self.amp_enabled,
                class_weights=self.class_weights,
                max_steps=cfg.max_steps,
                desc="Test",
            )
            print(format_metrics("Test :", test_metrics))
            (self.output / "test_metrics.json").write_text(
                json.dumps(
                    {
                        "best_epoch": best_epoch,
                        "best_val_miou": best_miou,
                        "test": test_metrics,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            result["test_metrics"] = test_metrics
            test_s = time.perf_counter() - test_t0

        if self.is_main:
            ep_times = [h["time"] for h in history]
            timing = {
                "started_at": started_at,
                "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "epochs": len(history),
                "fit_wall_s": round(time.perf_counter() - fit_t0, 1),
                "train_loop_wall_s": round(train_loop_s, 1),  # all epochs incl. val + checkpoint saves
                "train_s_total": round(sum(t["train_s"] for t in ep_times), 1),
                "val_s_total": round(sum(t["val_s"] for t in ep_times), 1),
                "test_s": None if test_s is None else round(test_s, 1),
                "mean_epoch_s": round(sum(t["epoch_s"] for t in ep_times) / max(len(ep_times), 1), 2),
                "mean_train_epoch_s": round(sum(t["train_s"] for t in ep_times) / max(len(ep_times), 1), 2),
                "device": str(self.device),
                "gpu_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
                "world_size": self.world_size,
                "global_batch_size": self.global_batch_size,
                "amp_enabled": self.amp_enabled,
                "hostname": platform.node(),
                "note": "wall clock on rank 0; train_s includes data loading; excludes model "
                        "build / dataset setup before fit()",
            }
            (self.output / "timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
            print(f"[TIME] run: {timing['fit_wall_s'] / 60:.1f} min "
                  f"(mean epoch {timing['mean_epoch_s']:.1f}s)")

        if self.ddp_enabled:
            dist.barrier()

        return result
