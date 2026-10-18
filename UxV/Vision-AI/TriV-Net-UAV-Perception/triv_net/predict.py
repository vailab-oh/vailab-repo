#!/usr/bin/env python3
import argparse
import copy
import inspect
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from train_model.data import normalize_img
from train_model.original_trivnet_training import (
    build_original_trivnet,
    decode_original_detections,
)
from train import set_global_seed
from visualization import (
    OLDTOWN_OD_CLASS_NAMES,
    depth_visualization,
    detection_visualization,
    segmentation_visualization,
)


def load_checkpoint(run_dir, device):
    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    if cfg.get("model", {}).get("architecture_family") != "original_trivnet":
        raise ValueError(
            "This predictor accepts original TriV-Net runs, not TriV-Net+ runs."
        )
    if cfg["seg_num_classes"] != 12 or cfg["od_num_classes"] != 5:
        raise ValueError(
            "The packaged visualizations use the current OldTown 12-SS/5-OD mapping."
        )
    set_global_seed(
        int(cfg.get("seed", 20260806)), bool(cfg.get("deterministic", True))
    )
    construction = copy.deepcopy(cfg)
    construction["model"]["pretrained"] = False
    model = build_original_trivnet(construction).to(device)
    options = (
        {"weights_only": True}
        if "weights_only" in inspect.signature(torch.load).parameters
        else {}
    )
    state = torch.load(str(run_dir / "best.pth"), map_location="cpu", **options)
    model.load_state_dict(state, strict=True)
    return cfg, model.eval()


def save_png(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError("Could not write {}".format(path))


@torch.no_grad()
def predict(args):
    if not 0 <= args.seg_alpha <= 1:
        raise ValueError("--seg-alpha must be between 0 and 1")
    device = torch.device(args.device)
    cfg, model = load_checkpoint(args.run_dir, device)
    height, width = int(cfg["height"]), int(cfg["width"])
    if height % 128 or width % 128:
        raise ValueError(
            "Legacy anchor input height and width must be divisible by 128"
        )
    rgb = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(args.image)
    rgb = cv2.resize(
        cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    images = normalize_img(rgb).unsqueeze(0).to(device)
    seg, depth, classification, regression = model(images)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    save_png(output / "input.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    outputs = ["input.png"]
    if seg is not None:
        save_png(
            output / "seg.png",
            segmentation_visualization(rgb, seg, "oldtown", args.seg_alpha),
        )
        mask = seg.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        save_png(output / "seg_train_id.png", mask)
        outputs.extend(["seg.png", "seg_train_id.png"])
    if depth is not None:
        save_png(output / "depth.png", depth_visualization(depth, height, width, 80.0))
        np.save(str(output / "depth_m.npy"), depth[0, 0].cpu().numpy())
        outputs.extend(["depth.png", "depth_m.npy"])
    if classification is not None:
        detection = decode_original_detections(
            model,
            images,
            classification,
            regression,
            score_thresh=args.score_threshold,
            nms_thresh=args.nms_threshold,
            topk=args.topk,
        )[0]
        save_png(
            output / "od.png",
            detection_visualization(rgb, detection, OLDTOWN_OD_CLASS_NAMES),
        )
        boxes = {key: value.tolist() for key, value in detection.items()}
        (output / "detections.json").write_text(
            json.dumps(boxes, indent=2) + "\n", encoding="utf-8"
        )
        outputs.extend(["od.png", "detections.json"])
    manifest = dict(
        model="original_trivnet",
        dataset="oldtown",
        run_dir=str(args.run_dir),
        image=str(args.image),
        height=height,
        width=width,
        outputs=outputs,
        segmentation_alpha=args.seg_alpha,
        depth_range_m=[0, 80],
        od_score_threshold=args.score_threshold,
        od_nms_threshold=args.nms_threshold,
        od_topk=args.topk,
        box_coordinates="xyxy in the resized input image",
    )
    (output / "prediction.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print("[saved] {}".format(output))


def main():
    parser = argparse.ArgumentParser(description="Run single-image TriV-Net inference.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory containing config.yaml and best.pth",
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="A NEW output directory"
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seg-alpha", type=float, default=1.0)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=100)
    predict(parser.parse_args())


if __name__ == "__main__":
    main()
