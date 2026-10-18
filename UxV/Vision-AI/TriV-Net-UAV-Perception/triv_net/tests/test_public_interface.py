import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
import yaml

import train
from predict import predict
from train_model.original_trivnet_training import build_original_trivnet
from visualization import OLDTOWN_OD_CLASS_NAMES, class_color


ROOT = Path(__file__).resolve().parents[1]


def test_od_labels_and_colors():
    assert OLDTOWN_OD_CLASS_NAMES == (
        "Street Light",
        "Traffic Light",
        "Road Sign",
        "Guard Rail",
        "Waste Bin",
    )
    assert class_color(0) == (75, 105, 135)
    assert class_color(1) == (142, 236, 108)


def test_epoch_summary_omits_disabled_tasks():
    cfg = {"model_conf": {"od": True, "Seg": False, "Depth": False}}
    message = train.epoch_summary(
        1, 50, cfg, {"loss": 3.0}, {"loss": 2.0, "od_map50": 0.4, "od_map50_95": 0.2}
    )
    assert "Epoch 1/50" in message
    assert "AP50=0.4000" in message
    assert "mIoU" not in message
    assert "AbsRel" not in message


def test_unrelated_model_variants_are_rejected():
    config_path = ROOT / "configs/original_trivnet/oldtown_original_trivnet_mtl.yaml"
    cfg = yaml.safe_load(config_path.read_text())
    for field, value in (("encoder", "EfficientNet"), ("decoder", "Bi-FPN")):
        changed = deepcopy(cfg)
        changed["model"][field] = value
        with pytest.raises(ValueError):
            build_original_trivnet(changed)


def test_quiet_cpu_training_and_prediction(tmp_path, monkeypatch, capsys):
    cfg = yaml.safe_load(
        (
            ROOT / "configs/original_trivnet/oldtown_original_trivnet_mtl.yaml"
        ).read_text()
    )
    data_root = tmp_path / "oldtown"
    for split in ("train", "val"):
        scene = data_root / split / "scene"
        for directory in ("image_left", "seg_train_id", "depth_left", "annotiation"):
            (scene / directory).mkdir(parents=True)
        for index in range(2):
            stem = "{:06d}_left".format(index)
            cv2.imwrite(
                str(scene / "image_left" / (stem + ".png")),
                np.full((480, 640, 3), 100 + index, dtype=np.uint8),
            )
            cv2.imwrite(
                str(scene / "seg_train_id" / (stem + "_train_id.png")),
                np.zeros((480, 640), dtype=np.uint8),
            )
            np.save(
                str(scene / "depth_left" / (stem + "_depth.npy")),
                np.full((480, 640), 10.0, dtype=np.float32),
            )
            (scene / "annotiation" / (stem + "_det.json")).write_text(
                json.dumps([{"class_id": 0, "bbox": [200, 150, 100, 80]}])
            )

    cfg.update(
        dataset_root=str(data_root),
        output_root=str(tmp_path / "runs"),
        height=128,
        width=256,
        epochs=1,
        batch_size=2,
        workers=0,
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))

    def offline_model(config):
        construction = deepcopy(config)
        construction["model"]["pretrained"] = False
        return build_original_trivnet(construction)

    monkeypatch.setattr(train, "build_original_trivnet", offline_model)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        sys, "argv", ["train.py", "--config", str(config_path), "--quiet"]
    )
    train.main()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""

    run_dir = next((tmp_path / "runs").iterdir())
    assert {path.name for path in run_dir.iterdir()} == {
        "best.pth",
        "last.pth",
        "config.yaml",
        "metrics.yaml",
        "losses.yaml",
        "model_info.yaml",
    }
    losses = yaml.safe_load((run_dir / "losses.yaml").read_text())[0]
    for split in ("train", "val"):
        expected = (
            30 * losses[split + "_od_loss"]
            + losses[split + "_seg_loss"]
            + 0.3 * losses[split + "_depth_loss"]
        )
        assert losses[split + "_loss"] == pytest.approx(expected)

    output = tmp_path / "prediction"
    predict(
        SimpleNamespace(
            run_dir=run_dir,
            image=data_root / "val/scene/image_left/000000_left.png",
            output_dir=output,
            device="cpu",
            seg_alpha=1.0,
            score_threshold=0.25,
            nms_threshold=0.5,
            topk=5,
        )
    )
    assert {path.name for path in output.iterdir()} == {
        "input.png",
        "seg.png",
        "seg_train_id.png",
        "depth.png",
        "depth_m.npy",
        "od.png",
        "detections.json",
        "prediction.json",
    }
    assert not list(tmp_path.rglob("anchor_*.npy"))
