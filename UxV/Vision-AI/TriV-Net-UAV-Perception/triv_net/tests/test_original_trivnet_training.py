from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_model.original_trivnet_training import (
    OriginalTriVNetODLoss,
    build_original_trivnet,
    decode_original_detections,
)
from train import validate_config


CONFIG_DIR = ROOT / "configs" / "original_trivnet"
MTL_CONFIG = CONFIG_DIR / "oldtown_original_trivnet_mtl.yaml"

EXPECTED_TASK_FLAGS = {
    "oldtown_original_trivnet_seg.yaml": {"Seg": True, "Depth": False, "od": False},
    "oldtown_original_trivnet_depth.yaml": {"Seg": False, "Depth": True, "od": False},
    "oldtown_original_trivnet_od.yaml": {"Seg": False, "Depth": False, "od": True},
    "oldtown_original_trivnet_mtl.yaml": {
        "Seg": True,
        "Depth": True,
        "od": True,
    },
}
CHECKPOINT_INIT_KEYS = {"resume", "init_checkpoint", "checkpoint_path", "weights"}


def _load_yaml(path):
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _mapping_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _mapping_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _mapping_keys(child)


@pytest.fixture(scope="module")
def original_mtl_forward():
    cfg = deepcopy(_load_yaml(MTL_CONFIG))
    cfg["model"]["pretrained"] = False

    torch.manual_seed(0)
    model = build_original_trivnet(cfg).cpu().eval()
    images = torch.randn(1, 3, 128, 256)
    with torch.no_grad():
        outputs = model(images)
    return model, images, outputs


def test_oldtown_configs_select_original_trivnet_without_checkpoint_init():
    config_paths = sorted(CONFIG_DIR.glob("oldtown_original_trivnet_*.yaml"))
    assert {path.name for path in config_paths} == set(EXPECTED_TASK_FLAGS)

    for path in config_paths:
        cfg = _load_yaml(path)
        assert cfg["dataset_root"] == "data/oldtown"
        assert cfg["model"]["architecture_family"] == "original_trivnet"
        assert cfg["seg_num_classes"] == 12
        assert cfg["model_conf"] == EXPECTED_TASK_FLAGS[path.name]
        validate_config(cfg)

        present_init_keys = CHECKPOINT_INIT_KEYS.intersection(_mapping_keys(cfg))
        assert not present_init_keys, (
            f"{path.name} unexpectedly initializes training from: "
            f"{sorted(present_init_keys)}"
        )


def test_original_trivnet_full_mtl_forward_shapes(original_mtl_forward):
    model, images, outputs = original_mtl_forward
    segmentation, depth, classification, regression = outputs

    assert model.cfg["model"]["pretrained"] is False
    assert images.shape == (1, 3, 128, 256)
    assert segmentation.shape == (1, 12, 128, 256)
    assert depth.shape == (1, 1, 128, 256)
    assert classification.shape == (1, 6138, 5)
    assert regression.shape == (1, 6138, 4)


def test_original_od_loss_is_finite_and_supports_backward(original_mtl_forward):
    model, images, (_, _, classification_output, regression_output) = (
        original_mtl_forward
    )
    classification = classification_output.detach().clone().requires_grad_(True)
    regression = regression_output.detach().clone().requires_grad_(True)

    anchors = model.anchors(images, dtype=regression.dtype)[0]
    height, width = images.shape[-2:]
    inside = (
        (anchors[:, 0] >= 0)
        & (anchors[:, 1] >= 0)
        & (anchors[:, 2] <= height)
        & (anchors[:, 3] <= width)
    )
    anchor = anchors[torch.where(inside)[0][0]]
    annotations = torch.tensor(
        [[[*anchor[[1, 0, 3, 2]].tolist(), 2.0]]],
        dtype=regression.dtype,
    )

    loss, mean_iou = OriginalTriVNetODLoss()(
        model, images, classification, regression, annotations
    )
    assert loss.ndim == 0
    assert mean_iou.ndim == 0
    assert torch.isfinite(loss)
    assert torch.isfinite(mean_iou)
    assert mean_iou >= 0.5

    loss.backward()
    assert classification.grad is not None
    assert regression.grad is not None
    assert torch.isfinite(classification.grad).all()
    assert torch.isfinite(regression.grad).all()


def test_original_detection_decode_honors_topk_and_image_bounds(original_mtl_forward):
    model, images, (_, _, classification, regression) = original_mtl_forward
    topk = 7
    detections = decode_original_detections(
        model,
        images,
        classification,
        regression,
        score_thresh=0.0,
        nms_thresh=0.5,
        topk=topk,
    )

    assert len(detections) == images.shape[0]
    rois = detections[0]["rois"]
    scores = detections[0]["scores"]
    class_ids = detections[0]["class_ids"]
    assert 0 < len(rois) <= topk
    assert rois.shape == (len(rois), 4)
    assert scores.shape == (len(rois),)
    assert class_ids.shape == (len(rois),)
    assert np.isfinite(rois).all()
    assert np.isfinite(scores).all()

    height, width = images.shape[-2:]
    assert (rois[:, 0] >= 0).all()
    assert (rois[:, 1] >= 0).all()
    assert (rois[:, 2] <= width - 1).all()
    assert (rois[:, 3] <= height - 1).all()
    assert (rois[:, 2] > rois[:, 0]).all()
    assert (rois[:, 3] > rois[:, 1]).all()
    assert ((class_ids >= 0) & (class_ids < 5)).all()
    assert (scores[:-1] >= scores[1:]).all()
