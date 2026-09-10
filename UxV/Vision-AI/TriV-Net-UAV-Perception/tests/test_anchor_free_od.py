from pathlib import Path
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_model.anchor_free_od import AnchorFreeODLoss, decode_anchor_free
from train_model.mtl_models import ODHead


def _outputs_for_box(box, stride=8, image_size=(64, 64)):
    image_h, image_w = image_size
    feature_h = image_h // stride
    feature_w = image_w // stride
    cx = (box[0] + box[2]) * 0.5
    cy = (box[1] + box[3]) * 0.5
    gx = int(cx / stride)
    gy = int(cy / stride)
    qx = (gx + 0.5) * stride
    qy = (gy + 0.5) * stride

    logits = torch.full((1, 1, feature_h, feature_w), -80.0)
    logits[0, 0, gy, gx] = 80.0
    box_ltrb = torch.zeros((1, 4, feature_h, feature_w))
    box_ltrb[0, :, gy, gx] = torch.tensor(
        [
            (qx - box[0]) / stride,
            (qy - box[1]) / stride,
            (box[2] - qx) / stride,
            (box[3] - qy) / stride,
        ]
    )
    return {
        "cls_logits": [logits],
        "box_ltrb": [box_ltrb],
        "strides": [stride],
    }


def _assert_perfect_round_trip(box):
    outputs = _outputs_for_box(box)
    annotations = torch.tensor([[box + [0.0]]])

    loss, od_iou = AnchorFreeODLoss(num_classes=1, box_weight=1.0)(
        outputs,
        annotations,
        image_size=(64, 64),
    )

    assert loss.item() < 1e-6
    torch.testing.assert_close(od_iou, torch.ones_like(od_iou), atol=1e-6, rtol=0)

    detections = decode_anchor_free(
        outputs,
        image_size=(64, 64),
        score_thresh=0.5,
        topk=10,
    )[0]
    assert len(detections["rois"]) == 1
    np.testing.assert_allclose(detections["rois"][0], box, atol=1e-6)
    np.testing.assert_array_equal(detections["class_ids"], [0])


def test_anchor_free_ltrb_round_trip_uses_grid_center():
    # GT center=(22, 27), assigned cell center q=(20, 28).
    _assert_perfect_round_trip([9.0, 13.0, 35.0, 41.0])


def test_anchor_free_ltrb_supports_small_box_outside_grid_center():
    # The positive cell center q=(12, 12) lies to the right of this narrow box.
    # Its right offset is negative and must remain representable.
    _assert_perfect_round_trip([9.0, 9.0, 11.0, 15.0])


def test_od_head_does_not_clamp_signed_ltrb_offsets():
    head = ODHead(channels=4, num_classes=1).eval()
    final_box_conv = head.box_heads[0][-1]
    with torch.no_grad():
        final_box_conv.weight.zero_()
        final_box_conv.bias.fill_(-1.0)

    pyramid = [torch.zeros((1, 4, 1, 1)) for _ in range(5)]
    outputs = head(pyramid)

    assert torch.all(outputs["box_ltrb"][0] < 0)


def test_od_head_rejects_legacy_box_encoding_checkpoint():
    head = ODHead(channels=4, num_classes=1)
    legacy_state = dict(head.state_dict())
    legacy_state.pop("_box_encoding_version")

    with pytest.raises(RuntimeError, match="_box_encoding_version"):
        head.load_state_dict(legacy_state, strict=True)


def test_decode_anchor_free_discards_invalid_boxes_before_nms():
    logits = torch.tensor([[[[20.0, 19.0]]]])
    box_ltrb = torch.zeros((1, 4, 1, 2))
    box_ltrb[0, :, 0, 0] = torch.tensor([0.5, 0.5, 0.5, 0.5])
    box_ltrb[0, :, 0, 1] = torch.tensor([-1.0, 0.5, -1.0, 0.5])
    outputs = {
        "cls_logits": [logits],
        "box_ltrb": [box_ltrb],
        "strides": [8],
    }

    detections = decode_anchor_free(
        outputs,
        image_size=(16, 16),
        score_thresh=0.5,
        topk=1,
    )[0]

    assert len(detections["rois"]) == 1
    np.testing.assert_allclose(detections["rois"][0], [0.0, 0.0, 8.0, 8.0], atol=1e-6)


def test_decode_anchor_free_discards_non_finite_offsets():
    outputs = {
        "cls_logits": [torch.tensor([[[[20.0]]]])],
        "box_ltrb": [torch.full((1, 4, 1, 1), float("inf"))],
        "strides": [8],
    }

    detections = decode_anchor_free(
        outputs,
        image_size=(16, 16),
        score_thresh=0.5,
        topk=1,
    )[0]

    assert detections == {"rois": [], "scores": [], "class_ids": []}
