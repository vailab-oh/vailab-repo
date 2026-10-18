from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from train_model.loss_balancing import LossBalancer
from train import run_epoch, validate_config


CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs/original_trivnet/oldtown_original_trivnet_mtl.yaml"
)


def configuration():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_mtl_uses_paper_weights_without_normalization_and_with_correct_gradients():
    cfg = configuration()
    validate_config(cfg)
    assert cfg["loss_balance"] == {"mode": "static"}
    assert cfg["loss_lambda"] == {"od": 30.0, "seg": 1.0, "depth": 0.3}
    balancer = LossBalancer(cfg, "cpu")
    losses = {
        "od": torch.tensor(2.0, requires_grad=True),
        "seg": torch.tensor(3.0, requires_grad=True),
        "depth": torch.tensor(4.0, requires_grad=True),
    }
    total, weights = balancer.combine(losses)
    assert total.item() == pytest.approx(64.2)
    assert weights == {"od": 30.0, "seg": 1.0, "depth": 0.3}
    total.backward()
    for task, expected in weights.items():
        assert losses[task].grad.item() == pytest.approx(expected)
    assert not balancer.fixed_weights.requires_grad
    assert balancer.summary()["normalized"] is False
    assert balancer.summary()["learned"] is False
    assert balancer.combine(losses)[1] == weights


@pytest.mark.parametrize("mode", ["gradnorm", "initial_loss"])
def test_adaptive_weighting_is_rejected(mode):
    cfg = configuration()
    cfg["loss_balance"]["mode"] = mode
    with pytest.raises(ValueError, match="fixed task weights"):
        validate_config(cfg)


def test_wrong_or_normalized_mtl_weights_are_rejected():
    cfg = configuration()
    for incorrect in (
        {"od": 1.0, "seg": 1.0, "depth": 1.0},
        {"od": 30.0 / 31.3, "seg": 1.0 / 31.3, "depth": 0.3 / 31.3},
    ):
        changed = deepcopy(cfg)
        changed["loss_lambda"] = incorrect
        with pytest.raises(ValueError, match="Expected fixed"):
            validate_config(changed)


@pytest.mark.parametrize(
    "task,flag", [("od", "od"), ("seg", "Seg"), ("depth", "Depth")]
)
def test_stl_preserves_unit_weight(task, flag):
    cfg = configuration()
    cfg["model_conf"] = {k: k == flag for k in ("od", "Seg", "Depth")}
    cfg["loss_lambda"] = {"od": 1.0, "seg": 1.0, "depth": 1.0}
    total, weights = LossBalancer(cfg, "cpu").combine({task: torch.tensor(7.0)})
    assert total.item() == 7.0
    assert weights == {task: 1.0}


def test_epoch_logging_uses_the_same_fixed_total_loss():
    class ToyModel(torch.nn.Module):
        def forward(self, images):
            b, _, h, w = images.shape
            return (
                torch.zeros(b, 12, h, w),
                torch.full((b, 1, h, w), 4.0),
                torch.tensor(3.0),
                None,
            )

    class DepthLoss:
        def __call__(self, prediction, gt, mask):
            return prediction.mean()

    class ODLoss:
        def __call__(self, model, images, classification, regression, annotations):
            return classification, torch.tensor(0.0)

    cfg = configuration()
    cfg.update(disable_tqdm=True, eval_od_metrics=False)
    images = torch.ones(2, 3, 8, 8)
    batch = (
        images,
        images,
        torch.zeros(2, 8, 8, dtype=torch.long),
        torch.ones(2, 1, 8, 8),
        torch.ones(2, 1, 8, 8, dtype=torch.bool),
        torch.zeros(2, 0, 5),
    )
    stats = run_epoch(
        ToyModel(),
        [batch],
        cfg,
        torch.device("cpu"),
        DepthLoss(),
        ODLoss(),
        LossBalancer(cfg, "cpu"),
    )
    expected = 30 * stats["od_loss"] + stats["seg_loss"] + 0.3 * stats["depth_loss"]
    assert stats["loss"] == pytest.approx(expected)
    assert stats["loss_weight_od"] == 30.0
    assert stats["loss_weight_seg"] == 1.0
    assert stats["loss_weight_depth"] == 0.3
