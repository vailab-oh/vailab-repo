import torch

from train_model.loss_balancing import LossBalancer


def test_single_task_static_weight_is_one():
    cfg = {
        "model_conf": {"Seg": True, "Depth": False, "od": False},
        "loss_balance": {"mode": "static"},
        "loss_lambda": {"seg": 1.0},
    }
    balancer = LossBalancer(cfg, torch.device("cpu"))
    total, weights = balancer.combine({"seg": torch.tensor(2.0)})

    assert total.item() == 2.0
    assert weights == {"seg": 1.0}


def test_gradnorm_keeps_positive_weights_normalized():
    cfg = {
        "model_conf": {"Seg": True, "Depth": True, "od": True},
        "loss_balance": {
            "mode": "gradnorm",
            "gradnorm_alpha": 1.5,
            "gradnorm_lr": 0.025,
        },
    }
    shared = torch.nn.Parameter(torch.tensor(1.0))
    losses = {
        "seg": shared.square(),
        "depth": (2.0 * shared).square(),
        "od": (4.0 * shared).square(),
    }
    initial = {name: float(value.detach()) for name, value in losses.items()}
    balancer = LossBalancer(cfg, torch.device("cpu"), initial_losses=initial)

    stats = balancer.gradnorm_step(losses, (shared,))
    weights = balancer.current_weights()

    assert stats is not None
    assert torch.all(weights > 0)
    torch.testing.assert_close(weights.sum(), torch.tensor(3.0))
