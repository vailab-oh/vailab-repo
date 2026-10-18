import math

import torch


TASK_NAMES = ("seg", "depth", "od")
PAPER_MTL_WEIGHTS = {"seg": 1.0, "depth": 0.3, "od": 30.0}


def active_task_names(cfg):
    conf = cfg.get("model_conf", {})
    enabled = {
        "seg": conf.get("Seg", False),
        "depth": conf.get("Depth", False),
        "od": conf.get("od", False),
    }
    return tuple(name for name in TASK_NAMES if enabled[name])


def validate_fixed_weights(cfg):
    tasks = active_task_names(cfg)
    if not tasks:
        raise ValueError("At least one task must be enabled.")
    if cfg.get("loss_balance", {}).get("mode", "static") != "static":
        raise ValueError(
            "Original TriV-Net uses fixed task weights, not adaptive weighting."
        )
    expected = PAPER_MTL_WEIGHTS if len(tasks) > 1 else {tasks[0]: 1.0}
    configured = cfg.get("loss_lambda", {})
    weights = {task: float(configured.get(task, expected[task])) for task in tasks}
    for task, value in weights.items():
        if not math.isfinite(value) or not math.isclose(
            value, expected[task], rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError(
                "Expected fixed {} weight {}, got {}. MTL uses OD=30, SS=1, DE=0.3; "
                "STL uses weight 1.".format(task, expected[task], value)
            )
    return tasks, weights


class LossBalancer:

    def __init__(self, cfg, device):
        self.tasks, self.weights = validate_fixed_weights(cfg)
        self.mode = "static"
        self.fixed_weights = torch.tensor(
            [self.weights[t] for t in self.tasks], dtype=torch.float32, device=device
        )

    def current_weight_dict(self):
        return dict(self.weights)

    def combine(self, task_losses):
        total = sum(
            self.fixed_weights[i] * task_losses[task]
            for i, task in enumerate(self.tasks)
        )
        return total, self.current_weight_dict()

    def summary(self):
        return {
            "mode": self.mode,
            "tasks": list(self.tasks),
            "weights": self.current_weight_dict(),
            "normalized": False,
            "learned": False,
        }
