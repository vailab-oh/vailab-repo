import torch


TASK_NAMES = ("seg", "depth", "od")


def active_task_names(cfg):
    conf = cfg.get("model_conf", {})
    enabled = {
        "seg": bool(conf.get("Seg", False)),
        "depth": bool(conf.get("Depth", False)),
        "od": bool(conf.get("od", False)),
    }
    return tuple(name for name in TASK_NAMES if enabled[name])


def normalized_positive_weights(values, total):
    weights = torch.as_tensor(values, dtype=torch.float32)
    if not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Loss weights must be finite and greater than zero.")
    return weights * (float(total) / weights.sum())


class LossBalancer:
    """Combines task losses using static, initial-loss, or GradNorm weights."""

    def __init__(self, cfg, device, initial_losses=None):
        self.device = device
        self.tasks = active_task_names(cfg)
        if not self.tasks:
            raise ValueError("At least one task must be enabled.")

        balance_cfg = cfg.get("loss_balance", {})
        self.mode = str(balance_cfg.get("mode", "static")).lower()
        if self.mode not in ("static", "initial_loss", "gradnorm"):
            raise ValueError("Unsupported loss_balance.mode: {}".format(self.mode))

        self.initial_losses = None
        if initial_losses is not None:
            self.initial_losses = {
                task: float(initial_losses[task])
                for task in self.tasks
            }
            if any(value <= 0 for value in self.initial_losses.values()):
                raise ValueError("Initial task losses must be greater than zero.")

        self.alpha = float(balance_cfg.get("gradnorm_alpha", 1.5))
        self.gradnorm_lr = float(balance_cfg.get("gradnorm_lr", 0.025))
        self.weight_optimizer = None

        if self.mode == "static":
            configured = cfg.get("loss_lambda", {})
            values = [float(configured.get(task, 1.0)) for task in self.tasks]
            self.fixed_weights = torch.tensor(values, dtype=torch.float32, device=device)
            if not torch.isfinite(self.fixed_weights).all() or (self.fixed_weights <= 0).any():
                raise ValueError("Static loss weights must be finite and greater than zero.")
            self.task_weights = None
        elif self.mode == "initial_loss":
            if self.initial_losses is None:
                raise ValueError("initial_loss mode requires calibrated initial losses.")
            inverse = [1.0 / self.initial_losses[task] for task in self.tasks]
            self.fixed_weights = normalized_positive_weights(inverse, len(self.tasks)).to(device)
            self.task_weights = None
        else:
            if self.initial_losses is None:
                raise ValueError("gradnorm mode requires calibrated initial losses.")
            self.fixed_weights = None
            self.task_weights = torch.nn.Parameter(
                torch.ones(len(self.tasks), dtype=torch.float32, device=device)
            )
            self.weight_optimizer = torch.optim.Adam([self.task_weights], lr=self.gradnorm_lr)

    def current_weights(self, detach=True):
        weights = self.task_weights if self.mode == "gradnorm" else self.fixed_weights
        return weights.detach().clone() if detach else weights

    def current_weight_dict(self):
        values = self.current_weights(detach=True).cpu().tolist()
        return {task: float(value) for task, value in zip(self.tasks, values)}

    def combine(self, task_losses):
        weights = self.current_weights(detach=True)
        total = None
        for index, task in enumerate(self.tasks):
            weighted = weights[index] * task_losses[task]
            total = weighted if total is None else total + weighted
        return total, self.current_weight_dict()

    def gradnorm_step(self, task_losses, shared_parameters):
        if self.mode != "gradnorm":
            return None

        parameters = tuple(parameter for parameter in shared_parameters if parameter.requires_grad)
        if not parameters:
            raise ValueError("GradNorm requires trainable shared parameters.")

        raw_norms = []
        for task in self.tasks:
            gradients = torch.autograd.grad(
                task_losses[task],
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
            squared_norm = None
            for gradient in gradients:
                if gradient is None:
                    continue
                value = gradient.detach().float().pow(2).sum()
                squared_norm = value if squared_norm is None else squared_norm + value
            if squared_norm is None:
                raise RuntimeError("Task {} has no gradient on the GradNorm shared layer.".format(task))
            raw_norms.append(torch.sqrt(squared_norm.clamp(min=1e-12)))

        raw_norms = torch.stack(raw_norms)
        weighted_norms = self.task_weights * raw_norms
        with torch.no_grad():
            loss_ratios = torch.tensor(
                [
                    float(task_losses[task].detach().item()) / self.initial_losses[task]
                    for task in self.tasks
                ],
                dtype=torch.float32,
                device=self.device,
            )
            relative_rates = loss_ratios / loss_ratios.mean().clamp(min=1e-12)
            targets = weighted_norms.detach().mean() * relative_rates.pow(self.alpha)

        gradnorm_loss = torch.abs(weighted_norms - targets).sum()
        self.weight_optimizer.zero_grad()
        gradnorm_loss.backward()
        self.weight_optimizer.step()

        with torch.no_grad():
            self.task_weights.clamp_(min=1e-3)
            self.task_weights.mul_(len(self.tasks) / self.task_weights.sum().clamp(min=1e-12))

        return {
            "gradnorm_loss": float(gradnorm_loss.detach().item()),
            "raw_gradient_norms": {
                task: float(value)
                for task, value in zip(self.tasks, raw_norms.detach().cpu().tolist())
            },
        }

    def summary(self):
        return {
            "mode": self.mode,
            "tasks": list(self.tasks),
            "initial_losses": self.initial_losses,
            "weights": self.current_weight_dict(),
            "gradnorm_alpha": self.alpha if self.mode == "gradnorm" else None,
            "gradnorm_lr": self.gradnorm_lr if self.mode == "gradnorm" else None,
        }
