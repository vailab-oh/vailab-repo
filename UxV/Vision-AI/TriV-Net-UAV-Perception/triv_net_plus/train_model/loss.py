"""Loss functions used by the final TriV-Net+ training pipeline."""

import torch
import torch.nn as nn


class silog_loss(nn.Module):
    """Scale-invariant logarithmic depth loss."""

    def __init__(self, variance_focus=0.85):
        super().__init__()
        self.variance_focus = float(variance_focus)

    def forward(self, depth_est, depth_gt, mask):
        valid = mask.bool()
        estimated = depth_est[valid].clamp(min=1e-4)
        target = depth_gt[valid].clamp(min=1e-4)
        if estimated.numel() == 0:
            return depth_est.sum() * 0.0

        difference = torch.log(estimated) - torch.log(target)
        value = difference.square().mean() - self.variance_focus * difference.mean().square()
        return torch.sqrt(value.clamp(min=0.0)) * 10.0
