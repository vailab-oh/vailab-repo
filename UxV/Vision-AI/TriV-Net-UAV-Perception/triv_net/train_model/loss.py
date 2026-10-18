import torch
import torch.nn as nn


class silog_loss(nn.Module):
    def __init__(self, variance_focus=0.85):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        depth_est = depth_est.squeeze()
        depth_gt = depth_gt.squeeze()
        mask = mask.squeeze().bool()

        depth_est_valid = depth_est[mask]
        depth_gt_valid = depth_gt[mask]

        d = torch.log(torch.clamp(depth_est_valid, min=1e-4)) - torch.log(
            torch.clamp(depth_gt_valid, min=1e-4)
        )

        mse_log = torch.mean(d**2)
        mean_log = torch.mean(d)

        return torch.sqrt(mse_log - self.variance_focus * mean_log**2) * 10.0


def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(
        torch.unsqueeze(a[:, 0], 1), b[:, 0]
    )
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(
        torch.unsqueeze(a[:, 1], 1), b[:, 1]
    )

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = (
        torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1)
        + area
        - iw * ih
    )

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU
