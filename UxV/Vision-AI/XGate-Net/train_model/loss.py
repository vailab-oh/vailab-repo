import torch
import torch.nn.functional as F
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceBCELossMultiClass(nn.Module):
    def __init__(self, smooth=1.0, bce_weight=1.0, dice_weight=1.0, pos_weight=None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.smooth = smooth
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets_1h):
        bce = self.bce(logits, targets_1h)

        probs = torch.sigmoid(logits)
        dims = (0, 2, 3)
        inter = (probs * targets_1h).sum(dims)
        denom = probs.sum(dims) + targets_1h.sum(dims)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        dice_loss = 1 - dice.mean()

        return self.bce_weight * bce + self.dice_weight * dice_loss

class Dice_BEC_Loss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Dice_BEC_Loss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(weight=weight)

    def forward(self, inputs, targets, smooth=1):

        bce_loss = self.bce(inputs, targets)
        inputs = torch.sigmoid(inputs)

        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return (1 - dice)+bce_loss
    
class CrossEntropyFromRGBLoss(nn.Module):
    def __init__(self, class_colors: dict, ignore_index=255, assume_bgr=True):
        super().__init__()
        self.ignore_index = ignore_index
        self.assume_bgr = assume_bgr
        items = sorted(class_colors.items(), key=lambda x: x[0])
        self.register_buffer("palette", torch.tensor([c for _, c in items], dtype=torch.uint8))

    def _rgb_to_index(self, seg_rgb: torch.Tensor) -> torch.Tensor:
        N, _, H, W = seg_rgb.shape
        rgb = (seg_rgb * 255.0).round().to(torch.uint8)
        pal = self.palette
        if self.assume_bgr:
            pal = pal[:, [2,1,0]]
        rgb_flat = rgb.permute(0,2,3,1).reshape(N, H*W, 3)
        pal_exp  = pal.view(1, pal.size(0), 1, 3)
        rgb_exp  = rgb_flat.unsqueeze(1)
        eq = (rgb_exp == pal_exp).all(dim=-1)
        matched, cls = eq.max(dim=1)
        labels = torch.full((N, H*W), self.ignore_index, device=seg_rgb.device, dtype=torch.long)
        labels[matched] = cls[matched]
        return labels.view(N, H, W)

    def forward(self, logits: torch.Tensor, targets_rgb: torch.Tensor):
        labels = self._rgb_to_index(targets_rgb)
        return F.cross_entropy(logits, labels, ignore_index=self.ignore_index)
    
class SegCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=-100):
        super(SegCrossEntropyLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, inputs, targets):
        return self.ce(inputs, targets)
    
class silog_loss(nn.Module):
    def __init__(self, variance_focus=0.85):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        depth_est = depth_est.squeeze()
        depth_gt  = depth_gt.squeeze()
        mask = mask.squeeze().bool()

        valid = mask & (depth_gt > 0)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=depth_gt.device)

        depth_est_v = depth_est[valid]
        depth_gt_v  = depth_gt[valid]

        d = torch.log(torch.clamp(depth_est_v, min=1e-4)) - \
            torch.log(torch.clamp(depth_gt_v , min=1e-4))

        mse_log  = torch.mean(d ** 2)
        mean_log = torch.mean(d)
        return torch.sqrt(mse_log - self.variance_focus * mean_log**2) * 10.0


def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU

class FocalLoss(nn.Module):

    def forward(self, classifications, regressions, anchors, annotations):
        xyxycls_B0 = None
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []
        device = classifications.device
        anchors = anchors.to(device)
        annotations = annotations.to(device)
        anchor = anchors[0, :, :]

        anchor_widths  = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights
        iou_scores = []
        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():
                    alpha_factor = torch.ones(classification.shape).cuda() * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(regression.sum() * 0.0)
                    iou_scores.append(classification.sum() * 0.0)

                else:
                    alpha_factor = torch.ones(classification.shape) * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(regression.sum() * 0.0)
                    iou_scores.append(classification.sum() * 0.0)
                continue

            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4])

            IoU_max, IoU_argmax = torch.max(IoU, dim=1)
            positive_indices = torch.ge(IoU_max, 0.5)
            if positive_indices.sum() > 0:
                mean_iou = IoU_max[positive_indices].mean()
            else:
                mean_iou = torch.tensor(0.0).cuda() if torch.cuda.is_available() else torch.tensor(0.0)
            iou_scores.append(mean_iou)


            targets = torch.ones(classification.shape) * -1

            if torch.cuda.is_available():
                targets = targets.cuda()

            targets[torch.lt(IoU_max, 0.4), :] = 0

            positive_indices = torch.ge(IoU_max, 0.5)

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            if torch.cuda.is_available():
                alpha_factor = torch.ones(targets.shape).cuda() * alpha
            else:
                alpha_factor = torch.ones(targets.shape) * alpha

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            cls_loss = focal_weight * bce

            if torch.cuda.is_available():
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
            else:
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))

            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))


            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths  = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y   = assigned_annotations[:, 1] + 0.5 * gt_heights

                gt_widths  = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                if torch.cuda.is_available():
                    targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
                else:
                    targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]])

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                if torch.cuda.is_available():
                    regression_losses.append(regression.sum() * 0.0)
                else:
                    regression_losses.append(regression.sum() * 0.0)
            if j == 0:
                if positive_indices.sum() > 0:
                    pos_anchors = anchor[positive_indices]
                    assigned_annotations_all = bbox_annotation[IoU_argmax, :]
                    pos_assigned = assigned_annotations_all[positive_indices]

                    pos_classes = pos_assigned[:, 4].long().unsqueeze(1)

                    pos_scores = classification[positive_indices, :]
                    pos_scores = pos_scores[torch.arange(pos_scores.size(0)), pos_classes.squeeze(1)]
                    pos_scores_unsqueezed = pos_scores.unsqueeze(1)

                    xyxycls_B0 = torch.cat([pos_anchors, pos_classes.float(), pos_scores_unsqueezed], dim=1)
                else:
                    xyxycls_B0 = None

        if len(classification_losses) == 0 or len(regression_losses) == 0 or len(iou_scores) == 0:
            zero = classifications.sum() * 0.0
            return (zero, zero, torch.zeros(1, device=classifications.device), None)
        
        return torch.stack(classification_losses).mean(dim=0, keepdim=True), \
            torch.stack(regression_losses).mean(dim=0, keepdim=True), \
            torch.stack(iou_scores).mean(dim=0, keepdim=True), \
            xyxycls_B0
