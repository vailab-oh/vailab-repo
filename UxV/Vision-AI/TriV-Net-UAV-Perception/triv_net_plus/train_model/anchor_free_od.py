import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms


def _box_iou_aligned(boxes1, boxes2):
    x1 = torch.maximum(boxes1[:, 0], boxes2[:, 0])
    y1 = torch.maximum(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.minimum(boxes1[:, 2], boxes2[:, 2])
    y2 = torch.minimum(boxes1[:, 3], boxes2[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    return inter / (area1 + area2 - inter).clamp(min=1e-6)


def _decode_level_boxes(ltrb, stride, image_size):
    image_h, image_w = image_size
    _, h, w = ltrb.shape
    xs = (torch.arange(w, device=ltrb.device, dtype=ltrb.dtype) + 0.5) * stride
    ys = (torch.arange(h, device=ltrb.device, dtype=ltrb.dtype) + 0.5) * stride
    boxes = torch.stack(
        [
            xs.unsqueeze(0) - ltrb[0] * stride,
            ys.unsqueeze(1) - ltrb[1] * stride,
            xs.unsqueeze(0) + ltrb[2] * stride,
            ys.unsqueeze(1) + ltrb[3] * stride,
        ],
        dim=-1,
    )
    finite = torch.isfinite(boxes).all(dim=-1)
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, image_w - 1)
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, image_h - 1)
    valid = finite & (boxes[..., 2] > boxes[..., 0]) & (boxes[..., 3] > boxes[..., 1])
    return boxes, valid


class AnchorFreeODLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.25, gamma=2.0, box_weight=5.0):
        super().__init__()
        self.num_classes = int(num_classes)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.box_weight = float(box_weight)

    def _level_for_box(self, width, height, num_levels):
        max_side = max(float(width), float(height))
        if max_side <= 64:
            return 0
        if max_side <= 128:
            return min(1, num_levels - 1)
        if max_side <= 256:
            return min(2, num_levels - 1)
        if max_side <= 512:
            return min(3, num_levels - 1)
        return num_levels - 1

    def forward(self, outputs, annotations, image_size):
        cls_logits = outputs["cls_logits"]
        box_ltrb = outputs["box_ltrb"]
        strides = outputs["strides"]
        device = cls_logits[0].device
        batch_size = cls_logits[0].shape[0]
        total_cls_loss = cls_logits[0].sum() * 0.0
        pred_boxes = []
        target_boxes = []
        positive_count = 0

        for level, (logits, pred_ltrb, stride) in enumerate(zip(cls_logits, box_ltrb, strides)):
            bsz, _, h, w = logits.shape
            cls_target = torch.zeros_like(logits)
            obj_mask = torch.zeros((bsz, 1, h, w), dtype=torch.bool, device=device)
            box_target = torch.zeros_like(pred_ltrb)

            for b in range(batch_size):
                valid = annotations[b]
                valid = valid[valid[:, 4] >= 0]
                for gt in valid:
                    x1, y1, x2, y2, cls = gt.tolist()
                    if x2 <= x1 or y2 <= y1:
                        continue
                    target_level = self._level_for_box(x2 - x1, y2 - y1, len(cls_logits))
                    if target_level != level:
                        continue
                    cx = (x1 + x2) * 0.5
                    cy = (y1 + y2) * 0.5
                    gx = int(cx / stride)
                    gy = int(cy / stride)
                    if gx < 0 or gx >= w or gy < 0 or gy >= h:
                        continue
                    cls_idx = int(cls)
                    if cls_idx < 0 or cls_idx >= self.num_classes:
                        continue
                    # Encode from the same fixed cell center used during decoding.
                    qx = (gx + 0.5) * stride
                    qy = (gy + 0.5) * stride
                    cls_target[b, cls_idx, gy, gx] = 1.0
                    obj_mask[b, 0, gy, gx] = True
                    box_target[b, :, gy, gx] = torch.tensor(
                        [(qx - x1) / stride, (qy - y1) / stride, (x2 - qx) / stride, (y2 - qy) / stride],
                        device=device,
                        dtype=pred_ltrb.dtype,
                    )

            prob = torch.sigmoid(logits)
            pt = prob * cls_target + (1.0 - prob) * (1.0 - cls_target)
            alpha = self.alpha * cls_target + (1.0 - self.alpha) * (1.0 - cls_target)
            cls_loss = F.binary_cross_entropy_with_logits(logits, cls_target, reduction="none")
            total_cls_loss = total_cls_loss + (alpha * (1.0 - pt).pow(self.gamma) * cls_loss).sum()

            pos = obj_mask.expand_as(pred_ltrb)
            if obj_mask.any():
                positive_count += int(obj_mask.sum().item())
                total_box_loss = F.smooth_l1_loss(pred_ltrb[pos], box_target[pos], reduction="sum")
                total_cls_loss = total_cls_loss + self.box_weight * total_box_loss

                batch_indices, ys, xs = torch.where(obj_mask[:, 0])
                px = (xs.float() + 0.5) * stride
                py = (ys.float() + 0.5) * stride
                pred = pred_ltrb[batch_indices, :, ys, xs] * stride
                tgt = box_target[batch_indices, :, ys, xs] * stride
                pred_boxes.append(torch.stack([px - pred[:, 0], py - pred[:, 1], px + pred[:, 2], py + pred[:, 3]], dim=1))
                target_boxes.append(torch.stack([px - tgt[:, 0], py - tgt[:, 1], px + tgt[:, 2], py + tgt[:, 3]], dim=1))

        normalizer = max(1, positive_count)
        loss = total_cls_loss / normalizer
        if pred_boxes:
            od_iou = _box_iou_aligned(torch.cat(pred_boxes), torch.cat(target_boxes)).mean()
        else:
            od_iou = torch.zeros((), device=device)
        return loss, od_iou


@torch.no_grad()
def decode_anchor_free(outputs, image_size, score_thresh=0.25, nms_thresh=0.5, topk=300):
    cls_logits = outputs["cls_logits"]
    box_ltrb = outputs["box_ltrb"]
    strides = outputs["strides"]
    batch_size = cls_logits[0].shape[0]
    image_h, image_w = image_size
    results = []

    for b in range(batch_size):
        boxes_all = []
        scores_all = []
        classes_all = []
        for logits, ltrb, stride in zip(cls_logits, box_ltrb, strides):
            scores = torch.sigmoid(logits[b])
            num_classes, h, w = scores.shape
            level_boxes, valid_locations = _decode_level_boxes(
                ltrb[b],
                stride,
                image_size,
            )
            valid_scores = torch.isfinite(scores) & valid_locations.unsqueeze(0)
            scores = scores.masked_fill(~valid_scores, float("-inf"))
            flat_scores = scores.reshape(-1)
            keep_count = min(int(topk), flat_scores.numel())
            top_scores, top_idx = torch.topk(flat_scores, k=keep_count)
            keep = top_scores >= score_thresh
            if not keep.any():
                continue
            top_scores = top_scores[keep]
            top_idx = top_idx[keep]
            cls = torch.div(top_idx, h * w, rounding_mode="trunc")
            loc = top_idx % (h * w)
            ys = torch.div(loc, w, rounding_mode="trunc")
            xs = loc % w
            boxes = level_boxes[ys, xs]
            boxes_all.append(boxes)
            scores_all.append(top_scores)
            classes_all.append(cls)

        if not boxes_all:
            results.append({"rois": [], "scores": [], "class_ids": []})
            continue

        boxes = torch.cat(boxes_all)
        scores = torch.cat(scores_all)
        classes = torch.cat(classes_all)
        keep_indices = []
        for cls_id in classes.unique():
            cls_keep = torch.where(classes == cls_id)[0]
            kept = nms(boxes[cls_keep], scores[cls_keep], nms_thresh)
            keep_indices.append(cls_keep[kept])
        keep_indices = torch.cat(keep_indices) if keep_indices else torch.empty(0, dtype=torch.long, device=boxes.device)
        keep_indices = keep_indices[scores[keep_indices].argsort(descending=True)[:topk]]

        results.append({
            "rois": boxes[keep_indices].detach().cpu().numpy(),
            "scores": scores[keep_indices].detach().cpu().numpy(),
            "class_ids": classes[keep_indices].detach().cpu().numpy(),
        })
    return results
