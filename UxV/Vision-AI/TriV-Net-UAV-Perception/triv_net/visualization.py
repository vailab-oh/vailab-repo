import cv2
import numpy as np
import torch.nn.functional as F


OLDTOWN_OD_CLASS_NAMES = (
    "Street Light",
    "Traffic Light",
    "Road Sign",
    "Guard Rail",
    "Waste Bin",
)


SEGMENTATION_PALETTES = {
    "oldtown": np.asarray(
        [((37 * i) % 255, (17 * i + 80) % 255, (97 * i + 30) % 255) for i in range(12)],
        dtype=np.uint8,
    )
}


def resize_dense_prediction(prediction, height, width, mode):
    return F.interpolate(
        prediction.float(),
        size=(height, width),
        mode=mode,
        align_corners=False if mode in ("bilinear", "bicubic") else None,
    )


def segmentation_visualization(image_rgb, logits, dataset, alpha):
    logits = resize_dense_prediction(
        logits, image_rgb.shape[0], image_rgb.shape[1], "bilinear"
    )
    mask = logits.argmax(dim=1)[0].cpu().numpy().astype(np.int64)
    palette = SEGMENTATION_PALETTES[dataset]
    colors_rgb = palette[np.clip(mask, 0, len(palette) - 1)]
    overlay_rgb = cv2.addWeighted(image_rgb, 1.0 - alpha, colors_rgb, alpha, 0.0)
    return cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)


def depth_visualization(depth, height, width, max_depth):
    depth = (
        resize_dense_prediction(depth, height, width, "bilinear")[0, 0].cpu().numpy()
    )
    depth = np.nan_to_num(depth, nan=0.0, posinf=max_depth, neginf=0.0)
    scaled = np.clip(depth, 0.0, max_depth) / max_depth
    return cv2.applyColorMap(
        np.rint(scaled * 255.0).astype(np.uint8), cv2.COLORMAP_PLASMA
    )


def class_color(class_id):
    return (
        int((67 * class_id + 40) % 220 + 35),
        int((131 * class_id + 70) % 220 + 35),
        int((193 * class_id + 100) % 220 + 35),
    )


def abbreviated_class_name(name):
    words = [word for word in name.replace("-", " ").split() if word]
    if len(words) == 1:
        return words[0] if len(words[0]) <= 5 else words[0][:5]
    return "".join(word[0].upper() for word in words)


def detection_visualization(image_rgb, detection, class_names):
    output = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    boxes = np.asarray(detection.get("rois", []), dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(detection.get("scores", []), dtype=np.float32).reshape(-1)
    class_ids = np.asarray(detection.get("class_ids", []), dtype=np.int64).reshape(-1)
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = np.rint(box).astype(np.int64)
        x1 = int(np.clip(x1, 0, output.shape[1] - 1))
        x2 = int(np.clip(x2, 0, output.shape[1] - 1))
        y1 = int(np.clip(y1, 0, output.shape[0] - 1))
        y2 = int(np.clip(y2, 0, output.shape[0] - 1))
        color = class_color(int(class_id))
        label_name = (
            class_names[int(class_id)]
            if 0 <= int(class_id) < len(class_names)
            else str(int(class_id))
        )
        label = abbreviated_class_name(label_name)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1
        )
        text_top = max(0, y1 - text_height - baseline - 4)
        text_right = min(output.shape[1] - 1, x1 + text_width + 4)
        cv2.rectangle(output, (x1, text_top), (text_right, y1), color, -1)
        cv2.putText(
            output,
            label,
            (x1 + 2, max(text_height, y1 - baseline - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return output
