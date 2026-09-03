from pathlib import Path
import json

import cv2
import numpy as np
from PIL import Image


DATA_ROOT = Path("data/neighborhood")
AREA_RATIO_THRESHOLD = 0.0001
HOUSE_AREA_RATIO_THRESHOLD = 0.001

OD_CLASSES = [
    {"class_id": 0, "class_name": "Pole", "class_name_ko": "전봇대", "seg_id": 10},
    {"class_id": 1, "class_name": "House", "class_name_ko": "집", "seg_id": 1},
    {"class_id": 2, "class_name": "Car", "class_name_ko": "자동차", "seg_id": 9},
]


def component_box(labels, component_id):
    ys, xs = np.where(labels == component_id)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return x1, y1, x2, y2


def boxes_overlap(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and bx1 < ax2 and ay1 < by2 and by1 < ay2


def merge_overlapping_boxes(boxes):
    merged = [list(box) for box in boxes]
    changed = True
    while changed:
        changed = False
        result = []
        used = [False] * len(merged)
        for i, box in enumerate(merged):
            if used[i]:
                continue
            x1, y1, x2, y2 = box
            used[i] = True
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                if boxes_overlap((x1, y1, x2, y2), merged[j]):
                    bx1, by1, bx2, by2 = merged[j]
                    x1 = min(x1, bx1)
                    y1 = min(y1, by1)
                    x2 = max(x2, bx2)
                    y2 = max(y2, by2)
                    used[j] = True
                    changed = True
            result.append([x1, y1, x2, y2])
        merged = result
    return [tuple(box) for box in merged]


def collect_annotations(seg):
    h, w = seg.shape
    annotations = []

    for item in OD_CLASSES:
        mask = (seg == item["seg_id"]).astype(np.uint8)
        if mask.max() == 0:
            continue

        area_ratio = HOUSE_AREA_RATIO_THRESHOLD if item["class_name"] == "House" else AREA_RATIO_THRESHOLD
        min_area = w * h * area_ratio
        num_labels, labels = cv2.connectedComponents(mask, connectivity=8)

        component_boxes = []
        for component_id in range(1, num_labels):
            box = component_box(labels, component_id)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            if (x2 - x1) * (y2 - y1) <= min_area:
                continue
            component_boxes.append((x1, y1, x2, y2))

        if item["class_name"] == "Car":
            component_boxes = merge_overlapping_boxes(component_boxes)

        for x1, y1, x2, y2 in component_boxes:
            bw, bh = x2 - x1, y2 - y1
            if bw * bh <= min_area:
                continue
            annotations.append(
                {
                    "class_id": int(item["class_id"]),
                    "class_name": item["class_name"],
                    "class_name_ko": item["class_name_ko"],
                    "bbox": [int(x1), int(y1), int(bw), int(bh)],
                }
            )

    return annotations


def main():
    total_files = 0
    total_boxes = 0
    total_empty = 0

    scene_dirs = sorted(p for p in DATA_ROOT.glob("*/*") if (p / "seg_train_id").exists())
    for scene_idx, scene_dir in enumerate(scene_dirs, start=1):
        ann_dir = scene_dir / "annotiation"
        ann_dir.mkdir(parents=True, exist_ok=True)

        scene_files = 0
        scene_boxes = 0
        scene_empty = 0
        for seg_path in sorted((scene_dir / "seg_train_id").glob("*_train_id.png")):
            seg = np.asarray(Image.open(seg_path))
            annotations = collect_annotations(seg)

            stem = seg_path.name.replace("_train_id.png", "")
            out_path = ann_dir / f"{stem}_det.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(annotations, f, ensure_ascii=False)

            scene_files += 1
            scene_boxes += len(annotations)
            if not annotations:
                scene_empty += 1

        total_files += scene_files
        total_boxes += scene_boxes
        total_empty += scene_empty
        print(
            f"{scene_idx}/{len(scene_dirs)} {scene_dir.relative_to(DATA_ROOT)} "
            f"files={scene_files} boxes={scene_boxes} empty={scene_empty}",
            flush=True,
        )

    print(f"total files={total_files} boxes={total_boxes} empty={total_empty}")


if __name__ == "__main__":
    main()
