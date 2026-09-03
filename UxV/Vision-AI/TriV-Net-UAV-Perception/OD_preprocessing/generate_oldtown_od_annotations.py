from pathlib import Path
import json

import cv2
import numpy as np
from PIL import Image


DATA_ROOT = Path("data/oldtown")
AREA_RATIO_THRESHOLD = 0.0001

OD_CLASSES = [
    {"class_id": 0, "class_name": "Street Light", "seg_id": 11},
    {"class_id": 1, "class_name": "Traffic Light", "seg_id": 5},
    {"class_id": 2, "class_name": "Road Sign", "seg_id": 2},
    {"class_id": 3, "class_name": "Guard Rail", "seg_id": 7},
    {"class_id": 4, "class_name": "Waste Bin", "seg_id": 6},
]


def box_from_component(labels, component_id):
    ys, xs = np.where(labels == component_id)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return x1, y1, x2, y2


def collect_annotations(seg):
    h, w = seg.shape
    min_area = w * h * AREA_RATIO_THRESHOLD
    annotations = []

    for item in OD_CLASSES:
        mask = (seg == item["seg_id"]).astype(np.uint8)
        if mask.max() == 0:
            continue

        num_labels, labels = cv2.connectedComponents(mask, connectivity=8)
        for component_id in range(1, num_labels):
            box = box_from_component(labels, component_id)
            if box is None:
                continue

            x1, y1, x2, y2 = box
            bw = x2 - x1
            bh = y2 - y1
            if bw * bh <= min_area:
                continue

            annotations.append(
                {
                    "class_id": int(item["class_id"]),
                    "class_name": item["class_name"],
                    "bbox": [int(x1), int(y1), int(bw), int(bh)],
                }
            )

    return annotations


def main():
    total_files = 0
    total_boxes = 0
    empty_files = 0

    for scene_dir in sorted(DATA_ROOT.glob("*/*")):
        seg_dir = scene_dir / "seg_train_id"
        if not scene_dir.is_dir() or not seg_dir.exists():
            continue

        ann_dir = scene_dir / "annotiation"
        ann_dir.mkdir(parents=True, exist_ok=True)

        scene_files = 0
        scene_boxes = 0
        scene_empty = 0
        for seg_path in sorted(seg_dir.glob("*_train_id.png")):
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
        empty_files += scene_empty
        print(
            f"{scene_dir.relative_to(DATA_ROOT)} files={scene_files} "
            f"boxes={scene_boxes} empty={scene_empty}",
            flush=True,
        )

    print(f"total files={total_files} boxes={total_boxes} empty={empty_files}")


if __name__ == "__main__":
    main()
