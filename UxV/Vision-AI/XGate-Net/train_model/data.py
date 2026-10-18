import csv
import hashlib
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


CLASS_COLORS = {
    0: (0, 0, 0), 1: (210, 0, 200), 2: (90, 200, 255), 3: (0, 199, 0),
    4: (90, 240, 0), 5: (140, 140, 140), 6: (100, 60, 100),
    7: (250, 100, 255), 8: (255, 255, 0), 9: (200, 200, 0),
    10: (255, 130, 0), 11: (80, 80, 80), 12: (160, 60, 60),
    13: (255, 127, 80), 14: (0, 139, 139),
}

normalize_img = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
to_tensor = transforms.ToTensor()


def scan_modality(root, folder, frame_folder, extensions):
    found = {}
    for path in sorted((root / folder).glob(f"*/*/frames/{frame_folder}/Camera_0/*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        digits = re.findall(r"\d+", path.stem)
        if not digits:
            raise ValueError(f"Missing frame ID: {path}")
        scene, condition = path.relative_to(root / folder).parts[:2]
        key = (scene, condition, int(digits[-1]))
        if key in found:
            raise ValueError(f"Duplicate {folder} frame {key}: {found[key]} and {path}")
        found[key] = path.resolve()
    return found


def read_image(path, flags):
    image = cv2.imread(str(path), flags)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


class Tartan(Dataset):

    def __init__(self, train, train_config, data_root="data"):
        split = "train" if train else "val"
        root = Path(data_root).resolve() / split
        rgb = scan_modality(root, "rgb", "rgb", {".jpg", ".jpeg", ".png"})
        seg = scan_modality(root, "label", "classSegmentation", {".png"})
        depth = scan_modality(root, "depth", "depth", {".png"})
        if not rgb:
            raise ValueError(f"No RGB images in {root}/rgb/*/*/frames/rgb/Camera_0/")
        if set(rgb) != set(seg) or set(rgb) != set(depth):
            raise ValueError(
                f"{split}: unmatched frames; RGB={len(rgb)}, seg={len(seg)}, depth={len(depth)}; "
                f"missing seg={sorted(set(rgb) - set(seg))[:5]}, "
                f"missing depth={sorted(set(rgb) - set(depth))[:5]}, "
                f"extra seg={sorted(set(seg) - set(rgb))[:5]}, "
                f"extra depth={sorted(set(depth) - set(rgb))[:5]}"
            )
        self.keys = sorted(rgb)
        self.image_front = [str(rgb[k]) for k in self.keys]
        self.seg_front = [str(seg[k]) for k in self.keys]
        self.depth_front = [str(depth[k]) for k in self.keys]
        self.W, self.H = int(train_config["width"]), int(train_config["height"])
        self.records = [
            {"scene": key[0], "condition": key[1], "frame": key[2],
             "rgb": self.image_front[i], "seg": self.seg_front[i], "depth": self.depth_front[i]}
            for i, key in enumerate(self.keys)
        ]
        self.manifest_sha256 = hashlib.sha256(
            json.dumps(self.records, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        rgb = read_image(self.image_front[index], cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.W, self.H), interpolation=cv2.INTER_LINEAR)

        seg = read_image(self.seg_front[index], cv2.IMREAD_COLOR)
        seg = cv2.resize(seg, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)

        depth = read_image(self.depth_front[index], cv2.IMREAD_ANYDEPTH).astype(np.float32)
        depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        depth = depth / 65535 * 80
        valid = (depth > 0) & (depth <= 80)
        return (normalize_img(rgb), to_tensor(seg), to_tensor(depth),
                torch.from_numpy(valid).unsqueeze(0))

    def save_manifest(self, path):
        with Path(path).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.records[0]))
            writer.writeheader()
            writer.writerows(self.records)


def validate_split(train_set, val_set):
    overlap = set(train_set.keys) & set(val_set.keys)
    if overlap:
        raise ValueError(f"Identical Scene/Condition/Frame IDs in train and val: {sorted(overlap)[:10]}")
