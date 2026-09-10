import glob
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset


normalize_img = torchvision.transforms.Compose((
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
))

ToTensor = torchvision.transforms.ToTensor()


def img_transform(img, resize, resize_dims, interpolation=cv2.INTER_LINEAR):
    img = cv2.resize(img, resize_dims, interpolation=interpolation)
    post_rot = torch.eye(3)
    post_tran = torch.zeros(3)
    post_rot[0, 0] = resize[0]
    post_rot[1, 1] = resize[1]
    return img, post_rot, post_tran


class Tartan(Dataset):
    def __init__(self, train, train_config):
        conf = train_config.get("model_conf", {})
        self.use_seg = bool(conf.get("Seg", False))
        self.use_depth = bool(conf.get("Depth", False))
        self.use_od = bool(conf.get("od", False))
        self.W = int(train_config["width"])
        self.H = int(train_config["height"])
        self.dataset_root = Path(train_config.get("dataset_root", "data/oldtown"))
        self.od_min_box_size = float(train_config.get("od_min_box_size", 1))
        values = train_config.get("od_class_values", None)
        self.od_class_to_index = {int(raw): idx for idx, raw in enumerate(values)} if values else None

        split = "train" if train else "val"
        selected = self._discover_split_samples(self.dataset_root / split)
        if not selected:
            raise ValueError(f"No usable samples found in {self.dataset_root / split}")

        self.image_front = [s["image"] for s in selected]
        self.bbs_front = [s.get("det") for s in selected] if self.use_od else [None] * len(selected)
        self.seg_front = [s.get("seg") for s in selected] if self.use_seg else [None] * len(selected)
        self.depth_front = [s.get("depth") for s in selected] if self.use_depth else [None] * len(selected)

        print("총 이미지 개수:", len(self.image_front))
        print("총 bbs 개수:", len([p for p in self.bbs_front if p]) if self.use_od else 0)
        print("총 seg 이미지 개수:", len([p for p in self.seg_front if p]) if self.use_seg else 0)
        print("총 Depth 이미지 개수:", len([p for p in self.depth_front if p]) if self.use_depth else 0)

    def _discover_split_samples(self, root):
        samples = []
        for image_path in sorted(glob.glob(str(root / "*" / "image_left" / "*_left.png"))):
            scene_dir = Path(image_path).parents[1]
            stem = Path(image_path).stem
            sample = {"image": image_path}
            if self.use_seg:
                sample["seg"] = str(scene_dir / "seg_train_id" / f"{stem}_train_id.png")
            if self.use_depth:
                sample["depth"] = str(scene_dir / "depth_left" / f"{stem}_depth.npy")
            if self.use_od:
                sample["det"] = str(scene_dir / "annotiation" / f"{stem}_det.json")
            if all(Path(v).is_file() for k, v in sample.items() if k != "image"):
                samples.append(sample)
        return samples

    def __len__(self):
        return len(self.image_front)

    def sample_augmentation(self):
        resize = (self.W / 640, self.H / 480)
        resize_dims = (self.W, self.H)
        return resize, resize_dims

    def get_img(self, idx):
        resize, resize_dims = self.sample_augmentation()
        path = self.image_front[idx]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, _, _ = img_transform(img, resize, resize_dims)
        img_ori = ToTensor(img)
        img = normalize_img(img)
        return img, img_ori

    def get_seg(self, idx):
        if not self.use_seg:
            return None
        path = self.seg_front[idx]
        mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(path)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = cv2.resize(mask.astype(np.uint8), (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy(mask.astype(np.int64)).long()

    def get_depth(self, idx):
        if not self.use_depth:
            return None, None
        path = self.depth_front[idx]
        depth = np.load(path).astype(np.float32)
        if depth.ndim == 3:
            depth = depth.squeeze()
        depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask = np.isfinite(depth) & (depth > 0.0) & (depth <= 80.0)
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        return ToTensor(depth), ToTensor(mask.astype(np.uint8)).bool()

    def resize_bbs(self, bbs, resize):
        if bbs.numel() == 0:
            return bbs
        x_scale, y_scale = resize
        bbs[:, [0, 2]] *= x_scale
        bbs[:, [1, 3]] *= y_scale
        bbs[:, [0, 2]] = bbs[:, [0, 2]].clamp(0, self.W - 1)
        bbs[:, [1, 3]] = bbs[:, [1, 3]].clamp(0, self.H - 1)
        return bbs

    def get_bbs_stack(self, idx):
        if not self.use_od:
            return torch.empty((0, 5), dtype=torch.float32)

        resize, _ = self.sample_augmentation()
        with open(self.bbs_front[idx], "r", encoding="utf-8") as f:
            objects = json.load(f)

        boxes = []
        for obj in objects:
            raw_class = int(obj["class_id"])
            if self.od_class_to_index is None:
                class_idx = raw_class
            else:
                if raw_class not in self.od_class_to_index:
                    continue
                class_idx = self.od_class_to_index[raw_class]
            x, y, w, h = [float(v) for v in obj["bbox"]]
            if w < self.od_min_box_size or h < self.od_min_box_size:
                continue
            boxes.append([x, y, x + w, y + h, class_idx])

        tensor = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 5), dtype=torch.float32)
        return self.resize_bbs(tensor, resize)

    def __getitem__(self, idx):
        imgs, imgs_ori = self.get_img(idx)
        seg = self.get_seg(idx)
        depth, depth_mask = self.get_depth(idx)
        annots = self.get_bbs_stack(idx)
        return imgs, imgs_ori, seg, depth, depth_mask, annots
