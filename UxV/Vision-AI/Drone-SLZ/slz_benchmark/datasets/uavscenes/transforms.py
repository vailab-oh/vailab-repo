from __future__ import annotations

import random
from typing import Sequence

import torch
import torch.nn.functional as F


def _resize_image(x, size):
    return F.interpolate(
        x.unsqueeze(0),
        size=size,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(0)


def _resize_nearest_2d(x, size):
    dtype = x.dtype
    y = F.interpolate(
        x[None, None].float(),
        size=size,
        mode="nearest",
    )[0, 0]
    return y.to(dtype)


def _resize_mask(mask, size):
    if mask is None:
        return None
    y = F.interpolate(
        mask[None].float(),
        size=size,
        mode="nearest",
    )[0]
    return y > 0.5


def _masked_bilinear(x, valid, size, eps=1e-6):
    """
    Bilinear resize without allowing invalid zero values to bleed into valid data.
    x:     [C,H,W]
    valid: [1,H,W] bool
    """
    if valid is None:
        return F.interpolate(
            x.unsqueeze(0),
            size=size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0), None

    vf = valid.float()
    num = F.interpolate(
        (x * vf).unsqueeze(0),
        size=size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    den = F.interpolate(
        vf.unsqueeze(0),
        size=size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    y = num / den.clamp_min(eps)
    valid_out = _resize_mask(valid, size)
    y = torch.where(valid_out.expand_as(y), y, torch.zeros_like(y))
    return y, valid_out


class Compose:
    def __init__(self, transforms: Sequence):
        self.transforms = list(transforms)

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class Resize:
    """
    Synchronized resize for RGB + all dense targets.

    - image: bilinear
    - semantic/slz/valid masks: nearest
    - depth/slope/roughness: masked bilinear
    - normal: masked bilinear + renormalization
    - K: scaled to resized image
    """

    def __init__(self, size: tuple[int, int]):
        self.size = tuple(map(int, size))

    def __call__(self, sample):
        old_h, old_w = sample["image"].shape[-2:]
        new_h, new_w = self.size

        if (old_h, old_w) == (new_h, new_w):
            return sample

        sample["image"] = _resize_image(sample["image"], self.size)

        for key in ("semantic", "slz"):
            if key in sample:
                sample[key] = _resize_nearest_2d(sample[key], self.size)

        depth_valid = sample.get("depth_valid")
        normal_valid = sample.get("normal_valid")
        geometry_valid = sample.get("geometry_valid")

        if "depth" in sample:
            sample["depth"], sample["depth_valid"] = _masked_bilinear(
                sample["depth"], depth_valid, self.size
            )

        if "normal" in sample:
            n, nv = _masked_bilinear(
                sample["normal"], normal_valid, self.size
            )
            norm = torch.linalg.vector_norm(n, dim=0, keepdim=True)
            nv = nv & (norm > 1e-6)
            n = torch.where(
                nv.expand_as(n),
                n / norm.clamp_min(1e-6),
                torch.zeros_like(n),
            )
            sample["normal"] = n
            sample["normal_valid"] = nv

        for key in ("slope", "roughness"):
            if key in sample:
                sample[key], _ = _masked_bilinear(
                    sample[key], geometry_valid, self.size
                )

        if "depth_valid" in sample and "normal_valid" in sample:
            sample["geometry_valid"] = (
                sample["depth_valid"] & sample["normal_valid"]
            )
        elif "depth_valid" in sample:
            sample["geometry_valid"] = sample["depth_valid"].clone()
        elif "normal_valid" in sample:
            sample["geometry_valid"] = sample["normal_valid"].clone()

        if "K" in sample:
            sx = new_w / float(old_w)
            sy = new_h / float(old_h)
            K = sample["K"].clone()
            K[0, 0] *= sx
            K[0, 2] *= sx
            K[1, 1] *= sy
            K[1, 2] *= sy
            sample["K"] = K

        return sample


class RandomHorizontalFlip:
    """
    Synchronized horizontal flip.

    Camera-frame normal:
        nx -> -nx

    If T_cam_to_map is loaded, a simple image reflection is not a proper SE(3)
    camera motion. Therefore pose-aware training is blocked by default.
    """

    def __init__(self, p=0.5, allow_pose_inconsistency=False):
        self.p = float(p)
        self.allow_pose_inconsistency = bool(allow_pose_inconsistency)

    def __call__(self, sample):
        if random.random() >= self.p:
            return sample

        if (
            "T_cam_to_map" in sample
            and not self.allow_pose_inconsistency
        ):
            raise RuntimeError(
                "Horizontal flip requested while T_cam_to_map is loaded. "
                "Disable hflip, set load_pose=False, or explicitly allow pose inconsistency."
            )

        spatial_keys = (
            "image",
            "semantic",
            "slz",
            "depth",
            "normal",
            "slope",
            "roughness",
            "depth_valid",
            "normal_valid",
            "geometry_valid",
        )
        for key in spatial_keys:
            if key in sample:
                sample[key] = torch.flip(sample[key], dims=(-1,))

        if "normal" in sample:
            # Reflection of camera x-axis.
            sample["normal"][0].mul_(-1.0)

        if "K" in sample:
            w = sample["image"].shape[-1]
            K = sample["K"].clone()
            K[0, 2] = (w - 1) - K[0, 2]
            sample["K"] = K

        return sample


class ColorJitter:
    """Photometric-only jitter: touches ``image`` alone, no mask/geometry sync needed
    (unlike ``RandomHorizontalFlip``/``Resize`` — nothing here moves a pixel).

    Applied to the ``[0, 1]``-range image (before ``NormalizeImage``); clamps back into
    ``[0, 1]`` afterward since torchvision's brightness/contrast/saturation adjustments
    aren't guaranteed to stay in range, and the dataset contract promises ``[0, 1]``
    pre-normalize.
    """

    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.5):
        from torchvision.transforms import ColorJitter as _ColorJitter

        self._jitter = _ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )
        self.p = float(p)

    def __call__(self, sample):
        if random.random() >= self.p:
            return sample
        sample["image"] = self._jitter(sample["image"]).clamp_(0.0, 1.0)
        return sample


class RandomGaussianBlur:
    """Photometric-only Gaussian blur: touches ``image`` alone, same reasoning as
    :class:`ColorJitter`."""

    def __init__(self, kernel_size=5, sigma=(0.1, 2.0), p=0.3):
        from torchvision.transforms import GaussianBlur as _GaussianBlur

        self._blur = _GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        self.p = float(p)

    def __call__(self, sample):
        if random.random() >= self.p:
            return sample
        sample["image"] = self._blur(sample["image"])
        return sample


class NormalizeImage:
    def __init__(
        self,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        self.mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(std, dtype=torch.float32)[:, None, None]

    def __call__(self, sample):
        image = sample["image"]
        sample["image"] = (
            image - self.mean.to(image.device)
        ) / self.std.to(image.device)
        return sample
