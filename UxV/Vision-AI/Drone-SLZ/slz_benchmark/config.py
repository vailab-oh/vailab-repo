"""Run configuration: a plain dataclass filled from argparse, or from a
``configs/*.yaml`` file via :func:`RunConfig.from_yaml`."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Image size: tensor / mask shape (H, W) = (640, 768); image notation W x H = 768 x 640.
CANONICAL_HEIGHT = 640
CANONICAL_WIDTH = 768


@dataclass
class RunConfig:
    # data
    root: Optional[Path] = None              # prepared dataset root (required)
    scheme: str = "loso"                     # see datasets/uavscenes/splits.py
    fold: Optional[int] = None
    # explicit manifest csv (evaluation only); wins over scheme/fold when set
    manifest: Optional[str] = None
    slz_encoding: str = "class_0_1_255"      # 0 Unsafe / 1 Safe / 255 Ignore
    height: int = CANONICAL_HEIGHT
    width: int = CANONICAL_WIDTH
    batch_size: int = 4
    num_workers: int = 4
    hflip_prob: float = 0.5
    # image-only photometric augmentation; the paper recipe (configs/*.yaml,
    # tools/run_experiments.py) uses 0.5 / 0.3
    jitter_prob: float = 0.0
    blur_prob: float = 0.0

    # model
    model: str = "segformer_b0"              # --model key (MODELS.md)
    # backbone id; ``None`` = the model key's default. train.py stores the concrete
    # name back before the run, so config.json never records ``null``.
    model_name: Optional[str] = None

    # optim / schedule
    epochs: int = 40
    lr: float = 6e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    poly_power: float = 0.9
    safe_weight: float = 1.0
    # how ``safe_weight`` was set: "default" | "manual" | "auto_inverse_freq_fold{N}"
    safe_weight_source: str = "default"
    grad_clip: float = 1.0

    # runtime
    seed: int = 42
    amp: bool = True
    deterministic: bool = False              # torch.use_deterministic_algorithms + cudnn
    max_steps: Optional[int] = None          # smoke runs only
    skip_test: bool = False

    output: Path = Path("outputs/run")

    @property
    def image_size(self) -> tuple[int, int]:
        return (self.height, self.width)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        import yaml

        data = yaml.safe_load(Path(path).read_text()) or {}
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")
        for key in ("root", "output"):
            if key in data:
                data[key] = Path(data[key])
        return cls(**data)
