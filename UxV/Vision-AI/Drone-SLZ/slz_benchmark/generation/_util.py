"""Small file, path and array helpers shared by the generator stages."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def log(msg: str):
    print(msg, flush=True)


def mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def path_exists(path: Path) -> bool:
    """True also for a dangling symlink."""
    return path.exists() or path.is_symlink()


def resolve_path(root: Path, value) -> Optional[Path]:
    """Manifest value -> path; relative values are resolved against ``root``."""
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    p = Path(s)
    return p if p.is_absolute() else root / p


def load_array(path: Path, key: Optional[str] = None):
    """Load a ``.npy`` or ``.npz`` array (``key`` picks one array of an ``.npz``)."""
    if path.suffix.lower() == ".npy":
        return np.load(path)
    if path.suffix.lower() == ".npz":
        z = np.load(path)
        if key and key in z.files:
            return z[key]
        if len(z.files) == 1:
            return z[z.files[0]]
        raise KeyError(f"{path}: keys={z.files}")
    raise ValueError(path)
