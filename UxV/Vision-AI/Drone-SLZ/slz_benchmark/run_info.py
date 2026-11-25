"""Run information recorded in every ``config.json``: git commit, package versions,
CUDA / GPU, and file hashes.

Best effort: a field that cannot be determined is set to ``null``, so collecting it never
stops a run.
"""

from __future__ import annotations

import hashlib
import importlib
import platform
import subprocess
from pathlib import Path
from typing import Optional

_PACKAGES = ("torch", "torchvision", "transformers", "timm", "numpy", "scipy", "fvcore")


def _pkg_version(name: str) -> Optional[str]:
    try:
        return importlib.import_module(name).__version__
    except Exception:
        try:
            return importlib.metadata.version(name)  # type: ignore[attr-defined]
        except Exception:
            return None


def git_commit(repo_root: str | Path | None = None) -> dict:
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parent.parent
    try:
        rev = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        status = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode()
        return {"commit": rev, "dirty": bool(status.strip())}
    except Exception:
        return {"commit": None, "dirty": None}


def cuda_env() -> dict:
    info: dict = {
        "cuda_version": None,
        "cudnn_version": None,
        "gpu_name": None,
        "torch_cuda_available": None,
    }
    try:
        import torch

        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = torch.version.cuda
        if torch.backends.cudnn.is_available():
            info["cudnn_version"] = torch.backends.cudnn.version()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


def sha256_files(paths) -> dict:
    """``{name: hexdigest}`` for each existing path; missing -> ``None``."""
    out: dict = {}
    for p in paths:
        p = Path(p)
        try:
            out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        except Exception:
            out[p.name] = None
    return out


def collect(repo_root: str | Path | None = None) -> dict:
    """The run-information block of ``config.json``."""
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": {name: _pkg_version(name) for name in _PACKAGES},
        "git": git_commit(repo_root),
        **cuda_env(),
    }
