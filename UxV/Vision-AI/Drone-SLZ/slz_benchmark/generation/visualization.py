"""SLZ colorization (``slz_viz``) and diagnostic figures of the generator. Uses cv2 / numpy / PIL only."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ._util import mkdir


def normal_rgb(Nw, valid):
    """Map [-1,1] world normal components to [0,255] RGB for visualization."""
    vis = ((np.clip(Nw, -1, 1) + 1.0) * 127.5).astype(np.uint8)
    vis[~valid] = 0
    # cv2.imwrite expects BGR. Input components are x,y,z; display as RGB=(x,y,z).
    return vis[..., ::-1]


def load_rgb(path, shape):
    H,W = shape
    if path is None or not path.exists():
        return np.zeros((H,W,3),np.uint8)
    im = cv2.imread(str(path))
    if im is None:
        return np.zeros((H,W,3),np.uint8)
    if im.shape[:2] != (H,W):
        im = cv2.resize(im,(W,H),interpolation=cv2.INTER_AREA)
    return im


def color_scalar(a, valid, vmax):
    x = np.zeros_like(a,np.float32)
    x[valid] = np.clip(a[valid]/max(vmax,1e-8),0,1)
    out = cv2.applyColorMap(np.round(x*255).astype(np.uint8),cv2.COLORMAP_TURBO)
    out[~valid] = 0
    return out


def overlay(rgb, mask, color=(0,255,0), alpha=.5):
    o = rgb.astype(np.float32).copy()
    c = np.asarray(color,np.float32)
    o[mask] = (1-alpha)*o[mask] + alpha*c
    return np.clip(o,0,255).astype(np.uint8)


def titled(im, title, width=420):
    H,W = im.shape[:2]
    h = max(1,round(H*width/W))
    im = cv2.resize(im,(width,h),interpolation=cv2.INTER_AREA)
    bar = np.full((40,width,3),245,np.uint8)
    cv2.putText(bar,title,(8,27),cv2.FONT_HERSHEY_SIMPLEX,.52,(20,20,20),1,cv2.LINE_AA)
    return np.vstack([bar,im])


def save_grid(path, panels, cols=3):
    imgs = [titled(im,t) for t,im in panels]
    mh = max(x.shape[0] for x in imgs)
    imgs = [np.vstack([x,np.full((mh-x.shape[0],x.shape[1],3),255,np.uint8)]) if x.shape[0]<mh else x for x in imgs]
    rows=[]
    for s in range(0,len(imgs),cols):
        g=imgs[s:s+cols]
        while len(g)<cols:
            g.append(np.full_like(imgs[0],255))
        rows.append(np.hstack(g))
    mkdir(path.parent)
    canvas = np.vstack(rows)
    ok = cv2.imwrite(str(path), canvas)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed: {path}")
    print(f"[VIZ] saved: {path}", flush=True)


def colorize_slz_label(label: np.ndarray) -> np.ndarray:
    """Colorize a class-index SLZ target for human inspection only.

    Training/evaluation MUST continue to use scenes/<scene>/slz/*.png:
      0   = Unsafe
      1   = Safe
      255 = Ignore

    Visualization colors are RGB:
      Unsafe = red, Safe = green, Ignore = gray.
    Unexpected values are magenta so encoding mistakes are obvious.
    """
    a = np.asarray(label, dtype=np.uint8)
    if a.ndim != 2:
        raise ValueError(f"SLZ label must be HxW class-index mask, got shape={a.shape}")

    vis = np.empty((*a.shape, 3), dtype=np.uint8)
    vis[...] = np.array([255, 0, 255], dtype=np.uint8)  # unexpected value
    vis[a == 0] = (255, 0, 0)      # Unsafe: red
    vis[a == 1] = (0, 255, 0)      # Safe: green
    vis[a == 255] = (128, 128, 128)  # Ignore: gray
    return vis


def save_slz_visualization(label_path: Path, viz_path: Path):
    label = np.array(Image.open(label_path))
    if label.ndim == 3:
        # A proper SLZ target is single-channel. Fail loudly rather than silently
        # colorizing a corrupted/palette-expanded training target.
        raise ValueError(f"Expected single-channel SLZ target: {label_path}, got {label.shape}")
    mkdir(viz_path.parent)
    Image.fromarray(colorize_slz_label(label), mode="RGB").save(viz_path)
