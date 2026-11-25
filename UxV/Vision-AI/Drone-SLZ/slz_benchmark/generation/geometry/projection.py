"""Scaled camera intrinsics, backprojection, up-vector and normal transforms. Imports no other ``generation`` module."""

from __future__ import annotations

import numpy as np


def scaled_K(info, shape):
    H, W = shape
    K = np.asarray(info["P3x3"], np.float64).copy()
    K[0, :] *= W / float(info["Width"])
    K[1, :] *= H / float(info["Height"])
    return K


def up_vec(axis):
    return {
        "x": np.array([1,0,0], np.float64),
        "-x": np.array([-1,0,0], np.float64),
        "y": np.array([0,1,0], np.float64),
        "-y": np.array([0,-1,0], np.float64),
        "z": np.array([0,0,1], np.float64),
        "-z": np.array([0,0,-1], np.float64),
    }[axis]


def up_in_camera(info, axis):
    R = np.asarray(info["T4x4"], np.float64)[:3, :3]  # cam -> world
    u = R.T @ up_vec(axis)
    return (u / max(np.linalg.norm(u), 1e-12)).astype(np.float32)


def backproject(D, info):
    H, W = D.shape
    K = scaled_K(info, D.shape)
    fx, fy = float(K[0,0]), float(K[1,1])
    cx, cy = float(K[0,2]), float(K[1,2])
    uu = np.arange(W, dtype=np.float32)[None, :]
    vv = np.arange(H, dtype=np.float32)[:, None]
    z = D.astype(np.float32, copy=False)
    P = np.empty((H, W, 3), np.float32)
    P[...,0] = (uu-cx)/fx*z
    P[...,1] = (vv-cy)/fy*z
    P[...,2] = z
    P[D <= 0] = 0
    return P, K


def normal_cam_to_world(N_cam, info):
    """Transform camera-coordinate normals to world coordinates."""
    R = np.asarray(info["T4x4"], np.float64)[:3, :3].astype(np.float32)
    Nw = N_cam.astype(np.float32) @ R.T
    L = np.linalg.norm(Nw, axis=2)
    good = np.isfinite(L) & (L > 1e-8)
    out = np.zeros_like(Nw, np.float32)
    out[good] = Nw[good] / L[good, None]
    return out


def orient_world_normals_up(Nw, valid, up_axis):
    """Resolve PCA/mesh normal sign ambiguity for fair visualization."""
    out = Nw.copy()
    u = up_vec(up_axis).astype(np.float32)
    dot = np.sum(out * u[None, None, :], axis=2)
    flip = valid & (dot < 0)
    out[flip] *= -1.0
    return out


def normal_angle_difference(N1, N2, valid):
    """Unsigned angular difference; n and -n are treated as the same surface normal."""
    c = np.abs(np.sum(N1 * N2, axis=2))
    a = np.degrees(np.arccos(np.clip(c, 0.0, 1.0))).astype(np.float32)
    a[~valid] = 0.0
    return a
