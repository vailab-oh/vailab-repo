"""Metric neighbourhood -> local PCA -> slope / roughness / validity / support."""

from __future__ import annotations

import cv2
import numpy as np

from .projection import backproject, up_in_camera, up_vec


def metric_radii(D, K, r_m, max_px):
    valid = D > 0
    z = np.maximum(D, 1e-6)
    rx = np.ceil(abs(float(K[0,0])) * r_m / z).astype(np.int32)
    ry = np.ceil(abs(float(K[1,1])) * r_m / z).astype(np.int32)
    rx = np.clip(rx, 1, max_px)
    ry = np.clip(ry, 1, max_px)
    rx[~valid] = 0
    ry[~valid] = 0
    return rx, ry


def variable_box_sum(src, rx, ry, row_chunk=128):
    H, W = src.shape
    I = cv2.integral(src.astype(np.float64), sdepth=cv2.CV_64F)
    out = np.zeros((H, W), np.float64)
    xx = np.arange(W, dtype=np.int32)[None, :]
    for ys in range(0, H, row_chunk):
        ye = min(ys + row_chunk, H)
        yy = np.arange(ys, ye, dtype=np.int32)[:, None]
        rxc, ryc = rx[ys:ye], ry[ys:ye]
        x0, x1 = np.maximum(xx-rxc,0), np.minimum(xx+rxc,W-1)
        y0, y1 = np.maximum(yy-ryc,0), np.minimum(yy+ryc,H-1)
        out[ys:ye] = I[y1+1,x1+1]-I[y0,x1+1]-I[y1+1,x0]+I[y0,x0]
    return out


def variable_box_area(rx, ry, row_chunk=128):
    H, W = rx.shape
    out = np.ones((H, W), np.float32)
    xx = np.arange(W, dtype=np.int32)[None, :]
    for ys in range(0, H, row_chunk):
        ye = min(ys+row_chunk, H)
        yy = np.arange(ys, ye, dtype=np.int32)[:, None]
        rxc, ryc = rx[ys:ye], ry[ys:ye]
        x0, x1 = np.maximum(xx-rxc,0), np.minimum(xx+rxc,W-1)
        y0, y1 = np.maximum(yy-ryc,0), np.minimum(yy+ryc,H-1)
        out[ys:ye] = ((x1-x0+1)*(y1-y0+1)).astype(np.float32)
    return out


def pca_geometry(D, info, r_m, min_support, min_points, max_px, row_chunk, up_axis):
    """
    C = E[XX^T] - mu mu^T
    n_p = eigenvector(lambda_min)
    slope = acos(|n_p^T u|)
    roughness = sqrt(lambda_min)
    """
    P, K = backproject(D, info)
    valid_d = D > 0
    rx, ry = metric_radii(D, K, r_m, max_px)
    area = variable_box_area(rx, ry, row_chunk)
    vf = valid_d.astype(np.float32)

    cnt = variable_box_sum(vf, rx, ry, row_chunk)
    den = np.maximum(cnt, 1.0)
    support = (cnt / np.maximum(area, 1.0)).astype(np.float32)

    x, y, z = P[...,0], P[...,1], P[...,2]
    mx = variable_box_sum(x*vf, rx, ry, row_chunk)/den
    my = variable_box_sum(y*vf, rx, ry, row_chunk)/den
    mz = variable_box_sum(z*vf, rx, ry, row_chunk)/den

    exx = variable_box_sum(x*x*vf, rx, ry, row_chunk)/den
    eyy = variable_box_sum(y*y*vf, rx, ry, row_chunk)/den
    ezz = variable_box_sum(z*z*vf, rx, ry, row_chunk)/den
    exy = variable_box_sum(x*y*vf, rx, ry, row_chunk)/den
    exz = variable_box_sum(x*z*vf, rx, ry, row_chunk)/den
    eyz = variable_box_sum(y*z*vf, rx, ry, row_chunk)/den

    H, W = D.shape
    C = np.empty((H,W,3,3), np.float32)
    C[...,0,0] = exx-mx*mx
    C[...,1,1] = eyy-my*my
    C[...,2,2] = ezz-mz*mz
    C[...,0,1] = C[...,1,0] = exy-mx*my
    C[...,0,2] = C[...,2,0] = exz-mx*mz
    C[...,1,2] = C[...,2,1] = eyz-my*mz
    C[~np.isfinite(C)] = 0

    base_valid = valid_d & (cnt >= min_points) & (support >= min_support)
    ids = np.flatnonzero(base_valid.ravel())
    lam = np.zeros(H*W, np.float32)
    normals = np.zeros((H*W,3), np.float32)
    eig_ok = np.zeros(H*W, bool)

    flatC = C.reshape(-1,3,3)
    for s in range(0, len(ids), 200000):
        ii = ids[s:s+200000]
        vals, vecs = np.linalg.eigh(flatC[ii])
        good = np.isfinite(vals[:,0]) & np.all(np.isfinite(vecs[:,:,0]), axis=1)
        lam[ii] = np.maximum(vals[:,0], 0).astype(np.float32)
        normals[ii] = vecs[:,:,0].astype(np.float32)
        eig_ok[ii] = good

    valid = base_valid & eig_ok.reshape(H,W)
    rough = np.sqrt(lam.reshape(H,W)).astype(np.float32)
    rough[~valid] = 0

    n = normals.reshape(H,W,3)
    u = up_in_camera(info, up_axis)
    c = np.abs(np.sum(n*u[None,None,:], axis=2))
    slope = np.degrees(np.arccos(np.clip(c,0,1))).astype(np.float32)
    slope[~valid] = 90

    return slope, rough, valid, K, support, n


def slope_from_world_normal(Nw, valid, up_axis):
    u = up_vec(up_axis).astype(np.float32)
    c = np.abs(np.sum(Nw * u[None, None, :], axis=2))
    s = np.degrees(np.arccos(np.clip(c, 0.0, 1.0))).astype(np.float32)
    s[~valid] = 90.0
    return s
