"""Camera projection, z-buffer rasterisation and vertex / face normals -> RGB-aligned mesh
depth / normal maps. ``numba`` is optional and only imported here.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    from numba import njit
except Exception:
    def njit(*args, **kwargs):
        def deco(fn):
            return fn
        return deco


@njit(cache=False)
def rasterize(u, v, C, Fv, VN, H, W, D, NF, NV):
    eps = 1e-6
    for fi in range(Fv.shape[0]):
        i0, i1, i2 = Fv[fi, 0], Fv[fi, 1], Fv[fi, 2]
        x0, y0, z0 = u[i0], v[i0], C[i0, 2]
        x1, y1, z1 = u[i1], v[i1], C[i1, 2]
        x2, y2, z2 = u[i2], v[i2], C[i2, 2]
        if z0 <= 0.05 or z1 <= 0.05 or z2 <= 0.05:
            continue
        minx = int(np.floor(min(x0, x1, x2)))
        maxx = int(np.ceil(max(x0, x1, x2)))
        miny = int(np.floor(min(y0, y1, y2)))
        maxy = int(np.ceil(max(y0, y1, y2)))
        if maxx < 0 or maxy < 0 or minx >= W or miny >= H:
            continue
        if minx < 0: minx = 0
        if miny < 0: miny = 0
        if maxx >= W: maxx = W - 1
        if maxy >= H: maxy = H - 1
        area = (x1-x0)*(y2-y0) - (x2-x0)*(y1-y0)
        if abs(area) < eps:
            continue
        inv = 1.0 / area

        e1x, e1y, e1z = C[i1,0]-C[i0,0], C[i1,1]-C[i0,1], C[i1,2]-C[i0,2]
        e2x, e2y, e2z = C[i2,0]-C[i0,0], C[i2,1]-C[i0,1], C[i2,2]-C[i0,2]
        nx = e1y*e2z - e1z*e2y
        ny = e1z*e2x - e1x*e2z
        nz = e1x*e2y - e1y*e2x
        nl = np.sqrt(nx*nx + ny*ny + nz*nz)
        if nl < eps:
            continue
        nx, ny, nz = nx/nl, ny/nl, nz/nl
        cx, cy, cz = (C[i0,0]+C[i1,0]+C[i2,0])/3.0, (C[i0,1]+C[i1,1]+C[i2,1])/3.0, (C[i0,2]+C[i1,2]+C[i2,2])/3.0
        if nx*cx + ny*cy + nz*cz > 0:
            nx, ny, nz = -nx, -ny, -nz

        n0x, n0y, n0z = VN[i0,0], VN[i0,1], VN[i0,2]
        n1x, n1y, n1z = VN[i1,0], VN[i1,1], VN[i1,2]
        n2x, n2y, n2z = VN[i2,0], VN[i2,1], VN[i2,2]

        for yy in range(miny, maxy+1):
            py = yy + 0.5
            for xx in range(minx, maxx+1):
                px = xx + 0.5
                w0 = ((x1-px)*(y2-py) - (x2-px)*(y1-py)) * inv
                w1 = ((x2-px)*(y0-py) - (x0-px)*(y2-py)) * inv
                w2 = 1.0 - w0 - w1
                if w0 >= -1e-4 and w1 >= -1e-4 and w2 >= -1e-4:
                    zz = w0*z0 + w1*z1 + w2*z2
                    if zz > 0.05 and zz < D[yy, xx]:
                        D[yy, xx] = zz
                        NF[yy, xx, 0] = nx
                        NF[yy, xx, 1] = ny
                        NF[yy, xx, 2] = nz
                        vx = w0*n0x + w1*n1x + w2*n2x
                        vy = w0*n0y + w1*n1y + w2*n2y
                        vz = w0*n0z + w1*n1z + w2*n2z
                        vl = np.sqrt(vx*vx + vy*vy + vz*vz)
                        if vl > eps:
                            vx, vy, vz = vx/vl, vy/vl, vz/vl
                        NV[yy, xx, 0] = vx
                        NV[yy, xx, 1] = vy
                        NV[yy, xx, 2] = vz


def vertex_normals(C: np.ndarray, F: np.ndarray, chunk=200000) -> np.ndarray:
    VN = np.zeros_like(C, dtype=np.float32)
    for s in range(0, F.shape[0], chunk):
        Fc = F[s:s+chunk]
        p0, p1, p2 = C[Fc[:,0]], C[Fc[:,1]], C[Fc[:,2]]
        cr = np.cross(p1-p0, p2-p0)
        a2 = np.linalg.norm(cr, axis=1).astype(np.float32)
        good = a2 > 1e-10
        fn = np.zeros_like(cr, dtype=np.float32)
        fn[good] = cr[good] / a2[good, None]
        cen = (p0 + p1 + p2) / 3.0
        flip = np.sum(fn * cen, axis=1) > 0
        fn[flip] *= -1
        weighted = fn * a2[:, None]
        for k in range(3):
            np.add.at(VN, Fc[:, k], weighted)
    n = np.linalg.norm(VN, axis=1)
    good = n > 1e-10
    VN[good] /= n[good, None]
    return VN.astype(np.float32)


def render_mesh(V, F, info: dict, shape: Tuple[int, int]):
    H, W = shape
    K = np.array(info["P3x3"], dtype=np.float64)
    K[0, :] *= W / float(info["Width"])
    K[1, :] *= H / float(info["Height"])
    T = np.array(info["T4x4"], dtype=np.float64)
    Tinv = np.linalg.inv(T)  # sampleinfos T4x4: camera->map, rendering: map->camera
    R, t = Tinv[:3, :3].astype(np.float32), Tinv[:3, 3].astype(np.float32)
    C = (np.asarray(V, dtype=np.float32) @ R.T + t).astype(np.float32)
    z = C[:, 2]
    u = (K[0,0] * C[:,0] / z + K[0,2]).astype(np.float32)
    v = (K[1,1] * C[:,1] / z + K[1,2]).astype(np.float32)
    zf, uf, vf = z[F], u[F], v[F]
    vis = (zf > 0.05).all(1) & (uf.max(1) >= 0) & (uf.min(1) < W) & (vf.max(1) >= 0) & (vf.min(1) < H)
    Fv = np.asarray(F[vis], dtype=np.int32)
    VN = vertex_normals(C, np.asarray(F, dtype=np.int32))
    D = np.full((H, W), np.inf, np.float32)
    NF = np.zeros((H, W, 3), np.float32)
    NV = np.zeros((H, W, 3), np.float32)
    rasterize(u, v, C, Fv, VN, H, W, D, NF, NV)
    valid = np.isfinite(D)
    D[~valid] = 0
    NF[~valid] = 0
    NV[~valid] = 0
    return D, NF, NV
