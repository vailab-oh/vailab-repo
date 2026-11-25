"""Projected metric radius -> depth-aware metric clearance -> safety buffer / UAV feasible
centers -> landable-region reconstruction. ``scipy`` is imported lazily in
``_edt_normalized``.
"""

from __future__ import annotations

import cv2
import numpy as np


def projected_radii_xy(D, K, radius_m):
    """Per-pixel projected semi-axes for a metric disk using mesh camera-Z depth."""
    fx = abs(float(K[0,0]))
    fy = abs(float(K[1,1]))
    rx = np.zeros_like(D, np.float32)
    ry = np.zeros_like(D, np.float32)
    v = D > 0
    z = np.maximum(D[v], 1e-6)
    rx[v] = fx * float(radius_m) / z
    ry[v] = fy * float(radius_m) / z
    return rx, ry


def _edt_normalized(clear_mask, K, border_as_blocked=False, pad_px=1):
    """Distance to blocked pixels in normalized-image coordinates.

    scipy.ndimage.distance_transform_edt supports anisotropic sampling.
    Using sampling=(1/fy, 1/fx) makes one horizontal/vertical pixel correspond
    to its normalized camera-plane distance.  Multiplying this distance by the
    candidate camera-Z depth approximates local metric ground/surface distance.
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except Exception as e:
        raise ImportError(
            "metric footprint filtering requires scipy "
            "(scipy.ndimage.distance_transform_edt)."
        ) from e

    fx = abs(float(K[0,0]))
    fy = abs(float(K[1,1]))
    src = clear_mask.astype(bool)

    if border_as_blocked:
        # A one-pixel zero border makes the image/FOV boundary a blocked source.
        padded = np.pad(src, 1, mode="constant", constant_values=False)
        dist = distance_transform_edt(
            padded,
            sampling=(1.0/max(fy,1e-9), 1.0/max(fx,1e-9)),
        )[1:-1,1:-1]
        return dist.astype(np.float32)

    # If the FOV boundary is not a physical obstacle, pad with clear pixels.
    # pad_px only has to exceed the largest projected radius we will query.
    pad_px = max(1, int(pad_px))
    padded = np.pad(src, pad_px, mode="constant", constant_values=True)
    dist = distance_transform_edt(
        padded,
        sampling=(1.0/max(fy,1e-9), 1.0/max(fx,1e-9)),
    )
    dist = dist[pad_px:-pad_px, pad_px:-pad_px]
    return dist.astype(np.float32)


def metric_clearance_from_blocked(
    blocked,
    D,
    K,
    radius_m,
    border_as_blocked=False,
    max_radius_px=512,
):
    """Depth-aware metric clearance from blocked pixels.

    Candidate pixel p is clear when:
        Z(p) * d_norm(p, nearest_blocked) >= radius_m

    where d_norm uses the camera intrinsics separately:
        dx_norm = du/fx
        dy_norm = dv/fy

    This avoids v10's max(fx,fy) circular-pixel approximation and therefore
    handles non-uniform image resizing correctly.
    """
    if radius_m <= 0:
        return np.ones_like(blocked, dtype=bool)

    rx, ry = projected_radii_xy(D, K, radius_m)
    max_needed = 1
    v = D > 0
    if np.any(v):
        max_needed = int(np.ceil(max(
            float(np.max(rx[v])),
            float(np.max(ry[v])),
        ))) + 2
    max_needed = min(max_needed, int(max_radius_px))

    dist_norm = _edt_normalized(
        ~blocked,
        K,
        border_as_blocked=border_as_blocked,
        pad_px=max_needed,
    )

    metric_dist = dist_norm * np.maximum(D, 0.0)

    # Treat raster pixels as finite cells rather than zero-area point samples.
    # This is the metric equivalent of v10's "+0.5 pixel" guard, while still
    # respecting fx/fy separately.  Using the larger local pixel footprint is
    # conservative and prevents sub-UAV safe islands from surviving merely
    # because distance was measured between pixel centers.
    fx = abs(float(K[0,0]))
    fy = abs(float(K[1,1]))
    half_pixel_m = (
        0.5 * np.maximum(D, 0.0) / max(min(fx, fy), 1e-9)
    )

    clear = (
        (D > 0)
        & (metric_dist >= float(radius_m) + half_pixel_m)
    )
    return clear


def _ellipse_kernel(rx_px, ry_px):
    rx = max(int(rx_px), 0)
    ry = max(int(ry_px), 0)
    if rx == 0 and ry == 0:
        return np.ones((1,1), np.uint8)
    return cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2*rx + 1, 2*ry + 1),
    )


def reconstruct_landable_region(
    feasible_centers,
    buffered_safe,
    D,
    K,
    uav_radius_m,
    max_radius_px=512,
):
    """Expand temporary feasible centers back to a landable surface region.

    Final SLZ semantics are a *landable region*, not only center points.

    Each surviving center is expanded by a depth-aware rasterized UAV footprint:
        rx = fx * R_uav / Z(center)
        ry = fy * R_uav / Z(center)

    Centers with the same rounded (rx, ry) are dilated together for efficiency.
    The union is intersected with ``buffered_safe``, so reconstruction can never
    re-introduce known unsafe or invalid pixels.
    """
    if uav_radius_m <= 0:
        return feasible_centers.copy()

    rx_f, ry_f = projected_radii_xy(D, K, uav_radius_m)
    rx_i = np.rint(rx_f).astype(np.int32)
    ry_i = np.rint(ry_f).astype(np.int32)
    rx_i = np.clip(rx_i, 0, int(max_radius_px))
    ry_i = np.clip(ry_i, 0, int(max_radius_px))

    region = np.zeros_like(feasible_centers, dtype=bool)
    pairs = np.stack(
        [rx_i[feasible_centers], ry_i[feasible_centers]],
        axis=1,
    )
    if len(pairs) == 0:
        return region

    for rx, ry in np.unique(pairs, axis=0):
        group = (
            feasible_centers
            & (rx_i == int(rx))
            & (ry_i == int(ry))
        )
        kernel = _ellipse_kernel(int(rx), int(ry))
        expanded = cv2.dilate(
            group.astype(np.uint8),
            kernel,
            iterations=1,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        ) > 0
        region |= expanded

    return region & buffered_safe
