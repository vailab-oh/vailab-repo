"""Terra mesh discovery, binary little-endian PLY parsing and an on-disk (V, F) cache."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .._util import log, mkdir


def mesh_group_from_scene(scene: str) -> str:
    """
    AMtown01/02/03 -> AMtown
    AMvalley01/02/03 -> AMvalley
    HKisland_GNSS01/02/03 -> HKisland_GNSS
    """
    if scene.endswith("_Evening"):
        return scene
    m = re.match(r"^(.*?)(0[1-3])$", scene)
    if m:
        return m.group(1)
    return scene


def find_meshes(mesh_root: Optional[Path], interval: str, scene: str) -> List[Path]:
    """
    Prefer Terra raw block meshes.

    Single-block group:
        {group}/terra_ply/Block/Block.ply

    Multi-block group, e.g. AMtown:
        {group}/terra_ply/BlockAB/BlockAB.ply
        {group}/terra_ply/BlockAY/BlockAY.ply
        {group}/terra_ply/BlockXB/BlockXB.ply
        {group}/terra_ply/BlockXY/BlockXY.ply

    Fallback:
        {group}/Mesh.ply
    """
    if mesh_root is None or not mesh_root.exists():
        return []

    group = mesh_group_from_scene(scene)
    terra_roots = [
        mesh_root / interval / "terra_3dmap_pointcloud_mesh",
        mesh_root / "terra_3dmap_pointcloud_mesh",
        mesh_root,
    ]

    group_dirs = []
    for tr in terra_roots:
        candidates = [tr / group]
        # If mesh_root itself is the group folder, also consider it.
        if tr.name == group or (tr / "terra_ply").exists():
            candidates.append(tr)
        for gd in candidates:
            if gd.exists() and gd not in group_dirs:
                group_dirs.append(gd)

    for gd in group_dirs:
        # 1) Single-block standard path.
        for p in [
            gd / "terra_ply" / "Block" / "Block.ply",
            gd / "terra_ply" / "Block" / "block.ply",
        ]:
            if p.exists():
                return [p]

        # 2) Multi-block terra_ply path.
        terra = gd / "terra_ply"
        if terra.exists():
            block_plys = []
            for sub in sorted([x for x in terra.iterdir() if x.is_dir()]):
                rep = sub / f"{sub.name}.ply"
                if rep.exists():
                    block_plys.append(rep)
            if block_plys:
                return block_plys

        # 3) Fallback to merged mesh.
        for p in [gd / "Mesh.ply", gd / "mesh.ply"]:
            if p.exists():
                return [p]

        # 4) Last fallback.
        any_blocks = sorted(gd.rglob("Block*.ply"))
        if any_blocks:
            reps = [p for p in any_blocks if p.parent.name == p.stem]
            return reps if reps else any_blocks[:1]
        any_mesh = sorted(gd.rglob("Mesh.ply"))
        if any_mesh:
            return [any_mesh[0]]

    return []


def _ply_scalar_dtype(ply_type: str):
    table = {
        "char": "i1", "int8": "i1",
        "uchar": "u1", "uint8": "u1",
        "short": "<i2", "int16": "<i2",
        "ushort": "<u2", "uint16": "<u2",
        "int": "<i4", "int32": "<i4",
        "uint": "<u4", "uint32": "<u4",
        "float": "<f4", "float32": "<f4",
        "double": "<f8", "float64": "<f8",
    }
    if ply_type not in table:
        raise ValueError(f"Unsupported PLY scalar type: {ply_type}")
    return np.dtype(table[ply_type])


def load_terra_ply(path: Path, cache_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robust loader for DJI Terra / VCGLIB binary_little_endian PLY.

    Supports both of these observed vertex formats:
      1) x y z
      2) x y z nx ny nz flags

    Supports face properties such as:
      property list uchar int vertex_indices
      property int flags
      property list uchar float texcoord
      property int texnumber

    Returns:
      V: (N, 3) float32 xyz
      F: (M, 3) int32 triangle indices

    Important:
      Some Terra PLY files include vertex normals and flags. Reading only xyz will
      misalign the file pointer and produce invalid face indices, e.g.
      "index 4139563 is out of bounds for axis 0 with size 1247145".
    """
    mkdir(cache_dir)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(path.resolve()))
    v_cache = cache_dir / f"{safe}.verts.npy"
    f_cache = cache_dir / f"{safe}.faces.npy"

    if v_cache.exists() and f_cache.exists():
        V = np.load(v_cache, mmap_mode="r")
        F = np.load(f_cache, mmap_mode="r")
        # Cache sanity check. If old buggy cache exists, ignore it and rebuild.
        if F.size == 0 or (F.min() >= 0 and F.max() < V.shape[0]):
            return V, F
        print(f"[WARN] invalid PLY cache detected, rebuilding: {path}", flush=True)
        try:
            v_cache.unlink()
            f_cache.unlink()
        except Exception:
            pass

    log(f"[PLY] loading {path}")

    with open(path, "rb") as f:
        fmt = ""
        n_vertices = None
        n_faces = None
        current_element = None
        vertex_props = []
        face_props = []

        while True:
            line_b = f.readline()
            if not line_b:
                raise RuntimeError("Unexpected EOF while reading PLY header")
            line = line_b.decode("latin1").strip()
            if line.startswith("format"):
                fmt = line
            elif line.startswith("element "):
                parts = line.split()
                current_element = parts[1]
                if current_element == "vertex":
                    n_vertices = int(parts[2])
                elif current_element == "face":
                    n_faces = int(parts[2])
            elif line.startswith("property "):
                parts = line.split()
                if current_element == "vertex":
                    # property float x
                    if parts[1] == "list":
                        raise RuntimeError("List property in vertex element is not supported")
                    vertex_props.append((parts[2], parts[1]))
                elif current_element == "face":
                    # property list uchar int vertex_indices
                    # property int flags
                    if parts[1] == "list":
                        face_props.append(("list", parts[4], parts[2], parts[3]))
                    else:
                        face_props.append(("scalar", parts[2], parts[1], None))
            elif line == "end_header":
                break

        if "binary_little_endian" not in fmt:
            raise RuntimeError(f"Only binary_little_endian PLY is supported. Got: {fmt}")
        if n_vertices is None or n_faces is None:
            raise RuntimeError(f"Failed to parse vertex/face count from {path}")

        # Read all vertex scalar properties with a structured dtype, then keep x,y,z.
        v_dtype = np.dtype([(name, _ply_scalar_dtype(tp)) for name, tp in vertex_props])
        verts_raw = np.fromfile(f, dtype=v_dtype, count=n_vertices)
        if len(verts_raw) != n_vertices:
            raise RuntimeError(f"Read {len(verts_raw)} vertices, expected {n_vertices}")

        if not all(k in verts_raw.dtype.names for k in ["x", "y", "z"]):
            raise RuntimeError(f"PLY vertex element has no x/y/z fields: {path}")

        V = np.stack(
            [
                verts_raw["x"].astype(np.float32),
                verts_raw["y"].astype(np.float32),
                verts_raw["z"].astype(np.float32),
            ],
            axis=1,
        )

        # Fast path for observed Terra/VCGLIB triangular face layouts.
        # We assume vertex_indices count=3 and texcoord count=6, which is true for
        # Terra triangle textured meshes. The count bytes are still read.
        #
        # Build dtype in exact header order.
        dtype_fields = []
        list_counter = 0
        for prop in face_props:
            kind, name, a, b = prop
            if kind == "list":
                count_type, item_type = a, b
                count_dtype = _ply_scalar_dtype(count_type)
                item_dtype = _ply_scalar_dtype(item_type)
                if name == "vertex_indices":
                    fixed_count = 3
                elif name == "texcoord":
                    fixed_count = 6
                else:
                    raise RuntimeError(f"Unsupported face list property: {name}")
                dtype_fields.append((f"{name}_count", count_dtype))
                dtype_fields.append((name, item_dtype, (fixed_count,)))
                list_counter += 1
            else:
                scalar_type = a
                dtype_fields.append((name, _ply_scalar_dtype(scalar_type)))

        face_dtype = np.dtype(dtype_fields)
        faces_raw = np.fromfile(f, dtype=face_dtype, count=n_faces)
        if len(faces_raw) != n_faces:
            raise RuntimeError(f"Read {len(faces_raw)} faces, expected {n_faces}")

        if "vertex_indices" not in faces_raw.dtype.names:
            raise RuntimeError(f"PLY face element has no vertex_indices: {path}")

        # Verify list counts. If not triangular, fail loudly instead of silently corrupting.
        if "vertex_indices_count" in faces_raw.dtype.names:
            counts = faces_raw["vertex_indices_count"]
            bad = counts != 3
            if np.any(bad):
                raise RuntimeError(
                    f"Non-triangle faces found in {path}: "
                    f"{int(np.sum(bad))} / {len(counts)} faces have vertex count != 3"
                )
        if "texcoord_count" in faces_raw.dtype.names:
            tc = faces_raw["texcoord_count"]
            bad_tc = tc != 6
            if np.any(bad_tc):
                raise RuntimeError(
                    f"Unexpected texcoord count in {path}: "
                    f"{int(np.sum(bad_tc))} / {len(tc)} faces have texcoord count != 6"
                )

        F = np.asarray(faces_raw["vertex_indices"], dtype=np.int32)

        # Final sanity filter. This should be zero for correct parsing.
        valid = (F >= 0).all(axis=1) & (F < V.shape[0]).all(axis=1)
        bad_n = int((~valid).sum())
        if bad_n > 0:
            bad_ratio = bad_n / max(len(F), 1)
            print(
                f"[WARN] removing invalid faces from {path}: "
                f"{bad_n}/{len(F)} ({bad_ratio*100:.4f}%)",
                flush=True,
            )
            F = F[valid]

    np.save(v_cache, V.astype(np.float32))
    np.save(f_cache, F.astype(np.int32))

    return np.load(v_cache, mmap_mode="r"), np.load(f_cache, mmap_mode="r")
