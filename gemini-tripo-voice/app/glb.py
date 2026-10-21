"""Extract part information from a segmented .glb without downloading the whole file.

Tripo's segmentation task returns ONE GLB (output.model_url) in which every part is its own
node/mesh. The task response itself carries no part list, so we read it from the GLB:

* A GLB starts with a 12-byte header, then a JSON chunk (the scene description). We stream only
  those first bytes and stop, so a 50 MB model costs a few KB of bandwidth.
* From the JSON we collect every mesh node, its name, and its world-space bounding box
  (computed from the POSITION accessor min/max + node transforms).
* v2 segmentation ("semantic labeling") is expected to give meaningful node names. If the names
  turn out to be generic ("mesh_3", "tripo_node_ab12..."), we report parts as "Part 1..N" with
  position/size hints and flag `labels_are_semantic=False` so Gemini says it is inferring.
"""
from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass
from typing import Optional

import httpx

MAX_PARTS_REPORTED = 20
_GENERIC_TOKENS = {
    "tripo", "mesh", "node", "object", "obj", "geometry", "geo", "part", "shape", "group",
    "scene", "root", "default", "primitive", "model", "segment", "seg", "component", "unnamed",
}


# --------------------------------------------------------------------------- reading
async def read_glb_json(url: str, *, timeout: float = 30.0) -> dict:
    """Stream just the JSON chunk of a GLB file and return it parsed."""
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as http:
        async with http.stream("GET", url) as resp:
            resp.raise_for_status()
            buf = bytearray()
            needed: Optional[int] = None
            async for chunk in resp.aiter_bytes(65536):
                buf.extend(chunk)
                if needed is None and len(buf) >= 20:
                    magic, _version, _length = struct.unpack("<4sII", bytes(buf[:12]))
                    json_len, json_type = struct.unpack("<I4s", bytes(buf[12:20]))
                    if magic != b"glTF" or json_type != b"JSON":
                        raise ValueError("Not a GLB file (bad header)")
                    needed = 20 + json_len
                if needed is not None and len(buf) >= needed:
                    break
    if needed is None or len(buf) < needed:
        raise ValueError("GLB ended before the JSON chunk was complete")
    return json.loads(bytes(buf[20:needed]).decode("utf-8"))


# --------------------------------------------------------------------------- math helpers
def _trs_matrix(node: dict) -> list[float]:
    """Node transform as a column-major 4x4 matrix (glTF convention)."""
    if "matrix" in node:
        return [float(v) for v in node["matrix"]]
    tx, ty, tz = node.get("translation", [0, 0, 0])
    qx, qy, qz, qw = node.get("rotation", [0, 0, 0, 1])
    sx, sy, sz = node.get("scale", [1, 1, 1])
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    r00, r01, r02 = 1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)
    r10, r11, r12 = 2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)
    r20, r21, r22 = 2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)
    return [
        r00 * sx, r10 * sx, r20 * sx, 0,
        r01 * sy, r11 * sy, r21 * sy, 0,
        r02 * sz, r12 * sz, r22 * sz, 0,
        tx, ty, tz, 1,
    ]


def _mat_mul(a: list[float], b: list[float]) -> list[float]:
    """a @ b for column-major 4x4 matrices."""
    out = [0.0] * 16
    for c in range(4):
        for r in range(4):
            out[c * 4 + r] = sum(a[k * 4 + r] * b[c * 4 + k] for k in range(4))
    return out


_IDENTITY = [1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0]


def _world_bounds(lo: list[float], hi: list[float], m: list[float]) -> tuple[list[float], list[float]]:
    xs, ys, zs = [], [], []
    for x in (lo[0], hi[0]):
        for y in (lo[1], hi[1]):
            for z in (lo[2], hi[2]):
                xs.append(m[0] * x + m[4] * y + m[8] * z + m[12])
                ys.append(m[1] * x + m[5] * y + m[9] * z + m[13])
                zs.append(m[2] * x + m[6] * y + m[10] * z + m[14])
    return [min(xs), min(ys), min(zs)], [max(xs), max(ys), max(zs)]


def _mesh_local_bounds(gltf: dict, mesh_index: int) -> Optional[tuple[list[float], list[float]]]:
    accessors = gltf.get("accessors", [])
    lo = hi = None
    for prim in gltf["meshes"][mesh_index].get("primitives", []):
        idx = prim.get("attributes", {}).get("POSITION")
        if idx is None or idx >= len(accessors):
            continue
        acc = accessors[idx]
        if "min" not in acc or "max" not in acc:
            continue
        if lo is None:
            lo, hi = list(acc["min"]), list(acc["max"])
        else:
            lo = [min(a, b) for a, b in zip(lo, acc["min"])]
            hi = [max(a, b) for a, b in zip(hi, acc["max"])]
    return (lo, hi) if lo is not None else None


# --------------------------------------------------------------------------- naming helpers
def _is_generic(name: str) -> bool:
    tokens = [t for t in re.split(r"[\s_\-.]+", name.lower()) if t]
    meaningful = [
        t for t in tokens
        if not t.isdigit() and t not in _GENERIC_TOKENS and not re.fullmatch(r"[0-9a-f]{6,}", t)
    ]
    return not meaningful


def _base_name(name: str) -> str:
    """'Wheel.003' / 'wheel_2' -> 'wheel' so repeated parts group together."""
    return re.sub(r"[\s_.\-]*\d+$", "", name.strip()) or name.strip()


def _display(name: str) -> str:
    return re.sub(r"[_\-]+", " ", name).strip()


def _band(value: float, lo: float, hi: float, names: tuple[str, str, str]) -> str:
    span = (hi - lo) or 1.0
    t = (value - lo) / span
    return names[0] if t < 0.33 else names[2] if t > 0.67 else names[1]


def _location(center: list[float], lo: list[float], hi: list[float]) -> str:
    """Rough, human-friendly position. Assumes glTF axes: +Y up, +Z toward the viewer."""
    bits = [
        _band(center[1], lo[1], hi[1], ("bottom", "middle", "top")),
        _band(center[0], lo[0], hi[0], ("left", "center", "right")),
        _band(center[2], lo[2], hi[2], ("back", "middle", "front")),
    ]
    interesting = [b for b in bits if b not in ("center", "middle")]
    return ", ".join(interesting) if interesting else "center of the model"


# --------------------------------------------------------------------------- main entry
def summarize_parts(gltf: dict) -> dict:
    """Turn the GLB JSON into a compact, speakable part list."""
    nodes = gltf.get("nodes", [])
    if not nodes or not gltf.get("meshes"):
        return {"labels_are_semantic": False, "total_instances": 0, "parts": [], "omitted_small_parts": 0}

    # Walk the scene graph, accumulating transforms, collecting every mesh-bearing node.
    scenes = gltf.get("scenes", [])
    if scenes:
        roots = scenes[gltf.get("scene", 0)].get("nodes", [])
    else:
        children = {c for n in nodes for c in n.get("children", [])}
        roots = [i for i in range(len(nodes)) if i not in children]

    instances: list[dict] = []
    visited: set[int] = set()

    def walk(index: int, parent: list[float]) -> None:
        if index in visited or index >= len(nodes):
            return
        visited.add(index)
        node = nodes[index]
        world = _mat_mul(parent, _trs_matrix(node))
        if "mesh" in node:
            local = _mesh_local_bounds(gltf, node["mesh"])
            if local:
                bmin, bmax = _world_bounds(local[0], local[1], world)
                name = node.get("name") or gltf["meshes"][node["mesh"]].get("name") or ""
                instances.append({"name": name, "min": bmin, "max": bmax})
        for child in node.get("children", []):
            walk(child, world)

    for root in roots:
        walk(root, _IDENTITY)

    if not instances:
        return {"labels_are_semantic": False, "total_instances": 0, "parts": [], "omitted_small_parts": 0}

    lo = [min(i["min"][a] for i in instances) for a in range(3)]
    hi = [max(i["max"][a] for i in instances) for a in range(3)]
    total_vol = max(((hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2])), 1e-12)

    semantic = [i for i in instances if i["name"] and not _is_generic(i["name"])]
    labels_are_semantic = len(semantic) >= max(1, len(instances) // 2)

    # Group repeated semantic parts ("wheel" x4); keep unlabeled ones individual.
    groups: dict[str, list[dict]] = {}
    unlabeled: list[dict] = []
    for inst in instances:
        if inst["name"] and not _is_generic(inst["name"]):
            groups.setdefault(_base_name(inst["name"]).lower(), []).append(inst)
        else:
            unlabeled.append(inst)

    def build(label: str, members: list[dict]) -> dict:
        gmin = [min(m["min"][a] for m in members) for a in range(3)]
        gmax = [max(m["max"][a] for m in members) for a in range(3)]
        center = [(gmin[a] + gmax[a]) / 2 for a in range(3)]
        vol = sum(
            max((m["max"][0] - m["min"][0]) * (m["max"][1] - m["min"][1]) * (m["max"][2] - m["min"][2]), 0.0)
            for m in members
        )
        frac = vol / total_vol
        return {
            "name": label,
            "count": len(members),
            "location": _location(center, lo, hi) + (" (copies spread across the model)" if len(members) > 1 else ""),
            "relative_size": "large" if frac > 0.25 else "medium" if frac > 0.05 else "small",
            "_vol": frac,
        }

    parts = [build(_display(_base_name(ms[0]["name"])), ms) for ms in groups.values()]
    unlabeled.sort(
        key=lambda m: -((m["max"][0] - m["min"][0]) * (m["max"][1] - m["min"][1]) * (m["max"][2] - m["min"][2]))
    )
    for n, inst in enumerate(unlabeled, start=1):
        parts.append(build(f"Unlabeled part {n}" if semantic else f"Part {n}", [inst]))

    parts.sort(key=lambda p: -p["_vol"])
    omitted = max(len(parts) - MAX_PARTS_REPORTED, 0)
    parts = parts[:MAX_PARTS_REPORTED]
    for p in parts:
        p.pop("_vol", None)

    return {
        "labels_are_semantic": labels_are_semantic,
        "total_instances": len(instances),
        "parts": parts,
        "omitted_small_parts": omitted,
    }


async def extract_parts(model_url: str) -> dict:
    """Download the GLB header/JSON and summarize its parts."""
    return summarize_parts(await read_glb_json(model_url))
