from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .smpl_parts import load_smpl_faces


@dataclass(frozen=True)
class AttachMeshResult:
    session_path: Path
    user_vertices_path: Path
    expert_vertices_path: Optional[Path]
    vertex_labels_path: Optional[Path]


def _load_vertices_and_faces(npz_path: Path) -> tuple[np.ndarray, Optional[np.ndarray]]:
    data = np.load(npz_path, allow_pickle=False)
    vertices = None
    for key in ("vertices", "verts", "pred_vertices"):
        if key in data:
            vertices = np.asarray(data[key], dtype=float)
            break
    if vertices is None:
        raise ValueError(
            f"Missing vertices in {npz_path} (expected one of: vertices/verts/pred_vertices)"
        )

    faces = None
    for key in ("faces", "triangles"):
        if key in data:
            faces = np.asarray(data[key], dtype=int)
            break

    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError(f"Expected vertices shape (T,V,3); got {vertices.shape} in {npz_path}")
    if faces is not None:
        if faces.ndim != 2 or faces.shape[-1] != 3:
            raise ValueError(f"Expected faces shape (F,3); got {faces.shape} in {npz_path}")
    return vertices, faces


def _resample_vertices_nearest(vertices: np.ndarray, target_frames: int) -> np.ndarray:
    if target_frames <= 0:
        raise ValueError("target_frames must be > 0")
    if vertices.shape[0] == target_frames:
        return vertices
    idx = np.linspace(0, vertices.shape[0] - 1, num=target_frames)
    idx_i = np.clip(np.round(idx).astype(int), 0, vertices.shape[0] - 1)
    return vertices[idx_i]


def _load_labels(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        labels = np.load(path)
    else:
        data = np.load(path, allow_pickle=False)
        if "labels" not in data:
            raise ValueError("NPZ must contain array 'labels' with shape (V,).")
        labels = data["labels"]
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(f"Expected labels shape (V,); got {labels.shape}")
    return labels


def _normalize_palette(palette: Any) -> list[list[float]]:
    if not isinstance(palette, list) or not palette:
        raise ValueError("palette must be a non-empty list of [r,g,b].")
    out: list[list[float]] = []
    for c in palette:
        if not isinstance(c, list) or len(c) != 3:
            raise ValueError("palette entries must be [r,g,b].")
        r, g, b = (float(c[0]), float(c[1]), float(c[2]))
        if max(r, g, b) > 1.0:
            r, g, b = r / 255.0, g / 255.0, b / 255.0
        out.append([r, g, b])
    return out


def _auto_palette(n: int) -> list[list[float]]:
    if n <= 0:
        return []
    out: list[list[float]] = []
    phi = 0.618033988749895
    h = 0.0
    for _ in range(n):
        h = (h + phi) % 1.0
        s = 0.65
        v = 0.95
        i = int(h * 6.0)
        f = h * 6.0 - i
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)
        i = i % 6
        if i == 0:
            r, g, b = v, t, p
        elif i == 1:
            r, g, b = q, v, p
        elif i == 2:
            r, g, b = p, v, t
        elif i == 3:
            r, g, b = p, q, v
        elif i == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q
        out.append([float(r), float(g), float(b)])
    return out


def attach_mesh_npz_to_session(
    *,
    session_path: Path,
    user_mesh_npz: Path,
    expert_mesh_npz: Optional[Path] = None,
    user_vertices_out: str = "user_smpl_vertices.f32",
    expert_vertices_out: str = "expert_smpl_vertices.f32",
    vertex_labels_npy: Optional[Path] = None,
    parts_meta_json: Optional[Path] = None,
    labels_out: str = "smpl_vertex_labels.u16",
    smpl_model_path: Optional[Path] = None,
) -> AttachMeshResult:
    """
    Attach SMPL-style mesh data to an existing session.json.

    The mesh is stored as binary float32 buffers inside the session directory for fast loading.
    """
    if not session_path.exists():
        raise FileNotFoundError(f"Session not found: {session_path}")

    session_dir = session_path.parent
    session = json.loads(session_path.read_text(encoding="utf-8"))
    timeline = session.get("timeline") or {}
    frame_count = len(timeline.get("t") or [])
    if frame_count <= 0:
        raise RuntimeError("Session timeline is missing or empty; cannot determine frame_count.")

    user_vertices, user_faces = _load_vertices_and_faces(user_mesh_npz)
    expert_vertices = None
    expert_faces = None
    if expert_mesh_npz:
        expert_vertices, expert_faces = _load_vertices_and_faces(expert_mesh_npz)

    faces = user_faces if user_faces is not None else expert_faces
    if faces is None and smpl_model_path is not None:
        faces = load_smpl_faces(smpl_model_path)
    if faces is None:
        raise RuntimeError(
            "No faces found in mesh NPZ outputs (and no SMPL model provided); viewer needs faces (F,3)."
        )

    user_vertices_r = _resample_vertices_nearest(user_vertices, frame_count).astype(np.float32)
    user_vertices_path = session_dir / user_vertices_out
    user_vertices_path.write_bytes(user_vertices_r.tobytes())

    expert_vertices_path = None
    if expert_vertices is not None:
        expert_vertices_r = _resample_vertices_nearest(expert_vertices, frame_count).astype(np.float32)
        expert_vertices_path = session_dir / expert_vertices_out
        expert_vertices_path.write_bytes(expert_vertices_r.tobytes())

    mesh: dict[str, Any] = {
        "type": "smpl",
        "vertex_count": int(user_vertices.shape[1]),
        "face_count": int(faces.shape[0]),
        "frame_count": int(frame_count),
        "vertices_dtype": "float32",
        "vertices_format": "f32le",
        "faces": faces.astype(int).tolist(),
    }

    labels_path_out: Optional[Path] = None
    if vertex_labels_npy:
        labels = _load_labels(vertex_labels_npy)
        if labels.shape[0] != user_vertices.shape[1]:
            raise RuntimeError(
                f"Label vertex count mismatch: labels={labels.shape[0]} vs mesh={user_vertices.shape[1]}"
            )
        labels_u16 = labels.astype(np.uint16, copy=False)
        labels_path_out = session_dir / labels_out
        labels_path_out.write_bytes(labels_u16.tobytes())

        parts: list[str] = []
        palette: list[list[float]] = []
        if parts_meta_json:
            meta = json.loads(parts_meta_json.read_text(encoding="utf-8"))
            parts = list(meta.get("parts") or [])
            palette = _normalize_palette(meta.get("palette"))
        if not palette:
            n = int(labels_u16.max()) + 1 if labels_u16.size else 0
            palette = _auto_palette(n)
        if not parts:
            parts = [f"part_{i}" for i in range(len(palette))]

        mesh["vertex_labels_path"] = labels_out
        mesh["labels_dtype"] = "uint16"
        mesh["labels_format"] = "u16le"
        mesh["parts"] = parts
        mesh["parts_palette"] = palette

    session["mesh"] = mesh
    session.setdefault("user", {})["mesh_vertices_path"] = user_vertices_out
    session.setdefault("expert", {})["mesh_vertices_path"] = (
        expert_vertices_out if expert_vertices_path else None
    )

    session_path.write_text(json.dumps(session, indent=2), encoding="utf-8")
    return AttachMeshResult(
        session_path=session_path,
        user_vertices_path=user_vertices_path,
        expert_vertices_path=expert_vertices_path,
        vertex_labels_path=labels_path_out,
    )

