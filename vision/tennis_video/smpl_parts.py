from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class SmplPartsMeta:
    parts: list[str]
    palette: list[list[float]]


def _load_smpl_model(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        return {k: data[k] for k in data.files}
    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as f:
            try:
                obj = pickle.load(f)
            except Exception:
                f.seek(0)
                obj = pickle.load(f, encoding="latin1")
        if not isinstance(obj, dict):
            raise ValueError("Expected SMPL pickle to contain a dict.")
        return obj
    raise ValueError("Unsupported SMPL model format; expected .pkl or .npz.")


def load_smpl_model_dict(path: Path) -> dict[str, Any]:
    return _load_smpl_model(path)


def load_smpl_faces(path: Path) -> np.ndarray:
    """
    Load SMPL faces from a model file.

    Common keys:
      - 'f' (SMPL pickles)
      - 'faces' / 'triangles' (some exports)
    """
    model = _load_smpl_model(path)
    faces = None
    for key in ("f", "faces", "triangles"):
        if key in model:
            faces = np.asarray(model[key], dtype=int)
            break
    if faces is None:
        raise ValueError("SMPL model is missing faces (expected key 'f' or 'faces').")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected faces shape (F,3); got {faces.shape}.")
    return faces


COARSE10_PARTS: list[str] = [
    "torso",
    "head",
    "upper_arm_l",
    "forearm_hand_l",
    "upper_arm_r",
    "forearm_hand_r",
    "thigh_l",
    "calf_foot_l",
    "thigh_r",
    "calf_foot_r",
]

COARSE10_PALETTE: list[list[float]] = [
    [0.49, 0.36, 1.00],  # torso (purple)
    [1.00, 0.86, 0.34],  # head (yellow)
    [0.35, 0.65, 1.00],  # upper arm L (blue)
    [0.00, 0.83, 1.00],  # forearm/hand L (cyan)
    [1.00, 0.56, 0.29],  # upper arm R (orange)
    [1.00, 0.36, 0.48],  # forearm/hand R (pink/red)
    [0.20, 0.83, 0.58],  # thigh L (green)
    [0.68, 0.93, 0.25],  # calf/foot L (lime)
    [0.36, 0.55, 1.00],  # thigh R (indigo)
    [0.95, 0.72, 1.00],  # calf/foot R (lavender)
]


def _joint_to_coarse10_part(j: int) -> int:
    # SMPL 24-joint ordering (common SMPL models):
    # 0 pelvis
    # 1 left_hip, 2 right_hip
    # 3 spine1
    # 4 left_knee, 5 right_knee
    # 6 spine2
    # 7 left_ankle, 8 right_ankle
    # 9 spine3
    # 10 left_foot, 11 right_foot
    # 12 neck
    # 13 left_collar, 14 right_collar
    # 15 head
    # 16 left_shoulder, 17 right_shoulder
    # 18 left_elbow, 19 right_elbow
    # 20 left_wrist, 21 right_wrist
    # 22 left_hand, 23 right_hand
    if j in {0, 3, 6, 9, 12, 13, 14}:
        return 0  # torso
    if j in {15}:
        return 1  # head
    if j in {16}:
        return 2  # upper arm L
    if j in {18, 20, 22}:
        return 3  # forearm/hand L
    if j in {17}:
        return 4  # upper arm R
    if j in {19, 21, 23}:
        return 5  # forearm/hand R
    if j in {1, 4}:
        return 6  # thigh L
    if j in {7, 10}:
        return 7  # calf/foot L
    if j in {2, 5}:
        return 8  # thigh R
    if j in {8, 11}:
        return 9  # calf/foot R
    return 0


def export_vertex_parts(
    smpl_model_path: Path,
    *,
    mode: Literal["coarse10"] = "coarse10",
) -> tuple[np.ndarray, SmplPartsMeta]:
    model = _load_smpl_model(smpl_model_path)
    if "weights" not in model:
        raise ValueError("SMPL model is missing 'weights' (V,24).")
    weights = np.asarray(model["weights"], dtype=float)
    if weights.ndim != 2 or weights.shape[1] != 24:
        raise ValueError(f"Expected weights shape (V,24); got {weights.shape}.")

    joint_idx = np.argmax(weights, axis=1).astype(int)

    if mode == "coarse10":
        labels = np.array([_joint_to_coarse10_part(int(j)) for j in joint_idx], dtype=np.uint16)
        meta = SmplPartsMeta(parts=COARSE10_PARTS, palette=COARSE10_PALETTE)
        return labels, meta

    raise ValueError(f"Unsupported mode: {mode}")


def write_vertex_parts_assets(
    smpl_model_path: Path,
    *,
    out_labels: Path,
    out_meta: Path,
    mode: Literal["coarse10"] = "coarse10",
) -> tuple[Path, Path]:
    labels, meta = export_vertex_parts(smpl_model_path, mode=mode)
    out_labels.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_labels, labels)
    out_meta.write_text(
        json.dumps({"parts": meta.parts, "palette": meta.palette}, indent=2), encoding="utf-8"
    )
    return out_labels, out_meta
