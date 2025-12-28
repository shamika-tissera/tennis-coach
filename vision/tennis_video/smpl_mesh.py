from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .pose_processing import (
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ANKLE,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)
from .smpl_parts import load_smpl_faces, load_smpl_model_dict


@dataclass(frozen=True)
class SmplModel:
    vertices: np.ndarray  # (V,3) float32
    weights: np.ndarray  # (V,24) float32
    joints: np.ndarray  # (24,3) float32
    faces: np.ndarray  # (F,3) int32


def _as_dense_regressor(reg) -> np.ndarray:
    if hasattr(reg, "toarray"):
        return np.asarray(reg.toarray(), dtype=float)
    return np.asarray(reg, dtype=float)


def load_smpl_model(path: Path) -> SmplModel:
    model = load_smpl_model_dict(path)
    if "v_template" not in model:
        raise ValueError("SMPL model missing 'v_template'.")
    if "weights" not in model:
        raise ValueError("SMPL model missing 'weights'.")
    if "J_regressor" not in model:
        raise ValueError("SMPL model missing 'J_regressor'.")

    v_template = np.asarray(model["v_template"], dtype=np.float32)
    weights = np.asarray(model["weights"], dtype=np.float32)
    j_reg = _as_dense_regressor(model["J_regressor"])
    if v_template.ndim != 2 or v_template.shape[1] != 3:
        raise ValueError(f"Expected v_template shape (V,3); got {v_template.shape}.")
    if weights.ndim != 2 or weights.shape[1] != 24 or weights.shape[0] != v_template.shape[0]:
        raise ValueError(f"Expected weights shape (V,24); got {weights.shape}.")
    if j_reg.ndim != 2 or j_reg.shape[0] != 24 or j_reg.shape[1] != v_template.shape[0]:
        raise ValueError(f"Expected J_regressor shape (24,V); got {j_reg.shape}.")

    joints = (j_reg @ v_template).astype(np.float32)
    faces = load_smpl_faces(path).astype(np.int32)
    return SmplModel(vertices=v_template, weights=weights, joints=joints, faces=faces)


def _rotation_from_to(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    eps = 1e-8
    a = np.asarray(a, dtype=float).reshape(3)
    b = np.asarray(b, dtype=float).reshape(3)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return np.eye(3, dtype=np.float32)
    a = a / na
    b = b / nb
    v = np.cross(a, b)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    s = float(np.linalg.norm(v))
    if s < eps:
        if c > 0.0:
            return np.eye(3, dtype=np.float32)
        axis = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=float)
        v2 = np.cross(a, axis)
        v2n = float(np.linalg.norm(v2))
        if v2n < eps:
            return -np.eye(3, dtype=np.float32)
        v2 = v2 / v2n
        r = -np.eye(3, dtype=float) + 2.0 * np.outer(v2, v2)
        return r.astype(np.float32)

    vx = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=float
    )
    r = np.eye(3, dtype=float) + vx + (vx @ vx) * ((1.0 - c) / (s * s))
    return r.astype(np.float32)


def _coco17_to_smpl24_joints(coco: np.ndarray) -> np.ndarray:
    """
    Map COCO-17 joints (in our normalized 3D space) to SMPL 24 joints.

    Expects coco in COCO ordering with indices from tennis_video.pose_processing.
    """
    coco = np.asarray(coco, dtype=np.float32)
    if coco.shape != (17, 3):
        raise ValueError(f"Expected coco joints shape (17,3); got {coco.shape}.")

    left_hip = coco[LEFT_HIP]
    right_hip = coco[RIGHT_HIP]
    pelvis = 0.5 * (left_hip + right_hip)
    left_sh = coco[LEFT_SHOULDER]
    right_sh = coco[RIGHT_SHOULDER]
    shoulder_mid = 0.5 * (left_sh + right_sh)
    trunk = shoulder_mid - pelvis

    spine1 = pelvis + 0.30 * trunk
    spine2 = pelvis + 0.60 * trunk
    spine3 = pelvis + 0.85 * trunk

    # COCO "nose" is index 0.
    head = coco[0]
    neck = shoulder_mid

    j = np.zeros((24, 3), dtype=np.float32)
    j[0] = pelvis
    j[1] = left_hip
    j[2] = right_hip
    j[3] = spine1
    j[4] = coco[LEFT_KNEE]
    j[5] = coco[RIGHT_KNEE]
    j[6] = spine2
    j[7] = coco[LEFT_ANKLE]
    j[8] = coco[RIGHT_ANKLE]
    j[9] = spine3
    j[10] = coco[LEFT_ANKLE]  # foot approx
    j[11] = coco[RIGHT_ANKLE]  # foot approx
    j[12] = neck
    j[13] = left_sh  # collar approx
    j[14] = right_sh  # collar approx
    j[15] = head
    j[16] = left_sh
    j[17] = right_sh
    j[18] = coco[LEFT_ELBOW]
    j[19] = coco[RIGHT_ELBOW]
    j[20] = coco[LEFT_WRIST]
    j[21] = coco[RIGHT_WRIST]
    j[22] = coco[LEFT_WRIST]  # hand approx
    j[23] = coco[RIGHT_WRIST]  # hand approx
    return j


def _child_index_for_smpl_joint() -> list[Optional[int]]:
    child: list[Optional[int]] = [None] * 24
    child[0] = 3
    child[3] = 6
    child[6] = 9
    child[9] = 12
    child[12] = 15

    child[1] = 4
    child[4] = 7
    child[7] = 10

    child[2] = 5
    child[5] = 8
    child[8] = 11

    child[16] = 18
    child[18] = 20
    child[20] = 22

    child[17] = 19
    child[19] = 21
    child[21] = 23
    return child


def _estimate_scale(rest_joints: np.ndarray, target_coco: np.ndarray) -> float:
    rest = np.asarray(rest_joints, dtype=float)
    coco = np.asarray(target_coco, dtype=float)
    try:
        target = float(np.linalg.norm(coco[LEFT_SHOULDER] - coco[RIGHT_SHOULDER]))
    except Exception:
        target = 1.0
    rest_dist = float(np.linalg.norm(rest[16] - rest[17]))
    if rest_dist <= 1e-8:
        return 1.0
    if target <= 1e-8:
        target = 1.0
    return float(target / rest_dist)


def smpl_vertices_from_coco17(
    coco_joints: np.ndarray,
    smpl: SmplModel,
    *,
    flip_y: bool = True,
) -> np.ndarray:
    """
    Generate SMPL mesh vertices from a (T,17,3) COCO joint sequence.

    This is a lightweight rigging approach (LBS using SMPL skinning weights) driven by
    per-joint direction alignment. It is not MMHuman3D-quality fitting, but produces
    a dense SMPL surface you can render and compare.
    """
    coco_joints = np.asarray(coco_joints, dtype=np.float32)
    if coco_joints.ndim != 3 or coco_joints.shape[1:] != (17, 3):
        raise ValueError(f"Expected coco_joints shape (T,17,3); got {coco_joints.shape}.")

    t = int(coco_joints.shape[0])
    if t <= 0:
        raise ValueError("No frames in coco_joints.")

    coco0 = coco_joints[0].copy()
    if flip_y:
        coco0[:, 1] *= -1.0
    scale = _estimate_scale(smpl.joints, coco0)

    v_rest = smpl.vertices.astype(np.float32) * float(scale)
    j_rest = smpl.joints.astype(np.float32) * float(scale)
    w = smpl.weights.astype(np.float32)
    child = _child_index_for_smpl_joint()

    out = np.empty((t, v_rest.shape[0], 3), dtype=np.float32)

    for f in range(t):
        coco = coco_joints[f].copy()
        if flip_y:
            coco[:, 1] *= -1.0
        # Center by pelvis (hip midpoint) to keep the mesh stable in the viewer.
        pelvis = 0.5 * (coco[LEFT_HIP] + coco[RIGHT_HIP])
        coco = coco - pelvis[None, :]

        j_tgt = _coco17_to_smpl24_joints(coco)

        r_all = np.zeros((24, 3, 3), dtype=np.float32)
        t_all = np.zeros((24, 3), dtype=np.float32)
        for j in range(24):
            c_idx = child[j]
            if c_idx is None:
                r = np.eye(3, dtype=np.float32)
            else:
                rest_dir = j_rest[c_idx] - j_rest[j]
                tgt_dir = j_tgt[c_idx] - j_tgt[j]
                r = _rotation_from_to(rest_dir, tgt_dir)
            r_all[j] = r
            t_all[j] = j_tgt[j] - (r @ j_rest[j])

        # Per-joint transform then weight-sum.
        v_acc = np.zeros((v_rest.shape[0], 3), dtype=np.float32)
        for j in range(24):
            vj = v_rest @ r_all[j].T + t_all[j][None, :]
            v_acc += w[:, j : j + 1] * vj
        out[f] = v_acc

    return out

