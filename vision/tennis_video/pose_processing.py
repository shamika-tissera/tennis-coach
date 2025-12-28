from __future__ import annotations
import math
from typing import Optional
import numpy as np

LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


def normalize_keypoints_2d(keypoints: np.ndarray) -> Optional[np.ndarray]:
    """Centers, scales, and rotates keypoints to a shoulder-aligned frame."""
    if keypoints.shape[0] <= RIGHT_ANKLE:
        return None
    coords = np.asarray(keypoints[:, :2], dtype=float)
    scores = np.asarray(keypoints[:, 2], dtype=float)
    if np.count_nonzero(scores > 0.2) < 6:
        return None
    left_hip = coords[LEFT_HIP]
    right_hip = coords[RIGHT_HIP]
    left_shoulder = coords[LEFT_SHOULDER]
    right_shoulder = coords[RIGHT_SHOULDER]
    hip_mid = 0.5 * (left_hip + right_hip)
    shoulder_vec = right_shoulder - left_shoulder
    shoulder_dist = np.linalg.norm(shoulder_vec)
    if shoulder_dist < 1e-3:
        return None
    coords_centered = coords - hip_mid
    coords_scaled = coords_centered / shoulder_dist
    angle = math.atan2(shoulder_vec[1], shoulder_vec[0])
    cos_a = math.cos(-angle)
    sin_a = math.sin(-angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    coords_rot = coords_scaled @ rot.T
    return coords_rot


def angle_between_2d(v1: np.ndarray, v2: np.ndarray) -> float:
    """Returns angle in radians between two 2D vectors."""
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
    dot = float(np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0))
    return float(math.acos(dot))


def _angle_at_joint(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at point b formed by a-b-c."""
    return angle_between_2d(a - b, c - b)


def compute_angle_vector(coords_norm: np.ndarray) -> np.ndarray:
    """Computes a fixed vector of joint angles from normalized coords."""
    right_elbow = _angle_at_joint(
        coords_norm[RIGHT_SHOULDER], coords_norm[RIGHT_ELBOW], coords_norm[RIGHT_WRIST]
    )
    left_elbow = _angle_at_joint(
        coords_norm[LEFT_SHOULDER], coords_norm[LEFT_ELBOW], coords_norm[LEFT_WRIST]
    )
    right_knee = _angle_at_joint(
        coords_norm[RIGHT_HIP], coords_norm[RIGHT_KNEE], coords_norm[RIGHT_ANKLE]
    )
    left_knee = _angle_at_joint(
        coords_norm[LEFT_HIP], coords_norm[LEFT_KNEE], coords_norm[LEFT_ANKLE]
    )
    shoulder_mid = 0.5 * (coords_norm[LEFT_SHOULDER] + coords_norm[RIGHT_SHOULDER])
    hip_mid = 0.5 * (coords_norm[LEFT_HIP] + coords_norm[RIGHT_HIP])
    trunk_vec = shoulder_mid - hip_mid
    vertical = np.array([0.0, 1.0])
    trunk_angle = angle_between_2d(trunk_vec, vertical)
    return np.array(
        [right_elbow, left_elbow, right_knee, left_knee, trunk_angle],
        dtype=float,
    )
