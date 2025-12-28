from __future__ import annotations
from typing import List, Tuple
import numpy as np
from .stroke_types import PoseStroke

ANGLE_NAMES = [
    "right_elbow",
    "left_elbow",
    "right_knee",
    "left_knee",
    "trunk",
]

ANGLE_TIPS = {
    "right_elbow": "Stabilize the hitting arm; reduce elbow flare near impact.",
    "left_elbow": "Stabilize the non-hitting arm; keep it supportive for balance.",
    "right_knee": "Improve lower-body stability; reduce knee collapse through impact.",
    "left_knee": "Improve lower-body stability; reduce knee collapse through impact.",
    "trunk": "Keep torso more upright and rotate smoothly through impact.",
}


def _resample_sequence(seq: np.ndarray, length: int) -> np.ndarray:
    """Resample a (T, D) sequence to fixed length via linear interpolation."""
    t_src = np.linspace(0.0, 1.0, num=seq.shape[0])
    t_dst = np.linspace(0.0, 1.0, num=length)
    out = []
    for d in range(seq.shape[1]):
        out.append(np.interp(t_dst, t_src, seq[:, d]))
    return np.stack(out, axis=1)


def angle_deviation_report(
    user: PoseStroke, expert: PoseStroke, resample_len: int = 100
) -> List[Tuple[str, float]]:
    """Return per-angle mean absolute deviation (radians) after length-normalization."""
    if user.angles.shape[1] != expert.angles.shape[1]:
        raise ValueError("User and expert angle dimensions differ.")
    user_r = _resample_sequence(user.angles, resample_len)
    exp_r = _resample_sequence(expert.angles, resample_len)
    mad = np.mean(np.abs(user_r - exp_r), axis=0)
    return list(zip(ANGLE_NAMES[: len(mad)], mad.tolist()))


def generate_feedback(
    user: PoseStroke, expert: PoseStroke, mild: float = 0.15, severe: float = 0.30
) -> List[str]:
    """
    Generate simple, actionable feedback based on per-angle deviations (radians).
    Thresholds: mild (<mild), moderate (<severe), high (>=severe).
    """
    deviations = angle_deviation_report(user, expert)
    feedback: list[str] = []
    for name, gap in deviations:
        if gap < mild:
            continue
        level = "moderate" if gap < severe else "high"
        tip = ANGLE_TIPS.get(name, "Refine consistency for this joint.")
        feedback.append(f"{name.replace('_', ' ').title()}: {level} deviation ({gap:.2f} rad). {tip}")
    if not feedback:
        feedback.append("Overall alignment is close; keep current form and consistency.")
    return feedback
