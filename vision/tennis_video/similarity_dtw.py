from __future__ import annotations
import numpy as np
from .stroke_types import PoseStroke


def dtw_distance(seq_usr: np.ndarray, seq_exp: np.ndarray) -> float:
    """Computes DTW average cost per step with L2 frame cost."""
    t1, t2 = seq_usr.shape[0], seq_exp.shape[0]
    if t1 == 0 or t2 == 0:
        return float("inf")
    dp = np.full((t1 + 1, t2 + 1), np.inf, dtype=float)
    dp[0, 0] = 0.0
    for i in range(1, t1 + 1):
        for j in range(1, t2 + 1):
            cost = np.linalg.norm(seq_usr[i - 1] - seq_exp[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[t1, t2] / (t1 + t2))


def form_distance(user: PoseStroke, expert: PoseStroke) -> float:
    """Returns DTW distance between user and expert angle sequences."""
    return dtw_distance(user.angles, expert.angles)
