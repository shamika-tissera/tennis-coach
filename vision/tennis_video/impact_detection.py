from __future__ import annotations

from typing import Optional, Protocol

import numpy as np

from .video_io import iterate_frames
from .pose_processing import LEFT_WRIST, RIGHT_WRIST


class PoseBackend(Protocol):
    def keypoints_from_frame(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]: ...


def _smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=float) / float(window)
    return np.vstack(
        [np.convolve(values[:, i], kernel, mode="same") for i in range(values.shape[1])]
    ).T


def _select_wrist(keypoints: np.ndarray, min_score: float) -> Optional[np.ndarray]:
    left = keypoints[LEFT_WRIST]
    right = keypoints[RIGHT_WRIST]
    if left[2] < min_score and right[2] < min_score:
        return None
    return left if left[2] >= right[2] else right


def estimate_impact_time(
    video_path: str,
    pose_backend: PoseBackend,
    min_kpt_score: float = 0.2,
    smooth_window: int = 5,
) -> float:
    """Estimate impact time by peak wrist speed over the whole clip."""
    times: list[float] = []
    points: list[np.ndarray] = []
    for _, t_frame, frame in iterate_frames(video_path):
        kpts = pose_backend.keypoints_from_frame(frame)
        if kpts is None:
            continue
        wrist = _select_wrist(kpts, min_kpt_score)
        if wrist is None:
            continue
        times.append(t_frame)
        points.append(wrist[:2].astype(float))

    if len(times) < 3:
        raise RuntimeError("Not enough wrist detections to estimate impact time.")

    t_arr = np.asarray(times, dtype=float)
    pts = np.vstack(points)
    pts_smooth = _smooth_series(pts, smooth_window)

    vel = np.gradient(pts_smooth, t_arr, axis=0)
    speeds = np.linalg.norm(vel, axis=1)
    speeds_smooth = np.convolve(speeds, np.ones(smooth_window) / smooth_window, mode="same")

    idx = int(np.argmax(speeds_smooth))
    return float(t_arr[idx])


def _local_maxima(values: np.ndarray) -> np.ndarray:
    """Return indices of simple 1D local maxima (excluding endpoints)."""
    if values.size < 3:
        return np.array([], dtype=int)
    left = values[1:-1] > values[:-2]
    right = values[1:-1] >= values[2:]
    return np.where(left & right)[0] + 1


def estimate_impact_times(
    video_path: str,
    pose_backend: PoseBackend,
    *,
    max_impacts: int = 5,
    min_separation_s: float = 1.0,
    min_kpt_score: float = 0.2,
    smooth_window: int = 5,
    min_peak_frac: float = 0.6,
) -> list[float]:
    """
    Estimate multiple impact times by picking multiple wrist-speed peaks.

    This is intended for clips containing multiple strokes. It returns up to `max_impacts`
    peaks, enforcing a minimum temporal separation.
    """
    if max_impacts <= 0:
        raise ValueError("max_impacts must be > 0")
    if min_separation_s < 0:
        raise ValueError("min_separation_s must be >= 0")
    if not (0.0 <= min_peak_frac <= 1.0):
        raise ValueError("min_peak_frac must be in [0, 1]")

    times: list[float] = []
    points: list[np.ndarray] = []
    for _, t_frame, frame in iterate_frames(video_path):
        kpts = pose_backend.keypoints_from_frame(frame)
        if kpts is None:
            continue
        wrist = _select_wrist(kpts, min_kpt_score)
        if wrist is None:
            continue
        times.append(t_frame)
        points.append(wrist[:2].astype(float))

    if len(times) < 3:
        raise RuntimeError("Not enough wrist detections to estimate impact times.")

    t_arr = np.asarray(times, dtype=float)
    pts = np.vstack(points)
    pts_smooth = _smooth_series(pts, smooth_window)

    vel = np.gradient(pts_smooth, t_arr, axis=0)
    speeds = np.linalg.norm(vel, axis=1)
    speeds_smooth = np.convolve(speeds, np.ones(smooth_window) / smooth_window, mode="same")

    peak_idx = _local_maxima(speeds_smooth)
    if peak_idx.size == 0:
        # Fallback: single best.
        idx = int(np.argmax(speeds_smooth))
        return [float(t_arr[idx])]

    peak_values = speeds_smooth[peak_idx]
    vmax = float(np.max(peak_values))
    if vmax <= 1e-9:
        idx = int(np.argmax(speeds_smooth))
        return [float(t_arr[idx])]

    # Threshold to remove very small peaks.
    keep = peak_values >= (min_peak_frac * vmax)
    peak_idx = peak_idx[keep] if np.any(keep) else peak_idx

    # Greedy select by peak magnitude with min separation.
    order = np.argsort(speeds_smooth[peak_idx])[::-1]
    selected: list[int] = []
    for j in order:
        idx = int(peak_idx[j])
        t = float(t_arr[idx])
        if any(abs(t - float(t_arr[k])) < min_separation_s for k in selected):
            continue
        selected.append(idx)
        if len(selected) >= max_impacts:
            break

    selected_times = sorted(float(t_arr[i]) for i in selected)
    if not selected_times:
        idx = int(np.argmax(speeds_smooth))
        selected_times = [float(t_arr[idx])]
    return selected_times
