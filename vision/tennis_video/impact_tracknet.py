from __future__ import annotations
from typing import Callable, Iterable, Tuple, Optional, List
import numpy as np
from .video_io import get_video_fps

# TrackNet callable: returns (frame_idx, x, y, score) per detection.
TrackNetFn = Callable[[str], Iterable[Tuple[int, float, float, float]]]


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values, kernel, mode="same")


def estimate_impact_time_tracknet(
    video_path: str,
    tracknet_fn: TrackNetFn,
    *,
    smooth_window: int = 5,
    min_conf: float = 0.2,
    stroke_window: Optional[Tuple[float, float]] = None,
) -> float:
    """Estimate impact time from TrackNet ball positions by peak speed change."""
    fps = get_video_fps(video_path)
    detections = [
        (idx, x, y, score)
        for idx, x, y, score in tracknet_fn(video_path)
        if score >= min_conf
    ]
    if len(detections) < 2:
        raise RuntimeError("Not enough TrackNet detections to estimate impact.")

    detections.sort(key=lambda d: d[0])
    frames = np.array([d[0] for d in detections], dtype=float)
    xy = np.array([[d[1], d[2]] for d in detections], dtype=float)

    # Keep only finite coordinates
    finite_mask = np.isfinite(xy).all(axis=1)
    frames = frames[finite_mask]
    xy = xy[finite_mask]
    if len(frames) < 2:
        raise RuntimeError("Not enough valid TrackNet detections to estimate impact.")

    times = frames / fps
    times, idx_unique = np.unique(times, return_index=True)
    xy = xy[idx_unique]
    if len(times) < 2:
        raise RuntimeError("Not enough unique-frame detections to estimate impact.")

    dt = np.diff(times)
    dxy = np.diff(xy, axis=0)
    mid_times = 0.5 * (times[1:] + times[:-1])

    valid = dt > 1e-6
    if not np.any(valid):
        raise RuntimeError("No valid time deltas in TrackNet detections.")

    speeds = np.full_like(dt, np.nan, dtype=float)
    speeds[valid] = np.linalg.norm(dxy[valid], axis=1) / dt[valid]
    speeds_valid = speeds[valid]
    mid_times_valid = mid_times[valid]

    if speeds_valid.size < 2:
        raise RuntimeError("Not enough valid speed samples to estimate impact.")

    speeds_smooth = _smooth(speeds_valid, smooth_window)
    accel = np.abs(np.gradient(speeds_smooth, mid_times_valid))
    scores = accel

    if stroke_window is not None:
        mask = (mid_times_valid >= stroke_window[0]) & (mid_times_valid <= stroke_window[1])
        if not np.any(mask):
            raise RuntimeError("No TrackNet detections in the specified stroke window.")
        scores = np.where(mask, scores, -np.inf)

    idx = int(np.argmax(scores))
    if not np.isfinite(scores[idx]) or scores[idx] < 0:
        raise RuntimeError("Could not determine impact from TrackNet detections.")

    return float(mid_times_valid[idx])


def estimate_impact_time_tracknet_from_detections(
    video_path: str,
    detections: Iterable[Tuple[int, float, float, float]],
    *,
    smooth_window: int = 5,
    min_conf: float = 0.2,
    stroke_window: Optional[Tuple[float, float]] = None,
) -> float:
    """Estimate impact time given precomputed TrackNet detections."""
    return estimate_impact_time_tracknet(
        video_path,
        lambda _: detections,
        smooth_window=smooth_window,
        min_conf=min_conf,
        stroke_window=stroke_window,
    )


def load_tracknet_csv(csv_path: str) -> List[Tuple[int, float, float, float]]:
    """Load TrackNet detections from CSV with columns: frame,x,y,score."""
    import csv

    detections: list[tuple[int, float, float, float]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"frame", "x", "y", "score"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("CSV must have columns: frame,x,y,score")
        for row in reader:
            detections.append(
                (
                    int(row["frame"]),
                    float(row["x"]),
                    float(row["y"]),
                    float(row["score"]),
                )
            )
    return detections
