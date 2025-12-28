import numpy as np
from .video_io import iterate_frames


def extract_video_window(
    video_path: str,
    t_impact: float,
    window_before: float,
    window_after: float,
) -> list[tuple[float, np.ndarray]]:
    """Returns [(tau, frame_bgr)] for frames within the window around impact."""
    tau_frames: list[tuple[float, np.ndarray]] = []
    start = t_impact - window_before
    end = t_impact + window_after
    for _, t_frame, frame in iterate_frames(video_path):
        if t_frame < start:
            continue
        if t_frame > end:
            break
        tau = t_frame - t_impact
        tau_frames.append((tau, frame))
    return tau_frames
