from collections.abc import Iterator
import cv2
import numpy as np


def get_video_fps(video_path: str) -> float:
    """Returns frames per second for a video."""
    cap = cv2.VideoCapture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        return float(fps)
    finally:
        cap.release()


def iterate_frames(video_path: str) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yields (frame_index, timestamp_seconds, frame_bgr) for each frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            t_frame = frame_idx / fps if fps > 0 else 0.0
            yield frame_idx, t_frame, frame
            frame_idx += 1
    finally:
        cap.release()
