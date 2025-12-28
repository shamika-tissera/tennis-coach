from __future__ import annotations
import argparse
import csv
import os

import cv2
import numpy as np


def get_video_fps(video_path: str) -> float:
    """Return FPS for a video path, or 0.0 when unavailable."""
    cap = cv2.VideoCapture(video_path)
    try:
        return float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate impact times by peak motion (optical flow) and update a manifest CSV."
    )
    parser.add_argument(
        "--csv-in",
        required=True,
        help="Input CSV with columns: video_path,t_impact,stroke_type,stroke_id",
    )
    parser.add_argument(
        "--csv-out",
        required=True,
        help="Output CSV with estimated t_impact filled when empty.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Median filter window (frames) for motion magnitude smoothing.",
    )
    parser.add_argument(
        "--search-start",
        type=float,
        default=0.05,
        help="Fraction of video to skip at the start when searching for impact.",
    )
    parser.add_argument(
        "--search-end",
        type=float,
        default=0.95,
        help="Fraction of video to skip at the end when searching for impact.",
    )
    return parser.parse_args()


def load_manifest(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_manifest(path: str, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def median_filter(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    pad = k // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = np.median(padded[i : i + k])
    return out


def estimate_impact_time(video_path: str, smooth_window: int, search_start: float, search_end: float) -> float | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or get_video_fps(video_path)
    ret, prev = cap.read()
    if not ret:
        cap.release()
        return None
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    motions: list[float] = []
    frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        mag = np.linalg.norm(flow, axis=2)
        motions.append(float(np.mean(mag)))
        prev_gray = gray
        frames += 1
    cap.release()
    if not motions:
        return None
    motion_arr = np.array(motions, dtype=float)
    motion_smooth = median_filter(motion_arr, smooth_window)
    start_idx = int(len(motion_smooth) * search_start)
    end_idx = max(start_idx + 1, int(len(motion_smooth) * search_end))
    roi = motion_smooth[start_idx:end_idx]
    if roi.size == 0:
        return None
    peak_rel = int(np.argmax(roi))
    impact_frame = start_idx + peak_rel + 1  # +1 because motions start from frame 1->2
    return impact_frame / fps


def main() -> None:
    args = parse_args()
    rows = load_manifest(args.csv_in)
    updated: list[dict[str, str]] = []
    for row in rows:
        t_val = row.get("t_impact", "").strip()
        if t_val:
            updated.append(row)
            continue
        video_path = row.get("video_path", "")
        if not video_path or not os.path.isfile(video_path):
            updated.append(row)
            continue
        est = estimate_impact_time(
            video_path,
            smooth_window=args.smooth_window,
            search_start=args.search_start,
            search_end=args.search_end,
        )
        if est is not None:
            row["t_impact"] = f"{est:.3f}"
            print(f"Estimated t_impact for {video_path}: {row['t_impact']}s")
        updated.append(row)
    save_manifest(args.csv_out, updated)
    print(f"Wrote updated manifest to {args.csv_out}")


if __name__ == "__main__":
    main()
