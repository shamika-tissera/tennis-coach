from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np
import torch

# Make TrackNet code importable
ROOT = Path(__file__).resolve().parents[1]
TRACKNET_ROOT = ROOT / "TrackNet-main"
if str(TRACKNET_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKNET_ROOT))

from model import BallTrackerNet  # type: ignore
from general import postprocess  # type: ignore


def read_video(path_video: str) -> tuple[list[np.ndarray], int]:
    cap = cv2.VideoCapture(path_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    return frames, fps


@torch.no_grad()
def infer_tracknet(
    frames: list[np.ndarray],
    model: torch.nn.Module,
    device: str = "cuda",
    height: int = 360,
    width: int = 640,
) -> list[tuple[int, float, float, float]]:
    """Run TrackNet on frames and return detections (frame_idx, x, y, score)."""
    detections: list[tuple[int, float, float, float]] = []
    for num in range(2, len(frames)):
        img = cv2.resize(frames[num], (width, height))
        img_prev = cv2.resize(frames[num - 1], (width, height))
        img_preprev = cv2.resize(frames[num - 2], (width, height))
        imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
        imgs = imgs.astype(np.float32) / 255.0
        imgs = np.rollaxis(imgs, 2, 0)
        inp = np.expand_dims(imgs, axis=0)

        out = model(torch.from_numpy(inp).float().to(device))
        output = out.argmax(dim=1).detach().cpu().numpy()
        x_pred, y_pred = postprocess(output)
        if x_pred is not None and y_pred is not None:
            detections.append((num, float(x_pred), float(y_pred), 1.0))
        else:
            detections.append((num, np.nan, np.nan, 0.0))
    return detections


def write_csv(detections: Iterable[tuple[int, float, float, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "x", "y", "score"])
        writer.writerows(detections)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run TrackNet on a video and emit detections as CSV (frame,x,y,score)."
    )
    p.add_argument("--video-path", required=True, help="Path to input video.")
    p.add_argument("--model-path", required=True, help="Path to TrackNet weights (.pth).")
    p.add_argument("--output-csv", required=True, help="Path to write detections CSV.")
    p.add_argument("--device", default="cuda:0", help="Device for inference (e.g., cuda:0 or cpu).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device
    model = BallTrackerNet()
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    frames, fps = read_video(args.video_path)
    if len(frames) < 3:
        raise RuntimeError("Video too short for TrackNet inference (need at least 3 frames).")
    detections = infer_tracknet(frames, model, device=device)
    write_csv(detections, Path(args.output_csv))
    print(f"Wrote {len(detections)} detections to {args.output_csv}")


if __name__ == "__main__":
    main()
