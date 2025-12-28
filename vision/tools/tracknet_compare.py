from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Iterable, Tuple
import csv
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRACKNET_ROOT = ROOT / "TrackNet-main"
if str(TRACKNET_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKNET_ROOT))

from model import BallTrackerNet  # type: ignore
from general import postprocess  # type: ignore

from tennis_video import (
    MMPoseBackend,
    analyze_video,
    config,
    estimate_impact_time_tracknet_from_detections,
    load_tracknet_csv,
    generate_feedback,
)
from tennis_video.stroke_types import load_pose_stroke


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
    device: str,
    height: int = 360,
    width: int = 640,
) -> list[tuple[int, float, float, float]]:
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
        description="Run TrackNet to get impact, then compare stroke vs expert templates."
    )
    p.add_argument("--video-path", required=True, help="Path to input video.")
    p.add_argument("--tracknet-weights", required=True, help="Path to TrackNet weights (.pt/.pth).")
    p.add_argument("--stroke-type", required=False, help="Stroke type (e.g., straight_shot).")
    p.add_argument("--expert", help="Path to expert template .npz; overrides stroke-type scan.")
    p.add_argument("--expert-root", default="expert_templates", help="Directory of expert templates.")
    p.add_argument("--top-k", type=int, default=3, help="Report top-k matches.")
    p.add_argument("--device", default=config.DEVICE, help="Device for TrackNet and pose (cpu/cuda:0).")
    p.add_argument("--pose-device", default=None, help="Device for pose; default uses --device.")
    p.add_argument("--det-score-thr", type=float, default=config.DET_SCORE_THR, help="Person det threshold.")
    p.add_argument("--window-before", type=float, default=config.WINDOW_BEFORE, help="Seconds before impact.")
    p.add_argument("--window-after", type=float, default=config.WINDOW_AFTER, help="Seconds after impact.")
    p.add_argument("--save-tracknet-csv", help="Optional path to save TrackNet detections CSV.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pose_device = args.pose_device or args.device

    # Run TrackNet
    tn_model = BallTrackerNet()
    tn_model.load_state_dict(torch.load(args.tracknet_weights, map_location=args.device))
    tn_model = tn_model.to(args.device)
    tn_model.eval()

    frames, fps = read_video(args.video_path)
    if len(frames) < 3:
        raise RuntimeError("Video too short for TrackNet inference (need at least 3 frames).")
    detections = infer_tracknet(frames, tn_model, device=args.device)
    if args.save_tracknet_csv:
        write_csv(detections, Path(args.save_tracknet_csv))
        print(f"Saved TrackNet detections to {args.save_tracknet_csv}")

    # Impact from TrackNet
    t_impact = estimate_impact_time_tracknet_from_detections(args.video_path, detections)
    print(f"TrackNet-estimated impact time: {t_impact:.3f}s")

    # Pose backend
    backend = MMPoseBackend(
        pose_config=config.POSE_CONFIG,
        pose_checkpoint=config.POSE_CHECKPOINT,
        det_config=config.DET_CONFIG,
        det_checkpoint=config.DET_CHECKPOINT,
        device=pose_device,
        det_score_thr=args.det_score_thr,
    )

    # Build user stroke once
    user_stroke = analyze_video.build_user_pose_stroke(
        video_path=args.video_path,
        t_impact=t_impact,
        stroke_id="user_stroke",
        stroke_type=args.stroke_type or "unknown",
        pose_backend=backend,
        window_before=args.window_before,
        window_after=args.window_after,
    )

    # Compare
    matches = []
    if args.expert:
        expert_stroke = load_pose_stroke(args.expert)
        d_form = analyze_video.form_distance(user_stroke, expert_stroke)
        s_form = analyze_video.form_similarity_score(d_form, config.D_MAX_FORM)
        matches.append((args.expert, expert_stroke.stroke_type, s_form, d_form, expert_stroke))
    else:
        if not args.stroke_type:
            raise SystemExit("Provide --stroke-type or --expert.")
        candidates = sorted(Path(args.expert_root).glob("**/*.npz"))
        for c in candidates:
            try:
                stroke = load_pose_stroke(str(c))
            except Exception:
                continue
            if stroke.stroke_type != args.stroke_type:
                continue
            d_form = analyze_video.form_distance(user_stroke, stroke)
            s_form = analyze_video.form_similarity_score(d_form, config.D_MAX_FORM)
            matches.append((str(c), stroke.stroke_type, s_form, d_form, stroke))

    if not matches:
        raise SystemExit("No expert templates matched the criteria.")

    matches.sort(key=lambda x: x[3])
    top_k = matches[: max(1, args.top_k)]

    print(f"User stroke frames: {len(user_stroke.tau)}, angles shape: {user_stroke.angles.shape}")
    print("Top matches:")
    for path, stype, s_form, d_form, _ in top_k:
        print(f"- {Path(path).name} [{stype}]  sim={s_form:.3f}  dist={d_form:.3f}")

    best_path, best_type, best_sim, best_dist, best_expert = top_k[0]
    print("\nBest match feedback:")
    for line in generate_feedback(user_stroke, best_expert):
        print(f"- {line}")


if __name__ == "__main__":
    main()
