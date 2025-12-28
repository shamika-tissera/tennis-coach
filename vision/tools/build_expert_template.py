from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tennis_video import (
    MMPoseBackend,
    build_user_pose_stroke,
    save_pose_stroke,
    config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and save an expert PoseStroke template.")
    parser.add_argument("video", help="Path to expert video file.")
    parser.add_argument("t_impact", type=float, help="Impact time in seconds from video start.")
    parser.add_argument("stroke_type", help="Stroke type label, e.g., forehand/backhand.")
    parser.add_argument(
        "--stroke-id",
        default=None,
        help="Identifier for the stroke; defaults to stroke_type plus basename.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output npz path. Defaults to expert_templates/<stroke_type>/<stroke_id>.npz",
    )
    parser.add_argument(
        "--window-before",
        type=float,
        default=config.WINDOW_BEFORE,
        help="Seconds before impact to include.",
    )
    parser.add_argument(
        "--window-after",
        type=float,
        default=config.WINDOW_AFTER,
        help="Seconds after impact to include.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stroke_id = args.stroke_id or f"{args.stroke_type}_{os.path.splitext(os.path.basename(args.video))[0]}"
    default_output = os.path.join("expert_templates", args.stroke_type, f"{stroke_id}.npz")
    output_path = args.output or default_output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    backend = MMPoseBackend(
        pose_config=config.POSE_CONFIG,
        pose_checkpoint=config.POSE_CHECKPOINT,
        det_config=config.DET_CONFIG,
        det_checkpoint=config.DET_CHECKPOINT,
        device=config.DEVICE,
        det_score_thr=config.DET_SCORE_THR,
    )

    stroke = build_user_pose_stroke(
        video_path=args.video,
        t_impact=args.t_impact,
        stroke_id=stroke_id,
        stroke_type=args.stroke_type,
        pose_backend=backend,
        window_before=args.window_before,
        window_after=args.window_after,
    )
    save_pose_stroke(stroke, output_path)
    print(f"Saved expert template to {output_path}")


if __name__ == "__main__":
    main()
