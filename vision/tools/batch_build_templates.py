from __future__ import annotations
import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Iterable

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
    parser = argparse.ArgumentParser(
        description="Batch-build expert PoseStroke templates from a CSV manifest."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="CSV with columns: video_path,t_impact,stroke_type[,stroke_id].",
    )
    parser.add_argument(
        "--output-root",
        default="expert_templates",
        help="Root directory to store templates (will create per stroke_type).",
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


def read_manifest(csv_path: str) -> Iterable[dict[str, str]]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("video_path") or not row.get("t_impact") or not row.get("stroke_type"):
                continue
            yield row


def main() -> None:
    args = parse_args()
    backend = MMPoseBackend(
        pose_config=config.POSE_CONFIG,
        pose_checkpoint=config.POSE_CHECKPOINT,
        det_config=config.DET_CONFIG,
        det_checkpoint=config.DET_CHECKPOINT,
        device=config.DEVICE,
        det_score_thr=config.DET_SCORE_THR,
    )

    os.makedirs(args.output_root, exist_ok=True)
    for row in read_manifest(args.csv):
        video_path = row["video_path"]
        t_impact = float(row["t_impact"])
        stroke_type = row["stroke_type"]
        stroke_id = row.get("stroke_id") or os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(args.output_root, stroke_type)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{stroke_id}.npz")
        print(f"Processing {video_path} -> {output_path}")
        stroke = build_user_pose_stroke(
            video_path=video_path,
            t_impact=t_impact,
            stroke_id=stroke_id,
            stroke_type=stroke_type,
            pose_backend=backend,
            window_before=args.window_before,
            window_after=args.window_after,
        )
        save_pose_stroke(stroke, output_path)
    print("Done.")


if __name__ == "__main__":
    main()
