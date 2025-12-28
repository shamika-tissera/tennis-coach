from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tennis_video.pose_backend_mediapipe import MediaPipeBackend
from tennis_video.pose_processing import (
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ANKLE,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    normalize_keypoints_2d,
)
from tennis_video.smpl_mesh import load_smpl_model, smpl_vertices_from_coco17
from tennis_video.video_io import iterate_frames


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a dense SMPL mesh from video using MediaPipe Pose (outputs NPZ with vertices+faces)."
    )
    p.add_argument("--video", required=True, help="Path to input video.")
    p.add_argument(
        "--out-npz",
        required=True,
        help="Output NPZ path (will contain 'vertices' (T,V,3) and 'faces' (F,3)).",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional cap on processed frames (0 means no cap).",
    )
    p.add_argument(
        "--min-kpt-score",
        type=float,
        default=0.15,
        help="Minimum avg keypoint score to keep a frame.",
    )
    p.add_argument(
        "--smpl-model",
        default=os.getenv("TCVISION_SMPL_MODEL", ""),
        help="Path to SMPL model (.pkl/.npz). Default: env TCVISION_SMPL_MODEL.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video).expanduser()
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    smpl_path = Path(args.smpl_model).expanduser() if args.smpl_model else None
    if smpl_path is None or not smpl_path.exists():
        raise SystemExit(
            "Missing SMPL model file. Set env TCVISION_SMPL_MODEL or pass --smpl-model "
            "(e.g., SMPL_NEUTRAL.pkl)."
        )

    smpl = load_smpl_model(smpl_path)
    backend = MediaPipeBackend()
    try:
        frames: list[np.ndarray] = []
        for frame_idx, _, frame in iterate_frames(str(video_path)):
            kpts = backend.keypoints_from_frame(frame)
            if kpts is None:
                continue
            if float(np.mean(kpts[:, 2])) < float(args.min_kpt_score):
                continue

            coords_norm = normalize_keypoints_2d(kpts)
            if coords_norm is None:
                continue

            # Convert to a simple normalized pseudo-3D layout: x/y in normalized shoulder units,
            # and a small left/right depth offset so rotation in the viewer looks 3D.
            # COCO indices:
            # 0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear.
            LEFT_EYE = 1
            RIGHT_EYE = 2
            LEFT_EAR = 3
            RIGHT_EAR = 4
            left_joints = {
                LEFT_EYE,
                LEFT_EAR,
                LEFT_SHOULDER,
                LEFT_ELBOW,
                LEFT_WRIST,
                LEFT_HIP,
                LEFT_KNEE,
                LEFT_ANKLE,
            }
            right_joints = {
                RIGHT_EYE,
                RIGHT_EAR,
                RIGHT_SHOULDER,
                RIGHT_ELBOW,
                RIGHT_WRIST,
                RIGHT_HIP,
                RIGHT_KNEE,
                RIGHT_ANKLE,
            }
            depth_scale = 0.15
            joints3d = np.zeros((17, 3), dtype=np.float32)
            joints3d[:, 0] = coords_norm[:, 0].astype(np.float32)
            joints3d[:, 1] = coords_norm[:, 1].astype(np.float32)
            for j in range(17):
                if j in left_joints:
                    joints3d[j, 2] = float(depth_scale)
                elif j in right_joints:
                    joints3d[j, 2] = float(-depth_scale)
                else:
                    joints3d[j, 2] = 0.0
            frames.append(joints3d)

            if args.max_frames and len(frames) >= int(args.max_frames):
                break

        if len(frames) < 3:
            raise SystemExit("Not enough pose frames detected to generate a mesh.")

        coco_seq = np.stack(frames, axis=0)
        verts = smpl_vertices_from_coco17(coco_seq, smpl, flip_y=True)

        out_npz = Path(args.out_npz).expanduser()
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_npz, vertices=verts, faces=smpl.faces)
        print(f"Wrote: {out_npz} (frames={verts.shape[0]}, verts={verts.shape[1]}, faces={smpl.faces.shape[0]})")
    finally:
        backend.close()


if __name__ == "__main__":
    main()
