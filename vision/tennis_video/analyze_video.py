from __future__ import annotations
import numpy as np
from .stroke_window import extract_video_window
from .pose_backend_mmpose import MMPoseBackend
from .stroke_types import PoseStroke
from .similarity_dtw import form_distance
from .form_scoring import form_similarity_score
from . import config


def build_user_pose_stroke(
    video_path: str,
    t_impact: float,
    stroke_id: str,
    stroke_type: str,
    pose_backend: MMPoseBackend,
    window_before: float,
    window_after: float,
) -> PoseStroke:
    """Builds a PoseStroke from a video window around impact."""
    tau_list: list[float] = []
    angles_list: list[np.ndarray] = []
    for tau, frame in extract_video_window(video_path, t_impact, window_before, window_after):
        angles = pose_backend.angle_vector_from_frame(frame)
        if angles is None:
            continue
        tau_list.append(tau)
        angles_list.append(angles)
    if not tau_list or not angles_list:
        raise RuntimeError("No valid pose angles extracted from video window.")
    tau_arr = np.array(tau_list, dtype=float)
    angles_arr = np.vstack(angles_list)
    return PoseStroke(
        stroke_id=stroke_id,
        stroke_type=stroke_type,
        tau=tau_arr,
        angles=angles_arr,
    )


def analyze_stroke_against_expert(
    video_path: str,
    t_impact: float,
    stroke_type: str,
    expert_stroke: PoseStroke,
    pose_backend: MMPoseBackend,
) -> tuple[PoseStroke, float, float]:
    """Runs full pipeline and returns (user_pose, s_form, d_form)."""
    user_stroke = build_user_pose_stroke(
        video_path=video_path,
        t_impact=t_impact,
        stroke_id="user_stroke",
        stroke_type=stroke_type,
        pose_backend=pose_backend,
        window_before=config.WINDOW_BEFORE,
        window_after=config.WINDOW_AFTER,
    )
    d_form = form_distance(user_stroke, expert_stroke)
    s_form = form_similarity_score(d_form, config.D_MAX_FORM)
    return user_stroke, s_form, d_form


if __name__ == "__main__":
    backend = MMPoseBackend(
        pose_config=config.POSE_CONFIG,
        pose_checkpoint=config.POSE_CHECKPOINT,
        det_config=config.DET_CONFIG,
        det_checkpoint=config.DET_CHECKPOINT,
        device=config.DEVICE,
        det_score_thr=config.DET_SCORE_THR,
    )
    expert_path = "expert_templates/forehand.npz"
    try:
        from .stroke_types import load_pose_stroke
        expert_stroke = load_pose_stroke(expert_path)
    except Exception:
        print(f"Could not load expert stroke from {expert_path}; using empty placeholder.")
        expert_stroke = PoseStroke("expert_forehand", "forehand", np.array([]), np.array([]))

    video_path = "sample_video.mp4"
    t_impact = 1.5
    user, s_form, d_form = analyze_stroke_against_expert(
        video_path=video_path,
        t_impact=t_impact,
        stroke_type=expert_stroke.stroke_type,
        expert_stroke=expert_stroke,
        pose_backend=backend,
    )
    print(f"Form similarity: {s_form:.3f}, distance: {d_form:.3f}")
