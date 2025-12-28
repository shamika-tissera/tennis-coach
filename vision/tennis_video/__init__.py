from __future__ import annotations

import importlib
from typing import Any

from .config import (
    D_MAX_FORM,
    DET_CHECKPOINT,
    DET_CONFIG,
    DET_SCORE_THR,
    DEVICE,
    POSE_CHECKPOINT,
    POSE_CONFIG,
    WINDOW_AFTER,
    WINDOW_BEFORE,
)

__all__ = [
    "WINDOW_BEFORE",
    "WINDOW_AFTER",
    "D_MAX_FORM",
    "DET_CONFIG",
    "DET_CHECKPOINT",
    "POSE_CONFIG",
    "POSE_CHECKPOINT",
    "DET_SCORE_THR",
    "DEVICE",
    "MMPoseBackend",
    "normalize_keypoints_2d",
    "compute_angle_vector",
    "PoseStroke",
    "ExpertLibrary",
    "save_pose_stroke",
    "load_pose_stroke",
    "load_expert_library_from_dir",
    "dtw_distance",
    "form_distance",
    "form_similarity_score",
    "score_stroke_against_template",
    "analyze_stroke_against_expert",
    "build_user_pose_stroke",
    "estimate_impact_time",
    "estimate_impact_times",
    "estimate_impact_time_tracknet",
    "estimate_impact_time_tracknet_from_detections",
    "load_tracknet_csv",
    "generate_feedback",
    "angle_deviation_report",
]

_LAZY: dict[str, tuple[str, str]] = {
    # Backends / pose
    "MMPoseBackend": ("tennis_video.pose_backend_mmpose", "MMPoseBackend"),
    "normalize_keypoints_2d": ("tennis_video.pose_processing", "normalize_keypoints_2d"),
    "compute_angle_vector": ("tennis_video.pose_processing", "compute_angle_vector"),
    # Stroke I/O
    "PoseStroke": ("tennis_video.stroke_types", "PoseStroke"),
    "ExpertLibrary": ("tennis_video.stroke_types", "ExpertLibrary"),
    "save_pose_stroke": ("tennis_video.stroke_types", "save_pose_stroke"),
    "load_pose_stroke": ("tennis_video.stroke_types", "load_pose_stroke"),
    "load_expert_library_from_dir": ("tennis_video.stroke_types", "load_expert_library_from_dir"),
    # Similarity / scoring
    "dtw_distance": ("tennis_video.similarity_dtw", "dtw_distance"),
    "form_distance": ("tennis_video.similarity_dtw", "form_distance"),
    "form_similarity_score": ("tennis_video.form_scoring", "form_similarity_score"),
    "score_stroke_against_template": ("tennis_video.form_scoring", "score_stroke_against_template"),
    # Video analysis
    "analyze_stroke_against_expert": ("tennis_video.analyze_video", "analyze_stroke_against_expert"),
    "build_user_pose_stroke": ("tennis_video.analyze_video", "build_user_pose_stroke"),
    # Impact detection
    "estimate_impact_time": ("tennis_video.impact_detection", "estimate_impact_time"),
    "estimate_impact_times": ("tennis_video.impact_detection", "estimate_impact_times"),
    "estimate_impact_time_tracknet": ("tennis_video.impact_tracknet", "estimate_impact_time_tracknet"),
    "estimate_impact_time_tracknet_from_detections": (
        "tennis_video.impact_tracknet",
        "estimate_impact_time_tracknet_from_detections",
    ),
    "load_tracknet_csv": ("tennis_video.impact_tracknet", "load_tracknet_csv"),
    # Feedback
    "generate_feedback": ("tennis_video.feedback", "generate_feedback"),
    "angle_deviation_report": ("tennis_video.feedback", "angle_deviation_report"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if not target:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY.keys()))
