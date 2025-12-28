from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np

from . import config
from .form_scoring import form_similarity_score
from .pose_processing import compute_angle_vector, normalize_keypoints_2d
from .similarity_dtw import form_distance
from .stroke_types import PoseStroke, load_pose_stroke
from .stroke_window import extract_video_window
from .feedback import generate_feedback
class PoseBackend(Protocol):
    def keypoints_from_frame(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]: ...

COCO17_JOINT_NAMES: list[str] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# COCO body skeleton edges.
COCO17_BONES: list[tuple[int, int]] = [
    (5, 6),  # shoulders
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),  # hips
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 5),
    (0, 6),
]

_COCO17_BODY_SLICE = slice(0, 17)

_LEFT_JOINTS = {
    COCO17_JOINT_NAMES.index("left_eye"),
    COCO17_JOINT_NAMES.index("left_ear"),
    COCO17_JOINT_NAMES.index("left_shoulder"),
    COCO17_JOINT_NAMES.index("left_elbow"),
    COCO17_JOINT_NAMES.index("left_wrist"),
    COCO17_JOINT_NAMES.index("left_hip"),
    COCO17_JOINT_NAMES.index("left_knee"),
    COCO17_JOINT_NAMES.index("left_ankle"),
}
_RIGHT_JOINTS = {
    COCO17_JOINT_NAMES.index("right_eye"),
    COCO17_JOINT_NAMES.index("right_ear"),
    COCO17_JOINT_NAMES.index("right_shoulder"),
    COCO17_JOINT_NAMES.index("right_elbow"),
    COCO17_JOINT_NAMES.index("right_wrist"),
    COCO17_JOINT_NAMES.index("right_hip"),
    COCO17_JOINT_NAMES.index("right_knee"),
    COCO17_JOINT_NAMES.index("right_ankle"),
}


@dataclass(frozen=True)
class ExpertMatch:
    template_path: str
    expert_stroke: PoseStroke
    form_distance: float
    form_similarity: float


def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "session"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _resample_sequence(seq: np.ndarray, length: int) -> np.ndarray:
    """Resample a (T, ...) array along time dimension to fixed length."""
    if length <= 0:
        raise ValueError("length must be > 0")
    if seq.shape[0] == length:
        return seq
    if seq.shape[0] < 2:
        return np.repeat(seq, repeats=length, axis=0)

    t_src = np.linspace(0.0, 1.0, num=seq.shape[0], dtype=float)
    t_dst = np.linspace(0.0, 1.0, num=length, dtype=float)

    flat = seq.reshape(seq.shape[0], -1)
    out = np.empty((length, flat.shape[1]), dtype=float)
    for d in range(flat.shape[1]):
        out[:, d] = np.interp(t_dst, t_src, flat[:, d])
    return out.reshape((length,) + seq.shape[1:])


def _keypoints_to_joints3d(coords_norm: np.ndarray, depth_scale: float) -> np.ndarray:
    """Convert (K,2) normalized coords to a simple (K,3) pseudo-3D layout."""
    joints3d = np.zeros((coords_norm.shape[0], 3), dtype=float)
    joints3d[:, 0] = coords_norm[:, 0]
    joints3d[:, 1] = coords_norm[:, 1]
    for idx in range(coords_norm.shape[0]):
        if idx in _LEFT_JOINTS:
            joints3d[idx, 2] = float(depth_scale)
        elif idx in _RIGHT_JOINTS:
            joints3d[idx, 2] = float(-depth_scale)
        else:
            joints3d[idx, 2] = 0.0
    return joints3d


def extract_pose_features_window(
    *,
    video_path: str,
    t_impact: float,
    pose_backend: PoseBackend,
    window_before: float,
    window_after: float,
    depth_scale: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract pose features from a video window.

    Returns:
      - tau: (T,)
      - angles: (T,5)
      - keypoints2d: (T,17,3) body keypoints (x,y,score)
      - joints3d: (T,17,3) pseudo-3D joints in normalized hip-centered space
    """
    tau_list: list[float] = []
    angles_list: list[np.ndarray] = []
    keypoints2d_list: list[np.ndarray] = []
    joints3d_list: list[np.ndarray] = []

    for tau, frame in extract_video_window(video_path, t_impact, window_before, window_after):
        keypoints = pose_backend.keypoints_from_frame(frame)
        if keypoints is None or keypoints.shape[0] < 17:
            continue

        keypoints_body = np.asarray(keypoints[_COCO17_BODY_SLICE], dtype=float)
        coords_norm = normalize_keypoints_2d(keypoints_body)
        if coords_norm is None:
            continue

        tau_list.append(float(tau))
        keypoints2d_list.append(keypoints_body)
        angles_list.append(compute_angle_vector(coords_norm))
        joints3d_list.append(_keypoints_to_joints3d(coords_norm, depth_scale=depth_scale))

    if not tau_list:
        raise RuntimeError("No valid poses extracted from video window.")

    tau_arr = np.asarray(tau_list, dtype=float)
    angles_arr = np.vstack(angles_list).astype(float)
    keypoints2d_arr = np.stack(keypoints2d_list, axis=0).astype(float)
    joints3d_arr = np.stack(joints3d_list, axis=0).astype(float)
    return tau_arr, angles_arr, keypoints2d_arr, joints3d_arr


def load_manifest_index(manifest_csv: Path) -> dict[str, dict[str, str]]:
    """Load a manifest CSV into a stroke_id -> row mapping."""
    with manifest_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        out: dict[str, dict[str, str]] = {}
        for row in reader:
            stroke_id = (row.get("stroke_id") or "").strip()
            if not stroke_id:
                continue
            out[stroke_id] = {k: (v or "").strip() for k, v in row.items()}
        return out


def _stroke_id_matches_view(stroke_id: str, view: Optional[str]) -> bool:
    if not view:
        return True
    view_norm = view.strip().lower()
    if not view_norm:
        return True
    # Convention used in manifests/templates: *_topview / *_sideview
    return stroke_id.lower().endswith(f"_{view_norm}")


def find_best_expert_match(
    *,
    user_stroke: PoseStroke,
    expert_root: Path,
    stroke_type: Optional[str],
    view: Optional[str] = None,
    top_k: int = 1,
) -> list[ExpertMatch]:
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    matches: list[ExpertMatch] = []
    for template_path in sorted(expert_root.glob("**/*.npz")):
        json_sidecar = template_path.with_suffix(template_path.suffix + ".json")
        if not json_sidecar.exists():
            continue
        try:
            expert_stroke = load_pose_stroke(str(template_path))
        except Exception:
            continue

        if stroke_type and expert_stroke.stroke_type != stroke_type:
            continue
        if not _stroke_id_matches_view(expert_stroke.stroke_id, view):
            continue

        d_form = float(form_distance(user_stroke, expert_stroke))
        s_form = float(form_similarity_score(d_form, config.D_MAX_FORM))
        matches.append(
            ExpertMatch(
                template_path=str(template_path),
                expert_stroke=expert_stroke,
                form_distance=d_form,
                form_similarity=s_form,
            )
        )

    matches.sort(key=lambda m: m.form_distance)
    return matches[:top_k]


def _impact_t_norm_from_tau(tau: np.ndarray) -> float:
    if tau.size == 0:
        return 0.5
    tau_min = float(np.min(tau))
    tau_max = float(np.max(tau))
    denom = tau_max - tau_min
    if denom <= 1e-9:
        return 0.5
    return float((0.0 - tau_min) / denom)


def build_web_session_payload(
    *,
    pose_backend: PoseBackend,
    user_video_path: str,
    t_impact_user: Optional[float],
    stroke_type: Optional[str],
    view: Optional[str],
    expert_template_path: Optional[str],
    expert_root: Path,
    expert_manifest_csv: Path,
    window_before: float = config.WINDOW_BEFORE,
    window_after: float = config.WINDOW_AFTER,
    resample_len: int = 120,
) -> dict[str, Any]:
    if t_impact_user is None:
        from .impact_detection import estimate_impact_time

        t_impact_user = float(estimate_impact_time(user_video_path, pose_backend))

    tau_u, angles_u, kpts_u, joints_u = extract_pose_features_window(
        video_path=user_video_path,
        t_impact=t_impact_user,
        pose_backend=pose_backend,
        window_before=window_before,
        window_after=window_after,
    )
    user_stroke = PoseStroke(
        stroke_id="user_stroke",
        stroke_type=stroke_type or "unknown",
        tau=tau_u,
        angles=angles_u,
    )

    if expert_template_path:
        expert_match = ExpertMatch(
            template_path=expert_template_path,
            expert_stroke=load_pose_stroke(expert_template_path),
            form_distance=float("nan"),
            form_similarity=float("nan"),
        )
    else:
        matches = find_best_expert_match(
            user_stroke=user_stroke,
            expert_root=expert_root,
            stroke_type=stroke_type,
            view=view,
            top_k=1,
        )
        if not matches:
            raise RuntimeError("No expert templates matched the criteria.")
        expert_match = matches[0]

    expert_stroke = expert_match.expert_stroke
    d_form = float(form_distance(user_stroke, expert_stroke))
    s_form = float(form_similarity_score(d_form, config.D_MAX_FORM))
    feedback = generate_feedback(user_stroke, expert_stroke)

    # Best-effort: map expert template -> expert video in manifest for 3D visualization.
    manifest = load_manifest_index(expert_manifest_csv) if expert_manifest_csv.exists() else {}
    expert_row = manifest.get(expert_stroke.stroke_id)
    expert_video_path = expert_row.get("video_path") if expert_row else None
    t_impact_expert = float(expert_row["t_impact"]) if expert_row and expert_row.get("t_impact") else None

    tau_e: Optional[np.ndarray] = None
    joints_e: Optional[np.ndarray] = None
    kpts_e: Optional[np.ndarray] = None
    if expert_video_path and t_impact_expert is not None:
        tau_e, _, kpts_e, joints_e = extract_pose_features_window(
            video_path=expert_video_path,
            t_impact=t_impact_expert,
            pose_backend=pose_backend,
            window_before=window_before,
            window_after=window_after,
        )

    # Resample to a shared playback length.
    joints_u_r = _resample_sequence(joints_u, resample_len).astype(np.float32)
    if joints_e is not None:
        joints_e_r = _resample_sequence(joints_e, resample_len).astype(np.float32)
    else:
        joints_e_r = None

    t = np.linspace(0.0, 1.0, num=resample_len, dtype=np.float32)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": _utc_now_iso(),
        "stroke_type": stroke_type or expert_stroke.stroke_type or "unknown",
        "view": view or None,
        "user": {
            "video_path": user_video_path,
            "t_impact": float(t_impact_user),
            "tau": tau_u.astype(float).tolist(),
            "keypoints2d": kpts_u.astype(float).tolist(),
            "impact_t_norm": _impact_t_norm_from_tau(tau_u),
            "joints3d_resampled": joints_u_r.tolist(),
        },
        "expert": {
            "template_path": expert_match.template_path,
            "stroke_id": expert_stroke.stroke_id,
            "video_path": expert_video_path,
            "t_impact": t_impact_expert,
            "tau": (tau_e.astype(float).tolist() if tau_e is not None else None),
            "keypoints2d": (kpts_e.astype(float).tolist() if kpts_e is not None else None),
            "impact_t_norm": (_impact_t_norm_from_tau(tau_e) if tau_e is not None else None),
            "joints3d_resampled": (joints_e_r.tolist() if joints_e_r is not None else None),
        },
        "timeline": {
            "t": t.tolist(),
            "impact_t_norm": float(window_before / (window_before + window_after))
            if (window_before + window_after) > 1e-9
            else 0.5,
        },
        "joints": {
            "names": COCO17_JOINT_NAMES,
            "bones": [list(b) for b in COCO17_BONES],
        },
        "metrics": {
            "form_distance": d_form,
            "form_similarity": s_form,
        },
        "feedback": feedback,
    }
    return payload


def write_web_session(
    *,
    sessions_dir: Path,
    session_id: str,
    payload: dict[str, Any],
) -> Path:
    session_dir = sessions_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=False)
    out_path = session_dir / "session.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def new_session_id(user_video_path: str) -> str:
    base = _slugify(Path(user_video_path).stem)
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%d_%H%M%S")
    ms = int(now.microsecond // 1000)
    return f"{ts}_{ms:03d}_{base}"
