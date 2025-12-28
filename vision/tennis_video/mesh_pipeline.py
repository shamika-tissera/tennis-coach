from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .mesh_generation import (
    MeshGenerationError,
    MeshGeneratorConfig,
    mesh_generator_config_from_env,
    run_external_mesh_generator,
)
from .mesh_session import AttachMeshResult, attach_mesh_npz_to_session
from .smpl_mesh import load_smpl_model, smpl_vertices_from_coco17
from .smpl_parts import write_vertex_parts_assets


@dataclass(frozen=True)
class MeshPipelineResult:
    user_mesh_npz: Path
    expert_mesh_npz: Optional[Path]
    attach_result: AttachMeshResult


def _resolve_path(p: str | Path, *, repo_root: Path) -> Path:
    path = Path(p).expanduser()
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _window_from_session_part(part: dict) -> Optional[tuple[float, float]]:
    try:
        t_impact = part.get("t_impact")
        tau = part.get("tau") or []
        if t_impact is None or not tau:
            return None
        return float(t_impact) + float(min(tau)), float(t_impact) + float(max(tau))
    except Exception:
        return None


def generate_and_attach_smpl_mesh(
    *,
    repo_root: Path,
    session_path: Path,
    include_expert: bool = True,
    generator_cfg: Optional[MeshGeneratorConfig] = None,
    smpl_model_path: Optional[Path] = None,
) -> MeshPipelineResult:
    if generator_cfg is None:
        generator_cfg = mesh_generator_config_from_env()
    if generator_cfg is None:
        raise MeshGenerationError(
            "Mesh generation is not configured. Set TCVISION_MESH_GENERATOR_CMD "
            "(uses placeholders: {video} {out_npz} {start} {end})."
        )

    session = json.loads(session_path.read_text(encoding="utf-8"))
    user = session.get("user") or {}
    expert = session.get("expert") or {}

    user_video = user.get("video_path")
    if not user_video:
        raise MeshGenerationError("Session is missing user.video_path.")
    user_video_path = _resolve_path(user_video, repo_root=repo_root)
    if not user_video_path.exists():
        raise MeshGenerationError(f"User video not found: {user_video_path}")

    user_window = _window_from_session_part(user)
    if user_window is None:
        raise MeshGenerationError("Session is missing user.t_impact/user.tau; cannot infer stroke window.")
    user_start, user_end = user_window

    expert_video_path = None
    expert_window = None
    if include_expert:
        expert_video = expert.get("video_path")
        if expert_video:
            expert_video_path = _resolve_path(expert_video, repo_root=repo_root)
            expert_window = _window_from_session_part(expert)

    with tempfile.TemporaryDirectory(prefix="tcvision_mesh_") as td:
        tmp = Path(td)
        user_npz = tmp / "user_mesh.npz"
        run_external_mesh_generator(
            video_path=user_video_path,
            out_npz=user_npz,
            start_s=user_start,
            end_s=user_end,
            cfg=generator_cfg,
        )

        expert_npz = None
        if include_expert and expert_video_path and expert_video_path.exists() and expert_window is not None:
            expert_npz = tmp / "expert_mesh.npz"
            start_s, end_s = expert_window
            run_external_mesh_generator(
                video_path=expert_video_path,
                out_npz=expert_npz,
                start_s=start_s,
                end_s=end_s,
                cfg=generator_cfg,
            )

        labels_npy = None
        meta_json = None
        if smpl_model_path is not None and smpl_model_path.exists():
            labels_npy = tmp / "smpl_vertex_labels.npy"
            meta_json = tmp / "smpl_parts_meta.json"
            write_vertex_parts_assets(
                smpl_model_path,
                out_labels=labels_npy,
                out_meta=meta_json,
            )

        attach_result = attach_mesh_npz_to_session(
            session_path=session_path,
            user_mesh_npz=user_npz,
            expert_mesh_npz=expert_npz,
            vertex_labels_npy=labels_npy,
            parts_meta_json=meta_json,
            smpl_model_path=smpl_model_path if smpl_model_path and smpl_model_path.exists() else None,
        )

        return MeshPipelineResult(user_mesh_npz=user_npz, expert_mesh_npz=expert_npz, attach_result=attach_result)


def generate_and_attach_smpl_mesh_from_session_joints(
    *,
    session_path: Path,
    smpl_model_path: Path,
    include_expert: bool = True,
) -> MeshPipelineResult:
    """
    Built-in mesh generation: derive a dense SMPL mesh from the session's existing (T,17,3)
    joints3d_resampled arrays (user/expert) using SMPL skinning weights.

    This avoids any OpenMMLab / MediaPipe dependencies and is meant as a reliable default
    for visualization. Requires a local SMPL model file.
    """
    if not session_path.exists():
        raise MeshGenerationError(f"Session not found: {session_path}")
    if not smpl_model_path.exists():
        raise MeshGenerationError(f"SMPL model not found: {smpl_model_path}")

    smpl = load_smpl_model(smpl_model_path)
    session = json.loads(session_path.read_text(encoding="utf-8"))

    def _joints(part: dict, *, label: str) -> np.ndarray:
        arr = part.get("joints3d_resampled")
        if arr is None:
            raise MeshGenerationError(f"Session is missing {label}.joints3d_resampled.")
        joints = np.asarray(arr, dtype=np.float32)
        if joints.ndim != 3 or joints.shape[1:] != (17, 3):
            raise MeshGenerationError(
                f"Expected {label}.joints3d_resampled shape (T,17,3); got {tuple(joints.shape)}."
            )
        return joints

    user = session.get("user") or {}
    expert = session.get("expert") or {}
    user_j = _joints(user, label="user")
    user_verts = smpl_vertices_from_coco17(user_j, smpl, flip_y=True)

    expert_verts = None
    if include_expert and (expert.get("joints3d_resampled") is not None):
        try:
            expert_j = _joints(expert, label="expert")
            expert_verts = smpl_vertices_from_coco17(expert_j, smpl, flip_y=True)
        except MeshGenerationError:
            expert_verts = None

    with tempfile.TemporaryDirectory(prefix="tcvision_mesh_internal_") as td:
        tmp = Path(td)
        user_npz = tmp / "user_mesh.npz"
        np.savez_compressed(user_npz, vertices=user_verts, faces=smpl.faces)

        expert_npz = None
        if expert_verts is not None:
            expert_npz = tmp / "expert_mesh.npz"
            np.savez_compressed(expert_npz, vertices=expert_verts, faces=smpl.faces)

        labels_npy = tmp / "smpl_vertex_labels.npy"
        meta_json = tmp / "smpl_parts_meta.json"
        write_vertex_parts_assets(smpl_model_path, out_labels=labels_npy, out_meta=meta_json)

        attach_result = attach_mesh_npz_to_session(
            session_path=session_path,
            user_mesh_npz=user_npz,
            expert_mesh_npz=expert_npz,
            vertex_labels_npy=labels_npy,
            parts_meta_json=meta_json,
            smpl_model_path=smpl_model_path,
        )

        return MeshPipelineResult(user_mesh_npz=user_npz, expert_mesh_npz=expert_npz, attach_result=attach_result)
