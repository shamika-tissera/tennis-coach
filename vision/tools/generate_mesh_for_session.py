from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tennis_video.mesh_generation import MeshGenerationError, mesh_generator_config_from_env
from tennis_video.mesh_pipeline import (
    generate_and_attach_smpl_mesh,
    generate_and_attach_smpl_mesh_from_session_joints,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a SMPL mesh for an existing web session using a configured generator command."
    )
    parser.add_argument("--session-id", required=True, help="Session id under sessions dir.")
    parser.add_argument(
        "--sessions-dir",
        default="data/web_sessions",
        help="Directory containing session folders.",
    )
    parser.add_argument(
        "--no-expert",
        action="store_true",
        help="Only generate the user mesh (skip expert).",
    )
    parser.add_argument(
        "--smpl-model",
        default=os.getenv("TCVISION_SMPL_MODEL"),
        help="Optional SMPL model path for faces + part labels (default: env TCVISION_SMPL_MODEL).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = mesh_generator_config_from_env()

    sessions_dir = Path(args.sessions_dir)
    session_path = sessions_dir / args.session_id / "session.json"
    smpl_model_path = Path(args.smpl_model).expanduser() if args.smpl_model else None
    if smpl_model_path is not None and not smpl_model_path.exists():
        smpl_model_path = None

    try:
        if smpl_model_path is not None:
            result = generate_and_attach_smpl_mesh_from_session_joints(
                session_path=session_path,
                smpl_model_path=smpl_model_path,
                include_expert=not bool(args.no_expert),
            )
        elif cfg is not None:
            result = generate_and_attach_smpl_mesh(
                repo_root=ROOT,
                session_path=session_path,
                include_expert=not bool(args.no_expert),
                generator_cfg=cfg,
                smpl_model_path=None,
            )
        else:
            raise SystemExit(
                "Mesh generation not configured. Set TCVISION_SMPL_MODEL (recommended) or TCVISION_MESH_GENERATOR_CMD."
            )
    except MeshGenerationError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Updated: {result.attach_result.session_path}")
    print(f"- user vertices: {result.attach_result.user_vertices_path}")
    if result.attach_result.expert_vertices_path:
        print(f"- expert vertices: {result.attach_result.expert_vertices_path}")
    if result.attach_result.vertex_labels_path:
        print(f"- vertex labels: {result.attach_result.vertex_labels_path}")


if __name__ == "__main__":
    main()
