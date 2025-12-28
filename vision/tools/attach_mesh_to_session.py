from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tennis_video.mesh_session import attach_mesh_npz_to_session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attach SMPL-style mesh data (faces + per-frame vertices) to an existing web session."
        )
    )
    parser.add_argument("--session-id", required=True, help="Session id under sessions dir.")
    parser.add_argument(
        "--sessions-dir",
        default="data/web_sessions",
        help="Directory containing session folders.",
    )
    parser.add_argument(
        "--user-mesh-npz",
        required=True,
        help="NPZ containing mesh arrays for the user (expects vertices + faces).",
    )
    parser.add_argument(
        "--expert-mesh-npz",
        default=None,
        help="Optional NPZ containing mesh arrays for the expert (expects vertices; faces optional).",
    )
    parser.add_argument(
        "--user-vertices-out",
        default="user_smpl_vertices.f32",
        help="Output file name written inside the session directory.",
    )
    parser.add_argument(
        "--expert-vertices-out",
        default="expert_smpl_vertices.f32",
        help="Output file name written inside the session directory.",
    )
    parser.add_argument(
        "--vertex-labels-npy",
        default=None,
        help="Optional .npy (V,) or .npz containing per-vertex part labels (expects key 'labels' if npz).",
    )
    parser.add_argument(
        "--parts-meta-json",
        default=None,
        help="Optional JSON containing {'parts': [...], 'palette': [[r,g,b],...]} for label colors.",
    )
    parser.add_argument(
        "--labels-out",
        default="smpl_vertex_labels.u16",
        help="Output labels file name written inside the session directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sessions_dir = Path(args.sessions_dir)
    session_dir = sessions_dir / args.session_id
    session_path = session_dir / "session.json"
    try:
        result = attach_mesh_npz_to_session(
            session_path=session_path,
            user_mesh_npz=Path(args.user_mesh_npz),
            expert_mesh_npz=Path(args.expert_mesh_npz) if args.expert_mesh_npz else None,
            user_vertices_out=args.user_vertices_out,
            expert_vertices_out=args.expert_vertices_out,
            vertex_labels_npy=Path(args.vertex_labels_npy) if args.vertex_labels_npy else None,
            parts_meta_json=Path(args.parts_meta_json) if args.parts_meta_json else None,
            labels_out=args.labels_out,
        )
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Updated: {result.session_path}")
    print(f"- user vertices: {result.user_vertices_path}")
    if result.expert_vertices_path:
        print(f"- expert vertices: {result.expert_vertices_path}")
    if result.vertex_labels_path:
        print(f"- vertex labels: {result.vertex_labels_path}")


if __name__ == "__main__":
    main()
