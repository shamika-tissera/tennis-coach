from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tennis_video.mesh_pipeline import generate_and_attach_smpl_mesh_from_session_joints


def _make_dummy_smpl_model(path: Path, *, vertex_count: int = 200, face_count: int = 380) -> None:
    rng = np.random.default_rng(0)

    v_template = rng.normal(scale=0.25, size=(vertex_count, 3)).astype(np.float32)
    v_template[:, 1] += 0.9  # lift above ground-ish

    weights = rng.uniform(0.0, 1.0, size=(vertex_count, 24)).astype(np.float32)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)

    j_reg = rng.uniform(0.0, 1.0, size=(24, vertex_count)).astype(np.float32)
    j_reg /= np.maximum(j_reg.sum(axis=1, keepdims=True), 1e-8)

    faces = rng.integers(0, vertex_count, size=(face_count, 3), dtype=np.int32)
    for i in range(face_count):
        a, b, c = int(faces[i, 0]), int(faces[i, 1]), int(faces[i, 2])
        if a == b:
            faces[i, 1] = (b + 1) % vertex_count
        if a == c or b == c:
            faces[i, 2] = (c + 2) % vertex_count

    np.savez_compressed(path, v_template=v_template, weights=weights, J_regressor=j_reg, f=faces)


def _make_dummy_session(session_path: Path, *, frames: int = 30) -> None:
    t = np.linspace(0.0, 1.0, num=frames, dtype=float)

    # Simple animated COCO17 pseudo-3D joints: a small torso sway + arm swing.
    joints = np.zeros((frames, 17, 3), dtype=np.float32)
    for i, ti in enumerate(t):
        sway = 0.07 * np.sin(2.0 * np.pi * ti)
        arm = 0.20 * np.sin(2.0 * np.pi * (ti + 0.15))
        joints[i, :, 0] = sway
        joints[i, :, 1] = 0.9
        joints[i, :, 2] = 0.0
        # shoulders
        joints[i, 5] = np.array([-0.2, 1.2, 0.0], dtype=np.float32)
        joints[i, 6] = np.array([0.2, 1.2, 0.0], dtype=np.float32)
        # elbows
        joints[i, 7] = np.array([-0.35, 1.05, 0.08], dtype=np.float32)
        joints[i, 8] = np.array([0.35, 1.05, -0.08], dtype=np.float32)
        # wrists (swing)
        joints[i, 9] = np.array([-0.48, 0.9, 0.12], dtype=np.float32)
        joints[i, 10] = np.array([0.48 + arm, 0.9, -0.12], dtype=np.float32)
        # hips
        joints[i, 11] = np.array([-0.15, 0.95, 0.0], dtype=np.float32)
        joints[i, 12] = np.array([0.15, 0.95, 0.0], dtype=np.float32)
        # knees/ankles
        joints[i, 13] = np.array([-0.15, 0.55, 0.02], dtype=np.float32)
        joints[i, 14] = np.array([0.15, 0.55, -0.02], dtype=np.float32)
        joints[i, 15] = np.array([-0.15, 0.15, 0.04], dtype=np.float32)
        joints[i, 16] = np.array([0.15, 0.15, -0.04], dtype=np.float32)

    session = {
        "schema_version": 1,
        "created_at": "smoke_test",
        "stroke_type": "smoke_test",
        "user": {"joints3d_resampled": joints.tolist()},
        "expert": {"joints3d_resampled": joints.tolist()},
        "timeline": {"t": t.tolist()},
    }
    session_path.write_text(json.dumps(session, indent=2), encoding="utf-8")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="tcvision_smoke_mesh_") as td:
        root = Path(td)
        session_dir = root / "session"
        session_dir.mkdir(parents=True, exist_ok=True)

        session_path = session_dir / "session.json"
        smpl_path = root / "DUMMY_SMPL_NEUTRAL.npz"

        _make_dummy_smpl_model(smpl_path)
        _make_dummy_session(session_path)

        result = generate_and_attach_smpl_mesh_from_session_joints(
            session_path=session_path, smpl_model_path=smpl_path, include_expert=True
        )

        updated = json.loads(result.attach_result.session_path.read_text(encoding="utf-8"))
        assert "mesh" in updated, "session.json missing mesh block"
        assert updated["mesh"]["type"] == "smpl"
        assert updated["user"].get("mesh_vertices_path"), "missing user mesh_vertices_path"
        assert updated["expert"].get("mesh_vertices_path"), "missing expert mesh_vertices_path"
        assert updated["mesh"].get("vertex_labels_path"), "missing vertex_labels_path"
        assert updated["mesh"].get("parts_palette"), "missing parts_palette"

        user_bin = result.attach_result.user_vertices_path
        labels_bin = result.attach_result.vertex_labels_path
        assert user_bin.exists() and user_bin.stat().st_size > 0
        assert labels_bin is not None and labels_bin.exists() and labels_bin.stat().st_size > 0

        print("ok: mesh pipeline smoke test passed")


if __name__ == "__main__":
    main()

